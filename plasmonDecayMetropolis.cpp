#include <core/Util.h>
#include <electronic/matrix.h>
#include <fstream>
#include <iterator>
#include <sstream>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/Units.h>
#include "BandStruct.h"
#include "Histogram.h"
#include "Epsilon.h"
#include "InputMap.h"

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Metropolis calculation of plasmon decay rate", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const int totalBlocks = inputMap.get("totalBlocks"); assert(totalBlocks>0);
	const int nKptsMetro = inputMap.get("nKptsMetro");
	const double dk = inputMap.get("dk");
	const int totalWalkers = inputMap.get("totalWalkers"); assert(totalWalkers>0);
	const double kPhi = inputMap.get("kPhi");
	const double Eplasmon = inputMap.get("Eplasmon") * eV;
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const double Eplasmon2 = inputMap.get("Eplasmon2") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const int eventSaveInterval = inputMap.get("eventSaveInterval", 0); //default 0 => no save
	matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("totalBlocks = %d\n", totalBlocks);
	logPrintf("nKptsMetro = %d\n", nKptsMetro);
	logPrintf("dK = %lg\n", dk);
	logPrintf("totalWalkers = %d\n", totalWalkers);
	logPrintf("kPhi = %lg\n", kPhi);
	logPrintf("Eplasmon = %lg\n", Eplasmon);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("Eplamson2 = %lg\n", Eplasmon2);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Initialize dielectric model:
	Epsilon eps("epsilon.txt");
	double omega = Eplasmon;
	eps.setFrequency(omega);
	
	//Initialize Wannier bandstructure:
	BandStruct bs("Wannier/wannier", mu, spinWeight);

	//Compute the normalization factor
	int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
	int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nKpts = nKptsN1/totalBlocks;
	double N1sum = 0., N1sumSq = 0.;
	StopWatch watchNorm("normalization"); watchNorm.start();
	logPrintf("Calculating normalization factor ... "); logFlush();
	for(int block=blockStart; block<blockStop; block++)
	{	Random::seed(block);
		double N1block = 0.;
		for(int nk =0; nk<nKpts; nk++)
		{	vector3<> kpnt; for(int j=0; j<3; j++) kpnt[j] = Random::uniform();
			double mk = bs.get_mk(kpnt, omega, T);
			N1block += exp(-0.5*mk/(T*T));
		}
		N1block /=  nKpts;
		N1sum += N1block;
		N1sumSq += std::pow(N1block,2);
	}
	watchNorm.stop();
	mpiUtil->allReduce(N1sum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(N1sumSq, MPIUtil::ReduceSum);
	double N1 = N1sum / totalBlocks;
	double N1std = sqrt(N1sumSq/totalBlocks - N1*N1)/sqrt(totalBlocks);
	logPrintf("N1 = %lg +/- %lg\n", N1, N1std);
	bool skipMetro = false;
	if(fabs(N1) < 1e-8)
	{	skipMetro = true;
		logPrintf("Warning: N1 is too small => no allowed transitions; skipping Metropolis sampling.\n");
	}
	
	// Metropolis sampling of BZ:
	logPrintf("Starting Metropolis sampling of BZ\n");
	std::vector<double> Econserve_rate;
	
	// Compute effective mode vector
	complex one(1.0,0.0);
	vector3<complex> zHat(0.0, 0.0, one);
	vector3<complex> kHat(cos(kPhi), sin(kPhi), 0.0);
	complex I(0.0,1.0);
	vector3<complex> sqrtGammaPrefac = (M_PI * sqrt(N1/((nKptsMetro*fabs(det(R)))*eps.modGammaMinus*omega*eps.Lquant)) ) * (kHat - I*(eps.k/eps.modGammaMinus)*zHat);
	logPrintf("sqrtGammaPrefac Real = %lg %lg %lg\n",  real(sqrtGammaPrefac[0]), real(sqrtGammaPrefac[1]), real(sqrtGammaPrefac[2]));
	logPrintf("sqrtGammaPrefac Imag = %lg %lg %lg\n",  imag(sqrtGammaPrefac[0]), imag(sqrtGammaPrefac[1]), imag(sqrtGammaPrefac[2]));
	
	//Prepare for event collection:
	double eventPrefac = N1*eventSaveInterval/(nKptsMetro*fabs(det(R)));
	struct Event
	{	double Ev, Ec;
		vector3<> vv, vc;
		vector3<complex> Pcv;
	};
	std::vector<Event> events;
	if(eventSaveInterval) events.reserve(totalWalkers / mpiUtil->nProcesses());
	
	const double weightCut = 1e-6;
	Histogram EcHist(-10*T, 0.5*T, Eplasmon+5*T);
	Histogram EvHist(-Eplasmon-5*T, 0.5*T, 10*T);
	double acceptRatioSum = 0., acceptRatioSumSq = 0., GammaSum = 0., GammaSumSq = 0.;
	int walkerStart = (totalWalkers * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); // MPI division
	int walkerStop = (totalWalkers * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	nKpts = nKptsMetro / totalWalkers;
	StopWatch watchMet("metropolis"); watchMet.start();
	if(!skipMetro) for(int walker=walkerStart; walker<walkerStop; walker++)
	{	Random::seed(walker);
		logPrintf("Metropolis walk# %d ... ", walker); logFlush();
		vector3<> kpntPrev;
		for(int j=0; j<3; j++)
			kpntPrev[j] = Random::uniform();
		vector3<> kpnt = kpntPrev;
		int nKptsTot = 0; //denominator of accept ratio
		int nKptsEquib = 0;
		bool equib = false;
		double mkPrev = INFINITY, GammaBlock = 0.;

		for(int ik=0; ik<nKpts; )
		{	// Calculate mk:
			double mk = bs.get_mk(kpnt, omega, T);

			// Metropolis accept - reject:
			if(exp(0.5*(mkPrev - mk)/(T*T)) > Random::uniform())
			{	mkPrev = mk;
				kpntPrev = kpnt;
				
				if(mk < 2*T*T) equib = true;
			
				if(equib)
				{	ik++;
					bool eventSave = eventSaveInterval && (ik % eventSaveInterval == 0);
					// Calculate transitions at current k-point:
					diagMatrix E = bs.getStates(kpnt);
					std::vector<matrix> Pk = bs.getDipoleMatElem(kpnt);
					std::vector<vector3<>> vk; if(eventSave) vk = bs.getVelocity(kpnt, R);
					for(int v=0; v<E.nRows(); v++) if(E[v]<10.*T)
					{	for(int c=0; c<E.nRows(); c++) if(E[c]>-10.*T)
						{	double mk_cv = BandStruct::mk_sub(E[c], E[v], Eplasmon, T);
							double weightEconserve = (0.5*spinWeight) * exp(0.5*(mk-mk_cv)/(T*T))/(T*sqrt(2*M_PI)); //weight contribution due to energy conservation (and spin)
							if(weightEconserve < weightCut) continue;
							// Effective matrix elements
							vector3<complex> Pk_cv; for(int j=0; j<3; j++) Pk_cv[j] = Pk[j](c,v);
							double weight = weightEconserve * dot(sqrtGammaPrefac, Pk_cv).norm(); //norm = abs^2
							//Include in statistics:
							GammaBlock += weight;
							EcHist.addEvent(E[c], weight);
							EvHist.addEvent(E[v], weight);
							//Save event if necessary:
							if(eventSave)
							{	Event event = { E[v], E[c], vk[v], vk[c], sqrt(eventPrefac * weightEconserve) * Pk_cv };
								events.push_back(event);
							}
						}
					}
				}
			}
			// Generate next kpoint
			for(int j=0; j<3; j++)
				kpnt[j] = kpntPrev[j] + dk * Random::normal();
			if(equib) nKptsTot++;
			else
			{	nKptsEquib++;
				if(nKptsEquib > nKpts/2) //heuristic to prevent getting stuck in local pockets
				{	logPrintf("\n\tReseting walker due to too many equilibration steps.\n");
					for(int j=0; j<3; j++) kpntPrev[j] = Random::uniform();
					kpnt = kpntPrev;
					mkPrev = INFINITY;
					nKptsEquib = 0;
				}
			}
		}
		double acceptRatio = (double)nKpts/nKptsTot;
		acceptRatioSum += acceptRatio;
		acceptRatioSumSq += std::pow(acceptRatio,2);
		GammaBlock *= totalWalkers; //prefactor is normalized for total kpts, not per block
		GammaSum += GammaBlock;
		GammaSumSq += std::pow(GammaBlock,2);
		logPrintf("acceptRatio = %lg  nKptsTot = %d  Gamma = %lg eV\n", acceptRatio, nKptsTot, GammaBlock/eV); logFlush();
	}
	watchMet.stop();
	
	//Acceptance ratio:
	mpiUtil->allReduce(acceptRatioSum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(acceptRatioSumSq, MPIUtil::ReduceSum);
	double acceptRatio = acceptRatioSum / totalWalkers;
	double acceptRatioStd = sqrt(acceptRatioSumSq/totalWalkers - acceptRatio*acceptRatio)/sqrt(totalWalkers);
	logPrintf("acceptRatio = %lg +/- %lg\n", acceptRatio, acceptRatioStd);

	//Decay rate:
	mpiUtil->allReduce(GammaSum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(GammaSumSq, MPIUtil::ReduceSum);
	double Gamma = GammaSum / totalWalkers;
	double GammaStd = sqrt(GammaSumSq/totalWalkers - Gamma*Gamma)/sqrt(totalWalkers);
	GammaStd = hypot(GammaStd, Gamma*N1std/N1); //propagate error in N1
	logPrintf("Gamma = %lg +/- %lg eV\n", Gamma/eV, GammaStd/eV);
	
	//Carrier distributions:
	char fname[256];
	sprintf(fname, "Distrib-%.1lfeV-metro.dat", Eplasmon/eV);
	EcHist.allReduce(MPIUtil::ReduceSum); EcHist.print(string("e")+fname, eV);
	EvHist.allReduce(MPIUtil::ReduceSum); EvHist.print(string("h")+fname, eV);

	//Write events:
	if(eventSaveInterval)
	{	MPIUtil::File fpEvent;
		sprintf(fname, "events-%.1lfeV-metro.dat", Eplasmon/eV);
		unsigned long nEventsPrev = 0; //number of events from previous processes
		for(int jProcess=0; jProcess<mpiUtil->nProcesses(); jProcess++)
		{	unsigned long nEvents = events.size();
			mpiUtil->bcast(nEvents, jProcess); //nEvents is now the number of events on jProcess
			if(jProcess < mpiUtil->iProcess()) nEventsPrev += nEvents;
		}
		mpiUtil->fopenWrite(fpEvent, fname);
		mpiUtil->fseek(fpEvent, nEventsPrev*sizeof(Event), SEEK_SET);
		mpiUtil->fwrite(events.data(), sizeof(Event), events.size(), fpEvent);
		mpiUtil->fclose(fpEvent);
	}
	
	finalizeSystem();
}
