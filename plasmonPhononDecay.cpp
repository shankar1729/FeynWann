#include <core/Util.h>
#include <electronic/matrix.h>
#include <fstream>
#include <iterator>
#include <sstream>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/Units.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "Histogram.h"
#include "Epsilon.h"
#include "LineWidth.h"
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
	const double spinWeight = inputMap.get("spinWeight");
	const double vl = inputMap.get("vl")* meter *2.41888e-17;// m/s in atomic units
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

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
	logPrintf("spinWeight = %lg\n", spinWeight);
	logPrintf("vl = %lg\n", vl);
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
	
	//Initalize line width of intermediate electronic states
	LineWidth lineWidth("ImSigma.dat");

	//Initialize Wannier bandstructure:
	BandStruct bs("wannier", mu);

	//Compute the normalization factor
	int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
	int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nKpts = nKptsN1/totalBlocks;
	double N1sum = 0., N1sumSq = 0., kappaSum = 0., kappaSumSq = 0.;
	StopWatch watchNorm("normalization"); watchNorm.start();
	logPrintf("Calculating normalization factor ... "); logFlush();
	for(int block=blockStart; block<blockStop; block++)
	{	Random::seed(block);
		double N1block = 0., kappaSqrdBlock = 0.;
		for(int nk1 =0; nk1<nKpts; nk1++)
		{	vector3<> kpnt1, kpnt2;
			for(int j=0; j<3; j++)
			{	kpnt1[j] = Random::uniform();
				kpnt2[j] = Random::uniform();
			}
			double mk1k2 = bs.get_mk1k2(kpnt1, kpnt2, omega, T);
			N1block += exp(-0.5*mk1k2/(T*T));
			diagMatrix Ek = bs.getStates(kpnt1);
			for(int n = 0; n<Ek.nRows(); n++)
			{	double dFdE =1/(T*std::pow(2*cosh(Ek[n]/(2*T)),2));
				kappaSqrdBlock += 4*M_PI*spinWeight*dFdE/fabs(det(R));
			}
		}
		N1block /=  nKpts;
		N1sum += N1block;
		N1sumSq += std::pow(N1block,2);
		kappaSqrdBlock /= nKpts;
		kappaSum += sqrt(kappaSqrdBlock);
		kappaSumSq +=kappaSqrdBlock;
	}
	watchNorm.stop();
	mpiUtil->allReduce(N1sum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(N1sumSq, MPIUtil::ReduceSum);
	mpiUtil->allReduce(kappaSum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(kappaSumSq, MPIUtil::ReduceSum);
	double N1 = N1sum / totalBlocks;
	double N1std = sqrt(N1sumSq/totalBlocks - N1*N1)/sqrt(totalBlocks);
	double kappa = kappaSum / totalBlocks;
	double kappaStd = sqrt(kappaSumSq/totalBlocks - kappa*kappa)/sqrt(totalBlocks);
	logPrintf("N1 = %lg +/- %lg\n", N1, N1std);
	logPrintf("kappa = %lg +/- %lg\n", kappa, kappaStd);
	bool skipMetro = false;
	if(fabs(N1) < 1e-8)
	{	skipMetro = true;
		logPrintf("Warning: N1 is too small => no allowed transitions; skipping Metropolis sampling.\n");
	}

	// Metropolis sampling of BZ:
	logPrintf("Starting Metropolis sampling of BZ\n");
	std::vector<double> Econserve_rate;
	
	// Compute effective mode vector
	nKpts = nKptsMetro / totalWalkers;
	complex one(1.0,0.0);
	vector3<complex> zHat(0.0, 0.0, one);
	vector3<complex> kHat(cos(kPhi), sin(kPhi), 0.0);
	complex I(0.0,1.0);
	vector3<complex> sqrtGammaPrefac = ( M_PI * sqrt(N1/(nKptsMetro*4*fabs(det(R))*eps.modGammaMinus*omega*eps.Lquant)) ) * (kHat - I*(eps.k/eps.modGammaMinus)*zHat);
	logPrintf("sqrtGammaPrefac Real = %lg %lg %lg\n",  real(sqrtGammaPrefac[0]), real(sqrtGammaPrefac[1]), real(sqrtGammaPrefac[2]));
	logPrintf("sqrtGammaPrefac Imag = %lg %lg %lg\n",  imag(sqrtGammaPrefac[0]), imag(sqrtGammaPrefac[1]), imag(sqrtGammaPrefac[2]));

	// Values for use in metropolis sampling
	const double weightCut = 1e-6, energyCut = 40*eV;
	Histogram EcHist(-10*T, 0.5*T, Eplasmon+5*T);
	Histogram EvHist(-Eplasmon-5*T, 0.5*T, 10*T);
	double acceptRatioSum = 0., acceptRatioSumSq = 0., GammaSum = 0., GammaSumSq = 0.;
	int walkerStart = (totalWalkers * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); // MPI division
	int walkerStop = (totalWalkers * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	StopWatch watchMet("metropolis"); watchMet.start();
	//Start metropolis sampling
	if(!skipMetro) for(int walker=walkerStart; walker<walkerStop; walker++)
	{	Random::seed(walker);
		logPrintf("Metropolis walk# %d ... ", walker); logFlush();
		vector3<> kpnt1Prev, kpnt2Prev;
		for(int j=0; j<3; j++)
		{	kpnt1Prev[j] = Random::uniform();
			kpnt2Prev[j] = Random::uniform();
		}
		vector3<> kpnt1 = kpnt1Prev, kpnt2 = kpnt2Prev;
		int nKptsTot = 0; //denominator of accept ratio
		int nKptsEquib = 0;
		bool equib = false;
		double mk1k2Prev = INFINITY, GammaBlock = 0.;

		for(int ik=0; ik<nKpts; )
		{	// Calculate mk:
			double mk1k2 = bs.get_mk1k2(kpnt1, kpnt2, omega, T);

			// Metropolis accept - reject:
			if(exp(0.5*(mk1k2Prev - mk1k2)/(T*T)) > Random::uniform())
			{	mk1k2Prev = mk1k2;
				kpnt1Prev = kpnt1;
				kpnt2Prev = kpnt2;
				
				if(mk1k2 < 2*T*T) equib = true;
			
				if(equib)
				{	ik++;
					// Calculate transitions at current k-point:
					diagMatrix E1 = bs.getStates(kpnt1);
					std::vector<matrix> Pk1 = bs.getDipoleMatElem(kpnt1);
					diagMatrix E2 = bs.getStates(kpnt2);
					std::vector<matrix> Pk2 = bs.getDipoleMatElem(kpnt2);
					diagMatrix P = bs.getPhononModes(kpnt1-kpnt2);
					std::vector<matrix> PePh = bs.getPhononMatElem(kpnt1,kpnt2);
					for(int v=0; v<E1.nRows(); v++) if(E1[v]<10.*T)
					{	for(int c=0; c<E2.nRows(); c++) if(E2[c]>-10.*T)
						{	double weight = 0.;
							for(int alpha=0; alpha<P.nRows(); alpha ++)
							{	for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
								{	double mk_cv = BandStruct::mk_sub(E2[c], E1[v], Eplasmon + ae*P[alpha], T);
									double weightEconserve = exp(0.5*(mk1k2-mk_cv)/(T*T))/(T*sqrt(2*M_PI)); //weight contribution due to energy conservation
									if(weightEconserve < weightCut) continue;
									// Effective matrix elements
									complex prefacDotP = 0.;
									for(int i=0; i<E1.nRows(); i++) // sum over the intermediate states
									{	if(E2[i]<energyCut && E1[i]<energyCut)
										{	complex E1i(E1[i], lineWidth(E1[i]));
											complex E2i(E2[i], lineWidth(E2[i]));
											double f1i = 1./(1.+exp(E1[i]/T));
											double f2i = 1./(1.+exp(E2[i]/T));
											for(int j=0; j<3; j++)
												prefacDotP += sqrtGammaPrefac[j] * 
													( Pk2[j](c,i) * (1.-f2i) * PePh[alpha](c,i) / (E2i-E2[c] + Eplasmon)
													+ Pk2[j](i,v) * (1.-f1i) * PePh[alpha](i,v) / (E1i-E1[v] - Eplasmon) );
										}
									}
									weight += (0.5*spinWeight) * weightEconserve * prefacDotP.norm(); //norm = abs^2
								}
							}
							//Include in statistics:
							GammaBlock += weight;
							EcHist.addEvent(E2[c], weight);
							EvHist.addEvent(E1[v], weight);
						}
					}
				}
			}
			// Generate next kpoints
			for(int j=0; j<3; j++)
			{	kpnt1[j] = kpnt1Prev[j] + dk * Random::normal();
				kpnt2[j] = kpnt2Prev[j] + dk * Random::normal();
			}
			if(equib) nKptsTot++;
			else
			{	nKptsEquib++;
				if(nKptsEquib > nKpts/2) //heuristic to prevent getting stuck in local pockets
				{	logPrintf("\n\tReseting walker due to too many equilibration steps.\n");
					for(int j=0; j<3; j++)
					{	kpnt1Prev[j] = Random::uniform();
						kpnt2Prev[j] = Random::uniform();
					}
					kpnt1 = kpnt1Prev; kpnt2 = kpnt2Prev;
					mk1k2Prev = INFINITY;
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
	GammaStd = hypot(GammaStd, Gamma*N1std/N1);
	logPrintf("Gamma = %lg +/- %lg eV\n", Gamma/eV, GammaStd/eV);
	
	//Carrier distributions:
	char fname[256];
	sprintf(fname, "Distrib-%.1lfeV-phonon.dat", Eplasmon/eV);
	EcHist.allReduce(MPIUtil::ReduceSum); EcHist.print(string("e")+fname, eV);
	EvHist.allReduce(MPIUtil::ReduceSum); EvHist.print(string("h")+fname, eV);

	//Experimental lineWidth
	complex arg = eps.epsilon / (eps.epsilon + 1);
	double omegaIm = fabs(sin(0.5*atan2(imag(arg),real(arg)))) * omega;
	logPrintf("Experimental Linewidth = %lg eV\n", omegaIm/eV);

	finalizeSystem();
}
