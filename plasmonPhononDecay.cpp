#include <core/Util.h>
#include <electronic/matrix.h>
#include <fstream>
#include <iterator>
#include <sstream>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/Units.h>
#include "bandStruct.h"
#include "histogram.h"
#include "epsilon.h"

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Metropolis calculation of plasmon decay rate", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	struct InputMap : std::map<string,double> //map class with a safe accessor that quits with error if key not found
	{	double get(string key) const
		{	auto iter = find(key);
			if(iter == end()) die("\nCould not find required entry '%s' in input.\n", key.c_str());
			return iter->second;
		}
	}
	inputMap;
	std::ifstream systemFile(inputFilename.c_str());
	if(!systemFile.is_open())
		die("Could not open system file '%s' for reading.\n", inputFilename.c_str());
	while(!systemFile.eof())
	{	string line; getline(systemFile, line); //line-by-line processing (comments can now be inline)
		trim(line);
		istringstream iss(line);
		string name; double val;
		if(iss >> name >> val)
			inputMap[name] = val;
	}
	systemFile.close();    
	
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
	logPrintf("spinWeight = %lg\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Initialize dielectric model:
	epsilon eps("epsilon.txt");
	double omega = Eplasmon;
	eps.setFrequency(omega);
	
	//Initialize Wannier bandstructure:
	bandStruct bs("wannier", mu);

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
		for(int nk1 =0; nk1<nKpts; nk1++)
		{	vector3<> kpnt1, kpnt2;
			for(int j=0; j<3; j++)
			{	kpnt1[j] = Random::uniform();
				kpnt2[j] = Random::uniform();
			}
			double mk1k2 = bs.get_mk1k2(kpnt1, kpnt2, omega, T);
			N1block += exp(-0.5*mk1k2/(T*T));
		}
		N1block /=  nKpts;
		N1sum += N1block;
		N1sumSq += std::pow(N1block,2);
	}
	watchNorm.stop();
	mpiUtil->allReduce(N1sum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(N1sumSq, MPIUtil::ReduceSum);
	double N1 = N1sum / totalBlocks;
	double N1std = sqrt(N1sumSq/totalBlocks - N1*N1);
	logPrintf("N1 = %lg +/- %lg\n", N1, N1std);

	// Metropolis sampling of BZ:
	logPrintf("Starting Metropolis sampling of BZ\n");
	std::vector<double> Econserve_rate;
	
	// Compute effective mode vector
	complex one(1.0,0.0);
	vector3<complex> zHat(0.0, 0.0, one);
	vector3<complex> kHat(cos(kPhi), sin(kPhi), 0.0);
	complex I(0.0,1.0);
	// WHY IS N1 INSIDE THE SQUARE_ROOT, SHOULD IT STILL BE??????????????????????????????
	vector3<complex> sqrtGammaPrefac = ((M_PI/nKptsMetro) * sqrt(N1/(4*(fabs(det(R)))*eps.modGammaMinus*omega*eps.Lquant)) ) * (kHat - I*(eps.k/eps.modGammaMinus)*zHat);
	logPrintf("sqrtGammaPrefac Real = %lg %lg %lg\n",  real(sqrtGammaPrefac[0]), real(sqrtGammaPrefac[1]), real(sqrtGammaPrefac[2]));
	logPrintf("sqrtGammaPrefac Imag = %lg %lg %lg\n",  imag(sqrtGammaPrefac[0]), imag(sqrtGammaPrefac[1]), imag(sqrtGammaPrefac[2]));

	const double weightCut = 1e-6;
	double Gamma = 0., M = 0.15;// FOR NOW DEFINE M AS A CONSTANT
	histogram EcHist(-10*T, 0.5*T, Eplasmon+5*T);
	histogram EvHist(-Eplasmon-5*T, 0.5*T, 10*T);
	double acceptRatioSum = 0., acceptRatioSumSq = 0.;
	int walkerStart = (totalWalkers * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); // MPI division
	int walkerStop = (totalWalkers * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	nKpts = nKptsMetro / totalWalkers;
	StopWatch watchMet("metropolis"); watchMet.start();
	for(int walker=walkerStart; walker<walkerStop; walker++)
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
		double mk1k2Prev = INFINITY;

		for(int ik=0; ik<nKpts; )
		{	// Calculate mk:
			double mk1k2 = bs.get_mk1k2(kpnt1, kpnt2, omega, T);

			// Metropolis accept - reject:
			if(exp(0.5*(mk1k2Prev - mk1k2)/(T*T)) > Random::uniform())
			{	mk1k2Prev = mk1k2;
				kpnt1Prev = kpnt1; kpnt2Prev = kpnt2;
				
				if(mk1k2 < 2*T*T) equib = true;
			
				if(equib)
				{	ik++;
					// Calculate transitions at current k-point:
					diagMatrix E1 = bs.getStates(kpnt1);
					std::vector<matrix> Pk1 = bs.getTransitions(kpnt1)
					diagMatrix E2 = bs.getStates(kpnt2);
					std::vector<matrix> Pk2 = bs.getTransitions(kpnt2);
					for(int v=0; v<E1.nRows(); v++) if(E1[v]<10.*T)
					{	for(int c=0; c<E2.nRows(); c++) if(E2[c]>-10.*T)
						{	double mk_cv = bandStruct::mk_sub(E2[c], E1[v], Eplasmon, T);
							double weightEconserve = exp(0.5*(mk1k2-mk_cv)/(T*T))/(T*sqrt(2*M_PI)); //weight contribution due to energy conservation
							if(weightEconserve < weightCut) continue;
							// Effective matrix elements
							complex prefacDotP = 0.;
							for(int i=0; i<E1.nRows(); i++) // sum over the intermediate states
							{	for(int j=0; j<3; j++)
									prefacDotP += sqrtGammaPrefac[j] * ((Pk2[j](c,i)*(1-1/(exp(E2[i]/T)+1)) * M)/(E2[i]-E1[v]) + (Pk1[j](i,v)*(1-1/(exp(E1[i]/T)+1))*M)/(E1[i]-E1[v]-Eplasmon));
							}
							double weight = (0.5*spinWeight) * weightEconserve * prefacDotP.norm(); //norm = abs^2
							//Include in statistics:
							Gamma += weight;
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
		double GammaSoFar = totalWalkers * Gamma / (walker - walkerStart + 1); //current estimate from this process
		logPrintf("acceptRatio = %lg  nKptsTot = %d  Gamma = %lg eV\n", acceptRatio, nKptsTot, GammaSoFar/eV); logFlush();
	}
	watchMet.stop();
	
	//Acceptance ratio:
	mpiUtil->allReduce(acceptRatioSum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(acceptRatioSumSq, MPIUtil::ReduceSum);
	double acceptRatio = acceptRatioSum / totalWalkers;
	double acceptRatioStd = sqrt(acceptRatioSumSq/totalWalkers - acceptRatio*acceptRatio);
	logPrintf("acceptRatio = %lg +/- %lg\n", acceptRatio, acceptRatioStd);

	//Decay rate:
	mpiUtil->allReduce(Gamma, MPIUtil::ReduceSum);
	logPrintf("Linewidth = %lg eV\n", Gamma/eV);
	
	//Carrier distributions:
	EcHist.allReduce(MPIUtil::ReduceSum); EcHist.print("eDistrib-2.8eV-metro17.dat", eV);
	EvHist.allReduce(MPIUtil::ReduceSum); EvHist.print("hDistrib-2.8eV-metro17.dat", eV);

	finalizeSystem();
}
