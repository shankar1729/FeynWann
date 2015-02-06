#include <core/Util.h>
#include <electronic/matrix.h>
#include <fstream>
#include <iterator>
#include <sstream>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include "Units.h"
#include "BandStruct.h"
#include "Histogram.h"
#include "Epsilon.h"
#include "InputMap.h"
#include "LineWidth.h"

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Metropolis calculation of two-plasmon decay rate", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const int totalBlocks = inputMap.get("totalBlocks"); assert(totalBlocks>0);
	const int nKptsMetro = inputMap.get("nKptsMetro");
	const double dk = inputMap.get("dk");
	const int totalWalkers = inputMap.get("totalWalkers"); assert(totalWalkers>0);
	const double kPhi1 = inputMap.get("kPhi");
	const double kPhi2 = inputMap.get("kPhi2");
	const double Eplasmon1 = inputMap.get("Eplasmon") * eV;
	const double Eplasmon2 = inputMap.get("Eplasmon2") * eV;
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	const double Squant = std::pow(10*Angstrom, 2) * std::pow(128.,-3); //1 nm^2 DIVIDED BY ~ 2 million (HACK)
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("totalBlocks = %d\n", totalBlocks);
	logPrintf("nKptsMetro = %d\n", nKptsMetro);
	logPrintf("dK = %lg\n", dk);
	logPrintf("totalWalkers = %d\n", totalWalkers);
	logPrintf("kPhi = %lg\n", kPhi1);
	logPrintf("kPhi2 = %lg\n", kPhi2);
	logPrintf("Eplasmon = %lg\n", Eplasmon1);
	logPrintf("Eplasmon2 = %lg\n", Eplasmon2);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
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
	Epsilon eps1("epsilon.txt"); eps1.setFrequency(Eplasmon1);
	Epsilon eps2(eps1); eps2.setFrequency(Eplasmon2);
	double EplasmonTot = Eplasmon1 + Eplasmon2;
	
	// Compute effective mode vectors
	complex one(1.0,0.0);
	vector3<complex> zHat(0.0, 0.0, one);
	vector3<complex> kHat1(cos(kPhi1), sin(kPhi1), 0.0);
	vector3<complex> kHat2(cos(kPhi2), sin(kPhi2), 0.0);
	complex I(0.0,1.0);
	std::vector< vector3<complex> > Ahat;
	Ahat.push_back(kHat1 - I*(eps1.k/eps1.modGammaMinus)*zHat);
	Ahat.push_back(kHat2 - I*(eps2.k/eps2.modGammaMinus)*zHat);
	
	//Initalize line width of intermediate electronic states
	LineWidth lineWidth("ImSigma.dat");

	//Initialize Wannier bandstructure:
	BandStruct bs("Wannier/wannier", mu, spinWeight, string(), Ahat);

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
			double mk = bs.get_mk(kpnt, EplasmonTot, T);
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
	
	double gammaPrefac = N1 * std::pow(M_PI,3) /
		( nKptsMetro * 2*fabs(det(R)) * (eps1.modGammaMinus + eps2.modGammaMinus)
		 * Eplasmon1 * Eplasmon2 * eps1.Lquant * eps2.Lquant * Squant );
	logPrintf("gammaPrefac = %lg\n",  gammaPrefac);
	
	const double weightCut = 1e-6;
	Histogram EcHist(-10*T, 0.5*T, EplasmonTot+5*T);
	Histogram EvHist(-EplasmonTot-5*T, 0.5*T, 10*T);
	double acceptRatioSum = 0., acceptRatioSumSq = 0., GammaSum = 0., GammaSumSq = 0.;
	std::vector<double> GammaConv(bs.getStates(vector3<>()).nRows(), 0.); //empty-state convergence
	std::vector<double> GammaConvCEDA(bs.getStates(vector3<>()).nRows(), 0.); //empty-state convergence with CEDA
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
			double mk = bs.get_mk(kpnt, EplasmonTot, T);

			// Metropolis accept - reject:
			if(exp(0.5*(mkPrev - mk)/(T*T)) > Random::uniform())
			{	mkPrev = mk;
				kpntPrev = kpnt;
				
				if(mk < 2*T*T) equib = true;
			
				if(equib)
				{	ik++;
					// Calculate transitions at current k-point:
					diagMatrix E = bs.getStates(kpnt);
					std::vector<matrix> AdotParr = bs.getDipoleMatElem(kpnt);
					const matrix& AdotP1 = AdotParr[0];
					const matrix& AdotP2 = AdotParr[1];
					matrix AAdotPP = bs.getDipoleSqMatElem(kpnt);
					for(int v=0; v<E.nRows(); v++) if(E[v]<10.*T)
					{	for(int c=0; c<E.nRows(); c++) if(E[c]>-10.*T)
						{	double mk_cv = BandStruct::mk_sub(E[c], E[v], EplasmonTot, T);
							double weightEconserve = (0.5*spinWeight) * exp(0.5*(mk-mk_cv)/(T*T))/(T*sqrt(2*M_PI)); //weight contribution due to energy conservation (and spin)
							if(weightEconserve < weightCut) continue;
							//Effective matrix element
							complex Meff = 0.; double weight = 0.;
							complex num1sum = 0., num2sum = 0.; //partial numerator sums for CEDA
							for(int l=0; l<E.nRows(); l++)
							{	complex El(E[l], lineWidth(E[l]));
								double Fl = 1./(1.+exp(E[l]/T)); //occupation
								complex num1 = AdotP1(c,l)*AdotP2(l,v); num1sum += num1;
								complex num2 = AdotP2(c,l)*AdotP1(l,v); num2sum += num2;
								complex den1 = one/(El-E[v]-Eplasmon2);
								complex den2 = one/(El-E[v]-Eplasmon1);
								Meff += (1.-Fl) * (num1*den1 + num2*den2);
								weight = gammaPrefac * weightEconserve * Meff.norm(); //norm = abs^2;
								GammaConv[l] += weight; //estimate based on truncating to i bands
								//CEDA corrections:
								double Ebar = bs.Eceda[l];
								double den1ceda = 1./(Ebar-E[v]-Eplasmon2);
								double den2ceda = 1./(Ebar-E[v]-Eplasmon1);
								complex MeffCEDA = Meff
									- den1ceda * num1sum
									- den2ceda * num2sum
									+ (den1ceda + den2ceda) * AAdotPP(c,v);
								weight = gammaPrefac * weightEconserve * MeffCEDA.norm(); //overwrite weight with CEDA (so that final result uses CEDA)
								GammaConvCEDA[l] += weight; //estimate based on truncating to i bands
							}
							//Include in statistics:
							GammaBlock += weight;
							EcHist.addEvent(E[c], weight);
							EvHist.addEvent(E[v], weight);
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
	sprintf(fname, "Distrib-%.1lfeV+%.1lfeV-metro.dat", Eplasmon1/eV, Eplasmon2/eV);
	EcHist.allReduce(MPIUtil::ReduceSum); EcHist.print(string("e")+fname, eV);
	EvHist.allReduce(MPIUtil::ReduceSum); EvHist.print(string("h")+fname, eV);

	//Empty-state convergence:
	mpiUtil->allReduce(GammaConv.data(), GammaConv.size(), MPIUtil::ReduceSum);
	mpiUtil->allReduce(GammaConvCEDA.data(), GammaConvCEDA.size(), MPIUtil::ReduceSum);
	sprintf(fname, "bandConvergence-%.1lfeV+%.1lfeV-metro.dat", Eplasmon1/eV, Eplasmon2/eV);
	FILE* fp = fopen(fname, "w");
	for(int i=0; i<int(GammaConv.size()); i++)
		fprintf(fp, "%d %lg %lg\n", i+1, GammaConv[i]/eV, GammaConvCEDA[i]/eV);
	fclose(fp);
	
	finalizeSystem();
}
