#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "Histogram.h"
#include "Epsilon.h"
#include "LineWidth.h"
#include "InputMap.h"
#include "Units.h"

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of all single-plasmon decay processes", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const double kPhi = inputMap.get("kPhi");
	const double EplasmonMax = inputMap.get("EplasmonMax") * eV;
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("kPhi = %lg\n", kPhi);
	logPrintf("EplasmonMax = %lg\n", EplasmonMax);
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
	Epsilon eps("epsilon.txt");

	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(2); //contract for zHat and kHat, which will be combined ina frequency dependent way
	Ahat[0] = vector3<complex>(cos(kPhi), sin(kPhi), 0.); //kHat
	Ahat[1] = vector3<complex>(0., 0., 1.); //zHat
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE", Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);
	
	//Initalize line width of intermediate electronic states
	LineWidth lineWidth("Wannier/wannier", bs);

	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	int nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	int nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	int nBands = bs.getStates(vector3<>()).nRows();
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	complex I(0,1);
	double directPrefac0 = (0.5*spinWeight) * std::pow(M_PI,2) / (nKpts * fabs(det(R))); //frequency independent part of prefac
	
	//Initialize histograms
	double gaussMargin = 5*T;
	double fermiMargin = 10*T;
	Histogram2D EcDirect(-fermiMargin, T, EplasmonMax+gaussMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D EcPhonon(-fermiMargin, T, EplasmonMax+gaussMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D EvDirect(-EplasmonMax-gaussMargin, T, fermiMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D EvPhonon(-EplasmonMax-gaussMargin, T, fermiMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram GammaDirect(gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram GammaPhonon(gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D convPhonon(1, 1, nBands,  gaussMargin, T, EplasmonMax-gaussMargin); //empty-state convergence for phonon
	
	//Monte Carlo loop:
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	logPrintf("\nProgress: "); logFlush();
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{
		//Generate a bunch of k-points:
		std::vector< vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		
		//Calculate electronic states and matrix elements for bunch:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector<diagMatrix> ImEarr = lineWidth(kArr);
		std::vector< std::vector<matrix> > Parr = bs.getDipoleMatElem(kArr);
		std::vector<diagMatrix> Farr = Earr; //convert to fillings:
		for(diagMatrix& F: Farr)
			for(double& f: F)
			{	double e = f/T; //E/T actually
				f = (e>30 ? exp(-e) : 1./(1. + exp(e))); //avoid overflow issues
			}
		
		//Direct transitions:
		for(int ik=0; ik<bunchSize; ik++)
		{	const diagMatrix& E = Earr[ik];
			const diagMatrix& F = Farr[ik];
			const std::vector<matrix>& P = Parr[ik];
			for(int v=0; v<E.nRows(); v++) if(E[v]<10.*T)
			{	for(int c=0; c<E.nRows(); c++) if(E[c]>-10.*T)
				{	double omega = E[c] - E[v]; //energy conservation
					if(omega<=0 || omega>=EplasmonMax) continue; //irrelevant event
					eps.setFrequency(omega, false);
					complex AdotPcv = P[0](c,v) - I*(eps.k/eps.modGammaMinus)*P[1](c,v);
					double directPrefac = directPrefac0 / (eps.modGammaMinus * omega * eps.Lquant);
					double weight = directPrefac * F[v] * (1.-F[c]) * AdotPcv.norm(); //norm=abs^2
					EcDirect.addEvent(E[c], omega, weight);
					EvDirect.addEvent(E[v], omega, weight);
					GammaDirect.addEvent(omega, weight);
				}
			}
		}
		
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunchesMine)));
			logFlush();
		}
	}
	logPrintf("done.\n"); logFlush();
	
	EcDirect.allReduce(MPIUtil::ReduceSum); EcDirect.print("hDistribAll-direct.dat", 1./eV, 1./eV, eV);
	EvDirect.allReduce(MPIUtil::ReduceSum); EvDirect.print("eDistribAll-direct.dat", 1./eV, 1./eV, eV);
	GammaDirect.allReduce(MPIUtil::ReduceSum); GammaDirect.print("GammaAll-direct.dat", 1./eV, 1./eV);
	
	finalizeSystem();
	return 0;
}
