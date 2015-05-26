#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include "BandStruct.h"
#include "Histogram.h"
#include "Epsilon.h"
#include "InputMap.h"
#include "Units.h"

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
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
	Epsilon eps("Wannier/epsilon.dat");

	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(2); //contract for zHat and kHat, which will be combined ina frequency dependent way
	Ahat[0] = vector3<complex>(cos(kPhi), sin(kPhi), 0.); //kHat
	Ahat[1] = vector3<complex>(0., 0., 1.); //zHat
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE", Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);
	
	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	long nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	long nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	int nBands = bs.getStates(vector3<>()).nRows();
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	complex I(0,1);
	double prefac0 = 4*M_PI; //frequency independent part of prefactor.  SHOULD THIS ALSO HAVE spinWeight?

	//Initalize sturctures to hold data
	double Gsum = 0., GsumSq=0.;
	
	//Monte Carlo loop:
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	logPrintf("\nProgress: "); logFlush();
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{	double Gblock=0.;
		
		//Generate a bunch of k-points:
		std::vector< vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		
		//Calculate electronic states and matrix elements for bunch:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector<diagMatrix> Farr = Earr; //convert to fillings:
		for(diagMatrix& F: Farr)
			for(double& f: F)
			{	double e = f/T; //E/T actually
				f = (e>30 ? exp(-e) : 1./(1. + exp(e))); //avoid overflow issues
			}
		
		//Phonon-assisted transitions:
		for(int ik1=0; ik1<bunchSize; ik1++)
		{	const diagMatrix& E1 = Earr[ik1];
			const diagMatrix& F1 = Farr[ik1];
			//phonon matrix elements for ik1 with rest of bunch:
			std::vector<matrix> gePhArr[bunchSize];
			bs.setPhononMatElemArray(kArr[ik1], kArr, gePhArr);
			//Loop over second k-point:
			for(int ik2=0; ik2<bunchSize; ik2++) if(ik2!=ik1) //avoid gamma-point phonon singularity
			{	const std::vector<matrix>& gePh = gePhArr[ik2];
				diagMatrix omegaPh = bs.getPhononModes(kArr[ik1] - kArr[ik2]);
				const diagMatrix& E2 = Earr[ik2];
				const diagMatrix& F2 = Farr[ik2];
				//Loops over bands and phonon modes:
				for(int v=0; v<nBands; v++) if(E1[v]<10.*T)
				{	for(int c=0; c<nBands; c++) if(E2[c]>-10.*T)
					{	for(int alpha=0; alpha<nModes; alpha++)
						{	for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
							{	double omega = E2[c] - E1[v] - ae*omegaPh[alpha]; //energy conservation
								if(omega<=0 || omega>=EplasmonMax) continue; //irrelevant event
								eps.setFrequency(omega, false);
								double phononPrefac = prefac0 * omegaPh[alpha];//SHOULD THIS HAVE *nKpairs??????
								if(!std::isfinite(phononPrefac) || phononPrefac<0.) continue; //avoid over-damped region
								double nPh = 1./(exp(omegaPh[alpha]/T) - 1.);
								double weightPrefac = phononPrefac * ((F1[v]-F2[c]) * nPh - F2[c]*(1-F1[v]));
								// Effective matrix elements
								complex Meff = gePh[alpha](c,v);
								double weight = weightPrefac * std::pow(abs(Meff),2);
								//Results using all available bands:
								Gblock +=weight;
							}
						}
					}
				}
				
			}
			Gsum +=Gblock/nKpairs;
			GsumSq +=std::pow(Gblock/nKpairs,2);
		}
		
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunchesMine)));
			logFlush();
		}
	}
	logPrintf("done.\n"); logFlush();
	
	mpiUtil->allReduce(Gsum, MPIUtil::ReduceSum);
        mpiUtil->allReduce(GsumSq, MPIUtil::ReduceSum);
	double G = Gsum;// / totalBlocks;
        double Gstd = sqrt(GsumSq - G*G);
        logPrintf("G = %lg +/- %lg\n", G, Gstd);

	finalizeSystem();
}
