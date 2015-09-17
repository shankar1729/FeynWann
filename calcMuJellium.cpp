#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "LineWidth.h"
#include "InputMap.h"
#include "Units.h"
#include "Histogram.h"
#include "Epsilon.h"

inline double fermi(double x) { return x>30. ? exp(-x) : 1./(1.+exp(x)); } //avoid overflow issues
inline double fermiPrime(double x) { return 0.25*(std::pow(tanh(0.5*x), 2) - 1.); } //avoid overflow issues

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Calculate mu(Te) within Jellium model", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const double Z = inputMap.get("Z"); //number of electrons per unit cell
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	const double TeMin = inputMap.get("TeMin") * Kelvin; //electron temperature grid start
	const double TeMax = inputMap.get("TeMax") * Kelvin; //electron temperature grid stop
	const double TeStep = inputMap.get("TeStep") * Kelvin; //electron temperature grid spacing
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("Z = %lg\n", Z);
	logPrintf("dE = %lg\n", dE);
	logPrintf("TeMin = %lg\n", TeMin);
	logPrintf("TeMax = %lg\n", TeMax);
	logPrintf("TeStep = %lg\n", TeStep);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	const int bunchSize = 32;
	
	//Initialize temperature grid:
	std::vector<double> TeArr(int(ceil((TeMax-TeMin)/TeStep)));
	for(size_t iT=0; iT<TeArr.size(); iT++)
		TeArr[iT] = TeMin + TeStep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", TeArr.front()/Kelvin, TeArr.back()/Kelvin, TeArr.size());

	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	long nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	long nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	
	std::vector< std::vector< vector3<> > > kArrArr(nBunchesMine);	
	double kf = std::pow(3 * M_PI * M_PI * nJellium,1./3);
	double Emin =  0.5 * kF * kF; //set Emin = Efermi to start with
	double Emax =  0.5 * kF * kF; //set Emax = Efermi to start with
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{	//Generate a bunch of k-points:
		std::vector< vector3<> >& kArr = kArrArr[iBunch];
		kArr.resize(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();

		//Calcualte energy for the bunch of k-points:
		std::vector<double> Earr(kArr.size());
		for(size_t ik=0; ik<kArr.size(); ik++) 
		{	vector3<> k = kArr[ik];
			Earr[ik] = 0.5 * (k[0]*kAr[0]+k[1]*k[1]+k[2]*k[2]);
			Emin = std::min(Emin,Earr[ik]);
			Emax = std::min(Emax,Earr[ik]);
		}
	
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunchesMine)));
			logFlush();
		}
	}
	mpiUtil->allReduce(Emin, MPIUtil::ReduceMin);
	mpiUtil->allReduce(Emax, MPIUtil::ReduceMax);
	Emin -=10*dE; //add some margin
	Emax +=10*dE; //add some margin
		
	//Calculate mu at each temperature:
	diagMatrix dmu(TeArr.size(), 0.);
	int iTstart, iTstop; TaskDivision(TeArr.size(), mpiUtil).myRange(iTstart, iTstop);
	for(int iT=iTstart; iT<iTstop; iT++)
	{	const double Te = TeArr[iT], invTe = 1./Te;
		//Bisect for chemical potential:
		double& dmuCur = dmu[iT];
		double dmuMin = Emin - 10*Te;
		double dmuMax = Emax + 10*Te;
		dmuCur = 0.5*(dmuMin + dmuMax);
		const double tol = 1e-9*Te;
		while(dmuMax-dmuMin > tol)
		{	//calculate number of electrons at current Z:
			double nElectrons = 0.;
			for(size_t ie=0; ie<dos.out.size(); ie++)
			{	double Ei = Emin + ie*dE;
				double fi = fermi(invTe*(Ei - dmuCur));
				nElectrons += dE * dos.out[ie] * fi;
			}
			((nElectrons>Z) ? dmuMax : dmuMin) = dmuCur;
			dmuCur = 0.5*(dmuMin + dmuMax);
		}
	}
	dmu.allReduce(MPIUtil::ReduceSum);
	
	finalizeSystem();
}
