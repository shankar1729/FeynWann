#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"
#include "Units.h"
#include "Histogram.h"
#include "Epsilon.h"

inline double fermi(double x) { return x>30. ? exp(-x) : 1./(1.+exp(x)); } //avoid overflow issues
inline double fermiPrime(double x) { return 0.25*(std::pow(tanh(0.5*x), 2) - 1.); } //avoid overflow issues


int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Ab initio parameters for Transient Absorption analysis", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	double mu = inputMap.get("mu"); //initial guess only - will be calculated self-consistently in this executable
	const double Z = inputMap.get("Z"); //number of electrons per unit cell
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	const double TlMin = inputMap.get("TeMin") * Kelvin; //electron temperature grid start
	const double TlMax = inputMap.get("TeMax") * Kelvin; //electron temperature grid stop
	const double TlStep = inputMap.get("TeStep") * Kelvin; //electron temperature grid spacing
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("mu = %lg\n", mu);
	logPrintf("Z = %lg\n", Z);
	logPrintf("dE = %lg\n", dE);
	logPrintf("TlMin = %lg\n", TeMin);
	logPrintf("TlMax = %lg\n", TeMax);
	logPrintf("TlStep = %lg\n", TeStep);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");
	
	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(1); //assume cubic symmetry and only calculate x-axis
	Ahat[0] = vector3<complex>(1., 0., 0.);
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE", Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);

	//Initialize temperature grid:
	std::vector<double> TlArr(int(ceil((TlMax-TlMin)/TlStep)));
	for(size_t iT=0; iT<TlArr.size(); iT++)
		TlArr[iT] = TlMin + TlStep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", TlArr.front()/Kelvin, TlArr.back()/Kelvin, TlArr.size());
	
	//Initialize energy grid:
	diagMatrix Egamma = bs.getStates(vector3<>());
	double Emin = Egamma.front(), Emax = Egamma.back(); //eigenvalues are sorted
	for(int i=0; i<10; i++)
	{	std::vector<vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		for(const diagMatrix& E: Earr)
		{	Emin = std::min(Emin, E.front());
			Emax = std::max(Emax, E.back());
		}
	}
	mpiUtil->allReduce(Emin, MPIUtil::ReduceMin);
	mpiUtil->allReduce(Emax, MPIUtil::ReduceMax);
	Emin -= 10*dE; //add some margin
	Emax += 10*dE;
	Histogram dos(Emin, dE, Emax); //density of states
	logPrintf("Initialized energy grid: %lg to %lg eV with %lu points.\n", Emin/eV, (Emin+dE*(dos.out.size()-1))/eV, dos.out.size());
	
	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	long nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	long nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	int nBands = Egamma.nRows();
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	double phononPrefac0 = 4 * std::pow(M_PI,2) * spinWeight / (nKpairs*fabs(det(R))); //frequency independent part of prefac
	double directPrefac0 = 4 * std::pow(M_PI,2) * spinWeight / (nKpts*fabs(det(R))); //frequency independent part of prefac
	
	//Singularity extrapolation parameters
	double extrapCoeff[] = {-19./12, 13./3, -7./4 }; //account for constant, 1/eta and eta^2 dependence
	//double extrapCoeff[] = { -1, 2.}; //account for constant and 1/eta dependence
	const int nExtrap = sizeof(extrapCoeff)/sizeof(double);
	const double eta = 0.1*eV;

	//-------- Pass 1: collect density of states, calculate Cl(Tl) ---------
	
	logPrintf("\nCollecting DOS: "); logFlush();
	std::vector< std::vector< vector3<> > > kArrArr(nBunchesMine); //use exact same set of MC k-points in the two passes for consistency
	const double dosWeight = spinWeight*(1./nKpts);
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{
		//Generate a bunch of k-points:
		std::vector< vector3<> >& kArr = kArrArr[iBunch];
		kArr.resize(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		
		//Collect DOS:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		for(const diagMatrix& E: Earr)
			for(const double& Ei: E)
				dos.addEvent(Ei, dosWeight);
		
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunchesMine)));
			logFlush();
		}
	}
	logPrintf("done.\n"); logFlush();
	dos.allReduce(MPIUtil::ReduceSum);
	dos.print("phononDOS.dat", 1./eV, eV);
	
	//Calculate Cl at each temperature:
	//--- check enough bands to contain Z:
	double Zmax = 0.;
	for(const double& g: dos.out)
		Zmax += dE * g;
	if(Zmax < Z)
		die("Current DOS can only support %lg electrons > %lg electrons specified.\n", Zmax, Z);
	int iTstart, iTstop; TaskDivision(TeArr.size(), mpiUtil).myRange(iTstart, iTstop);
	for(int iT=iTstart; iT<iTstop; iT++)
	{	const double Tl = TlArr[iT], invTl = 1./Tl;
		
		//Calculate lattice specific heat:
		double& ClCur = Cl[iT];
		ClCur = 0.;
		for(size_t ie=0; ie<dos.out.size(); ie++)
		{	double Ei = Emin + ie*dE;
			double x = invTl*Ei;
			double expFac = 1/std::pow(exp(x)-1,2);
			CeCur += dE * Ei * expFac  * dos.out[ie];
		}
	}
	Ce.allReduce(MPIUtil::ReduceSum);
	

	if(mpiUtil->isHead())
	{	const double Omega = fabs(det(R));
		const double ClSI = Joule/(Kelvin*pow(meter,3));
		ofstream ofs("phononCl.dat");
		ofs << "#T[K] Cl[J/m^3K] \n";
		for(size_t iT=0; iT<TlArr.size(); iT++)
			ofs << TlArr[iT]/Kelvin << '\t'
				<< Cl[iT]/(Omega*ClSI) << '\n';
		
	}
	
	finalizeSystem();
}
