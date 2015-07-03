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
	initSystemCmdline(argc, argv, "Phonon DOS and heat capacity", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const double vL = inputMap.get("vL") * meter*invSeconds; //longitudinal speed of sound
	const double vT = inputMap.get("vT") * meter*invSeconds; //transverse speed of sound (assumed x2)
	const double domegaPh = inputMap.get("domegaPh") * eV; //phonon energy resolution (should be much smaller than TD)
	const double TlMin = inputMap.get("TlMin") * Kelvin; //electron temperature grid start
	const double TlMax = inputMap.get("TlMax") * Kelvin; //electron temperature grid stop
	const double TlStep = inputMap.get("TlStep") * Kelvin; //electron temperature grid spacing
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("vL = %lg\n", vL);
	logPrintf("vT = %lg\n", vT);
	logPrintf("domegaPh = %lg\n", domegaPh);
	logPrintf("TlMin = %lg\n", TlMin);
	logPrintf("TlMax = %lg\n", TlMax);
	logPrintf("TlStep = %lg\n", TlStep);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");
	
	//Initialize Wannier bandstructure:
	BandStruct bs("Wannier/wannier", 0., 2., "Wannier/totalE");

	//Initialize temperature grid:
	std::vector<double> TlArr(int(ceil((TlMax-TlMin)/TlStep)));
	for(size_t iT=0; iT<TlArr.size(); iT++)
		TlArr[iT] = TlMin + TlStep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", TlArr.front()/Kelvin, TlArr.back()/Kelvin, TlArr.size());
	
	//Calculate Debye temperatures / energies (same in atomic units):
	const double kD  = std::pow(6*M_PI*M_PI/fabs(det(R)), 1./3);
	const double TdebyeL = vL * kD; logPrintf("Longitudinal Debye energy: %3.0lf K (%.1lf meV)\n", TdebyeL/Kelvin, TdebyeL/(1e-3*eV));
	const double TdebyeT = vT * kD; logPrintf("Transverse Debye energy:   %3.0lf K (%.1lf meV)\n", TdebyeT/Kelvin, TdebyeT/(1e-3*eV));
	
	//Initialize phonon energy grid:
	double omegaPhMax = std::max(TdebyeL, TdebyeT);
	for(int i=0; i<100; i++)
	{	vector3<> k;
		for(int j=0; j<3; j++)
			k[j] = Random::uniform();
		omegaPhMax = std::max(omegaPhMax, bs.getPhononModes(k).back());
	}
	mpiUtil->allReduce(omegaPhMax, MPIUtil::ReduceMax);
	omegaPhMax *= 1.25; //add some margin
	Histogram dos(0, domegaPh, omegaPhMax); //phonon density of states
	logPrintf("Initialized phonon energy grid: 0 to %lg eV with %lu points.\n", (domegaPh*(dos.out.size()-1))/eV, dos.out.size());
	
	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nkMine = (ikStop-ikStart); //number of k's on current process
	int ikInterval = std::max(1, int(round(nkMine/50.))); //interval for reporting progress
	int nKpts = nkMine; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	if(nModes != 3) logPrintf("WARNING: the Debye estimates are only valid if nModes = 3.\n");
	
	//-------- Collect density of states, calculate Cl(Tl) ---------
	
	logPrintf("\nCollecting DOS: "); logFlush();
	const double dosWeight = (1./nKpts);
	for(int ik=0; ik<nkMine; ik++)
	{
		//Generate a random k-point:
		vector3<> k;
		for(int j=0; j<3; j++)
			k[j] = Random::uniform();
		
		//Collect DOS:
		const diagMatrix omegaPh = bs.getPhononModes(k);
		for(const double& omega: omegaPh)
			dos.addEvent(omega, dosWeight);
		
		//Print progress:
		if((ik+1) % ikInterval == 0)
		{	logPrintf("%d%% ", int(round((ik+1)*100./nkMine)));
			logFlush();
		}
	}
	logPrintf("done.\n"); logFlush();
	dos.allReduce(MPIUtil::ReduceSum);
	dos.print("phononDOS.dat", 1./eV, eV);
	
	//Calculate Cl at each temperature:
	diagMatrix Cl(TlArr.size(), 0.), ClDebye(TlArr.size(), 0.);
	int iTstart, iTstop; TaskDivision(TlArr.size(), mpiUtil).myRange(iTstart, iTstop);
	const double dosPrefacDebyeL = fabs(det(R)) / (2*M_PI*M_PI * std::pow(vL,3)); 
	const double dosPrefacDebyeT = fabs(det(R)) / (2*M_PI*M_PI * std::pow(vT,3)); 
	for(int iT=iTstart; iT<iTstop; iT++)
	{	const double Tl = TlArr[iT], invTl = 1./Tl;
		
		//Calculate lattice specific heat using DOS:
		double& ClCur = Cl[iT]; ClCur = 0.;
		double& ClDebyeCur = ClDebye[iT]; ClDebyeCur = 0.;
		for(size_t ie=1; ie<dos.out.size(); ie++) //omit zero energy phonons to avoid 0/0 error
		{	double omegaPh = ie*domegaPh;
			double x = invTl * omegaPh;
			double g = 1./(exp(x)-1.);
			double g_Tl = g*(g+1)*x/Tl; //dg/dTl
			ClCur += domegaPh * omegaPh * g_Tl  * dos.out[ie];
			//Debye approximation:
			double dosDebyeL = dosPrefacDebyeL * (omegaPh<TdebyeL ? omegaPh*omegaPh : 0.);
			double dosDebyeT = dosPrefacDebyeT * (omegaPh<TdebyeT ? omegaPh*omegaPh : 0.); //per mode
			double dosDebye = dosDebyeL + 2*dosDebyeT; //2 transverse modes
			ClDebyeCur += domegaPh * omegaPh * g_Tl  * dosDebye;
		}
	}
	Cl.allReduce(MPIUtil::ReduceSum);
	ClDebye.allReduce(MPIUtil::ReduceSum);

	if(mpiUtil->isHead())
	{	const double Omega = fabs(det(R));
		const double ClSI = Joule/(Kelvin*pow(meter,3));
		ofstream ofs("phononCl.dat");
		ofs << "#T[K] Cl[J/m^3K] ClDebye[J/m^3K]\n";
		for(size_t iT=0; iT<TlArr.size(); iT++)
			ofs << TlArr[iT]/Kelvin << '\t'
				<< Cl[iT]/(Omega*ClSI) << '\t'
				<< ClDebye[iT]/(Omega*ClSI) << '\n';
	}
	
	finalizeSystem();
}
