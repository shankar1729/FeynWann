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
	const double Z = inputMap.get("Zjellium"); //jellium number of electrons per unit cell
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	const double TeMin = inputMap.get("TeMin") * Kelvin; //electron temperature grid start
	const double TeMax = inputMap.get("TeMax") * Kelvin; //electron temperature grid stop
	const double TeStep = inputMap.get("TeStep") * Kelvin; //electron temperature grid spacing
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("Zjellium = %lg\n", Z);
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

	//Initialize temperature grid:
	std::vector<double> TeArr(int(ceil((TeMax-TeMin)/TeStep)));
	for(size_t iT=0; iT<TeArr.size(); iT++)
		TeArr[iT] = TeMin + TeStep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", TeArr.front()/Kelvin, TeArr.back()/Kelvin, TeArr.size());

	const double Omega = fabs(det(R)); //unit cell volume
	const double nJellium = Z / Omega;
	const double kF = std::pow(3 * M_PI * M_PI * nJellium,1./3);
	const double Ef = 0.5 * kF * kF;

	diagMatrix dmu(TeArr.size(), 0.);
	int iTstart, iTstop; TaskDivision(TeArr.size(), mpiUtil).myRange(iTstart, iTstop);
	for(int iT=iTstart; iT<iTstop; iT++)
	{	const double Te = TeArr[iT], invTe = 1./Te;
		//Initialize energy grid:
		double Emin = 0;
		double Emax = Ef + 5*Te;
		double dE = 0.25*Te; //different from dE for carruer energies used later
		//Bisect for chemical potential:
		double& dmuCur = dmu[iT];
		double dmuMin = Emin - 10*Te;
		double dmuMax = Emax + 10*Te;
		dmuCur = 0.5*(dmuMin + dmuMax);
		const double tol = 1e-9*Te;
		while(dmuMax-dmuMin > tol)
		{	//calculate number of electrons at current Z:
			double nElectrons = 0.;
			for(double E=Emin; E<Emax; E+=dE)
			{	double f = fermi(invTe*(E - dmuCur));
				nElectrons += dE * f * sqrt(2*E)*Omega/(M_PI*M_PI);
			}
			((nElectrons>Z) ? dmuMax : dmuMin) = dmuCur;
			dmuCur = 0.5*(dmuMin + dmuMax);
		}
	}
	dmu.allReduce(MPIUtil::ReduceSum);
	
	if(mpiUtil->isHead())
	{	ofstream ofs("mu_Te_Jellium.dat");
                ofs << "#T[K] dmu[eV] \n";
                for(size_t iT=0; iT<TeArr.size(); iT++)
                        ofs << TeArr[iT]/Kelvin << '\t'
                                << dmu[iT]/eV << '\n';
	}

	finalizeSystem();
}
