#include "Units.h"
#include "InputMap.h"
#include "Interp1.h"
#include <core/Util.h>
#include <core/Operators.h>
#include <electronic/matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <fstream>

struct ePhRelax
{
	Interp1 dos, dosPh;
	
	ePhRelax(int argc, char** argv)
	{
		//Parse the command line:
		string inputFilename; bool dryRun, printDefaults;
		initSystemCmdline(argc, argv, "Electron-phonon relaxation using Boltzmann equation", inputFilename, dryRun, printDefaults);

		//Get the system parameters (mu, T, lattice vectors etc.)
		InputMap inputMap(inputFilename);	
		double mu = inputMap.get("mu"); //initial guess only - will be calculated self-consistently in this executable
		const double Z = inputMap.get("Z"); //number of electrons per unit cell
		const double T = inputMap.get("T") * Kelvin; //initial temperature in Kelvin (electron and lattice)
		const double Uabs = inputMap.get("Uabs") * Joule/std::pow(meter,3); //absorbed laser energy per unit volume in Joule/meter^3
		const double Eplasmon = inputMap.get("Eplasmon") * eV; //incident photon energy in eV
		const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
		double detR = fabs(det(R));
		
		logPrintf("\nInputs after conversion to atomic units:\n");
		logPrintf("mu = %lg\n", mu);
		logPrintf("Z = %lg\n", Z);
		logPrintf("T = %lg\n", T);
		logPrintf("Uabs = %lg\n", Uabs);
		logPrintf("Eplasmon = %lg\n", Eplasmon);
		logPrintf("R:\n");
		R.print(globalLog, " %lg ");
		if(dryRun)
		{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
			finalizeSystem();
			exit(0);
		}
		logPrintf("\n");
		
		//Read electron and phonon DOS (and convert to atomic units and per-unit volume):
		dos.init("dos.dat", eV, 1./(detR*eV));
		dosPh.init("phononDOS.dat", eV, 1./(detR*eV));
	}
	
	//Calculate lattice specific heat
	inline double Cl(double Tl) const
	{
	}
};

int main(int argc, char** argv)
{	ePhRelax e(argc, argv);
	
	finalizeSystem();
};
