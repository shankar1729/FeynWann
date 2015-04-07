#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include "BandStruct.h"
#include "LineWidth.h"
#include "Epsilon.h"
#include "InputMap.h"
#include "Units.h"
#include "Histogram.h"
#include <complex>

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Calibrated jellium estimates of intraband processes", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const double EplasmonMax = inputMap.get("EplasmonMax") * eV;
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	const int z = inputMap.get("z");
	const double sigma = inputMap.get("sigma") / Ohm;

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("EplasmonMax = %lg\n", EplasmonMax);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	logPrintf("z = %d\n", z);
	logPrintf("sigma = %lg\n", sigma);
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	const double cellVol = fabs(det(R)); //unit cell vol
	const double n = z / cellVol; // electron density
	const double tau = sigma /n;
	const double omega_p = 4*M_PI*n;
	//complex i = sqrt(-1);
 	
        //Initialize dielectric model:
        Epsilon eps("Wannier/epsilon.dat");
	double reDrudeEps;
	double imDrudeEps;

	ofstream ofs("drudeEpsilon.dat");
        for(double omega=0.01*eV; omega<10*eV; omega+=0.01*eV)
        {       eps.setFrequency(omega,false);
		reDrudeEps = omega_p*omega_p / (omega*omega + 1/(tau*tau));
		imDrudeEps = omega_p*omega_p / (omega*omega*omega*tau + omega/tau);
		//complex drudeEps = 1 - omega_p*omega_p / (omega*omega + i*omega/tau);
		double prefac = eps.exptLinewidth()/eps.epsilon.imag(); //ratio between plasmon linewidth and imaginary part of epsilo
                ofs << omega << '\t' << reDrudeEps << '\t' << imDrudeEps << '\t' << prefac*imDrudeEps << '\n';
                //ofs << omega << '\t' << real(drudeEps) << '\t' << imag(drudeEps) << '\t' << prefac*imag(drudeEps) << '\n';
	}

	finalizeSystem();
}
