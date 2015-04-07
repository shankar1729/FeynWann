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
	const double sigma = inputMap.get("sigma") / (Ohm * meter);

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
	const double omega_p = sqrt(4*M_PI*n);
	const double omega_pSqr = 4*M_PI*n;
	logPrintf("cellVol = %lg\n", cellVol); 	
	logPrintf("elctron denisty = %lg\n", n);
	logPrintf("tau = %lg\n", tau);
	logPrintf("omega_p = %lg Eh\n", omega_p);

        //Initialize dielectric model:
        Epsilon eps("Wannier/epsilon.dat");
	complex epsilon;

	ofstream ofs("drudeEpsilon.dat");
        for(double omega=0.01; omega<6; omega+=0.01)
        {       eps.setFrequency(omega*eV,false);
		complex denom = complex(1) / complex(omega,1/tau);
		epsilon = 1. - (omega_pSqr/omega)*denom;
		double prefac = eps.exptLinewidth()/eps.epsilon.imag(); //ratio between plasmon linewidth and imaginary part of epsilo
                ofs << omega*eV << '\t' << real(epsilon) << '\t' << imag(epsilon) << '\t' << prefac*imag(epsilon) << '\n';
	}

	finalizeSystem();
}
