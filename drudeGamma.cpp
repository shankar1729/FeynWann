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
	const double Zjellium = inputMap.get("Zjellium");
	const int tau = inputMap.get("tau")*fs;
	const double rho = inputMap.get("rho") * (Ohm * meter);
	const double sigma_o = 1/rho;

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("EplasmonMax = %lg\n", EplasmonMax);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	logPrintf("Zjellium = %lg\n", Zjellium);
	logPrintf("tau = %lg fs\n", tau/fs);
	logPrintf("rho = %lg ohm-m\n", rho/(Ohm*meter));
	logPrintf("sigma_o = %lg 1/(ohm-m)\n", sigma_o*Ohm*meter);
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Jellium parameters:
	double nJellium = Zjellium / fabs(det(R));
	double omegaPsq = 4*M_PI * nJellium; logPrintf("Jellium plasma frequency: %lg eV\n", sqrt(omegaPsq)/eV);
	double vF = std::pow(3*M_PI*M_PI*nJellium, 1./3); //same as kF in atomic units
	
	//Initialize dielectric model:
	Epsilon eps("Wannier/epsilon.dat");
	complex epsilon;

	ofstream ofs1("drudeEpsilon.dat");
	ofstream ofs2("drudeGamma.dat");	

	double gaussMargin = 5*T;
	for(double omega = gaussMargin; omega <= EplasmonMax-gaussMargin; omega += T)
	{	eps.setFrequency(omega, false);
		complex sigma = complex(sigma_o) / complex(1,-omega*tau);
		epsilon = 1.+ 4*M_PI*complex(-imag(sigma),real(sigma)) / complex(omega);
		double imEpsSurface = 0.75 * omegaPsq * eps.modGammaMinus * vF / std::pow(omega,3);
		double prefac = eps.exptLinewidth()/eps.epsilon.imag(); //ratio between plasmon linewidth and imaginary part of epsilon
		ofs1 << omega/eV << '\t' << real(epsilon) << '\t' << imag(epsilon) << '\n'; //Output energies in eV units
		ofs2 << omega/eV << '\t' << prefac*imag(epsilon)/eV << '\t' << prefac*imEpsSurface/eV << '\n'; //Output energies in eV units
	}

	finalizeSystem();
}
