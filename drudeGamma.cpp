#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include "BandStruct.h"
#include "LineWidth.h"
#include "Epsilon.h"
#include "InputMap.h"
#include <core/Units.h>
#include "Histogram.h"
#include <complex>

int main(int argc, char** argv)
{	
	InitParams ip = BandStruct::initialize(argc, argv, "Calibrated jellium estimates of intraband processes");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const double EplasmonMax = inputMap.get("EplasmonMax") * eV;
	const double T = inputMap.get("T") * Kelvin;
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	const double Zjellium = inputMap.get("Zjellium");
	const int tau = inputMap.get("tau")*fs;
	const double rho = inputMap.get("rho") * (Ohm * meter);
	const double sigma_o = 1/rho;

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("EplasmonMax = %lg\n", EplasmonMax);
	logPrintf("T = %lg\n", T);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	logPrintf("Zjellium = %lg\n", Zjellium);
	logPrintf("tau = %lg fs\n", tau/fs);
	logPrintf("rho = %lg ohm-m\n", rho/(Ohm*meter));
	logPrintf("sigma_o = %lg 1/(ohm-m)\n", sigma_o*Ohm*meter);
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Jellium parameters:
	double nJellium = Zjellium / fabs(det(R));
	double omegaPsq = 4*M_PI * nJellium; logPrintf("Jellium plasma frequency: %lg eV\n", sqrt(omegaPsq)/eV);
	double kF = std::pow(3*M_PI*M_PI*nJellium, 1./3);
	double vF = kF; //same as kF in atomic units
	
	//Initialize dielectric model:
	Epsilon eps("Wannier/epsilon.dat");
	complex epsilon;

	ofstream ofs1("drudeEpsilon.dat");
	ofstream ofs2("drudeGamma.dat");

	double domega = T;
	Histogram h(0, domega, EplasmonMax); //dummy histogram to match plasmonDceay grid exactly
	for(size_t jomega=0; jomega<h.out.size(); jomega++)
	{	double omega = jomega ? jomega*domega : 1e-3*domega; //avoid exact zero in calculation
		eps.setFrequency(omega, false);
		complex sigma = complex(sigma_o) / complex(1,-omega*tau);
		epsilon = 1.+ 4*M_PI*complex(-imag(sigma),real(sigma)) / complex(omega);
		double imEpsSurface = 0.75 * omegaPsq * eps.modGammaMinus * vF / std::pow(omega,3) //Khurgin's version
			* 2./(1.+std::pow(eps.modGammaMinus/eps.k, 2)) //Our main correction
			* ( 1. //+ higher-order corrections:
				- (1./4)*std::pow(omega/(0.5*kF*kF),2)*log(2*kF*kF/omega) //high-frequency correction
				+ (1./6)*std::pow(eps.modGammaMinus/eps.k,2)/(1.+std::pow(omega/(vF*eps.modGammaMinus),2)) //low-frequency correction
			);
		double prefac = eps.exptLinewidth()/eps.epsilon.imag(); //ratio between plasmon linewidth and imaginary part of epsilon
		omega = jomega*domega; //output exact zero
		ofs1 << omega/eV << '\t' << real(epsilon) << '\t' << imag(epsilon) << '\n'; //Output energies in eV units
		ofs2 << omega/eV << '\t' << prefac*imag(epsilon)/eV << '\t' << prefac*imEpsSurface/eV << '\n'; //Output energies in eV units
	}

	finalizeSystem();
}
