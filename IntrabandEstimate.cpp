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
 		
	ofstream ofs("drudeEpsilon.dat");
        for(double omega=0.01*eV; omega<10*eV; omega+=0.01*eV)
        {       complex drudeEps = 1 - omega_p^2 / (omega^2 + i*omega/tau);
                ofs << omega << '\t' << real(drudeEps) << '\t' << imag(drudeEps) << '\t' << '\n';
	}

	//Initialize dielectric model:
	Epsilon eps("drudeEpsilon.dat");
	
	//Output plasmon Gamma contributions from ImEps contributions:
	ImEpsPhonon.allReduce(MPIUtil::ReduceSum);
	ImEps2eh.allReduce(MPIUtil::ReduceSum);
	mpiUtil->allReduce(eeNumSim, MPIUtil::ReduceSum);
	mpiUtil->allReduce(ePhNumSim, MPIUtil::ReduceSum);
	if(mpiUtil->isHead())
	{	ofstream ofs("Drude-IntrabandEstimate.dat");
		for(size_t i=0; i<ImEpsPhonon.out.size(); i++)
		{	double omega = ImEpsPhonon.Emin + i * ImEpsPhonon.dE;
			eps.setFrequency(omega, false);
			double prefac = eps.exptLinewidth()/eps.epsilon.imag(); //ratio between plasmon linewidth and imaginary part of epsilon
			double eeImEps = ImEps2eh.out[i] * eeGamma0 / (std::pow(omega, 4) * eeNumSim / eeDen);
			double ePhImEps = ImEpsPhonon.out[i] * ePhGamma0 / (std::pow(omega, 4) * ePhNumSim / ePhDen);
			ofs << omega/eV << '\t' << prefac*eeImEps/eV << '\t' << prefac*ePhImEps/eV << '\n';
		}
	}
	
	finalizeSystem();
}
