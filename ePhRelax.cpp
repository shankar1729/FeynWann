#include "Units.h"
#include "InputMap.h"
#include "Interp1.h"
#include "Histogram.h"
#include <core/Util.h>
#include <core/Operators.h>
#include <electronic/matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <fstream>

struct ePhRelax
{
	Interp1 dos, dosPh;
	diagMatrix f0, fPert; //Initial Fermi and photon-perturbed distributions
	
	ePhRelax(int argc, char** argv)
	{
		//Parse the command line:
		string inputFilename; bool dryRun, printDefaults;
		initSystemCmdline(argc, argv, "Electron-phonon relaxation using Boltzmann equation", inputFilename, dryRun, printDefaults);

		//Get the system parameters (mu, T, lattice vectors etc.)
		InputMap inputMap(inputFilename);	
		const double Z = inputMap.get("Z"); //number of electrons per unit cell
		const double T = inputMap.get("T") * Kelvin; //initial temperature in Kelvin (electron and lattice)
		const double Uabs = inputMap.get("Uabs") * Joule/std::pow(meter,3); //absorbed laser energy per unit volume in Joule/meter^3
		const double Eplasmon = inputMap.get("Eplasmon") * eV; //incident photon energy in eV
		const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
		double detR = fabs(det(R));
		
		logPrintf("\nInputs after conversion to atomic units:\n");
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
		
		//Determine initial Fermi distribution:
		f0.resize(dos.xGrid.size());
		//--- Bisect for chemical potential:
		double dmuMin = dos.xGrid.front() - 10*T;
		double dmuMax = dos.xGrid.back() + 10*T;
		double dmu = 0.5*(dmuMin + dmuMax);
		const double tol = 1e-9*T;
		while(dmuMax-dmuMin > tol)
		{	//calculate number of electrons at current Z:
			double nElectrons = 0.;
			for(size_t ie=0; ie<dos.xGrid.size(); ie++)
			{	const double& Ei = dos.xGrid[ie];
				double& fi = f0[ie];
				fi = fermi((Ei - dmu)/T);
				nElectrons += dos.dx * dos.yGrid[0][ie] * fi * detR;
			}
			((nElectrons>Z) ? dmuMax : dmuMin) = dmu;
			dmu = 0.5*(dmuMin + dmuMax);
		}
		logPrintf("Initial Fermi distribution: dmu = %le eV\n", dmu/eV);
		//--- calculate density of states at the Fermi level:
		double dos0 = 0.;
		for(size_t ie=0; ie<dos.xGrid.size(); ie++)
			dos0 += dos.dx * dos.yGrid[0][ie] * fermiPrime((dos.xGrid[ie] - dmu)/T) * (-1./T);
		logPrintf("Density of states at Fermi level = %le /eV-cell\n", dos0*(eV*detR));
		
		//Perturb by photon-induced carrier density:
		//--- read carrier distributions from plasmonDecay:
		Histogram2D distribDirect("carrierDistribAll-direct.dat", 1./eV, 1./eV, 1.);
		Histogram2D distribPhonon("carrierDistribAll-phonon.dat", 1./eV, 1./eV, 1.);
		if(Eplasmon < distribDirect.omegaMin || Eplasmon > distribDirect.omegaMin + (distribDirect.nomega-1)*distribDirect.domega)
			die("Plasmon energy is out of the range available in carrierDistribAll-direct.dat")
		if(Eplasmon < distribPhonon.omegaMin || Eplasmon > distribPhonon.omegaMin + (distribPhonon.nomega-1)*distribPhonon.domega)
			die("Plasmon energy is out of the range available in carrierDistribAll-phonon.dat")
		//--- interpolate to required photon energy and carrier eenergy grid:
		fPert.resize(dos.xGrid.size());
		double Upert = 0.;
		for(size_t ie=0; ie<dos.xGrid.size(); ie++)
		{	const double& Ei = dos.xGrid[ie];
			double dni = distribDirect.interp1(Ei, Eplasmon) + distribPhonon.interp1(Ei, Eplasmon); //induced carrier number change at given energy
			fPert[ie] = dni / std::max(dos.yGrid[0][ie], 1e-3*dos0); //divide by DOS to get the effective filling change (regularize to avoid Infs)
			Upert += dni * Ei * dos.dx; //calculate energy of perturbation
		}
		fPert *= Uabs / Upert; //normalize to match absorbed laser energy per unit volume
		fPert += f0; //add initial Fermi distribution
	}
	
	//Calculate lattice specific heat
	inline double Cl(double Tl) const
	{	assert(dosPh.xMin==0.);
		const double& domegaPh = dosPh.dx;
		double result = 0.;
		for(size_t ie=1; ie<dosPh.xGrid.size(); ie++) //omit zero energy phonons to avoid 0/0 error
		{	double omegaPh = ie*domegaPh;
			double x = omegaPh/Tl;
			double g = 1./(exp(x)-1.);
			double g_Tl = g*(g+1)*x/Tl; //dg/dTl
			result += domegaPh * omegaPh * g_Tl  * dosPh.yGrid[0][ie];
		}
		return result;
	}
	
	inline double fermi(double x) { return x>30. ? exp(-x) : 1./(1.+exp(x)); } //avoid overflow issues
	inline double fermiPrime(double x) { return 0.25*(std::pow(tanh(0.5*x), 2) - 1.); } //avoid overflow issues
};

int main(int argc, char** argv)
{	ePhRelax e(argc, argv);
	
	finalizeSystem();
};
