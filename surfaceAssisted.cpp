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
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Numericla check of surface-assisted expressions", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const double EplasmonMax = inputMap.get("EplasmonMax") * eV;
	const double T = inputMap.get("T") * Kelvin;
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	const double Zjellium = inputMap.get("Zjellium");

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("EplasmonMax = %lg\n", EplasmonMax);
	logPrintf("T = %lg\n", T);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	logPrintf("Zjellium = %lg\n", Zjellium);
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Jellium parameters:
	const double nJellium = Zjellium / fabs(det(R));
	const double kF = std::pow(3*M_PI*M_PI*nJellium, 1./3);
	const double vF = kF; //same in atomic units
	
	//Initialize dielectric model:
	Epsilon eps("Wannier/epsilon.dat");
	
	//Initialize sampling parameters:
	const double qMax = sqrt(kF*kF + 2*EplasmonMax);
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	
	//Initialize histograms:
	Histogram GammaSurface(T, T, EplasmonMax-T);
	
	//Main MC sampling loop
	for(int ik=ikStart; ik<ikStop; ik++)
	{	//Generate random q with length < kF
		vector3<> q;
		while(true)
		{	for(int j=0; j<3; j++) //draw from bounding box
				q[j] = Random::uniform(-kF,kF);
			if(q.length() < kF) break; //done if inside sphere
		}
		//Generate random |qz'| < qMax
		vector3<> qPrime = q;
		qPrime[2] = Random::uniform(-qMax, qMax);
		//Get energy and momentum conserving omega and k self-consistently:
		double omega = 0.5*(qPrime.length_squared() - q.length_squared());
		if(omega<0 || omega>EplasmonMax) continue;
		eps.setFrequency(omega, false);
		while(true)
		{	if(!std::isfinite(eps.k)) break; //avoid over-damped region
			qPrime[0] = q[0] + eps.k; //exact xy-momentum conservation with plasmon
			double omegaNext = 0.5*(qPrime.length_squared() - q.length_squared());
			if(fabs(omegaNext-omega)<1e-8) break;
			omega = omegaNext;
			eps.setFrequency(omega, false);
		}
		//Filter unneeded events:
		if(qPrime.length() < kF) continue; //need unoccupied final state
		if(omega<0 || omega>EplasmonMax) continue;
		if(!std::isfinite(eps.k) || !std::isfinite(eps.modGammaMinus)) continue;
		//Collect event contributions:
		double qzSq = std::pow(q[2],2), qzPrimeSq = std::pow(qPrime[2],2);
		double gamma = eps.modGammaMinus, gammaSq = std::pow(gamma,2);
		double M = q[2]*qPrime[2]
			* (eps.k*(qzSq - qzPrimeSq) - (2*q[0] + eps.k)*gammaSq)
			/ (std::pow(qzSq + qzPrimeSq + gammaSq, 2) - 4.*qzSq*qzPrimeSq);
		double weight = (8.*omega*gamma*qMax)*(M*M)/((eps.k*eps.k + gammaSq)*nKptsN1);
		GammaSurface.addEvent(omega, weight);
	}
	GammaSurface.allReduce(MPIUtil::ReduceSum);
	
	if(mpiUtil->isHead())
	{	ofstream ofs("surfaceAssisted.dat");
		ofs << "omega Numerical Khurgin Analytical Analytical2\n";
		for(size_t i=0; i<GammaSurface.out.size(); i++)
		{	double omega = GammaSurface.Emin + i*GammaSurface.dE;
			eps.setFrequency(omega, false);
			double GammaRef = 0.75*vF*eps.modGammaMinus; //Khurgin
			double GammaCorrected = GammaRef * 2./(1.+std::pow(eps.modGammaMinus/eps.k, 2)); //our corrected expression (simpler version)
			double GammaCorrected2 = GammaCorrected
				* ( 1. - (1./4)*std::pow(omega/(0.5*kF*kF),2)*log(2*kF*kF/omega)
					+ (1./6)*std::pow(eps.modGammaMinus/eps.k,2)/(1.+std::pow(omega/(vF*eps.modGammaMinus),2)) );
			ofs << omega/eV
				<< '\t' << GammaSurface.out[i]*fs
				<< '\t' << GammaRef*fs
				<< '\t' << GammaCorrected*fs
				<< '\t' << GammaCorrected2*fs
				<< '\n';
		}
	}
	finalizeSystem();
}
