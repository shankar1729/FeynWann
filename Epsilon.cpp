#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include "Epsilon.h"
#include "Units.h"

Epsilon::Epsilon(string inputFilename)
{	//Get the Epsilon parameters
	std::ifstream epsFile(inputFilename.c_str());
	if(!epsFile.is_open())
		die("Could not open system file '%s' for reading.\n", inputFilename.c_str());
	logPrintf("---- Initializing dielectric model ----\n");
	while(!epsFile.eof())
	{	string line; getline(epsFile, line); //line-by-line processing (comments can now be inline)
		trim(line);
		if(line[0]=='#' || !line.length()) continue; //ignore comments and blank lines
		istringstream iss(line);
		string name; iss >> name;
		if(name == "omega_p")
		{	iss >> omega_p;
			omega_p *= eV;
			logPrintf("omega_p = %lg Eh\n", omega_p);
		}
		else if(name == "epsParams")
		{	double f, Gamma, omega0;
			iss >> f >> Gamma >> omega0;
			Gamma *= eV;
			omega0 *= eV;
			epsParams.push_back(vector3<>(f, Gamma, omega0));
			logPrintf( "epsParams:  f = %lg  Gamma = %lg Eh  omega0 = %lg Eh\n", f, Gamma, omega0);
		}
		else
		{	die("Error: invalid command '%s' in dielectric parameter file '%s'.\n", name.c_str(), inputFilename.c_str())
		}
	}
	logPrintf("\n");
	epsFile.close();
}

void Epsilon::setFrequency(double omegaIn)
{	// Calculate the dielectric at omega = Eplasmon
	double omega = omegaIn;
	complex omegaEpsilonPrime(1.0,0.0), one(1.0,0.0), den;
	double num;
	vector3<double> epsParam;
	complex I(0.0,1.0);
	epsilon = one;
	for(size_t iPole = 0; iPole < epsParams.size(); iPole++)
	{	epsParam = epsParams[iPole];
		num = epsParam[0]*(std::pow(omega_p,2));
		den = one/(std::pow(epsParam[2],2) - omega*omega - I * omega * epsParam[1]);
		epsilon += num *den;
		omegaEpsilonPrime += num * den*den * (std::pow(epsParam[2],2) + omega*omega);
	}

	// Plasmon mode deatils
	const double c = 1./7.29735257e-3;
	double realEpsilon = real(epsilon);
	k = (omega/c) * sqrt(realEpsilon/(realEpsilon+1));
	modGammaPlus = sqrt(k*k - (omega/c)*(omega/c));
	modGammaMinus = sqrt(k*k - (realEpsilon*(omega/c)*(omega/c)));
	Lquant = (1/(4*std::pow(modGammaPlus,3))) * (std::pow(modGammaPlus,2) + k*k + std::pow((omega/c),2))
		+ (1/(4*std::pow(modGammaMinus,3))) * ((std::pow(modGammaMinus,2)+k*k)*real(omegaEpsilonPrime)+(std::pow((real(epsilon*omega/c)),2)));
	logPrintf("omega = %lg  k = %lg  modGammaMinus  = %lg  modGammaPlus = %lg  Lquant = %lg\n", omega, k, modGammaMinus, modGammaPlus, Lquant);
}
