#include "epsilon.h"
#include <core/Util.h>
#include <core/Units.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
// -------------------------------------------- class epsilon --------------------------------------------

// Constructor
epsilon::epsilon(string inputFilename, double E)
{	//Get the epsilon parameters
	Eplasmon = E;
	std::ifstream systemFile(inputFilename.c_str());
        if(!systemFile.is_open())
                die("Could not open system file '%s' for reading.\n", inputFilename.c_str());
        while(!systemFile.eof())
        {	string line; getline(systemFile, line); //line-by-line processing (comments can now be inline)
		//trim(line);
		istringstream iss(line);
		string name; double val;
		if(iss >> name >> val)
		{	if( name == "omega_p")
        		{	omega_p = val*eV;
				logPrintf("omega_p = %lg\n", omega_p); 
			}	
		}
		string ename; double val1, val2, val3;
		if(iss >> ename >> val1 >> val2 >> val3)
		{	if (ename == "epsParams")
			{	epsParams.push_back(vector3<>(val1, val2*eV, val3*eV));
				logPrintf( "epsParams1 = %lg 2 =  %lg 3 = %lg\n", val1, val2, val3);
			}	
		}
	}
        systemFile.close();
}

double epsilon::getLquant()
{	// Calculate the dielectric at omega = Eplasmon
	double omega = Eplasmon;
	complex epsilon(1.0,0.0), omegaEpsilonPrime(1.0,0.0), one(1.0,0.0), den;
	double num;
	vector3<double> epsParam;
	complex I(0.0,1.0);
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
	logPrintf("k = %lg omega = %lg c = %lg modGammaMinus  = %lg modGammaPlus = %lg\n", k, omega, c, modGammaMinus, modGammaPlus);
	double Lquant = (1/(4*std::pow(modGammaPlus,3))) * (std::pow(modGammaPlus,2) + k*k + std::pow((omega/c),2)) + (1/(4*std::pow(modGammaPlus,3))) * ((std::pow(modGammaMinus,2)+k*k)*real(omegaEpsilonPrime)+(std::pow((real(epsilon*omega/c)),2)));
	return Lquant;
}


