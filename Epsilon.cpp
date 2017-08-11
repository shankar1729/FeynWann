#include <core/Util.h>
#include <core/Spline.h>
#include <core/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include "Epsilon.h"
#include <core/Units.h>

Epsilon::Epsilon(string filename)
{	logPrintf("\n---- Initializing dielectric model ----\n");
	
	//Read data from file:
	std::ifstream ifs(filename.c_str());
	if(!ifs.is_open())
		die("Could not open dielectric file '%s' for reading.\n", filename.c_str());
	std::vector<double> omegaArr, reArr, imArr;
	while(!ifs.eof())
	{	double omega, re, im;
		ifs >> omega >> re >> im;
		if(!ifs.good()) break;
		omegaArr.push_back(omega);
		reArr.push_back(re);
		imArr.push_back(im);
	}
	ifs.close();
	
	//Check frequency grid:
	omegaStart = omegaArr[0];
	omegaStop = omegaArr.back();
	domega = (omegaStop - omegaStart) / (omegaArr.size() - 1);
	for(size_t i=0; i<omegaArr.size(); i++)
		if(fabs(omegaStart + i*domega - omegaArr[i]) > 1e-3*domega)
			die("Dielectric file '%s' does not have a uniform frequency grid.\n", filename.c_str());
	
	//Report frequencies:
	logPrintf("omegaMax: %lg eV\n", omegaStop/eV);
	#define REPORTomega(name, epsCut) \
		for(size_t i=0; i<omegaArr.size()-1; i++) \
		{	if(reArr[i]<epsCut && reArr[i+1]>epsCut) \
			{	double omegaCut = (omegaArr[i]*(reArr[i+1]-epsCut) + omegaArr[i+1]*(epsCut-reArr[i])) / (reArr[i+1] - reArr[i]); \
				logPrintf("omega%s: %lg eV\n", #name, omegaCut/eV); \
				break; \
			} \
		}
	REPORTomega(SPP, -1.)
	REPORTomega(Dip, -2.)
	#undef REPORTomega
	
	//Initialize Drude fit for low frequencies:
	Gamma0 = omegaStart * imArr[0] / (1. - reArr[0]);
	OmegaPsq = hypot(1.-reArr[0], imArr[0]) * omegaStart * hypot(omegaStart,Gamma0);

	//Initialize quintic spline for remainder:
	epsilonStart = complex(reArr[0], imArr[0]);
	for(double& re: reArr) re -= epsilonStart.real();
	for(double& im: imArr) im -= epsilonStart.imag();
	coeffRe = QuinticSpline::getCoeff(reArr, true);
	coeffIm = QuinticSpline::getCoeff(imArr, true);
	
	logPrintf("\n");
}

void Epsilon::setFrequency(double omegaIn, bool print)
{	//Check frequency:
	omega = omegaIn;
	if(omega <= 0. || omega > omegaStop)
	{	epsilon = NAN;
		modGammaPlus = modGammaMinus = Lquant = k = NAN;
		if(print) logPrintf("omega = %lg  OUT OF RANGE\n", omega);
		return;
	}
	
	//Calculate epsilon and d(omega epsilon)/domega
	double omegaEpsilonPrime;
	if(omega < omegaStart)
	{	complex den = complex(1) / complex(omega,Gamma0); // = 1/(omega + I*Gamma0)
		epsilon = 1. - (OmegaPsq/omega)*den;
		omegaEpsilonPrime = 1. + OmegaPsq*(den*den).real();
	}
	else
	{	double x = (omega - omegaStart) / domega;
		complex deps(QuinticSpline::value(coeffRe.data(), x), QuinticSpline::value(coeffIm.data(), x));
		double deps_x = QuinticSpline::deriv(coeffRe.data(), x); //need real part only so far
		epsilon = epsilonStart + deps;
		omegaEpsilonPrime = epsilon.real() + omega * deps_x / domega;
	}

	//Plasmon mode details
	const double c = 1./7.29735257e-3;
	double epsRe = real(epsilon);
	k = (omega/c) * sqrt(epsRe/(epsRe+1));
	modGammaPlus = sqrt(k*k - (omega/c)*(omega/c));
	modGammaMinus = sqrt(k*k - (epsRe*(omega/c)*(omega/c)));
	Lquant = (1/(4*std::pow(modGammaPlus,3))) * (std::pow(modGammaPlus,2) + k*k + std::pow((omega/c),2))
		+ (1/(4*std::pow(modGammaMinus,3))) * ((std::pow(modGammaMinus,2)+k*k)*omegaEpsilonPrime+(std::pow((real(epsilon*omega/c)),2)));
	
	if(print) logPrintf("omega = %lg  k = %lg  modGammaMinus  = %lg  modGammaPlus = %lg  Lquant = %lg\n", omega, k, modGammaMinus, modGammaPlus, Lquant);
}

double Epsilon::exptLinewidth() const
{	return (omega / (2.*modGammaMinus*Lquant)) * (1. + std::pow(k/modGammaMinus, 2)) * imag(epsilon);
}
