#include "histogram.h"
#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>


//-------------------------- class histogram ------------------------------------------------

// Constructor
histogram::histogram(double Emin, double dE, double Emax)
{	// Make uniform energy gric
	int numBins = (Emax - Emin)/dE;
	double val = Emin;
	for(int i =0; i<numBins; i++, val += dE){
		Egrid.push_back(val);
		std::cout << "Egrid = " << val << std::endl;
	}
	Egrid.push_back(Emax);
	out.assign(Egrid.size(),0);
	std::cout << "Egrid = " << Emax << std::endl;
}

void histogram::addEvent(double energy, double weight)
{	double index = floor((energy-Emin)/dE);
	//std::cout << "val before = " << out[index] << std::endl;
	out[index] +=weight;
	//std::cout << "val after = " << out[index] << std::endl;
}
