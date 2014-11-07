#include "histogram.h"
#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>


//-------------------------- class histogram ------------------------------------------------

// Constructor
histogram::histogram(double emin, double de, double emax)
{	// Make uniform energy gric
	dE = de; Emin = emin; Emax = emax;
	int numBins = (Emax - Emin)/dE;
	double val = Emin;
	for(int i =0; i<numBins; i++, val += dE){
		Egrid.push_back(val);
	}
	Egrid.push_back(Emax);
	out.assign(Egrid.size(),0);
}

void histogram::addEvent(double energy, double weight)
{	double Energy = energy;
	int index = floor((Energy-Emin)/dE);
	//std::cout << "val before = " << out[index] << std::endl;
	out[index] +=weight;
	//std::cout << "val after = " << out[index] << std::endl;
}
