#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <vector>
#include <math.h>
#include <algorithm>

//---------------------------- class bandStruct---------------------------------

class histogram
{
public:
        double Emin, dE, Emax;
	std::vector<double> Egrid, out;
        histogram(double emin, double de, double emax);
        void addEvent(double energy, double weight);
};
#endif
