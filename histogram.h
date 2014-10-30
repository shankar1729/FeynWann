#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>

//---------------------------- class bandStruct---------------------------------

class histogram
{
public:
        double Emin, dE, Emax;
	std::vector<double> Egrid, out;
        histogram(double Emin, double dE, double Emax);
        void addEvent(double energy, double weight);
};
#endif
