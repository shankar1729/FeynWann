#ifndef EPSILON_H
#define EPSILON_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
// ------------------------------------ class epsilon ---------------------------------------------------

class epsilon
{
public:
	std::vector<vector3<double>> epsParams;
	double omega_p, Eplasmon, modGammaPlus, modGammaMinus, k;
	epsilon(string inputFilename, double Eplamson);
	double getLquant();
	double getModGammaMinus() { return modGammaMinus; }
	double getK() {return k; }
};
#endif
