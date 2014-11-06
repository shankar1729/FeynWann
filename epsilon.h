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
	double omega_p, omega, Eplasmon, modGammaPlus, modGammaMinus, Lquant, k;
	epsilon(string inputFilename);
	void setFrequency( double omegaIn);
};
#endif
