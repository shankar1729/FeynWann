#ifndef WANNIERMETROPOLIS_EPSILON_H
#define WANNIERMETROPOLIS_EPSILON_H

#include <core/string.h>
#include <core/vector3.h>
#include <vector>

class Epsilon
{	std::vector<vector3<double>> epsParams;
public:
	double omega_p, omega, Eplasmon, modGammaPlus, modGammaMinus, Lquant, k;
	Epsilon(string inputFilename);
	void setFrequency(double omegaIn);
};
#endif //WANNIERMETROPOLIS_EPSILON_H
