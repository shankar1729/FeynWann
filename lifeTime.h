#ifndef LIFETIME_H
#define LIFETIME_H

#include <core/string.h>
#include <vector>

class lifeTime
{
	double Emin, dE;
	std::vector<double> imSigma;
public:
	lifeTime(string inputFilename);
	double operator()(double energy) const;
};

#endif
