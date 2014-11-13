#ifndef LIFETIME_H
#define LIFETIME_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
// ------------------------------------ class lifeTime -- ---------------------------------------------------

class lifeTime
{
public:
	std::vector<double> E, imSigma;
	lifeTime(string inputFilename);
	double get_lifeTime( double energy);
};
#endif
