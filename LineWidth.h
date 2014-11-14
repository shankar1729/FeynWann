#ifndef WANNIERMETROPOLIS_LINEWIDTH_H
#define WANNIERMETROPOLIS_LINEWIDTH_H

#include <core/string.h>
#include <vector>

class LineWidth
{
	double Emin, dE;
	std::vector<double> imSigma;
public:
	LineWidth(string inputFilename);
	double operator()(double energy) const;
};

#endif //WANNIERMETROPOLIS_LINEWIDTH_H
