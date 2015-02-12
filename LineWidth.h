#ifndef WANNIERMETROPOLIS_LINEWIDTH_H
#define WANNIERMETROPOLIS_LINEWIDTH_H

#include <electronic/matrix.h>
#include <core/string.h>
#include <vector>
#include "BandStruct.h"

class LineWidth
{
public:
	LineWidth(string prefix, const BandStruct& bs);
	diagMatrix operator()(vector3<> k) const; //returns total ImSigma (e-e + e-ph)
	std::vector<diagMatrix> operator()(const std::vector< vector3<> >& k) const; //array version
private:
	int nBandsSq, nCells;
	const BandStruct& bs;
	matrix ImSigmaWannier; //Wannierized e-e and e-Ph contributions to ImSigma (combined columneiwse for efficient multiply)
};

#endif //WANNIERMETROPOLIS_LINEWIDTH_H
