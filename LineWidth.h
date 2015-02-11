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
private:
	const BandStruct& bs;
	matrix eeWannier, ePhWannier; //Wannierized e-e and e-Ph contributions to ImSigma
};

#endif //WANNIERMETROPOLIS_LINEWIDTH_H
