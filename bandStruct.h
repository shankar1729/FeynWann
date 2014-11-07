#ifndef BANDSTRUCT_H
#define BANDSTRUCT_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <vector>
#include <math.h>

//---------------------------- class bandStruct---------------------------------

class bandStruct
{	std::vector< vector3<int> > cellMap;
	int nBands;
	matrix hWannier, pxWannier, pyWannier, pzWannier;
	matrix phase, evecs; diagMatrix eigs;
	vector3<> kPointCache; //value of kpoint for which evecs and phase was computed
public:
	bandStruct(string filePrefix, double mu);
	diagMatrix getStates(vector3<double> kPoint);
	std::vector<matrix> getTransitions(vector3<double> kPoint);
	double get_mk(vector3<double> kPoint, double omega, double T); //calculate the energy conservation weight at a given k-point
};
#endif
