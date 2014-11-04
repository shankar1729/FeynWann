#ifndef BANDSTRUCT_H
#define BANDSTRUCT_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <vector>
#include <math.h>

//---------------------------- class bandStruct---------------------------------

class bandStruct
{
public:
	string filePrefix;
	std::vector< vector3<int> > cellMap;
	int nBands;
	matrix hWannier, pxWannier, pyWannier, pzWannier;
	matrix phase, evecs;
	vector3<> kPoint_evecs; //value of kpoint for which evecs and phase was computed
	bandStruct(string filePrefix);
	diagMatrix getStates(vector3<double> kPoint);
	std::vector<matrix> getTransitions(vector3<double> kPoint);
};
#endif
