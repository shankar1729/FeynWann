#ifndef BANDSTRUCT_H
#define BANDSTRUCT_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

//---------------------------- class bandStruct---------------------------------

class bandStruct
{
public:
	string filePrefix;
	std::vector< vector3<int> > cellMap;
	std::vector<matrix> hWannierArray, pxWannierArray, pyWannierArray, pzWannierArray;
	matrix evecs;
	vector3<> kPoint_evecs; //value of kpoint for which evecs was computed
	bandStruct(string filePrefix);
	diagMatrix getStates(vector3<double> kPoint);
	std::vector<matrix> getTransitions(vector3<double> kPoint);
};
#endif
