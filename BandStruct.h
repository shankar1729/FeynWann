#ifndef WANNIERMETROPOLIS_BANDSTRUCT_H
#define WANNIERMETROPOLIS_BANDSTRUCT_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <vector>
#include <math.h>

class BandStruct
{	std::vector< vector3<int> > cellMap;
	int nBands;
	matrix hWannier, pxWannier, pyWannier, pzWannier;
	matrix phase, evecs; diagMatrix eigs;
	vector3<> kPointCache; //value of kpoint for which evecs and phase was computed
public:
	BandStruct(string filePrefix, double mu);
	diagMatrix getStates(vector3<> kPoint);
	std::vector<matrix> getTransitions(vector3<> kPoint);
	double get_mk(vector3<> kPoint, double omega, double T); //calculate the energy conservation weight at a given k-point
	double get_mk1k2(vector3<> kPoint1, vector3<> kPoint2, double omega, double T); //calculate the energy conservation weight at a given k-point pair
	std::vector< vector3<> > getVelocity(vector3<> kPoint); //calculate velocities (in lattice coordinates)
	
	inline static double mk_sub(double Ec, double Ev, double omega, double T)
	{	return std::pow((Ec - Ev - omega),2) - 2*T*T * (logFermi(Ev/T) + logFermi(-Ec/T));
	}
	
	inline static double logFermi(double x) //calculate log(1/(1+exp(x)))
	{	return (x>30.) ? -x : -log(1+exp(x)); //avoid overflow issues
	}
};
#endif //WANNIERMETROPOLIS_BANDSTRUCT_H
