#ifndef WANNIERMETROPOLIS_BANDSTRUCT_H
#define WANNIERMETROPOLIS_BANDSTRUCT_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <vector>
#include <math.h>

class BandStruct
{	std::vector< vector3<int> > cellMap, phCellMap;
	int nBands, nModes;
	matrix hWannier, phWannier, phWannierMatrix, pxWannier, pyWannier, pzWannier;
	matrix phase, phPhase, evecs, phEvecs; diagMatrix eigs, phEigs;
	vector3<> kCache, qCache; //value of kpoint for which evecs and phase was computed
public:
	BandStruct(string filePrefix, double mu, bool usePhononStates=false);
	diagMatrix getStates(vector3<> k);
	diagMatrix getPhononModes(vector3<> k);
	std::vector<matrix> getDipoleMatElem(vector3<> k);
	std::vector<matrix> getPhononMatElem(vector3<> k1, vector3<> k2);
	double get_mk(vector3<> k, double omega, double T); //calculate the energy conservation weight at a given k-point
	double get_mk1k2(vector3<> k1, vector3<> k2, double omega, double T); //calculate the energy conservation weight at a given k-point pair
	std::vector< vector3<> > getVelocity(vector3<> k, const matrix3<>& R); //calculate velocities (in Cartesian coordinates; converted using lattice vectors R)
	
	inline static double mk_sub(double Ec, double Ev, double omega, double T)
	{	return std::pow((Ec - Ev - omega),2) - 2*T*T * (logFermi(Ev/T) + logFermi(-Ec/T));
	}
	
	inline static double logFermi(double x) //calculate log(1/(1+exp(x)))
	{	return (x>30.) ? -x : -log(1+exp(x)); //avoid overflow issues
	}
};
#endif //WANNIERMETROPOLIS_BANDSTRUCT_H
