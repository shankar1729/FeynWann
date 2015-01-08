#ifndef WANNIERMETROPOLIS_BANDSTRUCT_H
#define WANNIERMETROPOLIS_BANDSTRUCT_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <vector>
#include <list>
#include <memory>
#include <math.h>

class BandStruct
{	
public:
	BandStruct(string filePrefix, double mu, int spinWeight, string phononPrefix=string());
	diagMatrix getStates(vector3<> k) const;
	diagMatrix getPhononModes(vector3<> q) const;
	std::vector<matrix> getDipoleMatElem(vector3<> k) const;

	std::vector<matrix> getPhononMatElem(vector3<> k1, vector3<> k2) const;
	void setPhononMatElemArray(vector3<> k1, const std::vector< vector3<> >& k2arr, std::vector<matrix>* result) const; //get matrix elements for fixed k1 and ana array of k2 (for efficiency)
	
	double get_mk(vector3<> k, double omega, double T) const; //calculate the energy conservation weight at a given k-point
	double get_mk1k2(vector3<> k1, vector3<> k2, double omega, double T) const; //calculate the energy conservation weight at a given k-point pair
	std::vector< vector3<> > getVelocity(vector3<> k, const matrix3<>& R) const; //calculate velocities (in Cartesian coordinates; converted using lattice vectors R)
	
	inline static double mk_sub(double Ec, double Ev, double omega, double T)
	{	return std::pow((Ec - Ev - omega),2) - 2*T*T * (logFermi(Ev/T) + logFermi(-Ec/T));
	}
	
	inline static double logFermi(double x) //calculate log(1/(1+exp(x)))
	{	return (x>30.) ? -x : -log(1+exp(x)); //avoid overflow issues
	}
	
private:
	//Electrons:
	std::vector< vector3<int> > cellMap; //electron Wannier cell map
	int nBands; //number of Wannier bands for the electrons
	matrix hWannier, pWannier[3]; //Wannier hamiltonian and dipole matrix elements
	
	//Phonons:
	std::vector< vector3<int> > phononCellMap; //cell map for phonon force matrix
	int nModes; //number of phonon modes (polarizations)
	matrix omegaSqPh; //phonon force matrix
	
	//Electron-phonon interaction:
	matrix wannierHePh; //electron-phonon matrix elements in Wannier basis
	struct CellPair { vector3<int> iR1, iR2; };
	std::vector<CellPair> phononCellMapSq; //pairs of cells for which electron-phonon matrixe lemenst are stored
	
	//Caching of unitary rotations and eigenvalues (electrons as well as  phonons):
	struct CacheEntry
	{	vector3<> k; //wave-vector for cached entry
		matrix phase; //Fourier transform phase
		matrix evecs; //discrete unitary rotation
		diagMatrix eigs; //eigenvalues
	};
	std::list< std::shared_ptr<const CacheEntry> > electronCache, phononCache;
	//Note that the cache functions may update the cache, but are functionally const (henced marked as such). However they are NOT thread safe.
	std::shared_ptr<const CacheEntry> getElectronCache(vector3<> k) const;
	std::shared_ptr<const CacheEntry> getPhononCache(vector3<> q) const;
};

#endif //WANNIERMETROPOLIS_BANDSTRUCT_H
