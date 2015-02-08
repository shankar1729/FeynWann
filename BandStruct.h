#ifndef WANNIERMETROPOLIS_BANDSTRUCT_H
#define WANNIERMETROPOLIS_BANDSTRUCT_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <vector>
#include <memory>
#include <math.h>
#include <float.h>

class BandStruct
{	
public:
	BandStruct(
		string filePrefix, //!< filename prefix for Wannier band structure and matrix element files
		double mu, //!< Fermi level; all energies will subsequently be referenced against it
		int spinWeight, //!< 2 => non-relativistic, 1 => relativistic
		string phononPrefix=string(), //!< filename prefix for phonon dispersion files
		std::vector< vector3<complex> > Ahat = std::vector< vector3<complex> >() //!< list of relevant phonon polarizations (must specify if dipole matrix elements needed)
	);
	
	diagMatrix getStates(vector3<> k, double omegaMax=DBL_MAX, matrix* evecs=0) const; //if omegaMax is less than omegaMain (if available), use main centers alone. Optionally retrieve eigenvectors
	std::vector<diagMatrix> getStates(const std::vector< vector3<> >& kArr, double omegaMax=DBL_MAX, matrix* evecs=0) const; //array version of above
	diagMatrix getPhononModes(vector3<> q) const;
	std::vector<matrix> getDipoleMatElem(vector3<> k) const; //dipole matrix elements contracted against each specified Ahat in constructor
	matrix getDipoleSqMatElem(vector3<> k) const; //matrix elements of (Ahat1.P)(Ahat2.P); must have exactly two Ahat's in constructor

	std::vector<matrix> getPhononMatElem(vector3<> k1, vector3<> k2) const;
	void setPhononMatElemArray(vector3<> k1, const std::vector< vector3<> >& k2arr, std::vector<matrix>* result) const; //get matrix elements for fixed k1 and ana array of k2 (for efficiency)
	
	double get_mk(vector3<> k, double omega, double T) const; //calculate the energy conservation weight at a given k-point
	double get_mk1k2(vector3<> k1, vector3<> k2, double omega, double T) const; //calculate the energy conservation weight at a given k-point pair
	std::vector< vector3<> > getVelocity(vector3<> k, const matrix3<>& R, double omegaMax=DBL_MAX) const; //calculate velocities (in Cartesian coordinates; converted using lattice vectors R)
	
	inline static double mk_sub(double Ec, double Ev, double omega, double T)
	{	return std::pow((Ec - Ev - omega),2) - 2*T*T * (logFermi(Ev/T) + logFermi(-Ec/T));
	}
	
	inline static double logFermi(double x) //calculate log(1/(1+exp(x)))
	{	return (x>30.) ? -x : -log(1+exp(x)); //avoid overflow issues
	}
	
	std::vector<double> Eceda; //common energy denominator value for each number of Wannier bands used
	
	void setCacheSize(int cacheSize) { this->cacheSize = std::max(6, cacheSize); } //control cache size for electron and phonon states
private:
	//Electrons:
	std::vector< vector3<int> > cellMap; //electron Wannier cell map
	int nBands; //number of Wannier bands for the electrons
	int nMain, mainFirst; double omegaMain; //number of "main" Wannier centers, index of first main center and max frequency for which main window suffices
	matrix hWannier, pWannier, pSqWannier; //Wannier hamiltonian and dipole matrix elements
	matrix hWannierMain; //Wannier hamiltonian for the main centers alone
	
	int nPol; //number of photon polarizations in pre-contracted matrix elements
	int nPacked; //packed size of matrix elements (nBands x nBands when no main matrix elements)
	void compressMatElemArr(matrix& mArr) const; //replace a matrix element array by its packed version
	void transformMatElemArr(matrix& mArr, const matrix& rot) const; //replace a packed matrix element array by its transformed version with given transformation matrix rot
	void packMatElem(const matrix& m, matrix& mArr, int iCol) const; //pack matrix elements from m, getting rid of non-main by non-main entries, and store in i'th column of mArr
	matrix unpackMatElem(const matrix& mArr, int iCol) const; //unpack matrix elements from i'th column of mArr
	
	//Phonons:
	std::vector< vector3<int> > phononCellMap; //cell map for phonon force matrix
	int nModes; //number of phonon modes (polarizations)
	matrix omegaSqPh; //phonon force matrix
	
	//Electron-phonon interaction:
	matrix wannierHePh; //electron-phonon matrix elements in Wannier basis
	struct CellPair { vector3<int> iR1, iR2; };
	std::vector<CellPair> phononCellMapSq; //pairs of cells for which electron-phonon matrix elements are stored
	
	//Caching of unitary rotations and eigenvalues (electrons as well as  phonons):
	struct CacheEntry
	{	size_t rank; //Rank for the cache entry used to find the oldest entries to delete
		matrix phase; //Fourier transform phase
		matrix evecs; //discrete unitary rotation
		diagMatrix eigs; //eigenvalues
		int nBands() const { return eigs.nRows(); }
	};
	std::map<vector3<>, std::shared_ptr<const CacheEntry> > electronCache, mainCache, phononCache;
	size_t rankElectron, rankMain, rankPhonon;
	//Note that the cache functions may update the cache, but are functionally const (henced marked as such). However they are NOT thread safe.
	std::shared_ptr<const CacheEntry> getElectronCache(vector3<> k, double omegaMax=DBL_MAX) const;
	std::shared_ptr<const CacheEntry> getPhononCache(vector3<> q) const;
	std::vector< std::shared_ptr<const CacheEntry> > getElectronCache(const std::vector< vector3<> >& kArr, double omegaMax=DBL_MAX) const;
	std::vector< std::shared_ptr<const CacheEntry> > getPhononCache(const std::vector< vector3<> >& qArr) const;
	//--- Most of the implementation of caching mechanism collected here, and called by above:
	std::vector< std::shared_ptr<const CacheEntry> > getCache(
		const std::vector< vector3<> >& kArr, //!< array of k/q-points
		const std::map<vector3<>, std::shared_ptr<const CacheEntry> >& cache, //!< relevant cache
		const std::vector< vector3<int> >& cellMap, //!< relevant cell map
		const matrix& hWannierEff, //!< relevant Hamiltonian
		const size_t& rank, //!< relevant rank counter
		bool shouldSqrt //!< whether to take square-root of eigenvalues (needed for phonons)
	) const;
	size_t cacheSize;
};

#endif //WANNIERMETROPOLIS_BANDSTRUCT_H
