#ifndef WANNIERMC_WANNIERMC_H
#define WANNIERMC_WANNIERMC_H

#include "DistributedMatrix.h"

//! Parameters for initializing Wannier
struct WannierMCParams
{	string totalEprefix; //!< filename prefix for DFT outputs (default: Wannier/totalE)
	string phononPrefix; //!< filename prefix for phonon outputs (default: Wannier/phonon)
	string wannierPrefix; //!< filename prefix for wannier outputs (default: Wannier/wannier)
	bool needPhonons; //!< whether to initialize phonon-related quantities (default: false)
	bool needLinewidths; //!< whether to initialize line-widths (default: false)
	std::vector< vector3<complex> > Ahat; //!< list of relevant photon polarizations (default: none)
	
	WannierMCParams();
};


class WannierMC
{
public:
	static InitParams initialize(int argc, char** argv, const char* description); //wrap initSystemCmdLine from JDFTx
	static void finalize(); //wrap finalizeSystem from JDFTx
	
	const WannierMCParams& wmcp;
	WannierMC(const WannierMCParams& wmcp);
	
	//DFT / Wannier / Phonon parameters:
	matrix3<> R; //!< lattice vectors
	vector3<int> kfold; //!< k-point folding in original calculation
	vector3<bool> isTruncated; //!< whether each direction is truncated
	int nBands, spinWeight; //!< number of Wannier bands for the electrons and weight per spin channel
	double mu, nElectrons, nValence; //!< chemical potential (if a metal), number of electrons per unit cell and number of valence bands (if insulator)
	double eMinMain, eMaxMain; //!< energy range for main window (within which eigenvalues should be exact compared to DFT)
	int nPol; //!< number of photon polarizations in pre-contracted matrix elements
	int nModes; //!< number of phonon modes (polarizations)
	
private:
	//Electrons:
	std::vector< vector3<int> > cellMap; //electron Wannier cell map
	int nMain, mainFirst; double omegaMain; //number of "main" Wannier centers, index of first main center and max frequency for which main window suffices
	matrix hWannier, pWannier; //Wannier hamiltonian and dipole matrix elements
	matrix hWannierMain; //Wannier hamiltonian for the main centers alone
	
	int nPacked; //packed size of matrix elements (nBands x nBands when no main matrix elements)
	void compressMatElemArr(matrix& mArr) const; //replace a matrix element array by its packed version
	void transformMatElemArr(matrix& mArr, const matrix& rot) const; //replace a packed matrix element array by its transformed version with given transformation matrix rot
	void packMatElem(const matrix& m, matrix& mArr, int iCol) const; //pack matrix elements from m, getting rid of non-main by non-main entries, and store in i'th column of mArr
	matrix unpackMatElem(const matrix& mArr, int iCol) const; //unpack matrix elements from i'th column of mArr
	
	//Phonons:
	std::vector< vector3<int> > phononCellMap; //cell map for phonon force matrix
	matrix omegaSqPh; //phonon force matrix
	
	//Electron-phonon interaction:
	matrix wannierHePh; //electron-phonon matrix elements in Wannier basis
	struct CellPair { vector3<int> iR1, iR2; };
	std::vector<CellPair> phononCellMapSq; //pairs of cells for which electron-phonon matrix elements are stored
};

#endif //WANNIERMC_WANNIERMC_H
