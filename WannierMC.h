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
	bool needVelocity; //!< whether to initialize velocity (momentum) matrix elements
	WannierMCParams();
};

//! Wannier interpolator for electrons and phonons
class WannierMC
{
public:
	static InitParams initialize(int argc, char** argv, const char* description); //!< wrap initSystemCmdLine from JDFTx
	static void finalize(); //!< wrap finalizeSystem from JDFTx
	static vector3<> randomVector(MPIUtil* mpiUtil=0); //!< uniformly random vector in [0,1)^3, constant across mpi instance, if any
	
	const WannierMCParams& wmcp;
	WannierMC(const WannierMCParams& wmcp);
	void free(); //!< free matrices
	
	//! Electronic properties at a given wave vector
	struct StateE
	{	vector3<> k; //!< wave-vector in recip lattice coords
		diagMatrix E; //!< energy relative to Fermi level (WannierMC::mu)
		matrix v[3]; //!< velocity matrix elements in Cartesian coordinates, available if needVelocity = true
		diagMatrix ImE; //!< linewidth, available if needLinewidths = true
	};
	
	//! Phonon properties at a given wave vector
	struct StatePh
	{	vector3<> q; //!< wave-vector in recip lattice coords
		diagMatrix omega; //!< frequency
	};
	
	//! Electron-phonon matrix elements
	struct MatrixEPh
	{	const StateE* e1; //!< corresponding first electronic state
		const StateE* e2; //!< corresponding second electronic state
		const StatePh* ph; //!< corresponding phonon state
		std::vector<matrix> M; //!< nModes matrices of nBands x nBands matrix elements
	};
	
	typedef void (*eProcessFunc)(const StateE& state, void* params); //!< Callback function pointer for eLoop()
	typedef void (*phProcessFunc)(const StatePh& state, void* params); //!< Callback function pointer for phLoop()
	
	//! Calculate electronic properties for each k-point in a mesh offset by k0
	//! Calls provided callback function eProcess on each of them, along with provided params
	void eLoop(const vector3<>& k0, eProcessFunc eProcess, void* params);
	size_t eCountPerOffset() const { return Hw->nkTot; } //!< number of k's sampled per offset
	
	//! Calculate phonon properties for each q-point in a mesh offset by q0
	//! Calls provided callback function phProcess on each of them, along with provided params
	void phLoop(const vector3<>& q0, phProcessFunc phProcess, void* params);
	size_t phCountPerOffset() const { return OsqW->nkTot; } //!< number of q's sampled per offset
	
	//DFT / Wannier / Phonon parameters:
	matrix3<> R; //!< lattice vectors
	double Omega; //!< unit cell volume
	vector3<int> kfold; //!< k-point folding in original calculation
	vector3<int> phononSup; //!< phonon supercell in original calculation
	vector3<bool> isTruncated; //!< whether each direction is truncated
	int nBands, spinWeight; //!< number of Wannier bands for the electrons and weight per spin channel
	double mu, nElectrons, nValence; //!< chemical potential (if a metal), number of electrons per unit cell and number of valence bands (if insulator)
	double eMinMain, eMaxMain; //!< energy range for main window (within which eigenvalues should be exact compared to DFT)
	int nModes; //!< number of phonon modes (polarizations)
	
private:
	//Electrons:
	std::vector< vector3<int> > cellMap; //electron Wannier cell map
	std::shared_ptr<DistributedMatrix> Hw, Pw; //Wannier hamiltonian and dipole matrix elements
	std::shared_ptr<DistributedMatrix> ImSigma_eeW, ImSigma_ePhW; //linewidths in wannier basis
	void setState(int ik, StateE& state, matrix* Vptr=0); //!< set requested properties for ik in state, optionally retrieving eigenvectors in Vptr
	
	//Phonons:
	std::vector< vector3<int> > phononCellMap; //cell map for phonon force matrix
	std::shared_ptr<DistributedMatrix> OsqW; //phonon omega-squared matrix
	void setState(int iq, StatePh& state, matrix* Vptr=0); //!< set requested properties for iq in state, optionally retrieving eigenvectors in Vptr
	
	//Electron-phonon interaction:
	std::shared_ptr<DistributedMatrix> HePhW; //electron-phonon matrix elements in Wannier basis
};

#endif //WANNIERMC_WANNIERMC_H
