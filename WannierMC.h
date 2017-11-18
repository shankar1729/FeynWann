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


class WannierMC
{
public:
	static InitParams initialize(int argc, char** argv, const char* description); //wrap initSystemCmdLine from JDFTx
	static void finalize(); //wrap finalizeSystem from JDFTx
	
	const WannierMCParams& wmcp;
	WannierMC(const WannierMCParams& wmcp);
	void free(); //!< free matrices
	
	//DFT / Wannier / Phonon parameters:
	matrix3<> R; //!< lattice vectors
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
	
	//Phonons:
	std::vector< vector3<int> > phononCellMap; //cell map for phonon force matrix
	std::shared_ptr<DistributedMatrix> OsqW; //phonon omega-squared matrix
	
	//Electron-phonon interaction:
	std::shared_ptr<DistributedMatrix> HePhW; //electron-phonon matrix elements in Wannier basis
};

#endif //WANNIERMC_WANNIERMC_H
