#ifndef WANNIERMC_DISTRIBUTEDMATRIX_H
#define WANNIERMC_DISTRIBUTEDMATRIX_H

#include <core/matrix.h>

//! Distributed matrix elements / Hamiltonians for WannierMC
class DistributedMatrix
{
public:
	//Input parameters:
	const MPIUtil* mpiUtil; //!< MPI instance / communicator over whcih this is parallelized
	int nElemsTot; //!< total number of matrix elements (per cell/k) [if packed, only counts packed]
	const std::vector<vector3<int>>& cellMap; //!< cell map in real space
	const vector3<int>& kfold; //!< k-point folding
	bool squared; //!< whether cell map / kpoints are squared (eg. e-ph matrix elements)
	
	//Parameters set in the constructor:
	int nElems; //!< local number of matrix elements (per cell/k) [if packed, only counts packed]
	int iElemStart; //!< starting element on current process
	int nCellsTot; //!< size of cell map or its square, depending on squared
	int kfoldProd; //!< prod(kfold)
	int nkTot; //!< prod(kfold) or its square, depending on squared
	int nk; //!< number of k-points (or pairs, if squared) on current process
	int ikStart; //!< starting k-point (or pair, if squared) on current process
	std::vector<int> ikStartProc; //!< ikStart for each process in MPI group
	
	//! Initialize from file, containing complex or real elements as specified by realOnly
	//! (remaining parameters are as specified in the class)
	DistributedMatrix(string fname, bool realOnly, const MPIUtil* mpiUtil, int nElemsTot,
		const std::vector<vector3<int>>& cellMap, const vector3<int>& kfold, bool squared);
	~DistributedMatrix();
	
	void transform(vector3<> k0); //!< prepare results for k-point mesh offset by k0 (squared=false only)
	void transform(vector3<> k01, vector3<> k02); //!< prepare results for k-point mesh offsets k01 and k02 (squared=true only)
	const complex* getResult(int ik) const; //!< get pointer to result for k-point (or pair) index ik
private:
	ManagedArray<complex> mat; //!< input matrix elements
	ManagedArray<complex> buf; //!< buffer in which transformations happen and result is produced
	std::shared_ptr<struct PlanSet> planSet; //!< opaque pointer to required set of FFT plans
	std::vector<int> cellIndex; //!< index of cell or cell-pair in nkTot array
};

//Calculate flat index given 3D coordinates and sample counts
inline int calculateIndex(const vector3<int>& iv, const vector3<int>& S)
{	int i = 0;
	for(int iDir=0; iDir<3; iDir++)
	{	if(iDir) i *= S[iDir-1];
		i += positiveRemainder(iv[iDir], S[iDir]);
	}
	return i;
}

#endif //WANNIERMC_DISTRIBUTEDMATRIX_H
