#include "DistributedMatrix.h"
#include <fftw3-mpi.h>

template<typename scalar> void readMatrix(const MPIUtil* mpiUtil, string fname,
	int nElemsTot, int nElems, int iElemStart, int nCellsTot, complex* dest)
{
	logPrintf("Reading '%s' ... ", fname.c_str()); fflush(globalLog);
	size_t fsizeExpected = nCellsTot * nElemsTot * sizeof(scalar);
	MPIUtil::File fp;
	mpiUtil->fopenRead(fp, fname.c_str(), fsizeExpected);
	std::vector<scalar> column(nElems); //temporary storage to hold local portion of column
	for(int iCell=0; iCell<nCellsTot; iCell++)
	{	mpiUtil->fseek(fp, (iCell*nElemsTot+iElemStart)*sizeof(scalar), SEEK_SET);
		mpiUtil->fread(column.data(), sizeof(scalar), nElems, fp);
		for(int iElem=0; iElem<nElems; iElem++)
			dest[iCell+nCellsTot*iElem] = column[iElem]; //store transposed
	}
	mpiUtil->fclose(fp);
	logPrintf("done.\n");
}


DistributedMatrix::DistributedMatrix(string fname, bool realOnly, const MPIUtil* mpiUtil, int nElemsTot,
	const std::vector<vector3<int>>& cellMap, const vector3<int>& kfold, bool squared)
: mpiUtil(mpiUtil), nElemsTot(nElemsTot), cellMap(cellMap), kfold(kfold), squared(squared)
{
	nCellsTot = cellMap.size();
	nkTot = kfold[0]*kfold[1]*kfold[2];
	if(squared)
	{	nCellsTot *= nCellsTot;
		nkTot *= nkTot;
	}
	
	//Determine division:
	ptrdiff_t local_n0, local_0start, local_n1, local_1start;
	ptrdiff_t nTot = fftw_mpi_local_size_2d_transposed(nElemsTot, nkTot,
		mpiUtil->communicator(), &local_n0, &local_0start, &local_n1, &local_1start);
	nElems = local_n0; iElemStart = local_0start;
	nk = local_n1; ikStart = local_1start;
	
	//Allocate matrix and buffer:
	mat.init(nElems*nCellsTot);
	buf.init(nTot); //nTot = max(nElems*nkTot, nElemsTot*nk)
	
	//Read matrix:
	if(realOnly) readMatrix<double>(mpiUtil, fname, nElemsTot, nElems, iElemStart, nCellsTot, mat.data());
	else readMatrix<complex>(mpiUtil, fname, nElemsTot, nElems, iElemStart, nCellsTot, mat.data());
	
// 	//Debug matrix elements
// 	ostringstream oss; oss << "debug." << mpiWorld->iProcess();
// 	FILE* fp = fopen(oss.str().c_str(), "w");
// 	for(int iCell=0; iCell<nCellsTot; iCell++)
// 	{	for(int iElem=0; iElem<nElems; iElem++)
// 			fprintf(fp, "%19.12le ", mat.data()[iCell+nCellsTot*iElem].real());
// 		fprintf(fp, "\n");
// 	}
// 	fclose(fp);
}

DistributedMatrix::~DistributedMatrix()
{	
}

void DistributedMatrix::transform(vector3<int> k0)
{	assert(!squared);
}

void DistributedMatrix::transform(vector3<int> k01, vector3<int> k02)
{	assert(squared);
}

const complex* DistributedMatrix::getResult(int ik) const
{	int ikLocal = ik-ikStart;
	assert(ikLocal >= 0);
	assert(ikLocal < nk);
	return buf.data() + ikLocal*nElemsTot;
}
