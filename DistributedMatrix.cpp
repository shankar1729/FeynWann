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

struct PlanSet
{	fftw_plan transpose; //plan for MPI transpose
	fftw_plan fft; //Fourier transform plan for non-squared case
	fftw_plan fft1; //Fourier transform plan w.r.t iR1 for squared case
	fftw_plan fft2; //Fourier transform plan w.r.t iR2 for squared case
};

DistributedMatrix::DistributedMatrix(string fname, bool realOnly, const MPIUtil* mpiUtil, int nElemsTot,
	const std::vector<vector3<int>>& cellMap, const vector3<int>& kfold, bool squared)
: mpiUtil(mpiUtil), nElemsTot(nElemsTot), cellMap(cellMap), kfold(kfold), squared(squared)
{
	nCellsTot = cellMap.size();
	nkTot = kfold[0]*kfold[1]*kfold[2];
	int kfoldProd = nkTot;
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
	
	//Create FFTW plans:
	planSet = std::make_shared<PlanSet>();
	//--- transpose
	planSet->transpose = fftw_mpi_plan_many_transpose(nElemsTot, nkTot, 2,
		FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
		(double*)buf.data(), (double*)buf.data(),
		mpiUtil->communicator(), FFTW_MEASURE);
	if(!planSet->transpose) die_alone("MPI transpose plan creation failed.\n");
	//--- FFTs:
	if(squared)
	{	planSet->fft1 = fftw_plan_many_dft(3, &kfold[0], nElems*kfoldProd,
			(fftw_complex*)buf.data(), NULL, kfoldProd, 1, //Note: strided transform
			(fftw_complex*)buf.data(), NULL, kfoldProd, 1,
			-1, FFTW_MEASURE);
		if(!planSet->fft1) die_alone("Cell-map-squared FFT w.r.t cell 1 plan creation failed.\n");
		planSet->fft2 = fftw_plan_many_dft(3, &kfold[0], nElems*kfoldProd,
			(fftw_complex*)buf.data(), NULL, 1, kfoldProd,
			(fftw_complex*)buf.data(), NULL, 1, kfoldProd,
			+1, FFTW_MEASURE);
		if(!planSet->fft2) die_alone("Cell-map-squared FFT w.r.t cell 2 plan creation failed.\n");
	}
	else
	{	planSet->fft = fftw_plan_many_dft(3, &kfold[0], nElems,
			(fftw_complex*)buf.data(), NULL, 1, nkTot,
			(fftw_complex*)buf.data(), NULL, 1, nkTot,
			+1, FFTW_MEASURE);
		if(!planSet->fft) die_alone("Cell-map FFT plan creation failed.\n");
	}
}

DistributedMatrix::~DistributedMatrix()
{	//Clean-up FFTW plans:
	fftw_destroy_plan(planSet->transpose);
	if(squared)
	{	fftw_destroy_plan(planSet->fft1);
		fftw_destroy_plan(planSet->fft2);
	}
	else fftw_destroy_plan(planSet->fft);
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
