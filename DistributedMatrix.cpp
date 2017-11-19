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

inline int calculateIndex(const vector3<int>& iR, const vector3<int>& kfold)
{	int i = 0;
	for(int iDir=0; iDir<3; iDir++)
	{	if(iDir) i *= kfold[iDir-1];
		i += positiveRemainder(iR[iDir], kfold[iDir]);
	}
	return i;
}

DistributedMatrix::DistributedMatrix(string fname, bool realOnly, const MPIUtil* mpiUtil, int nElemsTot,
	const std::vector<vector3<int>>& cellMap, const vector3<int>& kfold, bool squared)
: mpiUtil(mpiUtil), nElemsTot(nElemsTot), cellMap(cellMap), kfold(kfold), squared(squared)
{
	nCellsTot = cellMap.size();
	kfoldProd = kfold[0]*kfold[1]*kfold[2];
	nkTot = kfoldProd;
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
	
	//Create FFTW plans:
	complex* bufData = buf.data();
	planSet = std::make_shared<PlanSet>();
	//--- transpose
	planSet->transpose = fftw_mpi_plan_many_transpose(nElemsTot, nkTot, 2,
		FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
		(double*)bufData, (double*)bufData,
		mpiUtil->communicator(), FFTW_MEASURE);
	if(!planSet->transpose) die_alone("MPI transpose plan creation failed.\n");
	//--- FFTs:
	if(squared)
	{	planSet->fft1 = fftw_plan_many_dft(3, &kfold[0], kfoldProd, //this has to be called nElems times
			(fftw_complex*)bufData, NULL, kfoldProd, 1, //Note: strided transform
			(fftw_complex*)bufData, NULL, kfoldProd, 1,
			-1, FFTW_MEASURE);
		if(!planSet->fft1) die_alone("Cell-map-squared FFT w.r.t cell 1 plan creation failed.\n");
		planSet->fft2 = fftw_plan_many_dft(3, &kfold[0], nElems*kfoldProd,
			(fftw_complex*)bufData, NULL, 1, kfoldProd,
			(fftw_complex*)bufData, NULL, 1, kfoldProd,
			+1, FFTW_MEASURE);
		if(!planSet->fft2) die_alone("Cell-map-squared FFT w.r.t cell 2 plan creation failed.\n");
	}
	else
	{	planSet->fft = fftw_plan_many_dft(3, &kfold[0], nElems,
			(fftw_complex*)bufData, NULL, 1, kfoldProd,
			(fftw_complex*)bufData, NULL, 1, kfoldProd,
			+1, FFTW_MEASURE);
		if(!planSet->fft) die_alone("Cell-map FFT plan creation failed.\n");
	}
	
	//Initialize cell (pair) index:
	cellIndex.resize(nCellsTot);
	auto iter = cellIndex.begin();
	if(squared)
	{	for(const vector3<int>& iR1: cellMap)
		{	int i1offset = kfoldProd * calculateIndex(iR1, kfold);
			for(const vector3<int>& iR2: cellMap)
				*(iter++) = i1offset + calculateIndex(iR2, kfold);
		}
	}
	else
	{	for(const vector3<int>& iR: cellMap)
			*(iter++) = calculateIndex(iR, kfold);
	}
	assert(iter == cellIndex.end());
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

void DistributedMatrix::transform(vector3<> k0)
{	static StopWatch watch("DistributedMatrix::transform1"); watch.start();
	assert(!squared);
	//Initialize offset phases:
	std::vector<complex> phase0(cellMap.size());
	auto phaseIter = phase0.begin();
	for(const vector3<int>& iR: cellMap)
		*(phaseIter++) = cis(2*M_PI*dot(iR, k0));
	//Reduce from mat to buf (apply offset phases and combine equivalent cells):
	buf.zero();
	complex* bufData = buf.data();
	const complex* matData = mat.data();
	for(int iElem=0; iElem<nElems; iElem++)
	{	auto cellIndexPtr = cellIndex.begin();
		for(const complex& phase0cur: phase0)
			bufData[iElem*nkTot+*(cellIndexPtr++)] += *(matData++) * phase0cur;
	}
	//Apply Fourier transform followed by MPI transpose:
	fftw_execute(planSet->fft);
	fftw_execute(planSet->transpose);
	watch.stop();
}

void DistributedMatrix::transform(vector3<> k01, vector3<> k02)
{	static StopWatch watch("DistributedMatrix::transform2"); watch.start();
	assert(squared);
	//Initialize offset phases:
	std::vector<complex> phase01(cellMap.size()), phase02(cellMap.size());
	auto phaseIter1 = phase01.begin();
	auto phaseIter2 = phase02.begin();
	for(const vector3<int>& iR: cellMap)
	{	*(phaseIter1++) = cis(-2*M_PI*dot(iR, k01));
		*(phaseIter2++) = cis(+2*M_PI*dot(iR, k02));
	}
	//Reduce from mat to buf (apply offset phases and combine equivalent cells):
	buf.zero();
	complex* bufData = buf.data();
	const complex* matData = mat.data();
	for(int iElem=0; iElem<nElems; iElem++)
	{	auto cellIndexPtr = cellIndex.begin();
		for(const complex& phase01cur: phase01)
		for(const complex& phase02cur: phase02)
			bufData[iElem*nkTot+*(cellIndexPtr++)] += *(matData++) * phase01cur * phase02cur;
	}
	//Apply Fourier transform followed by MPI transpose:
	for(int iElem=0; iElem<nElems; iElem++)
	{	fftw_execute_dft(planSet->fft1, (fftw_complex*)bufData, (fftw_complex*)bufData);
		bufData += nkTot;
	}
	fftw_execute(planSet->fft2);
	fftw_execute(planSet->transpose);
	watch.stop();
}

const complex* DistributedMatrix::getResult(int ik) const
{	int ikLocal = ik-ikStart;
	assert(ikLocal >= 0);
	assert(ikLocal < nk);
	return buf.data() + ikLocal*nElemsTot;
}
