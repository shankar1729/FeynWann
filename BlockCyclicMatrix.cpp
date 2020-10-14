#ifdef SCALAPACK_ENABLED
#include "BlockCyclicMatrix.h"
#include <core/Util.h>
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#include <mkl_lapack.h>
#include <core/Random.h>

inline complex sqrt(const complex& c)
{	double cAbs = c.abs();
	double x = sqrt(0.5*(cAbs+c.real())); //always positive => -infty branch cut on c
	double y = copysign(sqrt(0.5*(cAbs-c.real())), c.imag());
	return complex(x, y);
}

//Return list of indices in a given dimension (row or column) that belong to me in block-cyclic distribution
std::vector<int> distributedIndices(int nTotal, int blockSize, int iProcDim, int nProcsDim)
{	int zero = 0;
	int nMine = numroc_(&nTotal, &blockSize, &iProcDim, &zero, &nProcsDim);
	std::vector<int> myIndices; myIndices.reserve(nMine);
	int blockStride = blockSize * nProcsDim;
	int nBlocksMineMax = (nTotal + blockStride - 1) / blockStride;
	for(int iBlock=0; iBlock<nBlocksMineMax; iBlock++)
	{	int iStart = iProcDim*blockSize + iBlock*blockStride;
		int iStop = std::min(iStart+blockSize, nTotal);
		for(int i=iStart; i<iStop; i++)
			myIndices.push_back(i);
	}
	assert(int(myIndices.size()) == nMine);
	return myIndices;
}

BlockCyclicMatrix::BlockCyclicMatrix(int N, int blockSize, MPIUtil* mpiUtil) : N(N), blockSize(blockSize), mpiUtil(mpiUtil)
{
	//Calculate squarest possible process grid:
	int nProcesses = mpiUtil->nProcesses();
	nProcsRow = int(round(sqrt(nProcesses)));
	while(nProcesses % nProcsRow) nProcsRow--;
	nProcsCol = nProcesses / nProcsRow;

	//Initialize BLACS process grid:
	{	int unused=-1, what=0;
		blacs_get_(&unused, &what, &blacsContext);
		blacs_gridinit_(&blacsContext, "Row-major", &nProcsRow, &nProcsCol);
		blacs_gridinfo_(&blacsContext, &nProcsRow, &nProcsCol, &iProcRow, &iProcCol);
		assert(mpiUtil->iProcess() == iProcRow * nProcsCol + iProcCol); //this mapping is assumed below, so check
	}
	logPrintf("Initialized %d x %d process BLACS grid.\n", nProcsRow, nProcsCol);
	
	//Initialize matrix distribution:
	logPrintf("Setting up ScaLAPACK matrix with dimension %d\n", N); logFlush();
	if(N <= blockSize * (std::max(nProcsRow, nProcsCol) - 1))
		die("No data on some processes: reduce blockSize or # processes.\n");
	iRowsMine = distributedIndices(N, blockSize, iProcRow, nProcsRow); //indices of rows on current process
	iColsMine = distributedIndices(N, blockSize, iProcCol, nProcsCol); //indices of cols on current process
	nRowsMine = iRowsMine.size();
	nColsMine = iColsMine.size();
	nDataMine = nRowsMine * nColsMine;
	{	int zero=0, info;
		descinit_(desc, &N, &N, &blockSize, &blockSize, &zero, &zero, &blacsContext, &nRowsMine, &info);
		assert(info==0);
	}
}

#include <core/matrix.h>

//Read dense matrix from file into a distributed block cyclic matrix
BlockCyclicMatrix::Buffer BlockCyclicMatrix::readMatrix(string fname) const
{	matrix mat = zeroes(N, N);
	mat.read_real(fname.c_str());
	Buffer out(nDataMine);
	for(int iRow: iRowsMine)
		for(int iCol: iColsMine)
			out[localIndex(iRow, iCol)] = mat(iCol,iRow).real(); //switch row-major to col-major
	return out;
}

//Calculate error between distributed matrices
double BlockCyclicMatrix::matrixErr(const Buffer& A, const Buffer& B) const
{	double errSq = 0.;
	for(size_t i=0; i<nDataMine; i++)
		errSq += std::pow(A[i]-B[i], 2);
	mpiUtil->allReduce(errSq, MPIUtil::ReduceSum);
	return sqrt(errSq/(N*N));
}

//Calculate error between a distributed matrix and identity
double BlockCyclicMatrix::identityErr(const Buffer& A) const
{	double errSq = 0.;
	const double* Aptr = A.data();
	for(int iRow: iRowsMine)
		for(int iCol: iColsMine)
			errSq += std::pow(*(Aptr++) - (iRow==iCol ? 1. : 0.), 2);
	mpiUtil->allReduce(errSq, MPIUtil::ReduceSum);
	return sqrt(errSq/(N*N));
}

//Print all pieces of distributed block cyclic matrix
void BlockCyclicMatrix::printMatrix(const Buffer& mat, const char* name) const
{	ostringstream oss; 
	oss << "\nOn process " << mpiUtil->iProcess() << ":\n";
	for(int iRow=0; iRow<N; iRow++)
	{	for(int iCol=0; iCol<N; iCol++)
		{	int index = localIndex(iRow, iCol);
			oss.width(9);
			oss.precision(5);
			if(index<0) oss << "########"; else oss << std::fixed << mat[index];
		}
		oss << '\n';
	}
	string buf = oss.str();
	if(mpiUtil->isHead())
	{	if(strlen(name)>0)
			logPrintf("\n----------- Matrix %s -----------\n", name);
		for(int iProc=0; iProc<mpiUtil->nProcesses(); iProc++)
		{	if(iProc) mpiUtil->recv(buf, iProc, 0);
			logPrintf("%s", buf.c_str());
		}
		logFlush();
	}
	else mpiUtil->send(buf, 0, 0);
}

//Test with a random matrix:
void BlockCyclicMatrix::testRandom(double fillFactor) const
{
	//Create and diagonalize test matrix:
	Buffer A(nDataMine), VR, VL; 
	double* Adata = A.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
		{	Random::seed(iRow + N*iCol); //to make matrix independent of parallelization
			*(Adata++) = (Random::uniform() < fillFactor) ? Random::normal() : 0.;
		}
	Buffer Acopy(A); //run diagonalization on a destructible copy
	std::vector<complex> E = diagonalize(Acopy, VR, VL);
	
	//Determine error in eigen-decomposition:
	checkDiagonalization(A, VR, VL, E);
}

void BlockCyclicMatrix::checkDiagonalization(const Buffer& A, const Buffer& VR, const Buffer& VL, const std::vector<complex>& E) const
{
	//Report eigenvalue statistics:
	int nReal = 0;
	double EreMin = +DBL_MAX, EreMax = -DBL_MAX, EreMean = 0., EreSqSum = 0.;
	double EimMin = +DBL_MAX, EimMax = -DBL_MAX, EimMean = 0., EimSqSum = 0.;
	for(const complex& e: E)
	{	double re = e.real(), im = fabs(e.imag());
		if(not im) nReal++;
		EreMin = std::min(EreMin, re); EreMax = std::max(EreMax, re);
		EimMin = std::min(EimMin, im); EimMax = std::max(EimMax, im);
		EreMean += re; EreSqSum += std::pow(re,2);
		EimMean += im; EimSqSum += std::pow(im,2);
	}
	EreMean /= N; EimMean /= N;
	double EreStd = sqrt(EreSqSum/N - std::pow(EreMean,2));
	double EimStd = sqrt(EimSqSum/N - std::pow(EimMean,2));
	logPrintf("Eigenvalue statistics:\n");
	logPrintf("\t%d real and %d complex pairs\n", nReal, (N-nReal)/2);
	logPrintf("\tReal part:  min: %9.3lg  max: %9.3lg  mean: %9.3lg  std: %9.3lg\n", EreMin, EreMax, EreMean, EreStd);
	logPrintf("\t|Im part|:  min: %9.3lg  max: %9.3lg  mean: %9.3lg  std: %9.3lg\n", EimMin, EimMax, EimMean, EimStd);
	
	//Check VL-VR overlap:
	BlockCyclicMatrix::Buffer O, Oref(nDataMine);
	matMult(1., VL,true, VR,false, 0., O);
	double* OrefPtr = Oref.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
			*(OrefPtr++) = (iRow==iCol ? (E[iRow].imag() ? 0.5 : 1.) : 0.);
	logPrintf("RMSE VL^VR: %le\n", matrixErr(O,Oref));
	//printMatrix(O, "O");
	
	//Form matrix of eigenvalues:
	BlockCyclicMatrix::Buffer Emat(nDataMine);
	double* Edata = Emat.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
		{	double val = 0.;
			if(iRow==iCol) val = E[iRow].real();
			if((iRow==iCol+1) and (E[iRow].imag()<0.)) val = E[iRow].imag();
			if((iRow==iCol-1) and (E[iRow].imag()>0.)) val = E[iRow].imag();
			*(Edata++) = val;
		}
	BlockCyclicMatrix::Buffer lhs(nDataMine), rhs(nDataMine);
	
	//Right eigenvector error report:
	matMult(1., A,false, VR,false, 0., lhs);
	matMult(1., VR,false, Emat,false, 0., rhs);
	logPrintf("RMS A*VR-VR*E: %le\n", matrixErr(lhs,rhs));
	
	//Left eigenvector error report:
	matMult(1., VL,true, A,false, 0., lhs);
	matMult(1., Emat,false, VL,true, 0., rhs);
	logPrintf("RMS VL'*A-E*VL': %le\n", matrixErr(lhs,rhs));
}


std::vector<complex> BlockCyclicMatrix::diagonalize(Buffer& A, Buffer& VR, Buffer& VL, bool shouldBalance, bool shouldSort) const
{	static StopWatch watch("BlockCyclicMatrix::diagonalize"); watch.start();
	Buffer scale; std::vector<int> evalSort; //optional scale factors and eigenvalue sorting
	if(shouldBalance) scale = balance(A); //Balance matrix
	Buffer Q = hessenberg(A); //Hessenberg reduction
	std::vector<complex> evals = schur(A, Q); //Schur decomposition and eigenvalues
	if(shouldSort) evalSort = sortEvals(evals); //Sort eigenvalues
	getEvecs(A, Q, VR, VL,  //Get eigenvectors ...
		shouldBalance ? &scale : NULL, //... accounting for scale factors if balanced above
		shouldSort ? &evalSort : NULL); //... matching eigenvalue permutation if sorted above
	watch.stop();
	return evals;
}


BlockCyclicMatrix::Buffer BlockCyclicMatrix::balance(Buffer& A) const
{	static StopWatch watch("BlockCyclicMatrix::balance"); watch.start();
	assert(A.size()==nDataMine); 
	int iLo = 1, iHi = N;
	Buffer scaleFactors(N, 1.);
	logPrintf("Balancing matrix ... "); logFlush();
	int info = 0;
	pdgebal_("Scale", &N, A.data(), desc, &iLo, &iHi, scaleFactors.data(), &info);
	if(info < 0) die("Error in argument# %d to pdgebal.\n", -info);
	//Report range of scale factors:
	double scaleMin = +DBL_MAX, scaleMax = -DBL_MAX;
	for(const double s: scaleFactors)
	{	scaleMin = std::min(s, scaleMin);
		scaleMax = std::max(s, scaleMax);
	}
	logPrintf("done. Scale factor range: [ %lg , %lg ]\n", scaleMin, scaleMax);
	watch.stop();
	return scaleFactors;
}

BlockCyclicMatrix::Buffer BlockCyclicMatrix::hessenberg(Buffer& H) const
{	static StopWatch watch("BlockCyclicMatrix::hessenberg"); watch.start();
	assert(H.size()==nDataMine);
	
	//Hessenberg reduction by Householder transformations:
	int iLo = 1, iHi = N;
	int lwork = -1, one = 1, info = 0;
	Buffer work(1), tau(nColsMine);
	logPrintf("Hessenberg reduction ... "); logFlush();
	for(int pass=0; pass<2; pass++) //first pass is workspace query, next pass is actual calculation
	{	pdgehrd_(&N, &iLo, &iHi, H.data(), &one, &one, desc, tau.data(), work.data(), &lwork, &info);
		if(info < 0)
		{	int errCode = -info;
			if(errCode < 100) die("Error in argument# %d to pdgehrd.\n", errCode)
			else die("Error in entry %d of argument# %d to pdgehrd.\n", errCode%100, errCode/100)
		}
		if(pass) break; //done
		//After first-pass, use results of work-space query to allocate:
		lwork = int(work.data()[0]);
		work.resize(lwork);
	}
	logPrintf("done.\n");
	
	//Get orthogonal matrix correspnding to Householder transformations:
	//--- initialize Q to identity:
	Buffer Q(nDataMine);
	double* Qptr = Q.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
			*(Qptr++) = (iRow==iCol ? 1. : 0.);
	work[0] = 0; lwork = -1; //for workspace query
	logPrintf("Extracting rotations ... "); logFlush();
	for(int pass=0; pass<2; pass++) //first pass is workspace query, next pass is actual calculation
	{	pdormhr_("Left", "NoTrans", &N, &N, &iLo, &iHi, H.data(), &one, &one, desc,
			tau.data(), Q.data(), &one, &one, desc, work.data(), &lwork, &info);
		if(info < 0)
		{	int errCode = -info;
			if(errCode < 100) die("Error in argument# %d to pdormhr.\n", errCode)
			else die("Error in entry %d of argument# %d to pdormhr.\n", errCode%100, errCode/100)
		}
		if(pass) break; //done
		//After first-pass, use results of work-space query to allocate:
		lwork = int(work.data()[0]);
		work.resize(lwork);
	}
	logPrintf("done.\n");
	
	//Set H to strict upper Hessenberg form:
	double* Hdata = H.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
		{	if(iRow > iCol+1) *Hdata = 0.;
			Hdata++;
		}
	watch.stop();
	return Q;
}

extern "C" {
	//Tweaked version of pdhseqr to fix some workspace bugs implemented in PDHSEQRf.f
	void pdhseqrf_(const char* job, const char* compz, const int* n, const int* ilo, const int* ihi, double* h, const int* desch,
		double* wr, double* wi, double* z, const int* descz, double* work, const int* lwork, int* iwork, const int* liwork, int* info);
}

//Schur decomposition and eigenvalues:
std::vector<complex> BlockCyclicMatrix::schur(Buffer& H, Buffer& Q) const
{	static StopWatch watch("BlockCyclicMatrix::schur"); watch.start();
	assert(H.size()==nDataMine);
	assert(Q.size()==nDataMine);
	int iLo = 1, iHi = N;
	Buffer wr(N), wi(N); //real and imaginary parts of eigenvalues
	Buffer work(1); int lwork = -1, info = 0; //for workspace query
	std::vector<int> iwork(N); int liwork = -1; //for workspace query
	logPrintf("Schur decomposition ... "); logFlush();
	for(int pass=0; pass<2; pass++) //first pass is workspace query, next pass is actual calculation
	{	pdhseqr_("Schur", "Vectors", &N, &iLo, &iHi, H.data(), desc, wr.data(), wi.data(),
			Q.data(), desc, work.data(), &lwork, iwork.data(), &liwork, &info);
		if(info < 0)
		{	int errCode = -info;
			if(errCode < 100) die("Error in argument# %d to pdhseqr.\n", errCode)
			else die("Error in entry %d of argument# %d to pdhseqr.\n", errCode%100, errCode/100)
		}
		if(info > 0) die("Up to %d eigenvalues failed to converge.\n", info);
		if(pass) break; //done
		//After first-pass, use results of work-space query to allocate:
		int nExtra = 2*nDataMine + 10*blockSize*blockSize; //note bug in pdhseqr: workspace underestimated
		lwork = int(work.data()[0])+nExtra; work.resize(lwork);
		liwork = int(iwork.data()[0])+nExtra; iwork.resize(liwork);
	}
	logPrintf("done.\n");
	watch.stop();

	//Collect eigenvalues into complex array:
	std::vector<complex> evals(N);
	for(int i=0; i<N; i++)
		evals[i] = complex(wr[i], wi[i]);
	return evals;
}


std::vector<int> BlockCyclicMatrix::sortEvals(std::vector<complex>& evals) const
{	//Helper class to sort complex array by ascending real part and descending imaginary part
	struct IndexCompare
	{	const std::vector<complex>& array;
		IndexCompare(const std::vector<complex>& array) : array(array) {}
		bool operator()(int i1, int i2) const 
		{	complex diff = array[i1] - array[i2];
			return diff.real() ? (diff.real()<0.) : (diff.imag()>0.);
		}
	};
	//Determine sort indices:
	std::vector<int> sortIndex(N);
	for(int i=0; i<N; i++) sortIndex[i] = i;
	std::sort(sortIndex.begin(), sortIndex.end(), IndexCompare(evals));
	//Apply sort indices:
	std::vector<complex> out(N);
	for(int i=0; i<N; i++) out[i] = evals[sortIndex[i]];
	std::swap(out, evals);
	return sortIndex;
}

void BlockCyclicMatrix::getEvecs(const Buffer& T, const Buffer& Q, Buffer& VR, Buffer& VL, const Buffer* scaleFactors, const std::vector<int>* evalSort) const
{	static StopWatch watch("BlockCyclicMatrix::getEvecs"); watch.start();
	assert(T.size()==nDataMine);
	assert(Q.size()==nDataMine);
	if(scaleFactors) assert(int(scaleFactors->size())==N);
	
	//Underflow/overflow and precision constants:
	double uFlow = dlamch_("Safe minimum");
	double oFlow = 1./uFlow;
	dlabad_(&uFlow, &oFlow);
	const double prec = dlamch_("Precision");
	const double sNum = uFlow*(N/prec);
	const double bNum = (1.-prec)/sNum;
	
	//Collect column 1-norms and tri-diagonal portion of matrix on all processes:
	Buffer tNorm(N, 0.); //column 1-norms used for overflow mitigation below
	Buffer tDiag(N, 0.), tDiagU(N, 0.), tDiagL(N, 0.); //diagonal, upper and lower diagonal entries
	{	const double* Tdata = T.data();
		for(int j: iColsMine)
			for(int i: iRowsMine)
			{	const double& t = *(Tdata++);
				if(t) tNorm[j] += fabs(t);
				if(i == j) tDiag[i] = t;
				if(i+1 == j) tDiagU[i] = t;
				if(j+1 == i) tDiagL[j] = t;
			}
	}
	mpiUtil->allReduceData(tNorm, MPIUtil::ReduceSum);
	mpiUtil->allReduceData(tDiag, MPIUtil::ReduceSum);
	mpiUtil->allReduceData(tDiagU, MPIUtil::ReduceSum);
	mpiUtil->allReduceData(tDiagL, MPIUtil::ReduceSum);
	
	//Eigenalue inverse sort index:
	std::vector<int> sortIndexInv(N);
	if(evalSort) for(int i=0; i<N; i++) sortIndexInv[evalSort->at(i)] = i;
	else for(int i=0; i<N; i++) sortIndexInv[i] = i;
	
	//Temporaries for blas calls:
	int notTrans=0, isTrans=1, two=2, info=0;
	double oneD = 1.;
	double x[4], xNorm, scale; //1x1 or 2x2 matrix used in dlanl2; its norm and scale factor
	Buffer Z(nDataMine); //eigenvectors of T (multiplied by Q at the end)
	//Right eigenvector calculation:
	logPrintf("Computing right eigenvectors of Schur matrix ... "); logFlush();
	for(int ki=N-1; ki>=0; ki--)
	{	bool complexPair = ((ki>0) and (tDiagL[ki-1]!=0.));
		int kiBlockSize = complexPair ? 2  : 1; //current block size in ki
		int kiStart = ki + 1 - kiBlockSize; //start of current block in ki
		//Get the eigenvalue:
		double wr = tDiag[ki];
		double wi = complexPair ? sqrt(fabs(tDiagU[kiStart])) * sqrt(fabs(tDiagL[kiStart])) : 0.; //written like this to avoid over/under-flow
		double sMin = std::max(prec*(fabs(wr)+fabs(wi)), sNum); //small number threshold for this eigenvector (pair)
		//Construct RHS:
		Buffer rhs(kiBlockSize*N);
		if(complexPair)
		{	if(fabs(tDiagU[kiStart]) > fabs(tDiagL[kiStart]))
			{	rhs[kiStart] = 1.;
				rhs[ki+N] = wi/tDiagU[kiStart];
			}
			else
			{	rhs[kiStart] = -wi/tDiagL[kiStart];
				rhs[ki+N] = 1.;
			}
		}
		else rhs[ki] = 1.;
		for(int bk=0; bk<kiBlockSize; bk++)
		{	int iColMine = localColIndex(kiStart+bk);
			if(iColMine >= 0)
			{	int iRowMineStart, iRowMineStop;
				getRange(iRowsMine, 0, kiStart, iRowMineStart, iRowMineStop);
				for(int iRowMine=iRowMineStart; iRowMine<iRowMineStop; iRowMine++)
					rhs[iRowsMine[iRowMine]+bk*N] = -rhs[kiStart+bk+bk*N] * T[iRowMine+iColMine*nRowsMine]; //set on exacty one process
			}
			mpiUtil->allReduce(&rhs[bk*N], kiStart, MPIUtil::ReduceSum); //make available on all processes
		}
		//Solve upper quasi-triangular system (T[:kiStart,:kiStart] - (wr+i*wi))*x = scale*rhs
		for(int j=kiStart-1; j>=0.; j--)
		{	int jBlockSize = ((j>0) and (tDiagL[j-1]!=0.)) ? 2 : 1; //current block size in j
			int jStart = j+1-jBlockSize; //start of current block in j
			double T22[4] = { tDiag[jStart], tDiagL[jStart], tDiagU[jStart], tDiag[j] }; //2x2 diagonal block of T (only 1x1 valid/needed if jStart==j)
			dlaln2_(&notTrans, &jBlockSize, &kiBlockSize, &sMin, &oneD,
				T22, &two, &oneD, &oneD,
				&rhs[jStart], &N, &wr, &wi, x, &two,
				&scale, &xNorm, &info);
			//Scale relevant x to avoid overflow in rhs update:
			if((xNorm>1.) and (std::max(tNorm[jStart],tNorm[j])>bNum/xNorm))
			{	double xNormInv = 1./xNorm;
				for(int bk=0; bk<kiBlockSize; bk++)
					for(int bj=0; bj<jBlockSize; bj++)
						x[bj+2*bk] *= xNormInv;
				scale *= xNormInv;
			}
			if(scale != 1.)
				for(int bk=0; bk<kiBlockSize; bk++)
					cblas_dscal(ki+1, scale, &rhs[bk*N],1); //scale when needed
			//Update the right hand side
			for(int bk=0; bk<kiBlockSize; bk++)
			{	Buffer rhsUpdate(jStart);
				for(int bj=0; bj<jBlockSize; bj++)
				{	double xCur = x[bj+2*bk];
					rhs[jStart+bj + bk*N] = xCur;
					int iColMine = localColIndex(jStart+bj);
					if(iColMine >= 0)
					{	int iRowMineStart, iRowMineStop;
						getRange(iRowsMine, 0, jStart, iRowMineStart, iRowMineStop);
						for(int iRowMine=iRowMineStart; iRowMine<iRowMineStop; iRowMine++)
							rhsUpdate[iRowsMine[iRowMine]] -= xCur * T[iRowMine+iColMine*nRowsMine];
					}
				}
				if(jStart)
				{	mpiUtil->allReduceData(rhsUpdate, MPIUtil::ReduceSum);
					cblas_daxpy(jStart, 1., rhsUpdate.data(),1, &rhs[bk*N],1);
				}
			}
			j = jStart;
		}
		//Scale max entry to 1:
		double rhsMax = rhs[cblas_idamax(kiBlockSize*N, rhs.data(),1)];
		cblas_dscal(kiBlockSize*N, 1./fabs(rhsMax), rhs.data(),1);
		//Distribute the eigenvector to Z on relevant processes:
		for(int bk=0; bk<kiBlockSize; bk++)
		{	int colDest = sortIndexInv[kiStart+bk];
			int iColMine = localColIndex(colDest);
			if(iColMine >= 0)
			{	for(int iRowMine=0; iRowMine<nRowsMine; iRowMine++)
					Z[iRowMine+iColMine*nRowsMine] = rhs[iRowsMine[iRowMine]+bk*N];
			}
		}
		ki = kiStart;
	}
	logPrintf("done.\nRotating right eigenvectors to original basis ... "); logFlush();
	//--- multiply by Q
	matMult(1., Q,false, Z,false, 0.,VR);
	//--- account for scaleFactors if necessary:
	Buffer scaleMine, scaleMineInv;
	if(scaleFactors)
	{	scaleMine.reserve(nRowsMine);
		scaleMineInv.reserve(nRowsMine);
		for(int iRow: iRowsMine)
		{	double s = scaleFactors->at(iRow);
			scaleMine.push_back(s);
			scaleMineInv.push_back(1./s);
		}
		//Apply scale factors:
		double* VRdata = VR.data();
		for(int iColMine=0; iColMine<nColsMine; iColMine++)
			for(int iRowMine=0; iRowMine<nRowsMine; iRowMine++)
				*(VRdata++) *= scaleMine[iRowMine];
	}
	logPrintf("done.\n");
	
	//Left eigenvector calculation:
	logPrintf("Computing right eigenvectors of Schur matrix ... "); logFlush();
	for(int ki=0; ki<N; ki++)
	{	bool complexPair = (ki+1<N) and (tDiagL[ki]!=0.);
		int kiBlockSize = complexPair ? 2  : 1; //current block size in ki
		int kiStop = ki + kiBlockSize-1; //end of current block in ki
		//Get the eigenvalue:
		double wr = tDiag[ki];
		double mwi = complexPair ? -sqrt(fabs(tDiagL[ki]))*sqrt(fabs(tDiagU[ki])) : 0; //written like this to avoid over/under-flow
		double sMin = std::max(prec*(fabs(wr)+fabs(mwi)), sNum); //small number threshold for this eigenvector (pair)
		//Construct RHS:
		Buffer rhs(kiBlockSize*N);
		if(complexPair)
		{	if(fabs(tDiagU[ki]) > fabs(tDiagL[ki]))
			{	rhs[ki] = -mwi/tDiagU[ki];
				rhs[kiStop+N] = 1.;
			}
			else
			{	rhs[ki] = 1.;
				rhs[kiStop+N] = mwi/tDiagL[ki];
			}
		}
		else rhs[ki] = 1.;
		for(int bk=0; bk<kiBlockSize; bk++)
		{	int iRowMine = localRowIndex(ki+bk);
			if(iRowMine >= 0)
			{	int iColMineStart, iColMineStop;
				getRange(iColsMine, kiStop+1, N, iColMineStart, iColMineStop);
				for(int iColMine=iColMineStart; iColMine<iColMineStop; iColMine++)
					rhs[iColsMine[iColMine]+bk*N] = -rhs[ki+bk+bk*N] * T[iRowMine+iColMine*nRowsMine]; //set on exacty one process
			}
			mpiUtil->allReduce(&rhs[(kiStop+1)+bk*N], N-(kiStop+1), MPIUtil::ReduceSum); //make available on all processes
		}
		//Solve quasi-triangular system (T(kiStop+1:,kiStop+1:) - (wr-i*wi))*x = rhs
		double vCrit = bNum, vMax = 1.; //for scaling
		for(int j=kiStop+1; j<N; j++)
		{	int jBlockSize = ((j+1<N) and (tDiagL[j]!=0.)) ? 2 : 1; //current block size in j
			int jStop = j+jBlockSize-1; //end of current block in j
			//Scale to avoid overflow when forming RHS elements if needed:
			if(std::max(tNorm[j],tNorm[jStop]) > vCrit)
			{	double scaleFac = 1./vMax;
				for(int bk=0; bk<kiBlockSize; bk++)
					cblas_dscal(N-ki, scaleFac, &rhs[ki+bk*N],1);
				vMax = 1.;
				vCrit = bNum;
			}
			//Form RHS elements:
			for(int bk=0; bk<kiBlockSize; bk++)
			{	for(int bj=0; bj<jBlockSize; bj++)
				{	double rhsUpdate = 0.;
					int iColMine = localColIndex(j+bj);
					if(iColMine >= 0)
					{	int iRowMineStart, iRowMineStop;
						getRange(iRowsMine, kiStop+1, j, iRowMineStart, iRowMineStop);
						for(int iRowMine=iRowMineStart; iRowMine<iRowMineStop; iRowMine++)
							rhsUpdate -= T[iRowMine+iColMine*nRowsMine] * rhs[iRowsMine[iRowMine]+bk*N];
					}
					mpiUtil->allReduce(rhsUpdate, MPIUtil::ReduceSum);
					rhs[j+bj+bk*N] += rhsUpdate;
				}
			}
			//Solve kiBlockSize x jBlockSize complex equation to get x:
			double T22[4] = { tDiag[j], tDiagL[j], tDiagU[j], tDiag[jStop] }; //2x2 diagonal block of T (only 1x1 valid/needed if jStop==j)
			dlaln2_((jBlockSize==2 ? &isTrans : &notTrans),
				&jBlockSize, &kiBlockSize, &sMin, &oneD,
				T22, &two, &oneD, &oneD, 
				&rhs[j], &N, &wr, &mwi, x, &two,
				&scale, &xNorm, &info);
			//Scale if necessary:
			if(scale != 1.)
			{	for(int bk=0; bk<kiBlockSize; bk++)
					cblas_dscal(N-ki, scale, &rhs[ki+bk*N],1);
			}
			//Update solution:
			for(int bk=0; bk<kiBlockSize; bk++)
				for(int bj=0; bj<jBlockSize; bj++)
				{	rhs[(j+bj)+bk*N] = x[bj+2*bk];
					vMax = std::max(vMax, x[bj+2*bk]);
				}
			vCrit = bNum / vMax;
			j = jStop;
		}
		//Scale max entry to 1:
		double rhsMax = rhs[cblas_idamax(kiBlockSize*N, rhs.data(),1)];
		cblas_dscal(kiBlockSize*N, 1./fabs(rhsMax), rhs.data(),1);
		//Distribute the eigenvector to Z on relevant processes:
		for(int bk=0; bk<kiBlockSize; bk++)
		{	int colDest = sortIndexInv[ki+bk];
			int iColMine = localColIndex(colDest);
			if(iColMine >= 0)
			{	for(int iRowMine=0; iRowMine<nRowsMine; iRowMine++)
					Z[iRowMine+iColMine*nRowsMine] = rhs[iRowsMine[iRowMine]+bk*N];
			}
		}
		ki = kiStop;
	}
	logPrintf("done.\nRotating left eigenvectors to original basis ... "); logFlush();
	//--- multiply by Q
	matMult(1., Q,false, Z,false, 0.,VL);
	//--- account for scaleFactors if necessary:
	if(scaleFactors)
	{	double* VLdata = VL.data();
		for(int iColMine=0; iColMine<nColsMine; iColMine++)
			for(int iRowMine=0; iRowMine<nRowsMine; iRowMine++)
				*(VLdata++) *= scaleMineInv[iRowMine];
	}
	logPrintf("done.\n");
	
	//Fix normalization of eigenvectors:
	logPrintf("Normalizing left - right eigenvectors mutually ... "); logFlush();
	Z.clear(); tDiag.clear(); tDiagU.clear(); //only keep tDiagL to test real/complex eval below; free rest
	for(int ki=0; ki<N; ki++)
	{	int kiOrig = evalSort ? evalSort->at(ki) : ki; //only for e-val lookup in next line
		bool complexPair = (ki+1<N) and (tDiagL[kiOrig]!=0.);
		int kiBlockSize = complexPair ? 2 : 1;
		int kiStop = ki+kiBlockSize-1;
		//Fetch entire right and left eigenvector (pair) on all processes:
		Buffer VRcur(kiBlockSize*N), VLcur(kiBlockSize*N);
		for(int bk=0; bk<kiBlockSize; bk++)
		{	int iColMine = localColIndex(ki+bk);
			if(iColMine >= 0)
				for(int iRowMine=0; iRowMine<nRowsMine; iRowMine++)
				{	int iRow = iRowsMine[iRowMine];
					int index = iRowMine+iColMine*nRowsMine;
					VLcur[iRow+bk*N] = VL[index];
					VRcur[iRow+bk*N] = VR[index];
				}
		}
		mpiWorld->allReduceData(VRcur, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(VLcur, MPIUtil::ReduceSum);
		//Apply scaling:
		if(complexPair)
		{	//Determine max entry of each and mutual dot product:
			int iMaxNormR = -1; double maxNormR = 0.;
			int iMaxNormL = -1; double maxNormL = 0.;
			complex dotProduct = 0.;
			for(int iRow=0; iRow<N; iRow++)
			{	complex zR(VRcur[iRow], VRcur[iRow+N]); double normCurR = zR.norm();
				complex zL(VLcur[iRow], VLcur[iRow+N]); double normCurL = zL.norm();
				if(normCurR > maxNormR) { maxNormR = normCurR; iMaxNormR = iRow; }
				if(normCurL > maxNormL) { maxNormL = normCurL; iMaxNormL = iRow; }
				dotProduct += zL.conj()*zR;
			}
			//Make max abs entry = 1 first, and then apply complex phases to ensure dot(VL,VR) = id:
			complex zRmax(VRcur[iMaxNormR], VRcur[iMaxNormR+N]);
			complex zLmax(VLcur[iMaxNormL], VLcur[iMaxNormL+N]);
			if(dotProduct.norm() < prec) dotProduct = 1.; //unlikely (impossible?) corner case of orthogonal L-R evecs
			dotProduct /= (zLmax.conj() * zRmax); //acount for max-value pre-scaling done to the columns below
			complex dotFac = sqrt(dotProduct);
			complex scaleFacR = complex(1.,0)/(zRmax * dotFac);
			complex scaleFacL = complex(1.,0)/(zLmax * dotFac.conj());
			for(int iRow=0; iRow<N; iRow++)
			{	complex zR = scaleFacR * complex(VRcur[iRow], VRcur[iRow+N]);
				complex zL = scaleFacL * complex(VLcur[iRow], VLcur[iRow+N]);
				VRcur[iRow] = zR.real(); VRcur[iRow+N] = zR.imag();
				VLcur[iRow] = zL.real(); VLcur[iRow+N] = zL.imag();
			}
		}
		else
		{	//Determine max entry:
			int iMaxAbsR = -1; double maxAbsR = 0.;
			int iMaxAbsL = -1; double maxAbsL = 0.;
			double dotProduct = 0.;
			for(int iRow=0; iRow<N; iRow++)
			{	double absCurL = fabs(VLcur[iRow]); if(absCurL > maxAbsL) {	maxAbsL = absCurL; iMaxAbsL = iRow; }
				double absCurR = fabs(VRcur[iRow]); if(absCurR > maxAbsR) {	maxAbsR = absCurR; iMaxAbsR = iRow; }
				dotProduct += VLcur[iRow] * VRcur[iRow];
			}
			//Make max abs entry = 1, and then scale to ensure dot(VL,VR) = id:
			double zRmax = VRcur[iMaxAbsR];
			double zLmax = VLcur[iMaxAbsL];
			if(std::pow(dotProduct,2) < prec) dotProduct = 1.; //unlikely (impossible?) corner case of orthogonal L-R evecs
			dotProduct /= (zRmax * zLmax); //acount for max-value pre-scaling done to the columns below
			double dotFac = sqrt(fabs(dotProduct));
			double scaleFacR = 1./(zRmax * dotFac);
			double scaleFacL = 1./(zLmax * copysign(dotFac,dotProduct));
			for(int iRow=0; iRow<N; iRow++)
			{	VRcur[iRow] *= scaleFacR;
				VLcur[iRow] *= scaleFacL;
			}
		}
		//Set relevant pieces of eigenvector back:
		for(int bk=0; bk<kiBlockSize; bk++)
		{	int iColMine = localColIndex(ki+bk);
			if(iColMine >= 0)
				for(int iRowMine=0; iRowMine<nRowsMine; iRowMine++)
				{	int iRow = iRowsMine[iRowMine];
					int index = iRowMine+iColMine*nRowsMine;
					VL[index] = VLcur[iRow+bk*N];
					VR[index] = VRcur[iRow+bk*N];
				}
		}
		ki = kiStop;
	}
	logPrintf("done.\n");
	watch.stop();
}

void BlockCyclicMatrix::matMult(double alpha, const Buffer& A, bool transA, const Buffer& B, bool transB, double beta, Buffer& C) const
{	static StopWatch watch("BlockCyclicMatrix::matMult"); watch.start();
	assert(A.size()==nDataMine);
	assert(B.size()==nDataMine);
	if(beta) assert(C.size()==nDataMine); else C.resize(nDataMine);
	char transAchar = transA ? 'T' : 'N';
	char transBchar = transB ? 'T' : 'N';
	int one = 1;
	pdgemm_(&transAchar, &transBchar, &N, &N, &N, &alpha,
		A.data(), &one, &one, desc,
		B.data(), &one, &one, desc, &beta,
		C.data(), &one, &one, desc);
	watch.stop();
}


#endif //SCALAPACK_ENABLED
