#ifdef SCALAPACK_ENABLED
#include "BlockCyclicMatrix.h"
#include <core/Util.h>
#include <core/BlasExtra.h>
#include <mkl_blacs.h>
#include <mkl_pblas.h>
#include <mkl_scalapack.h>
#include <mkl_lapack.h>

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
	//Initialize row and column communicators:
	mpiRow = new MPIUtil(0, NULL, MPIUtil::ProcDivision(mpiUtil, 0, iProcRow)); assert(mpiRow->iProcess() == iProcCol);
	mpiCol = new MPIUtil(0, NULL, MPIUtil::ProcDivision(mpiUtil, 0, iProcCol)); assert(mpiCol->iProcess() == iProcRow);
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

BlockCyclicMatrix::~BlockCyclicMatrix()
{	delete mpiRow;
	delete mpiCol;
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
double BlockCyclicMatrix::identityErr(const Buffer& A, double* offDiagErr) const
{	double errSq = 0., offErrSq = 0.;
	const double* Aptr = A.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
		{	double errSqCur = std::pow(*(Aptr++) - (iRow==iCol ? 1. : 0.), 2);
			errSq += errSqCur;
			if(iRow!=iCol) offErrSq += errSqCur;
		}
	mpiUtil->allReduce(errSq, MPIUtil::ReduceSum);
	mpiUtil->allReduce(offErrSq, MPIUtil::ReduceSum);
	if(offDiagErr) *offDiagErr = sqrt(offErrSq/(N*N));
	return sqrt(errSq/(N*N));
}

//Print all pieces of distributed block cyclic matrix
void BlockCyclicMatrix::printMatrix(const Buffer& mat, const char* name) const
{	ostringstream oss; 
	oss << "\nOn process (" << iProcRow << ',' << iProcCol << ":\n";
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
		{	//Simple reproducible xorshift RNG:
			uint32_t x = iRow + N*iCol;
			x ^= x << 13; x ^= x >> 17; x ^= x << 5; double f = 2.3283e-10*x; //in [0,1)
			x ^= x << 13; x ^= x >> 17; x ^= x << 5; double val = 2.3283e-10*x - 0.5; //in [-0.5,0.5)
			*(Adata++) = (f < fillFactor) ? val : 0.;
		}
	Buffer Acopy(A); //run diagonalization on a destructible copy
	std::vector<complex> E = diagonalize(Acopy, VR, VL);
	
	//Determine error in eigen-decomposition:
	checkDiagonalization(A, VR, VL, E);
}

void BlockCyclicMatrix::checkDiagonalization(const Buffer& A, const Buffer& VR, const Buffer& VL, const std::vector<complex>& E) const
{
	//Get matrix norm:
	int one = 1;
	double Anorm = pdlange_("F", &N, &N, A.data(), &one, &one, desc, NULL);
	
	//Report eigenvalue statistics:
	int nReal = 0;
	double EreMin = +DBL_MAX, EreMax = -DBL_MAX, EreMean = 0., EreSqSum = 0.;
	double EimMin = +DBL_MAX, EimMax = -DBL_MAX, EimMean = 0., EimSqSum = 0.;
	for(const complex& e: E)
	{	double re = e.real(), im = fabs(e.imag());
		if(im < 1e-14*Anorm) nReal++;
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
	BlockCyclicMatrix::Buffer O;
	matMult(1., VL,true, VR,false, 0., O);
	logPrintf("RMSE VL^VR: %le\n", identityErr(O));
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


std::vector<complex> BlockCyclicMatrix::diagonalize(Buffer& A, Buffer& VR, Buffer& VL, bool shouldBalance) const
{	static StopWatch watch("BlockCyclicMatrix::diagonalize"); watch.start();
	Buffer scale; std::vector<int> evalSort; //optional scale factors and eigenvalue sorting
	if(shouldBalance) scale = balance(A); //Balance matrix
	Buffer Q = hessenberg(A); //Hessenberg reduction
	std::vector<complex> evals = schur(A, Q); //Schur decomposition and eigenvalues
	getEvecs(A, Q, VR, VL, shouldBalance ? &scale : NULL); //Transform Schur vectors to eigenvectors
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
		lwork = liwork = 2*std::max(int(work.data()[0]), int(iwork.data()[0])); //note bug in pdhseqr: liwork underestimated
		work.resize(lwork);
		iwork.resize(liwork);
	}
	logPrintf("done.\n");
	watch.stop();

	//Collect eigenvalues into complex array:
	std::vector<complex> evals(N);
	for(int i=0; i<N; i++)
		evals[i] = complex(wr[i], wi[i]);
	return evals;
}

//Eigenvector transformation
void BlockCyclicMatrix::getEvecs(const Buffer& T, const Buffer& Q, Buffer& VR, Buffer& VL, const Buffer* scaleFactors) const
{	static StopWatch watchLeft("BlockCyclicMatrix::leftEvecs"), watchRight("BlockCyclicMatrix::rightEvecs");
	static StopWatch watchLeft1("BlockCyclicMatrix::left1"), watchLeft2("BlockCyclicMatrix::left2");
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
	
	//Temporaries for blas calls:
	int notTrans=0, isTrans=1, two=2, info=0;
	double oneD = 1.;
	double x[4], xNorm, scale; //1x1 or 2x2 matrix used in dlanl2; its norm and scale factor
	Buffer Z(nDataMine); //eigenvectors of T (multiplied by Q at the end)
	
	//Left eigenvector calculation:
	watchLeft.start();
	logPrintf("Computing left eigenvectors of Schur matrix ... "); logFlush();
	
	watchLeft1.start();
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
			/*
			//Scale to avoid overflow when forming RHS elements if needed:
			if(std::max(tNorm[j],tNorm[jStop]) > vCrit)
			{	double scaleFac = 1./vMax;
				printf("Scaling PRE!\n");
				for(int bk=0; bk<kiBlockSize; bk++)
					cblas_dscal(N-ki, scaleFac, &rhs[ki+bk*N],1);
				vMax = 1.;
				vCrit = bNum;
			}
			*/
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
			if(scale != 1.) die_alone("Overflow encountered.\n");
			/*
			//Scale if necessary:
			if(scale != 1.)
			{	printf("Scaling POST!\n");
				for(int bk=0; bk<kiBlockSize; bk++)
					cblas_dscal(N-ki, scale, &rhs[ki+bk*N],1);
			}
			//Update solution:
			for(int bk=0; bk<kiBlockSize; bk++)
				for(int bj=0; bj<jBlockSize; bj++)
				{	rhs[(j+bj)+bk*N] = x[bj+2*bk];
					vMax = std::max(vMax, x[bj+2*bk]);
				}
			*/
			vCrit = bNum / vMax;
			j = jStop;
		}
		/*
		//Scale max entry to 1:
		double rhsMax = rhs[cblas_idamax(kiBlockSize*N, rhs.data(),1)];
		cblas_dscal(kiBlockSize*N, 1./fabs(rhsMax), rhs.data(),1);
		*/
		//Distribute the eigenvector to Z on relevant processes:
		for(int bk=0; bk<kiBlockSize; bk++)
		{	int iColMine = localColIndex(ki+bk);
			if(iColMine >= 0)
			{	for(int iRowMine=0; iRowMine<nRowsMine; iRowMine++)
					Z[iRowMine+iColMine*nRowsMine] = rhs[iRowsMine[iRowMine]+bk*N];
			}
		}
		ki = kiStop;
	}
	watchLeft1.stop();
	
	Buffer Zref(Z); //copy the result above
	
	//New attempt at parallelization:
	watchLeft2.start();
	//--- initialize Z to transpose(T):
	const int one = 1; double zeroD = 0.;
	pdgeadd_("T", &N, &N, &oneD, T.data(),&one,&one,desc, &zeroD, Z.data(),&one,&one,desc);
	//--- compute scale factors for initial RHS of the subsequent triangular solves
	//--- in the process, also zero-out block-diagonal part of transpose(T) stored in Z 
	//--- and prepare the indexing arrays required for synchonizing 2x2 blocks split across processes
	Buffer ZdiagMine; ZdiagMine.reserve(nColsMine);
	int iProcRowNext = (iProcRow+1) % nProcsRow, iProcRowPrev = (iProcRow-1) % nProcsRow;
	int iProcColNext = (iProcCol+1) % nProcsCol, iProcColPrev = (iProcCol-1) % nProcsCol;
	std::vector<int> iRowMinePadded; //iRowMine whose 2x2 operations will happen on this process, and -1 where data is needed from iProcRowNext
	std::vector<int> iPaddedRecv; //indices into iRowMinePadded for data that should be received from iProcRowNext
	std::vector<int> iRowMineSend; //iRowMine that need to be sent to iProcRowPrev
	int nBlocksPerRow = ceildiv(nRowsMine,blockSize);
	iRowMinePadded.reserve(nRowsMine + nBlocksPerRow);
	iPaddedRecv.reserve(nBlocksPerRow);
	iRowMineSend.reserve(nBlocksPerRow);
	for(int ki=0; ki<N; ki++)
	{	bool complexPair = (ki+1<N) and (tDiagL[ki]!=0.);
		int kiBlockSize = complexPair ? 2  : 1; //current block size in ki
		int kiStop = ki + kiBlockSize-1; //end of current block in ki
		int iRowMine, iColMine;
		#define IF_COL_MINE(iCol) iColMine = localColIndex(iCol); if(iColMine >= 0)
		#define ZERO_ENTRY_Z(iRow) iRowMine = localRowIndex(iRow); if(iRowMine >= 0) Z[iRowMine+iColMine*nRowsMine] = 0.;
		if(complexPair)
		{	double wi = sqrt(fabs(tDiagL[ki] * tDiagU[ki]));
			bool Ugreater = (fabs(tDiagU[ki]) > fabs(tDiagL[ki]));
			IF_COL_MINE(ki)     { ZdiagMine.push_back(Ugreater ? wi/tDiagU[ki] : 1.);  ZERO_ENTRY_Z(ki) ZERO_ENTRY_Z(kiStop) }
			IF_COL_MINE(kiStop) { ZdiagMine.push_back(Ugreater ? 1. : -wi/tDiagL[ki]); ZERO_ENTRY_Z(ki) ZERO_ENTRY_Z(kiStop) }
			//Update 2x2 sync arrays:
			iRowMine = localRowIndex(ki); int iRowMine2 = localRowIndex(kiStop);
			if(iRowMine >= 0)
			{	if(iRowMine2 >= 0)
				{	//both ki and kiStop on this process; no communication needed
					iRowMinePadded.push_back(iRowMine);
					iRowMinePadded.push_back(iRowMine2);
				}
				else
				{	//ki on this process, but kiStop on next process 
					iRowMinePadded.push_back(iRowMine);
					iRowMinePadded.push_back(-1); //need to recv kiStop from next process
					iPaddedRecv.push_back(iRowMinePadded.size()-1); //point to last entry added above
				}
			}
			else
			{	if(iRowMine2 >= 0)
				{	//kiStop on this process, but ki on prev process
					iRowMineSend.push_back(iRowMine2); //send data to prev process, which will process the 2x2 block
				}
				//else both not ki and kiStop not involved in this process
			}
		}
		else
		{	IF_COL_MINE(ki) { ZdiagMine.push_back(1.); ZERO_ENTRY_Z(ki) } 
			iRowMine = localRowIndex(ki);
			if(iRowMine >= 0) iRowMinePadded.push_back(iRowMine); //never need communictaion for 1x1 block
		}
		#undef IF_COL_MINE
		#undef ZERO_ENTRY_Z
		ki = kiStop;
	}
	
	{	ostringstream oss; oss << "Process (" << iProcRow << "," << iProcCol << "):";
		oss << " SEND(" << iRowMineSend.size() << ")"; for(int i: iRowMineSend) oss << ' ' << iRowsMine[i];
		oss << " RECV(" << iPaddedRecv.size() << ")"; for(int i: iPaddedRecv) { assert(i); assert(iRowMinePadded[i]==-1); oss << ' ' << iRowsMine[iRowMinePadded[i-1]]+1; }
		printf("\n%s", oss.str().c_str()); fflush(stdout);
	}
	
	//--- apply diagonal scaling factors
	//--- (diagonal blocks still left at zero, so Z is strictly lower triangular)
	{	double* Zdata = Z.data();
		for(int iColMine=0; iColMine<nColsMine; iColMine++)
		{	double diagCur = ZdiagMine[iColMine];
			cblas_dscal(nRowsMine, -diagCur, Zdata,1); //diagonal scaling of off-diagonal terms
			int iRowMine = localRowIndex(iColsMine[iColMine]);
			//if(iRowMine >= 0) Zdata[iRowMine] = diagCur; //set diagonal term
			Zdata += nRowsMine;
		}
	}
	//--- solve set of quasi-triangular systems (T - (wr-i*wi))*x = Z in parallel
	Buffer Tcur(nRowsMine*2), Zupdate(nColsMine*2);
	for(int j=0; j<N; j++)
	{	int jBlockSize = ((j+1<N) and (tDiagL[j]!=0.)) ? 2 : 1; //current block size in j
		int jStop = j+jBlockSize-1; //end of current block in j
		//Determine data ranges requiring opration for these j:
		int iColMineStop, iRowMineStop, iUnusedStart;
		getRange(iColsMine, 0, j, iUnusedStart, iColMineStop); //Note that iColMineStart = 0 
		getRange(iRowsMine, 0, j, iUnusedStart, iRowMineStop); //Note that iRowMineStart = 0 
		//Make T[:,j] available on all processes in each process row
		if(iRowMineStop)
			for(int bj=0; bj<jBlockSize; bj++)
			{	int whoseProcCol = ((j+bj) / blockSize) % nProcsCol;
				if(whoseProcCol == iProcCol)
					eblas_copy(&Tcur[nRowsMine*bj], &T[nRowsMine*localColIndex(j+bj)], iRowMineStop);
				mpiRow->bcast(&Tcur[nRowsMine*bj], iRowMineStop, whoseProcCol);
			}
		if(iColMineStop)
		{	//Update  Z(j,ki) -= sum_(ki < i < j) [ T(i,j) * Z(i,ki) ]
			//--- done as  Z(j,ki) -= sum_(i < j) [ T(i,j) * Z(i,ki) ] since Z is strictly lower triangular (diag part set later)
			//--- compute the local piece of the matrix product
			if(iRowMineStop)
				cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, iColMineStop, jBlockSize, iRowMineStop,
					1., Z.data(),nRowsMine, Tcur.data(),nRowsMine, 0., Zupdate.data(),nColsMine);
			//--- accumulate result on the appropriate process
			for(int bj=0; bj<jBlockSize; bj++)
			{	int whoseProcRow = ((j+bj) / blockSize) % nProcsRow;
				mpiCol->reduce(&Zupdate[nColsMine*bj], iColMineStop, MPIUtil::ReduceSum, whoseProcRow);
				if(whoseProcRow == iProcRow)
					cblas_daxpy(iColMineStop, -1., &Zupdate[nColsMine*bj],1, &Z[localRowIndex(j+bj)],nRowsMine);
			}
			//Solve kiBlockSize x jBlockSize complex equations to update Z:
			//--- prepare RHS, accounting for blocks split across processes if any
			
		}
		j = jStop;
	}
	//--- set the diagonal blocks of Z:
	for(int iColMine=0; iColMine<nColsMine; iColMine++)
	{	int iRowMine = localRowIndex(iColsMine[iColMine]);
		if(iRowMine >= 0)
			Z[iRowMine+iColMine*nRowsMine] = ZdiagMine[iColMine];
	}
	watchLeft2.stop();
	logPrintf("RMS Z-Zref: %le\n", matrixErr(Z,Zref));
	Z = Zref;
	
	logPrintf("done.\nRotating left eigenvectors to original basis ... "); logFlush();
	//--- multiply by Q
	matMult(1., Q,false, Z,false, 0.,VL);
	//--- account for scaleFactors if necessary:
	if(scaleFactors)
	{	//Collect scale factors relevant to my rows:
		Buffer scaleMineInv; scaleMineInv.reserve(nRowsMine);
		for(int iRow: iRowsMine)
			scaleMineInv.push_back(1./scaleFactors->at(iRow));
		//Apply scale factors:
		double* VLdata = VL.data();
		for(int iColMine=0; iColMine<nColsMine; iColMine++)
			for(int iRowMine=0; iRowMine<nRowsMine; iRowMine++)
				*(VLdata++) *= scaleMineInv[iRowMine];
	}
	logPrintf("done.\n");
	watchLeft.stop();
	
	//Right eigenvector calculation:
	watchRight.start();
	logPrintf("Computing right eigenvectors ... "); logFlush();
	//--- create transpose(VL) as an LHS matrix for inversion:
	Buffer VLT(nDataMine);
	pdgeadd_("T", &N, &N, &oneD, VL.data(),&one,&one,desc, &zeroD, VLT.data(),&one,&one,desc);
	//--- set VR to identity:
	VR.resize(nDataMine);
	double* VRdata = VR.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
			*(VRdata++) = (iRow==iCol) ? 1. : 0.;
	//--- update VR = inv(transpose(VL))
	info = 0;
	std::vector<int> pivot(N);
	pdgesv_(&N, &N, VLT.data(), &one, &one, desc, pivot.data(), VR.data(), &one, &one, desc, &info);
	if(info < 0)
	{	int errCode = -info;
		if(errCode < 100) die("Error in argument# %d to pdgesv.\n", errCode)
		else die("Error in entry %d of argument# %d to pdgesv.\n", errCode%100, errCode/100)
	}
	if(info > 0) die("Matrix singular at column# %d in pdgesv.\n", info);
	logPrintf("done.\n");
	watchRight.stop();
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
