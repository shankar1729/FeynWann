#ifdef SCALAPACK_ENABLED
#include "BlockCyclicMatrix.h"
#include <core/Util.h>
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

//Calculate error between distirbuted matrices
double BlockCyclicMatrix::matrixErr(const Buffer& A, const Buffer& B) const
{	double errSq = 0.;
	for(size_t i=0; i<nDataMine; i++)
		errSq += std::pow(A[i]-B[i], 2);
	mpiUtil->allReduce(errSq, MPIUtil::ReduceSum);
	return sqrt(errSq);
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


void BlockCyclicMatrix::balance(Buffer& A, int& iLo, int& iHi, Buffer& scaleFactors, bool shouldPermute, bool shouldScale) const
{	static StopWatch watch("BlockCyclicMatrix::balance"); watch.start();
	assert(A.size()==nDataMine); 
	iLo = 1; iHi = N;
	scaleFactors.assign(N, 1.);
	logPrintf("Balancing matrix ... "); logFlush();
	char job = shouldPermute ? (shouldScale ? 'B' : 'P') : (shouldScale ? 'S' : 'N');
	int info = 0;
	pdgebal_(&job, &N, A.data(), desc, &iLo, &iHi, scaleFactors.data(), &info);
	if(info < 0) die("Error in argument# %d to pdgebal.\n", -info);
	logPrintf("done.\n");
	logPrintf("Scale factors:"); for(double& s: scaleFactors) logPrintf(" %lg", s); logPrintf("\n");
	logPrintf("iLo: %d  iHi: %d\n", iLo, iHi);
	watch.stop();
}

void BlockCyclicMatrix::hessenberg(Buffer& H, int iLo, int iHi, Buffer& Q) const
{	static StopWatch watch("BlockCyclicMatrix::hessenberg"); watch.start();
	assert(H.size()==nDataMine); //Q is resized as needed
	
	//Hessenberg reduction by Householder transformations:
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
	Q.resize(nDataMine);
	double* Qptr = Q.data();
	for(int iCol: iColsMine)
		for(int iRow: iRowsMine)
			*(Qptr++) = (iRow==iCol ? 1. : 0.);
	work[0] = 0; lwork = -1; //for workspace query
	logPrintf("Extracting rotations ... "); logFlush();
	for(int pass=0; pass<2; pass++) //first pass is workspace query, next pass is actual calculation
	{	char side = 'L'; //irrelevant since we are multiplying by identity
		char trans = 'N'; //construct Q
		pdormhr_(&side, &trans, &N, &N, &iLo, &iHi, H.data(), &one, &one, desc,
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
}


//Schur decomposition and eigenvalues:
void BlockCyclicMatrix::schur(Buffer& H, int iLo, int iHi, Buffer& Q, std::vector<complex>& evals) const
{	static StopWatch watch("BlockCyclicMatrix::schur"); watch.start();
	assert(H.size()==nDataMine);
	assert(Q.size()==nDataMine);
	char job = 'T'; //Eigenvalues and Schur form
	char compz = 'V'; //Schur vectors transformed using Q provided at input
	Buffer wr(N), wi(N); //real and imaginary parts of eigenvalues
	Buffer work(1); int lwork = -1, info = 0; //for workspace query
	std::vector<int> iwork(1); int liwork = -1; //for workspace query
	logPrintf("Schur decomposition ... "); logFlush();
	for(int pass=0; pass<2; pass++) //first pass is workspace query, next pass is actual calculation
	{	pdhseqr_(&job, &compz, &N, &iLo, &iHi, H.data(), desc, wr.data(), wi.data(),
			Q.data(), desc, work.data(), &lwork, iwork.data(), &liwork, &info);
		if(info < 0)
		{	int errCode = -info;
			if(errCode < 100) die("Error in argument# %d to pdhseqr.\n", errCode)
			else die("Error in entry %d of argument# %d to pdhseqr.\n", errCode%100, errCode/100)
		}
		if(info > 0) die("Up to %d eigenvalues failed to converge.\n", info);
		if(pass) break; //done
		//After first-pass, use results of work-space query to allocate:
		lwork = int(work.data()[0]); work.resize(lwork);
		liwork = int(iwork.data()[0]); iwork.resize(liwork);
	}
	logPrintf("done.\n");
	watch.stop();
	//Collect eigenvalues into complex array:
	evals.resize(N);
	for(int i=0; i<N; i++)
		evals[i] = complex(wr[i], wi[i]);
}

void BlockCyclicMatrix::getEvecs(const Buffer& T, const Buffer& Q, Buffer& VL, Buffer& VR) const
{	static StopWatch watch("BlockCyclicMatrix::eigenvectors"); watch.start();
	
	if(mpiUtil->nProcesses() > 1) die("Parallelization not yet implemented.\n");
	assert(T.size()==nDataMine);
	assert(Q.size()==nDataMine);
	VL.resize(nDataMine);
	VR.resize(nDataMine);
	
	//Underflow/overflow and precision constants:
	double uFlow = dlamch_("Safe minimum");
	double oFlow = 1./uFlow;
	dlabad_(&uFlow, &oFlow);
	const double prec = dlamch_("Precision");
	const double sNum = uFlow*(N/prec);
	const double bNum = (1.-prec)/sNum;
	
	//Column 1-norms of strict upper-triangular part to control overflow below:
	Buffer tNorm(N, 0.);
	for(int j=1; j<N; j++)
		for(int i=0; i<j; i++)
			tNorm[j] += fabs(T[i+j*N]);
	
	//Temporaries for blas calls:
	int notTrans=0, isTrans=1, two=2, info=0;
	double oneD = 1.;
	double x[4], xNorm, scale; //1x1 or 2x2 matrix used in dlanl2; its norm and scale factor
	Buffer Z(N*N); //eigenvectors of T (multiplied by Q at the end)
	
	//Right eigenvector calculation:
	for(int ki=N-1; ki>=0; ki--)
	{	bool complexPair = (ki>0) and (T[ki+(ki-1)*N]!=0.);
		int kiBlockSize = complexPair ? 2  : 1; //current block size in ki
		int kiStart = ki + 1 - kiBlockSize; //start of current block in ki
		double* rhs = &Z[kiStart*N]; //initialized to zero above
		//Get the eigenvalue:
		double wr = T[ki+ki*N];
		double wi = 0., tU = 0., tL = 0.;
		if(complexPair)
		{	tU = T[kiStart+ki*N];
			tL = T[ki+kiStart*N];
			wi = sqrt(fabs(tU)) * sqrt(fabs(tL)); //written like this to avoid over/under-flow
		}
		double sMin = std::max(prec*(fabs(wr)+fabs(wi)), sNum); //small number threshold for this eigenvector (pair)
		//Construct RHS:
		if(complexPair)
		{	if(fabs(tU) > fabs(tL))
			{	rhs[kiStart] = 1.;
				rhs[ki+N] = wi/tU;
			}
			else
			{	rhs[kiStart] = -wi/tL;
				rhs[ki+N] = 1.;
			}
		}
		else rhs[ki] = 1.;
		for(int k=0; k<kiStart; k++)
			for(int bk=0; bk<kiBlockSize; bk++)
				rhs[k+bk*N] = -rhs[kiStart+bk+bk*N] * T[k+(kiStart+bk)*N];
		//Solve upper quasi-triangular system (T[:kiStart,:kiStart] - (wr+i*wi))*x = scale*rhs
		for(int j=kiStart-1; j>=0.; j--)
		{	int jBlockSize = ((j>0) and (T[j+(j-1)*N]!=0.)) ? 2 : 1; //current block size in j
			int jStart = j+1-jBlockSize; //start of current block in j
			dlaln2_(&notTrans, &jBlockSize, &kiBlockSize, &sMin, &oneD,
				&T[jStart+jStart*N], &N, &oneD, &oneD,
				rhs+jStart, &N, &wr, &wi, x, &two,
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
					cblas_dscal(ki+1, scale, rhs+bk*N,1); //scale when needed
			//Update the right hand side
			for(int bj=0; bj<jBlockSize; bj++)
				for(int bk=0; bk<kiBlockSize; bk++)
				{	rhs[jStart+bj + bk*N] = x[bj+2*bk];
					cblas_daxpy(jStart, -x[bj+2*bk], &T[(jStart+bj)*N],1, rhs+bk*N,1);
				}
			j = jStart;
		}
		//Scale max entry to 1:
		double rhsMax = rhs[cblas_idamax(kiBlockSize*N, rhs,1)];
		cblas_dscal(kiBlockSize*N, 1./fabs(rhsMax), rhs,1);
		ki = kiStart;
	}
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N,
		1., Q.data(),N, Z.data(),N, 0., VR.data(),N);
	
	//Left eigenvector calculation:
	Z.assign(N*N, 0.);
	for(int ki=0; ki<N; ki++)
	{	bool complexPair = (ki+1<N) and (T[(ki+1)+ki*N]!=0.);
		int kiBlockSize = complexPair ? 2  : 1; //current block size in ki
		int kiStop = ki + kiBlockSize-1; //end of current block in ki
		double* rhs = &Z[ki*N]; //initialized to zero above
		//Get the eigenvalue:
		double wr = T[ki+ki*N];
		double mwi = 0., tU = 0., tL = 0.;
		if(complexPair)
		{	tU = T[ki+kiStop*N];
			tL = T[kiStop+ki*N];
			mwi = -sqrt(fabs(tU)) * sqrt(fabs(tL)); //written like this to avoid over/under-flow
		}
		double sMin = std::max(prec*(fabs(wr)+fabs(mwi)), sNum); //small number threshold for this eigenvector (pair)
		//Construct RHS:
		if(complexPair)
		{	if(fabs(tU) > fabs(tL))
			{	rhs[ki] = -mwi/tU;
				rhs[kiStop+N] = 1.;
			}
			else
			{	rhs[ki] = 1.;
				rhs[kiStop+N] = mwi/tL;
			}
		}
		else rhs[ki] = 1.;
		for(int k=kiStop+1; k<N; k++)
			for(int bk=0; bk<kiBlockSize; bk++)
				rhs[k+bk*N] = -rhs[ki+bk+bk*N] * T[(ki+bk)+k*N];
		//Solve quasi-triangular system (T(kiStop+1:,kiStop+1:) - (wr-i*wi))*x = rhs
		double vCrit = bNum, vMax = 1.; //for scaling
		for(int j=kiStop+1; j<N; j++)
		{	int jBlockSize = ((j+1<N) and (T[(j+1)+j*N]!=0.)) ? 2 : 1; //current block size in j
			int jStop = j+jBlockSize-1; //end of current block in j
			//Scale to avoid overflow when forming RHS elements if needed:
			if(std::max(tNorm[j],tNorm[jStop]) > vCrit)
			{	double scaleFac = 1./vMax;
				for(int bk=0; bk<kiBlockSize; bk++)
					cblas_dscal(N-ki, scaleFac, rhs+ki+bk*N,1);
				vMax = 1.;
				vCrit = bNum;
			}
			//Form RHS elements:
			for(int bk=0; bk<kiBlockSize; bk++)
				for(int bj=0; bj<jBlockSize; bj++)
					rhs[j+bj+bk*N] -= cblas_ddot(j-kiStop-1, &T[(kiStop+1)+(j+bj)*N],1, rhs+(kiStop+1)+bk*N,1);
			//Solve kiBlockSize x jBlockSize complex equation to get x:
			dlaln2_((jBlockSize==2 ? &isTrans : &notTrans),
				&jBlockSize, &kiBlockSize, &sMin, &oneD,
				&T[j+j*N], &N, &oneD, &oneD, 
				rhs+j, &N, &wr, &mwi, x, &two,
				&scale, &xNorm, &info);
			//Scale if necessary:
			if(scale != 1.)
			{	for(int bk=0; bk<kiBlockSize; bk++)
					cblas_dscal(N-ki, scale, rhs+ki+bk*N,1);
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
		double rhsMax = rhs[cblas_idamax(kiBlockSize*N, rhs,1)];
		cblas_dscal(kiBlockSize*N, 1./fabs(rhsMax), rhs,1);
		ki = kiStop;
	}
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N,
		1., Q.data(),N, Z.data(),N, 0., VL.data(),N);
	watch.stop();
}

void BlockCyclicMatrix::matMult(double alpha, const Buffer& A, bool transA, const Buffer& B, bool transB, double beta, Buffer& C)
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
