/*-------------------------------------------------------------------
Copyright 2020 Ravishankar Sundararaman

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------*/

#ifndef FEYNWANN_BLOCKCYCLICMATRIX_H
#define FEYNWANN_BLOCKCYCLICMATRIX_H
#ifdef SCALAPACK_ENABLED

#include <core/MPIUtil.h>

//Wrapper to and extension of ScaLAPACK diagonalization routines
class BlockCyclicMatrix
{
public:
	typedef std::vector<double> Buffer; //!< used for local matrices and temporary working buffers for ScaLAPACK
	
	int N; //!< matrix dimension
	int blockSize; //!< block size for block-cyclic distribution
	MPIUtil* mpiUtil;
	int blacsContext;
	int desc[9]; //!< BLACS description of matrix distribution
	int nProcsRow, nProcsCol; //!< BLACS process grid dimensions
	int iProcRow, iProcCol; //!< Current process index in BLACS process grid
	int nRowsMine, nColsMine; //!< Number of rows and columns on current process
	size_t nDataMine; //!< Total number of local matrix entries
	std::vector<int> iRowsMine, iColsMine; //!< Indices of rows and columns that belng to current process

	BlockCyclicMatrix(int N, int blockSize, MPIUtil* mpiUtil); //!< Set up for diagnalization of NxN matrices parallelized over mpiUtil
	
	//---- Diagonalization and helper routines ----
	
	//! Balance a matrix A by row/column scaling and return scale factors
	Buffer balance(Buffer& A) const;
	
	//! Hessenberg reduction of a matrix H in place, and return rotations Q (Hin = Q Hout Q^T)
	Buffer hessenberg(Buffer& H) const;
	
	//! Schur decomposition of a Hessenberg matrix H in place, and return eigenvalues
	//! At exit, H is replaced with a quasi-upper triangular matrix T
	//! and the rotations Q are updated such that A = Q T Q^T,
	//! where A = Q H Q^T is the original matrix that was converted to Hessenberg form.
	std::vector<complex> schur(Buffer& H, Buffer& Q) const;
	
	//! Compute left and right eigenvectors given Shur decomposition of a non-symmetric matrix (equivalent to LAPACK dtrevc)
	//! Input: upper quasi-triangular matrix T and orthogonal matrix Q, such that matrix A = Q T Q^T
	//! Output: left and right eigenvectors of A in VL and VR
	//! Optionally correct the eigenvectors for scale factors used to balance A is scaleFactors is non-null
	void getEvecs(const Buffer& T, const Buffer& Q, Buffer& VL, Buffer& VR, Buffer* scaleFactors=0) const;
	
	//! C = beta C + alpha op(A) * op(B), where op = identity or transpose depending on transA and transB
	void matMult(double alpha, const Buffer& A, bool transA, const Buffer& B, bool transB, double beta, Buffer& C) const;

	//---- I/O and debugging ----
	Buffer readMatrix(string fname) const; //!< Read dense matrix from file
	double matrixErr(const Buffer& A, const Buffer& B) const; //!< Calculate error between two distributed matrices
	void printMatrix(const Buffer& mat, const char* name="") const; //!< Synchronized print of all pieces of a distributed matrix
	
	//---- Indexing utilties ----
	
	//!Get index into local storage given global indices and dimensions
	//! Returns -1 if corresponding value does not belong to current process
	inline int localIndex(int iRow, int iCol) const;

};

//Get index into local storage given global indices and dimensions
//Returns -1 if corresponding value does not belong to current process
inline int BlockCyclicMatrix::localIndex(int iRow, int iCol) const
{	//Identify row and column indices:
	#define InitIndices(dim) \
		int iBlock##dim##Global = i##dim / blockSize; \
		if(iBlock##dim##Global % nProcs##dim != iProc##dim) return -1; \
		int iBlock##dim = iBlock##dim##Global / nProcs##dim; /*local block index*/ \
		int iElem##dim = i##dim % blockSize; /*index within block*/ \
		int i##dim##Mine = iBlock##dim * blockSize + iElem##dim;
	InitIndices(Row)
	InitIndices(Col)
	#undef InitIndices
	//Compute flattened local index:
	return iColMine*nRowsMine + iRowMine;
}


#endif //SCALAPACK_ENABLED
#endif // FEYNWANN_BLOCKCYCLICMATRIX_H
