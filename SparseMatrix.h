/*-------------------------------------------------------------------
Copyright 2019 Ravishankar Sundararaman

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

#ifndef FEYNWANN_SPARSEMATRIX_H
#define FEYNWANN_SPARSEMATRIX_H

#include <core/matrix.h>

//----- Rudimentary triplet format square sparse matrix with restricted but fast, inline operations ----
struct SparseEntry
{	int i, j;
	complex val;
	//Accessors supporting dagger operation:
	template<bool dagger=false> int I() const { return dagger ? j : i; }
	template<bool dagger=false> int J() const { return dagger ? i : j; }
	template<bool dagger=false> complex Val() const { return dagger ? val.conj() : val; }
};
struct SparseMatrix : public std::vector<SparseEntry>
{
private:
	int nr, nc;
public:
    inline SparseMatrix(int nRows=0, int nCols=0, int nNZexpected=0) { init(nRows, nCols, nNZexpected); }
	inline void init(int nRows, int nCols, int nNZexpected=0)
	{	nr = nRows; nc = nCols;
		if(nNZexpected) reserve(nNZexpected);
	}
	//Accessors supporting dagger operation:
	template<bool dagger=false> int nRows() const { return dagger ? nc : nr; }
	template<bool dagger=false> int nCols() const { return dagger ? nr : nc; }
};

//Multiply S1 * M * S2 for sparse matrices S1 and S2 (each optionally daggered), and dense matrix M
template<bool dag1, bool dag2> SparseMatrix SMS(const SparseMatrix& S1, const matrix& M, const SparseMatrix& S2)
{	assert(S1.nCols<dag1>() == M.nRows());
	assert(M.nCols() == S2.nRows<dag2>());
	SparseMatrix result(S1.nRows<dag1>(), S2.nCols<dag2>(), S1.size()*S2.size());
	const complex* m = M.data();
	for(const SparseEntry& s1: S1)
		for(const SparseEntry& s2: S2)
		{	SparseEntry sr;
			sr.i = s1.I<dag1>();
			sr.j = s2.J<dag2>();
			sr.val = s1.Val<dag1>() * m[M.index(s1.J<dag1>(),s2.I<dag2>())] * s2.Val<dag2>();
			result.push_back(sr);
		}
	return result;
}

//Extract diagonal part of product of sparse matrices:
inline diagMatrix diagSS(const SparseMatrix& S1, const SparseMatrix& S2)
{	assert(S1.nCols() == S2.nRows()); //for S1 * S2 to be meaningful
	assert(S1.nRows() == S2.nCols()); //for result to be square
	diagMatrix result(S1.nRows());
	for(const SparseEntry& s1: S1)
		for(const SparseEntry& s2: S2)
			if(s1.i==s2.j && s1.j==s2.i)
				result[s1.i] += (s1.val * s2.val).real();
	return result;
}

//Accumulate product of sparse matrix with dense matrix on left to dense matrix:
inline void axpyProd(double alpha, const matrix& M, const SparseMatrix& S, matrix& R)
{	assert(M.nCols() == S.nRows());
	int nRows = M.nRows();
	assert(R.nRows() == nRows);
	assert(R.nCols() == S.nCols());
	complex* r = R.data();
	const complex* m = M.data();
	for(const SparseEntry& s: S)
	{	complex prefac = alpha * s.val;
		complex* rCur = r + nRows*s.j;
		const complex* mCur = m + nRows*s.i;
		for(int k=0; k<nRows; k++)
			*(rCur++) += *(mCur++) * prefac;
	}
}

//Multiply sparse matrix with dense matrix on left:
inline matrix operator*(const matrix& M, const SparseMatrix& S)
{	matrix R = zeroes(M.nRows(), S.nCols());
	axpyProd(1., M, S, R);
	return R;
}

//Accumulate product of sparse matrix with dense matrix on left to dense matrix:
inline void axpyProd(double alpha, const SparseMatrix& S, const matrix& M, matrix& R)
{	assert(S.nCols() == M.nRows());
	int nCols = M.nCols();
	assert(R.nRows() == S.nRows());
	assert(R.nCols() == nCols);
	complex* r = R.data();
	const complex* m = M.data();
	for(const SparseEntry& s: S)
	{	complex prefac = alpha * s.val;
		complex* rCur = r + s.i;
		const complex* mCur = m + s.j;
		for(int k=0; k<nCols; k++)
		{	(*rCur) += prefac * (*mCur);
			rCur += R.nRows();
			mCur += M.nRows();
		}
	}
}

//Multiply sparse matrix with dense matrix on right:
inline matrix operator*(const SparseMatrix& S, const matrix& M)
{	matrix R = zeroes(S.nRows(), M.nCols());
	axpyProd(1., S, M, R);
	return R;
}

#endif //FEYNWANN_SPARSEMATRIX_H
