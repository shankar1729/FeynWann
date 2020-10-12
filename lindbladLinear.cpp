/*-------------------------------------------------------------------
Copyright 2019 Ravishankar Sundararaman, Adela Habib

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

#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <commands/command.h>
#include "FeynWann.h"
#include "Histogram.h"
#include "InputMap.h"
#include "LindbladFile.h"
#include "Integrator.h"
#include "BlockCyclicMatrix.h"
#include <core/Units.h>
#include <slepceps.h>

//Slightly more graceful wrapper to CHKERRQ() macro from Petsc:
PetscInt iErr = 0;
#define CHECKERR(codeLine) \
	iErr = codeLine; \
	CHKERRQ(iErr);

inline matrix dot(const matrix* P, vector3<complex> pol)
{	return pol[0]*P[0] + pol[1]*P[1] + pol[2]*P[2];
}

//Construct identity - X:
inline matrix bar(const matrix& X)
{	matrix Xbar(X);
	complex* XbarData = Xbar.data();
	for(int j=0; j<X.nCols(); j++)
		for(int i=0; i<X.nRows(); i++)
		{	(*XbarData) = (i==j ? 1. : 0.) - (*XbarData);
			XbarData++;
		}
	return Xbar;
}
inline diagMatrix bar(const diagMatrix& X)
{	diagMatrix Xbar(X);
	for(double& x: Xbar) x = 1. - x;
	return Xbar;
}

//Lindblad initialization, time evolution and measurement operators using FeynWann callback
struct LindbladLinear : public Integrator<DM1>
{	
	int stepID; //current time and reporting step number
	
	const double dmu, T, invT; //!< Fermi level position relative to neutral value / VBM, and temperature
	const bool spectrumMode; //!< if yes (diagonalization), evolveMat includes coherent part
	const bool sparseDiag; //!< if yes (sparse diagonalization), use SLEPc (preconditioner is also initialized), else use ScaLAPACK
	const int blockSize; //!< block size in ScaLAPACK matrix distribution
	const double pumpOmega, pumpA0, pumpTau; const vector3<complex> pumpPol; //!< pump parameters
	const bool pumpBfield; const vector3<> pumpB; //pump parameters for Bfield mode
	const double omegaMin, domega, omegaMax; const int nomega; //!< probe frequency grid
	const double tau; const std::vector<vector3<complex>> pol; //!< probe parameters
	const double dE; //!< energy resolution for distribution functions
	
	const bool ePhEnabled; //!< whether e-ph coupling is enabled
	const bool verbose; //!< whether to print more detailed stats during evolution
	const string checkpointFile; //!< file name to save checkpoint data to
	bool spinorial; //!< whether spin is available
	int spinWeight; //!< weight of spin in BZ integration
	matrix3<> R; double Omega; //!< lattice vectors and unit cell volume
	
	size_t nk, nkTot; //!< number of selected k-points overall and original total k-points effectively used in BZ sampling
	size_t ikStart, ikStop, nkMine; //!< range and number of selected k-points on this process
	TaskDivision kDivision;
	inline bool isMine(size_t ik) const { return kDivision.isMine(ik); } //!< check if k-point index is local
	inline int whose(size_t ik) const { return kDivision.whose(ik); } //!< find out which process (in mpiWorld) this k-point belongs to

	struct State : LindbladFile::Kpoint
	{	int innerStop; //end of active inner window range (relative to outer window)
		diagMatrix rho0; //equilibrium / initial density matrix (diagonal)
		matrix pumpPD; //P matrix elements at pump polarization x energy conservation delta (D), but without A0 and time factor
	};
	std::vector<State> state; //!< all information read from lindbladInit output (e and e-ph properties) + extra local variables above
	std::vector<int> nInnerAll; //!< nInner for all k-points on all processes
	double Emin, Emax; //!< energy range of active space across all k (for spin and number density output)
	
	LindbladLinear(double dmu, double T, bool spectrumMode, bool sparseDiag, int blockSize,
		double pumpOmega, double pumpA0, double pumpTau, vector3<complex> pumpPol, bool pumpBfield, vector3<> pumpB,
		double omegaMin, double omegaMax, double domega, double tau, std::vector<vector3<complex>> pol, double dE,
		bool ePhEnabled, bool verbose, string checkpointFile)
	: stepID(0),
		dmu(dmu), T(T), invT(1./T), spectrumMode(spectrumMode), sparseDiag(sparseDiag), blockSize(blockSize),
		pumpOmega(pumpOmega), pumpA0(pumpA0), pumpTau(pumpTau), pumpPol(pumpPol), pumpBfield(pumpBfield), pumpB(pumpB),
		omegaMin(omegaMin), domega(domega), omegaMax(omegaMax), nomega(1+int(round((omegaMax-omegaMin)/domega))),
		tau(tau), pol(pol), dE(dE), ePhEnabled(ePhEnabled), verbose(verbose), checkpointFile(checkpointFile),
		Emin(+DBL_MAX), Emax(-DBL_MAX)
	{
	}
	
	//---- Flat density matrix storage and access functions ----
	DM1 drho; //!< flat array of density matrix changes of all k stored on this process
	std::vector<size_t> nInnerPrev; //cumulative nInner for each k, which is the offset into the Eall array for each k
	std::vector<size_t> nRhoPrev; //cumulative nInner^2 for each k, which is the offset into the global rho structure for each k
	std::vector<size_t> rhoOffset; //!< array of offsets into process's rho for each k (essentially nRhoPrev - nRhoPrev[ikStart])
	std::vector<size_t> rhoSize; //!< total size of rho on each process
	size_t rhoOffsetGlobal; //!< offset of current process rho data in the overall data
	size_t rhoSizeTot; //!< total size of rho
	
	//Get an NxN complex Hermitian matrix from a real array of length N^2
	inline matrix getRho(const double* rhoData, int N) const
	{	matrix out(N, N); complex* outData = out.data();
		for(int i=0; i<N; i++)
			for(int j=0; j<=i; j++)
			{	int i1 = i+N*j, i2 = j+N*i;
				if(i==j)
					outData[i1] = rhoData[i1];
				else
					outData[i2] = (outData[i1] = complex(rhoData[i1],rhoData[i2])).conj();
			}
		return out;
	}
	
	//Accumulate a diagonal matrix to a real array of length N^2
	inline void accumRho(const diagMatrix& in, double* rhoData) const
	{	const int N = in.nRows();
		for(int i=0; i<N; i++)
		{	*(rhoData) += in[i];
			rhoData += (N+1); //advance to next diagonal entry
		}
	}
	
	//Accumulate an NxN matrix and its Hermitian conjugate to a real array of length N^2
	inline void accumRhoHC(const matrix& in, double* rhoData) const
	{	const complex* inData = in.data();
		const int N = in.nRows();
		for(int i=0; i<N; i++)
			for(int j=0; j<=i; j++)
			{	int i1 = i+N*j, i2 = j+N*i;
				if(i==j)
					rhoData[i1] += 2*inData[i1].real();
				else
				{	complex hcSum = inData[i1] + inData[i2].conj();
					rhoData[i1] += hcSum.real();
					rhoData[i2] += hcSum.imag();
				}
			}
	}
	
	//--------- Time evolution sparse matrix and SLEPc conversion -----------
	
	struct Triplet { int i, j; double val; bool local; }; //entry in triplet format matrix (along with tage for process locality)for initial construction
	Mat evolveMat; //Time evolution operator
	//Mat precondMat; //Preconditioning matrix
	Vec vRho, vRhoDot; //!< temporary copies of drho and rdhoDot data in Petsc format
	
	//Clean up Petsc quantities
	PetscErrorCode cleanup()
	{	if(spectrumMode and (not sparseDiag))
		{	//TODO: dense cleanup
		}
		else
		{	CHECKERR(MatDestroy(&evolveMat));
			CHECKERR(VecDestroy(&vRho));
			CHECKERR(VecDestroy(&vRhoDot));
			//if(spectrumMode) CHECKERR(MatDestroy(&precondMat));
		}
		return 0;
	}
	
	//Initialize a distributed square matrix M from (distributed) triplet format in entries:
	PetscErrorCode matInit(Mat& M, const std::vector<Triplet> entries)
	{	int N = rhoSizeTot;
		int Nmine = rhoSize[mpiWorld->iProcess()];
		//Determine non-zero sizes:
		std::vector<int> nnzD(N), nnzO(N); //number of process-diagonal and process off-diagonal entries by row
		for(const Triplet& entry: entries)
			(entry.local ? nnzD : nnzO)[entry.i]++;
		MPI_Allreduce(MPI_IN_PLACE, nnzD.data(), N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, nnzO.data(), N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		for(size_t i=rhoOffsetGlobal; i<rhoOffsetGlobal+Nmine; i++)
		{	nnzD[i] = std::min(nnzD[i], Nmine);
			nnzO[i] = std::min(nnzO[i], N - Nmine);
		}
		CHECKERR(MatCreateAIJ(PETSC_COMM_WORLD, Nmine, Nmine, N, N,
			0, nnzD.data()+rhoOffsetGlobal, 0, nnzO.data()+rhoOffsetGlobal, &M));
		CHECKERR(MatSetOption(M, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
		for(const Triplet& entry: entries)
			CHECKERR(MatSetValue(M, entry.i, entry.j, entry.val, ADD_VALUES));
		CHECKERR(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
		CHECKERR(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
		return 0;
	}
	
	//Initialize a block-diagonal-inverse preconditioner:
	PetscErrorCode blockDiagonalInvert(const Mat& M, Mat& K)
	{	//Determine non-zero sizes and allocate matrix:
		int N = rhoSizeTot;
		int Nmine = rhoSize[mpiWorld->iProcess()];
		std::vector<int> nnzD(Nmine, 0), nnzO(Nmine, 0);
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	size_t nRhoCur = nRhoPrev[ik+1]-nRhoPrev[ik];
			for(size_t iRho=nRhoPrev[ik]; iRho<nRhoPrev[ik+1]; iRho++)
				nnzD[iRho-rhoOffsetGlobal] = nRhoCur; //all diagonal; nnzO = 0
		}
		CHECKERR(MatCreateAIJ(PETSC_COMM_WORLD, Nmine, Nmine, N, N, 0, nnzD.data(), 0, nnzO.data(), &K));
		//Create index sets to extract diagonal blocks:
		std::vector<IS> isArr(nkMine);
		for(size_t ik=ikStart; ik<ikStop; ik++)
			CHECKERR(ISCreateStride(PETSC_COMM_SELF, nRhoPrev[ik+1]-nRhoPrev[ik], nRhoPrev[ik], 1, &isArr[ik-ikStart]));
		//Get diagonal blocks:
		Mat* blocks;
		CHECKERR(MatCreateSubMatrices(M, nkMine, isArr.data(), isArr.data(), MAT_INITIAL_MATRIX, &blocks));
		for(IS& is: isArr) CHECKERR(ISDestroy(&is));
		//Invert each diagonal block:
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	Mat& block = blocks[ik-ikStart];
			size_t nRhoCur = nRhoPrev[ik+1]-nRhoPrev[ik];
			//Get data as a dense block:
			matrix blockInv = zeroes(nRhoCur, nRhoCur);
			CHECKERR(MatConvert(block, MATDENSE, MAT_INPLACE_MATRIX, &block));
			const double* pBlock; CHECKERR(MatDenseGetArrayRead(block, &pBlock));
			eblas_daxpy(blockInv.nData(), 1., pBlock,1, (double*)blockInv.data(),2); //copy to real parts in complex matrix
			CHECKERR(MatDenseRestoreArrayRead(block, &pBlock));
			//Invert block and set it in K:
			blockInv = inv(blockInv);
			std::vector<double> blockInvRe(nRhoCur*nRhoCur, 0.); //real part of the inverse
			eblas_daxpy(blockInv.nData(), 1., (const double*)blockInv.data(),2, blockInvRe.data(),1); //extract real part
			std::vector<int> indices(nRhoCur);
			for(size_t iRhoCur=0; iRhoCur<nRhoCur; iRhoCur++)
				indices[iRhoCur] = nRhoPrev[ik] + iRhoCur; //global index for each row/column
			CHECKERR(MatSetValues(K, nRhoCur,indices.data(), nRhoCur,indices.data(), blockInvRe.data(), INSERT_VALUES));
		}
		CHECKERR(MatDestroySubMatrices(nkMine, &blocks));
		//Assemble matrix:
		CHECKERR(MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY));
		CHECKERR(MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY));
		return 0;
	}
	
	//--------- Blacs / ScaLAPACK interface for dense diagonalization -----------
	#ifdef SCALAPACK_ENABLED
	std::shared_ptr<BlockCyclicMatrix> bcm;
	#endif
	
	//--------- Initialize -------------
	PetscErrorCode initialize(string inFile)
	{
		//Read header and check parameters:
		MPIUtil::File fp;
		mpiWorld->fopenRead(fp, inFile.c_str());
		LindbladFile::Header h; h.read(fp, mpiWorld);
		if(dmu<h.dmuMin or dmu>h.dmuMax)
			die("dmu = %lg eV is out of range [ %lg , %lg ] eV specified in lindbladInit.\n", dmu/eV, h.dmuMin/eV, h.dmuMax/eV);
		if(T > h.Tmax)
			die("T = %lg K is larger than Tmax = %lg K specified in lindbladInit.\n", T/Kelvin, h.Tmax/Kelvin);
		if((not pumpBfield) and (pumpOmega > h.pumpOmegaMax))
			die("pumpOmega = %lg eV is larger than pumpOmegaMax = %lg eV specified in lindbladInit.\n", pumpOmega/eV, h.pumpOmegaMax/eV);
		if(omegaMax > h.probeOmegaMax)
			die("omegaMax = %lg eV is larger than probeOmegaMax = %lg eV specified in lindbladInit.\n", omegaMax/eV, h.probeOmegaMax/eV);
		nk = h.nk;
		nkTot = h.nkTot;
		spinorial = h.spinorial;
		spinWeight = h.spinWeight;
		R = h.R; Omega = fabs(det(R));
		if(ePhEnabled != h.ePhEnabled)
			die("ePhEnabled = %s differs from the mode specified in lindbladInit.\n", boolMap.getString(ePhEnabled));
		if(pumpBfield and (not spinorial))
			die("Bfield pump mode requires spin matrix elements from a spinorial calculation.\n");
		
		//Read k-point offsets:
		std::vector<size_t> byteOffsets(h.nk);
		mpiWorld->freadData(byteOffsets, fp);
		
		//Divide k-points between processes:
		kDivision.init(nk, mpiWorld);
		kDivision.myRange(ikStart, ikStop);
		nkMine = ikStop-ikStart;
		state.resize(nkMine);
		nInnerAll.resize(nk);
		
		//Read k-point info and initialize states:
		
	if((not spectrumMode) or sparseDiag) { //HACK
		
		mpiWorld->fseek(fp, byteOffsets[ikStart], SEEK_SET);
		for(size_t ikMine=0; ikMine<nkMine; ikMine++)
		{	State& s = state[ikMine];
			
			//Read base info from LindbladFile:
			((LindbladFile::Kpoint&)s).read(fp, mpiWorld, h);
			nInnerAll[ikStart+ikMine] = s.nInner;
			
			//Initialize extra quantities in state:
			s.innerStop = s.innerStart + s.nInner;
			//--- Active energy range:
			Emin = std::min(Emin, s.E[s.innerStart]);
			Emax = std::max(Emax, s.E[s.innerStop-1]);
			//--- Pump matrix elements with energy conservation
			if(not pumpBfield)
			{	s.pumpPD = dot(s.P, pumpPol)(0,s.nInner, s.innerStart,s.innerStop); //restrict to inner active
				double normFac = sqrt(pumpTau/sqrt(M_PI));
				complex* PDdata = s.pumpPD.data();
				for(int b2=s.innerStart; b2<s.innerStop; b2++)
					for(int b1=s.innerStart; b1<s.innerStop; b1++)
					{	//Multiply energy conservation:
						double tauDeltaE = pumpTau*(s.E[b1] - s.E[b2] - pumpOmega);
						*(PDdata++) *= normFac * exp(-0.5*tauDeltaE*tauDeltaE);
					}
			}
			
			//Set initial occupations:
			s.rho0.resize(s.nInner);
			for(int b=0; b<s.nInner; b++)
				s.rho0[b] = fermi((s.E[b+s.innerStart]-dmu)*invT);
		}
		
	} //HACK
		
		mpiWorld->fclose(fp);
		
		//Synchronize energy range:
		mpiWorld->allReduce(Emin, MPIUtil::ReduceMin);
		mpiWorld->allReduce(Emax, MPIUtil::ReduceMax);
		logPrintf("Electron energy grid from %lg eV to %lg eV with spacing %lg eV.\n", Emin/eV, Emax/eV, dE/eV);
		
		//Make nInner for all k available on all processes:
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
			mpiWorld->bcast(nInnerAll.data()+kDivision.start(jProc),
				kDivision.stop(jProc)-kDivision.start(jProc), jProc);
		
		//Compute sizes of and offsets into flattened rho for all processes:
		rhoOffset.resize(nk);
		rhoSize.resize(mpiWorld->nProcesses());
		rhoSizeTot = 0;
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
		{	size_t jkStart = kDivision.start(jProc);
			size_t jkStop = kDivision.stop(jProc);
			size_t offset = 0; //start at 0 for each process's chunk
			for(size_t jk=jkStart; jk<jkStop; jk++)
			{	rhoOffset[jk] = offset;
				offset += nInnerAll[jk]*nInnerAll[jk];
			}
			rhoSize[jProc] = offset;
			if(jProc == mpiWorld->iProcess()) rhoOffsetGlobal = rhoSizeTot;
			rhoSizeTot += offset; //cumulative over all processes
		}
		
		//Initialize rho:
		drho.assign(rhoSize[mpiWorld->iProcess()], 0.);
		
		//Initialize sparse matrix corresponding to net time evolution (if required):
		if(ePhEnabled)
		{	//Make inner-window energies available for all processes:
			nInnerPrev.assign(nk+1, 0); //cumulative nInner for each k (offset into the Eall array)
			nRhoPrev.assign(nk+1, 0); //cumulative nInner^2 for each k (offset into global rho)
			for(size_t ik=0; ik<nk; ik++)
			{	nInnerPrev[ik+1] = nInnerPrev[ik] + nInnerAll[ik];
				nRhoPrev[ik+1] = nRhoPrev[ik] +  nInnerAll[ik]*nInnerAll[ik];
			}
			std::vector<double> Eall(nInnerPrev.back()); //inner window energies for all k
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	const State& s = state[ik-ikStart];
				const double* Ei = &(s.E[s.innerStart]);
				std::copy(Ei, Ei+s.nInner, Eall.begin()+nInnerPrev[ik]);
			}
			for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
			{	size_t iEstart = nInnerPrev[kDivision.start(jProc)];
				size_t iEstop = nInnerPrev[kDivision.stop(jProc)];
				mpiWorld->bcast(&Eall[iEstart], iEstop-iEstart, jProc);
			}
			//Collect matrix elements in triplet format:
			std::vector<Triplet> evolveEntries;
		
		if((not spectrumMode) or sparseDiag) { //HACK
		
			State* sPtr = state.data();
			logPrintf("Initialing time evolution operator ... "); logFlush();
			for(size_t ik1=ikStart; ik1<ikStop; ik1++)
			{	State& s = *(sPtr++);
				const double* E1 = &(s.E[s.innerStart]);
				const int& nInner1 = nInnerAll[ik1];
				const int N1 = nInner1*nInner1; //number of density matrix entries
				const int& nRhoPrev1 = nRhoPrev[ik1];
				const int whose1 = mpiWorld->iProcess();
				const diagMatrix& f1 = s.rho0;
				const diagMatrix f1bar = bar(f1);
				//Coherent evolution (only in spectrum mode):
				if(spectrumMode)
				{	for(int a=0; a<nInner1; a++)
						for(int b=0; b<nInner1; b++)
							if(a != b)
							{	evolveEntries.push_back(Triplet{nRhoPrev1+a+b*nInner1, nRhoPrev1+b+a*nInner1, E1[b]-E1[a]});
								//diagVals[i*n+j] = complex(0, eps(i)-eps(j));
							}
				}
				//Electron-phonon part:
				const double prefacEph = 2*M_PI/nkTot; //factor of 2 from the +h.c. contribution
				std::vector<LindbladFile::GePhEntry>::iterator g = s.GePh.begin();
				while(g != s.GePh.end())
				{	const size_t& ik2 = g->jk;
					const int& nInner2 = nInnerAll[ik2];
					const int N2 = nInner2*nInner2; //number of density matrix entries
					const int& nRhoPrev2 = nRhoPrev[ik2];
					const int whose2 = whose(ik2);
					const double* E2 = &(Eall[nInnerPrev[ik2]]);
					diagMatrix f2(nInner2); for(int b2=0; b2<nInner2; b2++) f2[b2] = fermi(invT*(E2[b2]-dmu));
					const diagMatrix f2bar = bar(f2);
					//Store results in dense complex blocks of the superoperator first:
					matrix L12 = zeroes(N1,N2); complex* L12data = L12.data();
					matrix L21 = zeroes(N2,N1); complex* L21data = L21.data();
					matrix L11 = zeroes(N1,N1); complex* L11data = L11.data();
					matrix L22 = zeroes(N2,N2); complex* L22data = L22.data();
					#define L(i,a,b, j,c,d) L##i##j##data[L##i##j.index(a+b*nInner##i, c+d*nInner##j)] //access superoperator block element
					//Loop over all connections to the same ik2:
					while((g != s.GePh.end()) and (g->jk == ik2))
					{	g->G.init(nInner1, nInner2);
						g->initA(T);
						//Loop over A- and A+
						for(int pm=0; pm<2; pm++) 
						{	const SparseMatrix& Acur = pm ? g->Ap : g->Am;
							const diagMatrix& f1cur = pm ? f1 : f1bar;
							const diagMatrix& f2cur = pm ? f2bar : f2;
							//Loop oover all pairs of non-zero entries:
							for(const SparseEntry& s1: Acur)
							{	int a = s1.i, b = s1.j; //to match derivation's notation
								for(const SparseEntry& s2: Acur)
								{	int c = s2.i, d = s2.j; //to match derivation's notation
									complex M = prefacEph * (s1.val * s2.val.conj());
									L(1,a,c, 2,b,d) += f1cur[a] * M;
									L(2,d,b, 1,c,a) += f2cur[d] * M;
									if(b == d) for(int e=0; e<nInner1; e++) L(1,e,c, 1,e,a) -= f2cur[b] * M;
									if(a == c) for(int e=0; e<nInner2; e++) L(2,e,b, 2,e,d) -= f1cur[c] * M;
								}
							}
						}
						//Move to next element:
						g++;
					}
					#undef L
					//Convert from complex to real input and real outputs (based on h.c. symmetry):
					#define CreateRandInv(i) \
						SparseMatrix R##i(N##i,N##i,2*N##i), Rinv##i(N##i,N##i,2*N##i); \
						for(int a=0; a<nInner##i; a++) \
						{	for(int b=0; b<a; b++) \
							{	int ab = a+b*nInner##i, ba = b+a*nInner##i; \
								R##i.push_back(SparseEntry{ab,ab,complex(1,0)}); R##i.push_back(SparseEntry{ab,ba,complex(0,+1)}); \
								R##i.push_back(SparseEntry{ba,ab,complex(1,0)}); R##i.push_back(SparseEntry{ba,ba,complex(0,-1)}); \
								Rinv##i.push_back(SparseEntry{ab,ab,complex(+0.5,0)}); Rinv##i.push_back(SparseEntry{ab,ba,complex(+0.5,0)}); \
								Rinv##i.push_back(SparseEntry{ba,ab,complex(0,-0.5)}); Rinv##i.push_back(SparseEntry{ba,ba,complex(0,+0.5)}); \
							} \
							int aa = a+a*nInner##i; \
							R##i.push_back(SparseEntry{aa,aa,1.}); \
							Rinv##i.push_back(SparseEntry{aa,aa,1.}); \
						}
					CreateRandInv(1)
					CreateRandInv(2)
					#undef CreateRandInv
					L12 = Rinv1 * (L12 * R2);
					L21 = Rinv2 * (L21 * R1);
					L11 = Rinv1 * (L11 * R1);
					L22 = Rinv2 * (L22 * R2);
					//Extract non-zero entries in triplet:
					#define EXTRACT_NNZ(i,j) \
					{	bool isLocal = (whose##i == whose##j); \
						const complex* data = L##i##j.data(); \
						for(int col=0; col<L##i##j.nCols(); col++) \
						{	for(int row=0; row<L##i##j.nRows(); row++) \
							{	double M = (data++)->real(); \
								if(M) \
								{	evolveEntries.push_back(Triplet{row+nRhoPrev##i, col+nRhoPrev##j, M, isLocal}); \
								} \
							} \
						} \
					}
					EXTRACT_NNZ(1,2)
					EXTRACT_NNZ(2,1)
					EXTRACT_NNZ(1,1)
					EXTRACT_NNZ(2,2)
					#undef EXTRACT_NNZ
				}
			}
			logPrintf("done.\n"); 
			
		} //HACK
			
			//Convert from triplet to appropriate format:
			if(spectrumMode and (not sparseDiag))
			{	//Convert to dense matrix for ScaLAPACK:
				#ifdef SCALAPACK_ENABLED
				const int nRows = 22; //HACK rhoSizeTot
				bcm = std::make_shared<BlockCyclicMatrix>(nRows, blockSize, mpiWorld);
				
				//############ HACK ########################
				//Read test matrix:
				BlockCyclicMatrix::Buffer H = bcm->readMatrix("testMatrix.bin");
				BlockCyclicMatrix::Buffer VLref = bcm->readMatrix("testMatrix.VL.bin");
				BlockCyclicMatrix::Buffer VRref = bcm->readMatrix("testMatrix.VR.bin");
				
				BlockCyclicMatrix::Buffer scale = bcm->balance(H); //Balance matrix
				BlockCyclicMatrix::Buffer Q = bcm->hessenberg(H); //Hessenberg reduction
				std::vector<complex> evals = bcm->schur(H, Q); //Schur decomposition and eigenvalues
				std::vector<int> evalSort = bcm->sortEvals(evals); //Sort eigenvalues
				BlockCyclicMatrix::Buffer VL, VR; bcm->getEvecs(H, Q, VL, VR, &scale, &evalSort); //Get eigenvectors
				
				//Fix normalization of eigenvectors:
				std::vector<double*> VdataArr(2);
				VdataArr[0] = VL.data();
				VdataArr[1] = VR.data();
				for(double* Vdata: VdataArr)
				{	for(int iCol=0; iCol<nRows; iCol++)
					{	double* Qcur = Vdata + iCol*nRows;
						if(evals[iCol].imag()) //complex eigenvector pair
						{	double* Qnext = Qcur + nRows;
							//Determine max entry:
							int iMaxAbs = -1; double maxAbs = 0.;
							for(int iRow=0; iRow<nRows; iRow++)
							{	double absCur = complex(Qcur[iRow], Qnext[iRow]).abs();
								if(absCur > maxAbs)
								{	maxAbs = absCur;
									iMaxAbs = iRow;
								}
							}
							//Make max abs entry = 1:
							complex scaleFac = complex(1.,0)/complex(Qcur[iMaxAbs], Qnext[iMaxAbs]);
							for(int iRow=0; iRow<nRows; iRow++)
							{	complex scaled = scaleFac * complex(Qcur[iRow], Qnext[iRow]);
								Qcur[iRow] = scaled.real();
								Qnext[iRow] = scaled.imag();
							}
							iCol++; //extra column handled
						}
						else
						{	//Determine max entry:
							int iMaxAbs = -1; double maxAbs = 0.;
							for(int iRow=0; iRow<nRows; iRow++)
							{	double absCur = fabs(Qcur[iRow]);
								if(absCur > maxAbs)
								{	maxAbs = absCur;
									iMaxAbs = iRow;
								}
							}
							//Make max abs entry = 1:
							double scaleFac = 1./Qcur[iMaxAbs];
							for(int iRow=0; iRow<nRows; iRow++)
								Qcur[iRow] *= scaleFac;
						}
					}
				}
				
				
				//Check decomposition by multiplying:
				{	BlockCyclicMatrix::Buffer VLTVR;
					bcm->matMult(1., VL,true, VR,false, 0., VLTVR);
					bcm->printMatrix(VLTVR, "VL^T * VR");
				}
				bcm->printMatrix(VL, "VL");
				bcm->printMatrix(VR, "VR");
				
				//Print eigenvalues:
				logPrintf("\nEigenvalues:\n");
				for(int i=0; i<nRows; i++)
				{	logPrintf("\t%11.8lf%+11.8lfj (original index: %d)\n", evals[i].real(), evals[i].imag(), evalSort[i]);
				}
				
				logPrintf("Eigenvector errors: %le left, %le right\n", bcm->matrixErr(VL,VLref), bcm->matrixErr(VR,VRref));
				//############ HACK ########################
				#endif
			}
			else
			{	//Convert to Petsc matrix:
				logPrintf("Converting to PETSc sparse matrix ... "); logFlush();
				matInit(evolveMat, evolveEntries);
				MatInfo info; CHECKERR(MatGetInfo(evolveMat, MAT_GLOBAL_SUM, &info));
				logPrintf("done. Net sparsity: %.0lf non-zero in %lu x %lu matrix (%.1lf%% fill)\n",
					info.nz_used, rhoSizeTot, rhoSizeTot, info.nz_used*100./(rhoSizeTot*rhoSizeTot));
				logFlush();
				CHECKERR(MatCreateVecs(evolveMat, &vRho, &vRhoDot));
				if(spectrumMode)
				{	/*
					logPrintf("Initializing block-diagonal-inverse preconditioner ... "); logFlush();
					CHECKERR(blockDiagonalInvert(evolveMat, precondMat));
					CHECKERR(MatGetInfo(precondMat, MAT_GLOBAL_SUM, &info));
					logPrintf("done. Net sparsity: %.0lf non-zero in %lu x %lu matrix (%.1lf%% fill)\n",
						info.nz_used, rhoSizeTot, rhoSizeTot, info.nz_used*100./(rhoSizeTot*rhoSizeTot));
					logFlush();
					*/
				}
			}
		}
		logPrintf("\n"); logFlush();
		return 0;
	}
	
	//Calculate change in probe response due to current drho:
	diagMatrix calcDeltaImEps(double t, const DM1& drho) const
	{	static StopWatch watch("Lindblad::calcImEps");
		size_t nImEps = pol.size() * nomega;
		if(nImEps==0) return diagMatrix(); //no probe specified
		watch.start();
		diagMatrix dimEps(nImEps);
		//Collect contributions from each k at this process:
		const State* sPtr = state.data();
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	const State& s = *(sPtr++);
			const matrix drhoCurSub = getRho(drho.data()+rhoOffset[ik], s.nInner);
			//Expand density matrix:
			matrix drhoCur = zeroes(s.nOuter, s.nOuter);
			drhoCur.set(s.innerStart,s.innerStop, s.innerStart,s.innerStop, drhoCurSub);
			//Expand probe matrix elements:
			std::vector<matrix> Ppol(pol.size(), zeroes(s.nOuter, s.nOuter));
			for(int iDir=0; iDir<3; iDir++)
			{	//Expand Cartesian component:
				const matrix& PiSub = s.P[iDir]; //nInner x nOuter
				matrix Pi = zeroes(s.nOuter, s.nOuter);
				Pi.set(s.innerStart,s.innerStop, 0,s.nOuter, PiSub);
				Pi.set(0,s.nOuter, s.innerStart,s.innerStop, dagger(PiSub));
				//Update each polarization:
				for(int iPol=0; iPol<int(pol.size()); iPol++)
					Ppol[iPol] += pol[iPol][iDir] * Pi;
			}
			//Probe response:
			for(int iomega=0; iomega<nomega; iomega++)
			{	double omega = omegaMin + iomega*domega;
				double prefac = (4*std::pow(M_PI,2)*spinWeight)/(nkTot * Omega * std::pow(std::max(omega, 1./tau), 3));
				//Energy conservation and phase factors for all pair of bands at this frequency:
				std::vector<complex> delta(s.nOuter*s.nOuter);
				complex* deltaData = delta.data();
				double normFac = sqrt(tau/sqrt(M_PI));
				for(int b2=0; b2<s.nOuter; b2++)
					for(int b1=0; b1<s.nOuter; b1++)
					{	double tauDeltaE = tau*(s.E[b1] - s.E[b2] - omega);
						*(deltaData++) = normFac * exp(-0.5*tauDeltaE*tauDeltaE) * cis(t*(s.E[b1]-s.E[b2]));
					}
				//Loop over polarizations:
				for(int iPol=0; iPol<int(pol.size()); iPol++)
				{	//Multiply matrix elements with energy conservation:
					matrix P = Ppol[iPol];
					eblas_zmul(P.nData(), delta.data(),1, P.data(),1); //P-
					matrix Pdag = dagger(P); //P+
					//Compute change in rho due to probe (summed over excitation/deexcitational already):
					diagMatrix deltaRhoDiag = diag(Pdag*drhoCur*P + P*drhoCur*Pdag - drhoCur*P*Pdag - Pdag*P*drhoCur);
					dimEps[iPol*nomega+iomega] += prefac * dot(s.E, deltaRhoDiag);
				}
			}
		}
		//Accumulate contributions from all processes on head:
		mpiWorld->reduceData(dimEps, MPIUtil::ReduceSum);
		watch.stop();
		return dimEps;
	}
	
	//Write change in imEps to plain-text file:
	void writeDeltaImEps(string fname, const diagMatrix& dimEps) const
	{	if(mpiWorld->isHead())
		{	ofstream ofs(fname);
			ofs << "#omega[eV]";
			for(int iPol=0; iPol<int(pol.size()); iPol++)
				ofs << " dImEps" << (iPol+1);
			ofs << "\n";
			for(int iomega=0; iomega<nomega; iomega++)
			{	double omega = omegaMin + iomega*domega;
				ofs << omega/eV;
				for(int iPol=0; iPol<int(pol.size()); iPol++)
					ofs << '\t' << dimEps[iPol*nomega+iomega];
				ofs << '\n';
			}
		}
	}
	
	//Apply pump using perturbation theory (instantly go from before to after pump, skipping time evolution)
	void applyPump()
	{	static StopWatch watch("Lindblad::applyPump"); 
		watch.start();
		const State* sPtr = state.data();
		//Perturb each k separately:
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	const State& s = *(sPtr++);
			if(pumpBfield)
			{	//Construct Hamiltonian including magnetic field contribution:
				matrix Htot(s.E(s.innerStart, s.innerStart+s.nInner));
				for(int iDir=0; iDir<3; iDir++) //Add Zeeman Hamiltonian
					Htot -= pumpB[iDir] * s.S[iDir];
				//Set rho to Fermi function of this perturbed Hamiltonian:
				diagMatrix Epert; matrix Vpert;
				Htot.diagonalize(Vpert, Epert);
				diagMatrix fPert(s.nInner);
				for(int b=0; b<s.nInner; b++)
					fPert[b] = fermi((Epert[b]-dmu)*invT);
				matrix rhoPert = Vpert * fPert * dagger(Vpert);
				accumRhoHC(0.5*(rhoPert-s.rho0), drho.data()+rhoOffset[ik]);
			}
			else
			{	const diagMatrix& rho0 = s.rho0;
				matrix rho0bar(bar(rho0)); //1-rho0
				//Compute and apply perturbation:
				matrix P = s.pumpPD; //P-
				matrix Pdag = dagger(P); //P+
				matrix deltaRho;
				for(int s=-1; s<=+1; s+=2)
				{	deltaRho += rho0bar*P*rho0*Pdag - Pdag*rho0bar*P*rho0;
					std::swap(P, Pdag); //P- <--> P+
				}
				accumRhoHC((M_PI*pumpA0*pumpA0) * deltaRho, drho.data()+rhoOffset[ik]);
			}
		}
		watch.stop();
	}
	
	//Time evolution operator returning drho/dt
	DM1 compute(double t, const DM1& drho)
	{	static StopWatch watchEph("Lindblad::compute::ePh");
		
		DM1 drhoDot(drho.size(), 0.);
		
		//E-ph relaxation contribution:
		if(ePhEnabled)
		{	watchEph.start();
			//Convert interaction picture rho data to Schrodinger picture version in PETSc format:
			double* vRhoPtr;  VecGetArray(vRho, &vRhoPtr);
			eblas_zero(drho.size(), vRhoPtr);
			std::vector<complex> phase(drho.size());
			complex* phaseData = phase.data();
			const State* sPtr = state.data();
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	const State& s = *(sPtr++);
				const double* Einner = s.E.data() + s.innerStart;
				matrix drhoCur = getRho(drho.data()+rhoOffset[ik], s.nInner);
				complex* drhoData = drhoCur.data();
				for(int bCol=0; bCol<s.nInner; bCol++)
					for(int bRow=0; bRow<s.nInner; bRow++)
					{	complex phase = 0.5*cis(t*(Einner[bCol]-Einner[bRow])); //factor of 1/2 in order to use accumRhoHC
						*(drhoData++) *= phase;
						*(phaseData++) = phase.conj(); //cache the reverse phase for below
					}
				accumRhoHC(drhoCur, vRhoPtr+rhoOffset[ik]);
			}
			VecRestoreArray(vRho, &vRhoPtr);
			//Apply sparse operator using PETSc:
			MatMult(evolveMat, vRho, vRhoDot);
			//Copy Schrodinger picture rhoDot data in PETSc format back to interaction picture:
			const double* vRhoDotPtr;  VecGetArrayRead(vRhoDot, &vRhoDotPtr);
			sPtr = state.data();
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	const State& s = *(sPtr++);
				matrix rhoDotCur = getRho(vRhoDotPtr+rhoOffset[ik], s.nInner);
				eblas_zmul(rhoDotCur.nData(), phase.data()+rhoOffset[ik],1, rhoDotCur.data(),1);
				accumRhoHC(rhoDotCur, drhoDot.data()+rhoOffset[ik]);
			}
			VecRestoreArrayRead(vRhoDot, &vRhoDotPtr);
			watchEph.stop();
		}
		
		if(verbose)
		{	//Report current statistics:
			double drhoDotMax = 0., drhoEigMin = +DBL_MAX, drhoEigMax = -DBL_MAX;
			const State* sPtr = state.data();
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	const State& s = *(sPtr++);
				//max(rhoDot)
				const matrix drhoDotCur = getRho(drhoDot.data()+rhoOffset[ik], s.nInner);
				drhoDotMax = std::max(drhoDotMax, drhoDotCur.data()[cblas_izamax(drhoDotCur.nData(), drhoDotCur.data(), 1)].abs());
				//eig(rho):
				const matrix drhoCur = getRho(drho.data()+rhoOffset[ik], s.nInner);
				matrix V; diagMatrix f;
				drhoCur.diagonalize(V, f);
				drhoEigMin = std::min(drhoEigMin, f.front());
				drhoEigMax = std::max(drhoEigMax, f.back());
			}
			mpiWorld->reduce(drhoDotMax, MPIUtil::ReduceMax);
			mpiWorld->reduce(drhoEigMax, MPIUtil::ReduceMax);
			mpiWorld->reduce(drhoEigMin, MPIUtil::ReduceMin);
			logPrintf("\n\tComputed at t[fs]: %lg  max(drhoDot): %lg drhoEigRange: [ %lg %lg ] ",
				t/fs, drhoDotMax, drhoEigMin, drhoEigMax); logFlush();
		}
		else logPrintf("(t[fs]: %lg) ", t/fs);
		logFlush();
		
		return drhoDot;
	}
	
	//Print / dump quantities at each checkpointed step / eigenmode
	void report(double t, const DM1& drho, complex eig, double eigErr) const
	{	static StopWatch watch("Lindblad::report"); watch.start();
		ostringstream ossID; ossID << stepID;
		//Compute total energy and distributions:
		int nDist = spinorial ? 4 : 1; //number distribution only, or also spin distribution
		std::vector<Histogram> dist(nDist, Histogram(Emin, dE, Emax));
		const double prefac = spinWeight*(1./nkTot); //BZ integration weight
		double Etot = 0., dfMax = 0.; vector3<> Stot;
		const State* sPtr = state.data();
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	const State& s = *(sPtr++);
			const matrix drhoCur = getRho(drho.data()+rhoOffset[ik], s.nInner);
			//Energy and distribution:
			const complex* drhoData = drhoCur.data();
			for(int b=0; b<s.nInner; b++)
			{	double weight = prefac * drhoData->real();
				const double& Ecur = s.E[b+s.innerStart];
				Etot += weight * Ecur;
				dfMax = std::max(dfMax, fabs(drhoData->real()));
				dist[0].addEvent(Ecur, weight);
				drhoData += (s.nInner+1); //advance to next diagonal entry
			}
			//Spin distribution (if available):
			if(spinorial)
			{	const complex* drhoData = drhoCur.data();
				vector3<const complex*> Sdata; for(int k=0; k<3; k++) Sdata[k] = s.S[k].data();
				std::vector<vector3<>> Sband(s.nInner); //spin expectation by band S_b := sum_a S_ba drho_ab
				const double* Einner = s.E.data() + s.innerStart;
				for(int b2=0; b2<s.nInner; b2++)
				{	for(int b1=0; b1<s.nInner; b1++)
					{	complex weight = prefac * (*(drhoData++)).conj() * cis((Einner[b1]-Einner[b2])*t);
						for(int iDir=0; iDir<3; iDir++)
							Sband[b2][iDir] += (weight * (*(Sdata[iDir]++))).real();
					}
					Stot += Sband[b2];
				}
				//Collect distribution based on per-band spin:
				for(int b=0; b<s.nInner; b++)
				{	const double& E = s.E[b+s.innerStart];
					int iEvent; double tEvent;
					if(dist[1].eventPrecalc(E, iEvent, tEvent))
					{	for(int iDir=0; iDir<3; iDir++)
							dist[iDir+1].addEventPrecalc(iEvent, tEvent, Sband[b][iDir]);
					}
				}
			}
		}
		mpiWorld->reduce(Etot, MPIUtil::ReduceSum);
		mpiWorld->reduce(Stot, MPIUtil::ReduceSum);
		mpiWorld->reduce(dfMax, MPIUtil::ReduceMax);
		for(Histogram& h: dist) h.reduce(MPIUtil::ReduceSum);
		if(mpiWorld->isHead())
		{	//Report step ID and energy:
			if(spectrumMode)
			{	double decayTime = -1./eig.real();
				double period = (2*M_PI)/fabs(eig.imag());
				logPrintf("Mode: %2d  DecayTime[fs]: %11.5lg  Period[fs]: %11.5lg  RelErr: %11.5lg",
					stepID, decayTime/fs, period/fs, eigErr);
			}
			else logPrintf("Integrate: Step: %4d   t[fs]: %6.1lf   Etot[eV]: %.6lf", stepID, t/fs, Etot/eV);
			logPrintf("   dfMax: %6.4lf", dfMax);
			if(spinorial) logPrintf("   S: [ %11.4lg %11.4lg %11.4lg ]", Stot[0],  Stot[1],  Stot[2]);
			logPrintf("\n"); logFlush();
			//Save distribution functions:
			ofstream ofs("dist."+ossID.str());
			ofs << "#E-mu/VBM[eV] n[eV^-1]";
			if(spinorial)
				ofs << "Sx[eV^-1] Sy[eV^-1] Sz[eV^-1]";
			ofs << "\n";
			for(int iE=0; iE<dist[0].nE; iE++)
			{	double E = Emin + iE*dE;
				ofs << E/eV;
				for(int iDist=0; iDist<nDist; iDist++)
					ofs << '\t' << dist[iDist].out[iE]*eV;
				ofs << '\n';
			}
		}
		//Write checkpoint file if needed:
		if(checkpointFile.length())
		{
			#ifdef MPI_SAFE_WRITE
			if(mpiWorld->isHead())
			{	FILE* fp = fopen(checkpointFile.c_str(), "w");
				fwrite(&stepID, sizeof(int), 1, fp);
				fwrite(&t, sizeof(double), 1, fp);
				//Data from head:
				fwrite(drho.data(), sizeof(double), drho.size(), fp);
				//Data from remaining processes:
				for(int jProc=1; jProc<mpiWorld->nProcesses(); jProc++)
				{	DM1 buf(rhoSize[jProc]);
					mpiWorld->recvData(buf, jProc, 0); //recv data to be written
					fwrite(buf.data(), sizeof(double), buf.size(), fp);
				}
				fclose(fp);
			}
			else mpiWorld->sendData(drho, 0, 0); //send to head for writing
			#else
			//Write in parallel using MPI I/O:
			MPIUtil::File fp; mpiWorld->fopenWrite(fp, checkpointFile.c_str());
			//--- Write current step and time as a header:
			if(mpiWorld->isHead())
			{	mpiWorld->fwrite(&stepID, sizeof(int), 1, fp);
				mpiWorld->fwrite(&t, sizeof(double), 1, fp);
			}
			//--- Move to location of this process's data:
			size_t offset = sizeof(int) + sizeof(double); //offset due to header
			for(int jProc=0; jProc<mpiWorld->iProcess(); jProc++)
				offset += sizeof(double) * rhoSize[jProc]; //offset due to data from previous processes
			mpiWorld->fseek(fp, offset, SEEK_SET);
			//--- Write this process's data:
			mpiWorld->fwrite(drho.data(), sizeof(double), drho.size(), fp);
			mpiWorld->fclose(fp);
			#endif
		}
		watch.stop();
		//Probe responses if present:
		diagMatrix imEps = calcDeltaImEps(t, drho);
		if(imEps.size())
			writeDeltaImEps("dimEps."+ossID.str(), imEps);
		//Increment stepID:
		((LindbladLinear*)this)->stepID++;
	}
	void report(double t, const DM1& drho) const { report(t, drho, complex(), 0.); }
};

inline void print(FILE* fp, const vector3<complex>& v, const char* format="%lg ")
{	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].real()); std::fprintf(fp, "] + 1j*");
	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].imag()); std::fprintf(fp, "]\n");
}
inline vector3<complex> normalize(const vector3<complex>& v) { return v * (1./sqrt(v[0].norm() + v[1].norm() + v[2].norm())); }

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Lindblad linearized dynamics or spectrum in an ab initio Wannier basis");
	int argcSlepc=1; CHECKERR(SlepcInitialize(&argcSlepc, &argv, (char*)0, "")); //don't let slepc see the actual command line (too many conflicts)
	
	//Get the system parameters:
	InputMap inputMap(ip.inputFilename);
	//--- doping / temperature
	const double dmu = inputMap.get("dmu", 0.) * eV; //optional: shift in fermi level from neutral value / VBM in eV (default: 0)
	const double T = inputMap.get("T") * Kelvin; //temperature in Kelvin (ambient phonon T = initial electron T)
	const string mode = inputMap.getString("mode"); //RealTime or Spectrum or SpectrumSparse
	if(mode!="RealTime" and mode!="Spectrum" and mode!="SpectrumSparse")
		die("\nmode must be 'RealTime', 'Spectrum' or 'SpectrumSparse'\n");
	const bool spectrumMode = (mode == "Spectrum" or mode == "SpectrumSparse");
	const bool sparseDiag = (mode == "SpectrumSparse");
	#ifndef SCALAPACK_ENABLED
	if(spectrumMode and (not sparseDiag))
		die("\nSpectrum (dense diagonalization) mode requires linking with ScaLAPACK.\n");
	#endif
	const int blockSize = int(inputMap.get("blockSize", 64));
	//--- eiegen-decomposition parameters (required and used only in spectrum mode)
	const int nEigs = int(inputMap.get("nEigs", spectrumMode ? NAN : 0.)); //number of eigenvectors to compute
	const double eigTol = inputMap.get("eigTol", 1e-7); //convergence threshold on eigenvalues
	const int innerIter = int(inputMap.get("innerIter", 10)); //number of iterations for inner linear-solve inversion
	const double innerTol = inputMap.get("innerTol", 1e-3); //convergence threshold for inner linear-solve inversion
	//--- time evolution parameters (required and used only in real time mode)
	const double dt = inputMap.get("dt", spectrumMode ? 0. : NAN) * fs; //time interval between reports
	const double tStop = inputMap.get("tStop", spectrumMode ? 0. : NAN) * fs; //stopping time for simulation
	const double tStep = inputMap.get("tStep", 0.) * fs; //if non-zero, time step for fixed-step (non-adaptive) integrator
	const double tolAdaptive = inputMap.get("tolAdaptive", 1e-3); //relative tolerance for adaptive integrator
	//--- pump / Bfield (required and used only in real time mode)
	const string pumpMode = spectrumMode ? "Bfield" : inputMap.getString("pumpMode"); //must be Perturb or Bfield (Evolve not allowed)
	if(pumpMode!="Perturb" and pumpMode!="Bfield")
		die("\npumpMode must be 'Perturb' or 'Bfield' (Evolve not supported by lindbladLinear)'\n");
	const double Tesla = Joule/(Ampere*meter*meter);
	const vector3<> pumpB = inputMap.getVector("pumpB", vector3<>()) * Tesla; //perturbing initial magnetic field in Tesla (used only in Bfield mode)
	const double pumpOmega = inputMap.get("pumpOmega", (spectrumMode or pumpMode=="Bfield") ? 0. : NAN) * eV; //pump frequency in eV (used only in Evolve or Perturb modes)
	const double pumpA0 = inputMap.get("pumpA0", (spectrumMode or pumpMode=="Bfield") ? 0. : NAN); //pump pulse amplitude / intensity (Units TBD)
	const double pumpTau = inputMap.get("pumpTau", (spectrumMode or pumpMode=="Bfield") ? 0. : NAN)*fs; //Gaussian pump pulse width (sigma of amplitude) in fs
	const vector3<complex> pumpPol = normalize(
		complex(1,0)*inputMap.getVector("pumpPolRe", vector3<>(1.,0.,0.)) +  //Real part of polarization
		complex(0,1)*inputMap.getVector("pumpPolIm", vector3<>(0.,0.,0.)) ); //Imag part of polarization
	//--- probes (used in both modes; parameters required only if one or more polarizations specified)
	std::vector<vector3<complex>> pol;
	while(true)
	{	int iPol = int(pol.size())+1;
		ostringstream oss; oss << iPol;
		string polName = oss.str();
		vector3<> polRe = inputMap.getVector("polRe"+polName, vector3<>(0.,0.,0.)); //Real part of polarization
		vector3<> polIm = inputMap.getVector("polIm"+polName, vector3<>(0.,0.,0.)); //Imag part of polarization
		if(polRe.length_squared() + polIm.length_squared() == 0.) break; //End of probe polarizations
		pol.push_back(normalize(complex(1,0)*polRe + complex(0,1)*polIm));
	}
	const double omegaMin = inputMap.get("omegaMin", pol.size() ? NAN : 0.) * eV; //start of frequency grid for probe response
	const double omegaMax = inputMap.get("omegaMax", pol.size() ? NAN : 0.) * eV; //end of frequency grid for probe response
	const double domega = inputMap.get("domega", pol.size() ? NAN : 0.) * eV; //frequency resolution for probe calculation
	const double tau = inputMap.get("tau", pol.size() ? NAN :  0.) * fs; //Gaussian probe pulse width (sigma of amplitude) in fs
	//--- general options
	const double dE = inputMap.get("dE") * eV; //energy resolution for distribution functions
	const string ePhMode = inputMap.getString("ePhMode"); //must be Off or DiagK (add FullK in future)
	if(ePhMode!="Off" and ePhMode!="DiagK")
		die("\nePhMode must be 'Off' or 'DiagK'\n");
	const bool ePhEnabled = (ePhMode != "Off");
	if(spectrumMode and not ePhEnabled)
		die("\nePhMode must be 'DiagK' in Spectrum mode\n");
	const string verboseMode = inputMap.has("verbose") ? inputMap.getString("verbose") : "no"; //must be yes or no
	if(verboseMode!="yes" and verboseMode!="no")
		die("\nverboseMode must be 'yes' or 'no'\n");
	const bool verbose = (verboseMode=="yes");
	const string inFile = inputMap.has("inFile") ? inputMap.getString("inFile") : "ldbd.dat"; //input file name
	const string checkpointFile = (inputMap.has("checkpointFile") and (not spectrumMode)) ? inputMap.getString("checkpointFile") : ""; //checkpoint file name
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("dmu = %lg\n", dmu);
	logPrintf("T = %lg\n", T);
	logPrintf("mode = %s\n", mode.c_str());
	if(spectrumMode)
	{	logPrintf("nEigs = %d\n", nEigs);
		logPrintf("eigTol = %lg\n", eigTol);
		logPrintf("innerIter = %d\n", innerIter);
		logPrintf("innerTol = %lg\n", innerTol);
	}
	else
	{	logPrintf("dt = %lg\n", dt);
		logPrintf("tStop = %lg\n", tStop);
		logPrintf("tStep = %lg\n", tStep);
		logPrintf("tolAdaptive = %lg\n", tolAdaptive);
		logPrintf("pumpMode = %s\n", pumpMode.c_str());
		if(pumpMode == "Bfield")
		{	logPrintf("pumpB = "); pumpB.print(globalLog, " %lg ");
		}
		else
		{	logPrintf("pumpOmega = %lg\n", pumpOmega);
			logPrintf("pumpA0 = %lg\n", pumpA0);
			logPrintf("pumpTau = %lg\n", pumpTau);
			logPrintf("pumpPol = "); print(globalLog, pumpPol);
		}
	}
	if(pol.size())
	{	for(int iPol=0; iPol<int(pol.size()); iPol++)
		{	logPrintf("pol%d = ", iPol+1);
			print(globalLog, pol[iPol]);
		}
		logPrintf("omegaMin = %lg\n", omegaMin);
		logPrintf("omegaMax = %lg\n", omegaMax);
		logPrintf("domega = %lg\n", domega);
		logPrintf("tau = %lg\n", tau);
	}
	logPrintf("dE = %lg\n", dE);
	logPrintf("ePhMode = %s\n", ePhMode.c_str());
	logPrintf("verbose = %s\n", verboseMode.c_str());
	logPrintf("inFile = %s\n", inFile.c_str());
	if(not spectrumMode) logPrintf("checkpointFile = %s\n", checkpointFile.c_str());
	logPrintf("\n");
	
	//Create and initialize lindblad calculator:
	LindbladLinear lbl(dmu, T, spectrumMode, sparseDiag, blockSize,
		pumpOmega, pumpA0, pumpTau, pumpPol, (pumpMode=="Bfield"), pumpB,
		omegaMin, omegaMax, domega, tau, pol, dE,
		ePhEnabled, verbose, checkpointFile);
	CHECKERR(lbl.initialize(inFile));
	logPrintf("Initialization completed successfully at t[s]: %9.2lf\n\n", clock_sec());
	logFlush();
	
	logPrintf("%lu active k-points parallelized over %d processes.\n", lbl.nk, mpiWorld->nProcesses());
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		CHECKERR(SlepcFinalize());
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	if(spectrumMode and sparseDiag)
	{
		//----------- Sparse diagonalization using SLEPc ---------------
		
		//Create the eigensolver and set various options:
		EPS eps; CHECKERR(EPSCreate(PETSC_COMM_WORLD, &eps));
		CHECKERR(EPSSetDimensions(eps, nEigs, PETSC_DEFAULT, PETSC_DEFAULT));
		CHECKERR(EPSSetTolerances(eps, eigTol, PETSC_DEFAULT));
		CHECKERR(EPSSetOperators(eps, lbl.evolveMat, NULL));
		CHECKERR(EPSSetProblemType(eps, EPS_NHEP)); //Non-Hermitian
		CHECKERR(EPSSetType(eps, EPSJD)); //Jacobi-Davidson algorithm
		CHECKERR(EPSSetTarget(eps, 0)); //Target eigenvalues closes to 0. using spectral transformation below
		CHECKERR(EPSSetWhichEigenpairs(eps, EPS_TARGET_REAL));
		CHECKERR(EPSSetExtraction(eps, EPS_HARMONIC));
		typedef PetscErrorCode (*SlepcMonitor)(EPS,PetscInt,PetscInt,PetscScalar*,PetscScalar*,PetscReal*,PetscInt,void*);
		typedef PetscErrorCode (*SlepcContextDestroy)(void**);
		PetscViewerAndFormat* vf; CHECKERR(PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_DEFAULT, &vf));
		CHECKERR(EPSMonitorSet(eps, (SlepcMonitor)EPSMonitorFirst, (void*)vf, (SlepcContextDestroy)PetscViewerAndFormatDestroy));
		
		//--- Spectral transformation to get to smallest eigenvalues:
		ST st; CHECKERR(EPSGetST(eps, &st));
		CHECKERR(STSetType(st, STPRECOND));
		KSP ksp; CHECKERR(STGetKSP(st, &ksp)); //Krylov method that acts as preconditioner for JD
		CHECKERR(KSPSetType(ksp, KSPGMRES)); //Select Krylov method eg. BCGSL, GMRES etc.
		CHECKERR(KSPSetTolerances(ksp, innerTol, PETSC_DEFAULT, PETSC_DEFAULT, innerIter));
		//--- Set custom preconditioner for inner solve:
		PC pc; CHECKERR(KSPGetPC(ksp, &pc));
		CHECKERR(PCSetType(pc, PCGAMG)); //eg. SOR, GAMG (Multigrid)
		/* //Custom block-diagonal preconditioner (does not seem to be working well)
		CHECKERR(PCSetType(pc, PCMAT));
		CHECKERR(PCSetOperators(pc, lbl.precondMat, NULL));
		CHECKERR(STPrecondSetMatForPC(st, lbl.precondMat));
		*/

		//Set deflation space (trace of density matrix conserved):
		Vec nullVec; CHECKERR(MatCreateVecs(lbl.evolveMat, &nullVec, NULL));
		double* nullVecPtr;  CHECKERR(VecGetArray(nullVec, &nullVecPtr));
		{	for(const LindbladLinear::State& s: lbl.state)
				for(int col=0; col<s.nInner; col++)
					for(int row=0; row<s.nInner; row++)
						*(nullVecPtr++) = (row==col ? 1. : 0.);
		}
		CHECKERR(VecRestoreArray(nullVec, &nullVecPtr));
		CHECKERR(EPSSetDeflationSpace(eps, 1, &nullVec));
		CHECKERR(VecDestroy(&nullVec));
		
		//Solve the eigensystem:
		CHECKERR(EPSSolve(eps));
		
		//Display solution:
		logPrintf("\n"); logFlush();
		int nConverged; CHECKERR(EPSGetConverged(eps, &nConverged));
		for(int iEig=0; iEig<nConverged; iEig++)
		{	complex eig; CHECKERR(EPSGetEigenpair(eps, iEig, &eig.real(), &eig.imag(), lbl.vRho, lbl.vRhoDot));
			double err; CHECKERR(EPSGetErrorEstimate(eps, iEig, &err));
			//Convert eigenvector:
			DM1& evec =  lbl.drho;
			const Vec& vEvec = eig.imag()>=0 ? lbl.vRho : lbl.vRhoDot;
			const double* vEvecData; CHECKERR(VecGetArrayRead(vEvec, &vEvecData));
			eblas_copy(evec.data(), vEvecData, evec.size());
			CHECKERR(VecRestoreArrayRead(vEvec, &vEvecData));
			//Report:
			lbl.stepID = iEig;
			lbl.report(0., evec, eig, err);
		}
		logPrintf("\n"); logFlush();
		
		//Clean up:
		CHECKERR(EPSDestroy(&eps));
	}
	else if(spectrumMode and (not sparseDiag))
	{
		//----------- Dense diagonalization using ScaLAPACK ---------------
		
	}
	else if(not ePhEnabled)
	{	//Simple probe-pump-probe with no relaxation:
		lbl.report(-dt, lbl.drho);
		lbl.applyPump(); //takes care of optical pump or B-field excitation
		lbl.report(0., lbl.drho);
	}
	else
	{	double tStart = 0.;
		bool checkpointExists = false;
		if(mpiWorld->isHead())
			checkpointExists = (checkpointFile.length()>0) and (fileSize(checkpointFile.c_str())>0);
		mpiWorld->bcast(checkpointExists);
		if(checkpointExists)
		{	logPrintf("Reading checkpoint from '%s' ... ", checkpointFile.c_str()); logFlush(); 
			//Determine offset of current process data and total expected file length:
			size_t offset = sizeof(int)+sizeof(double); //offset due to header
			for(int jProc=0; jProc<mpiWorld->iProcess(); jProc++)
				offset += sizeof(double) * lbl.rhoSize[jProc]; //offset due to data from previous processes
			size_t fsizeExpected = offset;
			for(int jProc=mpiWorld->iProcess(); jProc<mpiWorld->nProcesses(); jProc++)
				fsizeExpected += sizeof(double) * lbl.rhoSize[jProc];
			mpiWorld->bcast(fsizeExpected);
			//Open check point file and rrad time header:
			MPIUtil::File fp; mpiWorld->fopenRead(fp, checkpointFile.c_str(), fsizeExpected);
			mpiWorld->fread(&(lbl.stepID), sizeof(int), 1, fp);
			mpiWorld->fread(&tStart, sizeof(double), 1, fp);
			mpiWorld->bcast(tStart);
			//Read density matrix from check point file:
			mpiWorld->fseek(fp, offset, SEEK_SET);
			mpiWorld->fread(lbl.drho.data(), sizeof(double), lbl.drho.size(), fp);
			mpiWorld->fclose(fp);
			logPrintf("done.\n");
		}
		else
		{	//Do an initial report akin to above and apply the pump/B-field:
			lbl.report(-dt, lbl.drho);
			lbl.applyPump(); //takes care of optical pump or B-field excitation
			tStart = 0.; //integrate will report at t=0 below, before evolving ePh relaxation
		}
		//Evolve:
		if(tStep) //Fixed-step integrator:
			lbl.integrateFixed(lbl.drho, tStart, tStop, tStep, dt);
		else //Adaptive integrator:
			lbl.integrateAdaptive(lbl.drho, tStart, tStop, tolAdaptive, dt);
	}
	
	//Cleanup:
	CHECKERR(lbl.cleanup());
	CHECKERR(SlepcFinalize());
	FeynWann::finalize();
	return 0;
}
