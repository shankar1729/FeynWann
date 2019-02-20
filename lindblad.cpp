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
#include "FeynWann.h"
#include "Histogram.h"
#include "InputMap.h"
#include "Integrator.h"
#include <core/Units.h>

inline matrix dot(const matrix* P, vector3<complex> pol)
{	return pol[0]*P[0] + pol[1]*P[1] + pol[2]*P[2];
}

//Time evolution data type and required functions:
typedef std::vector<matrix> DM1; //array of density matrices at each k
double dot(const DM1& x, const DM1& y)
{	assert(x.size()==y.size());
	double result = 0.;
	for(size_t i=0; i<x.size(); i++)
		result += dot(x[i], y[i]);
	mpiWorld->allReduce(result, MPIUtil::ReduceSum, true);
	return result;
}
void axpy(double a, const DM1& x, DM1& y)
{	assert(x.size()==y.size());
	for(size_t i=0; i<x.size(); i++)
		axpy(a, x[i], y[i]);
}
DM1& operator*=(DM1& x, double a)
{	for(matrix& x_i: x) x_i *= a;
	return x;
}
DM1 clone(const DM1& x) { return x; }

//----- Triplet format square sparse matrix with restricted operations ----
struct SparseEntry
{	int i, j;
	complex val;
};
typedef std::vector<SparseEntry> SparseMatrix;

//Multiply dagger(S)*M*S for sparse matrix S and dense matrix M
SparseMatrix SdagMS(const SparseMatrix& S, const matrix& M)
{	SparseMatrix result; result.reserve(S.size()*S.size());
	const complex* m = M.data();
	for(const SparseEntry& s1: S)
		for(const SparseEntry& s2: S)
		{	SparseEntry sr;
			sr.i = s1.j;
			sr.j = s2.j;
			sr.val = s1.val.conj() * m[M.index(s1.i,s2.i)] * s2.val;
			result.push_back(sr);
		}
	return result;
}

//Multiply S*M*dagger(S) for sparse matrix S and dense matrix M
SparseMatrix SMSdag(const SparseMatrix& S, const matrix& M)
{	SparseMatrix result; result.reserve(S.size()*S.size());
	const complex* m = M.data();
	for(const SparseEntry& s1: S)
		for(const SparseEntry& s2: S)
		{	SparseEntry sr;
			sr.i = s1.i;
			sr.j = s2.i;
			sr.val = s1.val * m[M.index(s1.j,s2.j)] * s2.val.conj();
			result.push_back(sr);
		}
	return result;
}

//Multiply sparse matrix with dense matrix:
matrix operator*(const matrix& M, const SparseMatrix& S)
{	int N = M.nRows(); //assumed square
	matrix R = zeroes(N, N);
	complex* r = R.data();
	const complex* m = M.data();
	for(const SparseEntry& s: S)
	{	complex* rCur = r + N*s.j;
		const complex* mCur = m + N*s.i;
		for(int k=0; k<N; k++)
			*(rCur++) += *(mCur++) * s.val;
	}
	return R;
}


//Lindblad initialization, time evolution and measurement operators using FeynWann callback
struct Lindblad : public Integrator<DM1>
{	
	FeynWann& fw;
	const std::vector<vector3<>>& k0; //!< k-point offsets
	const size_t oStart, oStop, noMine; //!< range of offsets handled by this process group
	const size_t ikStart, ikStop, nkMine; //!< range of k-points within offset handled by this process
	const size_t nkOffset, nkTot; //!< numbe rof k-points per offset, and total k-points effectively used in BZ sampling
	
	int stepID; //current time and reporting step number
	DM1 rho; //!< current density matrices (indexed by offset and ik)
	
	const double dmu, T, invT; //!< Fermi level position relative to neutral value / VBM, and temperature
	const double pumpOmega, pumpA0, pumpTau; const vector3<complex> pumpPol; const bool pumpEvolve; //!< pump parameters
	const double omegaMin, domega; const int nomega; //!< probe frequency grid
	const double tau; const std::vector<vector3<complex>> pol; //!< probe parameters
	const double dE; //!< energy resolution for distribution functions
	
	const bool ePhEnabled; //!< whether e-ph coupling is enabled
	const double ePhDelta; //!< Gaussian energy conservation width
	const bool verbose; //!< whether to print more detailed stats during evolution
	
	Lindblad(FeynWann& fw, const std::vector<vector3<>>& k0, int oStart, int oStop,
		double dmu, double T, double pumpOmega, double pumpA0, double pumpTau, vector3<complex> pumpPol, bool pumpEvolve,
		double omegaMin, double omegaMax, double domega, double tau, std::vector<vector3<complex>> pol, double dE,
		bool ePhEnabled, double ePhDelta, bool verbose)
	: fw(fw), k0(k0), oStart(oStart), oStop(oStop), noMine(oStop-oStart),
		ikStart(fw.Hw->ikStart), ikStop(ikStart+fw.Hw->nk), nkMine(ikStop-ikStart),
		nkOffset(fw.eCountPerOffset()), nkTot(nkOffset*k0.size()), stepID(0),
		rho(noMine * nkMine), dmu(dmu), T(T), invT(1./T),
		pumpOmega(pumpOmega), pumpA0(pumpA0), pumpTau(pumpTau), pumpPol(pumpPol), pumpEvolve(pumpEvolve),
		omegaMin(omegaMin), domega(domega), nomega(1+int(round((omegaMax-omegaMin)/domega))),
		tau(tau), pol(pol), dE(dE), ePhEnabled(ePhEnabled), ePhDelta(ePhDelta), verbose(verbose)
	{
	}
	
	//--------- Initialize -------------
	
	//Entry in e-ph coupling list below:
	struct GePhEntry
	{	SparseMatrix G; //coupling matrix to partner
		double omegaPh; //corresponding phonon frequency
	};
	typedef std::vector<std::vector<GePhEntry>> GePhSet; //first index is global index of other k-point, second is phonon mode
	//---Serialize for MPI communications:
	void sendGePh(const std::vector<GePhEntry>& gArr, int dest, int tag)
	{	ostringstream oss;
		//Overall size:
		size_t nMatrices = gArr.size();
		oss.write((const char*)&nMatrices, sizeof(size_t));
		//Write each sparse matrix:
		for(const GePhEntry& g: gArr)
		{	size_t nEntries = g.G.size();
			oss.write((const char*)&nEntries, sizeof(size_t));
			oss.write((const char*)g.G.data(), sizeof(SparseEntry)*nEntries);
			oss.write((const char*)&g.omegaPh, sizeof(double));
		}
		//Send over MPI:
		mpiGroup->send(oss.str(), dest, tag);
	}
	void recvGePh(std::vector<GePhEntry>& gArr, int src, int tag)
	{	//Receive over MPI:
		string buf;
		mpiGroup->recv(buf, src, tag);
		istringstream iss(buf);
		//Overall size:
		size_t nMatrices = 0;
		iss.read((char*)&nMatrices, sizeof(size_t));
		gArr.resize(nMatrices);
		//Read each matrix:
		for(GePhEntry& g: gArr)
		{	size_t nEntries = 0;
			iss.read((char*)&nEntries, sizeof(size_t));
			g.G.resize(nEntries);
			iss.read((char*)g.G.data(), sizeof(SparseEntry)*nEntries);
			iss.read((char*)&g.omegaPh, sizeof(double));
		}
	}
	
	//Cache required properties per state
	struct State
	{	diagMatrix E; //energy eigenvalues (i.e. H0)
		diagMatrix rho0; //equilibrium / initial density matrix (diagonal)
		std::vector<matrix> P; //P matrix elements for each probe polarization (energy conservation delta (D) not included)
		std::vector<matrix> S; //spin matrix elements (if available)
		matrix pumpPD; //P matrix elements at pump polarization x energy conservation delta (D), but without A0 and time factor
		GePhSet GePh; //list of e-ph partners
	};
	std::vector<State> state;
	double Emin, Emax; //range of energies in grid
	
	inline void initializeE(const FeynWann::StateE& stateE, int o)
	{	//Identify destination for results:
		size_t index = stateE.ik-ikStart + nkMine*(o-oStart);
		State& s = state[index];
		
		//Cache required properties to state:
		//--- Energies
		s.E = stateE.E;
		Emin = std::min(Emin, s.E.front());
		Emax = std::max(Emax, s.E.back());
		//--- Probe matrix elements (without energy conservation)
		s.P.clear(); s.P.reserve(pol.size());
		for(const vector3<complex>& pol_i: pol)
			s.P.push_back(dot(stateE.v, pol_i));
		//--- Pump matrix elements
		s.pumpPD = dot(stateE.v, pumpPol);
		double normFac = sqrt(pumpTau/sqrt(M_PI));
		complex* PDdata = s.pumpPD.data();
		for(int b2=0; b2<fw.nBands; b2++)
			for(int b1=0; b1<fw.nBands; b1++)
			{	//Multiply energy conservation:
				double tauDeltaE = pumpTau*(s.E[b1] - s.E[b2] - pumpOmega);
				*(PDdata++) *= normFac * exp(-0.5*tauDeltaE*tauDeltaE);
			}
		//--- Spin matrix elements
		if(fw.fwp.needSpin)
			s.S.assign(stateE.S, stateE.S+3);
		
		//Set rho to initial occupations:
		s.rho0.resize(fw.nBands);
		for(int b=0; b<fw.nBands; b++)
		{	double expArg = (s.E[b]-dmu)*invT;
			s.rho0[b] = (expArg < -30.) ? 1.
				: ((expArg > +30.) ? 0.
				: 1./(1.+exp(expArg)) );
		}
		rho[index] = s.rho0;
	}
	struct InitEparams { Lindblad* lb; size_t o; };
	static void initializeE(const FeynWann::StateE& stateE, void* params)
	{	InitEparams& p = *((InitEparams*)params);
		p.lb->initializeE(stateE, p.o);
	}
	
	std::vector<GePhSet> GePhGroup; //temporary GePh storage within process group for initializeEph: first index is index of k in group
	std::vector<int> whoseIndex; //which process (mpiWorld index) owns each global k-point index
	inline void initializeEph(const FeynWann::MatrixEph& mat, int o1, int o2)
	{	//Identify destination for results:
		size_t index1 = mat.e1->ik + nkOffset*(o1-oStart); //net index of state 1 in group (may not be mine)
		size_t index2 = mat.e2->ik + nkOffset*o2; //global index of state 2
		double sigmaInv = 1./ePhDelta;
		double deltaPrefac = sqrt(sigmaInv/sqrt(M_PI));
		for(int alpha=0; alpha<fw.nModes; alpha++)
		{	GePhEntry g;
			g.omegaPh = mat.ph->omega[alpha];
			if(g.omegaPh < 1e-6) continue; //avoid zero frequency phonons
			const complex* Gdata = mat.M[alpha].data();
			for(int n2=0; n2<fw.nBands; n2++)
				for(int n1=0; n1<fw.nBands; n1++)
				{	double deltaEbySigma = sigmaInv*(mat.e1->E[n1] - mat.e2->E[n2] - g.omegaPh);
					if(fabs(deltaEbySigma) < 3. and (mat.e1->E[n1] > mat.e2->E[n2]))
					{	SparseEntry s;
						s.i = n1;
						s.j = n2;
						s.val = *(Gdata++) * (deltaPrefac*exp(-0.5*deltaEbySigma*deltaEbySigma)); //apply e-conservation factor
						g.G.push_back(s);
					}
					else Gdata++; //Neglect because far from energy conserving
				}
			if(g.G.size()) GePhGroup[index1][index2].push_back(g);
		}
	}
	struct InitEphParams { Lindblad* lb; size_t o1, o2; };
	static void initializeEph(const FeynWann::MatrixEph& mat, void* params)
	{	InitEphParams& p = *((InitEphParams*)params);
		p.lb->initializeEph(mat, p.o1, p.o2);
	}
	
	void initialize()
	{	state.resize(rho.size());
		Emin = +DBL_MAX;
		Emax = -DBL_MAX;
		
		//Initialize eLoop:
		size_t oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
		logPrintf("\nInitializing electronic quantities: "); logFlush();
		for(size_t o=oStart; o<oStop; o++)
		{	InitEparams params = {this, o};
			fw.eLoop(k0[o], Lindblad::initializeE, &params);
			//Print progress:
			if((o-oStart+1)%oInterval==0) { logPrintf("%d%% ", int(round((o-oStart+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		
		//Synchronize energy range:
		mpiWorld->allReduce(Emin, MPIUtil::ReduceMin);
		mpiWorld->allReduce(Emax, MPIUtil::ReduceMax);
		logPrintf("Electron energy grid from %lg eV to %lg eV with spacing %lg eV.\n", Emin/eV, Emax/eV, dE/eV);
		
		//Initialize ePhLoop (if needed):
		if(ePhEnabled)
		{	logPrintf("Initializing e-ph quantities: "); logFlush();
			size_t noPairsMine = noMine*k0.size();
			size_t oPairInterval = std::max(1, int(round(noPairsMine/50.))); //interval for reporting progress
			GePhGroup.assign(noMine*nkOffset, GePhSet(nkTot)); //temporary storage by group-index k1 and global index k2
			for(size_t o1=oStart; o1<oStop; o1++)
				for(size_t o2=0; o2<k0.size(); o2++)
				{	InitEphParams params = {this, o1, o2};
					fw.ePhLoop(k0[o1], k0[o2], Lindblad::initializeEph, &params);
					//Print progress:
					size_t oPair = (o1-oStart)*k0.size()+o2;
					if((oPair+1)%oPairInterval==0) { logPrintf("%d%% ", int(round((oPair+1)*100./noPairsMine))); logFlush(); }
				}
			logPrintf("done.\n"); logFlush();
			//Redistribute GePhGroup to the appropriate state:
			logPrintf("Redistributing e-ph quantities: "); logFlush();
			size_t nIndex1 = GePhGroup.size();
			size_t index1interval = std::max(1, int(round(nIndex1/20.))); //interval for reporting progress
			int iProc = mpiGroup->iProcess(); //current process in group
			for(size_t o1=oStart; o1<oStop; o1++)
			{	for(int jProc=0; jProc<mpiGroup->nProcesses(); jProc++) //who has the state for index1
					for(int ik=fw.Hw->ikStartProc[jProc]; ik<fw.Hw->ikStartProc[jProc+1]; ik++) //all ik belonging to jProc
					{	size_t index1 = ik + (o1-oStart)*nkOffset; //belongs on jProc, but this is group index
						State* state1 = 0;
						if(jProc == iProc)
						{	state1 = state.data() + (ik-ikStart + (o1-oStart)*nkMine); //this is the local index
							state1->GePh.resize(nkTot);
						}
						//Map out who has which sets of index2:
						std::vector<std::pair<int,int>> nGwhose(nkTot); //number of ePh entries and owner process, per index2
						for(size_t index2=0; index2<nkTot; index2++)
						{	nGwhose[index2].first = int(GePhGroup[index1][index2].size()); //number of ePh entries
							nGwhose[index2].second = iProc; //current process; will be replaced by process that owns the ePh entries on the reduce below
						}
						#ifdef MPI_ENABLED
						if(mpiGroup->nProcesses() > 1)
						{	MPI_Allreduce(MPI_IN_PLACE, nGwhose.data(), nkTot, MPI_2INT, MPI_MAXLOC, mpiGroup->communicator());
						} //else results are already up to date on each process (each owns all the data of that group)
						#endif
						//Transfer to the correct process:
						for(size_t index2=0; index2<nkTot; index2++)
						{	if(not nGwhose[index2].first) continue; //no matrix elements to transfer
							int srcProc = nGwhose[index2].second; //need to transfer from srcProc to jProc
							if(iProc == srcProc)
							{	if(iProc == jProc) //State is local
									state1->GePh[index2] = GePhGroup[index1][index2];
								else //Belongs elsewhere; send over MPI
									sendGePh(GePhGroup[index1][index2], jProc, index2);
							}
							else if(iProc == jProc) //Belongs here, but this is not srcProc, so recv over MPI
								recvGePh(state1->GePh[index2], srcProc, index2);
						}
						//Print progress:
						if((index1+1)%index1interval==0) { logPrintf("%d%% ", int(round((index1+1)*100./nIndex1))); logFlush(); }
					}
			}
			logPrintf("done.\n"); logFlush();
			//Initialize global array of who (in mpiWorld) owns which k-point:
			std::vector<std::pair<int,int>> whoseTemp(nkTot);
			for(size_t o1=oStart; o1<oStop; o1++)
				for(size_t ik=ikStart; ik<ikStop; ik++)
					whoseTemp[ik+o1*nkOffset] = std::make_pair(1, mpiWorld->iProcess());
			#ifdef MPI_ENABLED
			if(mpiWorld->nProcesses() > 1)
			{	MPI_Allreduce(MPI_IN_PLACE, whoseTemp.data(), nkTot, MPI_2INT, MPI_MAXLOC, mpiWorld->communicator());
			} //else results are already up to date on the only process (which necessarily owns all the data)
			#endif
			whoseIndex.resize(nkTot);
			for(size_t index=0; index<nkTot; index++)
				whoseIndex[index] = whoseTemp[index].second; //only keep the process indices from above
		}
		logPrintf("\n");
	}
	
	
	//Calculate probe response at current rho (update this->imEps)
	diagMatrix calcImEps() const
	{	static StopWatch watch("Lindblad::calcImEps");
		size_t nImEps = pol.size() * nomega;
		if(nImEps==0) return diagMatrix(); //no probe specified
		watch.start();
		diagMatrix imEps(nImEps);
		//Collect contributions from each k at this process:
		const matrix* rhoPtr = rho.data();
		const State* sPtr = state.data();
		for(size_t o=oStart; o<oStop; o++)
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	const matrix& rhoCur = *(rhoPtr);
				matrix rhoBar = eye(fw.nBands) - rhoCur; //1-rho
				//Probe response:
				for(int iomega=0; iomega<nomega; iomega++)
				{	double omega = omegaMin + iomega*domega;
					double prefac = (4*M_PI*fw.spinWeight)/(nkTot * fw.Omega * std::pow(std::max(omega, 1./tau), 3));
					//Energy conservation factors for all pair of bands at this frequency:
					std::vector<double> delta(fw.nBands*fw.nBands);
					double* deltaData = delta.data();
					double normFac = sqrt(tau/sqrt(M_PI));
					for(int b2=0; b2<fw.nBands; b2++)
						for(int b1=0; b1<fw.nBands; b1++)
						{	double tauDeltaE = tau*(sPtr->E[b1] - sPtr->E[b2] - omega);
							*(deltaData++) = normFac * exp(-0.5*tauDeltaE*tauDeltaE);
						}
					//Loop over polarizations:
					for(int iPol=0; iPol<int(pol.size()); iPol++)
					{	//Multiply matrix elements with energy conservation:
						matrix P = sPtr->P[iPol];
						eblas_zmuld(P.nData(), delta.data(),1, P.data(),1); //P-
						matrix Pdag = dagger(P); //P+
						//Loop over directions of excitations:
						diagMatrix deltaRhoDiag(fw.nBands);
						for(int s=-1; s<=+1; s+=2)
						{	deltaRhoDiag += diag(rhoBar*P*rhoCur*Pdag - Pdag*rhoBar*P*rhoCur);
							std::swap(P, Pdag); //P- <--> P+
						}
						imEps[iPol*nomega+iomega] += prefac * dot(sPtr->E, deltaRhoDiag);
					}
				}
				//Advance pointers for next k:
				rhoPtr++;
				sPtr++;
			}
		//Accumulate contributions from all processes on head:
		mpiWorld->reduceData(imEps, MPIUtil::ReduceSum);
		watch.stop();
		return imEps;
	}
	
	//Write current imEps to plain-text file:
	void writeImEps(string fname, const diagMatrix& imEps) const
	{	if(mpiWorld->isHead())
		{	ofstream ofs(fname);
			ofs << "#omega[eV]";
			for(int iPol=0; iPol<int(pol.size()); iPol++)
				ofs << " ImEps" << (iPol+1);
			ofs << "\n";
			for(int iomega=0; iomega<nomega; iomega++)
			{	double omega = omegaMin + iomega*domega;
				ofs << omega/eV;
				for(int iPol=0; iPol<int(pol.size()); iPol++)
					ofs << '\t' << imEps[iPol*nomega+iomega];
				ofs << '\n';
			}
		}
	}
	
	//Apply pump using perturbation theory (instantly go from before to after pump, skipping time evolution)
	void applyPump()
	{	static StopWatch watch("Lindblad::applyPump"); 
		if(pumpEvolve) return; //only use this function when perturbing instantly
		watch.start();
		matrix* rhoPtr = rho.data();
		const State* sPtr = state.data();
		//Perturb each k separately:
		for(size_t o=oStart; o<oStop; o++)
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	matrix& rhoCur = *(rhoPtr);
				matrix rhoBar = eye(fw.nBands) - rhoCur; //1-rho
				//Compute and apply perturbation:
				matrix P = sPtr->pumpPD; //P-
				matrix Pdag = dagger(P); //P+
				matrix deltaRho;
				for(int s=-1; s<=+1; s+=2)
				{	deltaRho += rhoBar*P*rhoCur*Pdag - Pdag*rhoBar*P*rhoCur;
					std::swap(P, Pdag); //P- <--> P+
				}
				rhoCur += (M_PI*pumpA0*pumpA0) * (deltaRho + dagger(deltaRho));
				//Advance pointers for next k:
				rhoPtr++;
				sPtr++;
			}
		watch.stop();
	}
	
	//Time evolution operator returning drho/dt
	DM1 compute(double t, const DM1& rho)
	{	static StopWatch watchPump("Lindblad::compute::Pump");
		static StopWatch watchEph("Lindblad::compute::ePh");
		DM1 rhoDot(nkMine*noMine, zeroes(fw.nBands, fw.nBands));
		matrix id = eye(fw.nBands); //identity used below repeatedly
		//Pump contribution:
		if(pumpEvolve)
		{	watchPump.start();
			double prefac = sqrt(M_PI)*pumpA0*pumpA0/pumpTau * exp(-(t*t)/(pumpTau*pumpTau));
			//Each k contributes separately:
			matrix* rhoDotPtr = rhoDot.data();
			const matrix* rhoPtr = rho.data();
			const State* sPtr = state.data();
			for(size_t o=oStart; o<oStop; o++)
				for(size_t ik=ikStart; ik<ikStop; ik++)
				{	const matrix& rhoCur = *(rhoPtr);
					const matrix rhoBar = id - rhoCur; //1-rho
					//Compute and apply perturbation:
					matrix P = sPtr->pumpPD; //P-
					matrix Pdag = dagger(P); //P+
					for(int s=-1; s<=+1; s+=2)
					{	*(rhoDotPtr) += prefac * (rhoBar*P*rhoCur*Pdag - Pdag*rhoBar*P*rhoCur); //+ h.c. added together below
						std::swap(P, Pdag); //P- <--> P+
					}
					//Advance pointers for next k:
					rhoDotPtr++;
					rhoPtr++;
					sPtr++;
				}
			watchPump.stop();
		}
		//E-ph relaxation contribution:
		if(ePhEnabled)
		{	watchEph.start();
			const double prefac = M_PI/nkTot;
			//Loop over second k index by going over all offsets and ik
			int iProc = mpiWorld->iProcess(); //current process
			for(size_t o2=0; o2<k0.size(); o2++)
				for(size_t ik2=0; ik2<nkOffset; ik2++)
				{	size_t index2 = ik2 + o2*nkOffset; //global index2
					int jProc = whoseIndex[index2]; //who owns index2
					//Make current rho2 available on all processes:
					matrix rho2 = (jProc==iProc)
						? rho[ik2-ikStart+(o2-oStart)*nkMine] //local index2, only on process that owns it
						: zeroes(fw.nBands, fw.nBands);
					mpiWorld->bcastData(rho2, jProc);
					const matrix rho2bar = id - rho2;
					matrix rho2dot = zeroes(fw.nBands, fw.nBands); //contributions to remote derivative
					//Loop over rho1 local to each process:
					for(size_t index1=0; index1<rho.size(); index1++) //local index1
					{	matrix& rho1dot = rhoDot[index1];
						const matrix& rho1 = rho[index1];
						//const matrix rho1bar = id - rho1;
						for(const GePhEntry& g: state[index1].GePh[index2])
						{	//Phonon occupation factor:
							/* Old dense implementation with temperature 
							double omegaPhByT = g.omegaPh/T;
							double nPh = omegaPhByT>36 ? 0. : 1./(exp(std::max(1e-3, omegaPhByT)) - 1.); //avoid overflow
							matrix A = dagger(g.G) * rho1 * (prefac*(nPh+1));
							matrix B = g.G * rho2bar;
							matrix C = rho1bar * g.G;
							matrix D = rho2 * dagger(g.G) * (prefac*nPh);
							rho1dot += C*D - B*A; //+ h.c. added together below
							rho2dot += A*B - D*C; //+ h.c. added together below
							*/
							//Sparse implementation currently at T=0:
							rho1dot -= prefac * (rho1 * SMSdag(g.G, rho2bar));
							rho2dot += prefac * (rho2bar * SdagMS(g.G, rho1));
						}
					}
					//Collect remote contributions:
					mpiWorld->allReduceData(rho2dot, MPIUtil::ReduceSum);
					if(jProc==iProc) rhoDot[ik2-ikStart+(o2-oStart)*nkMine] += rho2dot; //local index2, only on process that owns it
				}
			watchEph.stop();
		}
		
		//Add + h.c. (omited everywhere above):
		for(matrix& m: rhoDot)
			m += dagger(m);
		
		if(verbose)
		{	//Report current statistics:
			double rhoDotMax = 0.;
			for(const matrix& m: rhoDot)
				rhoDotMax = std::max(rhoDotMax, m.data()[cblas_izamax(m.nData(), m.data(), 1)].abs());
			double rhoEigMin = +DBL_MAX, rhoEigMax = -DBL_MAX;
			for(const matrix& m: rho)
			{	matrix V; diagMatrix f;
				m.diagonalize(V, f);
				rhoEigMin = std::min(rhoEigMin, f.front());
				rhoEigMax = std::max(rhoEigMax, f.back());
			}
			mpiWorld->reduce(rhoDotMax, MPIUtil::ReduceMax);
			mpiWorld->reduce(rhoEigMax, MPIUtil::ReduceMax);
			mpiWorld->reduce(rhoEigMin, MPIUtil::ReduceMin);
			logPrintf("\n\tComputed at t[fs]: %lg  max(rhoDot): %lg rhoEigRange: [ %lg %lg ] ",
				t/fs, rhoDotMax, rhoEigMin, rhoEigMax); logFlush();
		}
		else logPrintf("(t[fs]: %lg) ", t/fs);
		logFlush();
		return rhoDot;
	}
	
	//Print / dump quantities at each checkpointed step
	void report(double t, const DM1& rho) const
	{	static StopWatch watch("Lindblad::report"); watch.start();
		ostringstream ossID; ossID << stepID;
		//Compute total energy and distributions:
		int nDist = fw.fwp.needSpin ? 4 : 1; //number distribution only, or also spin distribution
		std::vector<Histogram> dist(nDist, Histogram(Emin, dE, Emax));
		const double prefac = fw.spinWeight * (1./nkTot);
		double Etot = 0., dfMax = 0.;
		const matrix* rhoPtr = rho.data();
		const State* sPtr = state.data();
		for(size_t o=oStart; o<oStop; o++)
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	matrix drho = *(rhoPtr) - sPtr->rho0;
				//Energy and distribution:
				const complex* drhoData = drho.data();
				for(int b=0; b<fw.nBands; b++)
				{	double weight = prefac * drhoData->real();
					Etot += weight * sPtr->E[b];
					dfMax = std::max(dfMax, fabs(drhoData->real()));
					dist[0].addEvent(sPtr->E[b], weight);
					drhoData += (fw.nBands+1); //advance to next diagonal entry
				}
				//Spin distribution of available:
				if(fw.fwp.needSpin)
				{	const complex* drhoData = drho.data();
					vector3<const complex*> Sdata; for(int k=0; k<3; k++) Sdata[k] = sPtr->S[k].data();
					for(int b2=0; b2<fw.nBands; b2++)
					{	int i = b2*fw.nBands; //offset into data
						for(int b1=0; b1<=b2; b1++) //use Hermitian symmetry
						{	complex weight = ((b1==b2 ? 1 : 2) * prefac) * drhoData[i];
							//Precalculate histogram position:
							double E = 0.5*(sPtr->E[b1] + sPtr->E[b2]);
							int iEvent; double tEvent;
							if(dist[1].eventPrecalc(E, iEvent, tEvent))
							{	//Collect spin densities:
								for(int k=0; k<3; k++)
									dist[k+1].addEventPrecalc(iEvent, tEvent, (weight * Sdata[k][i]).real());
							}
							//Advance to next entry of Hermitian matrix:
							i++;
						}
					}
				}
				//Advance pointers for next k:
				rhoPtr++;
				sPtr++;
			}
		mpiWorld->reduce(Etot, MPIUtil::ReduceSum);
		mpiWorld->reduce(dfMax, MPIUtil::ReduceMax);
		for(Histogram& h: dist) h.reduce(MPIUtil::ReduceSum);
		if(mpiWorld->isHead())
		{	//Report step ID and energy:
			logPrintf("Integrate: Step: %4d   t[fs]: %6.1lf   Etot[eV]: %.6lf   dfMax: %.3lg\n", stepID, t/fs, Etot/eV, dfMax);
			//Save distribution functions:
			ofstream ofs("dist."+ossID.str());
			ofs << "#E-mu/VBM[eV] n[eV^-1]";
			if(fw.fwp.needSpin)
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
		watch.stop();
		//Probe responses if present:
		diagMatrix imEps = calcImEps();
		if(imEps.size())
			writeImEps("imEps."+ossID.str(), imEps);
		((Lindblad*)this)->stepID++;
	}
};

inline void print(FILE* fp, const vector3<complex>& v, const char* format="%lg ")
{	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].real()); std::fprintf(fp, "] + 1j*");
	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].imag()); std::fprintf(fp, "]\n");
}
inline vector3<complex> normalize(const vector3<complex>& v) { return v * (1./sqrt(v[0].norm() + v[1].norm() + v[2].norm())); }

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Lindblad dynamics in an ab initio Wannier basis");

	//Get the system parameters:
	InputMap inputMap(ip.inputFilename);
	//--- kpoints
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	vector3<int> NkMult;
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	//--- doping / temperature
	const double dmu = inputMap.get("dmu", 0.) * eV; //optional: shift in fermi level from neutral value / VBM in eV (default: 0)
	const double T = inputMap.get("T") * Kelvin; //temperature in Kelvin (ambient phonon T = initial electron T)
	//--- pump
	const string pumpMode = inputMap.getString("pumpMode"); //must be Perturb or Evolve
	if(pumpMode!="Evolve" and pumpMode!="Perturb")
		die("\npumpMode must be 'Evolve' or 'Perturb'\n");
	const double pumpOmega = inputMap.get("pumpOmega") * eV; //pump frequency in eV
	const double pumpA0 = inputMap.get("pumpA0"); //pump pulse amplitude / intensity (Units TBD)
	const double pumpTau = inputMap.get("pumpTau")*fs; //Gaussian pump pulse width in fs
	const vector3<complex> pumpPol = normalize(
		complex(1,0)*inputMap.getVector("pumpPolRe", vector3<>(1.,0.,0.)) +  //Real part of polarization
		complex(0,1)*inputMap.getVector("pumpPolIm", vector3<>(0.,0.,0.)) ); //Imag part of polarization
	//--- probes
	const double omegaMin = inputMap.get("omegaMin") * eV; //start of frequency grid for probe response
	const double omegaMax = inputMap.get("omegaMax") * eV; //end of frequency grid for probe response
	const double domega = inputMap.get("domega") * eV; //frequency resolution for probe calculation
	const double tau = inputMap.get("tau") * fs; //Gaussian probe pulse width in fs
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
	const double dE = inputMap.get("dE") * eV; //energy resolution for distribution functions
	const double dt = inputMap.get("dt") * fs; //time interval between reports
	const double tStop = inputMap.get("tStop") * fs; //stopping time for simulation
	
	const string ePhMode = inputMap.getString("ePhMode"); //must be Off or DiagK (add FullK in future)
	if(ePhMode!="Off" and ePhMode!="DiagK")
		die("\nePhMode must be 'Off' or 'DiagK'\n");
	const bool ePhEnabled = (ePhMode != "Off");
	const double ePhDelta = inputMap.get("ePhDelta") * eV; //energy conservation width for e-ph coupling
	const string verboseMode = inputMap.getString("verbose"); //must be yes or no
	if(verboseMode!="yes" and verboseMode!="no")
		die("\nverboseMode must be 'yes' or 'no'\n");
	const bool verbose = (verboseMode=="yes");
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	logPrintf("dmu = %lg\n", dmu);
	logPrintf("T = %lg\n", T);
	logPrintf("pumpMode = %s\n", pumpMode.c_str());
	logPrintf("pumpOmega = %lg\n", pumpOmega);
	logPrintf("pumpA0 = %lg\n", pumpA0);
	logPrintf("pumpTau = %lg\n", pumpTau);
	logPrintf("pumpPol = "); print(globalLog, pumpPol);
	logPrintf("omegaMin = %lg\n", omegaMin);
	logPrintf("omegaMax = %lg\n", omegaMax);
	logPrintf("domega = %lg\n", domega);
	logPrintf("tau = %lg\n", tau);
	for(int iPol=0; iPol<int(pol.size()); iPol++)
	{	logPrintf("pol%d = ", iPol+1);
		print(globalLog, pol[iPol]);
	}
	logPrintf("dE = %lg\n", dE);
	logPrintf("dt = %lg\n", dt);
	logPrintf("tStop = %lg\n", tStop);
	logPrintf("ePhMode = %s\n", ePhMode.c_str());
	logPrintf("ePhDelta = %lg\n", ePhDelta);
	logPrintf("verbose = %s\n", verboseMode.c_str());
	
	//Initialize FeynWann:
	FeynWannParams fwp;
	fwp.needVelocity = true;
	fwp.needSpin = true;
	fwp.needPhonons = ePhEnabled;
	FeynWann fw(fwp);
	
	//Construct mesh of k-offsets:
	std::vector<vector3<>> k0;
	vector3<int> NkFine;
	for(int iDir=0; iDir<3; iDir++)
	{	if(fw.isTruncated[iDir] && NkMult[iDir]!=1)
		{	logPrintf("Setting NkMult = 1 along truncated direction %d.\n", iDir+1);
			NkMult[iDir] = 1; //no multiplication in truncated directions
		}
		NkFine[iDir] = fw.kfold[iDir] * NkMult[iDir];
	}
	matrix3<> NkFineInv = inv(Diag(vector3<>(NkFine)));
	vector3<int> ikMult;
	for(ikMult[0]=0; ikMult[0]<NkMult[0]; ikMult[0]++)
	for(ikMult[1]=0; ikMult[1]<NkMult[1]; ikMult[1]++)
	for(ikMult[2]=0; ikMult[2]<NkMult[2]; ikMult[2]++)
		k0.push_back(NkFineInv * ikMult);
	logPrintf("Effective interpolated k-mesh dimensions: ");
	NkFine.print(globalLog, " %d ");
	size_t nOffsets = k0.size();
	size_t nKeff = nOffsets * fw.eCountPerOffset();
	logPrintf("Effectively sampled %s: %lu\n", "nKpts", nKeff);
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Initialize sampling parameters:
	int oStart=0, oStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsets, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	
	//Create and initialize lindblad calculator:
	Lindblad lb(fw, k0, oStart, oStop, dmu, T,
		pumpOmega, pumpA0, pumpTau, pumpPol, (pumpMode=="Evolve"),
		omegaMin, omegaMax, domega, tau, pol, dE, ePhEnabled, ePhDelta, verbose);
	lb.initialize();
	logPrintf("Initialization completed successfully at t[s]: %9.2lf\n\n", clock_sec());
	logFlush();
	
	if(pumpMode=="Perturb" and (not ePhEnabled))
	{	//Simple probe-pump-probe with no relaxation:
		lb.report(-dt, lb.rho);
		lb.applyPump();
		lb.report(0., lb.rho);
	}
	else
	{	double tStart = 0.;
		if(pumpMode=="Perturb")
		{	//Do an initial report akin to above and apply the pump:
			lb.report(-dt, lb.rho);
			lb.applyPump();
			tStart = 0.; //integrate will report at t=0 below, before evolving ePh relaxation
		}
		else
		{	//Set start time to a multiple of dt that covers pulse:
			tStart = -dt * ceil(5.*tau/dt);
		}
		//Evolve:
		lb.integrateAdaptive(lb.rho, tStart, tStop, 1e-3, dt);
	}
	
	//Cleanup:
	fw.free();
	FeynWann::finalize();
	return 0;
}
