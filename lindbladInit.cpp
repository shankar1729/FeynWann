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
#include <core/Units.h>
#include "FeynWann.h"
#include "Histogram.h"
#include "InputMap.h"
#include "SparseMatrix.h"

//Reverse iterator for pointers:
template<class T> constexpr std::reverse_iterator<T*> reverse(T* i) { return std::reverse_iterator<T*>(i); }

//Lindblad initialization using FeynWann callback
struct LindbladInit
{	
	FeynWann& fw;
	const std::vector<vector3<>>& k0; //!< k-point offsets
	const vector3<int>& NkFine; //!< effective k-point mesh sampled
	const size_t nkTot; //!< total k-points effectively used in BZ sampling
	
	const double dmuMin, dmuMax, Tmax;
	const double pumpOmegaMax;
	
	const bool ePhEnabled; //!< whether e-ph coupling is enabled
	const double ePhDelta; //!< Gaussian energy conservation width
	
	size_t oStart, oStop; //!< range of offstes handled by this process group
	size_t noMine, oInterval; //!< number of offsets on this process group and reporting interval
	
	
	LindbladInit(FeynWann& fw, const std::vector<vector3<>>& k0, const vector3<int>& NkFine,
		double dmuMin, double dmuMax, double Tmax, double pumpOmegaMax,
		bool ePhEnabled, double ePhDelta)
	: fw(fw), k0(k0), NkFine(NkFine), nkTot(fw.eCountPerOffset()*k0.size()),
		dmuMin(dmuMin), dmuMax(dmuMax), Tmax(Tmax), pumpOmegaMax(pumpOmegaMax),
		ePhEnabled(ePhEnabled), ePhDelta(ePhDelta)
	{
		//Initialize sampling parameters:
		if(mpiGroup->isHead())
			TaskDivision(k0.size(), mpiGroupHead).myRange(oStart, oStop);
		mpiGroup->bcast(oStart);
		mpiGroup->bcast(oStop);
		noMine = oStop-oStart;
		oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	}
	
	//--------- k-point selection -------------
	
	double EvMax, EcMin; //VBM and CBM estimates
	inline void eRange(const FeynWann::StateE& state)
	{	for(const double& E: state.E)
		{	if(E<dmuMin and E>EvMax) EvMax = E;
			if(E>dmuMax and E<EcMin) EcMin = E;
		}
	}
	static void eRange(const FeynWann::StateE& state, void* params)
	{	((LindbladInit*)params)->eRange(state);
	}
	
	double Estart, Estop; //energy range for k selection
	std::vector<vector3<>> k; //selected k-points
	std::vector<double> E; //all band energies for selected k-points
	inline void kSelect(const FeynWann::StateE& state)
	{	bool active = false;
		for(double E: state.E)
			if(E>=Estart and E<=Estop)
			{	active = true;
				break;
			}
		if(active)
		{	k.push_back(state.k);
			E.insert(E.end(), state.E.begin(), state.E.end());
		}
	}
	static void kSelect(const FeynWann::StateE& state, void* params)
	{	((LindbladInit*)params)->kSelect(state);
	}
	void kpointSelect()
	{
		//Determine energy range:
		EvMax = -DBL_MAX;
		EcMin = +DBL_MAX;
		fw.eLoop(vector3<>(), LindbladInit::eRange, this);
		mpiWorld->allReduce(EvMax, MPIUtil::ReduceMax);
		mpiWorld->allReduce(EcMin, MPIUtil::ReduceMin);
		//--- add margins of max phonon energy, energy conservation width and fermiPrime width
		double Emargin =7.*Tmax; //neglect below 10^-3 occupation deviation from equilibrium
		Estart = std::min(EcMin - pumpOmegaMax, EvMax) - Emargin;
		Estop = std::max(EvMax + pumpOmegaMax, EcMin) + Emargin;
		logPrintf("Active energy range: %.3lf to %.3lf eV\n", Estart/eV, Estop/eV);
		
		//Select k-points:
		logPrintf("Scanning k-points with active states: "); logFlush();
		for(size_t o=oStart; o<oStop; o++)
		{	fw.eLoop(k0[o], LindbladInit::kSelect, this);
			//Print progress:
			if((o-oStart+1)%oInterval==0) { logPrintf("%d%% ", int(round((o-oStart+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		
		//Synchronize selected k and E across all processes:
		//--- determine nk on each process and compute cumulative counts
		std::vector<size_t> nkPrev(mpiWorld->nProcesses()+1);
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
		{	size_t nkCur = k.size();
			mpiWorld->bcast(nkCur, jProc); //nkCur = k.size() on jProc in all processes
			nkPrev[jProc+1] = nkPrev[jProc] + nkCur; //cumulative count
		}
		size_t nkSelected = nkPrev.back();
		//--- broadcast k and E:
		{	//Set k and E in position in global arrays:
			std::vector<vector3<>> k(nkSelected);
			std::vector<double> E(nkSelected*fw.nBands);
			std::copy(this->k.begin(), this->k.end(), k.begin()+nkPrev[mpiWorld->iProcess()]);
			std::copy(this->E.begin(), this->E.end(), E.begin()+nkPrev[mpiWorld->iProcess()]*fw.nBands);
			//Broadcast:
			for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
			{	size_t ikStart = nkPrev[jProc], nk = nkPrev[jProc+1]-ikStart;
				mpiWorld->bcast(k.data()+ikStart, nk, jProc);
				mpiWorld->bcast(E.data()+ikStart*fw.nBands, nk*fw.nBands, jProc);
			}
			//Store to class variables:
			std::swap(k, this->k);
			std::swap(E, this->E);
		}
		logPrintf("Found %lu k-points with active states from %lu total k-points (%.0fx reduction)\n\n",
			nkSelected, nkTot, round(nkTot*1./nkSelected));
	}
	
	//--------- k-pair selection -------------
	std::vector<std::vector<size_t>> kpartners; //list of e-ph coupled k2 for each k1
	std::vector<std::pair<size_t,size_t>> kpairs; //pairs of k1 and k2
	std::map<size_t,size_t> kIndexMap; //map from k-point mesh index to index in selected set
	inline size_t kIndex(vector3<> k)
	{	size_t index=0;
		for(int iDir=0; iDir<3; iDir++)
		{	double ki = k[iDir] - floor(k[iDir]); //wrapped to [0,1)
			index = (size_t)round(NkFine[iDir]*(index+ki));
		}
		return index;
	}
	inline void kpSelect(const FeynWann::StatePh& state)
	{	const double omegaPhCut = 1e-6;
		//Find pairs of momentum conserving electron states with this q:
		for(size_t ik1=0; ik1<k.size(); ik1++)
		{	const vector3<>& k1 = k[ik1];
			vector3<> k2 = k1 - state.q; //momentum conservation
			const std::map<size_t,size_t>::iterator iter = kIndexMap.find(kIndex(k2));
			if(iter != kIndexMap.end())
			{	size_t ik2 = iter->second;
				//Check energy conservation for pair of bands within active range:
				//--- determine ranges of all E1 and E2:
				const double *E1begin = E.data()+ik1*fw.nBands, *E1end = E1begin+fw.nBands;
				const double *E2begin = E.data()+ik2*fw.nBands, *E2end = E2begin+fw.nBands;
				//--- narrow to active energy ranges:
				E1begin = std::lower_bound(E1begin, E1end, Estart);
				E1end = &(*std::lower_bound(reverse(E1end), reverse(E1begin), Estop, std::greater<double>()))+1;
				E2begin = std::lower_bound(E2begin, E2end, Estart);
				E2end = &(*std::lower_bound(reverse(E2end), reverse(E2begin), Estop, std::greater<double>()))+1;
				//--- check energy ranges:
				bool Econserve = false;
				for(const double* E1=E1begin; E1<E1end; E1++) //E1 in active range
				{	for(const double* E2=E2begin; E2<E2end; E2++) //E2 in active range
					{	for(const double omegaPh: state.omega) if(omegaPh>omegaPhCut) //loop over non-zero phonon frequencies
						{	double deltaE = (*E1) - (*E2) - omegaPh; //energy conservation violation
							if(fabs(deltaE) < 4*ePhDelta) //else negligible at the 10^-3 level for a Gaussian
							{	Econserve = true;
								break;
							}
						}
						if(Econserve) break;
					}
					if(Econserve) break;
				}
				if(Econserve) kpairs.push_back(std::make_pair(ik1,ik2));
			}
		}
	}
	static void kpSelect(const FeynWann::StatePh& state, void* params)
	{	((LindbladInit*)params)->kpSelect(state);
	}
	void kpairSelect()
	{	
		//Initialize kIndexMap for searching selected k-points:
		for(size_t ik=0; ik<k.size(); ik++)
			kIndexMap[kIndex(k[ik])] = ik;
		
		//Find momentum-conserving k-pairs for which energy conservation is also possible for some bands:
		logPrintf("Scanning k-pairs with e-ph coupling: "); logFlush();
		for(size_t o=oStart; o<oStop; o++)
		{	fw.phLoop(k0[o], LindbladInit::kpSelect, this);
			//Print progress:
			if((o-oStart+1)%oInterval==0) { logPrintf("%d%% ", int(round((o-oStart+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		
		//Synchronize selected kpairs across all processes:
		//--- determine nk on each process and compute cumulative counts
		std::vector<size_t> nkpPrev(mpiWorld->nProcesses()+1);
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
		{	size_t nkpCur = kpairs.size();
			mpiWorld->bcast(nkpCur, jProc); //nkCur = k.size() on jProc in all processes
			nkpPrev[jProc+1] = nkpPrev[jProc] + nkpCur; //cumulative count
		}
		size_t nkpairs = nkpPrev.back();
		//--- broadcast k and E:
		{	//Set k and E in position in global arrays:
			std::vector<std::pair<size_t,size_t>> kpairs(nkpairs);
			std::copy(this->kpairs.begin(), this->kpairs.end(), kpairs.begin()+nkpPrev[mpiWorld->iProcess()]);
			//Broadcast:
			for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
			{	size_t ikpStart = nkpPrev[jProc], nkp = nkpPrev[jProc+1]-ikpStart;
				mpiWorld->bcast(((size_t*)kpairs.data())+ikpStart*2, nkp*2, jProc);
			}
			//Store to class variables:
			std::swap(kpairs, this->kpairs);
		}
		//--- report:
		size_t nkpairsTot = k.size()*k.size();
		logPrintf("Found %lu k-pairs with e-ph coupling from %lu total pairs of selected k-points (%.0fx reduction)\n",
			nkpairs, nkpairsTot, round(nkpairsTot*1./nkpairs));
		//--- initialize kpartners (list of k2 by k1):
		kpartners.resize(k.size());
		for(auto kpair: kpairs)
			kpartners[kpair.first].push_back(kpair.second);
		size_t nPartnersMin = k.size(), nPartnersMax = 0;
		for(std::vector<size_t>& kp: kpartners)
		{	std::sort(kp.begin(), kp.end()); //sort k2 within each k1 array
			const size_t& nPartners = kp.size();
			if(nPartners < nPartnersMin) nPartnersMin = nPartners;
			if(nPartners > nPartnersMax) nPartnersMax = nPartners;
		}	
		logPrintf("Number of partners per k-point:  min: %lu  max: %lu  mean: %.1lf\n\n", nPartnersMin, nPartnersMax, nkpairs*1./k.size());
	}

	//--------- Initialize -------------
	/*
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
		std::vector<matrix> P; //momentum matrix elements
		std::vector<matrix> S; //spin matrix elements (if available)
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
		//--- Momentum matrix elements
		s.P.assign(stateE.v, stateE.v+3);
		//--- Spin matrix elements
		if(fw.fwp.needSpin)
			s.S.assign(stateE.S, stateE.S+3);
	}
	struct InitEparams { LindbladInit* lb; size_t o; };
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
	struct InitEphParams { LindbladInit* lb; size_t o1, o2; };
	static void initializeEph(const FeynWann::MatrixEph& mat, void* params)
	{	InitEphParams& p = *((InitEphParams*)params);
		p.lb->initializeEph(mat, p.o1, p.o2);
	}
	
	void initialize()
	{	Emin = +DBL_MAX;
		Emax = -DBL_MAX;
		
		//Initialize eLoop:
		size_t oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
		logPrintf("\nInitializing electronic quantities: "); logFlush();
		for(size_t o=oStart; o<oStop; o++)
		{	InitEparams params = {this, o};
			fw.eLoop(k0[o], LindbladInit::initializeE, &params);
			//Print progress:
			if((o-oStart+1)%oInterval==0) { logPrintf("%d%% ", int(round((o-oStart+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		
		//Synchronize energy range:
		mpiWorld->allReduce(Emin, MPIUtil::ReduceMin);
		mpiWorld->allReduce(Emax, MPIUtil::ReduceMax);
		logPrintf("Electron energy range: %lg eV to %lg eV.\n", Emin/eV, Emax/eV);
		
		//Initialize ePhLoop (if needed):
		if(ePhEnabled)
		{	logPrintf("Initializing e-ph quantities: "); logFlush();
			size_t noPairsMine = noMine*k0.size();
			size_t oPairInterval = std::max(1, int(round(noPairsMine/50.))); //interval for reporting progress
			GePhGroup.assign(noMine*nkOffset, GePhSet(nkTot)); //temporary storage by group-index k1 and global index k2
			for(size_t o1=oStart; o1<oStop; o1++)
				for(size_t o2=0; o2<k0.size(); o2++)
				{	InitEphParams params = {this, o1, o2};
					fw.ePhLoop(k0[o1], k0[o2], LindbladInit::initializeEph, &params);
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
	}*/
};

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Initialize sparse matrices for Lindblad dynamics");

	//Get the system parameters:
	InputMap inputMap(ip.inputFilename);
	//--- kpoints
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	vector3<int> NkMult;
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	//--- doping / temperature
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional: lowest shift in fermi level from neutral value / VBM in eV (default: 0)
	const double dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional: highest shift in fermi level from neutral value / VBM in eV (default: 0)
	const double Tmax = inputMap.get("Tmax") * Kelvin; //maximum temperature in Kelvin (ambient phonon T = initial electron T)
	//--- pump
	const double pumpOmegaMax = inputMap.get("pumpOmegaMax") * eV; //maximum pump frequency in eV
	const string ePhMode = inputMap.getString("ePhMode"); //must be Off or DiagK (add FullK in future)
	const bool ePhEnabled = (ePhMode != "Off");
	const double ePhDelta = inputMap.get("ePhDelta") * eV; //energy conservation width for e-ph coupling
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("Tmax = %lg\n", Tmax);
	logPrintf("pumpOmegaMax = %lg\n", pumpOmegaMax);
	logPrintf("ePhMode = %s\n", ePhMode.c_str());
	logPrintf("ePhDelta = %lg\n", ePhDelta);
	
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
	
	//Create and initialize lindblad calculator:
	LindbladInit lb(fw, k0, NkFine, dmuMin, dmuMax, Tmax, pumpOmegaMax, ePhEnabled, ePhDelta);
	
	//First pass (e only): select k-points
	lb.kpointSelect();
	
	//Second pass (ph only): select k pairs
	lb.kpairSelect();
	
	//Final pass: output electronic and e-ph quantities
	
	//Cleanup:
	fw.free();
	FeynWann::finalize();
	return 0;
}
