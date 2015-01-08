#include <core/Util.h>
#include <core/Units.h>
#include <core/Thread.h>
#include "BandStruct.h"
#include "InputMap.h"

int main(int argc, char** argv)
{	int nPhysicalCores = nProcsAvailable; //before overriden by initSystem()
	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "LInearized-Boltzmann calculation of resistivity", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n"); logFlush();

	//Initialize Wannier bandstructure:
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE");

	//Compile list of k-points and bands within fermi level:
	vector3<int> Nk(1,1,1); Nk *= 16; //32;
	int NkProd = Nk[0]*Nk[1]*Nk[2];
	int ik0start = (Nk[0] * mpiUtil->iProcess()) / mpiUtil->nProcesses();
	int ik0stop = (Nk[0] * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	vector3<int> ikStride(Nk[1]*Nk[2], Nk[2], 1);
	struct State
	{	vector3<int> ik; //integer kmesh coordinates
		int b; //band index
		double e, f, fPrime; //energy, fermi filling, and df/de
		vector3<> v; //band velocity
	};
	std::vector<State> statesCur; //states from current process
	State state;
	matrix3<> invDiagNk = inv(Diag(vector3<>(Nk)));
	for(state.ik[0]=ik0start; state.ik[0]<ik0stop; state.ik[0]++)
	for(state.ik[1]=0; state.ik[1]<Nk[1]; state.ik[1]++)
	for(state.ik[2]=0; state.ik[2]<Nk[2]; state.ik[2]++)
	{	vector3<> k = invDiagNk * state.ik;
		diagMatrix E = bs.getStates(k);
		//First check if this k is being used:
		bool need_k = false;
		for(state.b=0; state.b<E.nRows(); state.b++)
			if(fabs(E[state.b]) < 10*T)
			{	need_k = true;
				break;
			}
		if(!need_k) continue;
		//Calculate velocity and add relevant states:
		std::vector< vector3<> > V = bs.getVelocity(k, R);
		for(state.b=0; state.b<E.nRows(); state.b++)
			if(fabs(E[state.b]) < 10*T)
			{	state.e = E[state.b];
				state.f = 1./(1 + exp(state.e/T));
				state.fPrime = -1/(T*std::pow(2*cosh(state.e/(2*T)),2));
				state.v = V[state.b];
				statesCur.push_back(state);
			}
	}
	//--- synchronize states:
	int nStatesCur = statesCur.size();
	int nStates = nStatesCur; mpiUtil->allReduce(nStates, MPIUtil::ReduceSum);
	std::vector<State> states(nStates);
	char* statePtr = (char*)(&states[0]);
	for(int jProcess=0; jProcess<mpiUtil->nProcesses(); jProcess++)
	{	int nStates_j = nStatesCur; mpiUtil->bcast(nStates_j, jProcess);
		size_t nBytes_j = nStates_j * sizeof(State);
		if(jProcess==mpiUtil->iProcess())
			memcpy(statePtr, statesCur.data(), nBytes_j);
		mpiUtil->bcast(statePtr, nBytes_j, jProcess);
		statePtr += nBytes_j;
	}
	logPrintf("nStates = %d\n", nStates);

	//Print mean Fermi velocity (debug):
	double vFsqSum = 0.; double weightSum = 0.;
	for(const State& state: states)
	{	vFsqSum += state.v.length_squared() * state.fPrime;
		weightSum += state.fPrime;
	}
	double vF = sqrt(vFsqSum / weightSum);
	logPrintf("vF = %lg\n", vF); logFlush();
	
	//Map k-points to state indices:
	std::map<vector3<int>, std::vector<int> > stateMap;
	for(int iState=0; iState<nStates; iState++)
		stateMap[states[iState].ik].push_back(iState);
	
	//Get list of unique k-points:
	std::vector< vector3<int> > ikList;
	for(auto entry: stateMap)
		ikList.push_back(entry.first);
	
	//Calculate phonon frequencies for all integer k-mesh displacements:
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	diagMatrix omegaPhMesh(NkProd*nModes, 0.);
	for(state.ik[0]=ik0start; state.ik[0]<ik0stop; state.ik[0]++)
	for(state.ik[1]=0; state.ik[1]<Nk[1]; state.ik[1]++)
	for(state.ik[2]=0; state.ik[2]<Nk[2]; state.ik[2]++)
	{	int offset = dot(state.ik, ikStride)*nModes;
		omegaPhMesh.set(offset, offset+nModes, bs.getPhononModes(invDiagNk * state.ik));
	}
	omegaPhMesh.allReduce(MPIUtil::ReduceSum);
	
	//Generate list of k-point pairs with relevant coupling:
	double weightCut = 1e-6;
	double EconserveExpFac = -0.5/(T*T), EconservePrefac = 1./(sqrt(2*M_PI)*T); //energy conserving Gaussian parameters
	std::vector< std::vector<int> > kpairs(ikList.size());
	int nPairs = 0;
	for(int kIndex1=0; kIndex1<int(ikList.size()); kIndex1++)
	for(int kIndex2=0; kIndex2<kIndex1; kIndex2++) //ignore same-k contributions (zero due to translational invariance)
	{	vector3<int> ik1 = ikList[kIndex1];
		vector3<int> ik2 = ikList[kIndex2];
		vector3<int> ikDiff = ik1 - ik2;
		for(int dir=0; dir<3; dir++)
			ikDiff[dir] = positiveRemainder(ikDiff[dir], Nk[dir]);
		int offsetPh = dot(ikDiff, ikStride)*nModes;
		diagMatrix omegaPhCur = omegaPhMesh(offsetPh, offsetPh+nModes);
		bool needPair = false;
		for(int iState1: stateMap[ik1])
		for(int iState2: stateMap[ik2])
		{	double fWeight = T * std::max(fabs(states[iState1].fPrime), fabs(states[iState2].fPrime)); //effective weight due to occupations
			for(double omegaPh: omegaPhCur)
			for(int ae=-1; ae<=+1; ae+=2)
			{	double EconserveWeight = EconservePrefac * exp(EconserveExpFac * std::pow(states[iState2].e - states[iState1].e - ae*omegaPh,2));
				if(fWeight * EconserveWeight > weightCut)
					needPair = true;
			}
		}
		if(needPair)
		{	nPairs++;
			//Try to even out the distribution of the first index in stored pairs:
			if(kpairs[kIndex1].size() < kpairs[kIndex2].size())
				kpairs[kIndex1].push_back(kIndex2);
			else
				kpairs[kIndex2].push_back(kIndex1);
		}
	}
	int nPairsTot = int((ikList.size()*(ikList.size()+1))/2);
	logPrintf("%d kpoints, %d of %d pairs (%d%%) relevant\n",
		int(ikList.size()), nPairs, nPairsTot, int((100*nPairs)/nPairsTot) );
	logFlush();
	//--- generate blocks of kpairSets with block size constraint:
	const int maxBlockSize = 64;
	struct KpairSet { int kIndex1; std::vector<int> kIndex2set; };
	std::vector<KpairSet> kpairSets;
	kpairSets.reserve(ikList.size() * ceildiv(ceildiv(nPairs, int(ikList.size())), maxBlockSize));
	for(int kIndex1=0; kIndex1<int(ikList.size()); kIndex1++)
	{	int nBlocks = ceildiv(int(kpairs[kIndex1].size()), maxBlockSize);
		for(int iBlock=0; iBlock<nBlocks; iBlock++)
		{	size_t blockStart = (kpairs[kIndex1].size() * iBlock)/nBlocks;
			size_t blockStop = (kpairs[kIndex1].size() * (iBlock+1))/nBlocks;
			KpairSet kpairSet;
			kpairSet.kIndex1 = kIndex1;
			kpairSet.kIndex2set.assign(kpairs[kIndex1].begin()+blockStart, kpairs[kIndex1].begin()+blockStop);
			kpairSets.push_back(kpairSet);
		}
	}
	
	//Fill matrix 'S' in the derivation:
	matrix S = zeroes(nStates, nStates);
	double prefacS = M_PI/NkProd;
	double tauInvNum = 0.;
	int pairSetStart = (kpairSets.size() * mpiUtil->iProcess()) / mpiUtil->nProcesses();
	int pairSetStop = (kpairSets.size() * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	for(int pairSet=pairSetStart; pairSet<pairSetStop; pairSet++)
	{	const KpairSet& kpairSet = kpairSets[pairSet];
		int kIndex1 = kpairSet.kIndex1;
		//Get e-ph elements for all pairs in set together:
		std::vector< std::vector<matrix> > MePhArr(kpairSet.kIndex2set.size());
		{	vector3<> k1 = invDiagNk * ikList[kIndex1];
			std::vector< vector3<> > k2arr;
			k2arr.reserve(kpairSet.kIndex2set.size());
			for(int kIndex2: kpairSet.kIndex2set)
				k2arr.push_back(invDiagNk * ikList[kIndex2]);
			bs.setPhononMatElemArray(k1, k2arr, MePhArr.data());
		}
		//Now process one pair at a time:
		for(size_t iBlockEntry=0; iBlockEntry<kpairSet.kIndex2set.size(); iBlockEntry++)
		{	int kIndex2 = kpairSet.kIndex2set[iBlockEntry];
			vector3<int> ik1 = ikList[kIndex1];
			vector3<int> ik2 = ikList[kIndex2];
			vector3<> k1 = invDiagNk * ik1;
			vector3<> k2 = invDiagNk * ik2;
			diagMatrix omegaPh = bs.getPhononModes(k1-k2);
			std::vector<matrix>& MePh = MePhArr[iBlockEntry];
			for(int swapped=0; swapped<2; swapped++)
			{	for(int iState1: stateMap[ik1])
				for(int iState2: stateMap[ik2])
				{	double Scur = 0.;
					const State& state1 = states[iState1];
					const State& state2 = states[iState2];
					for(int alpha=0; alpha<omegaPh.nRows(); alpha ++)
					{	double gk = 1./(exp(omegaPh[alpha]/T) - 1.); //Bose factor
						double Msq_by_omega = MePh[alpha](state1.b,state2.b).norm() / omegaPh[alpha];
						for(int ae=-1; ae<=+1; ae+=2)
						{	double delta = EconservePrefac * exp(EconserveExpFac * std::pow(state2.e - state1.e - ae*omegaPh[alpha],2));
							Scur += prefacS * (gk+0.5 + ae*(0.5-state1.f)) * delta * Msq_by_omega;
							tauInvNum += prefacS * state1.fPrime * (gk+0.5 + ae*0.5) * delta * Msq_by_omega;
						}
					}
					S.set(iState1,iState2, Scur);
				}
				if(!swapped) //need to do this only after first time (at most 2 passes through loop)
				{	std::swap(ik1, ik2);
					std::swap(k1, k2);
					for(matrix& M: MePh) M = dagger(M);
				}
			}
		}
	}
	S.allReduce(MPIUtil::ReduceSum);
	mpiUtil->allReduce(tauInvNum, MPIUtil::ReduceSum);
	
	//Do the final processing on head alone (but-multithreaded)
	if(mpiUtil->isHead())
	{	//Make sure head can use all cores of machine (other processes ons ame node are now done)
		nProcsAvailable = nPhysicalCores;
		resumeOperatorThreading();
		
		//Construct matrix 'B' in the derivation:
		matrix ones(1,nStates);
		for(int iState=0; iState<nStates; iState++)
			ones.set(0,iState, 1.);
		matrix Ssum = ones * S; // = Sum_l S_{l,i} in derivation
		diagMatrix DiagSsum(nStates);
		for(int iState=0; iState<nStates; iState++)
			DiagSsum[iState] = Ssum(0,iState).real();
		matrix B = S - DiagSsum;
		
		//Invert B safely (handling its null space properly):
		matrix invB;
		{	matrix BU, BVdag; diagMatrix BS;
			B.svd(BU, BS, BVdag);
			diagMatrix BSinv(BS); for(double& s: BSinv) s = (s<1e-6*BS[0] ? 0. : 1./s);
			invB = dagger(BVdag) * BSinv * dagger(BU);
		}
		
		//Construct velocity and fPrime matrices:
		matrix V(nStates, 3); diagMatrix fPrime(nStates);
		for(int iState=0; iState<nStates; iState++)
		{	const State& state = states[iState];
			for(int dir=0; dir<3; dir++)
				V.set(iState, dir, state.v[dir]);
			fPrime[iState] = state.fPrime;
		}
		
		//Calculate conductivity tensor using Boltzmann equation:
		matrix sigmaMat = (spinWeight/(NkProd*fabs(det(R)))) * dagger(V) * invB * (fPrime * V);
		logPrintf("\nConductivity tensor [atomic units]:\n");
		sigmaMat.print_real(globalLog, " %12.5le ");
		logPrintf("\n");
		double sigma = (1./3) * trace(sigmaMat).real();
		
		//Report resistivity:
		const double invSeconds = 2.418884326505e-17;
		const double Coulomb = Joule/eV;
		const double Volt = Joule/Coulomb;
		const double Ampere = Coulomb*invSeconds;
		const double Ohm = Volt/Ampere;
		const double fs = 1e-15/invSeconds;
		
		// Calculate Resistivity
		double rho = 1./sigma;
		logPrintf("Resistivity = %lg ohm-m\n", rho/(Ohm*meter));
		
		//Comparison with the simple estimate:
		double Tt = (-spinWeight/(3.*NkProd)) * trace(dagger(V) * fPrime * V).real();
		double Gamma = (spinWeight/(3.*NkProd)) * trace(dagger(V) * B * (fPrime * V)).real();
		double rhoSimple = fabs(det(R))*Gamma/(Tt*Tt);
		logPrintf("T = %lg\n", Tt);
		logPrintf("Gamma = %lg\n", Gamma);
		logPrintf("Resistivity(simple) = %lg ohm-m\n", rhoSimple/(Ohm*meter));
		
		//Mean lifetime:
		double tauInv = tauInvNum / weightSum;
		logPrintf("tau = %lg fs\n", (1./tauInv)/fs);
	}
	
	finalizeSystem();
}
