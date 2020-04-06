#include "LindbladFile.h"
#include "InputMap.h"
#include <core/Units.h>
#include <deque>

struct TripletMatrix
{	//Triplet format:
	const int nRows, nCols;
	struct Entry
	{	int i,j; double Mij; 
		Entry(int i, int j, double Mij) : i(i), j(j), Mij(Mij) {}
	};
	std::vector<Entry> entries; //MPI divided; each process has a subset
	
	TripletMatrix(int nRows, int nCols, int nNZestimate=0) : nRows(nRows), nCols(nCols)
	{	if(nNZestimate) entries.reserve(nNZestimate / mpiWorld->nProcesses());
	}
	
	//Matrix multiply:
	matrix operator*(const matrix& v) const
	{	static StopWatch watch("TripletMatrix::operator*"); watch.start();
		matrix out = zeroes(nRows, v.nCols());
		complex* outData = out.data(); const complex* vData = v.data();
		for(const Entry& entry: entries)
			for(int col=0; col<v.nCols(); col++)
				outData[out.index(entry.i, col)] += entry.Mij * vData[v.index(entry.j, col)];
		mpiWorld->allReduceData(out, MPIUtil::ReduceSum);
		watch.stop();
		return out;
	}
	
	//Calculate Sum_i Mij:
	diagMatrix DiagSumRows() const
	{	diagMatrix out(nCols, 0.);
		for(const Entry& entry: entries)
			out[entry.j] += entry.Mij;
		mpiWorld->allReduceData(out, MPIUtil::ReduceSum);
		return out;
	}
	
	complex safe_dot(const matrix& A, const matrix& B) const
	{	complex result = trace(dagger(A) * B);
		mpiWorld->bcast(&result.real(), 2);
		return result;
	}
	
	double safe_nrm2(const matrix& A) const
	{	return sqrt(safe_dot(A,A).real());
	}
	
	//Solve A x = b iteratively using something like GMRES/DIIS (pass in an initial guess for x):
	void applyInverse(const matrix& b, matrix& x) const
	{	const int nIterations = 100;
		const int maxHistory = 10;
		const double threshold = 1e-5;
		std::deque<matrix> xPrev, rPrev;
		matrix overlap(maxHistory, maxHistory);
		matrix r = (*this) * x - b; //initial residual
		double rNormThresh = threshold * safe_nrm2(r);
		logPrintf("\n");
		for(int iter=0; iter<=nIterations; iter++)
		{	//Truncate history if necessary:
			if((int)xPrev.size() >= maxHistory)
			{	size_t ndim = xPrev.size();
				if(ndim>1) overlap.set(0,ndim-1, 0,ndim-1, overlap(1,ndim, 1,ndim));
				xPrev.pop_front();
				rPrev.pop_front();
			}
			
			//Update history and report:
			rPrev.push_back(r);
			xPrev.push_back(x);
			double rNorm = safe_nrm2(r);
			logPrintf("DIIS: Cycle: %2i  |Residual|: %.3e\n", iter, rNorm);
			if(iter==nIterations) { logPrintf("DIIS: Convergence threshold not reached in %d iterations.\n", iter); break; }
			if(rNorm < rNormThresh) { logPrintf("DIIS: Converged: |Residual| < %lg |Initial residual|.\n", threshold); break; }
			
			//DIIS mixing:
			//--- Update the overlap matrix
			size_t ndim = xPrev.size();
			for(size_t j=0; j<ndim; j++)
			{	complex thisOverlap = safe_dot(rPrev[j], r);
				overlap.set(j, ndim-1, thisOverlap);
				overlap.set(ndim-1, j, thisOverlap.conj());
			}
			//--- reset if singular
			bool singular = false;
			if(ndim > 1)
			{	matrix U, Vdag; diagMatrix S;
				matrix(overlap(0,ndim, 0,ndim)).svd(U, S, Vdag);
				if(S.back() < S.front()*threshold)
				{	logPrintf("DIIS: Singularity in overlap, resetting history.\n");
					singular = true;
					xPrev.assign(1, x);
					rPrev.assign(1, r);
					overlap.set(0,0, overlap(ndim-1,ndim-1));
				}
			}
			if(!singular)
			{	//--- Invert the residual overlap matrix to get the minimum of residual
				matrix cOverlap(ndim+1, ndim+1); //Add row and column to enforce normalization constraint
				cOverlap.set(0, ndim, 0, ndim, overlap(0,ndim, 0,ndim));
				for(size_t j=0; j<ndim; j++)
				{	cOverlap.set(j, ndim, 1);
					cOverlap.set(ndim, j, 1);
				}
				cOverlap.set(ndim, ndim, 0);
				matrix cOverlap_inv = inv(cOverlap);
				//---- find best x and corresponding r in current subspace:
				x.zero();
				r.zero();
				for(size_t j=0; j<ndim; j++)
				{	complex alpha = cOverlap_inv(j, ndim);
					x +=  alpha * xPrev[j];
					r += alpha * rPrev[j];
				}
			}
			
			//Expand subspace:
			matrix d = r; //search direction
			double r_r = safe_dot(r,r).real();
			while(true)
			{	matrix Ad = (*this) * d;
				double Ad_Ad = safe_dot(Ad,Ad).real();
				complex Ad_r = safe_dot(Ad,r);
				if(Ad_r.abs() < threshold * sqrt(Ad_Ad * r_r))
				{	//Need to try a new search direction (try random):
					logPrintf("DIIS: Singularity in subspace expansion, randomizing search direction.\n");
					if(mpiWorld->isHead())
						randomize(d);
					mpiWorld->bcastData(d);
					continue;
				}
				complex alpha = -Ad_r / Ad_Ad;
				x += alpha * d;
				r += alpha * Ad;
				break;
			}
		}
	}
};


int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Linearized-Boltzmann calculation of resistivity");

	//Read input file:
	InputMap inputMap(ip.inputFilename);
	const double T = inputMap.get("T") * Kelvin; //must be less than Tmax specified while generating inFile
	const double dmu = inputMap.get("dmu", 0.) * eV; //must be within [dmuMin,dmuMax] specified while generating inFile
	const string inFile = inputMap.has("inFile") ? inputMap.getString("inFile") : "ldbd.dat"; //input file name
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("T = %lg\n", T);
	logPrintf("dmu = %lg\n", dmu);
	logPrintf("inFile = %s\n", inFile.c_str());
	logPrintf("\n");
	
	//Read inFile generated by lindbladInit:
	logPrintf("Reading '%s': ", inFile.c_str()); logFlush();
	//--- Read header and check parameters:
	MPIUtil::File fp;
	mpiWorld->fopenRead(fp, inFile.c_str());
	LindbladFile::Header h; h.read(fp, mpiWorld);
	if(dmu<h.dmuMin or dmu>h.dmuMax)
	{	FeynWann::finalize();
		die("dmu = %lg eV is out of range [ %lg , %lg ] eV specified in lindbladInit.\n", dmu/eV, h.dmuMin/eV, h.dmuMax/eV);
	}
	if(T > h.Tmax)
	{	FeynWann::finalize();
		die("T = %lg K is larger than Tmax = %lg K specified in lindbladInit.\n", T/Kelvin, h.Tmax/Kelvin);
	}
	double Omega = fabs(det(h.R)); //unit cell volume
	if(not h.ePhEnabled)
	{	FeynWann::finalize();
		die("resistivityBTE requires lindbladInit to be run with e-ph enabled.\n");
	}
	std::vector<size_t> byteOffsets(h.nk);
	mpiWorld->freadData(byteOffsets, fp);
	//--- Read energies and matrix elements divided by k-points:
	TaskDivision kDivision(h.nk, mpiWorld);
	size_t ikStart, ikStop;
	kDivision.myRange(ikStart, ikStop);
	size_t nkMine = ikStop-ikStart;
	size_t ikInterval = std::max(1, int(round(nkMine/50.))); //interval for reporting progress
	std::vector<LindbladFile::Kpoint> kpoint(nkMine);
	std::vector<int> nInnerAll(h.nk); //number of active bands at each k
	mpiWorld->fseek(fp, byteOffsets[ikStart], SEEK_SET);
	for(size_t ik=ikStart; ik<ikStop; ik++)
	{	LindbladFile::Kpoint& kp = kpoint[ik-ikStart];
		kp.read(fp, mpiWorld, h);
		nInnerAll[ik] = kp.nInner;
		//Print progress:
		if((ik-ikStart+1)%ikInterval==0) { logPrintf("%d%% ", int(round((ik-ikStart+1)*100./nkMine))); logFlush(); }
	}
	mpiWorld->fclose(fp);
	logPrintf("done.\n"); logFlush();
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");

	//Determine total number of bands by k on all processes:
	for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
		mpiWorld->bcast(nInnerAll.data()+kDivision.start(jProc),
			kDivision.stop(jProc)-kDivision.start(jProc), jProc);
	std::vector<size_t> stateOffset(h.nk+1); //global index of first active band within each k
	for(size_t ik=0; ik<h.nk; ik++)
		stateOffset[ik+1] = stateOffset[ik] + nInnerAll[ik];
	size_t stateOffsetMine = stateOffset[ikStart]; //offset of states stored in current process
	size_t nStatesMine = stateOffset[ikStop] - stateOffsetMine; //number of states in current process
	size_t nStatesTot = stateOffset.back();
	
	//Separate energies by band:
	struct State
	{	double e, f, mfPrime; //energy, fermi filling, and -df/de
		vector3<> v; //band velocity
	};
	std::vector<State> state(nStatesMine);
	std::vector<double> Eall(nStatesTot); //energies for all processes
	const double invT = 1./T;
	double vFsqSum = 0., weightSum = 0.;
	State* s = state.data();
	size_t nNZbound = 0; //uper bound number of non-zero e-ph matrix elements
	for(size_t ik=ikStart; ik<ikStop; ik++)
	{	const LindbladFile::Kpoint& kp = kpoint[ik-ikStart];
		for(int b=0; b<kp.nInner; b++)
		{	s->e = kp.E[kp.innerStart+b];
			Eall[stateOffset[ik]+b] = s->e;
			double EminusMuByT = (s->e - dmu)*invT, fbar;
			fermi(EminusMuByT, s->f, fbar);
			s->mfPrime = invT * s->f * fbar;
			for(int iDir=0; iDir<3; iDir++)
				s->v[iDir] = kp.P[iDir](b, kp.innerStart+b).real();
			//Fermi surface sums:
			vFsqSum += s->mfPrime * s->v.length_squared();
			weightSum += s->mfPrime;
			//e-ph matrix element count:
			size_t jk = -1;
			for(const LindbladFile::GePhEntry& g: kp.GePh)
			{	if(jk != g.jk) nNZbound += nInnerAll[g.jk];
				jk = g.jk;
			}
			s++;
		}
	}
	mpiWorld->allReduce(vFsqSum, MPIUtil::ReduceSum);
	mpiWorld->allReduce(weightSum, MPIUtil::ReduceSum);
	for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
	{	size_t iEstart = stateOffset[kDivision.start(jProc)];
		size_t iEstop = stateOffset[kDivision.stop(jProc)];
		mpiWorld->bcast(&Eall[iEstart], iEstop-iEstart, jProc);
	}
	double vF = sqrt(vFsqSum / weightSum);
	double gEf = h.spinWeight*weightSum/h.nkTot;
	logPrintf("vF = %lg\n", vF); logFlush();
	logPrintf("g(Ef) = %lg\n", gEf); logFlush();
	
	//Construct BTE sparse matrix:
	logPrintf("\nConstructing BTE matrix: "); logFlush();
	TripletMatrix S(nStatesTot, nStatesTot, nNZbound);
	double prefacS = 2*M_PI/h.nkTot;
	double tauInvNum = 0.;
	for(size_t ik=ikStart; ik<ikStop; ik++)
	{	const LindbladFile::Kpoint& kp = kpoint[ik-ikStart];
		const State* s1 = state.data() + (stateOffset[ik]-stateOffsetMine);
		auto gStart = kp.GePh.begin();
		while(gStart != kp.GePh.end())
		{	size_t jk = gStart->jk;
			//Find range of g with same jk:
			auto gStop = gStart; gStop++;
			while((gStop != kp.GePh.end()) and (gStop->jk == jk)) gStop++;
			//Collect contributions:
			matrix T = zeroes(nInnerAll[ik], nInnerAll[jk]); //real=T+, imag=T- contribution in derivation
			complex* Tdata = T.data();
			for(auto g=gStart; g!=gStop; g++)
			{	double nPh = bose(invT*g->omegaPh); //phonon occupation
				for(const SparseEntry& e: g->G)
				{	double term = prefacS * e.val.norm(); //prefactors * |e-ph matrix element|^2 * energy conservation factor
					double nfbar_i = nPh + 1 - s1[e.i].f;
					double nf_j = nPh*(nPh+1)/nfbar_i; //nPh + fj by detailed balance
					Tdata[T.index(e.i,e.j)] += term * complex(nfbar_i, nf_j); //re->T+, im->T-
					//Scattering time average:
					double fj = nf_j - nPh, mfjPrime = invT*fj*(1.-fj);
					tauInvNum += term * (nfbar_i * s1[e.i].mfPrime + nf_j * mfjPrime);
				}
			}
			for(int bj=0; bj<nInnerAll[jk]; bj++)
				for(int bi=0; bi<nInnerAll[ik]; bi++)
				{	if(Tdata->norm())
					{	size_t iState = stateOffset[ik] + bi;
						size_t jState = stateOffset[jk] + bj;
						S.entries.push_back(TripletMatrix::Entry(iState,jState, Tdata->real()));
						S.entries.push_back(TripletMatrix::Entry(jState,iState, Tdata->imag()));
					}
					Tdata++;
				}
			//Move to next jk set:
			gStart = gStop;
		}
		//Print progress:
		if((ik-ikStart+1)%ikInterval==0) { logPrintf("%d%% ", int(round((ik-ikStart+1)*100./nkMine))); logFlush(); }
	}
	logPrintf("done.\n"); logFlush();
	mpiWorld->allReduce(tauInvNum, MPIUtil::ReduceSum);
	
	//Convert 'S' to matrix 'B' in the derivation:
	diagMatrix DiagSsum = S.DiagSumRows();
	for(size_t iState=stateOffsetMine; iState<stateOffsetMine+nStatesMine; iState++)
		S.entries.push_back(TripletMatrix::Entry(iState, iState, -DiagSsum[iState]));
	const TripletMatrix& B = S; //call it B for clarity in the following
	
	//Construct velocity and -fPrime matrices:
	matrix V = zeroes(nStatesTot, 3);
	diagMatrix mfPrime(nStatesTot, 0.);
	for(size_t iState=stateOffsetMine; iState<stateOffsetMine+nStatesMine; iState++)
	{	const State& s = state[iState-stateOffsetMine];
		for(int dir=0; dir<3; dir++)
			V.set(iState, dir, s.v[dir]);
		mfPrime[iState] = s.mfPrime;
	}
	mpiWorld->allReduceData(V, MPIUtil::ReduceSum);
	mpiWorld->allReduceData(mfPrime, MPIUtil::ReduceSum);
	matrix mfPrimeV = mfPrime * V;
	
	//Simple estimate (analgous to resistivity.cpp, but on uniform k-mesh):
	double Tt = (h.spinWeight/(3.*h.nkTot)) * trace(dagger(V) * mfPrimeV).real();
	double Gamma = (-h.spinWeight/(3.*h.nkTot)) * trace(dagger(V) * (B * mfPrimeV)).real();
	double rhoSimple = Omega*Gamma/(Tt*Tt);
	double tauDrude = Tt / Gamma;
	
	//Calculate inv(B) * fPrimeV iteratively:
	matrix invB_mfPrimeV = tauDrude * mfPrimeV; //Drude model as initial guess
	B.applyInverse(mfPrimeV, invB_mfPrimeV);
	
	//Calculate conductivity tensor using Boltzmann equation:
	matrix sigmaMat = (-h.spinWeight/(h.nkTot*Omega)) * dagger(V) * invB_mfPrimeV;
	matrix rhoMat = inv(sigmaMat);
	double sigma = (1./3) * trace(sigmaMat).real();
	double rho = 1./sigma;
	double tauInv = tauInvNum / weightSum;
	
	//Report:
	double invRhoUnit = 1./(1e-9*Ohm*meter);
	logPrintf("\n");
	logPrintf("tauDrude         = %12lg fs\n", tauDrude/fs);
	logPrintf("tau              = %12lg fs\n", (1./tauInv)/fs);
	logPrintf("Resistivity(RTA) = %12lg nOhm-m\n", rhoSimple*invRhoUnit);
	logPrintf("Resistivity      = %12lg nOhm-m\n", rho*invRhoUnit);
	logPrintf("Resistivity tensor [nOhm-m]:\n"); 
	matrix(rhoMat*invRhoUnit).print_real(globalLog, " %12lg ");
	logPrintf("\n");
	
	FeynWann::finalize();
}
