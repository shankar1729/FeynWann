#include <lindblad/Lindblad.h>
#include <commands/command.h>
#include <core/Units.h>
#include <Histogram.h>


Lindblad::Lindblad(const LindbladParams& lp)
: lp(lp), stepID(0), Emin(+DBL_MAX), Emax(-DBL_MAX),
	K(1./3, 1./3, 0), Kp(-1./3, -1./3, 0)
{
	//Read header and check parameters:
	MPIUtil::File fp;
	mpiWorld->fopenRead(fp, lp.inFile.c_str());
	LindbladFile::Header h; h.read(fp, mpiWorld);
	if(lp.dmu<h.dmuMin or lp.dmu>h.dmuMax)
		die("dmu = %lg eV is out of range [ %lg , %lg ] eV specified in lindbladInit.\n", lp.dmu/eV, h.dmuMin/eV, h.dmuMax/eV);
	if(lp.T > h.Tmax)
		die("T = %lg K is larger than Tmax = %lg K specified in lindbladInit.\n", lp.T/Kelvin, h.Tmax/Kelvin);
	if((not lp.pumpBfield) and (lp.pumpOmega > h.pumpOmegaMax))
		die("pumpOmega = %lg eV is larger than pumpOmegaMax = %lg eV specified in lindbladInit.\n", lp.pumpOmega/eV, h.pumpOmegaMax/eV);
	if(lp.omegaMax > h.probeOmegaMax)
		die("omegaMax = %lg eV is larger than probeOmegaMax = %lg eV specified in lindbladInit.\n", lp.omegaMax/eV, h.probeOmegaMax/eV);
	nk = h.nk;
	nkTot = h.nkTot;
	spinorial = h.spinorial;
	spinWeight = h.spinWeight;
	R = h.R; Omega = fabs(det(R));
	if(lp.ePhEnabled != h.ePhEnabled)
		die("ePhEnabled = %s differs from the mode specified in lindbladInit.\n", boolMap.getString(lp.ePhEnabled));
	if(lp.pumpBfield and (not spinorial))
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
	if(lp.valleyMode != ValleyNone) isKall.resize(nk);
	
	//Read k-point info and initialize states:
	mpiWorld->fseek(fp, byteOffsets[ikStart], SEEK_SET);
	for(size_t ikMine=0; ikMine<nkMine; ikMine++)
	{	State& s = state[ikMine];
		s.ik = ikStart + ikMine;
		
		//Read base info from LindbladFile:
		((LindbladFile::Kpoint&)s).read(fp, mpiWorld, h);
		nInnerAll[s.ik] = s.nInner;
		
		//Initialize extra quantities in state:
		s.innerStop = s.innerStart + s.nInner;
		//--- Active energy range:
		Emin = std::min(Emin, s.E[s.innerStart]);
		Emax = std::max(Emax, s.E[s.innerStop-1]);
		//--- Pump matrix elements with energy conservation
		if(not lp.pumpBfield)
		{	s.pumpPD = dot(s.P, lp.pumpPol)(0,s.nInner, s.innerStart,s.innerStop); //restrict to inner active
			double normFac = sqrt(lp.pumpTau/sqrt(M_PI));
			complex* PDdata = s.pumpPD.data();
			for(int b2=s.innerStart; b2<s.innerStop; b2++)
				for(int b1=s.innerStart; b1<s.innerStop; b1++)
				{	//Multiply energy conservation:
					double tauDeltaE = lp.pumpTau*(s.E[b1] - s.E[b2] - lp.pumpOmega);
					*(PDdata++) *= normFac * exp(-0.5*tauDeltaE*tauDeltaE);
				}
		}
		if(lp.valleyMode != ValleyNone) isKall[s.ik] = isKvalley(s.k);
		
		//Set initial occupations:
		s.rho0.resize(s.nInner);
		for(int b=0; b<s.nInner; b++)
			s.rho0[b] = fermi((s.E[b+s.innerStart] - lp.dmu) * lp.invT);
		
		//Initialize H0 used for interaction picture:
		s.E0 = s.E(s.innerStart, s.innerStop); //default: diagonal using energies from data file
		if(lp.Bext.length_squared() and spinorial)
		{	matrix H0(s.E0);
			for(int iDir=0; iDir<3; iDir++) //Add Zeeman Hamiltonian
				H0 -= lp.Bext[iDir] * s.S[iDir];
			H0.diagonalize(s.V0, s.E0); //now have diagonal basis for off-diagonal H0
		}
		
		//Initialize density matrix and time derivative:
		s.rho = matrix(s.rho0);
		s.rhoDot = zeroes(s.nInner, s.nInner);
		s.phase = eye(s.nInner);
	}
	mpiWorld->fclose(fp);
	if(lp.valleyMode != ValleyNone) mpiWorld->allReduceData(isKall, MPIUtil::ReduceMax);

	
	//Synchronize energy range:
	mpiWorld->allReduce(Emin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(Emax, MPIUtil::ReduceMax);
	logPrintf("Electron energy grid from %lg eV to %lg eV with spacing %lg eV.\n", Emin/eV, Emax/eV, lp.dE/eV);
	
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
	drho.assign(rhoSize[mpiWorld->iProcess()], 0.);

	//Make inner-window energies available for all processes (if needed):
	if(lp.ePhEnabled)
	{	nInnerPrev.assign(nk+1, 0); //cumulative nInner for each k (offset into the Eall array)
		nRhoPrev.assign(nk+1, 0); //cumulative nInner^2 for each k (offset into global rho)
		for(size_t ik=0; ik<nk; ik++)
		{	nInnerPrev[ik+1] = nInnerPrev[ik] + nInnerAll[ik];
			nRhoPrev[ik+1] = nRhoPrev[ik] +  nInnerAll[ik]*nInnerAll[ik];
		}
		Eall.resize(nInnerPrev.back());
		for(const State& s: state)
		{	const double* Ei = &(s.E[s.innerStart]);
			std::copy(Ei, Ei+s.nInner, Eall.begin()+nInnerPrev[s.ik]);
		}
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
		{	size_t iEstart = nInnerPrev[kDivision.start(jProc)];
			size_t iEstop = nInnerPrev[kDivision.stop(jProc)];
			mpiWorld->bcast(&Eall[iEstart], iEstop-iEstart, jProc);
		}
	}
	logPrintf("\n"); logFlush();
}


void Lindblad::calculate()
{
	if(not (lp.pumpEvolve or lp.ePhEnabled))
	{	//Simple probe - one-shot-pump - probe with no relaxation:
		report(-lp.dt, drho);
		applyPump(); //one-shot optical pump or B-field excitation
		report(0., drho);
	}
	else
	{	//Pump and time evolve with continuous probe:
		//Initialization:
		double tStart = 0.;
		if(readCheckpoint(tStart))
		{	//tStart and stepID already set to end of previously checkpointed run.
			//No other initialization needed.
		}
		else
		{	if(lp.pumpEvolve)
			{	//Set start time to a multiple of dt that covers pulse:
				tStart = -lp.dt * ceil(5.*lp.tau/lp.dt);
				//pump will be included in evolution below
			}
			else
			{	report(-lp.dt, drho); //initial-state report
				applyPump(); //takes care of optical pump or B-field excitation
				tStart = 0.; //integrate will report at t=0 below, before evolving ePh relaxation
			}
		}
		
		//Evolution:
		if(lp.tStep) //Fixed-step integrator:
			integrateFixed(drho, tStart, lp.tStop, lp.tStep, lp.dt);
		else //Adaptive integrator:
			integrateAdaptive(drho, tStart, lp.tStop, lp.tolAdaptive, lp.dt);
	}
}


void Lindblad::applyPump()
{	static StopWatch watch("Lindblad::applyPump"); 
	if(lp.pumpEvolve) return; //only use this function when perturbing instantly
	watch.start();
	//Perturb each k separately:
	for(const State& s: state)
	{	if(lp.pumpBfield)
		{	//Construct Hamiltonian including magnetic field contribution:
			matrix Htot(s.E(s.innerStart, s.innerStart+s.nInner));
			for(int iDir=0; iDir<3; iDir++) //Add Zeeman Hamiltonian
				Htot -= lp.pumpB[iDir] * s.S[iDir];
			//Set rho to Fermi function of this perturbed Hamiltonian:
			diagMatrix Epert; matrix Vpert;
			Htot.diagonalize(Vpert, Epert);
			diagMatrix fPert(s.nInner);
			for(int b=0; b<s.nInner; b++)
				fPert[b] = fermi((Epert[b] - lp.dmu) * lp.invT);
			matrix rhoPert = Vpert * fPert * dagger(Vpert);
			accumRhoHC(0.5*(rhoPert-s.rho0), drho.data()+rhoOffset[s.ik]);
		}
		else
		{	const diagMatrix& rho0 = s.rho0;
			diagMatrix rho0bar = bar(rho0); //1-rho0
			//Compute and apply perturbation:
			matrix P = s.pumpPD; //P-
			matrix Pdag = dagger(P); //P+
			matrix deltaRho;
			for(int s=-1; s<=+1; s+=2)
			{	deltaRho += rho0bar*P*rho0*Pdag - Pdag*rho0bar*P*rho0;
				std::swap(P, Pdag); //P- <--> P+
			}
			accumRhoHC((M_PI*std::pow(lp.pumpA0, 2)) * deltaRho, drho.data()+rhoOffset[s.ik]);
		}
	}
	watch.stop();
}


void Lindblad::setState(double t, const DM1& drho, State& s) const
{	s.rhoDot.zero();
	s.drho = getRho(drho.data()+rhoOffset[s.ik], s.nInner);
	s.rho = s.rho0 + s.drho;

	//Calculate and apply phases:
	//--- in diagonal basis
	std::vector<complex> phaseDiag(s.nInner);
	for(int b=0; b<s.nInner; b++)
		phaseDiag[b] = cis(-t * s.E0[b]);
	//--- convert to required basis
	if(s.V0)
	{	s.phase = (s.V0 * phaseDiag) * dagger(s.V0); //construct unitary e^(-i H0 t) transform
		s.drho = s.phase * s.drho * dagger(s.phase); //apply this transform to drho
		s.rho = s.phase * s.rho * dagger(s.phase); //apply this transform to rho
	}
	else
	{	complex* phaseData = s.phase.data();
		complex* drhoData = s.drho.data();
		complex* rhoData = s.rho.data();
		for(int bCol=0; bCol<s.nInner; bCol++)
			for(int bRow=0; bRow<s.nInner; bRow++)
			{	complex phaseCur = phaseDiag[bRow] * phaseDiag[bCol].conj();
				*(phaseData++) = phaseCur; //store multiplicative phase for each element
				*(drhoData++) *= phaseCur; //apply multiplicative phase to drho
				*(rhoData++) *= phaseCur; //apply multiplicative phase to rho
			}
	}
}


void Lindblad::getStateDot(const State& s, DM1& rhoDot) const
{	//Convert rhoDot from Schrodinger to interaction picture:
	matrix rhoDotCur;
	if(s.V0)
		rhoDotCur = dagger(s.phase) * s.rhoDot * s.phase; //reverse unitary transform for phase
	else
	{	rhoDotCur = clone(s.rhoDot);
		complex* rhoDotData = rhoDotCur.data();
		const complex* phaseData = s.phase.data();
		for(int bCol=0; bCol<s.nInner; bCol++)
			for(int bRow=0; bRow<s.nInner; bRow++)
				*(rhoDotData++) *= (phaseData++)->conj(); //conjugated multiplicative phase
	}
	//Set into rhoDot with H.C. term:
	accumRhoHC(rhoDotCur, rhoDot.data()+rhoOffset[s.ik]);
}


bool Lindblad::readCheckpoint(double& t)
{	//Check checkpoimnt availability:
	bool checkpointExists = false;
	if(mpiWorld->isHead())
		checkpointExists = ((lp.checkpointFile.length() > 0) //checkpoint specified
			and (fileSize(lp.checkpointFile.c_str())>0)); // ... and is readable
	mpiWorld->bcast(checkpointExists);

	if(checkpointExists)
	{	logPrintf("Reading checkpoint from '%s' ... ", lp.checkpointFile.c_str()); logFlush(); 

		//Determine offset of current process data and total expected file length:
		size_t headerLength = sizeof(int) + sizeof(double);
		size_t offset = headerLength + sizeof(double)*rhoOffsetGlobal;
		size_t fsizeExpected = headerLength + sizeof(double)*rhoSizeTot;

		//Open check point file and read step/time header:
		MPIUtil::File fp;
		mpiWorld->fopenRead(fp, lp.checkpointFile.c_str(), fsizeExpected);
		mpiWorld->fread(&stepID, sizeof(int), 1, fp);
		mpiWorld->fread(&t, sizeof(double), 1, fp);
		mpiWorld->bcast(t);

		//Read density matrix from check point file:
		mpiWorld->fseek(fp, offset, SEEK_SET);
		mpiWorld->fread(drho.data(), sizeof(double), drho.size(), fp);
		mpiWorld->fclose(fp);
		logPrintf("done.\n");
	}
	return checkpointExists;
}


void Lindblad::writeCheckpoint(double t) const
{
	if(not lp.checkpointFile.length()) return; //checkpoint disabled
#ifdef MPI_SAFE_WRITE
	if(mpiWorld->isHead())
	{	FILE* fp = fopen(lp.checkpointFile.c_str(), "w");
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
	MPIUtil::File fp;
	mpiWorld->fopenWrite(fp, lp.checkpointFile.c_str());
	//--- Write current step and time as a header:
	if(mpiWorld->isHead())
	{	mpiWorld->fwrite(&stepID, sizeof(int), 1, fp);
		mpiWorld->fwrite(&t, sizeof(double), 1, fp);
	}
	//--- Move to location of this process's data:
	size_t offset = sizeof(int) + sizeof(double); //offset due to header
	offset += sizeof(double)*rhoOffsetGlobal; //offset due to data from previous processes
	mpiWorld->fseek(fp, offset, SEEK_SET);
	//--- Write this process's data:
	mpiWorld->fwrite(drho.data(), sizeof(double), drho.size(), fp);
	mpiWorld->fclose(fp);
#endif
}


vector3<> Lindblad::getB(double t) const
{	double omegaFreq  = 0.001; //Main Larmor frequency in Hartrees (time unit is 1/140 fs).
	double deltaOmega = 0.1*omegaFreq; //Sets the magnitude of the perturbing field. deltaOmega = gamma * deltaB
	double piPulseDuration = M_PI/deltaOmega;
	double tDelay = 100*1000*fs;
	vector3<> deltaBoff; //zero field
	vector3<> deltaBon = (0.5*deltaOmega) * vector3<>(cos(omegaFreq*t), sin(omegaFreq*t), 0); // perturbing B field when on
	//Modulation:
	if(t < 0.)
		return deltaBoff;
	//--- pi/2 pulse is on here
	if(t < 0.5*piPulseDuration)
		return deltaBon;
	if(t < 0.5*piPulseDuration + tDelay)
		return deltaBoff;
	//--- pi pulse is on here
	if(t < 1.5*piPulseDuration + tDelay)
		return deltaBon;
	else
		return deltaBoff;
}


DM1 Lindblad::compute(double t, const DM1& drho)
{	double pumpPrefac = lp.pumpEvolve
		? sqrt(M_PI) * std::pow(lp.pumpA0, 2) * exp(-(t*t)/std::pow(lp.pumpTau, 2)) / lp.pumpTau
		: 0.;
	vector3<> Bcur = lp.spinEcho ? getB(t) : vector3<>();
	
	for(State& s: state)
	{	//Convert interaction picture input to Schrodinger picture within state:
		setState(t, drho, s);
		
		//---- k-diagonal contributions -----
		
		//Pump:
		if(lp.pumpEvolve)
		{	matrix P = s.pumpPD; //P-
			matrix Pdag = dagger(P); //P+
			const matrix rhoBar = s.rho; //1-rho
			for(int sign=-1; sign<=+1; sign+=2)
			{	s.rhoDot += pumpPrefac * (rhoBar * P * s.rho * Pdag
										- Pdag * rhoBar * P * s.rho); //+HC added by getRhoDot()
				std::swap(P, Pdag); //P- <--> P+
			}
		}
		
		//Time-dependent magnetic field contribution:
		if(lp.spinEcho)
		{	assert(spinorial);
			matrix deltaH = s.S[0]*Bcur[0] + s.S[1]*Bcur[1] + s.S[2]*Bcur[2];
			s.rhoDot += complex(0, 1) * s.rho * deltaH; //+HC added by getRhoDot() completes commutator
		}
	}
	
	//Scattering contributions (e-ph, defects) that couple k:
	rhoDotScatter();
	
	//Convert final result back to interaction picture:
	DM1 rhoDot(drho.size());
	for(const State& s: state)
		getStateDot(s, rhoDot);
	
	if(lp.verbose)
	{	//Report current statistics:
		double rhoDotMax = 0., rhoEigMin = +DBL_MAX, rhoEigMax = -DBL_MAX;
		for(const State& s: state)
		{	//max(rhoDot)
			rhoDotMax = std::max(rhoDotMax, s.rhoDot.data()[cblas_izamax(s.rhoDot.nData(), s.rhoDot.data(), 1)].abs());
			//eig(rho):
			matrix V; diagMatrix f;
			s.rho.diagonalize(V, f);
			rhoEigMin = std::min(rhoEigMin, f.front());
			rhoEigMax = std::max(rhoEigMax, f.back());
		}
		mpiWorld->reduce(rhoDotMax, MPIUtil::ReduceMax);
		mpiWorld->reduce(rhoEigMax, MPIUtil::ReduceMax);
		mpiWorld->reduce(rhoEigMin, MPIUtil::ReduceMin);
		logPrintf("\n\tComputed at t[fs]: %lg  max(rhoDot): %lg rhoEigRange: [ %lg %lg ] ",
			t/fs, rhoDotMax, rhoEigMin, rhoEigMax); logFlush();
	}
	else logPrintf("* "); //just a visual progress bar
	logFlush();

	return rhoDot;
}


void Lindblad::report(double t, const DM1& drho) const
{	static StopWatch watch("Lindblad::report"); watch.start();
	ostringstream ossID; ossID << stepID;

	//Total energy and distributions:
	int nDist = lp.saveDist
		? (spinorial ? 4 : 1) //number distribution only, or also spin distribution
		: 0; //don't save distributions
	std::vector<Histogram> dist(nDist, Histogram(Emin, lp.dE, Emax));
	const double prefac = spinWeight*(1./nkTot); //BZ integration weight
	double Etot = 0., dfMax = 0.; vector3<> Stot;
	for(const State& s: state)
	{	setState(t, drho, (State&)s); //update Schrodinger-picture quantities
		
		//Energy and distribution:
		const complex* drhoData = s.drho.data();
		for(int b=0; b<s.nInner; b++)
		{	double weight = prefac * drhoData->real();
			const double& Ecur = s.E[b+s.innerStart];
			Etot += weight * Ecur;
			dfMax = std::max(dfMax, fabs(drhoData->real()));
			if(lp.saveDist)
				dist[0].addEvent(Ecur, weight);
			drhoData += (s.nInner+1); //advance to next diagonal entry
		}
		
		//Spin distribution (if available):
		if(spinorial)
		{	const complex* drhoData = s.drho.data();
			vector3<const complex*> Sdata; for(int k=0; k<3; k++) Sdata[k] = s.S[k].data();
			std::vector<vector3<>> Sband(s.nInner); //spin expectation by band S_b := sum_a S_ba drho_ab
			for(int b2=0; b2<s.nInner; b2++)
			{	for(int b1=0; b1<s.nInner; b1++)
				{	complex weight = prefac * (*(drhoData++)).conj();
					for(int iDir=0; iDir<3; iDir++)
						Sband[b2][iDir] += (weight * (*(Sdata[iDir]++))).real();
				}
				Stot += Sband[b2];
			}
			//Collect distribution based on per-band spin:
			if(lp.saveDist)
			{	for(int b=0; b<s.nInner; b++)
				{	const double& E = s.E[b+s.innerStart];
					int iEvent; double tEvent;
					if(dist[1].eventPrecalc(E, iEvent, tEvent))
					{	for(int iDir=0; iDir<3; iDir++)
							dist[iDir+1].addEventPrecalc(iEvent, tEvent, Sband[b][iDir]);
					}
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
		logPrintf("Integrate: Step: %4d   t[fs]: %6.1lf   Etot[eV]: %.2le   dfMax: %.2le", stepID, t/fs, Etot/eV, dfMax);
		if(spinorial) logPrintf("   S: [ %16.15lg %16.15lg %16.15lg ]", Stot[0],  Stot[1],  Stot[2]);
		logPrintf("\n"); logFlush();
		
		//Save distribution functions:
		if(lp.saveDist)
		{	ofstream ofs("dist."+ossID.str());
			ofs << "#E-mu/VBM[eV] n[eV^-1]";
			if(spinorial)
				ofs << "Sx[eV^-1] Sy[eV^-1] Sz[eV^-1]";
			ofs << "\n";
			for(int iE=0; iE<dist[0].nE; iE++)
			{	double E = Emin + iE*lp.dE;
				ofs << E/eV;
				for(int iDist=0; iDist<nDist; iDist++)
					ofs << '\t' << dist[iDist].out[iE]*eV;
				ofs << '\n';
			}
		}
	}
	
	//Other file outputs:
	writeCheckpoint(t);
	writeImEps("imEps." + ossID.str());
	((Lindblad*)this)->stepID++; //Increment stepID
	watch.stop();
}


//Write probe response at current rho
void Lindblad::writeImEps(string fname) const
{	static StopWatch watch("Lindblad::calcImEps");
	size_t nImEps = lp.pol.size() * lp.nomega;
	if(nImEps==0) return; //no probe specified

	watch.start();
	diagMatrix imEps(nImEps);
	
	//Collect contributions from each k at this process:
	for(const State& s: state)
	{
		//Expand density matrix:
		matrix rhoCur = zeroes(s.nOuter, s.nOuter);
		if(s.innerStart) rhoCur.set(0,s.innerStart, 0,s.innerStart, eye(s.innerStart));
		rhoCur.set(s.innerStart,s.innerStop, s.innerStart,s.innerStop, s.rho);
		matrix rhoBar = bar(rhoCur); //1-rho
		
		//Expand probe matrix elements:
		std::vector<matrix> Ppol(lp.pol.size(), zeroes(s.nOuter, s.nOuter));
		for(int iDir=0; iDir<3; iDir++)
		{	//Expand Cartesian component:
			const matrix& PiSub = s.P[iDir]; //nInner x nOuter
			matrix Pi = zeroes(s.nOuter, s.nOuter);
			Pi.set(s.innerStart,s.innerStop, 0,s.nOuter, PiSub);
			Pi.set(0,s.nOuter, s.innerStart,s.innerStop, dagger(PiSub));
			//Update each polarization:
			for(int iPol=0; iPol<int(lp.pol.size()); iPol++)
				Ppol[iPol] += lp.pol[iPol][iDir] * Pi;
		}

		//Probe response:
		for(int iomega=0; iomega<lp.nomega; iomega++)
		{	double omega = lp.omegaMin + iomega*lp.domega;
			double prefac = (4*std::pow(M_PI,2)*spinWeight)/(nkTot * Omega * std::pow(std::max(omega, 1./lp.tau), 3));
			
			//Energy conservation and phase factors for all pair of bands at this frequency:
			std::vector<complex> delta(s.nOuter*s.nOuter);
			complex* deltaData = delta.data();
			double normFac = sqrt(lp.tau/sqrt(M_PI));
			for(int b2=0; b2<s.nOuter; b2++)
				for(int b1=0; b1<s.nOuter; b1++)
				{	double tauDeltaE = lp.tau*(s.E[b1] - s.E[b2] - omega);
					*(deltaData++) = normFac * exp(-0.5*tauDeltaE*tauDeltaE);
				}
			
			//Loop over polarizations:
			for(int iPol=0; iPol<int(lp.pol.size()); iPol++)
			{	//Multiply matrix elements with energy conservation:
				matrix P = Ppol[iPol];
				eblas_zmul(P.nData(), delta.data(),1, P.data(),1); //P-
				matrix Pdag = dagger(P); //P+
				
				//Loop over directions of excitations:
				diagMatrix deltaRhoDiag(s.nOuter);
				for(int s=-1; s<=+1; s+=2)
				{	deltaRhoDiag += diag(rhoBar*P*rhoCur*Pdag - Pdag*rhoBar*P*rhoCur);
					std::swap(P, Pdag); //P- <--> P+
				}
				imEps[iPol*lp.nomega+iomega] += prefac * dot(s.E, deltaRhoDiag);
			}
		}
	}

	//Accumulate contributions from all processes to head and write:
	mpiWorld->reduceData(imEps, MPIUtil::ReduceSum);
	if(mpiWorld->isHead())
	{	ofstream ofs(fname);
		ofs << "#omega[eV]";
		for(int iPol=0; iPol<int(lp.pol.size()); iPol++)
			ofs << " ImEps" << (iPol+1);
		ofs << "\n";
		for(int iomega=0; iomega<lp.nomega; iomega++)
		{	double omega = lp.omegaMin + iomega*lp.domega;
			ofs << omega/eV;
			for(int iPol=0; iPol<int(lp.pol.size()); iPol++)
				ofs << '\t' << imEps[iPol*lp.nomega+iomega];
			ofs << '\n';
		}
	}	
	watch.stop();
}
