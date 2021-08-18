#include <lindblad/Lindblad.h>
#include <commands/command.h>
#include <core/Units.h>


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
		
		//Read base info from LindbladFile:
		((LindbladFile::Kpoint&)s).read(fp, mpiWorld, h);
		nInnerAll[ikStart+ikMine] = s.nInner;
		
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
		if(lp.valleyMode != ValleyNone) isKall[ikStart+ikMine] = isKvalley(s.k);
		
		//Set initial occupations:
		s.rho0.resize(s.nInner);
		for(int b=0; b<s.nInner; b++)
			s.rho0[b] = fermi((s.E[b+s.innerStart] - lp.dmu) * lp.invT);
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
{
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


DM1 Lindblad::compute(double t, const DM1& v)
{
	return DM1(); //TODO
}


void Lindblad::report(double t, const DM1& v) const
{
	//TODO
}
