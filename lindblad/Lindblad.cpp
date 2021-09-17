#include <lindblad/Lindblad.h>
#include <commands/command.h>
#include <core/Units.h>


//---- implementation of class LindbladParams ----


void LindbladParams::initialize()
{	invT = 1./T;
	nomega = 1 + int(round((omegaMax-omegaMin)/domega));
	
	//Spin echo setup:
	spinEchoOmega = 2.*Bext.length();
	if(spinEchoB.length_squared())
	{
		//Check specified magnetic fields:
		const double tol = 1E-6;
		if(not Bext.length_squared())
			die("\nSpin echo requires non-zero magnetic field Bext.\n\n");
		vector3<> BextHat = normalize(Bext); //unit vector of precession axis
		vector3<> spinEchoBhat = normalize(spinEchoB); //unit vector of starting rotational field
		if(fabs(dot(spinEchoBhat, BextHat)) >= tol)
			die("\nspinEchoB must be perpendicular to Bext\n\n");
		
		//Ensure initial conditions:
		if(not pumpBfield)
			die("\nSpin echo requires initial spin state to be set with B-field pump mode.\n\n");
		if(fabs(cross(pumpB, BextHat).length()) >= tol * pumpB.length())
			die("\nSpin echo requires pumpB and Bext to be parallel.\n\n");
		
		//Compute rotation matrix and period:
		spinEchoRot.set_col(0, spinEchoBhat);
		spinEchoRot.set_col(1, cross(BextHat, spinEchoBhat));
		spinEchoRot.set_col(2, BextHat);
		spinEchoFlipTime = M_PI/(2.*spinEchoB.length());
	}
	else spinEchoFlipTime = 0.; //can use as a flag to check if spin echo being used
}


vector3<> LindbladParams::spinEchoTransform(vector3<> v, double t) const
{	vector3<> vRef = v * spinEchoRot; //rotate to reference orientation (Bext along z)
	double c = cos(spinEchoOmega*t), s = sin(spinEchoOmega*t);
	vector3<> vLabRef(c*vRef[0]-s*vRef[1], s*vRef[0]+c*vRef[1], vRef[2]); //rotate to lab frame in reference axes
	return spinEchoRot * vLabRef; //rotate back from reference to real axes
}


vector3<> LindbladParams::spinEchoGetB(double t) const
{	vector3<> deltaBoff; //zero field
	if(not spinEchoFlipTime)
		return deltaBoff;

	//Modulated field:
	vector3<> deltaBon = spinEchoTransform(spinEchoB, -t); //perturbing B field when on (spinEchoB in rotating frame -> lab frame)
	if(t < 0.)
		return deltaBoff;
	//--- pi/2 pulse is on here
	if(t < 0.5*spinEchoFlipTime)
		return deltaBon;
	if(t < spinEchoDelay)
		return deltaBoff;
	//--- pi pulse is on here
	if(t < spinEchoDelay + spinEchoFlipTime)
		return deltaBon;
	else
		return deltaBoff;
}


//---- implementation of class Lindblad (except dynamics functions) ----


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
	if(lp.Bext.isNonzero() and (not spinorial))
		die("Bext requires spin matrix elements from a spinorial calculation.\n");
	if(lp.spinEchoFlipTime and (not spinorial))
		die("Spin echo measurement requires spin matrix elements from a spinorial calculation.\n");
	
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
