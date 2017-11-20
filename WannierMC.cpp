#include "WannierMC.h"
#include <core/BlasExtra.h>
#include <core/Random.h>
#include <fftw3-mpi.h>
#include "config.h"

WannierMCParams::WannierMCParams()
: totalEprefix("Wannier/totalE"), phononPrefix("Wannier/phonon"), wannierPrefix("Wannier/wannier"),
needPhonons(false), needLinewidths(false), needVelocity(false)
{
	
}

InitParams WannierMC::initialize(int argc, char** argv, const char* description)
{	InitParams ip;
	ip.packageName = PACKAGE_NAME;
	ip.versionString = VERSION_STRING;
	ip.versionHash = GIT_HASH;
	ip.description = description;
	initSystemCmdline(argc, argv, ip);
	fftw_mpi_init();
	return ip;
}

void WannierMC::finalize()
{	fftw_mpi_cleanup();
	finalizeSystem();
}

vector3<> WannierMC::randomVector(MPIUtil* mpiUtil)
{	vector3<> v;
	for(int iDir=0; iDir<3; iDir++)
		v[iDir] = Random::uniform();
	if(mpiUtil) mpiUtil->bcast(&v[0], 3);
	return v;
}

//Read matrix from file accounting for real-only or complex storage based on spinWeight
void readMatrix(matrix& m, string fname, int spinWeight)
{	logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
	if(spinWeight==1) m.read(fname.c_str()); else m.read_real(fname.c_str());
	logPrintf("done.\n"); logFlush();
}

WannierMC::WannierMC(const WannierMCParams& wmcp)
: wmcp(wmcp), spinWeight(0), mu(NAN), nElectrons(0), nValence(0)
{	
	//Read relevant parameters from totalE.out:
	logPrintf("\nReading '%s.out' ... ", wmcp.totalEprefix.c_str()); logFlush();
	ifstream ifs(wmcp.totalEprefix + ".out");
	if(!ifs.is_open()) die("could not open file.\n");
	bool initDone = false; //whether finished reading the initialization part of totalE.out
	int nBandsDFT = 0; //number of DFT bands (>= this->nBands = # Wannier bands)
	while(!ifs.eof())
	{	string line; getline(ifs, line);
		if(line.find("Initializing the grid") != string::npos)
		{	getline(ifs, line); //skip the line containing "R = "
			for(int j=0; j<3; j++)
			{	getline(ifs, line);
				sscanf(line.c_str(), "[ %lf %lf %lf ]", &R(j,0), &R(j,1), &R(j,2));
			}
			Omega = fabs(det(R));
		}
		else if(line.find("kpoint-folding") != string::npos)
		{	istringstream iss(line); string buf;
			iss >> buf >> kfold[0] >> kfold[1] >> kfold[2];
		}
		else if(line.find("spintype") != string::npos)
		{	istringstream iss(line); string buf, spinString;
			iss >> buf >> spinString;
			if(spinString == "no-spin")
				spinWeight = 2;
			else if(spinString == "spin-orbit")
				spinWeight = 1;
			else
				die("Spin-polarized modes not yet supported.\n");
		}
		else if(line.find("coulomb-interaction") != string::npos)
		{	istringstream iss(line); string cmdName, typeString, dirString;
			iss >> cmdName >> typeString >> dirString;
			if(typeString == "Periodic")
			{	isTruncated = vector3<bool>(false, false, false);
			}
			else if(typeString == "Slab")
			{	isTruncated = vector3<bool>(false, false, false);
				if(dirString == "100") isTruncated[0] = true;
				else if(dirString == "010") isTruncated[1] = true;
				else if(dirString == "001") isTruncated[2] = true;
				else die("Unrecognized truncation direction '%s'\n", dirString.c_str());
			}
			else if(typeString == "Wire" || typeString == "Cylindrical")
			{	isTruncated = vector3<bool>(true, true, true);
				if(dirString == "100") isTruncated[0] = false;
				else if(dirString == "010") isTruncated[1] = false;
				else if(dirString == "001") isTruncated[2] = false;
				else die("Unrecognized truncation direction '%s'\n", dirString.c_str());
			}
			else if(typeString == "Isolated" || typeString == "Spherical")
			{	isTruncated = vector3<bool>(true, true, true);
			}
			else die("Unrecognized truncation type '%s'\n", typeString.c_str());
		}
		else if(line.find("Initialization completed") != string::npos)
		{	initDone = true;
		}
		else if(initDone && (line.find("FillingsUpdate:") != string::npos))
		{	istringstream iss(line); string buf;
			iss >> buf >> buf >> mu >> buf >> nElectrons;
		}
		else if(line.find("nElectrons:") == 0) //nElectrons, nBands, nStates line
		{	istringstream iss(line); string buf;
			iss >> buf >> nElectrons >> buf >> nBandsDFT;
		}
	}
	ifs.close();
	logPrintf("done.\n"); logFlush();
	if(std::isnan(mu))
	{	mu = 0.;
		logPrintf("NOTE: mu unavailable (setting to zero); must be semiconductor/insulator.\n");
		nValence = int(round(nElectrons/spinWeight)); //nValence is set to zero when mu is available
		if(fabs(nValence*spinWeight-nElectrons > 1e-6))
			die("Number of electrons incompatible with semiconductor / insulator.\n");
	}
	logPrintf("mu = %lg\n", mu);
	logPrintf("nElectrons = %lg\n", nElectrons);
	logPrintf("nBandsDFT = %d\n", nBandsDFT);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("kfold = "); kfold.print(globalLog, " %d ");
	logPrintf("isTruncated = "); isTruncated.print(globalLog, " %d ");
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	
	//Read relevant parameters from phonon.out:
	logPrintf("\nReading '%s.out' ... ", wmcp.phononPrefix.c_str()); logFlush();
	ifs.open(wmcp.phononPrefix + ".out");
	if(!ifs.is_open()) die("could not open file.\n");
	nModes = 0;
	while(!ifs.eof())
	{	string line; getline(ifs, line);
		if(line.find("phonon  \\") != string::npos)
		{	//at start of phonon command print
			string key;
			while(key!="supercell" && (!ifs.eof()))
				ifs >> key; //search for supercell keyword
			ifs >> phononSup[0] >> phononSup[1] >> phononSup[2];
			if(!ifs.good()) die("Failed to read phonon supercell dimensions.\n");
		}
		string cmdName; istringstream(line) >> cmdName;
		if(cmdName == "ion")
			nModes += 3; //3 modes per atom in unit cell
		if(line.find("Unit cell calculation") != string::npos)
			break; //don't need anything else after this from phonon.out
	}
	ifs.close();
	if(!phononSup.length_squared()) die("Failed to read phonon supercell dimensions.\n");
	logPrintf("done.\n"); logFlush();
	logPrintf("nModes = %d\n", nModes);
	logPrintf("phononSup = "); phononSup.print(globalLog, " %d ");
	for(int iDir=0; iDir<3; iDir++)
	{	kfoldSup[iDir] = kfold[iDir] / phononSup[iDir];
		if(kfoldSup[iDir] * phononSup[iDir] != kfold[iDir])
			die("kfold is not a multiple of phononSup.\n");
	}
	
	//Read cell map
	ifs.open(wmcp.wannierPrefix + ".mlwfCellMap");
	string headerLine; getline(ifs, headerLine); //read and ignore header line
	vector3<int> cm;
	double x,y,z;
	while(ifs >> cm[0] >> cm[1] >> cm[2] >> x >> y >> z)
		cellMap.push_back(cm);
	ifs.close();
	
	//Find number of wannier centers from Wannier band contrib file:
	{	string fname = wmcp.wannierPrefix + ".mlwfBandContrib";
		logPrintf("\nReading '%s' ... ", fname.c_str()); logFlush();
		FILE* fp = fopen(fname.c_str(), "r");
		if(!fp) die("could not open for reading.\n");
		nBands = 0; //number of Wannier centers
		while(!feof(fp))
		{	int nMin_b, nMax_b; double eMin_b, eMax_b;
			if(fscanf(fp, "%d %d %lf %lf", &nMin_b, &nMax_b, &eMin_b, &eMax_b) != 4) break;
			nBands = std::max(nBands, nMax_b+1); //number of Wannier centers
		}
		fclose(fp);
		logPrintf("done.\n");
		assert(nBands <= nBandsDFT);
	}
	logPrintf("nBands = %d\n", nBands);
	logPrintf("\n");
	
	//Read wannier hamiltonian
	bool realOnly = (spinWeight==2);
	string fname = wmcp.wannierPrefix + ".mlwfH";
	Hw = std::make_shared<DistributedMatrix>(fname, realOnly,
		mpiGroup, nBands*nBands, cellMap, kfold, false);
	
	
	if(wmcp.needPhonons)
	{	//Read phonon cell map
		string fname = wmcp.wannierPrefix + ".mlwfCellMapPh";
		logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
		ifs.open(fname.c_str());
		getline(ifs, headerLine); //read and ignore header line
		while(ifs >> cm[0] >> cm[1] >> cm[2] >> x >> y >> z)
			phononCellMap.push_back(cm);
		ifs.close();
		logPrintf("done.\n"); logFlush();
		
		//Read phonon force matrix
		fname = wmcp.wannierPrefix + ".mlwfOmegaSqPh";
		OsqW = std::make_shared<DistributedMatrix>(fname, true, //phonon omegaSq is always real
			mpiGroup, nModes*nModes, phononCellMap, phononSup, false);
		
		//Read electron-phonon matrix elements
		fname = wmcp.wannierPrefix + ".mlwfHePh";
		HePhW = std::make_shared<DistributedMatrix>(fname, realOnly,
			mpiGroup, nModes*nBands*nBands, phononCellMap, phononSup, true);
	}
	
	//Velocity matrix elements
	if(wmcp.needVelocity)
	{	fname = wmcp.wannierPrefix + ".mlwfP";
		Pw = std::make_shared<DistributedMatrix>(fname, realOnly,
			mpiGroup, 3*nBands*nBands, cellMap, kfold, false);
	}
	
	//Linewidths:
	if(wmcp.needLinewidths)
	{	//e-e:
		fname = wmcp.wannierPrefix + ".mlwfImSigma_ee";
		ImSigma_eeW = std::make_shared<DistributedMatrix>(fname, realOnly,
			mpiGroup, nBands*nBands, cellMap, kfold, false);
		//e-e:
		fname = wmcp.wannierPrefix + ".mlwfImSigma_ePh";
		ImSigma_ePhW = std::make_shared<DistributedMatrix>(fname, realOnly,
			mpiGroup, nBands*nBands, cellMap, kfold, false);
	}
	
	logPrintf("\n");
}

void WannierMC::free()
{	Hw = 0;
	Pw = 0;
	ImSigma_eeW = 0;
	ImSigma_ePhW = 0;
	OsqW = 0;
	HePhW = 0;
}

//Get iMatrix'th matrix of specified dimensions from pointer src, assuming they are stored contiguously there in column-major order)
inline matrix getMatrix(const complex* src, int nRows, int nCols, int iMatrix=0)
{	matrix result(nRows, nCols);
	eblas_copy(result.data(), src + iMatrix*result.nData(), result.nData());
	return result;
}

//Prepare and broadcast matrices on custom communicator:
inline void bcast(diagMatrix& m, int nRows, MPIUtil* mpiUtil, int root)
{	m.resize(nRows);
	mpiUtil->bcast(m.data(), nRows, root);
}
inline void bcast(matrix& m, int nRows, int nCols, MPIUtil* mpiUtil, int root)
{	if(!m) m.init(nRows, nCols);
	mpiUtil->bcast(m.data(), m.nData(), root);
}

template<typename T> vector3<T> elemwiseProd(vector3<int> a, vector3<T> b)
{	return vector3<T>(a[0]*b[0], a[1]*b[1], a[2]*b[2]);
}

//Loop i till iStop, sampling a 3D mesh of dimensions S
//At each point, set fractional coordinates x offset by x0 and run code
#define PartialLoop3D(S, i, iStop, x, x0, code) \
	vector3<int> i##v( \
		i / (S[2]*S[1]), \
		(i/S[2]) % S[1], \
		i % S[2] ); \
	vector3<> i##Frac(1./S[0], 1./S[1], 1./S[2]); \
	while(i<iStop) \
	{	\
		x = x0 + vector3<>(i##v[0]*i##Frac[0], i##v[1]*i##Frac[1], i##v[2]*i##Frac[2]); \
		code \
		\
		i++; if(i==iStop) break; \
		i##v[2]++; \
		if(i##v[2]==S[2]) \
		{	i##v[2]=0; \
			i##v[1]++; \
			if(i##v[1]==S[1]) \
			{	i##v[1] = 0; \
				i##v[0]++; \
			} \
		} \
	}

void WannierMC::eLoop(const vector3<>& k0, WannierMC::eProcessFunc eProcess, void* params)
{	//Run Fourier transforms with this offset:
	Hw->transform(k0);
	if(wmcp.needVelocity)
		Pw->transform(k0);
	if(wmcp.needLinewidths)
	{	ImSigma_eeW->transform(k0);
		ImSigma_ePhW->transform(k0);
	}
	//Call eProcess for k-points on present process:
	int ik = Hw->ikStart;
	int ikStop = ik + Hw->nk;
	StateE state;
	PartialLoop3D(kfold, ik, ikStop, state.k, k0,
		setState(ik, state);
		eProcess(state, params);
	)
}

void WannierMC::phLoop(const vector3<>& q0, WannierMC::phProcessFunc phProcess, void* params)
{	assert(wmcp.needPhonons);
	//Run Fourier transforms with this offset:
	OsqW->transform(q0);
	//Call phProcess for q-points on present process:
	int iq = OsqW->ikStart;
	int iqStop = iq + OsqW->nk;
	StatePh state;
	PartialLoop3D(phononSup, iq, iqStop, state.q, q0,
		setState(iq, state);
		phProcess(state, params);
	)
}

void WannierMC::ePhLoop(const vector3<>& k01, const vector3<>& k02, WannierMC::ePhProcessFunc ePhProcess, void* params)
{	static StopWatch watchBcast("WannierMC::ePhLoop:bcast"), watchRotations("WannierMC::ePhLoop:rotations");
	assert(wmcp.needPhonons);
	int prodKfold = Hw->nkTot;
	int prodSup = OsqW->nkTot;
	int prodSupSq = HePhW->nkTot;
	assert(prodSupSq == prodSup*prodSup);
	//Initialize electronic states for 1 and 2:
	#define PrepareElecStates(i) \
		std::vector<StateE> e##i(prodKfold); /* States */ \
		std::vector<matrix> V##i(prodKfold); /* Eigenvectors */ \
		{	Hw->transform(k0##i); \
			if(wmcp.needVelocity) \
				Pw->transform(k0##i); \
			if(wmcp.needLinewidths) \
			{	ImSigma_eeW->transform(k0##i); \
				ImSigma_ePhW->transform(k0##i); \
			} \
			int ik = Hw->ikStart; \
			int ikStop = ik + Hw->nk; \
			PartialLoop3D(kfold, ik, ikStop, e##i[ik].k, k0##i, \
				setState(ik, e##i[ik], &V##i[ik]); \
			) \
			/* Make available on all processes of group */ \
			if(mpiGroup->nProcesses() > 1) \
			{	watchBcast.start(); \
				for(int whose=0; whose<mpiGroup->nProcesses(); whose++) \
					for(int ik=Hw->ikStartProc[whose]; ik<Hw->ikStartProc[whose+1]; ik++) \
						bcastState(e##i[ik], mpiGroup, whose, &V##i[ik]); \
				watchBcast.stop(); \
			} \
		}
	PrepareElecStates(1) //prepares e1 and V1
	PrepareElecStates(2) //prepares e2 and V2
	#undef PrepareElecStates
	//Loop over phonon q offsets:
	vector3<> kfoldInv(1./kfold[0], 1./kfold[1], 1./kfold[2]);
	vector3<int> iqSup;
	for(iqSup[0]=0; iqSup[0]<kfoldSup[0]; iqSup[0]++)
	for(iqSup[1]=0; iqSup[1]<kfoldSup[1]; iqSup[1]++)
	for(iqSup[2]=0; iqSup[2]<kfoldSup[2]; iqSup[2]++)
	{	//Prepare phonon states:
		vector3<> q0 = k01 - k02 + elemwiseProd(iqSup, kfoldInv);
		OsqW->transform(q0);
		std::vector<StatePh> ph(prodSup);
		std::vector<matrix> Vph(prodSup);
		{	int iq = OsqW->ikStart;
			int iqStop = iq + OsqW->nk;
			PartialLoop3D(phononSup, iq, iqStop, ph[iq].q, q0,
				setState(iq, ph[iq], &Vph[iq]);
			)
			//Make available on all processes of group:
			if(mpiGroup->nProcesses() > 1)
			{	watchBcast.start();
				for(int whose=0; whose<mpiGroup->nProcesses(); whose++)
					for(int iq=OsqW->ikStartProc[whose]; iq<OsqW->ikStartProc[whose+1]; iq++)
					{	mpiGroup->bcast(&ph[iq].q[0], 3, whose);
						bcast(ph[iq].omega, nModes, mpiGroup, whose);
						bcast(Vph[iq], nModes, nModes, mpiGroup, whose);
					}
				watchBcast.stop();
			}
		}
		//Loop over k2 supercell offsets:
		vector3<int> ik2sup;
		for(ik2sup[0]=0; ik2sup[0]<kfoldSup[0]; ik2sup[0]++)
		for(ik2sup[1]=0; ik2sup[1]<kfoldSup[1]; ik2sup[1]++)
		for(ik2sup[2]=0; ik2sup[2]<kfoldSup[2]; ik2sup[2]++)
		{	vector3<> k02cur = k02 + elemwiseProd(ik2sup, kfoldInv);
			vector3<int> ik1sup = iqSup + ik2sup; //momentum conservation
			vector3<> k01cur = q0 + k02cur; //momentum conservation
			//Calculate electron-phonon matrix elements:
			HePhW->transform(k01cur, k02cur);
			int ikPair = 0;
			int ikPairStart = HePhW->ikStart;
			int ikPairStop = ikPairStart + HePhW->nk;
			int ik1 = 0; vector3<> k1;
			PartialLoop3D(phononSup, ik1, prodSup, k1, k01cur,
				int ik2 = 0; vector3<> k2;
				PartialLoop3D(phononSup, ik2, prodSup, k2, k02cur,
					if(ikPair>=ikPairStart && ikPair<ikPairStop)
					{	watchRotations.start();
						//Get the matrix elements for all modes together:
						MatrixEph m;
						matrix Mall = getMatrix(HePhW->getResult(ikPair), nBands*nBands, nModes);
						//Apply associated phonon transformation:
						int iqIndex = calculateIndex(ik1v - ik2v, phononSup);
						m.ph = &ph[iqIndex];
						Mall = Mall * Vph[iqIndex]; //to phonon eigenbasis
						//Identify associated electronic states:
						int ik1net = calculateIndex(ik1sup + elemwiseProd(kfoldSup, ik1v), kfold);
						int ik2net = calculateIndex(ik2sup + elemwiseProd(kfoldSup, ik2v), kfold);
						m.e1 = &e1[ik1net];
						m.e2 = &e2[ik2net];
						//Extract matrices for each phonon mode:
						m.M.resize(nModes);
						for(int iMode=0; iMode<nModes; iMode++)
							m.M[iMode] = sqrt(0.5/m.ph->omega[iMode]) //frequency-dependent phonon amplitude
								* (dagger(V1[ik1net]) * getMatrix(Mall.data(), nBands, nBands, iMode) * V2[ik2net]); //to E1 and E2 eigenbasis
						watchRotations.stop();
						//Invoke call-back function:
						ePhProcess(m, params);
					}
					ikPair++;
				)
			)
		}
	}
}


void WannierMC::setState(int ik, WannierMC::StateE& state, matrix* Vptr)
{	//Get and diagonalize Hamiltonian:
	matrix Vk, Hk = getMatrix(Hw->getResult(ik), nBands, nBands);
	Hk.diagonalize(Vk, state.E);
	for(double& E: state.E) E -= mu; //reference to Fermi level
	if(Vptr) *Vptr = Vk;
	//Velcoity matrix, if needed:
	if(wmcp.needVelocity)
	{	state.vVec.resize(nBands);
		for(int iDir=0; iDir<3; iDir++)
		{	state.v[iDir] = complex(0,1) //Since P was stored with i omitted (to make it real when possible)
				* (dagger(Vk) * getMatrix(Pw->getResult(ik), nBands, nBands, iDir) * Vk);
			//Extract diagonal parts for convenience:
			for(int b=0; b<nBands; b++)
				state.vVec[b][iDir] = state.v[iDir](b,b).real();
		}
	}
	//Linewidths, if needed:
	if(wmcp.needLinewidths)
	{	//e-e linewidth:
		state.ImE = diag(dagger(Vk) * getMatrix(ImSigma_eeW->getResult(ik), nBands, nBands) * Vk);
		//add e-ph linewidth:
		diagMatrix logImE_ePh = diag(dagger(Vk) * getMatrix(ImSigma_ePhW->getResult(ik), nBands, nBands) * Vk);
		for(int b=0; b<nBands; b++)
			state.ImE[b] += exp(logImE_ePh[b]); //e-ph linewidth interpolated in logarithm
	}
}

void WannierMC::bcastState(WannierMC::StateE& state, MPIUtil* mpiUtil, int root, matrix* Vptr)
{	if(mpiUtil->nProcesses()==1) return; //noc ommunictaion needed
	mpiUtil->bcast(&state.k[0], 3, root);
	//Energy and eigenvectors:
	bcast(state.E, nBands, mpiUtil, root);
	if(Vptr)
		bcast(*Vptr, nBands, nBands, mpiUtil, root);
	//Velcoity matrix, if needed:
	if(wmcp.needVelocity)
	{	for(int iDir=0; iDir<3; iDir++)
			bcast(state.v[iDir], nBands, nBands, mpiUtil, root);
		state.vVec.resize(nBands);
		mpiUtil->bcast(&state.vVec[0][0], 3*nBands, root);
	}
	//Linewidths, if needed:
	if(wmcp.needLinewidths)
		bcast(state.ImE, nBands, mpiUtil, root);
}


void WannierMC::setState(int iq, WannierMC::StatePh& state, matrix* Vptr)
{	assert(wmcp.needPhonons);
	//Get and diagonalize force matrix:
	matrix Vq, Osqq = getMatrix(OsqW->getResult(iq), nModes, nModes);
	Osqq.diagonalize(Vq, state.omega);
	for(double& omega: state.omega) omega = sqrt(std::max(0.,omega)); //convert to phonon frequency; discard imaginary
	if(Vptr) *Vptr = Vq;
}

/*
void WannierMC::setPhononMatElemArray(vector3<> k1, const std::vector< vector3<> >& k2arr, std::vector<matrix>* result) const
{	static StopWatch watch("WannierMC::getPhononMatElem"), watchFT("WannierMC::getPhononMatEl_FT"); watch.start();
	int nk2 = k2arr.size();
	//Compute double Fourier transform for fixed k1 and all k2 together:
	watchFT.start();
	matrix phase1(phononCellMap.size(), 1);
	matrix phase2(phononCellMap.size(), nk2);
	for(size_t iCell=0; iCell<phononCellMap.size(); iCell++)
		phase1.set(iCell,0, cis(-2*M_PI * dot(phononCellMap[iCell],k1)));
	for(int ik2=0; ik2<nk2; ik2++)
	{	vector3<> k2 = k2arr[ik2];
		//Calculate Fourier transform phase:
		for(size_t iCell=0; iCell<phononCellMap.size(); iCell++)
			phase2.set(iCell,ik2, cis(2*M_PI * dot(phononCellMap[iCell],k2)));
	}
	matrix HePhArr = wannierHePh * phase1; //fourier transform for k1
	HePhArr.reshape(0, phononCellMap.size());
	HePhArr = HePhArr * phase2; //fourier transform for each k2; now a nBands*nBands*nModes x nk2 matrix
	watchFT.stop();
	//Get electronic caches for all k-points together:
	std::vector< vector3<> > kAll = k2arr;
	kAll.push_back(k1);
	std::vector< std::shared_ptr<const CacheEntry> > cElAll = getElectronCache(kAll);
	const CacheEntry& cEl1 = *(cElAll.back());
	//Get phonon caches for all k-points together:
	std::vector< vector3<> > qAll(nk2);
	for(int ik2=0; ik2<nk2; ik2++)
		qAll[ik2] = k1 - k2arr[ik2];
	std::vector< std::shared_ptr<const CacheEntry> > cPhAll = getPhononCache(qAll);
	//Now process one k2 at a time:
	for(int ik2=0; ik2<nk2; ik2++)
	{	matrix HePh = HePhArr(0,HePhArr.nRows(), ik2,ik2+1);
		HePh.reshape(nPacked, nModes); //now each column is matrix elements for a specific nuclear displacement
		//Apply unitary transformations:
		HePh = HePh * cPhAll[ik2]->evecs; //phonon unitary rotation
		const CacheEntry& cEl2 = *(cElAll[ik2]);
		result[ik2].resize(nModes);
		for(int alpha=0; alpha<nModes; alpha++)
			result[ik2][alpha] = sqrt(0.5/cPhAll[ik2]->eigs[alpha]) //frequency-dependent phonon amplitude
				* (dagger(cEl1.evecs) * unpackMatElem(HePh, alpha) * cEl2.evecs); //electron unitary rotations
	}
	watch.stop();
}
*/
