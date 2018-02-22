#include "WannierMC.h"
#include <core/BlasExtra.h>
#include <core/Random.h>
#include <fftw3-mpi.h>
#include "config.h"

WannierMCParams::WannierMCParams()
: iSpin(0), totalEprefix("Wannier/totalE"), phononPrefix("Wannier/phonon"), wannierPrefix("Wannier/wannier"),
needSymmetries(false), needCellWeights(false), needPhonons(false), needVelocity(false),
needLinewidth_ee(false), needLinewidth_ePh(false), needLinewidthP_ePh(false)
{
}

//Fillings grid on [0,1] for which to calculate e-ph linewidths
inline std::vector<double> getFgrid(int nInterp)
{	std::vector<double> fGrid(nInterp+1);
	double df = 1./nInterp;
	for(int i=0; i<=nInterp; i++)
		fGrid[i] = i*df;
	return fGrid;
}
const std::vector<double> WannierMCParams::fGrid_ePh = getFgrid(4);

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

std::vector<vector3<int>> readCellMap(string fname)
{	logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
	ifstream ifs(fname); if(!ifs.is_open()) die("could not open file.\n");
	string headerLine; getline(ifs, headerLine); //read and ignore header line
	std::vector<vector3<int>> cellMap;
	vector3<int> cm; //lattice coords version (store)
	vector3<> Rcm; //cartesian version (ignore)
	while(ifs >> cm[0] >> cm[1] >> cm[2] >> Rcm[0] >> Rcm[1] >> Rcm[2])
		cellMap.push_back(cm);
	ifs.close();
	logPrintf("done.\n");
	return cellMap;
}

WannierMC::WannierMC(const WannierMCParams& wmcp)
: wmcp(wmcp), nSpins(0), nSpinor(0), spinWeight(0), mu(NAN), nElectrons(0)
{	
	//Read relevant parameters from totalE.out:
	string fname = wmcp.totalEprefix + ".out";
	logPrintf("\nReading '%s' ... ", fname.c_str()); logFlush();
	ifstream ifs(fname); if(!ifs.is_open()) die("could not open file.\n");
	bool initDone = false; //whether finished reading the initialization part of totalE.out
	int nBandsDFT = 0; //number of DFT bands (>= this->nBands = # Wannier bands)
	int nStatesDFT = 0; //number of reduced k-pts * spins in DFT
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
			{	nSpins = 1;
				nSpinor = 1;
			}
			else if(spinString == "z-spin")
			{	nSpins = 2;
				nSpinor = 1;
			}
			else //non-collinear modes
			{	nSpins = 1;
				nSpinor = 2;
			}
			spinWeight = 2/(nSpins*nSpinor);
			if(wmcp.iSpin<0 || wmcp.iSpin>=nSpins)
				die("iSpin = %d not in interval [0,nSpins), where nSpins = %d for this system.\n\n", wmcp.iSpin, nSpins);
			spinSuffix = (nSpins==1 ? "" : (wmcp.iSpin==0 ? "Up" : "Dn"));
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
		else if(line.find("Initialization completed") == 0)
		{	initDone = true;
		}
		else if(initDone && (line.find("FillingsUpdate:") != string::npos))
		{	istringstream iss(line); string buf;
			iss >> buf >> buf >> mu >> buf >> nElectrons;
		}
		else if(initDone && (line.find("# Ionic positions in") != string::npos))
		{	atpos.clear(); //read last version (if many ionic steps in totalE.out)
			atNames.clear();
			bool cartesian = (line.find("cartesian") != string::npos);
			while(true)
			{	getline(ifs, line);
				istringstream iss(line);
				string cmd, atName; vector3<> x;
				iss >> cmd >> atName >> x[0] >> x[1] >> x[2]; //rest (move flag etc. not needed)
				if(cmd != "ion") break;
				if(cartesian) x = inv(R) * x; //convert to lattice
				atpos.push_back(x);
				atNames.push_back(atName);
			}
		}
		else if(line.find("nElectrons:") == 0) //nElectrons, nBands, nStates line
		{	istringstream iss(line); string buf;
			iss >> buf >> nElectrons >> buf >> nBandsDFT >> buf >> nStatesDFT;
		}
	}
	ifs.close();
	logPrintf("done.\n"); logFlush();
	if(!nSpins)
		die("Could not determine spin configuration from DFT output file.");
	if(std::isnan(mu))
	{	logPrintf("NOTE: mu unavailable; assuming semiconductor/insulator and setting to VBM.\n");
		int nValence = int(round(nElectrons/(nSpins*spinWeight))); //number of valence bands
		if(fabs(nValence*nSpins*spinWeight-nElectrons > 1e-6))
			die("Number of electrons incompatible with semiconductor / insulator.\n");
		//Read DFT eigenvalues file:
		ManagedArray<double> Edft; Edft.init(nBandsDFT*nStatesDFT);
		fname = wmcp.totalEprefix + ".eigenvals";
		logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
		Edft.read(fname.c_str());
		logPrintf("done.\n");
		//Find VBM:
		mu = -DBL_MAX;
		for(int q=0; q<nStatesDFT; q++)
			mu = std::max(mu, Edft.data()[q*nBandsDFT+nValence-1]); //highest valence eigenvalue at each q
	}
	logPrintf("mu = %lg\n", mu);
	logPrintf("nElectrons = %lg\n", nElectrons);
	logPrintf("nBandsDFT = %d\n", nBandsDFT);
	logPrintf("nSpins = %d\n", nSpins);
	logPrintf("nSpinor = %d\n", nSpinor);
	logPrintf("spinSuffix = '%s'\n", spinSuffix.c_str());
	logPrintf("kfold = "); kfold.print(globalLog, " %d ");
	logPrintf("isTruncated = "); isTruncated.print(globalLog, " %d ");
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	logPrintf("Atoms with fractional coordinates:\n");
	for(unsigned i=0; i<atpos.size(); i++)
		logPrintf("\t%2s %19.15lf %19.15lf %19.15lf\n",
			atNames[i].c_str(), atpos[i][0], atpos[i][1], atpos[i][2]);
	logPrintf("\n");
	
	//Read symmetries if required
	if(wmcp.needSymmetries)
	{	fname = wmcp.totalEprefix + ".sym";
		logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
		ifs.open(fname); if(!ifs.is_open()) die("could not open file.\n");
		sym.clear();
		while(!ifs.eof())
		{	SpaceGroupOp op;
			for(int i=0; i<3; i++) for(int j=0; j<3; j++) ifs >> op.rot(i,j); //rotation
			for(int i=0; i<3; i++) ifs >> op.a[i]; //translation
			if(ifs.good()) sym.push_back(op);
		}
		ifs.close();
		logPrintf("done. Read %lu symmetries.\n", sym.size());
	}
	
	//Read cell map
	cellMap = readCellMap(wmcp.wannierPrefix + ".mlwfCellMap" + spinSuffix);
	
	//Find number of wannier centers from Wannier band contrib file:
	{	fname = wmcp.wannierPrefix + ".mlwfBandContrib" + spinSuffix;
		logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
		FILE* fp = fopen(fname.c_str(), "r");
		if(!fp) die("could not open file.\n");
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
	realPartOnly = (nSpinor==1);
	fname = wmcp.wannierPrefix + ".mlwfH" + spinSuffix;
	Hw = std::make_shared<DistributedMatrix>(fname, realPartOnly,
		mpiGroup, nBands*nBands, cellMap, kfold, false);
	
	//Read cell weights (if needed):
	if(wmcp.needCellWeights)
	{	fname = wmcp.wannierPrefix + ".mlwfCellWeights" + spinSuffix;
		logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
		cellWeights.init(nBands*nBands, cellMap.size());
		cellWeights.read_real(fname.c_str());
		logPrintf("done.\n");
	}
	
	if(wmcp.needPhonons)
	{	//Read relevant parameters from phonon.out:
		fname = wmcp.phononPrefix + ".out";
		logPrintf("\nReading '%s' ... ", fname.c_str()); logFlush();
		ifs.open(fname); if(!ifs.is_open()) die("could not open file.\n");
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
		logPrintf("\n");
		
		//Read phonon cell map
		phononCellMap = readCellMap(wmcp.wannierPrefix + ".mlwfCellMapPh" + spinSuffix);
		
		//Read phonon force matrix
		fname = wmcp.wannierPrefix + ".mlwfOmegaSqPh" + spinSuffix;
		OsqW = std::make_shared<DistributedMatrix>(fname, true, //phonon omegaSq is always real
			mpiGroup, nModes*nModes, phononCellMap, phononSup, false);
		
		//Read electron-phonon matrix elements
		fname = wmcp.wannierPrefix + ".mlwfHePh" + spinSuffix;
		HePhW = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, nModes*nBands*nBands, phononCellMap, phononSup, true);
	}
	
	//Velocity matrix elements
	if(wmcp.needVelocity)
	{	fname = wmcp.wannierPrefix + ".mlwfP" + spinSuffix;
		Pw = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, 3*nBands*nBands, cellMap, kfold, false);
	}
	
	//Linewidths:
	if(wmcp.needLinewidth_ee)
	{	//e-e:
		fname = wmcp.wannierPrefix + ".mlwfImSigma_ee" + spinSuffix;
		ImSigma_eeW = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, nBands*nBands, cellMap, kfold, false);
	}
	if(wmcp.needLinewidth_ePh)
	{	//e-ph:
		fname = wmcp.wannierPrefix + ".mlwfImSigma_ePh" + spinSuffix;
		ImSigma_ePhW = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, nBands*nBands*WannierMCParams::fGrid_ePh.size(), cellMap, kfold, false);
	}
	if(wmcp.needLinewidthP_ePh)
	{	//e-ph:
		fname = wmcp.wannierPrefix + ".mlwfImSigmaP_ePh" + spinSuffix;
		ImSigmaP_ePhW = std::make_shared<DistributedMatrix>(fname, realPartOnly,
			mpiGroup, nBands*nBands*WannierMCParams::fGrid_ePh.size(), cellMap, kfold, false);
	}
	
	logPrintf("\n");
}

void WannierMC::free()
{	Hw = 0;
	Pw = 0;
	ImSigma_eeW = 0;
	ImSigma_ePhW = 0;
	ImSigmaP_ePhW = 0;
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
{	static StopWatch watchCallback("WannierMC::eLoop:callback");
	//Run Fourier transforms with this offset:
	Hw->transform(k0);
	if(wmcp.needVelocity)
		Pw->transform(k0);
	if(wmcp.needLinewidth_ee) ImSigma_eeW->transform(k0);
	if(wmcp.needLinewidth_ePh) ImSigma_ePhW->transform(k0);
	if(wmcp.needLinewidthP_ePh) ImSigmaP_ePhW->transform(k0);
	//Call eProcess for k-points on present process:
	int ik = Hw->ikStart;
	int ikStop = ik + Hw->nk;
	StateE state;
	PartialLoop3D(kfold, ik, ikStop, state.k, k0,
		state.ik = ik;
		setState(state);
		watchCallback.start();
		eProcess(state, params);
		watchCallback.stop();
	)
}

void WannierMC::phLoop(const vector3<>& q0, WannierMC::phProcessFunc phProcess, void* params)
{	static StopWatch watchCallback("WannierMC::phLoop:callback");
	assert(wmcp.needPhonons);
	//Run Fourier transforms with this offset:
	OsqW->transform(q0);
	//Call phProcess for q-points on present process:
	int iq = OsqW->ikStart;
	int iqStop = iq + OsqW->nk;
	StatePh state;
	PartialLoop3D(phononSup, iq, iqStop, state.q, q0,
		state.iq = iq;
		setState(state);
		watchCallback.start();
		phProcess(state, params);
		watchCallback.stop();
	)
}

void WannierMC::ePhLoop(const vector3<>& k01, const vector3<>& k02, WannierMC::ePhProcessFunc ePhProcess, void* params)
{	static StopWatch watchBcast("WannierMC::ePhLoop:bcast"); 
	static StopWatch watchRotations("WannierMC::ePhLoop:rotations");
	static StopWatch watchCallback("WannierMC::ePhLoop:callback");
	assert(wmcp.needPhonons);
	int prodKfold = Hw->nkTot;
	int prodSup = OsqW->nkTot;
	int prodSupSq = HePhW->nkTot;
	assert(prodSupSq == prodSup*prodSup);
	//Initialize electronic states for 1 and 2:
	#define PrepareElecStates(i) \
		std::vector<StateE> e##i(prodKfold); /* States */ \
		{	Hw->transform(k0##i); \
			if(wmcp.needVelocity) \
				Pw->transform(k0##i); \
			if(wmcp.needLinewidth_ee) ImSigma_eeW->transform(k0##i); \
			if(wmcp.needLinewidth_ePh) ImSigma_ePhW->transform(k0##i); \
			if(wmcp.needLinewidthP_ePh) ImSigmaP_ePhW->transform(k0##i); \
			int ik = Hw->ikStart; \
			int ikStop = ik + Hw->nk; \
			PartialLoop3D(kfold, ik, ikStop, e##i[ik].k, k0##i, \
				e##i[ik].ik = ik; \
				setState(e##i[ik]); \
			) \
			/* Make available on all processes of group */ \
			if(mpiGroup->nProcesses() > 1) \
			{	watchBcast.start(); \
				for(int whose=0; whose<mpiGroup->nProcesses(); whose++) \
					for(int ik=Hw->ikStartProc[whose]; ik<Hw->ikStartProc[whose+1]; ik++) \
					{	e##i[ik].ik = ik; \
						bcastState(e##i[ik], mpiGroup, whose); \
					} \
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
		{	int iq = OsqW->ikStart;
			int iqStop = iq + OsqW->nk;
			PartialLoop3D(phononSup, iq, iqStop, ph[iq].q, q0,
				ph[iq].iq = iq;
				setState(ph[iq]);
			)
			//Make available on all processes of group:
			if(mpiGroup->nProcesses() > 1)
			{	watchBcast.start();
				for(int whose=0; whose<mpiGroup->nProcesses(); whose++)
					for(int iq=OsqW->ikStartProc[whose]; iq<OsqW->ikStartProc[whose+1]; iq++)
					{	ph[iq].iq = iq;
						bcastState(ph[iq], mpiGroup, whose);
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
						Mall = Mall * m.ph->U; //to phonon eigenbasis
						//Identify associated electronic states:
						int ik1net = calculateIndex(ik1sup + elemwiseProd(kfoldSup, ik1v), kfold);
						int ik2net = calculateIndex(ik2sup + elemwiseProd(kfoldSup, ik2v), kfold);
						m.e1 = &e1[ik1net];
						m.e2 = &e2[ik2net];
						//Extract matrices for each phonon mode:
						const double omegaPhCut = 1e-6;
						m.M.resize(nModes);
						for(int iMode=0; iMode<nModes; iMode++)
							m.M[iMode] = sqrt(m.ph->omega[iMode]<omegaPhCut ? 0. : 0.5/m.ph->omega[iMode]) //frequency-dependent phonon amplitude
								* (dagger(m.e1->U) * getMatrix(Mall.data(), nBands, nBands, iMode) * m.e2->U); //to E1 and E2 eigenbasis
						watchRotations.stop();
						//Invoke call-back function:
						watchCallback.start();
						ePhProcess(m, params);
						watchCallback.stop();
					}
					ikPair++;
				)
			)
		}
	}
}

void WannierMC::symmetrize(matrix3<>& m) const
{	matrix3<> mOut;
	matrix3<> invR = inv(R);
	for(const SpaceGroupOp& op: sym)
	{	matrix3<> rot = R * op.rot * invR; //convert to Cartesian
		mOut += rot * m * (~rot);
	}
	m = mOut *(1./sym.size());
	//Set near-zero to exact zero:
	double mCut = 1e-14*sqrt(trace((~m)*m));
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			if(fabs(m(i,j)) < mCut)
				m(i,j) = 0.;
}

void WannierMC::setState(WannierMC::StateE& state)
{	static StopWatch watchRotations("WannierMC::setState:rotations");
	//Get and diagonalize Hamiltonian:
	matrix Hk = getMatrix(Hw->getResult(state.ik), nBands, nBands);
	Hk.diagonalize(state.U, state.E);
	for(double& E: state.E) E -= mu; //reference to Fermi level
	watchRotations.start();
	//Velcoity matrix, if needed:
	if(wmcp.needVelocity)
	{	state.vVec.resize(nBands);
		for(int iDir=0; iDir<3; iDir++)
		{	state.v[iDir] = complex(0,-1) //Since P was stored with -i omitted (to make it real when possible)
				* (dagger(state.U) * getMatrix(Pw->getResult(state.ik), nBands, nBands, iDir) * state.U);
			//Extract diagonal parts for convenience:
			for(int b=0; b<nBands; b++)
				state.vVec[b][iDir] = state.v[iDir](b,b).real();
		}
	}
	//Linewidths, ad needed:
	if(wmcp.needLinewidth_ee)
		state.ImSigma_ee = diag(dagger(state.U) * getMatrix(ImSigma_eeW->getResult(state.ik), nBands, nBands) * state.U);
	if(wmcp.needLinewidth_ePh)
	{	state.logImSigma_ePhArr.resize(WannierMCParams::fGrid_ePh.size());
		for(unsigned iMat=0; iMat<state.logImSigma_ePhArr.size(); iMat++)
			state.logImSigma_ePhArr[iMat] = diag(dagger(state.U) * getMatrix(ImSigma_ePhW->getResult(state.ik), nBands, nBands, iMat) * state.U);
	}
	if(wmcp.needLinewidthP_ePh)
	{	state.logImSigmaP_ePhArr.resize(WannierMCParams::fGrid_ePh.size());
		for(unsigned iMat=0; iMat<state.logImSigmaP_ePhArr.size(); iMat++)
			state.logImSigmaP_ePhArr[iMat] = diag(dagger(state.U) * getMatrix(ImSigmaP_ePhW->getResult(state.ik), nBands, nBands, iMat) * state.U);
	}
	watchRotations.stop();
}

void WannierMC::bcastState(WannierMC::StateE& state, MPIUtil* mpiUtil, int root)
{	if(mpiUtil->nProcesses()==1) return; //no communictaion needed
	mpiUtil->bcast(&state.k[0], 3, root);
	//Energy and eigenvectors:
	bcast(state.E, nBands, mpiUtil, root);
	bcast(state.U, nBands, nBands, mpiUtil, root);
	//Velcoity matrix, if needed:
	if(wmcp.needVelocity)
	{	for(int iDir=0; iDir<3; iDir++)
			bcast(state.v[iDir], nBands, nBands, mpiUtil, root);
		state.vVec.resize(nBands);
		mpiUtil->bcast(&state.vVec[0][0], 3*nBands, root);
	}
	//Linewidths, if needed:
	if(wmcp.needLinewidth_ee) bcast(state.ImSigma_ee, nBands, mpiUtil, root);
	if(wmcp.needLinewidth_ePh)
	{	state.logImSigma_ePhArr.resize(WannierMCParams::fGrid_ePh.size());
		for(diagMatrix& d: state.logImSigma_ePhArr) bcast(d, nBands, mpiUtil, root);
	}
	if(wmcp.needLinewidthP_ePh)
	{	state.logImSigmaP_ePhArr.resize(WannierMCParams::fGrid_ePh.size());
		for(diagMatrix& d: state.logImSigmaP_ePhArr) bcast(d, nBands, mpiUtil, root);
	}
}


void WannierMC::setState(WannierMC::StatePh& state)
{	assert(wmcp.needPhonons);
	//Get and diagonalize force matrix:
	matrix Osqq = getMatrix(OsqW->getResult(state.iq), nModes, nModes);
	Osqq.diagonalize(state.U, state.omega);
	for(double& omega: state.omega) omega = sqrt(std::max(0.,omega)); //convert to phonon frequency; discard imaginary
}

void WannierMC::bcastState(WannierMC::StatePh& state, MPIUtil* mpiUtil, int root)
{	if(mpiUtil->nProcesses()==1) return; //no communictaion needed
	mpiUtil->bcast(&state.q[0], 3, root);
	bcast(state.omega, nModes, mpiUtil, root);
	bcast(state.U, nModes, nModes, mpiUtil, root);
}

//----------- class WannierMC::StateE -------------
inline double interpQuartic(const std::vector<diagMatrix>& Y, int n, double f)
{	//Get bernstein coeffs
	double a0 = Y[0][n];
	double a4 = Y[4][n];
	double a1 = (1./12)*(-13.*Y[0][n]+48.*Y[1][n]-36.*Y[2][n]+16.*Y[3][n]-3.*Y[4][n]);
	double a3 = (1./12)*(-13.*Y[4][n]+48.*Y[3][n]-36.*Y[2][n]+16.*Y[1][n]-3.*Y[0][n]);
	double a2 = (1./18)*(13.*(Y[0][n]+Y[4][n])-64.*(Y[1][n]+Y[3][n])+120.*Y[2][n]);
	//Evaluate bernstein polynomial
	//--- 1
	double b0 = a0+f*(a1-a0);
	double b1 = a1+f*(a2-a1);
	double b2 = a2+f*(a3-a2);
	double b3 = a3+f*(a4-a3);
	//--- 2
	double c0 = b0+f*(b1-b0);
	double c1 = b1+f*(b2-b1);
	double c2 = b2+f*(b3-b2);
	//--- 3
	double d0 = c0+f*(c1-c0);
	double d1 = c1+f*(c2-c1);
	//--- 4
	return d0+f*(d1-d0);
}
double WannierMC::StateE::ImSigma_ePh(int n, double f) const
{	return exp(interpQuartic(logImSigma_ePhArr, n, f));
}
double WannierMC::StateE::ImSigmaP_ePh(int n, double f) const
{	return exp(interpQuartic(logImSigmaP_ePhArr, n, f));
}
