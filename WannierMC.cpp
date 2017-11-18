#include "WannierMC.h"
#include <core/BlasExtra.h>
#include <core/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <set>
#include <core/Units.h>
#include <fftw3-mpi.h>
#include "config.h"

WannierMCParams::WannierMCParams()
: totalEprefix("Wannier/totalE"), phononPrefix("Wannier/phonon"), wannierPrefix("Wannier/wannier"),
needPhonons(false), needLinewidths(false)
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
}

/*
diagMatrix WannierMC::getStates(vector3<> k, double omegaMax, matrix* evecs) const
{	return getStates(std::vector< vector3<> >(1, k), omegaMax, evecs)[0];
}

std::vector<diagMatrix> WannierMC::getStates(const std::vector< vector3<> >& kArr, double omegaMax, matrix* evecs) const
{	static StopWatch watch("WannierMC::getStates"); watch.start();
	std::vector< std::shared_ptr<const CacheEntry> > ceArr = getElectronCache(kArr, omegaMax);
	std::vector< diagMatrix > eigs(kArr.size());
	for(size_t ik=0; ik<kArr.size(); ik++)
	{	eigs[ik] = ceArr[ik]->eigs;
		if(evecs) evecs[ik] = ceArr[ik]->evecs;
	}
	watch.stop();
	return eigs;
}


diagMatrix WannierMC::getPhononModes(vector3<> q) const
{	static StopWatch watch("WannierMC::getPhononModes"); watch.start();
	std::shared_ptr<const CacheEntry> ce = getPhononCache(q);
	watch.stop();
	return ce->eigs;
}

std::vector<matrix> WannierMC::getDipoleMatElem(vector3<> k) const
{	return getDipoleMatElem(std::vector<vector3<>>(1, k))[0];
}

std::vector< std::vector<matrix> > WannierMC::getDipoleMatElem(const std::vector< vector3<> >& kArr) const
{	static StopWatch watch("WannierMC::getDipoleMatElem"); watch.start();
	assert(nPol);
	std::vector< std::shared_ptr<const WannierMC::CacheEntry> > ceArr = getElectronCache(kArr);
	//Collect phases:
	matrix phase(cellMap.size(), kArr.size());
	for(size_t ik=0; ik<kArr.size(); ik++)
		phase.set(0,phase.nRows(), ik,ik+1, ceArr[ik]->phase);
	//Single matrix multiply for Fourier transforms:
	matrix PkArr = pWannier * phase;
	//Apply unitary rotations individually:
	std::vector< std::vector<matrix> > out(kArr.size());
	for(size_t ik=0; ik<kArr.size(); ik++)
	{	const matrix& evecs = ceArr[ik]->evecs;
		matrix Pk = PkArr(0,PkArr.nRows(), ik,ik+1);
		Pk.reshape(nPacked, nPol);
		out[ik].resize(nPol);
		for(int iPol=0; iPol<nPol; iPol++)
			out[ik][iPol] = dagger(evecs) * unpackMatElem(Pk,iPol) * evecs; //switch to eigenbasis of Hk
	}
	watch.stop();
	return out;
}

std::vector<matrix> WannierMC::getPhononMatElem(vector3<> k1, vector3<> k2) const
{	std::vector<matrix> result;
	setPhononMatElemArray(k1, std::vector< vector3<> >(1, k2), &result);
	return result;
}

void WannierMC::setPhononMatElemArray(vector3<> k1, const std::vector< vector3<> >& k2arr, std::vector<matrix>* result) const
{	static StopWatch watch("WannierMC::getPhononMatElem"), watchFT("WannierMC::getPhononMatEl_FT"); watch.start();
	int nk2 = k2arr.size();
	//Compute double Fourier transform for fixed k1 and all k2 together:
	watchFT.start();
	std::vector< vector3<> > kMeanArr(nk2);
	matrix phase1(phononCellMap.size(), 1);
	matrix phase2(phononCellMap.size(), nk2);
	for(size_t iCell=0; iCell<phononCellMap.size(); iCell++)
		phase1.set(iCell,0, cis(-2*M_PI * dot(phononCellMap[iCell],k1)));
	for(int ik2=0; ik2<nk2; ik2++)
	{	vector3<> k2 = k2arr[ik2];
		//Get bisecting k-point (within nearest image convention):
		vector3<> kDiff = k2 - k1;
		for(int j=0; j<3; j++) kDiff[j] -= floor(kDiff[j]+0.5);
		vector3<> kMean = k1 + 0.5*kDiff;
		kMeanArr[ik2] = kMean;
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

std::vector< vector3<> > WannierMC::getVelocity(vector3<> k, double omegaMax) const
{	static StopWatch watch("WannierMC::getVelocity"); watch.start();
	std::shared_ptr<const CacheEntry> ce = getElectronCache(k, omegaMax);
	int nBandsEff = ce->nBands();
	const matrix& hWannierEff = nBandsEff==nBands ? hWannier : hWannierMain;
	std::vector< vector3<> > v(nBandsEff);
	for(int j = 0; j < 3; j++)
	{	matrix phasePrime = ce->phase;
		complex* phasePrimeData = phasePrime.data();
		for(size_t ic=0; ic<cellMap.size(); ic++)
			phasePrimeData[ic] *= complex(0,cellMap[ic][j]); //multiply phase by I*iR[j] to get phasePrime
		matrix dHdk = hWannierEff * phasePrime;
		dHdk.reshape(nBandsEff, nBandsEff);
		diagMatrix vj = diag(dagger(ce->evecs) * dHdk * ce->evecs);
		for(int b=0; b<nBandsEff; b++) v[b][j] = vj[b];
	}
	for(vector3<>& vb: v) vb = R * vb; //Convert to Cartesian
	watch.stop();
	return v;
}


//------------ Cache functions -------------

std::shared_ptr<const WannierMC::CacheEntry> WannierMC::getElectronCache(vector3<> k, double omegaMax) const
{	return getElectronCache(std::vector< vector3<> >(1, k), omegaMax)[0];
}

std::shared_ptr<const WannierMC::CacheEntry> WannierMC::getPhononCache(vector3<> q) const
{	return getPhononCache(std::vector< vector3<> >(1, q))[0];
}

std::vector< std::shared_ptr<const WannierMC::CacheEntry> > WannierMC::getElectronCache(const std::vector< vector3<> >& kArr, double omegaMax) const
{	if(omegaMax<omegaMain)
		return getCache(kArr, mainCache, cellMap, hWannierMain, rankMain, false);
	else
		return getCache(kArr, electronCache, cellMap, hWannier, rankElectron, false);
}

std::vector< std::shared_ptr<const WannierMC::CacheEntry> > WannierMC::getPhononCache(const std::vector< vector3<> >& qArr) const
{	return getCache(qArr, phononCache, phononCellMap, omegaSqPh, rankPhonon, true);
}

std::vector< std::shared_ptr<const WannierMC::CacheEntry> > WannierMC::getCache( const std::vector< vector3<> >& kArr, 
		const std::map<vector3<>, std::shared_ptr<const CacheEntry> >& cache, const std::vector< vector3<int> >& cellMap,
		const matrix& hWannierEff, const size_t& rank, bool shouldSqrt) const
{	static StopWatch watch("WannierMC::getCache");
	std::vector< std::shared_ptr<const WannierMC::CacheEntry> > ceArr(kArr.size());
	std::vector< vector3<> > kNew; //k's for which results not yet available in cache
	//Check cache first:
	for(size_t ik=0; ik<kArr.size(); ik++)
	{	const vector3<>& k = kArr[ik];
		auto iter = cache.find(k);
		if(iter == cache.end()) kNew.push_back(k);
		else ceArr[ik] = iter->second;
	}
	if(!kNew.size()) return ceArr;
	//Generate ones not found in cache:
	watch.start(); //only time when actually producing new entries
	//--- combine Fourier transforms into a single BLAS3:
	matrix phase(cellMap.size(), kNew.size());
	for(size_t ikNew=0; ikNew<kNew.size(); ikNew++)
	{	const vector3<>& k = kNew[ikNew];
		for(size_t ic=0; ic<cellMap.size(); ic++)
			phase.set(ic,ikNew, cis(2*M_PI*dot(cellMap[ic],k)));
	}
	matrix HkNew = hWannierEff * phase;
	//--- need to do the remainder (diagonalization etc.) individually:
	typedef std::map<vector3<>, std::shared_ptr<const CacheEntry> > Cache;
	Cache& cacheMod = (Cache&)cache; //!< modifiable copy
	size_t ikNew = 0;
	for(size_t ik=0; ik<kArr.size(); ik++)
	{	if(ceArr[ik]) continue; //result already obtained from cache
		const vector3<>& k = kNew[ikNew];
		auto ce = std::make_shared<CacheEntry>();
		ce->rank = (((size_t&)rank)++);
		//--- collect phase, Hamiltonian at current k and diagonalize:
		ce->phase = phase(0,phase.nRows(), ikNew,ikNew+1);
		matrix Hk = HkNew(0,HkNew.nRows(), ikNew,ikNew+1);
		int nBandsEff = round(sqrt(HkNew.nRows()));
		Hk.reshape(nBandsEff, nBandsEff);
		Hk = dagger_symmetrize(Hk);
		Hk.diagonalize(ce->evecs, ce->eigs);
		if(shouldSqrt)
		{	for(double& x: ce->eigs)
				x = sqrt(fabs(x));
		}
		ceArr[ik] = ce;
		cacheMod[k] = ce; //add to cache
		ikNew++;
	}
	assert(ikNew == kNew.size());
	//Clean up cache if necessary:
	if(cacheMod.size() > 2*cacheSize)
	{	size_t rankMin = rank - cacheSize; //remove everything older
		for(auto iter=cacheMod.begin(); iter!=cacheMod.end(); )
		{	auto iterNext = iter; iterNext++;
			if(iter->second->rank < rankMin)
				cacheMod.erase(iter);
			iter = iterNext;
		}
	}
	watch.stop();
	return ceArr;
}
*/
