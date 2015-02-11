#include "BandStruct.h"
#include <core/Util.h>
#include <core/BlasExtra.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <set>
#include "Units.h"

//Read matrix from file accounting for real-only or complex storage based on spinWeight
void readMatrix(matrix& m, string fname, int spinWeight)
{	if(spinWeight==1) m.read(fname.c_str()); else m.read_real(fname.c_str());
}

BandStruct::BandStruct(string prefix, double mu, int spinWeight, string phononPrefix, std::vector< vector3<complex> > Ahat)
: spinWeight(spinWeight), nPol(Ahat.size()), cacheSize(6)
{	//Read cell map
	ifstream ifs(prefix + ".mlwfCellMap");
	string headerLine; getline(ifs, headerLine); //read and ignore header line
	vector3<int> cm;
	double x,y,z;
	while(ifs >> cm[0] >> cm[1] >> cm[2] >> x >> y >> z)
		cellMap.push_back(cm);
	ifs.close();

	//Read wannier hamiltonian
	string fnameH = prefix + ".mlwfH";
	nBands = sqrt(fileSize(fnameH.c_str()) / ((spinWeight==1 ? sizeof(complex) : sizeof(double)) * cellMap.size()));
	hWannier.init(nBands*nBands, cellMap.size());
	readMatrix(hWannier, fnameH, spinWeight);

	//Offset wannier Hamiltonian by mu:
	for(size_t ic=0; ic<cellMap.size(); ic++)
		if(!cellMap[ic].length_squared()) //diagonal element
		{	matrix id = eye(nBands); id.reshape(nBands*nBands, 1);
			hWannier.set(0,nBands*nBands, ic,ic+1, hWannier(0,nBands*nBands, ic,ic+1) - mu * id);
		}

	//Read CEDA energies:
	Eceda.resize(nBands);
	{	string fname = prefix + ".mlwfCEDA";
		FILE* fp = fopen(fname.c_str(), "r");
		if(!fp) die("Could not open %s for reading.\n", fname.c_str());
		for(int b=0; b<nBands; b++)
		{	if(fscanf(fp, "%lf", &Eceda[b]) != 1)
				die("Error reading CEDA energies from %s.\n", fname.c_str());
				Eceda[b] -= mu;
		}
		fclose(fp);
	}
	
	//Read momentum matrix elements
	if(nPol)
	{	pWannier.init(nBands*nBands*3, cellMap.size());
		readMatrix(pWannier, prefix + ".mlwfP", spinWeight);
	}

	//Read momentum-squared matrix elements
	if(nPol == 2)
	{	pSqWannier.init(nBands*nBands*6, cellMap.size());
		readMatrix(pSqWannier, prefix + ".mlwfPsq", spinWeight);
	}

	//Initialize main window (if available):
	nMain = 0; mainFirst = 0; omegaMain = 0.;
	{	//read Wannier band contrib file
		string fname = prefix + ".mlwfBandContrib";
		FILE* fp = fopen(fname.c_str(), "r");
		if(!fp) die("Could not open %s for reading.\n", fname.c_str());
		double eMin = INFINITY, eMax = -INFINITY;
		mainFirst = 0;
		int mainLast = nBands;
		while(!feof(fp))
		{	int nMin_b, nMax_b; double eMin_b, eMax_b;
			if(fscanf(fp, "%d %d %lf %lf", &nMin_b, &nMax_b, &eMin_b, &eMax_b) != 4) break;
			//select tightest possible range that contains Fermi level:
			if(eMin_b<=mu && mu<=eMax_b && (nMax_b-nMin_b)<(mainLast-mainFirst))
			{	eMin = eMin_b; eMax = eMax_b;
				mainFirst = nMin_b; mainLast = nMax_b;
			}
		}
		fclose(fp);
		nMain = mainLast - mainFirst + 1;
		if(nMain < nBands) //otherwise no point
		{	omegaMain = std::min(mu-eMin, eMax-mu);
			if(omegaMain < 0.) omegaMain = 0.; //if Fermi level does not lie in [eMin,eMax]
		}
	}
	if(omegaMain)
	{	logPrintf("Initialized main window of half-width %lf eV with %d of %d Wannier centers.\n", omegaMain/eV, nMain, nBands);
		assert(nMain > 0 && nMain < nBands); //this is checked by Wannier as well
		//Initialize compressed Hamiltonian:
		hWannierMain.init(nMain*nMain, cellMap.size());
		for(size_t ic=0; ic<cellMap.size(); ic++)
		{	matrix H = hWannier(0,nBands*nBands, ic,ic+1);
			H.reshape(nBands, nBands);
			H = H(mainFirst,mainFirst+nMain, mainFirst,mainFirst+nMain); //extract the "main" part
			H.reshape(nMain*nMain, 1);
			hWannierMain.set(0,nMain*nMain, ic,ic+1, H);
		}
	}
	else nMain = nBands; //so that all Wannier centers are always used
	
	if(phononPrefix.length())
	{	//Read phonon cell map
		ifs.open((phononPrefix + ".phononCellMap").c_str());
		getline(ifs, headerLine); //read and ignore header line
		while(ifs >> cm[0] >> cm[1] >> cm[2] >> x >> y >> z)
			phononCellMap.push_back(cm);
		ifs.close();

		//Read phonon force matrix
		string phFile = phononPrefix + ".phononOmegaSq";
		nModes = sqrt(fileSize(phFile.c_str())/(sizeof(double)*phononCellMap.size())); //phonon omegaSq is always real
		omegaSqPh.init(nModes*nModes, phononCellMap.size());
		omegaSqPh.read_real(phFile.c_str());
		
		//Read phononCellMapSqPh
		ifs.open((prefix + ".mlwfCellMapSqPh").c_str());
		getline(ifs, headerLine); // read and ignore header line
		CellPair cp;
		while(ifs >> cp.iR1[0] >> cp.iR1[1] >> cp.iR1[2] >> cp.iR2[0] >> cp.iR2[1] >> cp.iR2[2])
			phononCellMapSq.push_back(cp);
		ifs.close();
		
		//Check the order of pairs in phononCellMapSqPh:
		auto pairIter = phononCellMapSq.begin();
		for(const vector3<int>& iR1: phononCellMap)
		for(const vector3<int>& iR2: phononCellMap)
		{	if(not (pairIter->iR1==iR1 and pairIter->iR2==iR2))
				die("Phonon cell map squared is not in required order (with outer index iR1 and inner index iR2)\n");
			pairIter++;
		}
		
		//Read phonon matrix elements
		wannierHePh.init(nModes*nBands*nBands, phononCellMapSq.size());
		readMatrix(wannierHePh, prefix + ".mlwfHePh", spinWeight);
		
		//Reshape matrix elements for two-phase transform:
		//--- make the second index iR1 alone, and lump iR2 into the first index
		//--- (possible because of the order enforced above)
		wannierHePh.reshape(nModes*nBands*nBands*phononCellMap.size(), phononCellMap.size());
	}
	
	//Pack matrix elements:
	nPacked = nMain*nMain + 2*nMain*(nBands-nMain);
	if(pWannier) compressMatElemArr(pWannier);
	if(pSqWannier) compressMatElemArr(pSqWannier);
	if(wannierHePh) compressMatElemArr(wannierHePh);
	
	//Pre-contract photon polarizations wherever applicable:
	int nPol = Ahat.size();
	//--- momentum matrix elements
	if(pWannier)
	{	matrix rot(3, Ahat.size());
		for(int iPol=0; iPol<nPol; iPol++)
			for(int iDir=0; iDir<3; iDir++)
				rot.set(iDir, iPol, Ahat[iPol][iDir]);
		transformMatElemArr(pWannier, rot);
	}
	//--- momentum-squared matrix elements
	if(pSqWannier)
	{	matrix rot(6, 1); //express contraction with symmetric tensor Ahat1 * Ahat2'
		for(int iDir=0; iDir<3; iDir++)
		{	//Diagonal
			rot.set(iDir,0, Ahat[0][iDir]*Ahat[1][iDir]);
			//Off-diagonal
			int jDir = (iDir + 1) % 3;
			int kDir = (iDir + 2) % 3;
			rot.set(iDir+3,0, Ahat[0][jDir]*Ahat[1][kDir] + Ahat[0][kDir]*Ahat[1][jDir]);
		}
		transformMatElemArr(pSqWannier, rot);
	}
}

diagMatrix BandStruct::getStates(vector3<> k, double omegaMax, matrix* evecs) const
{	return getStates(std::vector< vector3<> >(1, k), omegaMax, evecs)[0];
}

std::vector<diagMatrix> BandStruct::getStates(const std::vector< vector3<> >& kArr, double omegaMax, matrix* evecs) const
{	static StopWatch watch("BandStruct::getStates"); watch.start();
	std::vector< std::shared_ptr<const CacheEntry> > ceArr = getElectronCache(kArr, omegaMax);
	std::vector< diagMatrix > eigs(kArr.size());
	for(size_t ik=0; ik<kArr.size(); ik++)
	{	eigs[ik] = ceArr[ik]->eigs;
		if(evecs) evecs[ik] = ceArr[ik]->evecs;
	}
	watch.stop();
	return eigs;
}


diagMatrix BandStruct::getPhononModes(vector3<> q) const
{	static StopWatch watch("BandStruct::getPhononModes"); watch.start();
	std::shared_ptr<const CacheEntry> ce = getPhononCache(q);
	watch.stop();
	return ce->eigs;
}

std::vector<matrix> BandStruct::getDipoleMatElem(vector3<> k) const
{	static StopWatch watch("BandStruct::getDipoleMatElem"); watch.start();
	assert(nPol);
	std::shared_ptr<const CacheEntry> ce = getElectronCache(k);
	matrix Pk = pWannier * ce->phase;
	Pk.reshape(nPacked, nPol);
	std::vector<matrix> pk(nPol);
	for(int iPol=0; iPol<nPol; iPol++)
		pk[iPol] = dagger(ce->evecs) * unpackMatElem(Pk,iPol) * ce->evecs; //switch to eigenbasis of Hk
	watch.stop();
	return pk;
}

matrix BandStruct::getDipoleSqMatElem(vector3<> k) const
{	static StopWatch watch("BandStruct::getDipoleSqMatElem"); watch.start();
	assert(nPol == 2);
	std::shared_ptr<const CacheEntry> ce = getElectronCache(k);
	matrix Psq = unpackMatElem(pSqWannier * ce->phase, 0);
	watch.stop();
	return Psq;
}

std::vector<matrix> BandStruct::getPhononMatElem(vector3<> k1, vector3<> k2) const
{	std::vector<matrix> result;
	setPhononMatElemArray(k1, std::vector< vector3<> >(1, k2), &result);
	return result;
}

void BandStruct::setPhononMatElemArray(vector3<> k1, const std::vector< vector3<> >& k2arr, std::vector<matrix>* result) const
{	static StopWatch watch("BandStruct::getPhononMatElem"), watchFT("BandStruct::getPhononMatEl_FT"); watch.start();
	int nk2 = k2arr.size();
	//Compute double Fourier transform for fixed k1 and all k2 together:
	watchFT.start();
	std::vector< vector3<> > kMeanArr(nk2);
	matrix phase1(phononCellMap.size(), 1);
	matrix phase2(phononCellMap.size(), nk2);
	for(size_t iCell=0; iCell<phononCellMap.size(); iCell++)
		phase1.set(iCell,0, cis(2*M_PI * dot(phononCellMap[iCell],k1)));
	for(int ik2=0; ik2<nk2; ik2++)
	{	vector3<> k2 = k2arr[ik2];
		//Get bisecting k-point (within nearest image convention):
		vector3<> kDiff = k2 - k1;
		for(int j=0; j<3; j++) kDiff[j] -= floor(kDiff[j]+0.5);
		vector3<> kMean = k1 + 0.5*kDiff;
		kMeanArr[ik2] = kMean;
		//Calculate Fourier transform phase:
		for(size_t iCell=0; iCell<phononCellMap.size(); iCell++)
			phase2.set(iCell,ik2, cis(-2*M_PI * dot(phononCellMap[iCell],k2)));
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
			result[ik2][alpha] = dagger(cEl1.evecs) * unpackMatElem(HePh, alpha) * cEl2.evecs; //electron unitary rotations
	}
	watch.stop();
}

double BandStruct::get_mk(vector3<> k, double omega, double T) const
{	diagMatrix E = getStates(k, omega);
	double mk = INFINITY;
	for(int v=0; v<E.nRows(); v++) if(E[v]<10.*T)
		for(int c=0; c<E.nRows(); c++) if(E[c]>-10.*T)
			mk = std::min(mk, mk_sub(E[c], E[v], omega, T));
	return mk;
}

double BandStruct::get_mk1k2(vector3<> k1, vector3<> k2, double omega, double T) const
{	diagMatrix E1 = getStates(k1, omega);
	diagMatrix E2 = getStates(k2, omega);
	diagMatrix P = getPhononModes(k1-k2);
 	double mk1k2 = INFINITY;
	for(int v=0; v<E1.nRows(); v++) if (E1[v]<10.*T)
		for(int c=0; c<E2.nRows(); c++) if(E2[c]>-10*T)
			for(int alpha=0; alpha<nModes; alpha++)
				for(int ae=-1; ae<=+1; ae+=2)
					mk1k2 = std::min(mk1k2, mk_sub(E2[c], E1[v], omega + ae*P[alpha], T));
	return mk1k2;
}

std::vector< vector3<> > BandStruct::getVelocity(vector3<> k, const matrix3<>& R, double omegaMax) const
{	static StopWatch watch("BandStruct::getVelocity"); watch.start();
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

void BandStruct::compressMatElemArr(matrix& mArr) const
{	if(nMain == nBands) return; //no compression possible
	assert(mArr.nRows() % (nBands*nBands) == 0);
	int nMatsPerCol = mArr.nRows() / (nBands*nBands);
	matrix mPacked(nMatsPerCol*nPacked, mArr.nCols());
	matrix m(nBands, nBands);
	const complex* src = mArr.dataPref();
	for(int iMat=0; iMat<nMatsPerCol*mArr.nCols(); iMat++)
	{	callPref(eblas_copy)(m.dataPref(), src, nBands*nBands); src += nBands*nBands;
		packMatElem(m, mPacked, iMat);
	}
	std::swap(mPacked, mArr);
}

void BandStruct::transformMatElemArr(matrix& mArr, const matrix& rot) const
{	assert(mArr.nRows() % nPacked == 0); //must be in packed form
	int nMatsPerCol = mArr.nRows() / nPacked;
	assert(nMatsPerCol % rot.nRows() == 0);
	int nSetsPerCol = nMatsPerCol / rot.nRows();
	matrix m(nPacked, rot.nRows());
	matrix mArrOut(nSetsPerCol*rot.nCols()*nPacked, mArr.nCols());
	const complex* dataIn = mArr.dataPref();
	complex* dataOut = mArrOut.dataPref();
	for(int iSet=0; iSet<nSetsPerCol*mArr.nCols(); iSet++)
	{	callPref(eblas_copy)(m.dataPref(), dataIn, m.nData()); //read set into a matrix
		matrix mOut = m * rot; //apply transofrmation
		callPref(eblas_copy)(dataOut, mOut.dataPref(), mOut.nData()); //store set
		dataIn += m.nData();
		dataOut += mOut.nData();
	}
	std::swap(mArrOut, mArr);
}

void BandStruct::packMatElem(const matrix& m, matrix& mArr, int iCol) const
{	const complex* src = m.dataPref();
	complex* dest = mArr.dataPref() + iCol*nPacked;
	callPref(eblas_copy)(dest, src+mainFirst*nBands, nMain*nBands); dest += nMain*nBands;
	for(int b=0; b<nBands; b++)
		if(b<mainFirst || b>=(mainFirst+nMain))
		{	callPref(eblas_copy)(dest, src+mainFirst+b*nBands, nMain); dest += nMain;
		}
}

matrix BandStruct::unpackMatElem(const matrix& mArr, int iCol) const
{	matrix m = zeroes(nBands, nBands);
	const complex* src = mArr.dataPref() + iCol*nPacked;
	complex* dest = m.dataPref();
	callPref(eblas_copy)(dest+mainFirst*nBands, src, nMain*nBands); src += nMain*nBands;
	for(int b=0; b<nBands; b++)
		if(b<mainFirst || b>=(mainFirst+nMain))
		{	callPref(eblas_copy)(dest+mainFirst+b*nBands, src, nMain); src += nMain;
		}
	return m;
}

//------------ Cache functions -------------

std::shared_ptr<const BandStruct::CacheEntry> BandStruct::getElectronCache(vector3<> k, double omegaMax) const
{	return getElectronCache(std::vector< vector3<> >(1, k), omegaMax)[0];
}

std::shared_ptr<const BandStruct::CacheEntry> BandStruct::getPhononCache(vector3<> q) const
{	return getPhononCache(std::vector< vector3<> >(1, q))[0];
}

std::vector< std::shared_ptr<const BandStruct::CacheEntry> > BandStruct::getElectronCache(const std::vector< vector3<> >& kArr, double omegaMax) const
{	if(omegaMax<omegaMain)
		return getCache(kArr, mainCache, cellMap, hWannierMain, rankMain, false);
	else
		return getCache(kArr, electronCache, cellMap, hWannier, rankElectron, false);
}

std::vector< std::shared_ptr<const BandStruct::CacheEntry> > BandStruct::getPhononCache(const std::vector< vector3<> >& qArr) const
{	return getCache(qArr, phononCache, phononCellMap, omegaSqPh, rankPhonon, true);
}

std::vector< std::shared_ptr<const BandStruct::CacheEntry> > BandStruct::getCache( const std::vector< vector3<> >& kArr, 
		const std::map<vector3<>, std::shared_ptr<const CacheEntry> >& cache, const std::vector< vector3<int> >& cellMap,
		const matrix& hWannierEff, const size_t& rank, bool shouldSqrt) const
{	static StopWatch watch("BandStruct::getCache");
	std::vector< std::shared_ptr<const BandStruct::CacheEntry> > ceArr(kArr.size());
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
			phase.set(ic,ikNew, cis(-2*M_PI*dot(cellMap[ic],k)));
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

