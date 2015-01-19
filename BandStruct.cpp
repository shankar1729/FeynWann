#include "BandStruct.h"
#include <core/Util.h>
#include <core/Units.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

BandStruct::BandStruct(string prefix, double mu, int spinWeight, string phononPrefix)
{	//Read cell map
	ifstream ifs(prefix + ".mlwfCellMap");
	string headerLine; getline(ifs, headerLine); //read and ignore header line
	vector3<int> cm;
	double x,y,z;
	while(ifs >> cm[0] >> cm[1] >> cm[2] >> x >> y >> z)
		cellMap.push_back(cm);
	ifs.close();

	//Read wannier hamiltonian
	string hFile = prefix + ".mlwfH";
	nBands = sqrt(fileSize(hFile.c_str()) / ((spinWeight==1 ? sizeof(complex) : sizeof(double)) * cellMap.size()));
	hWannier.init(nBands*nBands, cellMap.size());
	if(spinWeight==1) hWannier.read(hFile.c_str()); else hWannier.read_real(hFile.c_str());

	//Offset wannier Hamiltonian by mu:
	for(size_t ic=0; ic<cellMap.size(); ic++)
		if(!cellMap[ic].length_squared()) //diagonal element
		{	matrix id = eye(nBands); id.reshape(nBands*nBands, 1);
			hWannier.set(0,nBands*nBands, ic,ic+1, hWannier(0,nBands*nBands, ic,ic+1) - mu * id);
		}
	
	// Read momentum matrix elements
	string dirNames[3] = { "x", "y", "z" };
	for(int j=0; j<3; j++)
	{	pWannier[j].init(nBands*nBands, cellMap.size());
		string pFile = prefix + ".mlwfP" + dirNames[j];
		if(spinWeight==1) pWannier[j].read(pFile.c_str()); else pWannier[j].read_real(pFile.c_str());
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
		
		//Read phonon matrix elements
		wannierHePh.init(nModes*nBands*nBands, phononCellMapSq.size());
		matrix wannierHePh_mode(nBands*nBands, phononCellMapSq.size()); //matrix elements for a single mode (stored contiguously)
		FILE* fp = fopen((prefix + ".mlwfHePh").c_str(), "r");
		for(int alpha=0; alpha<nModes; alpha++)
		{	if(spinWeight==1) wannierHePh_mode.read(fp); else wannierHePh_mode.read_real(fp);
			wannierHePh.set(alpha*nBands*nBands,(alpha+1)*nBands*nBands, 0,phononCellMapSq.size(), wannierHePh_mode);
		}
		fclose(fp);
	}
}

diagMatrix BandStruct::getStates(vector3<> k, double omegaMax) const
{   static StopWatch watch("BandStruct::getStates"); watch.start();
	std::shared_ptr<const CacheEntry> ce = getElectronCache(k, omegaMax);
	watch.stop();
	return ce->eigs;
}

diagMatrix BandStruct::getPhononModes(vector3<> q) const
{	static StopWatch watch("BandStruct::getPhononModes"); watch.start();
	std::shared_ptr<const CacheEntry> ce = getPhononCache(q);
	watch.stop();
	return ce->eigs;
}

std::vector<matrix> BandStruct::getDipoleMatElem(vector3<> k) const
{	static StopWatch watch("BandStruct::getDipoleMatElem"); watch.start();
	std::shared_ptr<const CacheEntry> ce = getElectronCache(k);
	std::vector<matrix> pk(3);
	for(int j=0; j<3; j++)
	{	matrix Pk = pWannier[j] * ce->phase;
		Pk.reshape(nBands, nBands);
		pk[j] = dagger(ce->evecs) * Pk * ce->evecs; //switch to eigenbasis of Hk
	}
	watch.stop();
	return pk;
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
	matrix phase(phononCellMapSq.size(), 2*nk2);
	for(int ik2=0; ik2<nk2; ik2++)
	{	vector3<> k2 = k2arr[ik2];
		//Get bisecting k-point (within nearest image convention):
		vector3<> kDiff = k2 - k1;
		for(int j=0; j<3; j++) kDiff[j] -= floor(kDiff[j]+0.5);
		vector3<> kMean = k1 + 0.5*kDiff;
		kMeanArr[ik2] = kMean;
		//Calculate Fourier transform phase:
		for(size_t icp=0; icp<phononCellMapSq.size(); icp++)
		{	const CellPair& cp = phononCellMapSq[icp];
			phase.set(icp,ik2, cis(2*M_PI * (dot(cp.iR1,k1) - dot(cp.iR2,k2))));
			phase.set(icp,ik2+nk2, cis(2*M_PI * dot(cp.iR1 - cp.iR2, kMean)));
		}
	}
	matrix HePhPair = wannierHePh * phase; //now a nBands*nBands*nModes x nk2*2 matrix (columns of all k2 results first, followed by all kMean's)
	watchFT.stop();
	//Now process one k2 at a time:
	std::shared_ptr<const CacheEntry> cEl1 = getElectronCache(k1); //only cache common to all of below
	for(int ik2=0; ik2<nk2; ik2++)
	{	vector3<> k2 = k2arr[ik2];
		vector3<> kMean = kMeanArr[ik2];
		matrix HePh = HePhPair(0,HePhPair.nRows(), ik2,ik2+1); HePh.reshape(nBands*nBands, nModes); //now each column is matrix elements for a specific nuclear displacement
		//Translational invariance correction by subtracting reference at kMean: [WARNING: Assumes no optical phonon modes!]
		matrix HePhRef = HePhPair(0,HePhPair.nRows(), ik2+nk2,ik2+nk2+1); HePhRef.reshape(nBands*nBands, nModes); //corresponding to HePh at kMean
		std::shared_ptr<const CacheEntry> cElMean = getElectronCache(kMean);
		for(int alpha=0; alpha<nModes; alpha++)
		{	matrix Href = HePhRef(0,HePhRef.nRows(), alpha,alpha+1); //select the current mode at kMean
			Href.reshape(nBands, nBands);
			Href = dagger(cElMean->evecs) * Href * cElMean->evecs; //switch to eigenbasis
			for(int b1=0; b1<nBands; b1++)
				for(int b2=0; b2<nBands; b2++)
					if(fabs(cElMean->eigs[b1] - cElMean->eigs[b2]) > 1e-4)
						Href.set(b1,b2, 0.); //not in degenerate subspace; don't correct
			Href = cElMean->evecs * Href * dagger(cElMean->evecs); //switch back to Wannier basis
			Href.reshape(nBands*nBands, 1);
			HePhRef.set(0,HePh.nRows(), alpha,alpha+1, Href); //set the reference value
		}
		HePh -= HePhRef;
		//Apply unitary transformations:
		HePh = HePh * getPhononCache(k1-k2)->evecs; //phonon unitary rotation
		std::shared_ptr<const CacheEntry> cEl2 = getElectronCache(k2);
		result[ik2].resize(nModes);
		for(int alpha=0; alpha<nModes; alpha++)
		{	result[ik2][alpha] = HePh(0,HePh.nRows(), alpha,alpha+1); //select the current mode
			result[ik2][alpha].reshape(nBands, nBands);
			result[ik2][alpha] = dagger(cEl1->evecs) * result[ik2][alpha] * cEl2->evecs; //electron unitary rotations
		}
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
	const matrix& hWannierEff = ce->nBands==nBands ? hWannier : hWannierMain;
	std::vector< vector3<> > v(ce->nBands);
	for(int j = 0; j < 3; j++)
	{	matrix phasePrime = ce->phase;
		complex* phasePrimeData = phasePrime.data();
		for(size_t ic=0; ic<cellMap.size(); ic++)
			phasePrimeData[ic] *= complex(0,cellMap[ic][j]); //multiply phase by I*iR[j] to get phasePrime
		matrix dHdk = hWannierEff * phasePrime;
		dHdk.reshape(ce->nBands, ce->nBands);
		diagMatrix vj = diag(dagger(ce->evecs) * dHdk * ce->evecs);
		for(int b=0; b<ce->nBands; b++) v[b][j] = vj[b];
	}
	for(vector3<>& vb: v) vb = R * vb; //Convert to Cartesian
	watch.stop();
	return v;
}

std::shared_ptr<const BandStruct::CacheEntry> BandStruct::getElectronCache(vector3<> k, double omegaMax) const
{	int nBandsEff = omegaMax<omegaMain ? nMain : nBands;
	//Check cache first:
	for(const auto& entry: electronCache)
		if(entry->k == k && entry->nBands == nBandsEff)
			return entry;
	//Not found in cache; generate:
	auto ce = std::make_shared<CacheEntry>();
	ce->nBands = nBandsEff;
	ce->k = k;
	//--- calculate phase factors for each cell:
	ce->phase.init(cellMap.size(), 1);
	for(size_t ic=0; ic<cellMap.size(); ic++)
		ce->phase.set(ic,0, cis(2*M_PI*dot(cellMap[ic],k)));
	//--- compute and diagonalize Hamiltonian for k:
	const matrix& hWannierEff = ce->nBands==nBands ? hWannier : hWannierMain;
	matrix Hk = hWannierEff * ce->phase;
	Hk.reshape(ce->nBands, ce->nBands);
	Hk = dagger_symmetrize(Hk);
	Hk.diagonalize(ce->evecs, ce->eigs);
	//--- update cache:
	BandStruct& bs = *((BandStruct*)this); //modifiable version of this
	if(electronCache.size() == 6) bs.electronCache.pop_front(); //discard oldest cache entry
	bs.electronCache.push_back(ce); //add new one
	return ce;
}

std::shared_ptr<const BandStruct::CacheEntry> BandStruct::getPhononCache(vector3<> q) const
{	//Check cache first:
	for(const auto& entry: phononCache)
		if(entry->k == q)
			return entry;
	//Not found in cache; generate:
	auto ce = std::make_shared<CacheEntry>();
	ce->k = q;
	//--- calculate phase factors for each cell:
	ce->phase.init(phononCellMap.size(),1);
	for(size_t ic=0; ic<phononCellMap.size(); ic++)
		ce->phase.set(ic,0,cis(2*M_PI*dot(phononCellMap[ic], q)));
	//--- compute and diagonalize force matrix for q:
	matrix omegaSq_q = omegaSqPh * ce->phase;
	omegaSq_q.reshape(nModes,nModes);
	omegaSq_q = dagger_symmetrize(omegaSq_q);
	omegaSq_q.diagonalize(ce->evecs, ce->eigs);
	for(double& x: ce->eigs) x = sqrt(fabs(x)); //switch from omegaSq to omega
	//--- update cache:
	BandStruct& bs = *((BandStruct*)this); //modifiable version of this
	if(phononCache.size() == 1) bs.phononCache.pop_front(); //discard oldest cache entries
	bs.phononCache.push_back(ce); //add new one
	return ce;
}
