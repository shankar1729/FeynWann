#include "BandStruct.h"
#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

BandStruct::BandStruct(string prefix, double mu, bool usePhononStates)
{	// Read cell map
	ifstream readCellMap(prefix + ".mlwfCellMap");
	string headerLine; getline(readCellMap, headerLine); //read and ignore header line
	vector3<int> cm;
	double x,y,z;
	while(readCellMap >> cm[0] >> cm[1] >> cm[2] >> x >> y >> z)
		cellMap.push_back(cm);
	readCellMap.close();

	// Read wannier hamiltonian
	string hFile = prefix + ".mlwfH";
	nBands = sqrt(fileSize(hFile.c_str())/(16*cellMap.size())); //16 converts from bytes to number of complex numbers
	hWannier.init(nBands*nBands, cellMap.size()); hWannier.read(hFile.c_str());

	// Offset wannier Hamiltonian by mu:
	for(size_t ic=0; ic<cellMap.size(); ic++)
		if(!cellMap[ic].length_squared()) //diagonal element
		{	matrix id = eye(nBands); id.reshape(nBands*nBands, 1);
			hWannier.set(0,nBands*nBands, ic,ic+1, hWannier(0,nBands*nBands, ic,ic+1) - mu * id);
		}
	
	// Read momentum matrix elements
	string dirNames[3] = { "x", "y", "z" };
	for(int j=0; j<3; j++)
	{	pWannier[j].init(nBands*nBands, cellMap.size());
		pWannier[j].read((prefix + ".mlwfP" + dirNames[j]).c_str());
	}
	
	if(usePhononStates)
	{	// Read phonon cell map
		ifstream readPhCellMap("totalE.phononCellMap");
		string phHeaderLine; getline(readPhCellMap, phHeaderLine); //read and ignore header line
		while(readPhCellMap >> cm[0] >> cm[1] >> cm[2] >> x >> y >> z)
			phononCellMap.push_back(cm);
		readPhCellMap.close();

		// Read phononCellMapSqPh
		ifstream readCellMapSq(prefix + ".mlwfCellMapSqPh");
		string headerLineSq ; getline(readCellMapSq, headerLineSq); // read and ignore header line
		CellPair cp;
		while(readCellMapSq >> cp.iR1[0] >> cp.iR1[1] >> cp.iR1[2] >> cp.iR2[0] >> cp.iR2[1] >> cp.iR2[2])
			phononCellMapSq.push_back(cp);
	
		// Read wannier phonon hamiltonian
		string phFile = "totalE.phononOmegaSq";
		nModes = sqrt(fileSize(phFile.c_str())/(8*phononCellMap.size())); //8 converts from bytes to number of complex numbers
		omegaSqPh.init(nModes*nModes, phononCellMap.size()); omegaSqPh.read_real(phFile.c_str());

		
		// Read phonon matrix elements
		wannierHePh.init(nModes*nModes, phononCellMap.size(), nBands*nBands);
		wannierHePh.read("wannier.mlwfHePh");
	}
}

diagMatrix BandStruct::getStates(vector3<> k) const
{   static StopWatch watch("BandStruct::getStates"); watch.start();
	const CacheEntry& ce = getElectronCache(k);
	watch.stop();
	return ce.eigs;
}

diagMatrix BandStruct::getPhononModes(vector3<> q) const
{	static StopWatch watch("BandStruct::getPhononModes"); watch.start();
	const CacheEntry& ce = getPhononCache(q);
	watch.stop();
	return ce.eigs;
}

std::vector<matrix> BandStruct::getDipoleMatElem(vector3<> k) const
{	static StopWatch watch("BandStruct::getDipoleMatElem"); watch.start();
	const CacheEntry& ce = getElectronCache(k);
	std::vector<matrix> pk(3);
	for(int j=0; j<3; j++)
	{	matrix Pk = pWannier[j] * ce.phase;
		Pk.reshape(nBands, nBands);
		pk[j] = transpose(ce.evecs) * Pk * ce.evecs; //switch to eigenbasis of Hk
	}
	watch.stop();
	return pk;
}

std::vector<matrix> BandStruct::getPhononMatElem(vector3<> k1, vector3<> k2) const
{	//TODO
	die("Not yet implemented.\n");
	return std::vector<matrix>();
}

double BandStruct::get_mk(vector3<> k, double omega, double T) const
{	diagMatrix E = getStates(k);
	double mk = INFINITY;
	for(int v=0; v<nBands; v++) if(E[v]<10.*T)
		for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
			mk = std::min(mk, mk_sub(E[c], E[v], omega, T));
	return mk;
}

double BandStruct::get_mk1k2(vector3<> k1, vector3<> k2, double omega, double T) const
{	vector3<> q = k1 - k2;
	diagMatrix E1 = getStates(k1), E2 = getStates(k2), P = getPhononModes(q);
 	double mk1k2 = INFINITY;
	for(int v=0; v<nModes; v++) if (E1[v]<10.*T)
		for(int c=0; c<nBands; c++) if(E2[c]>-10*T)
			for(int alpha=0; alpha<nModes; alpha++)
				for(int ae=-1; ae<=+1; ae+=2)
					mk1k2 = std::min(mk1k2, mk_sub(E2[c], E1[v], omega + ae*P[alpha], T));
	return mk1k2;
}

std::vector< vector3<> > BandStruct::getVelocity(vector3<> k, const matrix3<>& R) const
{	static StopWatch watch("BandStruct::getVelocity"); watch.start();
	const CacheEntry& ce = getElectronCache(k);
	std::vector< vector3<> > v(nBands);
	for(int j = 0; j < 3; j++)
	{	matrix phasePrime = ce.phase;
		complex* phasePrimeData = phasePrime.data();
		for(size_t ic=0; ic<cellMap.size(); ic++)
			phasePrimeData[ic] *= complex(0,cellMap[ic][j]); //multiply phase by I*iR[j] to get phasePrime
		matrix dHdk = hWannier * phasePrime;
		dHdk.reshape(nBands, nBands);
		diagMatrix vj = diag(dagger(ce.evecs) * dHdk * ce.evecs);
		for(int b = 0; b<nBands; b++) v[b][j] = vj[b];
	}
	for(vector3<>& vb: v) vb = R * vb; //Convert to Cartesian
	watch.stop();
	return v;
}

const BandStruct::CacheEntry& BandStruct::getElectronCache(vector3<> k) const
{	//Check cache first:
	for(const auto& entry: electronCache)
		if(entry->k == k)
			return *entry;
	//Not found in cache; generate:
	auto ce = std::make_shared<CacheEntry>();
	ce->k = k;
	//--- calculate phase factors for each cell:
	ce->phase.init(cellMap.size(), 1);
	for(size_t ic=0; ic<cellMap.size(); ic++)
		ce->phase.set(ic,0, cis(2*M_PI*dot(cellMap[ic],k)));
	//--- compute and diagonalize Hamiltonian for k:
	matrix Hk = hWannier * ce->phase;
	Hk.reshape(nBands, nBands);
	Hk = dagger_symmetrize(Hk);
	Hk.diagonalize(ce->evecs, ce->eigs);
	//--- update cache:
	BandStruct& bs = *((BandStruct*)this); //modifiable version of this
	if(electronCache.size() == 2) bs.electronCache.pop_front(); //discard oldest cache entry
	bs.electronCache.push_back(ce); //add new one
	return *ce;
}

const BandStruct::CacheEntry& BandStruct::getPhononCache(vector3<> q) const
{	//Check cache first:
	for(const auto& entry: phononCache)
		if(entry->k == q)
			return *entry;
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
	for(double& x: ce->eigs) x = sqrt(x); //switch from omegaSq to omega
	//--- update cache:
	BandStruct& bs = *((BandStruct*)this); //modifiable version of this
	if(phononCache.size() == 1) bs.phononCache.pop_front(); //discard oldest cache entries
	bs.phononCache.push_back(ce); //add new one
	return *ce;
}
