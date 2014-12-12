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
	pxWannier.init(nBands*nBands, cellMap.size()); pxWannier.read((prefix + ".mlwfPx").c_str());
	pyWannier.init(nBands*nBands, cellMap.size()); pyWannier.read((prefix + ".mlwfPz").c_str());
	pzWannier.init(nBands*nBands, cellMap.size()); pzWannier.read((prefix + ".mlwfPz").c_str());
	
	kCache *= NAN; //indicate that cache is invalid
	
	if(usePhononStates)
	{	// Read phonon cell map
		ifstream readPhCellMap("totalE.phononCellMap");
		string phHeaderLine; getline(readPhCellMap, phHeaderLine); //read and ignore header line
		vector3<int> pcm;
		double px,py,pz;
		while(readPhCellMap >> pcm[0] >> pcm[1] >> pcm[2] >> px >> py >> pz)
			phCellMap.push_back(pcm);
		readPhCellMap.close();

		// Read cellMapSqPh
		ifstream readCellMapSq(prefix + ".mlwfCellMapSqPh");
		string headerLineSq ; getline(readCellMapSq, headerLineSq); // read and ignore header line
		struct CellPair { vector3<int> iR1, iR2; };
		std::vector<CellPair> cellMapSq;
		vector3<int> iR1, iR2;
		while(readCellMapSq >> iR1[0] >> iR1[1] >> iR1[2] >> iR2[0] >> iR2[1] >> iR2[2])
		{	CellPair cellPair = {iR1,iR2};
			cellMapSq.push_back(cellPair);
		}
	
		// Read wannier phonon hamiltonian
		string phFile = "totalE.phononOmegaSq";
		nModes = sqrt(fileSize(phFile.c_str())/(8*phCellMap.size())); //8 converts from bytes to number of complex numbers
		phWannier.init(nModes*nModes, phCellMap.size()); phWannier.read_real(phFile.c_str());

		qCache *=NAN; // indicate that cache is invalid
		
		// Read phonon matrix elements
		phWannierMatrix.init(nModes*nModes, phCellMap.size(), nBands*nBands); phWannierMatrix.read("wannier.mlwfHePh");

		// Read phonon cell map
		
	}
}

diagMatrix BandStruct::getStates(vector3<> k)
{   static StopWatch watch("BandStruct::getStates");
	if(k == kCache) return eigs;
	watch.start();
	//Calculate phase factors for each cell:
	phase.init(cellMap.size(), 1);
	for(size_t ic=0; ic<cellMap.size(); ic++)
		phase.set(ic,0, cis(2*M_PI*dot(cellMap[ic],k)));
	//Compute Hamiltonian for k:
	matrix Hk = hWannier * phase;
	Hk.reshape(nBands, nBands);
	Hk = dagger_symmetrize(Hk);
	//Diagonalize:
	Hk.diagonalize(evecs, eigs);
	kCache = k;
	watch.stop();
	return eigs;
}

diagMatrix BandStruct::getPhononModes(vector3<> q)
{	if(q == qCache) return phEigs;
	//Calculate phase factors for each cell:
	phPhase.init(phCellMap.size(),1);
	for(size_t ic=0; ic<phCellMap.size(); ic++)
		phPhase.set(ic,0,cis(2*M_PI*dot(phCellMap[ic], q)));
	//Compute Hamiltonian for q:
	matrix Hq = phWannier *phPhase;
	Hq.reshape(nModes,nModes);
	Hq = dagger_symmetrize(Hq);
	//Diagonalize:
	Hq.diagonalize(phEvecs, phEigs);
	qCache = q;
	for(double& x: phEigs) x = sqrt(x);
	return phEigs;
}

std::vector<matrix> BandStruct::getDipoleMatElem(vector3<> k)
{	static StopWatch watch("BandStruct::getDipoleMatElem"); watch.start();
	if(!(k == kCache)) getStates(k); //Update evecs and phase if necessary
	// Compute transitions at k
	matrix Pkx = pxWannier * phase; Pkx.reshape(nBands, nBands);
	matrix Pky = pyWannier * phase; Pky.reshape(nBands, nBands);
	matrix Pkz = pzWannier * phase; Pkz.reshape(nBands, nBands);
	// Change basis of Px, Py, Pz to eigenbasis of Hk
	std::vector<matrix> pk(3);
	pk[0] = transpose(evecs) * Pkx * evecs;
	pk[1] = transpose(evecs) * Pky * evecs;
	pk[2] = transpose(evecs) * Pkz * evecs;
	watch.stop();
	return pk;
}

std::vector<matrix> BandStruct::getPhononMatElem(vector3<> k1, vector3<> k2)
{	if(!(k1 == kCache)) getStates(k1); // Update evecs and phase if necessary
	matrix Pk1 = phWannierMatrix * phase;
}

double BandStruct::get_mk(vector3<> k, double omega, double T)
{	diagMatrix E = getStates(k);
	double mk = INFINITY;
	for(int v=0; v<nBands; v++) if(E[v]<10.*T)
		for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
			mk = std::min(mk, mk_sub(E[c], E[v], omega, T));
	return mk;
}

double BandStruct::get_mk1k2(vector3<> k1, vector3<> k2, double omega, double T)
{	vector3<> q = k1 - k2;
	diagMatrix E1 = getStates(k1), E2 = getStates(k2), P = getPhononModes(q);
 	double mk1k2 = INFINITY;
	for(int v=0; v<nModes; v++) if (E1[v]<10.*T)
		for(int c=0; c<nBands; c++) if(E2[c]>-10*T)
			for(int pn=0; pn<nModes; pn++)
				for(int ae = -1; ae<=1; ae+=2)
					mk1k2 = std::min(mk1k2, mk_sub(E2[c],E1[v],omega+ae*P[pn], T));
	return mk1k2;
}

std::vector< vector3<> > BandStruct::getVelocity(vector3<> k, const matrix3<>& R)
{	static StopWatch watch("BandStruct::getVelocity"); watch.start();
	if(!(k == kCache)) getStates(k); //Update evecs and phase if necessary
	std::vector< vector3<> > v(nBands);
	matrix Vk = evecs;
	complex I(0.0,1.0);
	for(int j = 0; j < 3; j++)
	{	matrix phasePrime;
		phasePrime.init(cellMap.size(), 1);
		for(size_t ic=0; ic<cellMap.size(); ic++)
		{	auto R = cellMap[ic];
			phasePrime.set(ic,0, I*R[j]*cis(2*M_PI*dot(R,k)));
		}
		matrix dHdk = hWannier * phasePrime;
		dHdk.reshape(nBands, nBands);
		diagMatrix vj = diag(dagger(Vk) * dHdk * Vk);
		for( int b = 0; b<nBands; b++) v[b][j] = vj[b];
	}
	for(vector3<>& vb: v) vb = R * vb; //Convert to Cartesian
	watch.stop();
	return v;
}
