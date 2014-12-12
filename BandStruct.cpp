#include "BandStruct.h"
#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

BandStruct::BandStruct(string prefix, double mu, bool usePhononStates=0)
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
	
	kPointCache *= NAN; //indicate that cache is invalid
	
	if usePhononStates
	{	// Read phonon cell map
		ifstream readPhCellMap("totalE.phononCellMap");
		string phHeaderLine; getline(readPhCellMap, phHeaderLine); //read and ignore header line
		vector3<int> pcm;
		double px,py,pz;
		while(readPhononCellMap >> pcm[0] >> pcm[1] >> pcm[2] >> px >> py >> pz)
			phCellMap.push_back(pcm);
		readPhCellMap.close();

		// Read cellMapSqPh
		ifstream readCellMapSq(prefix + ".mlwfCellMapSqPh");
		string headerLineSq ; getline(readCellMapSq, headerLineSq); // read and ignore header line
		struct CellPair { vector3<> iR1, iR2 }
		std::vector<CellPair> cellMapSq;
		vector3<int> iR1, iR2;
		while(readCellMapSq >> iR1[0] >> iR1[1] >> iR1[2] >> iR2[0] >> iR2[1] >> iR2[2])
		{	CellPair cellPair = {iR1,iR2};
			cellMapSq.push_back(cellPair);
		}
	
		// Read wannier phonon hamiltonian
		string phFile = "totalE.phononOmegaSq";
		pnBands = sqrt(fileSize(phFile.c_str())/(8*phCellMap.size())); //8 converts from bytes to number of complex numbers
		phWannier.init(pnBands*pnBands, phCellMap.size()); phWannier.read_real(phFile.c_str());

		qPointCache *=NAN; // indicate that cache is invalid
		
		// Read phonon matrix elements
		phWannierMatrix.init(pbBands*pnBands, phCellMap.size(), nBands*nBands); phWannierMatrix.read(("wannier.mlwfHePh").c_str());

		// Read phonon cell map
		
	}
}

diagMatrix BandStruct::getStates(vector3<> kPoint)
{   static StopWatch watch("BandStruct::getStates");
	if(kPoint == kPointCache) return eigs;
	watch.start();
	//Calculate phase factors for each cell:
	phase.init(cellMap.size(), 1);
	for(size_t ic=0; ic<cellMap.size(); ic++)
		phase.set(ic,0, cis(2*M_PI*dot(cellMap[ic],kPoint)));
	//Compute Hamiltonian for kPoint:
	matrix Hk = hWannier * phase;
	Hk.reshape(nBands, nBands);
	Hk = dagger_symmetrize(Hk);
	//Diagonalize:
	Hk.diagonalize(evecs, eigs);
	kPointCache = kPoint;
	watch.stop();
	return eigs;
}

diagMatrix BandStruct::getPhStates(vector3<> qPoint)
{	if(qPoint == qPointCache) return phEigs;
	//Calculate phase factors for each cell:
	phPhase.init(phCellMap.size(),1);
	for(size_t ic=0; ic<phCellMap.size(); ic++)
		phPhase.set(ic,0,cis(2*M_PI*dot(phCellMap[ic],Kpoint)));
	//Compute Hamiltonian for qPoint:
	matrix Hq = phWannier *phPhase;
	Hq.reshape(npBands,npBands);
	Hq = dagger_symmetrize(Hq);
	//Diagonalize:
	Hq.diagonalize(phEvecs, phEigs);
	qPointCache = qPoint;
	for(double& x: phEigs) x = sqrt(x);
	return phEigs;
}

std::vector<matrix> BandStruct::getTransitions(vector3<> kPoint)
{	static StopWatch watch("BandStruct::getTransitions"); watch.start();
	if(!(kPoint == kPointCache)) getStates(kPoint); //Update evecs and phase if necessary
	// Compute transitions at kPoint
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

std::vector<matrix> BandStruct::getPhTransitions(vector3<> kPoint1, kPoint2)
{	if(!(kPoint1 == kPointCache)) getStates(kPoint1); // Update evecs and phase if necessary
	matrix Pk1 = phWannierMatrix * phase;	
}

double BandStruct::get_mk(vector3<> kPoint, double omega, double T)
{	diagMatrix E = getStates(kPoint);
	double mk = INFINITY;
	for(int v=0; v<nBands; v++) if(E[v]<10.*T)
	{	for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
		{	mk = std::min(mk, mk_sub(E[c], E[v], omega, T));
		}
	}
	return mk;
}

double BandStruct::get_mk1k2(vector3<> kPoint1, vector3<> kPoint2, double omega, double T)
{	diagMatrix E1 = getStates(kPoint1), E2 = getStates(kPoint2);
	double mk1k2 = INFINITY;
	for(int v=0; v<nBands; v++) if(E1[v]<10.*T)
	{	for(int c=0; c<nBands; c++) if(E2[c]>-10.*T)
		{	mk1k2 = std::min(mk1k2, mk_sub(E2[c], E1[v], omega, T));
		}
	}
	return mk1k2;
}

double BandStruct::get_mk1k2ph(vector3<> kPoint1, vector3<> kPoint2, double omega, doubleT)
{	vector3<> qPoint = kPoint1 - kPoint2;
	diagMatrix E1 = getStates(kPoint1), E2 = getStates(kPoint2), P = getPhStates(qPoint);
 	double mk1k2 = INFINITY;
	for(int v=0; v<pnBands; v++) if (E1[v]<10.*T)
	{	for(int c=0; c<nBands; c++) if(E2[c]>-10*T)
		{	for(int pn=0; pn<pnBands; pn++)
			{	for(int ae = -1; ae<=1; ae+=2)
				{	mk1k2 = std::min(mk1k2, mk_sub(E2[c],E1[v],omega+ae*P[pn], T)
				}
			}	
		}
	}
	return mk1k2;
}

std::vector< vector3<> > BandStruct::getVelocity(vector3<> kPoint, const matrix3<>& R)
{	static StopWatch watch("BandStruct::getVelocity"); watch.start();
	if(!(kPoint == kPointCache)) getStates(kPoint); //Update evecs and phase if necessary
	std::vector< vector3<> > v(nBands);
	matrix Vk = evecs;
	complex I(0.0,1.0);
	for(int j = 0; j < 3; j++)
	{	matrix phasePrime;
		phasePrime.init(cellMap.size(), 1);
		for(size_t ic=0; ic<cellMap.size(); ic++)
		{	auto R = cellMap[ic];
			phasePrime.set(ic,0, I*R[j]*cis(2*M_PI*dot(R,kPoint)));
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
