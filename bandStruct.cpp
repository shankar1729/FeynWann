#include "bandStruct.h"
#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

//---------------------------- class bandStruct---------------------------------

//Constructor 
bandStruct::bandStruct(string prefix, double mu)
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
}

diagMatrix bandStruct::getStates(vector3<double> kPoint)
{   static StopWatch watch("bandStruct::getStates");
	if(kPoint == kPointCache) return eigs;
	watch.start();
	//Calculate phase factors for each cell:
	phase.init(cellMap.size(), 1);
	for(size_t ic=0; ic<cellMap.size(); ic++)
		phase.set(ic,0, cis(2*M_PI * dot(cellMap[ic],kPoint)));
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

std::vector<matrix> bandStruct::getTransitions(vector3<double> kPoint)
{	static StopWatch watch("bandStruct::getTransitions"); watch.start();
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

double bandStruct::get_mk(vector3<double> kPoint, double omega, double T)
{	diagMatrix E = getStates(kPoint);
	double mk = INFINITY;
	for(int v=0; v<nBands; v++) if(E[v]<10.*T)
	{	for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
		{	mk = std::min(mk, mk_sub(E[c], E[v], omega, T));
		}
	}
	return mk;
}

double bandStruct::get_mk1k2(vector3<double> kPoint1, vector3<double> kPoint1, double omega, double T)
{       diagMatrix E1 = getStates(kPoint1), E2 = getStates(kPoint2);
        double mk1k2 = INFINITY;
        for(int v=0; v<nBands; v++) if(E[v]<10.*T)
        {       for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
                {       mk1k2 = std::min(mk1k2, mk_sub(E2[c], E1[v], omega, T));
                }
        }
        return mk1k2;
}
