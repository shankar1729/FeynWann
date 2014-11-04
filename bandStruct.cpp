#include "bandStruct.h"
#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

//---------------------------- class bandStruct---------------------------------

//Constructor 
bandStruct::bandStruct(string prefix)
{	// Read cell map
	string filePrefix = prefix;
	ifstream readCellMap(filePrefix + ".mlwfCellMap");
	string headerLine; getline(readCellMap, headerLine); //read and ignore header line
	vector3<int> cm;
	double x,y,z;
	while(readCellMap >> cm[0] >> cm[1] >> cm[2] >> x >> y >> z)
		cellMap.push_back(cm);
	readCellMap.close();

	// Read wannier hamiltonian
	string hFile = filePrefix + ".mlwfH";
	nBands = sqrt(fileSize(hFile.c_str())/(16*cellMap.size())); //16 converts from bytes to number of complex numbers
	hWannier.init(nBands*nBands, cellMap.size()); hWannier.read(hFile.c_str());

	// Read momentum matrix elements
	pxWannier.init(nBands*nBands, cellMap.size()); pxWannier.read((filePrefix + ".mlwfPx").c_str());
	pyWannier.init(nBands*nBands, cellMap.size()); pyWannier.read((filePrefix + ".mlwfPz").c_str());
	pzWannier.init(nBands*nBands, cellMap.size()); pzWannier.read((filePrefix + ".mlwfPz").c_str());        
}

diagMatrix bandStruct::getStates(vector3<double> kPoint)
{   static StopWatch watch("bandStruct::getStates"); watch.start();
	//Calculate phase factors for each cell:
	phase.init(cellMap.size(), 1);
	for(size_t ic=0; ic<cellMap.size(); ic++)
		phase.set(ic,0, cis(2*M_PI * dot(cellMap[ic],kPoint)));
	//Compute Hamiltonian for kPoint:
	matrix Hk = hWannier * phase;
	Hk.reshape(nBands, nBands);
	Hk = dagger_symmetrize(Hk);
	//Diagonalize:
	diagMatrix eigs; //Note evecs is remembered for use in kPoint
	Hk.diagonalize(evecs, eigs);
	kPoint_evecs = kPoint;
	watch.stop();
	return eigs;
}

std::vector<matrix> bandStruct::getTransitions(vector3<double> kPoint)
{	static StopWatch watch("bandStruct::getTransitions"); watch.start();
	if(!(kPoint == kPoint_evecs)) getStates(kPoint); //Update evecs and phase if necessary
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
