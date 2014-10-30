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
{       // Read cell map
        string filePrefix = prefix;
        ifstream readCellMap(filePrefix + ".mlwfCellMap");
        string headerLine; getline(readCellMap, headerLine); //read and ignore header line
        vector3<int> cm;
        double x,y,z;
        while(readCellMap >> cm[0] >> cm[1] >> cm[2] >> x >> y >> z)
                cellMap.push_back(cm);
        readCellMap.close();
        const int cms = cellMap.size();

        // Read wannier hamiltonian
        string hFile = filePrefix + ".mlwfH";
        const char * cps = hFile.c_str();
        int hWannierSize = fileSize(cps)/16; //convert from bytes to number of complex numbers
        matrix hWannier(hWannierSize,1);
        hWannier.read(cps);
        int numBands = sqrt(hWannierSize/cms);
        hWannier.reshape(numBands,numBands*cms);
        hWannierArray.resize(cms);
        for (int ii = 0; ii<cms; ii++)
                hWannierArray[ii] = hWannier(0,numBands,0+ii*numBands,numBands+ii*numBands);


        // Read momentum matrix elements
        string pxFile = filePrefix + ".mlwfPx";
        string pyFile = filePrefix + ".mlwfPy";
        string pzFile = filePrefix + ".mlwfPz";
        const char * cpx = pxFile.c_str();
	const char * cpy = pyFile.c_str();
	const char * cpz =  pzFile.c_str();
        matrix pxWannier(hWannierSize,1);
	matrix pyWannier(hWannierSize,1);
	matrix pzWannier(hWannierSize,1);
        pxWannier.read(cpx);
	pyWannier.read(cpy);
	pzWannier.read(cpz);
        pxWannier.reshape(numBands,numBands*cms);
	pyWannier.reshape(numBands,numBands*cms);
	pzWannier.reshape(numBands,numBands*cms);
        pxWannierArray.resize(cms);
	pyWannierArray.resize(cms);
	pzWannierArray.resize(cms);
        for (int ii = 0; ii<cms; ii++){
                pxWannierArray[ii] = pxWannier(0,numBands,0+ii*numBands, numBands+ii*numBands);
        	pyWannierArray[ii] = pyWannier(0,numBands,0+ii*numBands, numBands+ii*numBands);
        	pzWannierArray[ii] = pzWannier(0,numBands,0+ii*numBands, numBands+ii*numBands);
	}
}

diagMatrix bandStruct::getStates(vector3<double> kPoint)
{       // Compute & record interpolated band strucutre;
        //matrix evecs;
        diagMatrix eigs;
        matrix Hk,Hkh;
        const int cmsize = cellMap.size();
        for (int ic = 0; ic<cmsize; ic++)
                Hk +=  hWannierArray[ic] * cis(2*M_PI * dot(cellMap[ic],kPoint));
        Hkh = dagger_symmetrize(Hk);
        Hkh.diagonalize(evecs, eigs);
	
	return eigs;
}

std::vector<matrix> bandStruct::getTransitions(vector3<double> kPoint)
{	// Compute transitions at kPoint
	matrix Pkx, Pky, Pkz, px, py, pz;
        const int cmsiz = cellMap.size();
	for (int ip = 0; ip<cmsiz; ip++){
		Pkx += pxWannierArray[ip]*cis(2*M_PI * dot(cellMap[ip],kPoint));
		Pky += pyWannierArray[ip]*cis(2*M_PI * dot(cellMap[ip],kPoint));
		Pkz += pzWannierArray[ip]*cis(2*M_PI * dot(cellMap[ip],kPoint));
	}
	// Change basis of Px, Py, Pz to eigenbasis of Hk
	px = transpose(evecs) * Pkx * evecs;
	py = transpose(evecs) * Pky * evecs;
	pz = transpose(evecs) * Pkz * evecs;
	PkWannierArray.resize(3);
	PkWannierArray[0] = px;
	PkWannierArray[1] = py;
	PkWannierArray[2] = pz;

        return PkWannierArray;
}                                                
