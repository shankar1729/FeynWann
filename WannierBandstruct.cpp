#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
using std::cout;
using std::endl;
using std::vector;

int main(int argc, char** argv)
{       initSystem(argc, argv);
	
	// Read cell map
	ifstream readCellMap("wannier.mlwfCellMap");
	vector< vector3<int> > cellMap;
	string headerLine; getline(readCellMap, headerLine); //read and ignore header line
	vector3<int> cm;
	double x,y,z;
	while(readCellMap >> cm[0] >> cm[1] >> cm[2] >> x >> y >> z)
		cellMap.push_back(cm);
	readCellMap.close();	
	const int numCellMapPoints = cellMap.size();

	// Read wannier hamiltonian
	off_t hWannierSize = fileSize("wannier.mlwfH")/16; //convert from bytes to number of complex numbers
	matrix hWannier(hWannierSize,1);
	hWannier.read("wannier.mlwfH");
	int numBands = sqrt(hWannierSize/numCellMapPoints);
	hWannier.reshape(numBands,numBands*numCellMapPoints);	
	matrix hWannierArray[numCellMapPoints];
	for (int ii = 0; ii<numCellMapPoints; ii++)
		hWannierArray[ii] = hWannier(0,numBands,0+ii*numBands,numBands+ii*numBands);

	// Read band structure k-points
	ifstream readKpoints("bandstruct.kpoints");
        getline(readKpoints, headerLine); getline(readKpoints,headerLine);//read and ignore two  header line
        vector3<double> kps;
	vector< vector3<double> > kPoints;
        double kpointWeight;
	string kp;
        while(readKpoints >> kp >> kps[0] >> kps[1] >> kps[2] >> kpointWeight)
                kPoints.push_back(kps);
        readKpoints.close();

	// Compute & record interpolated band strucutre;
	int numKpoints = kPoints.size();
	matrix evecs;
	diagMatrix eigs;
	FILE * eigsTxt;
	eigsTxt = fopen("WannierBandstruct.eigenvals","w+");
	for(int ik=0; ik < numKpoints; ik++){
		matrix Hk,Hkh;
		for (int ic = 0; ic< numCellMapPoints; ic++)
			Hk +=  hWannierArray[ic] * cis(2*M_PI * dot(cellMap[ic],kPoints.at(ik)));
  		Hkh = dagger_symmetrize(Hk);
		Hkh.diagonalize(evecs, eigs);
		eigs.print(eigsTxt);
	}

        finalizeSystem();
        return 0;
}
