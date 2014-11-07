#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include "bandStruct.h"

int main(int argc, char** argv)
{       initSystem(argc, argv);
	
	// Read band structure k-points
	ifstream readKpoints("bandstruct.kpoints");
        string headerLine;
	getline(readKpoints, headerLine); getline(readKpoints,headerLine);//read and ignore two  header line
        vector3<double> kps;
        double kpointWeight;
	string kp;
	std::vector< vector3<double> > kPoints;
        while(readKpoints >> kp >> kps[0] >> kps[1] >> kps[2] >> kpointWeight)
                kPoints.push_back(kps);
        readKpoints.close();
	int numKpoints = kPoints.size();
	std::cout << "size of kPoints vector = " << numKpoints << std::endl;
	std::cout << "size of kpoint vector element = " << sizeof(kPoints.at(1))/8 << std::endl;

	// Compute & record interpolated band strucutre;
	diagMatrix eigs;
	FILE * eigsTxt;
	eigsTxt = fopen("WannierBandstruct.eigenvals","w+");
	bandStruct bs("wannier", 0.);	
	for(int ik=0; ik < numKpoints; ik++){
		vector3<double> kpnt = kPoints.at(ik);
		eigs = bs.getStates(kpnt);
		eigs.print(eigsTxt);
	}

	logPrintf("\nHello world from process %d of %d using JDFTx!\n\n", mpiUtil->iProcess(), mpiUtil->nProcesses());

        finalizeSystem();
        return 0;
}
