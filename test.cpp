#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include "histogram.h"
using std::vector;
using std::cout;

int main(int argc, char** argv)
{	
	histogram hist(1,0.5,10);
	hist.addEvent(2,5.4);
	//initSystem(argc, argv);
	
/*	FILE * pFile;
	pFile = fopen ("myfile.txt" , "w+");

	int nRows = 3;
	int nCols = 3;
	matrix mat (nRows,nCols);
	randomize(mat);
	cout << "numRows = "<< mat.nRows() << " numCols = " << mat.nCols() << std::endl;
	//mat.print(pFile);

	matrix hmat = dagger_symmetrize(mat);
	//hmat.print(pFile);

	matrix evecs (nRows,nCols);
	diagMatrix eigs (nRows,nCols);
	hmat.diagonalize(evecs, eigs);	
	//evecs.print(pFile);
	eigs.print(pFile);
	cout << nrm2(hmat - evecs * eigs * dagger(evecs));	

	matrix m (1,nCols);
	std::vector<complex> v(nCols,3.5);
	memcpy(m.data(),v.data(),m.nData()*sizeof(complex));
	//m.print(pFile);
*/
	/* initialize random seed:
	srand (time(NULL));
	double randnum;

	for( int i = 0; i < mat.nRows(); i++ ){
		for( int j = 0; j < mat.nCols(); j++){
			// generate random number betweeb 0 and 1000
			randnum = rand() % 1000;
			mat.set(i,j,randnum);
		}
	}*/
/*
	logPrintf("\nHello world from process %d of %d using JDFTx!\n\n", mpiUtil->iProcess(), mpiUtil->nProcesses());
	
	finalizeSystem();
*/	return 0;
}

