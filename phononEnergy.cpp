#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"
#include "Units.h"
#include "Histogram.h"
#include "Epsilon.h"

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Phonon energy", inputFilename, dryRun, printDefaults);

	//Initialize Wannier bandstructure:
	BandStruct bs("Wannier/wannier", 0., 2., "Wannier/totalE");

	//Initialize k-array
	double M = 50;
	std::vector<vector3<>> kPoints(M*5);
	vector3<> k;
	for(int m=0; m<M; m++)
	{	//Gamma to X: (0,0,0) to (0,0.5,0.5)
		k[0] = 0.0;
		k[1] = 0.5*m/M;
		k[2] = 0.5*m/M;
		kPoints[m] = k;
	}
	for(int m=0; m<M; m++)
	{	// X to W: (0,0.5,0.5) to (0.25,0.75,0.5)
		k[0] = 0.25*(m/M);
		k[1] = 0.5+0.25*m/M;
		k[2] = 0.5;
		kPoints[M+m] = k;
	}
	for(int m=0; m<M; m++)
	{	// W to L: (0.25,0.75,0.5) to (0.5,0.5,0.5)
		k[0] = 0.25+0.25*m/M;
		k[1] = 0.75-0.25*m/M;
		k[2] = 0.5;
		kPoints[2*M+m] = k;
	}
	for(int m=0; m<M; m++)
	{	// L to Gamma: (0.5, 0.5, 0.5) to (0,0,0)
		k[0] = 0.5-0.5*m/M;
		k[1] = 0.5-0.5*m/M;
		k[2] = 0.5-0.5*m/M;
		kPoints[3*M+m] = k;
	}
	for(int m=0; m<M; m++)
	{	//Gamma to K: (0,0,0) to (0.375, 0.75, 0.375)
		k[0] = 0.375 * m/M;
		k[1] = 0.75 * m/M;
		k[2] = 0.375*m/M;
		kPoints[4*M+m] = k;
	} 

	//calcualte phonon energy grid:
	std::vector<double> phononEnergy1(M*5);
	std::vector<double> phononEnergy3(M*5);
	std::vector<double> phononEnergy2(M*5);
	diagMatrix phononModes;
	for(int i=0; i<(M*5); i++)
	{	phononModes =  bs.getPhononModes(kPoints[i]);
		phononEnergy1[i] = phononModes[0];
		phononEnergy2[i] = phononModes[1];
		phononEnergy3[i] = phononModes[2];
	}

	//write to file
	ofstream ofs("phononEnergy.dat");
	ofs << "#kx	ky	kz	phononEnergy\n";
	for(int iT=0; iT<M*5; iT++)
	{	k = kPoints[iT];
		ofs << kPoints[iT][0] << '\t'
		<< kPoints[iT][1] << '\t'
		<< kPoints[iT][2] << '\t'
		<< phononEnergy1[iT]/eV << '\t'
		<< phononEnergy2[iT]/eV << '\t'
		<< phononEnergy3[iT]/eV << '\n';
	}
	
	finalizeSystem();
}
