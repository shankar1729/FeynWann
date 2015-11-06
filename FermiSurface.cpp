#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <algorithm>
#include "BandStruct.h"
#include "LineWidth.h"
#include "InputMap.h"
#include "Units.h"

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Generate event list for transport modules", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const double mu = inputMap.get("mu");
	const int spinWeight = round(inputMap.get("spinWeight"));

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("mu = %lg\n", mu);
	logPrintf("spinWeight = %d\n", spinWeight);

	const int nk = 64;
	matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * 0.5; //unit fcc lattice
	
	//Initialize bandstructure and linewidth
	BandStruct bs("Wannier/wannier", mu, spinWeight);
	LineWidth lineWidth("Wannier/wannier", bs);
	bs.setCacheSize(nk);
	
	//Calculate band energies and lifetimes:
	int nBands = bs.getStates(vector3<>()).nRows();
	int ik0start, ik0stop; TaskDivision(nk, mpiUtil).myRange(ik0start, ik0stop);
	int nk0mine = ik0stop - ik0start;
	typedef std::pair<double,double> dpair;
	std::vector<std::vector<dpair> > Epair(nBands, std::vector<dpair>(nk0mine*nk*nk));
	for(int ik0=ik0start; ik0<ik0stop; ik0++)
		for(int ik1=0; ik1<nk; ik1++)
		{	//Initialize k-points in bunch:
			std::vector< vector3<> > kArr(nk);
			for(int ik2=0; ik2<nk; ik2++)
				kArr[ik2] = R * (-1 + vector3<>(ik0,ik1,ik2) * (2./(nk-1)));
			//Calculate properties of bunch:
			std::vector<diagMatrix> Earr = bs.getStates(kArr);
			std::vector<diagMatrix> ImEarr = lineWidth(kArr, 0, 1); //e-ph only
			//Store into 3D grid:
			for(int ik2=0; ik2<nk; ik2++)
			{	size_t index = ((ik0-ik0start)*nk + ik1)*nk + ik2;
				for(int b=0; b<nBands; b++)
					Epair[b][index] = std::make_pair(Earr[ik2][b], 1./(2.*ImEarr[ik2][b])/fs);
			}
		}
	
	//Output bands which cross the Fermi level:
	for(int b=0; b<nBands; b+=(spinWeight==1 ? 2 : 1)) //pick only one from degenerate relativistic bands
	{	//Determine energy range:
		double Emin = std::min_element(Epair[b].begin(), Epair[b].end())->first;
		double Emax = std::max_element(Epair[b].begin(), Epair[b].end())->first;
		mpiUtil->allReduce(Emin, MPIUtil::ReduceMin);
		mpiUtil->allReduce(Emax, MPIUtil::ReduceMax);
		if(Emin>=0. || Emax<=0.) continue; //band does not cross Fermi level
		//Output binary file:
		char fname[256]; MPIUtil::File fp;
		sprintf(fname, "FermiSurface.%d.bin", b);
		mpiUtil->fopenWrite(fp, fname);
		mpiUtil->fseek(fp, ik0start*nk*nk*sizeof(dpair), SEEK_SET);
		mpiUtil->fwrite(Epair[b].data(), sizeof(dpair), Epair[b].size(), fp);
		mpiUtil->fclose(fp);
	}
	
	finalizeSystem();
}