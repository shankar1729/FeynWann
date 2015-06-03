#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"
#include "Units.h"
#include <fstream>      // std::ifstream
#include <algorithm>    // std::lower_bound
#include "Histogram.h"

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of electron-phonon coupling", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const int totalBlocks = inputMap.get("totalBlocks"); assert(totalBlocks>0);
	double mu = inputMap.get("mu");
	const double rT = inputMap.get("rT") * Kelvin; //room temperature, used for energy conserving delta parameters
	double Te = inputMap.get("Te") * Kelvin; //electron temperature
	const double Tl = inputMap.get("Tl") * Kelvin; //lattice temperature
	const double EplasmonMax = inputMap.get("EplasmonMax") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("totalBlocks = %d\n", totalBlocks);
	logPrintf("mu = %lg\n", mu);
	logPrintf("rT = %lg\n", rT);
	logPrintf("Te = %lg\n", Te);
	logPrintf("Tl = %lg\n", Tl);
	logPrintf("EplasmonMax = %lg\n", EplasmonMax);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Read in Te and mu from text file
	std::ifstream ifs("CeTnew2.dat");
	std::vector<double> TeArr, TeArr2, muArr;
	double tt, cc, mumu;;
	while(ifs >> tt >> cc >> mumu)
	{	TeArr.push_back(tt * Kelvin);
		TeArr2.push_back(fabs(tt * Kelvin -Te));
		muArr.push_back(mumu * eV);
	}
	// Adjust mu to account for Te value
	int TeIndex=0;
	for (unsigned int i=0; i<TeArr.size(); i++)
	{	if (TeArr2[i]<TeArr2[TeIndex])
			TeIndex=i;
	}
	Te = TeArr[TeIndex]; mu = muArr[TeIndex];
	logPrintf("Temp dependent adjusted values: Te = %lg, mu = %lg\n", Te, mu);//print values for debugging

	//Initialize Wannier bandstructure:
	const int bunchSize = 32;
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE");
	bs.setCacheSize(2*bunchSize);
	
	//Initialize histograms for matrix elements
	diagMatrix E0 = bs.getStates(vector3<>()); //Gamma point eigenvalues
	double Emin = *std::min_element(E0.begin(), E0.end());
	Histogram MepNum(Emin, rT, EplasmonMax);
	Histogram MepDen(Emin, rT, EplasmonMax);

	// Compute G
	double Gsum = 0., GsumSq = 0.;
	logPrintf("Calculating G... "); logFlush();
	int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
	int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nKptsMin = nKptsN1/totalBlocks;
	double Omega = fabs(det(R)); //unit cell volume
	const double Emax = 10*rT; //max energy from Fermi level to consider
	double EconserveExpFac = -0.5/(rT*rT), EconservePrefac = 1./(sqrt(2*M_PI)*rT); //energy conserving Gaussian parameters
	for(int block=blockStart; block<blockStop; block++)
	{	Random::seed(block);
		double Gblock = 0.;
		double nKpts = 0.;
		double nKpairs = 0.;
		while(nKpts < nKptsMin)
		{	//Get a bunch of k points:
			std::vector< vector3<> > kArr(bunchSize);
			for(vector3<>& k: kArr)
 				for(int j=0; j<3; j++)
 					k[j] = Random::uniform();
 			nKpts += bunchSize;
			nKpairs += bunchSize*(bunchSize-1);
			
			//Get energies for selected bunch:
			std::vector<diagMatrix> Earr = bs.getStates(kArr, Emax);
			
			diagMatrix omegaPh[bunchSize];
			std::vector<matrix> gePh[bunchSize];
			for(int ik1=0; ik1<bunchSize; ik1++)
			{	//Calculate phonon stuff for each pair of k-points involving ik1
				bs.setPhononMatElemArray(kArr[ik1], kArr, gePh);
				for(int ik2=0; ik2<bunchSize; ik2++)
					omegaPh[ik2] = bs.getPhononModes(kArr[ik1] - kArr[ik2]);
	
				for(int v=0; v<Earr[ik1].nRows(); v++)
				{	for(int ik2=0; ik2<bunchSize; ik2++)
						if(ik2 != ik1)
							for(int c=0; c<Earr[ik2].nRows(); c++)
							{	double fi = 1./(exp(Earr[ik1][v]/Te)+1);
								double fj = 1./(exp(Earr[ik2][c]/Te)+1);
								for(int alpha=0; alpha<omegaPh[ik2].nRows(); alpha++)
								{	double nPh = 1./(exp(omegaPh[ik2][alpha]/Tl) - 1.);
									double gePhSq = gePh[ik2][alpha](c,v).norm();
									double delta = EconservePrefac * exp(EconserveExpFac * std::pow(Earr[ik1][v]-Earr[ik2][c] + omegaPh[ik2][alpha],2));
									double occFactors = (fi-fj)*nPh  - fj*(1-fi);
									Gblock += 2 * M_PI * spinWeight * omegaPh[ik2][alpha] * gePhSq * occFactors * delta;
									//Matrix element squared (weighted by energy conservation)
									MepNum.addEvent(Earr[ik1][v], delta * gePhSq);
									MepNum.addEvent(Earr[ik2][c], delta * gePhSq);
									MepDen.addEvent(Earr[ik1][v], delta);
									MepDen.addEvent(Earr[ik2][c], delta);
								}
							}
				}
			}
		}
		Gblock /= (nKpairs * (Tl - Te));
		Gsum += Gblock; GsumSq += std::pow(Gblock,2);
	}

	mpiUtil->allReduce(Gsum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(GsumSq, MPIUtil::ReduceSum);
	double G = Gsum / totalBlocks;
	double Gstd = sqrt(GsumSq/totalBlocks - G*G)/sqrt(totalBlocks);
	double Gscale = 1./(Omega * Joule*invSeconds/(std::pow(meter,3)*Kelvin)); //per unit cell and switch to SI
	logPrintf("G = %lg +/- %lg W/(m^3 K)\n", G*Gscale, Gstd*Gscale);
	
	MepNum.allReduce(MPIUtil::ReduceSum);
	MepDen.allReduce(MPIUtil::ReduceSum);
	if(mpiUtil->isHead())
	{	ofstream ofs("Mep.dat");
		for(size_t i=0; i<MepNum.out.size(); i++)
			ofs << (MepNum.Emin + i*MepNum.dE)/eV << '\t' << MepNum.out[i]/MepDen.out[i] << '\n';
	}

	
	finalizeSystem();
}
