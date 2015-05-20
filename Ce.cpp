#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"
#include "Units.h"

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of electronic heat capacity", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const int totalBlocks = inputMap.get("totalBlocks"); assert(totalBlocks>0);
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("totalBlocks = %d\n", totalBlocks);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Initialize Wannier bandstructure:
	const int bunchSize = 32;
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE");
	bs.setCacheSize(2*bunchSize);
	
	// Compute Ce
	double CeSum = 0., CeSumSq = 0.;
	logPrintf("Calculating Ce... "); logFlush();
	int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
	int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nKptsMin = nKptsN1/totalBlocks;
	const double Emax = 10*T; //max energy from Fermi level to consider
	for(int block=blockStart; block<blockStop; block++)
	{	Random::seed(block);
		double CeBlock = 0.;;
		double nKpts = 0.; int nBunches = 0;
		while(nKpts < nKptsMin)
		{	//Get a bunch of k-points with states near the Fermi level:
			std::vector< vector3<> > kArr; kArr.reserve(bunchSize);
			while(kArr.size() < bunchSize)
			{	//Diagonalize Hamiltonians at a set of random k-points:
				std::vector< vector3<> > kTmp(bunchSize);
				for(vector3<>& k: kTmp)
					for(int j=0; j<3; j++)
						k[j] = Random::uniform();
				std::vector<diagMatrix> Etmp = bs.getStates(kTmp, Emax);
				//Add k-points with appropriate states:
				int nFound = 0, nAdded = 0;
				for(int ik=0; ik<bunchSize; ik++)
				{	bool worthwhile = false;
					for(int b=0; b<Etmp[ik].nRows(); b++)
						if(fabs(Etmp[ik][b]) < Emax)
						{	worthwhile = true;
							break;
						}
					if(worthwhile)
					{	nFound++;
						if(kArr.size() < bunchSize)
						{	kArr.push_back(kTmp[ik]);
							nAdded++;
						}
					}
				}
				nKpts += bunchSize * (nFound ? nAdded * (1./nFound) : 1.); //number of k-points examined to get the relevant ones (needed for normalization)
			}
			nBunches++;
			
			//Get energies for selected bunch:
			std::vector<diagMatrix> Earr = bs.getStates(kArr, Emax);
			
			for(int ik1=0; ik1<bunchSize; ik1++)
			{	//Calculate heat capacity for each k-point ik1i
				for(int v=0; v<Earr[ik1].nRows(); v++) //for each band
				{	//kb=1 in atomic units, E-mu already happened in bandstruct
					double dfdT = (Earr[ik1][v] * std::pow(1/cosh(Earr[ik1][v]/(2*T)),2)) / (4*T*T);
					CeBlock += dfdT * Earr[ik1][v];
				}
			}
		}
		CeSum += CeBlock; CeSumSq += std::pow(CeBlock,2);
	}

	mpiUtil->allReduce(CeSum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(CeSumSq, MPIUtil::ReduceSum);
	double Ce = CeSum / totalBlocks;
	double CeStd = sqrt(CeSumSq/totalBlocks - Ce*Ce)/sqrt(totalBlocks);
	logPrintf("Ce = %lg +/- %lg J/eV\n", Ce/(Joule/eV), CeStd/(Joule/eV));
	
	finalizeSystem();
}
