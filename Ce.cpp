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
	double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const double Z = inputMap.get("Z");
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("totalBlocks = %d\n", totalBlocks);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("Z (electrons per unit cell) = %lg\n", Z);
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
	
	double Omega = fabs(det(R)); //unit cell volume
	const double Emax = 10*T; //max energy from Fermi level to consider

	double dmuT = 0.; //temperature dependent mu correction
	while(true)
	{	// Compute Ce
		double CeSum = 0., CeSumSq = 0.;
		double Nsum = 0., NsumSq = 0., dNdmuSum = 0.;
		int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
		int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
		int nKptsMin = nKptsN1/totalBlocks;
		for(int block=blockStart; block<blockStop; block++)
		{	Random::seed(block);
			double CeBlock = 0., NBlock = 0., dNdmuBlock=0.;
			double nKpts = 0.;
			while(nKpts < nKptsMin)
			{	//Get a bunch of k-points:
				std::vector< vector3<> > kArr(bunchSize);
				for(vector3<>& k: kArr)
					for(int j=0; j<3; j++)
						k[j] = Random::uniform();
				nKpts += bunchSize;
				//Get energies for selected bunch:
				std::vector<diagMatrix> Earr = bs.getStates(kArr, Emax);
				for(int ik1=0; ik1<bunchSize; ik1++)
				{	//Calculate heat capacity for each k-point ik1i
					for(int v=0; v<Earr[ik1].nRows(); v++) //for each band
					{	//kb=1 in atomic units, E-mu already happened in bandstruct
						double x = (Earr[ik1][v] - dmuT) / T; 
						double f = 1. / (1. + exp(x));
						double dfdx = -1./std::pow(2*cosh(0.5*x), 2);
						double dfdmu = dfdx * (-1./T);
						double dfdT = dfdx * (-x/T); //chain rule
						CeBlock += dfdT * Earr[ik1][v];
						NBlock += f;
						dNdmuBlock += dfdmu;
					}
				}
			}
			double w = spinWeight / nKpts;
			CeSum += w*CeBlock; CeSumSq += std::pow(w*CeBlock,2);
			Nsum += w*NBlock; NsumSq += std::pow(w*NBlock,2);
			dNdmuSum += w*dNdmuBlock;
		}
		//Results at current mu offset:
		mpiUtil->allReduce(CeSum, MPIUtil::ReduceSum);
		mpiUtil->allReduce(CeSumSq, MPIUtil::ReduceSum);
		mpiUtil->allReduce(Nsum, MPIUtil::ReduceSum);
		mpiUtil->allReduce(NsumSq, MPIUtil::ReduceSum);
		mpiUtil->allReduce(dNdmuSum, MPIUtil::ReduceSum);
		double Ce = CeSum / totalBlocks;
		double CeStd = sqrt(CeSumSq/totalBlocks - Ce*Ce)/sqrt(totalBlocks);
		double CeScale = 1./(Omega * Joule/(std::pow(meter,3)*Kelvin)); //per unit cell and switch to SI
		double N = Nsum / totalBlocks;
		double Nstd = sqrt(NsumSq/totalBlocks - N*N)/sqrt(totalBlocks);
		double dNdmu = dNdmuSum / totalBlocks;
		logPrintf("dmuT=%+lf  N=%lf+/-%lf:  Ce = %.1lf +/- %.1lf J/(m^3 K)\n", dmuT, N,Nstd, Ce*CeScale, CeStd*CeScale);
		fflush(globalLog);
		//Update mu offset:
		if(fabs(N-Z) < Nstd) break; //can't converge more accurately than error in N
		dmuT += (Z - N) / dNdmu;
	}
	
	finalizeSystem();
}
