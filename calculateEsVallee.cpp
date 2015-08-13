#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "InputMap.h"
#include "Units.h"

int main(int argc, char** argv)
{       string inputFilename; bool dryRun, printDefaults;
        initSystemCmdline(argc, argv, "Calculation of q_TF and E_S from Vallee paper", inputFilename, dryRun, printDefaults);

        //Get the system parameters (mu, T, lattice vectors etc.)
        InputMap inputMap(inputFilename);
        const int nKptsN1 = inputMap.get("nKptsN1");
        const int totalBlocks = inputMap.get("totalBlocks"); assert(totalBlocks>0);
        const double T = inputMap.get("T") * eV;
	const double Ef = inputMap.get("Ef"); // jellium fermi energy
	const double epsilon_b = inputMap.get("epsilon_b"); // from vallee paper
	const double beta = inputMap.get("beta"); // from vallee paper, q_s = beta * q_TF

        logPrintf("\nInputs after conversion to atomic units:\n");
        logPrintf("nKptsN1 = %d\n", nKptsN1);
        logPrintf("totalBlocks = %d\n", totalBlocks);
        logPrintf("T = %lg\n", T);
        logPrintf("Ef = %lg\n", Ef);
        logPrintf("epsilon_b = %lg\n", epsilon_b);
        logPrintf("beta = %lg\n", beta);
        
	const int bunchSize = 32;
	
	const double epsilon_o = 1/(4*M_PI);;

	// Compute qTF
	double qSqSum = 0., qSqSumSq=0.;
	int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
	int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nKptsMin = nKptsN1/totalBlocks;
	for(int block=blockStart; block<blockStop; block++)
	{	Random::seed(block);
                double qSqBlock = 0.;
                double nKpts = 0.; int nBunches = 0;
                while(nKpts < nKptsMin)
                {       //Get a bunch of k-points
                        std::vector< vector3<> > kArr; kArr.reserve(bunchSize);
                        while(kArr.size() < bunchSize)
                        {	vector3<> k;
                                for(int j=0; j<3; j++)
					k[j] = Random::uniform();
				kArr.push_back(k);
				nKpts += bunchSize;
                        }
                        nBunches++;

			//Get energies and q_TF squared sum for selected bunch
                        for(int ik=0; ik<bunchSize; ik++)
			{	double Ek = (kArr[ik][0]*kArr[ik][0]+kArr[ik][1]*kArr[ik][1]+kArr[ik][2]*kArr[ik][2])/2;
				double dFdE = -1 / (2*T*cosh((Ef-Ek)/T) + 2*T);
				qSqBlock += dFdE / (epsilon_o*epsilon_b);

			}
		}
		qSqSum += qSqBlock/nKpts;
		qSqSumSq += std::pow(qSqBlock/nKpts,2);
	}

	mpiUtil->allReduce(qSqSum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(qSqSumSq, MPIUtil::ReduceSum);
	double qTFSq = qSqSum / totalBlocks;
	double qTFSqstd = sqrt(fabs(qSqSumSq)/totalBlocks - fabs(qTFSq*qTFSq))/sqrt(totalBlocks);
        logPrintf("qTF squared = %lg +/- %lg\n", qTFSq, qTFSqstd);

	double qTF = sqrt(fabs(qTFSq));
	double qs = beta * qTF;
	double Es = qs*qs/2;
	logPrintf("Es = %lg\n", Es);
	finalizeSystem();
}
