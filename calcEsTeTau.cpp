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
	const double T = inputMap.get("T")*eV; // jellium fermi energy
	const double epsilon_b = inputMap.get("epsilon_b"); // from vallee paper
	const double beta = inputMap.get("beta"); // from vallee paper, q_s = beta * q_TF
	const double Zjellium = inputMap.get("Zjellium");
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	const double TeMin = inputMap.get("TeMin") * Kelvin; //electron temperature grid start
	const double TeMax = inputMap.get("TeMax") * Kelvin; //electron temperature grid stop
	const double TeStep = inputMap.get("TeStep") * Kelvin; //electron temperature grid spacing
	const double Tl = inputMap.get("Tl") * Kelvin; //lattice temperature

        logPrintf("\nInputs after conversion to atomic units:\n");
        logPrintf("nKptsN1 = %d\n", nKptsN1);
        logPrintf("totalBlocks = %d\n", totalBlocks);
        logPrintf("T = %lg\n", T);
        logPrintf("epsilon_b = %lg\n", epsilon_b);
        logPrintf("beta = %lg\n", beta);
	logPrintf("Zjellium = %lg\n", Zjellium);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	logPrintf("TeMin = %lg\n", TeMin);
	logPrintf("TeMax = %lg\n", TeMax);
	logPrintf("TeStep = %lg\n", TeStep);
	logPrintf("Tl = %lg\n", Tl);
        
	const int bunchSize = 32;
	
	const double epsilon_o = 1/(4*M_PI);;
	const double nJellium = Zjellium / fabs(det(R));
	logPrintf("nJellium = %lg per bohr^3, %lg per m^3\n", nJellium, nJellium*meter*meter*meter);
	const double kF = std::pow(3 * M_PI * M_PI * nJellium,1./3);
	const double EfJellium = 0.5 * kF * kF;
	logPrintf("EfJellium = %lf hartrees, %lg eV, %lg Joules\n",EfJellium, EfJellium/eV, EfJellium/Joule);

	// Compute qTF
	double qSqSum = 0., qSqSumSq=0.;
	int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
	int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nKptsMin = nKptsN1/totalBlocks;
	for(int block=blockStart; block<blockStop; block++)
	{	Random::seed(block);
		double qSqBlock = 0.;
		double nKpts = 0.;
                while(nKpts < nKptsMin)
		{	//Get a bunch of k-points
			std::vector< vector3<> > kArr(bunchSize);
			for(vector3<>& k: kArr)
			{	for(int j=0; j<3; j++)
					k[j] = Random::uniform();
				nKpts += 1;
			}

			//Get energies and q_TF squared sum for selected bunch
			for(int ik=0; ik<bunchSize; ik++)
			{	double Ek = (kArr[ik][0]*kArr[ik][0]+kArr[ik][1]*kArr[ik][1]+kArr[ik][2]*kArr[ik][2])/2;
				double dFdE = -1 / (2*T*cosh((EfJellium-Ek)/T) + 2*T);
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

	// ==========================  Calcualte Te dependence of Tau ==========================================
	
	//Initialize temperature grid:
	std::vector<double> TeArr(int(ceil((TeMax-TeMin)/TeStep)));
	for(size_t iT=0; iT<TeArr.size(); iT++)
		TeArr[iT] = TeMin + TeStep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", TeArr.front()/Kelvin, TeArr.back()/Kelvin, TeArr.size());

	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{
		//Retrieve k-point bunch:
		const std::vector< vector3<> >& kArr = kArrArr[iBunch];

		//Calculate electronic states and matrix elements and T_e contribution to lifetime for bunch:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector<diagMatrix> ImEarr = lineWidth(kArr);
		std::vector< std::vector<matrix> > Parr = bs.getDipoleMatElem(kArr);
		std::vector< std::vector<diagMatrix> > Farr(bunchSize); //fillings by k-point, temperature and band
		std::vector< std::vector<double> > invTauTe(bunchSize);
		for(int ik=0; ik<bunchSize; ik++)
		{	Farr[ik].resize(TeArr.size());
			double Ejel = Ejellium(kArr[ik]);
			double lPrefac = -1 / (32 * std::pow(M_PI,3) * std::pow(4*M_PI*epsB,2) * Es * sqrt(Ejel));
			for(size_t iT=0; iT<TeArr.size(); iT++)
			{	double invTe = 1./TeArr[iT];
				Farr[ik][iT] = Earr[ik];
				for(double& f: Farr[ik][iT]) //convert to fillings:
					f = fermi(invTe*(f-dmu[iT]));
				// Calcualte T_e contribution to lifetime
				for(int ik1=0; ik1<bunchSize; ik1++)
				{       for(int ik2=0; ik2<bunchSize; ik2++)
					{	double& dmuCur = dmu[iT];
						double f = fermi(invTe*(Ejel - dmuCur));
						double E1jel = Ejellium(kArr[ik1]);
						double f1 = fermi(invTe*(E1jel - dmuCur));
						double E2jel = Ejellium(kArr[ik2]);
						double f2 = fermi(invTe*(E2jel - dmuCur));
						double E3jel = E1jel + Ejellium(kArr[ik1]) - Ejellium(kArr[ik2]);
						double f3 = fermi(invTe*(E3jel - dmuCur));
						double occFactor = f * f1 * (1-f2) * (1-f3);
						double topLim = std::min(std::pow(sqrt(E1jel)+sqrt(E3jel),2),std::pow(sqrt(Ejel)+sqrt(E2jel),2));
						double lowLim = std::max(std::pow(sqrt(E1jel)-sqrt(E3jel),2),std::pow(sqrt(Ejel)-sqrt(E2jel),2));
						double arg = argLW(topLim,Es) - argLW(lowLim,Es);
						invTauTe[ik][iT] += lPrefac*arg*occFactor/(bunchSize*bunchSize);
					}
				}
			}
		}

	}
	finalizeSystem();
}
