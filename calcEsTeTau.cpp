#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "InputMap.h"
#include "Units.h"

inline double fermi(double x) { return x>30. ? exp(-x) : 1./(1.+exp(x)); } //avoid overflow issues
inline double argLW(double E,double Es) { return sqrt(E)/(E+Es) + 1/sqrt(Es) * atan(sqrt(E/Es)); }

int main(int argc, char** argv)
{       string inputFilename; bool dryRun, printDefaults;
        initSystemCmdline(argc, argv, "Calculation of q_TF and E_S from Vallee paper", inputFilename, dryRun, printDefaults);

        //Get the system parameters (mu, T, lattice vectors etc.)
        InputMap inputMap(inputFilename);
        const int nKptsN1 = inputMap.get("nKptsN1");
        const int totalBlocks = inputMap.get("totalBlocks"); assert(totalBlocks>0);
	const double T = inputMap.get("T")*eV; // jellium fermi energy
	const double beta = inputMap.get("beta"); // from vallee paper, q_s = beta * q_TF
	const double Zjellium = inputMap.get("Zjellium");
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	const double TeMin = inputMap.get("TeMin") * Kelvin; //electron temperature grid start
	const double TeMax = inputMap.get("TeMax") * Kelvin; //electron temperature grid stop
	const double TeStep = inputMap.get("TeStep") * Kelvin; //electron temperature grid spacing
	const double Tl = inputMap.get("Tl") * Kelvin; //lattice temperature
	const double Es = inputMap.get("Es"); // Es in hartrees, as defined in Vallee paper
	const double epsB = inputMap.get("epsilonB"); // epsilon_b, as defined in Vallee paper
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	double mu = inputMap.get("mu"); //initial guess only - will be calculated self-consistently in this executable

        logPrintf("\nInputs after conversion to atomic units:\n");
        logPrintf("nKptsN1 = %d\n", nKptsN1);
        logPrintf("totalBlocks = %d\n", totalBlocks);
        logPrintf("T = %lg\n", T);
        logPrintf("beta = %lg\n", beta);
	logPrintf("Zjellium = %lg\n", Zjellium);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	logPrintf("TeMin = %lg\n", TeMin);
	logPrintf("TeMax = %lg\n", TeMax);
	logPrintf("TeStep = %lg\n", TeStep);
	logPrintf("Tl = %lg\n", Tl);
 	logPrintf("dE = %lg\n", dE);
	logPrintf("mu = %lg\n", mu);
	logPrintf("epsilon_b = %lg\n", epsB);
	logPrintf("Es: = %lg\n", Es);       

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
				qSqBlock += dFdE / (epsilon_o*epsB);
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
	double EsCalc = qs*qs/2;
	logPrintf("Es = %lg\n", EsCalc);

	// ==========================  Calcualte Te dependence of Tau ==========================================
	
	//Initialize temperature grid:
	std::vector<double> TeArr(int(ceil((TeMax-TeMin)/TeStep)));
	for(size_t iT=0; iT<TeArr.size(); iT++)
		TeArr[iT] = TeMin + TeStep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", TeArr.front()/Kelvin, TeArr.back()/Kelvin, TeArr.size());

	double iT = 0;
	//Initialize energy grid:
	std::vector<double> EArr(int(ceil((2*eV+10*TeArr[iT])/dE)));
	for(size_t iE=0; iE<EArr.size(); iE++)
		EArr[iE] = mu - 1*eV - 5*TeArr[iT] + dE*iE;

	std::vector<double> invTauTe(EArr.size());
	//std::vector< std::vector<double> > invTauTe(EArr.size());
	double invTe = 1./TeArr[iT];
	for(size_t iE=0; iE<EArr.size(); iE++)
	{	double lPrefac = -1 / (32 * std::pow(M_PI,3) * std::pow(1/(4*M_PI)*epsB,2) * Es * sqrt(EArr[iE]));
		for(size_t iE1=0; iE1<EArr.size(); iE1++)
		{	for(size_t iE2=0; iE2<EArr.size(); iE2++)
			{	//double& dmuCur = dmu[iT];
				double dmuCur = mu;
				double f = fermi(invTe*(EArr[iE] - dmuCur));
				double f1 = fermi(invTe*(EArr[iE1] - dmuCur));
				double f2 = fermi(invTe*(EArr[iE2] - dmuCur));
				double E3 = EArr[iE] + EArr[iE1] - EArr[iE2];
				double f3 = fermi(invTe*(E3 - dmuCur));
				double occFactor = f * f1 * (1-f2) * (1-f3);
				double topLim = std::min(std::pow(sqrt(EArr[iE1])+sqrt(E3),2),std::pow(sqrt(EArr[iE])+sqrt(EArr[iE2]),2));
				double lowLim = std::max(std::pow(sqrt(EArr[iE1])-sqrt(E3),2),std::pow(sqrt(EArr[iE])-sqrt(EArr[iE2]),2));
				double arg = argLW(topLim,Es) - argLW(lowLim,Es);
				invTauTe[iE] += lPrefac*arg*occFactor*dE*dE;
				//invTauTe[iE][iT] += lPrefac*arg*occFactor*dE*dE;
			}
		}

	}
	ofstream ofs("invTauTe.dat");
	ofs << "#E[eV] invTauTe[invSeconds]i\n";
	for(size_t iE=0; iE<EArr.size(); iE++)
		ofs << EArr[iE]/eV << '\t' << invTauTe[iE]/invSeconds << '\n';

	finalizeSystem();
}
