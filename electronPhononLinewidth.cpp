#include <core/Util.h>
#include <electronic/matrix.h>
#include <electronic/ColumnBundle.h>
#include <electronic/Everything.h>
#include <fstream>
#include <iterator>
#include <sstream>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/LatticeUtils.h>
#include <commands/parser.h>
#include "BandStruct.h"
#include "InputMap.h"
#include "Units.h"

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Metropolis calculation of two-plasmon decay rate", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	
	logPrintf("\nInputs after conversion to atomic units:\n");
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
	
	//Construct representation of irreducible wedge using dummy JDFTx input:
	const int Nk = 36; double invNk = 1./Nk; //k-point sampling for e-ph, preferably a multiple of the pure electronic one below
	std::vector< std::pair<string,string> > jdftInputs;
	jdftInputs.push_back(std::make_pair<string,string>("kpoint-folding", "12 12 12")); //Hard-coded k-point sampling used to generate Wannier functions
	jdftInputs.push_back(std::make_pair<string,string>("lattice", "face-centered cubic 2")); //to make setup lightweight, only k-mesh used
	jdftInputs.push_back(std::make_pair<string,string>("dump", "End DOS")); //to make supercell available
	Everything e;
	logPrintf("\n------- Dummy JDFTx setup to get k-mesh ------\n");
	logSuspend();
	parse(jdftInputs, e, false);
	e.setup();
	logResume();
	
	const Supercell& supercell = *(e.coulombParams.supercell);
	logPrintf("%d of %d kpoints in irreducible wedge.\n", e.eInfo.nStates, int(supercell.kmesh.size()));
	
	//Initialize Wannier band structure with electron phonon matrix elements:
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE");
	bs.setCacheSize(2*Nk+5);
	
	//Calculate lifetimes for states in irreducible wedge:
	int nBands = bs.getStates(vector3<>()).nRows();
	std::vector<diagMatrix> Gamma(e.eInfo.nStates, diagMatrix(nBands, 0.)); //decay rates in irreducible wedge
	std::vector<diagMatrix> E(e.eInfo.nStates, diagMatrix(nBands, 0.)); //eigenvalues in irreducible wedge
	double prefacGamma = M_PI/(Nk*Nk*Nk);
	double EconserveScaleFac = 1./T, EconservePrefac = 1./(M_PI*T); //energy conserving Lorentzian parameters
	std::vector< std::vector<matrix> > MePhArr(Nk);
	//Loop over blocks of one dimension of second k-point
	std::vector< vector3<> > k2arr(Nk);
	for(int ik0=0; ik0<Nk; ik0++)
	for(int ik1=0; ik1<Nk; ik1++)
	{	//Print progress:
		if(!ik1)
		{	logPrintf("Working on ik0 = %d of %d\n", ik0+1, Nk);
			logFlush();
		}
		//Initialize k2 array:
		for(int ik2=0; ik2<Nk; ik2++)
			k2arr[ik2] = (vector3<>(ik0,ik1,ik2) + 0.5) * invNk;
		//Loop over first k-point (irreducible wedge):
		for(int q=e.eInfo.qStart; q<e.eInfo.qStop; q++)
		{	vector3<> k1 = e.eInfo.qnums[q].k;
			E[q] = bs.getStates(k1);
			const diagMatrix& E1 = E[q];
			bs.setPhononMatElemArray(k1, k2arr, MePhArr.data());
			for(int ik2=0; ik2<Nk; ik2++)
			{	const vector3<>& k2 = k2arr[ik2];
				diagMatrix E2 = bs.getStates(k2);
				diagMatrix omegaPh = bs.getPhononModes(k1 - k2);
				const std::vector<matrix>& MePh = MePhArr[ik2];
				
				for(int b1=0; b1<E1.nRows(); b1++)
				{	for(int b2=0; b2<E2.nRows(); b2++)
					{	double f2 = 1./(exp(E2[b2]/T)+1);
						for(int alpha=0; alpha<omegaPh.nRows(); alpha ++)
						{	double gk = 1./(exp(omegaPh[alpha]/T) - 1.);
							double Msq_by_omega = MePh[alpha](b2,b1).norm() / omegaPh[alpha];
							for(int ae=-1; ae<=+1; ae+=2)
							{	double delta = EconservePrefac / (1. + std::pow(EconserveScaleFac * (E2[b2]-E1[b1] - ae*omegaPh[alpha]),2));
								double occFactors = (gk+0.5 - ae*(0.5-f2));
								Gamma[q][b1] += prefacGamma * occFactors * delta * Msq_by_omega;
							}
						}
					}
				}
			}
		}
	}
	FILE* fp = 0;
	if(mpiUtil->isHead()) fp = fopen("ImSigma_ePh.dat", "w");
	for(int q=0; q<e.eInfo.nStates; q++)
	{	Gamma[q].bcast(e.eInfo.whose(q));
		E[q].bcast(e.eInfo.whose(q));
		if(fp)
			for(int b=0; b<nBands; b++)
				fprintf(fp, "%+19.12le %19.12le\n", E[q][b]/eV, Gamma[q][b]/(1./fs));
	}
	if(fp) fclose(fp);
	
	//Wannierize the lifetimes:
	std::map<vector3<int>,double> cellMap = getCellMap(e.gInfo.R, supercell.Rsuper); //Similar to BandStruct::cellMap, but includes weights for boundary symmetrization
	size_t ikStart = (supercell.kmesh.size() * mpiUtil->iProcess()) / mpiUtil->nProcesses();
	size_t ikStop = (supercell.kmesh.size() * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	size_t nkMine = std::max(size_t(1), ikStop-ikStart); //avoid zero size matrices below
	matrix phase = zeroes(nkMine, cellMap.size());
	matrix ImSigmaWannierTilde = zeroes(nBands*nBands, nkMine);
	double kWeight = 1./supercell.kmesh.size();
	for(size_t ik=ikStart; ik<ikStop; ik++)
	{	//Apply unitary rotations at current k:
		vector3<> k = supercell.kmesh[ik];
		matrix evecs; bs.getStates(k, DBL_MAX, &evecs);
		matrix ImSigmaSub = evecs * Gamma[supercell.kmeshTransform[ik].iReduced] * dagger(evecs);
		callPref(eblas_copy)(ImSigmaWannierTilde.dataPref()+ImSigmaWannierTilde.index(0,ik-ikStart), ImSigmaSub.dataPref(), ImSigmaSub.nData());
		//Calculate required phases:
		int iCell = 0;
		for(auto cell: cellMap)
			phase.set(ik-ikStart, iCell++, cell.second * kWeight * cis(2*M_PI*dot(k, cell.first)));
	}
	//Fourier transform to Wannier space and save
	matrix ImSigmaWannier = ImSigmaWannierTilde * phase;
	ImSigmaWannier.allReduce(MPIUtil::ReduceSum);
	if(mpiUtil->isHead())
		ImSigmaWannier.dump("wannier.mlwfImSigma_ePh", spinWeight==2);
	
	finalizeSystem();
	return 0;
}
