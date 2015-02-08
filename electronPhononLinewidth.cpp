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
	const int Nk = 48; double invNk = 1./Nk; //k-point sampling for e-ph, preferably a multiple of the pure electronic one below
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
	bs.setCacheSize(Nk + e.eInfo.nStates);
	
	//Calculate lifetimes for states in irreducible wedge:
	int nBands = bs.getStates(vector3<>()).nRows();
	std::vector<diagMatrix> ImSigma(e.eInfo.nStates, diagMatrix(nBands, 0.)); //imaginary part of self-energy in irreducible wedge
	double prefacImSigma = 0.5 * M_PI/(Nk*Nk*Nk); //Note factor of 0.5 between decay rate and ImSigma due to squaring of wavefunctions to probability
	double EconserveScaleFac = 1./T, EconservePrefac = 1./(M_PI*T); //energy conserving Lorentzian parameters
	std::vector< std::vector<matrix> > MePhArr(Nk);
	//Loop over blocks of one dimension of second k-point
	std::vector< vector3<> > k2arr(Nk);
	int ik01start = (Nk*Nk * mpiUtil->iProcess()) / mpiUtil->nProcesses();
	int ik01stop = (Nk*Nk * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nk01mine = ik01stop-ik01start;
	int nk01interval = std::max(1, int(round(nk01mine/50.))); //interval for reporting progress
	for(int ik01=ik01start; ik01<ik01stop; ik01++)
	{	int ik0 = ik01 / Nk;
		int ik1 = ik01 - ik0 * Nk;
		//Print progress:
		int nk01done = ik01-ik01start+1;
		if(nk01done % nk01interval == 0)
		{	logPrintf("Working on nk01 = %d of %d\n", nk01done, nk01mine);
			logFlush();
		}
		//Initialize k2 array:
		for(int ik2=0; ik2<Nk; ik2++)
			k2arr[ik2] = (vector3<>(ik0,ik1,ik2) + 0.5) * invNk;
		//Loop over first k-point (irreducible wedge):
		for(int q=0; q<e.eInfo.nStates; q++)
		{	vector3<> k1 = e.eInfo.qnums[q].k;
			bs.setPhononMatElemArray(k1, k2arr, MePhArr.data());
			const diagMatrix& E1 = bs.getStates(k1);
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
								ImSigma[q][b1] += prefacImSigma * occFactors * delta * Msq_by_omega;
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
	{	ImSigma[q].allReduce(MPIUtil::ReduceSum);
		if(fp)
		{	diagMatrix E = bs.getStates(e.eInfo.qnums[q].k);
			for(int b=0; b<nBands; b++)
				fprintf(fp, "%+19.12le %19.12le\n", E[b]/eV, ImSigma[q][b]/(1./fs));
		}
	}
	if(fp) fclose(fp);
	
	//Raw binary output in JDFTx format: (the version actually used for final Wannierization)
	e.eInfo.write(ImSigma, "totalE.ImSigma_ePh", nBands);
	
	//Wannierize logImSigma (for debug only - will move to JDFTx/Wannier once e-e formalism is complete):
	std::map<vector3<int>,double> cellMap = getCellMap(e.gInfo.R, supercell.Rsuper); //Similar to BandStruct::cellMap, but includes weights for boundary symmetrization
	size_t ikStart = (supercell.kmesh.size() * mpiUtil->iProcess()) / mpiUtil->nProcesses();
	size_t ikStop = (supercell.kmesh.size() * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	size_t nkMine = std::max(size_t(1), ikStop-ikStart); //avoid zero size matrices below
	matrix phase = zeroes(nkMine, cellMap.size());
	matrix LogImSigmaWannierTilde = zeroes(nBands*nBands, nkMine);
	double kWeight = 1./supercell.kmesh.size();
	for(size_t ik=ikStart; ik<ikStop; ik++)
	{	//Apply unitary rotations at current k:
		vector3<> k = supercell.kmesh[ik];
		matrix evecs; bs.getStates(k, DBL_MAX, &evecs);
		diagMatrix LogImSigma;
		for(double IS: ImSigma[supercell.kmeshTransform[ik].iReduced])
			LogImSigma.push_back(log(IS));
		matrix LogImSigmaSub = evecs * LogImSigma * dagger(evecs);
		callPref(eblas_copy)(LogImSigmaWannierTilde.dataPref()+LogImSigmaWannierTilde.index(0,ik-ikStart), LogImSigmaSub.dataPref(), LogImSigmaSub.nData());
		//Calculate required phases:
		int iCell = 0;
		for(auto cell: cellMap)
			phase.set(ik-ikStart, iCell++, cell.second * kWeight * cis(2*M_PI*dot(k, cell.first)));
	}
	//Fourier transform to Wannier space and save
	matrix LogImSigmaWannier = LogImSigmaWannierTilde * phase;
	LogImSigmaWannier.allReduce(MPIUtil::ReduceSum);
	if(mpiUtil->isHead())
		LogImSigmaWannier.dump("wannier.mlwfLogImSigma_ePh", spinWeight==2);
	
	finalizeSystem();
	return 0;
}
