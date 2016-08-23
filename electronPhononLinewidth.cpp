#include <core/Util.h>
#include <electronic/matrix.h>
#include <electronic/ColumnBundle.h>
#include <electronic/Everything.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/LatticeUtils.h>
#include <commands/parser.h>
#include "BandStruct.h"
#include "InputMap.h"
#include "Units.h"

class Wannierizer
{	const BandStruct& bs; const std::vector<diagMatrix>& ImSigma; const std::vector<vector3<>>& kArr;
public:
	Wannierizer(const BandStruct& bs, const std::vector<diagMatrix>& ImSigma, const std::vector<vector3<>>& kArr)
	: bs(bs), ImSigma(ImSigma), kArr(kArr) { }
	
	void save(string fname) const
	{	//Calculate cell weights:
		int nCells = bs.cellMap.size();
		std::vector<std::vector<int>> cellSets; cellSets.reserve(nCells); //cells grouped by translatinal equivalence
		std::vector<vector3<>> cellSupArr; cellSupArr.reserve(nCells); //unique cells in supercell coordinates
		PeriodicLookup<vector3<>> plook(cellSupArr, (~bs.R)*bs.R, nCells);
		for(int iCell=0; iCell<nCells; iCell++)
		{	vector3<> cellSup;
			for(int j=0; j<3; j++)
				cellSup[j] = bs.cellMap[iCell][j] * (1./bs.kfold[j]);
			size_t setIndex = plook.find(cellSup);
			if(setIndex == string::npos) //not yet found, create new set:
			{	plook.addPoint(cellSets.size(), cellSup);
				cellSupArr.push_back(cellSup);
				cellSets.push_back(std::vector<int>(1,iCell));
			}
			else cellSets[setIndex].push_back(iCell);
		}
		assert(int(cellSets.size()) == bs.kfold[0]*bs.kfold[1]*bs.kfold[2]);
		diagMatrix cellWeights(nCells);
		for(const std::vector<int>& cellSet: cellSets)
			for(int iCell: cellSet)
				cellWeights[iCell] = 1./(cellSet.size()*cellSets.size());
		//Transform from Bloch to Wannier:
		int qStart, qStop; TaskDivision(kArr.size(), mpiUtil).myRange(qStart, qStop);
		std::vector<vector3<>> kArrMine(kArr.begin()+qStart, kArr.begin()+qStop);
		auto ceArr = bs.getElectronCache(kArrMine);
		matrix ImSigmaRS(bs.nBands*bs.nBands, ceArr.size());
		matrix phase(ceArr.size(), nCells);
		for(int q=qStart; q<qStop; q++)
		{	int dq = q - qStart;
			std::shared_ptr<const BandStruct::CacheEntry> ce = ceArr[dq];
			diagMatrix logImSigma_q(bs.nBands);
			for(int b=0; b<bs.nBands; b++)
				logImSigma_q[b] = log(ImSigma[q][b]);
			matrix ImSigma_q = ce->evecs * logImSigma_q * dagger(ce->evecs);
			ImSigma_q.reshape(bs.nBands*bs.nBands, 1);
			ImSigmaRS.set(0,ImSigmaRS.nRows(), dq,dq+1, ImSigma_q);
			phase.set(dq,dq+1, 0,nCells, dagger(ce->phase) * cellWeights);
		}
		matrix ImSigmaWannier = ImSigmaRS * phase;
		ImSigmaWannier.allReduce(MPIUtil::ReduceSum);
		if(mpiUtil->isHead())
			ImSigmaWannier.dump(fname.c_str(), bs.spinWeight==2);
	}
};

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Electron-phonon scattering contribution to electron linewidth", inputFilename, dryRun, printDefaults);

	//Read input file:
	InputMap inputMap(inputFilename);
	const double T = inputMap.get("T") * Kelvin;
	const int NkMult = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("T = %lg\n", T);
	logPrintf("NkMult = %d\n", NkMult);

	//Initialize Wannier bandstructure:
	const int bunchSize = 32;
	BandStruct bs("Wannier/totalE", "Wannier/wannier", true);
	bs.setCacheSize(2*bunchSize);
	
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	vector3<int> NkIn = bs.kfold;
	vector3<int> NkOut = NkIn * NkMult;
	vector3<int> strideIn(NkIn[1]*NkIn[2], NkIn[2], 1);
	int prodNkIn = NkIn[0]*NkIn[1]*NkIn[2];
	bs.setCacheSize(NkOut[2]*4);
	
	//Calculate lifetimes for states in input k-point mesh:
	int nBands = bs.nBands;
	std::vector<diagMatrix> ImSigma(prodNkIn, diagMatrix(nBands, 0.)); //imaginary part of self-energy on NkIn mesh
	std::vector<vector3<>> kInArr(prodNkIn);
	{	vector3<> ikIn;
		for(ikIn[0]=0; ikIn[0]<NkIn[0]; ikIn[0]++)
		for(ikIn[1]=0; ikIn[1]<NkIn[1]; ikIn[1]++)
		for(ikIn[2]=0; ikIn[2]<NkIn[2]; ikIn[2]++)
		{	vector3<>& kIn = kInArr[dot(ikIn,strideIn)];
			for(int j=0; j<3; j++)
				kIn[j] = ikIn[j] * (1./NkIn[j]); //use Gamma-centered mesh for in
		}
	}
	double prefacImSigma = 0.5 * 2*M_PI/(NkOut[0]*NkOut[1]*NkOut[2]); //Note factor of 0.5 between decay rate and ImSigma due to squaring of wavefunctions to probability
	double EconserveScaleFac = 1./T, EconservePrefac = 1./(M_PI*T); //energy conserving Lorentzian parameters
	std::vector< std::vector<matrix> > gePhArr(NkOut[2]);
	//Loop over blocks of one dimension of second k-point
	std::vector< vector3<> > kOutArr(NkOut[2]);
	int NkOut01 = NkOut[0] * NkOut[1];
	int ik01start = (NkOut01 * mpiUtil->iProcess()) / mpiUtil->nProcesses();
	int ik01stop = (NkOut01 * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nk01mine = ik01stop-ik01start;
	int nk01interval = std::max(1, int(round(nk01mine/50.))); //interval for reporting progress
	vector3<int> ikOut;
	for(int ik01=ik01start; ik01<ik01stop; ik01++)
	{	ikOut[0] = ik01 / NkOut[1];
		ikOut[1] = ik01 - ikOut[0] * NkOut[1];
		//Print progress:
		int nk01done = ik01-ik01start+1;
		if(nk01done % nk01interval == 0)
		{	logPrintf("Working on nk01 = %d of %d\n", nk01done, nk01mine);
			logFlush();
		}
		//Initialize k2 array:
		for(ikOut[2]=0; ikOut[2]<NkOut[2]; ikOut[2]++)
			for(int j=0; j<3; j++)
				kOutArr[ikOut[2]][j] = (ikOut[j] + 0.5) / NkOut[j]; //use Gamma-offset mesh for out
		//Loop over first k-point (irreducible wedge):
		for(int q=0; q<prodNkIn; q++)
		{	const vector3<>& kIn = kInArr[q];
			bs.setPhononMatElemArray(kIn, kOutArr, gePhArr.data());
			const diagMatrix& Ein = bs.getStates(kIn);
			for(int ikOut2=0; ikOut2<NkOut[2]; ikOut2++)
			{	const vector3<>& kOut = kOutArr[ikOut2];
				diagMatrix Eout = bs.getStates(kOut);
				diagMatrix omegaPh = bs.getPhononModes(kIn - kOut);
				const std::vector<matrix>& gePh = gePhArr[ikOut2];
				
				for(int bIn=0; bIn<Ein.nRows(); bIn++)
				{	for(int bOut=0; bOut<Eout.nRows(); bOut++)
					{	double fOut = bs.nValence
							? (bOut<bs.nValence ? 1. : 0.) //insulator/semiconductor
							: 1./(exp(Eout[bOut]/T)+1); //metal (energies referenced to mu)
						for(int alpha=0; alpha<omegaPh.nRows(); alpha ++)
						{	double nPh = 1./(exp(omegaPh[alpha]/T) - 1.);
							for(int ae=-1; ae<=+1; ae+=2)
							{	double delta = EconservePrefac / (1. + std::pow(EconserveScaleFac * (Eout[bOut]-Ein[bIn] - ae*omegaPh[alpha]),2));
								double occFactors = (nPh+0.5 - ae*(0.5-fOut));
								ImSigma[q][bIn] += prefacImSigma * occFactors * delta * gePh[alpha](bOut,bIn).norm();
							}
						}
					}
				}
			}
		}
	}
	FILE* fp = 0;
	if(mpiUtil->isHead()) fp = fopen("ImSigma_ePh.dat", "w");
	for(int q=0; q<prodNkIn; q++)
	{	ImSigma[q].allReduce(MPIUtil::ReduceSum);
		if(fp)
		{	diagMatrix E = bs.getStates(kInArr[q]);
			for(int b=0; b<nBands; b++)
				fprintf(fp, "%+19.12le %19.12le\n", E[b], ImSigma[q][b]);
		}
	}
	if(fp) fclose(fp);
	
	//Wannierized output:
	Wannierizer(bs, ImSigma, kInArr).save("Wannier/wannier.mlwfImSigma_ePh");
	
	finalizeSystem();
	return 0;
}
