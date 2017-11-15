#include <core/Util.h>
#include <core/matrix.h>
#include <electronic/ColumnBundle.h>
#include <electronic/Everything.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/LatticeUtils.h>
#include <commands/parser.h>
#include "BandStruct.h"
#include "InputMap.h"
#include <core/Units.h>

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
		int qStart, qStop; TaskDivision(kArr.size(), mpiWorld).myRange(qStart, qStop);
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
		matrix ImSigmaWannier = ceArr.size() ? ImSigmaRS * phase : zeroes(bs.nBands*bs.nBands, nCells);
		ImSigmaWannier.allReduce(MPIUtil::ReduceSum);
		if(mpiWorld->isHead())
			ImSigmaWannier.dump(fname.c_str(), bs.spinWeight==2);
	}
};

inline bool eigsEqual(const diagMatrix& E1, const diagMatrix& E2, double Emin, double Emax, double tol)
{	if(E1.nRows() != E2.nRows()) return false;
	for(int b=0; b<E1.nRows(); b++)
	{	if(E1[b]>=Emin && E1[b]<=Emax && fabs(E1[b]-E2[b])>tol)
			return false;
	}
	return true;
}

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Electron-phonon scattering contribution to electron linewidth", inputFilename, dryRun, printDefaults);

	//Read input file:
	InputMap inputMap(inputFilename);
	const double T = inputMap.get("T") * Kelvin;
	const double EconserveWidth = inputMap.get("EconserveWidth") * eV;
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	vector3<int> NkMult;
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	vector3<> k0; //optional input to get linewidths at single k-point
	k0[0] = inputMap.get("k0x", INFINITY);
	k0[1] = inputMap.get("k0y", INFINITY);
	k0[2] = inputMap.get("k0z", INFINITY);
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("T = %lg\n", T);
	logPrintf("EconserveWidth = %lg\n", EconserveWidth);
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	if(std::isfinite(k0.length_squared()))
	{	logPrintf("k0 = ");
		k0.print(globalLog, " %lf ");
	}
	//Initialize Wannier bandstructure:
	const int bunchSize = 32;
	BandStruct bs("Wannier/totalE", "Wannier/wannier", true);
	bs.setCacheSize(2*bunchSize);
	
	vector3<int> NkIn = bs.kfold, NkOut;
	for(int j=0; j<3; j++)
		NkOut[j] = NkIn[j] * (bs.isTruncated[j] ? 1 : NkMult[j]); //multiply k-points in periodic directions
	logPrintf("NkFine = "); NkOut.print(globalLog, " %d ");
	vector3<> kInOff; //extra offset for k-point meshes (none for mesh mode)
	if(std::isfinite(k0.length_squared()))
	{	NkIn = vector3<int>(1,1,1);
		kInOff = k0; //offset k-meshes by supplied special point
	}
	vector3<int> strideIn(NkIn[1]*NkIn[2], NkIn[2], 1);
	int prodNkIn = NkIn[0]*NkIn[1]*NkIn[2];
	
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Calculate lifetimes for states in input k-point mesh:
	int nBands = bs.nBands;
	std::vector<vector3<>> kInArr(prodNkIn), kInReduced;
	std::vector<int> iReduced(prodNkIn, -1);
	{	vector3<> ikIn;
		std::vector<diagMatrix> Earr(prodNkIn);
		for(ikIn[0]=0; ikIn[0]<NkIn[0]; ikIn[0]++)
		for(ikIn[1]=0; ikIn[1]<NkIn[1]; ikIn[1]++)
		for(ikIn[2]=0; ikIn[2]<NkIn[2]; ikIn[2]++)
		{	int iIn = dot(ikIn,strideIn);
			vector3<>& kIn = kInArr[iIn];
			for(int j=0; j<3; j++)
				kIn[j] = kInOff[j] + ikIn[j] * (1./NkIn[j]); //use Gamma-centered mesh for in
			Earr[iIn] = bs.getStates(kIn);
		}
		//Reduce k-points based on degeneracies:
		for(int i1=0; i1<prodNkIn; i1++)
			if(iReduced[i1]<0)
			{	iReduced[i1] = kInReduced.size();
				for(int i2=i1+1; i2<prodNkIn; i2++)
					if(eigsEqual(Earr[i1], Earr[i2], bs.eMinMain, bs.eMaxMain, 1e-5))
						iReduced[i2] = kInReduced.size();
				kInReduced.push_back(kInArr[i1]);
			}
		if(prodNkIn>1) logPrintf("Symmetry degeneracies reduced %d input k-points to %lu\n", prodNkIn, kInReduced.size());
	}
	std::vector<vector3<>> kOutArr(NkOut[0] * NkOut[1] * NkOut[2]);
	vector3<int> strideOut(NkOut[1]*NkOut[2], NkOut[2], 1);
	{	vector3<> ikOut;
		for(ikOut[0]=0; ikOut[0]<NkOut[0]; ikOut[0]++)
		for(ikOut[1]=0; ikOut[1]<NkOut[1]; ikOut[1]++)
		for(ikOut[2]=0; ikOut[2]<NkOut[2]; ikOut[2]++)
		{	vector3<>& kOut = kOutArr[dot(ikOut,strideOut)];
			for(int j=0; j<3; j++)
				kOut[j] = kInOff[j] + (ikOut[j] + (bs.isTruncated[j] ? 0.0 : 0.5)) / NkOut[j]; //use Gamma-offset mesh for out (offset along periodic directions alone)
		}
	}
	double prefacImSigma = 0.5 * 2*M_PI/(kOutArr.size()); //Note factor of 0.5 between decay rate and ImSigma due to squaring of wavefunctions to probability
	double EconserveExpFac = -0.5/std::pow(EconserveWidth,2), EconservePrefac = 1./(sqrt(2.*M_PI)*EconserveWidth); //energy conserving Lorentzian parameters
	//Loop over blocks of one dimension of second k-point
	int ikOutStart = (kOutArr.size() * mpiWorld->iProcess()) / mpiWorld->nProcesses();
	int ikOutStop = (kOutArr.size() * (mpiWorld->iProcess()+1)) / mpiWorld->nProcesses();
	int nkOutBlocks = ceildiv(ikOutStop-ikOutStart, bunchSize);
	int nkOutInterval = std::max(1, int(round(nkOutBlocks/50.))); //interval for reporting progress
	std::vector<diagMatrix> ImSigmaReduced(kInReduced.size(), diagMatrix(nBands, 0.)); //imaginary part of self-energy on reduced mesh
	std::vector<diagMatrix> ImSigmaReducedP(kInReduced.size(), diagMatrix(nBands, 0.)); //momentum-relaxation version of above
	vector3<int> ikOut;
	for(int ikOut=ikOutStart; ikOut<ikOutStop; ikOut+=bunchSize)
	{	//Print progress:
		int ikOutBlocks = (ikOut-ikOutStart)/bunchSize+1;
		if(ikOutBlocks % nkOutInterval == 0)
		{	logPrintf("Working on ikOutBlock = %d of %d\n", ikOutBlocks, nkOutBlocks);
			logFlush();
		}
		//Initialize k2 array:
		int bunchSizeCur = std::min(bunchSize, ikOutStop-ikOut);
		std::vector<vector3<>> kOutCur(kOutArr.begin()+ikOut, kOutArr.begin()+ikOut+bunchSizeCur);
		std::vector< std::vector<matrix> > gePhArr(bunchSizeCur);
		//Loop over first k-point (irreducible wedge):
		for(size_t q=0; q<kInReduced.size(); q++)
		{	const vector3<>& kIn = kInReduced[q];
			const diagMatrix& Ein = bs.getStates(kIn);
			std::vector<vector3<>> vIn = bs.getVelocity(kIn);
			bs.setPhononMatElemArray(kIn, kOutCur, gePhArr.data());
			for(int ik2=0; ik2<bunchSizeCur; ik2++)
			{	const vector3<>& kOut = kOutCur[ik2];
				diagMatrix Eout = bs.getStates(kOut);
				std::vector<vector3<>> vOut = bs.getVelocity(kOut);
				diagMatrix omegaPh = bs.getPhononModes(kIn - kOut);
				const std::vector<matrix>& gePh = gePhArr[ik2];
				
				for(int bIn=0; bIn<Ein.nRows(); bIn++)
				{	for(int bOut=0; bOut<Eout.nRows(); bOut++)
					{	double fOut = bs.nValence
							? (bOut<bs.nValence ? 1. : 0.) //insulator/semiconductor
							: 1./(exp(Eout[bOut]/T)+1); //metal (energies referenced to mu)
						double cosThetaScatter = dot(vIn[bIn], vOut[bOut])
							/ sqrt(std::max(1e-16, vIn[bIn].length_squared() * vOut[bOut].length_squared()));
						for(int alpha=0; alpha<omegaPh.nRows(); alpha ++)
						{	double nPh = 1./(exp(omegaPh[alpha]/T) - 1.);
							for(int ae=-1; ae<=+1; ae+=2)
							{	double EconserveExponent = EconserveExpFac * std::pow((Eout[bOut]-Ein[bIn] - ae*omegaPh[alpha]),2);
								if(EconserveExponent < -15.) continue;
								double delta = EconservePrefac * exp(EconserveExponent);
								double occFactors = (nPh+0.5 - ae*(0.5-fOut));
								double ImSigmaContrib = prefacImSigma * occFactors * delta * gePh[alpha](bOut,bIn).norm();
								ImSigmaReduced[q][bIn] += ImSigmaContrib;
								ImSigmaReducedP[q][bIn] += ImSigmaContrib * (1.-cosThetaScatter);
							}
						}
					}
				}
			}
		}
	}
	
	//Collect results and map to full mesh:
	for(size_t q=0; q<kInReduced.size(); q++)
	{	ImSigmaReduced[q].allReduce(MPIUtil::ReduceSum);
		ImSigmaReducedP[q].allReduce(MPIUtil::ReduceSum);
	}
	std::vector<diagMatrix> ImSigma(prodNkIn), ImSigmaP(prodNkIn); //imaginary part of self-energy (and momentum-relaxation version) on full input mesh
	for(int q=0; q<prodNkIn; q++)
	{	ImSigma[q] = ImSigmaReduced[iReduced[q]];
		ImSigmaP[q] = ImSigmaReducedP[iReduced[q]];
	}
	
	if(mpiWorld->isHead())
	{	if(prodNkIn==1) logPrintf("\n#E ImSigma_ePh ImSigmaP_ePh\n");
		FILE* fp = (prodNkIn==1) ? globalLog : fopen("ImSigma_ePh.dat", "w"); //for special k-point mode, only report in log file
		for(size_t q=0; q<kInReduced.size(); q++)
		{	diagMatrix E = bs.getStates(kInReduced[q]);
			for(int b=0; b<nBands; b++)
				fprintf(fp, "%+19.12le %19.12le %19.12le\n", E[b], ImSigmaReduced[q][b], ImSigmaReducedP[q][b]);
		}
		if(prodNkIn>1) fclose(fp);
	}
	if(prodNkIn==1) { logPrintf("\n"); finalizeSystem(); return 0; } //Skip wannierization for special k-point mode
	
	//Wannierized output:
	Wannierizer(bs, ImSigma, kInArr).save("Wannier/wannier.mlwfImSigma_ePh");
	Wannierizer(bs, ImSigmaP, kInArr).save("Wannier/wannier.mlwfImSigmaP_ePh");
	
	finalizeSystem();
	return 0;
}
