#include "WannierMC.h"
#include "InputMap.h"
#include <core/Units.h>
#include <core/LatticeUtils.h>
#include <algorithm>

template<typename T> T prod(const vector3<T>& v) { return v[0]*v[1]*v[2]; }

struct CollectEph
{	
	const WannierMC& wmc;
	const double T;
	const double prefacImSigma;
	const double EconserveExpFac, EconservePrefac; //energy conserving Gaussian exponential and pre-factor
	std::vector<diagMatrix> ImSigma, ImSigmaP; //e-ph linewidth without and with momentum direction factors
	std::vector<diagMatrix> E; //save electron energies on DFT mesh for later
	double wOffsetCur; //weight factor of current offset (due to symmetry reduction)
	
	CollectEph(const WannierMC& wmc, double T, double EconserveWidth, const vector3<int>& NkMult)
	: wmc(wmc), T(T),
		prefacImSigma(0.5 * 2*M_PI/(prod(wmc.kfold)*prod(NkMult))), //Factor of 0.5 in ImSigma because of psi^2 -> n
		EconserveExpFac(-0.5/std::pow(EconserveWidth,2)),
		EconservePrefac(1./(sqrt(2.*M_PI)*EconserveWidth)),
		ImSigma(prod(wmc.kfold), diagMatrix(wmc.nBands)),
		ImSigmaP(prod(wmc.kfold), diagMatrix(wmc.nBands)),
		E(prod(wmc.kfold))
	{
	}
	
	//---- Max group velocity calculation for NkMult hint ----
	vector3<> dEdkMax; //w.r.t k in reciprocal lattice coordinates
	void vgMaxCollect(const WannierMC::StateE& state)
	{	matrix3<> G = (2*M_PI)*inv(wmc.R);
		for(int b=0; b<wmc.nBands; b++)
			if(fabs(state.E[b]) < 5*eV) //only consider states in medium proximity to Fermi level
			{	vector3<> dEdk = G * state.vVec[b]; //w.r.t k in reciprocal lattice coordinates
				for(int iDir=0; iDir<3; iDir++)
					dEdkMax[iDir] = std::max(dEdkMax[iDir], fabs(dEdk[iDir]));
			}
	}
	static void vgMaxProcess(const WannierMC::StateE& state, void* params)
	{	((CollectEph*)params)->vgMaxCollect(state);
	}
	
	//---- Main e-ph scattering linewidth kernel ----
	void process(const WannierMC::MatrixEph& mEph)
	{	const WannierMC::StateE& e1 = *(mEph.e1);
		const WannierMC::StateE& e2 = *(mEph.e2);
		const WannierMC::StatePh& ph = *(mEph.ph);
		const int nBands = e1.E.nRows();
		const int nModes = ph.omega.nRows();
		//Loop over electronic state 1:
		for(int b1=0; b1<nBands; b1++)
		{	const double& E1 = e1.E[b1];
			const vector3<>& v1 = e1.vVec[b1];
			//Loop over electronic state 2:
			for(int b2=0; b2<nBands; b2++)
			{	const double& E2 = e2.E[b2];
				const vector3<>& v2 = e2.vVec[b2];
				double f2 = wmc.nValence
					? (b2<wmc.nValence ? 1. : 0.) //insulator/semiconductor
					: 1./(exp(E2/T)+1); //metal (energies referenced to mu)
				double cosThetaScatter = dot(v1, v2) / sqrt(std::max(1e-16, v1.length_squared() * v2.length_squared()));
				//Loop over phonon modes:
				for(int alpha=0; alpha<nModes; alpha++)
				{	const double& omegaPh = ph.omega[alpha];
					double nPh = 1./(exp(omegaPh/T) - 1.);
					for(int ae=-1; ae<=+1; ae+=2)
					{	double EconserveExponent = EconserveExpFac * std::pow((E2-E1 - ae*omegaPh),2);
						if(EconserveExponent < -15.) continue; //the exponential below will be negligible
						double delta = EconservePrefac * exp(EconserveExponent);
						double occFactors = (nPh+0.5 - ae*(0.5-f2));
						double ImSigmaContrib = wOffsetCur * prefacImSigma * occFactors * delta * mEph.M[alpha](b2,b1).norm();
						ImSigma[e1.ik][b1] += ImSigmaContrib;
						ImSigmaP[e1.ik][b1] += ImSigmaContrib * (1.-cosThetaScatter);
					}
				}
			}
		}
		//Save E1 for final output:
		if(!E[e1.ik].size()) E[e1.ik] = e1.E;
	}
	static void ePhProcess(const WannierMC::MatrixEph& mEph, void* params)
	{	((CollectEph*)params)->process(mEph);
	}
	
	//---- Wannierization ----
	int cStart, cStop; //range of cells handled here
	matrix mlwfImSigma, mlwfImSigmaP, phase;
	
	void wannierize(const WannierMC::StateE& state)
	{	//Convert to log for the interpolation:
		diagMatrix logImSigma(ImSigma[state.ik]); for(double& x: logImSigma) x = log(x);
		diagMatrix logImSigmaP(ImSigmaP[state.ik]); for(double& x: logImSigmaP) x = log(x);
		//Switch to Wannier basis:
		matrix logImSigmaW = state.U * logImSigma * dagger(state.U);
		matrix logImSigmaPW = state.U * logImSigmaP * dagger(state.U);
		//Save as a column in a matrix containing all k:
		int iCol = state.ik - wmc.Hw->ikStart;
		int colLength = wmc.nBands * wmc.nBands;
		eblas_copy(mlwfImSigma.data()+iCol*colLength, logImSigmaW.data(), colLength);
		eblas_copy(mlwfImSigmaP.data()+iCol*colLength, logImSigmaPW.data(), colLength);
		//Calculate corresponding phases for Fourier transform:
		for(int c=cStart; c<cStop; c++)
			phase.set(iCol, c-cStart, cis(-2*M_PI*dot(state.k, wmc.cellMap[c])));
	}
	static void eProcess(const WannierMC::StateE& state, void* params)
	{	((CollectEph*)params)->wannierize(state);
	}
	
	void dumpWannierized(matrix& m, string fname) const
	{	m = m * phase; //Fourier transform
		mpiGroup->allReduce(m.data(), m.nData(), MPIUtil::ReduceSum); //Collect results within groups
		if(mpiGroup->isHead())
		{	//expand to all cells version (with zeroes where unavailable currently)
			matrix mEx = zeroes(m.nRows(), wmc.cellMap.size());
			mEx.set(0,m.nRows(), cStart,cStop, m);
			//Collect results between group heads
			mpiGroupHead->allReduce(mEx.data(), mEx.nData(), MPIUtil::ReduceSum);
			//Output from world head:
			if(mpiGroupHead->isHead())
			{	eblas_zmul(wmc.cellWeights.nData(), wmc.cellWeights.data(),1, mEx.data(),1); //Apply weight factors
				mEx.dump(fname.c_str(), wmc.spinWeight==2); //Output
			}
		}
	}
};


//Report ImSigma for N states closes to Fermi level
class FermiImSigmaReport
{	const size_t N;
	std::multimap<double, std::pair<double,double>> cache;
public:
	FermiImSigmaReport(size_t N) : N(N) {}
	
	void addState(double E, double ImSigma)
	{	auto entry = std::make_pair(fabs(E), std::make_pair(E, ImSigma));
		if(cache.size() < N)
			cache.insert(entry);
		else
		{	if(entry.first < cache.rbegin()->first)
			{	//current one better than worst entry in cache
				cache.erase(--cache.end()); //remove worst entry
				cache.insert(entry); //add currnet one
			}
		}
	}
	
	void report() const
	{	for(auto entry: cache)
			logPrintf("\t%+9.6lf %14.12lf\n", entry.second.first, entry.second.second);
	}
};


int main(int argc, char** argv)
{   InitParams ip =  WannierMC::initialize(argc, argv, "Electron-phonon scattering contribution to electron linewidth.");

	//Read input file:
	InputMap inputMap(ip.inputFilename);
	const double T = inputMap.get("T") * Kelvin;
	const double EconserveWidth = inputMap.get("EconserveWidth") * eV;
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	vector3<int> NkMult;
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("T = %lg\n", T);
	logPrintf("EconserveWidth = %lg\n", EconserveWidth);
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	
	//Initialize WannierMC:
	WannierMCParams wmcp;
	wmcp.needSymmetries = true;
	wmcp.needCellWeights = true;
	wmcp.needPhonons = true;
	wmcp.needVelocity = true;
	WannierMC wmc(wmcp);
	
	//Check NkMult compatibility with symmetries:
	for(const SpaceGroupOp& op: wmc.sym)
	{	//Similar to Symmetries::checkFFTbox in JDFTx
		matrix3<int> mMesh = Diag(NkMult) * op.rot;
		for(int i=0; i<3; i++)
			for(int j=0; j<3; j++)
				if(mMesh(i,j) % NkMult[j] == 0)
					mMesh(i,j) /= NkMult[j];
				else
				{	logPrintf("NkMult not commensurate with symmetry matrix:\n");
					op.rot.print(globalLog, " %2d ");
					op.a.print(globalLog, " %lg ");
					die("NkMult not commensurate with symmetries.\n");
				}
	}
	
	//Construct NkMult mesh:
	std::vector<vector3<>> kMult;
	vector3<> kOffset;
	vector3<int> NkFine;
	for(int iDir=0; iDir<3; iDir++)
	{	kOffset[iDir] = wmc.isTruncated[iDir] ? 0. : 0.5; //offset from Gamma in periodic directions
		if(wmc.isTruncated[iDir] && NkMult[iDir]!=1)
		{	logPrintf("Setting NkMult = 1 along truncated direction %d.\n", iDir+1);
			NkMult[iDir] = 1; //no multiplication in truncated directions
		}
		NkFine[iDir] = wmc.kfold[iDir] * NkMult[iDir];
	}
	matrix3<> NkMultInv = inv(Diag(vector3<>(NkMult)));
	vector3<int> ikMult;
	for(ikMult[0]=0; ikMult[0]<NkMult[0]; ikMult[0]++)
	for(ikMult[1]=0; ikMult[1]<NkMult[1]; ikMult[1]++)
	for(ikMult[2]=0; ikMult[2]<NkMult[2]; ikMult[2]++)
		kMult.push_back(NkMultInv * (ikMult + kOffset));
	logPrintf("Effective interpolated k-mesh dimensions: ");
	NkFine.print(globalLog, " %d ");
	
	//Initialize collect helper class:
	CollectEph cEph(wmc, T, EconserveWidth, NkMult);
	//Estimate minimum NkMult:
	wmc.eLoop(vector3<>(), CollectEph::vgMaxProcess, &cEph);
	mpiGroup->allReduce(&cEph.dEdkMax[0], 3, MPIUtil::ReduceMax);
	vector3<int> NkMultMin;
	for(int iDir=0; iDir<3; iDir++)
	{	double dkMax = EconserveWidth / cEph.dEdkMax[iDir]; //max dk in recip coords such that dE within EconserveWidth
		NkMultMin[iDir] = ceil(1./(wmc.kfold[iDir]*dkMax)); //multiplication factor that will keep dk of mesh smaller than that
	}
	logPrintf("\nFor dE ~ EconserveWidth, NkMult ~ ");
	NkMultMin.print(globalLog, " %d ");
	
	//Reduce under symmetries (simplified version of Symmetries::reduceKmesh from JDFTx):
	std::vector<vector3<>> k02; //array of k2-mesh offsets
	std::vector<double> wk02; //corresponding weights
	//--- Compile list of inversions to check:
	std::vector<int> invertList;
	invertList.push_back(+1);
	invertList.push_back(-1);
	for(const SpaceGroupOp& op: wmc.sym)
		if(op.rot==matrix3<int>(-1,-1,-1))
		{	invertList.resize(1); //inversion explicitly found in symmetry list, so remove from invertList
			break;
		}
	matrix3<> G = 2*M_PI*inv(wmc.R), GGT = G*(~G);
	matrix3<> kfoldInv = inv(Diag(vector3<>(wmc.kfold)));
	if(mpiWorld->isHead())
	{	//compile kpoint map:
		PeriodicLookup<vector3<>> plook(kMult, GGT);
		std::vector<bool> kDone(kMult.size(), false);
		for(size_t iSrc=0; iSrc<kMult.size(); iSrc++)
			if(!kDone[iSrc])
			{	double w = 0.; //weight of current point
				for(int invert: invertList)
					for(const SpaceGroupOp& op: wmc.sym)
					{	size_t iDest = plook.find(invert * kMult[iSrc] * op.rot);
						if(iDest!=string::npos && (!kDone[iDest]))
						{	kDone[iDest] = true; //iDest in iSrc's orbit
							w += 1.; //increase weight of iSrc
						}
					}
				//add corresponding offset:
				k02.push_back(kfoldInv * kMult[iSrc]);
				wk02.push_back(w);
			}
	}
	//--- make available on all processes
	int nOffsets = k02.size();
	mpiWorld->bcast(nOffsets);
	k02.resize(nOffsets);
	wk02.resize(nOffsets);
	mpiWorld->bcast(&k02[0][0], 3*nOffsets);
	mpiWorld->bcast(wk02.data(), nOffsets);
	logPrintf("\n%lu offsets in NkMult mesh reduced to %d under symmetries.\n", kMult.size(), nOffsets);
	
	logPrintf("\n");
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		wmc.free();
		WannierMC::finalize();
		return 0;
	}
	
	//Initialize sampling parameters:
	int oStart=0, oStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsets, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	
	//Collect results for each offset
	logPrintf("Collecting ImSigma_ePh: "); logFlush();
	for(int o=oStart; o<oStop; o++)
	{	//Process with a random offset:
		cEph.wOffsetCur = wk02[o];
		wmc.ePhLoop(vector3<>(), k02[o], CollectEph::ePhProcess, &cEph);
		//Print progress:
		if((o-oStart+1)%oInterval==0) { logPrintf("%d%% ", int(round((o-oStart+1)*100./noMine))); logFlush(); }
	}
	logPrintf("done.\n"); logFlush();
	
	//Collect results from all processes:
	for(diagMatrix& d: cEph.ImSigma) d.allReduce(MPIUtil::ReduceSum);
	for(diagMatrix& d: cEph.ImSigmaP) d.allReduce(MPIUtil::ReduceSum);
	
	//Symmetrize:
	std::vector<vector3<>> kmesh; //DFT k-point mesh
	vector3<int> ikmesh;
	for(ikmesh[0]=0; ikmesh[0]<wmc.kfold[0]; ikmesh[0]++)
	for(ikmesh[1]=0; ikmesh[1]<wmc.kfold[1]; ikmesh[1]++)
	for(ikmesh[2]=0; ikmesh[2]<wmc.kfold[2]; ikmesh[2]++)
		kmesh.push_back(kfoldInv * ikmesh);
	PeriodicLookup<vector3<>> plook(kmesh, GGT);
	std::vector<bool> kDone(kmesh.size(), false);
	std::vector<int> iReduced;
	for(size_t i0=0; i0<kmesh.size(); i0++)
		if(!kDone[i0])
		{	//Find orbit of this k-points under symmetries:
			std::vector<int> iEquiv;
			diagMatrix ImSigmaMean(wmc.nBands), ImSigmaMeanP(wmc.nBands);
			for(int invert: invertList)
				for(const SpaceGroupOp& op: wmc.sym)
				{	size_t i = plook.find(invert * kmesh[i0] * op.rot);
					if(i!=string::npos && (!kDone[i]))
					{	kDone[i] = true; //i will be covered in i0's orbit
						iEquiv.push_back(i);
						ImSigmaMean += cEph.ImSigma[i];
						ImSigmaMeanP += cEph.ImSigmaP[i];
					}
				}
			//Symmetrize within orbit:
			ImSigmaMean *= (1./iEquiv.size());
			ImSigmaMeanP *= (1./iEquiv.size());
			for(int i: iEquiv)
			{	cEph.ImSigma[i] = ImSigmaMean;
				cEph.ImSigmaP[i] = ImSigmaMeanP;
			}
			iReduced.push_back(i0);
		}
	logPrintf("Symmetrized ImSigma for %lu k-points in mesh in %lu orbits.\n", kmesh.size(), iReduced.size());
	
	//Collect electronic energies on all processes:
	for(int i: iReduced)
	{	int root = cEph.E[i].size() ? mpiGroup->iProcess() : mpiGroup->nProcesses(); //my process ID or N, depending on whether I have E[i]
		mpiGroup->allReduce(root, MPIUtil::ReduceMin); //lowest process number which has E[i] available
		cEph.E[i].resize(wmc.nBands);
		mpiGroup->bcast(cEph.E[i].data(), wmc.nBands, root);
	}
	
	//Output linewidths and energies in text file:
	if(mpiWorld->isHead())
	{	FermiImSigmaReport fr(10);
		const char* fname = "ImSigma_ePh.dat";
		logPrintf("Dumping '%s' ... ", fname); fflush(globalLog);
		FILE* fp = fopen(fname, "w");
		for(int i: iReduced)
			for(int b=0; b<wmc.nBands; b++)
			{	fprintf(fp, "%+19.12le %19.12le %19.12le\n",
					cEph.E[i][b], cEph.ImSigma[i][b], cEph.ImSigmaP[i][b]);
				fr.addState(cEph.E[i][b], cEph.ImSigma[i][b]);
			}
		fclose(fp);
		logPrintf("done.\n");
		logPrintf("\nEnergy and ImSigma [Eh] for few states closest to Fermi level:\n");
		fr.report();
		logPrintf("HINT: check convergence of above numbers with NkMult.\n\n");
	}
	
	//Wannierize output:
	//--- divide output cells over MPI groups:
	cEph.cStart = cEph.cStop = 0;
	if(mpiGroup->isHead())
		TaskDivision(wmc.cellMap.size(), mpiGroupHead).myRange(cEph.cStart, cEph.cStop);
	mpiGroup->bcast(cEph.cStart);
	mpiGroup->bcast(cEph.cStop);
	int ncMine = std::max(1, cEph.cStop - cEph.cStart);
	int nkMine = std::max(1, wmc.Hw->nk);
	//--- Wannierize
	cEph.mlwfImSigma = zeroes(wmc.nBands*wmc.nBands, nkMine);
	cEph.mlwfImSigmaP = zeroes(wmc.nBands*wmc.nBands, nkMine);
	cEph.phase = zeroes(nkMine, ncMine);
	wmc.eLoop(vector3<>(), CollectEph::eProcess, &cEph);
	cEph.phase *= (1./kmesh.size()); //inverse transform normalizing factor
	cEph.dumpWannierized(cEph.mlwfImSigma, wmcp.wannierPrefix + ".mlwfImSigma_ePh");
	cEph.dumpWannierized(cEph.mlwfImSigmaP, wmcp.wannierPrefix + ".mlwfImSigmaP_ePh");
	
	wmc.free();
	WannierMC::finalize();
	return 0;
}
