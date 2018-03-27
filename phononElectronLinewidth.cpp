#include "WannierMC.h"
#include "InputMap.h"
#include <core/Units.h>
#include <core/LatticeUtils.h>
#include <electronic/TetrahedralDOS.h>
#include <algorithm>

template<typename T> T prod(const vector3<T>& v) { return v[0]*v[1]*v[2]; }

struct CollectEph
{	
	const WannierMC& wmc;
	const double prefacG;
	const double EconserveExpFac, EconservePrefac; //energy conserving (fermi-surface constraining) Gaussian exponential and pre-factor
	std::vector<diagMatrix> G; //Fermi-surface integrated e-ph coupling (for each phonon mode on DFT electronic k-mesh)
	std::vector<diagMatrix> omegaPh; //save phonon energies on full qmesh for final outputs
	std::vector<vector3<>> qmesh; //phonon q-mesh (full version i.e. unreduced)
	double wOffsetCur; //weight factor of current offset (due to symmetry reduction)
	
	CollectEph(const WannierMC& wmc, double EconserveWidth, const vector3<int>& NkMult)
	: wmc(wmc),
		prefacG(wmc.spinWeight * 2*M_PI/(prod(wmc.kfold)*prod(NkMult))),
		EconserveExpFac(-0.5/std::pow(EconserveWidth,2)),
		EconservePrefac(1./(sqrt(2.*M_PI)*EconserveWidth)),
		G(prod(wmc.kfold), diagMatrix(wmc.nModes)),
		omegaPh(prod(wmc.kfold), diagMatrix(wmc.nModes)),
		qmesh(prod(wmc.kfold))
	{
	}
	
	//Calculate Fermi-surface delta function and set hasContrib=true if any non-zero
	diagMatrix delta(diagMatrix E, bool& hasContrib)
	{	diagMatrix result(E.nRows());
		for(int b=0; b<E.nRows(); b++)
		{	double deltaExponent = EconserveExpFac * (E[b]*E[b]);
			if(deltaExponent < -15.) continue; //the exponential below will be negligible
			result[b] = EconservePrefac * exp(deltaExponent);
			hasContrib = true;
		}
		return result;
	}
	
	//---- Main Fermi-surface-integrated e-ph coupling kernel ----
	void process(const WannierMC::MatrixEph& mEph)
	{	const WannierMC::StateE& e1 = *(mEph.e1);
		const WannierMC::StateE& e2 = *(mEph.e2);
		const WannierMC::StatePh& ph = *(mEph.ph);
		//Svae phonon wave-vectors and frequencies for final outputs
		qmesh[ph.iqFine] = ph.q;
		omegaPh[ph.iqFine] = ph.omega;
		//Calculate Fermi-surface-constraining delta functions:
		bool hasContrib = false;
		diagMatrix delta1 = delta(e1.E, hasContrib); if(!hasContrib) return; //both k's must have a band at Ef
		diagMatrix delta2 = delta(e2.E, hasContrib); if(!hasContrib) return; //both k's must have a band at Ef
		//Loop over electronic state 1:
		for(int b1=0; b1<wmc.nBands; b1++) if(delta1[b1])
		{	//Loop over electronic state 2:
			for(int b2=0; b2<wmc.nBands; b2++) if(delta2[b2])
			{	double contrib = wOffsetCur * prefacG * delta1[b1] * delta2[b2];
				//Loop over phonon modes:
				for(int alpha=0; alpha<wmc.nModes; alpha++)
					G[ph.iqFine][alpha] += contrib * mEph.M[alpha](b2,b1).norm();
			}
		}
	}
	static void ePhProcess(const WannierMC::MatrixEph& mEph, void* params)
	{	((CollectEph*)params)->process(mEph);
	}
};

int main(int argc, char** argv)
{   InitParams ip =  WannierMC::initialize(argc, argv, "Electron-phonon scattering contribution to phonon linewidth.");

	//Read input file:
	InputMap inputMap(ip.inputFilename);
	const double EconserveWidth = inputMap.get("EconserveWidth") * eV;
	const int iSpin = inputMap.get("iSpin", 0); //spin channel (default 0)
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	vector3<int> NkMult;
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("EconserveWidth = %lg\n", EconserveWidth);
	logPrintf("iSpin = %d\n", iSpin);
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	
	//Initialize WannierMC:
	WannierMCParams wmcp;
	wmcp.iSpin = iSpin;
	wmcp.needSymmetries = true;
	wmcp.needCellWeights = true;
	wmcp.needPhonons = true;
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
	vector3<int> NkFine;
	vector3<> q0; //phonon q-mesh offset
	for(int iDir=0; iDir<3; iDir++)
	{	q0[iDir] = wmc.isTruncated[iDir] ? 0. : 0.5/wmc.kfold[iDir]; //offset from Gamma in periodic directions
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
		kMult.push_back(NkMultInv * ikMult);
	logPrintf("Effective interpolated k-mesh dimensions: ");
	NkFine.print(globalLog, " %d ");
	
	//Reduce under symmetries (simplified version of Symmetries::reduceKmesh from JDFTx):
	std::vector<vector3<>> k0; //array of electron k-mesh offsets
	std::vector<double> wk0; //corresponding weights
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
				k0.push_back(kfoldInv * kMult[iSrc]);
				wk0.push_back(w);
			}
	}
	//--- make available on all processes
	int nOffsets = k0.size();
	mpiWorld->bcast(nOffsets);
	k0.resize(nOffsets);
	wk0.resize(nOffsets);
	mpiWorld->bcast(&k0[0][0], 3*nOffsets);
	mpiWorld->bcast(wk0.data(), nOffsets);
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
	logPrintf("Collecting Gph: "); logFlush();
	CollectEph cEph(wmc, EconserveWidth, NkMult);
	for(int o=oStart; o<oStop; o++)
	{	//Process with selected offset:
		cEph.wOffsetCur = wk0[o];
		wmc.ePhLoop(q0+k0[o], k0[o], CollectEph::ePhProcess, &cEph);
		//Print progress:
		if((o-oStart+1)%oInterval==0) { logPrintf("%d%% ", int(round((o-oStart+1)*100./noMine))); logFlush(); }
	}
	logPrintf("done.\n"); logFlush();
	
	//Collect results from all processes:
	for(diagMatrix& g: cEph.G) g.allReduce(MPIUtil::ReduceSum);
	for(diagMatrix& o: cEph.omegaPh) o.allReduce(MPIUtil::ReduceMax);
	mpiWorld->allReduce(&cEph.qmesh[0][0], 3*cEph.qmesh.size(), MPIUtil::ReduceMax);
	
	//Symmetrize:
	PeriodicLookup<vector3<>> plook(cEph.qmesh, GGT);
	std::vector<bool> kDone(cEph.qmesh.size(), false);
	std::vector<int> iReduced;
	std::vector<vector3<>> qReduced;
	std::vector<double> qWeight;
	for(size_t i0=0; i0<cEph.qmesh.size(); i0++)
		if(!kDone[i0])
		{	//Find orbit of this k-points under symmetries:
			std::vector<int> iEquiv;
			diagMatrix Gmean(wmc.nModes);
			for(int invert: invertList)
				for(const SpaceGroupOp& op: wmc.sym)
				{	size_t i = plook.find(invert * cEph.qmesh[i0] * op.rot);
					if(i!=string::npos && (!kDone[i]))
					{	kDone[i] = true; //i will be covered in i0's orbit
						iEquiv.push_back(i);
						Gmean += cEph.G[i];
					}
				}
			//Symmetrize within orbit:
			Gmean *= (1./iEquiv.size());
			for(int i: iEquiv)
				cEph.G[i] = Gmean;
			iReduced.push_back(i0);
			qReduced.push_back(cEph.qmesh[i0]);
			qWeight.push_back(iEquiv.size()*(1./cEph.qmesh.size()));
		}
	logPrintf("Symmetrized Gph for %lu k-points in mesh in %lu orbits.\n", cEph.qmesh.size(), iReduced.size());
	
	//Output linewidths and energies in text file:
	if(mpiWorld->isHead())
	{	string fname = "Gph" + wmc.spinSuffix + ".dat";
		logPrintf("Dumping '%s' ... ", fname.c_str()); fflush(globalLog);
		FILE* fp = fopen(fname.c_str(), "w");
		for(int i: iReduced)
			for(int b=0; b<wmc.nModes; b++)
				fprintf(fp, "%+16.12lf %16.12lf\n", cEph.omegaPh[i][b], cEph.G[i][b]);
		fclose(fp);
		logPrintf("done.\n");
		//q-mesh and weights:
		fname = "Gph" + wmc.spinSuffix + ".qList";
		logPrintf("Dumping '%s' ... ", fname.c_str()); fflush(globalLog);
		fp = fopen(fname.c_str(), "w");
		for(size_t i=0; i<qReduced.size(); i++)
			fprintf(fp, "%12.10lf %12.10lf %12.10lf  %14.12lf\n",
				qReduced[i][0], qReduced[i][1], qReduced[i][2], qWeight[i]);
		fclose(fp);
		logPrintf("done.\n");
	}
	
	wmc.free();
	WannierMC::finalize();
	return 0;
}
