#include "WannierMC.h"
#include "InputMap.h"
#include <core/Units.h>
#include <core/Random.h>

struct ResistivityCollect
{	std::vector<double> dmu; //doping levels
	double T; //temperature
	std::vector<double> g, tauInv; //density of states and scattering time
	std::vector<matrix3<>> Tcur, Gamma; //above with velocity matrices multiplied

	ResistivityCollect(const std::vector<double>& dmu, double T, double EconserveWidth) : dmu(dmu), T(T),
		g(dmu.size()), tauInv(dmu.size()), Tcur(dmu.size()), Gamma(dmu.size())
	{
	}
	
	void collect(const WannierMC::StateE& state)
	{	const int nBands = state.E.nRows();
		for(int b=0; b<nBands; b++)
		{	const double& E = state.E[b];
			const vector3<>& v = state.vVec[b];
			matrix3<> vdotv = outer(v, v);
			for(unsigned iMu=0; iMu<dmu.size(); iMu++)
			{	double dFdE = -1./(T*std::pow(2*cosh((E-dmu[iMu])/(2*T)),2));
				Tcur[iMu] += vdotv * (-dFdE);
				g[iMu] += (-dFdE);
				Gamma[iMu] += vdotv * ((-dFdE) * (2*state.ImSigmaP_ePh[b]));
				tauInv[iMu] += (-dFdE) * (2*state.ImSigma_ePh[b]);
			}
		}
	}
	static void eProcess(const WannierMC::StateE& state, void* params)
	{	((ResistivityCollect*)params)->collect(state);
	}
};

//Functions for printing with error estimates (implemented at bottom of file)
void reportResult(const std::vector<matrix3<>>& result, string resultName, double unit, string unitName);
void reportResult(const std::vector<double>& result, string resultName, double unit, string unitName);

//Eliminate direction slabDir from tensor (for 2D case normal to slabDir):
inline void slabConstrain(matrix3<>& M, int slabDir)
{	if(slabDir >= 0)
	{	for(int jDir=0; jDir<3; jDir++)
		{	M(jDir, slabDir) = 0.;
			M(slabDir, jDir) = 0.;
		}
		M(slabDir, slabDir) = 1.;
	}
}

//Trace directions other than slabDir:
inline double trace(const matrix3<>& M, int slabDir)
{	double result = 0;
	for(int jDir=0; jDir<3; jDir++)
		if(jDir != slabDir)
			result += M(jDir,jDir);
	return result;
}


int main(int argc, char** argv)
{	InitParams ip = WannierMC::initialize(argc, argv, "Monte Carlo estimate of resistivity");

	//Read input file:
	InputMap inputMap(ip.inputFilename);
	const int nOffsets = inputMap.get("nOffsets"); assert(nOffsets>0);
	const int nBlocks = inputMap.get("nBlocks"); assert(nBlocks>0);
	const double T = inputMap.get("T") * Kelvin;
	const double EconserveWidth = inputMap.get("EconserveWidth", T/eV) * eV; //energy conservation width (default to T)
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional shift in chemical potential from neutral value; start of range (default to 0)
	const double dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional shift in chemical potential from neutral value; end of range (default to 0)
	const int dmuCount = inputMap.get("dmuCount", 1); assert(dmuCount>0); //number of chemical potential shifts
	const int slabDir = inputMap.get("slabDir", -1); assert(slabDir<3); //0-based index of direction to eliminate; default -1 => don't eliminate any (keep 3D)
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("nBlocks = %d\n", nBlocks);
	logPrintf("T = %lg\n", T);
	logPrintf("EconserveWidth = %lg\n", EconserveWidth);
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("dmuCount = %d\n", dmuCount);
	logPrintf("slabDir = %d\n", slabDir);
	
	//Initialize WannierMC:
	WannierMCParams wmcp;
	wmcp.needVelocity = true;
	wmcp.needLinewidth_ePh = true;
	wmcp.needLinewidthP_ePh = true;
	WannierMC wmc(wmcp);
	
	//dmu array:
	std::vector<double> dmu(dmuCount, dmuMin); //set first value here
	for(int iMu=1; iMu<dmuCount; iMu++) //set remaining values (if any)
		dmu[iMu] = dmuMin + iMu*(dmuMax-dmuMin)/(dmuCount-1);
	
	//Handle dimensionality:
	double Omega = wmc.Omega;
	double rhoUnit = 1e-9*Ohm*meter;
	string rhoUnitName="nOhm-m";
	string rhoName = "Resistivity";
	if(slabDir>=0)
	{	Omega /= wmc.R.column(slabDir).length(); //convert to area excluding this dimension
		rhoUnit = Ohm;
		rhoUnitName = "Ohm";
		rhoName = "SheetResistance";
	}
	
	//Initialize sampling parameters:
	int nOffsetsPerBlock = ceildiv(nOffsets, nBlocks);
	size_t nKptsPerBlock = wmc.eCountPerOffset() * nOffsetsPerBlock;
	logPrintf("Effectively sampled nKpts: %lu\n", nKptsPerBlock * nBlocks);
	int oStart = 0, oStop = 0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsetsPerBlock, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets (per block) handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		wmc.free();
		WannierMC::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Compute resistivity:
	double prefacDOS = wmc.spinWeight*(1./nKptsPerBlock);
	#define DeclareArray2D(type, name) std::vector<std::vector<type>> name(dmuCount, std::vector<type>(nBlocks))
	DeclareArray2D(matrix3<>, Tarr); DeclareArray2D(matrix3<>, GammaArr); DeclareArray2D(matrix3<>, rhoArr);
	DeclareArray2D(double, rhoBarArr); DeclareArray2D(double, tauArr); DeclareArray2D(double, tauDrudeArr);
	DeclareArray2D(double, vFarr); DeclareArray2D(double, gArr); 
	#undef DeclareArray2D
	for(int block=0; block<nBlocks; block++)
	{	logPrintf("Working on block %d of %d: ", block+1, nBlocks); logFlush();
		ResistivityCollect rc(dmu, T, EconserveWidth);
		for(int o=0; o<noMine; o++)
		{	Random::seed(block*nOffsetsPerBlock+o+oStart); //to make results independent of MPI division
			//Process with a random offset:
			vector3<> k0 = wmc.randomVector(mpiGroup); //must be constant across group
			wmc.eLoop(k0, ResistivityCollect::eProcess, &rc);
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		for(int iMu=0; iMu<dmuCount; iMu++)
		{	//Accumulate between processes:
			mpiWorld->allReduce(&rc.Tcur[iMu](0,0), 3*3, MPIUtil::ReduceSum);
			mpiWorld->allReduce(&rc.Gamma[iMu](0,0), 3*3, MPIUtil::ReduceSum);
			mpiWorld->allReduce(rc.g[iMu], MPIUtil::ReduceSum);
			mpiWorld->allReduce(rc.tauInv[iMu], MPIUtil::ReduceSum);
			//Apply normalizing factors:
			rc.Tcur[iMu] *= prefacDOS;
			rc.g[iMu] *= prefacDOS;
			rc.Gamma[iMu] *= prefacDOS; rc.Gamma[iMu] = 0.5*(rc.Gamma[iMu] + (~rc.Gamma[iMu])); //symmetrize
			rc.tauInv[iMu] *= prefacDOS;
			slabConstrain(rc.Tcur[iMu], slabDir); //eliminate out-of-plane components if necessary
			slabConstrain(rc.Gamma[iMu], slabDir); //eliminate out-of-plane components if necessary
			//Store relevant quantities:
			Tarr[iMu][block] = rc.Tcur[iMu];
			GammaArr[iMu][block] = rc.Gamma[iMu];
			rhoArr[iMu][block] = Omega * (inv(rc.Tcur[iMu]) * rc.Gamma[iMu] * inv(rc.Tcur[iMu]));
			rhoBarArr[iMu][block] = trace(rhoArr[iMu][block], slabDir) / (slabDir>=0 ? 2. : 3.);
			tauArr[iMu][block] = rc.g[iMu] / rc.tauInv[iMu];
			tauDrudeArr[iMu][block] = trace(rc.Tcur[iMu], slabDir) / trace(rc.Gamma[iMu], slabDir);
			vFarr[iMu][block] = sqrt(trace(rc.Tcur[iMu], slabDir)/rc.g[iMu]);
			gArr[iMu][block] = rc.g[iMu];
			//Fix slab direction values physically:
			if(slabDir>=0.)
			{	Tarr[iMu][block](slabDir,slabDir) = 0.;
				GammaArr[iMu][block](slabDir,slabDir) = 0.;
				rhoArr[iMu][block](slabDir,slabDir) = INFINITY;
			}
		}
	}
	
	for(int iMu=0; iMu<dmuCount; iMu++)
	{	logPrintf("\nResults for dmu = %lg eV:\n", dmu[iMu]/eV);
		reportResult(Tarr[iMu], "T", 1, "");
		reportResult(GammaArr[iMu], "Gamma", 1, "");
		reportResult(rhoArr[iMu], rhoName, rhoUnit, rhoUnitName);
		reportResult(rhoBarArr[iMu], rhoName, rhoUnit, rhoUnitName);
		reportResult(tauDrudeArr[iMu], "tauDrude", fs, "fs");
		reportResult(tauArr[iMu], "tau", fs, "fs");
		reportResult(vFarr[iMu], "vF", 1, "");
		reportResult(gArr[iMu], "g(eF)", 1, "");
	}
	
	wmc.free();
	WannierMC::finalize();
}

//Report a tensor with error estimates
void reportResult(const std::vector<matrix3<>>& result, string resultName, double unit, string unitName)
{	matrix3<> resultMean, resultStd;
	for(int i=0; i<3; i++)
	{	for(int j=0; j<3; j++)
		{	double sum = 0., sumSq = 0.; int N = 0;
			for(size_t block=0; block<result.size(); block++)
			{	N++;
				sum += result[block](i,j);
				sumSq += std::pow(result[block](i,j), 2);
			}
			resultMean(i,j) = sum/N;
			resultStd(i,j) = sqrt(sumSq/N - std::pow(sum/N,2));
		}
		char mOpen[] = "/|\\", mClose[] = "\\|/";
		logPrintf("%20s%c", i==1 ? (resultName + " = ").c_str() : "", mOpen[i]);
		for(int j=0; j<3; j++) logPrintf(" %12lg", resultMean(i,j)/unit);
		logPrintf(" %c%5s%c", mClose[i], i==1 ? " +/- " : "", mOpen[i]);
		for(int j=0; j<3; j++) logPrintf(" %12lg", resultStd(i,j)/unit);
		logPrintf(" %c %s\n", mClose[i], i==1 ? unitName.c_str() : "");
	}
	logPrintf("\n");
}

//Report a scalar with error estimates:
void reportResult(const std::vector<double>& result, string resultName, double unit, string unitName)
{	double sum = 0., sumSq = 0.; int N = 0;
	for(size_t block=0; block<result.size(); block++)
	{	N++;
		sum += result[block];
		sumSq += std::pow(result[block], 2);
	}
	double resultMean = sum/N;
	double resultStd = sqrt(sumSq/N - std::pow(sum/N,2));
	logPrintf("%17s = %12lg +/- %12lg %s\n", resultName.c_str(), resultMean/unit, resultStd/unit, unitName.c_str());
}
