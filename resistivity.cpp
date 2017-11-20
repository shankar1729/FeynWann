#include "WannierMC.h"
#include "InputMap.h"
#include <core/Units.h>
#include <core/Random.h>

struct ResistivityCollect
{	std::vector<double> dmu; //doping levels
	double T; //temperature
	double EconserveExpFac, EconservePrefac; //energy conserving Gaussian exponential and prefactor
	std::vector<double> g, tauInv; //density of states and scattering time
	std::vector<matrix3<>> Tcur, Gamma; //above with velocity matrices multiplied

	ResistivityCollect(const std::vector<double>& dmu, double T, double EconserveWidth) : dmu(dmu), T(T),
		EconserveExpFac(-0.5/std::pow(EconserveWidth,2)), EconservePrefac(1./(sqrt(2*M_PI)*EconserveWidth)), //energy conserving Gaussian parameters
		g(dmu.size()), tauInv(dmu.size()), Tcur(dmu.size()), Gamma(dmu.size())
	{
	}
	
	void collect(const WannierMC::MatrixEph& m)
	{	const WannierMC::StateE& e1 = *(m.e1);
		const WannierMC::StateE& e2 = *(m.e2);
		const WannierMC::StatePh& ph = *(m.ph);
		const int nBands = e1.E.nRows();
		const int nModes = ph.omega.nRows();
		//Loop over electron 1:
		for(int b1=0; b1<nBands; b1++)
		{	const double& E1 = e1.E[b1];
			const vector3<>& v1 = e1.vVec[b1];
			matrix3<> v1dotv1 = outer(v1, v1);
			for(unsigned iMu=0; iMu<dmu.size(); iMu++)
			{	double dFdE1 = -1./(T*std::pow(2*cosh((E1-dmu[iMu])/(2*T)),2));
				Tcur[iMu] += v1dotv1*(-dFdE1);
				g[iMu] += (-dFdE1);
				for(int b2=0; b2<nBands; b2++)
				{	const double& E2 = e2.E[b2];
					const vector3<>& v2 = e2.vVec[b2];
					matrix3<> v1dotv2 = outer(v1, v2);
					double f2 = 1./(exp((E2-dmu[iMu])/T)+1);
					for(int alpha=0; alpha<nModes; alpha++)
					{	double nPh = 1./(exp(ph.omega[alpha]/T) - 1.);
						double gePhSq = m.M[alpha](b2,b1).norm();
						for(int ae=-1; ae<=+1; ae+=2) //absorb or emit phonon
						{	double deltaExp = EconserveExpFac * std::pow(E2 - E1 - ae*ph.omega[alpha],2);
							if(deltaExp < -15.) continue; //delta will be negligible
							double delta = EconservePrefac * exp(deltaExp);
							double occFactors = (-dFdE1) * (nPh+0.5 - ae*(0.5-f2));
							Gamma[iMu] += (v1dotv1 -  v1dotv2) * (occFactors * delta * gePhSq);
							tauInv[iMu] += occFactors * delta * gePhSq;
						}
					}
				}
			}
		}
	}
	
	static void ePhProcess(const WannierMC::MatrixEph& m, void* params)
	{	((ResistivityCollect*)params)->collect(m);
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
	wmcp.needPhonons = true;
	wmcp.needVelocity = true;
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
	size_t nKpairsPerBlock = wmc.ePhCountPerOffset() * nOffsetsPerBlock;
	logPrintf("Effectively sampled nKpairs: %lu\n", nKpairsPerBlock * nBlocks);
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
	double prefacT = wmc.spinWeight*1./nKpairsPerBlock;
	double prefacGamma = wmc.spinWeight*(2*M_PI)/nKpairsPerBlock;
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
			//Process with a random offset pair:
			vector3<> k01 = wmc.randomVector(mpiGroup); //must be constant across group
			vector3<> k02 = wmc.randomVector(mpiGroup); //must be constant across group
			wmc.ePhLoop(k01, k02, ResistivityCollect::ePhProcess, &rc);
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
			rc.Tcur[iMu] *= prefacT;
			rc.g[iMu] *= prefacT;
			rc.Gamma[iMu] *= prefacGamma; rc.Gamma[iMu] = 0.5*(rc.Gamma[iMu] + (~rc.Gamma[iMu])); //symmetrize
			rc.tauInv[iMu] *= prefacGamma;
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
