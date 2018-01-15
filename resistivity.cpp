#include "WannierMC.h"
#include "InputMap.h"
#include <core/Units.h>
#include <core/Random.h>

struct ResistivityCollect
{	std::vector<double> dmu; //doping levels
	double T; //temperature
	std::vector<double> n, g, vSq, tau; //carrier number, density of states, |v|^2 and e-ph life time
	std::vector<matrix3<>> vvTau; //scattering time * velocity outer product

	ResistivityCollect(const std::vector<double>& dmu, double T) : dmu(dmu), T(T),
		n(dmu.size()), g(dmu.size()), vSq(dmu.size()), tau(dmu.size()), vvTau(dmu.size())
	{
	}
	
	void collect(const WannierMC::StateE& state)
	{	const int nBands = state.E.nRows();
		double invT = 1./T;
		for(int b=0; b<nBands; b++)
		{	const double& E = state.E[b];
			const vector3<>& v = state.vVec[b];
			matrix3<> vdotv = outer(v, v);
			for(unsigned iMu=0; iMu<dmu.size(); iMu++)
			{	double expArg = (E-dmu[iMu])*invT;
				if(fabs(expArg)>30.) //avoid over/underflow
				{	if(expArg<0.) n[iMu] += 1.; //still need to count towards n if occupied
					continue; //negligible contribution to rest
				}
				double expTerm = exp(expArg);
				double f = 1./(1.+expTerm);
				double dfdE = -f*f*expTerm*invT;
				n[iMu] += f;
				g[iMu] += (-dfdE);
				vSq[iMu] += (-dfdE) * v.length_squared();
				tau[iMu] += (-dfdE) / (2*state.ImSigma_ePh(b,f));
				vvTau[iMu] += ((-dfdE) / (2*state.ImSigmaP_ePh(b,f))) * vdotv;
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
	double Nconduction = inputMap.get("Nconduction", 0.); //optional number of DFT electrons to be counted as conduction (default to 0); set for metal mobility calc
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional shift in chemical potential from neutral value; start of range (default to 0)
	const double dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional shift in chemical potential from neutral value; end of range (default to 0)
	const int dmuCount = inputMap.get("dmuCount", 1); assert(dmuCount>0); //number of chemical potential shifts
	const int slabDir = inputMap.get("slabDir", -1); assert(slabDir<3); //0-based index of direction to eliminate; default -1 => don't eliminate any (keep 3D)
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("nBlocks = %d\n", nBlocks);
	logPrintf("T = %lg\n", T);
	logPrintf("Nconduction = %lg\n", Nconduction);
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("dmuCount = %d\n", dmuCount);
	logPrintf("slabDir = %d\n", slabDir);
	
	//Initialize WannierMC:
	WannierMCParams wmcp;
	wmcp.needSymmetries = true;
	wmcp.needVelocity = true;
	wmcp.needLinewidth_ePh = true;
	wmcp.needLinewidthP_ePh = true;
	std::shared_ptr<WannierMC> wmc = std::make_shared<WannierMC>(wmcp);
	
	//dmu array:
	std::vector<double> dmu(dmuCount, dmuMin); //set first value here
	for(int iMu=1; iMu<dmuCount; iMu++) //set remaining values (if any)
		dmu[iMu] = dmuMin + iMu*(dmuMax-dmuMin)/(dmuCount-1);
	
	//Handle dimensionality:
	double Omega = wmc->Omega;
	double rhoUnit = 1e-9*Ohm*meter;
	string rhoUnitName="nOhm-m";
	string rhoName = "Resistivity";
	const double cm = 1e-2*meter;
	const double cm2byVs = cm*cm/(Volt*sec);
	double densityUnit = std::pow(cm,-3);
	string densityUnitName = "cm^-3";
	if(slabDir>=0)
	{	Omega /= wmc->R.column(slabDir).length(); //convert to area excluding this dimension
		rhoUnit = Ohm;
		rhoUnitName = "Ohm";
		rhoName = "SheetResistance";
		densityUnit = std::pow(cm,-2);
		densityUnitName = "cm^-2";
	}
	
	//Initialize sampling parameters:
	int nOffsetsPerBlock = ceildiv(nOffsets, nBlocks);
	size_t nKptsPerBlock = wmc->eCountPerOffset() * nOffsetsPerBlock;
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
		wmc = 0;
		WannierMC::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Collect integrals involved in resistivity:
	double prefacDOS = wmc->spinWeight*(1./nKptsPerBlock);
	std::vector<std::vector<std::shared_ptr<ResistivityCollect>>> rcArr(wmc->nSpins);
	std::vector<string> spinSuffixes(wmc->nSpins);
	for(int iSpin=0; iSpin<wmc->nSpins; iSpin++)
	{	//Update WannierMC for spin channel if necessary:
		if(iSpin>0)
		{	wmc = 0; //free memory from previous spin
			wmcp.iSpin = iSpin;
			wmc = std::make_shared<WannierMC>(wmcp);
		}
		spinSuffixes[iSpin] = wmc->spinSuffix;
		rcArr[iSpin].resize(nBlocks);
		for(int block=0; block<nBlocks; block++)
		{	logPrintf("Working on block %d of %d: ", block+1, nBlocks); logFlush();
			rcArr[iSpin][block] = std::make_shared<ResistivityCollect>(dmu, T);
			ResistivityCollect& rc = *rcArr[iSpin][block];
			for(int o=0; o<noMine; o++)
			{	Random::seed(block*nOffsetsPerBlock+o+oStart); //to make results independent of MPI division
				//Process with a random offset:
				vector3<> k0 = wmc->randomVector(mpiGroup); //must be constant across group
				wmc->eLoop(k0, ResistivityCollect::eProcess, &rc);
				//Print progress:
				if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
			}
			for(int iMu=0; iMu<dmuCount; iMu++)
			{	//Accumulate between processes:
				mpiWorld->allReduce(rc.n[iMu], MPIUtil::ReduceSum);
				mpiWorld->allReduce(rc.g[iMu], MPIUtil::ReduceSum);
				mpiWorld->allReduce(rc.vSq[iMu], MPIUtil::ReduceSum);
				mpiWorld->allReduce(rc.tau[iMu], MPIUtil::ReduceSum);
				mpiWorld->allReduce(&rc.vvTau[iMu](0,0), 3*3, MPIUtil::ReduceSum);
				//Apply normalizing factors:
				rc.n[iMu] *= prefacDOS;
				rc.n[iMu] -= (wmc->nElectrons - Nconduction)/wmc->nSpins; //convert to number of free carriers per unit cell
				rc.g[iMu] *= prefacDOS;
				rc.vSq[iMu] *= prefacDOS;
				rc.tau[iMu] *= prefacDOS;
				rc.vvTau[iMu] *= prefacDOS;
				slabConstrain(rc.vvTau[iMu], slabDir); //eliminate out-of-plane components if necessary
				wmc->symmetrize(rc.vvTau[iMu]); //follow symmetries of unit cell
			}
			logPrintf("done.\n"); logFlush();
		}
	}
	
	//Generate channel for sum over spins if necessary:
	if(wmc->nSpins > 1)
	{	spinSuffixes.push_back(""); //no suffix for total
		rcArr.resize(wmc->nSpins+1);
		rcArr.back().resize(nBlocks);
		for(int block=0; block<nBlocks; block++)
		{	rcArr.back()[block] = std::make_shared<ResistivityCollect>(dmu, T);
			ResistivityCollect& rcTot = *rcArr.back()[block];
			for(int iSpin=0; iSpin<wmc->nSpins; iSpin++)
				for(int iMu=0; iMu<dmuCount; iMu++)
				{	rcTot.n[iMu] += rcArr[iSpin][block]->n[iMu];
					rcTot.g[iMu] += rcArr[iSpin][block]->g[iMu];
					rcTot.vSq[iMu] += rcArr[iSpin][block]->vSq[iMu];
					rcTot.tau[iMu] += rcArr[iSpin][block]->tau[iMu];
					rcTot.vvTau[iMu] += rcArr[iSpin][block]->vvTau[iMu];
				}
		}
	}
	
	//Compute resistivity and related quantities along with statistics for each mu:
	for(int iMu=0; iMu<dmuCount; iMu++)
	{	logPrintf("\nResults for dmu = %lg eV:\n", dmu[iMu]/eV);
		for(size_t iSpin=0; iSpin<rcArr.size(); iSpin++)
		{	string spinSuffix = spinSuffixes[iSpin];
			//Compute quantities for each block:
			std::vector<matrix3<>> rhoArr(nBlocks), mobArr(nBlocks);
			std::vector<double> rhoBarArr(nBlocks), mobBarArr(nBlocks); 
			std::vector<double> tauArr(nBlocks), tauDrudeArr(nBlocks);
			std::vector<double> mEffArr(nBlocks), vFarr(nBlocks);
			std::vector<double> gArr(nBlocks), nArr(nBlocks);
			for(int block=0; block<nBlocks; block++)
			{	const ResistivityCollect& rc = *rcArr[iSpin][block];
				rhoArr[block] = Omega * inv(rc.vvTau[iMu]);
				mobArr[block] = rc.vvTau[iMu]/fabs(rc.n[iMu]);
				if(slabDir>=0.)
				{	rhoArr[block](slabDir,slabDir) = INFINITY;
					mobArr[block](slabDir,slabDir) = 0.;
				}
				rhoBarArr[block] = trace(rhoArr[block], slabDir) / (slabDir>=0 ? 2. : 3.);
				mobBarArr[block] = trace(mobArr[block], slabDir) / (slabDir>=0 ? 2. : 3.);
				tauArr[block] = rc.tau[iMu] / rc.g[iMu];
				tauDrudeArr[block] = trace(rc.vvTau[iMu], slabDir) / rc.vSq[iMu];
				mEffArr[block] = tauDrudeArr[block] / mobBarArr[block]; //mobility-effective-mass
				vFarr[block] = sqrt(rc.vSq[iMu] / rc.g[iMu]);
				gArr[block] = rc.g[iMu];
				nArr[block] = rc.n[iMu];
			}
			//Report with statistics:
			reportResult(rhoArr, rhoName+spinSuffix, rhoUnit, rhoUnitName);
			reportResult(mobArr, "Mobility"+spinSuffix, cm2byVs, "cm^2/(V.s)");
			reportResult(rhoBarArr, rhoName+spinSuffix, rhoUnit, rhoUnitName);
			reportResult(mobBarArr, "Mobility"+spinSuffix, cm2byVs, "cm^2/(V.s)");
			reportResult(tauDrudeArr, "tauDrude"+spinSuffix, fs, "fs");
			reportResult(tauArr, "tau"+spinSuffix, fs, "fs");
			reportResult(mEffArr, "mEff"+spinSuffix, 1, "");
			reportResult(vFarr, "vF"+spinSuffix, 1, "");
			reportResult(gArr, "g"+spinSuffix+"(eF)", 1, "");
			reportResult(nArr, "Ncarriers"+spinSuffix, 1, "cell^-1");
			reportResult(nArr, "nCarriers"+spinSuffix, (Omega*densityUnit), densityUnitName);
			logPrintf("\n");
		}
	}
	
	wmc = 0;
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
