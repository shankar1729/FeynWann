#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"
#include <core/Units.h>

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
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of resistivity", inputFilename, dryRun, printDefaults);

	//Read input file:
	InputMap inputMap(inputFilename);
	const int nKpts = inputMap.get("nKpts");
	const int nBlocks = inputMap.get("nBlocks"); assert(nBlocks>0);
	const double T = inputMap.get("T") * Kelvin;
	const double EconserveWidth = inputMap.get("EconserveWidth", T/eV) * eV; //energy conservation width (default to T)
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional shift in chemical potential from neutral value; start of range (default to 0)
	const double dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional shift in chemical potential from neutral value; end of range (default to 0)
	const int dmuCount = inputMap.get("dmuCount", 1); assert(dmuCount>0); //number of chemical potential shifts
	const int slabDir = inputMap.get("slabDir", -1); assert(slabDir<3); //0-based index of direction to eliminate; default -1 => don't eliminate any (keep 3D)
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKpts = %d\n", nKpts);
	logPrintf("nBlocks = %d\n", nBlocks);
	logPrintf("T = %lg\n", T);
	logPrintf("EconserveWidth = %lg\n", EconserveWidth);
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("dmuCount = %d\n", dmuCount);
	logPrintf("slabDir = %d\n", slabDir);
	
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

	/* //DEBUG: check translation invariance of e-ph matrix elements:
	if(mpiUtil->isHead())
	{	FILE* fp = fopen("test_gePh.dat", "w");
		for(int block=0; block<200; block++)
		{	vector3<> k1; for(int j=0; j<3; j++) k1[j] = Random::uniform();
			std::vector< vector3<> > k2arr(bunchSize);
			for(vector3<>& k2: k2arr) for(int j=0; j<3; j++) k2[j] = k1[j] + Random::normal(0., 0.05, 2);
			k2arr[0] = k1;
			std::vector<matrix> gePh[bunchSize];
			bs.setPhononMatElemArray(k1, k2arr, gePh);
			for(int ik2=0; ik2<bunchSize; ik2++)
			{	vector3<> dk = k2arr[ik2] - k1;
				for(int j=0; j<3; j++) dk[j] -= floor(0.5+dk[j]);
				double dkMag = (dk * inv(bs.R)).length();
				fprintf(fp, "%lf", dkMag);
				for(int mode=0; mode<3; mode++) //pick only acoustic-like modes
					for(int b=0; b<bs.nBands; b++)
						fprintf(fp, " %le", gePh[ik2][mode](b,b).abs());
				fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}
	die("Testing.\n"); */
	
	//dmu array:
	std::vector<double> dmu(dmuCount, dmuMin); //set first value here
	for(int iMu=1; iMu<dmuCount; iMu++) //set remaining values (if any)
		dmu[iMu] = dmuMin + iMu*(dmuMax-dmuMin)/(dmuCount-1);
	
	//Handle dimensionality:
	double Omega = fabs(det(bs.R));
	double rhoUnit = 1e-9*Ohm*meter;
	string rhoUnitName="nOhm-m";
	string rhoName = "Resistivity";
	if(slabDir>=0)
	{	Omega /= bs.R.column(slabDir).length(); //convert to area excluding this dimension
		rhoUnit = Ohm;
		rhoUnitName = "Ohm";
		rhoName = "SheetResistance";
	}
	
	//Compute resistivity:
	#define DeclareArray2D(type, name) std::vector<std::vector<type>> name(dmuCount, std::vector<type>(nBlocks))
	DeclareArray2D(matrix3<>, Tarr); DeclareArray2D(matrix3<>, GammaArr); DeclareArray2D(matrix3<>, rhoArr);
	DeclareArray2D(double, rhoBarArr); DeclareArray2D(double, tauArr); DeclareArray2D(double, tauDrudeArr);
	DeclareArray2D(double, vFarr); DeclareArray2D(double, gArr); 
	#undef DeclareArray2D
	int nKptsMin = ceildiv(nKpts, nBlocks*mpiUtil->nProcesses()); //number of k points per block per process
	const double Emax = std::max(fabs(dmuMin), fabs(dmuMax)) + 10*T + 5*EconserveWidth; //max energy from Fermi level to consider
	double EconserveExpFac = -0.5/std::pow(EconserveWidth,2), EconservePrefac = 1./(sqrt(2*M_PI)*EconserveWidth); //energy conserving Gaussian parameters
	for(int block=0; block<nBlocks; block++)
	{	logPrintf("Working on block %d of %d ... ", block+1, nBlocks); logFlush();
		std::vector<matrix3<>> Tcur(dmuCount), Gamma(dmuCount);
		std::vector<double> g(dmuCount), tauInv(dmuCount);
		double nKpts = 0.; int nBunches = 0;
		while(nKpts < nKptsMin)
		{	//Get a bunch of k-points with states near the Fermi level:
			std::vector< vector3<> > kArr; kArr.reserve(bunchSize);
			while(kArr.size() < size_t(bunchSize))
			{	//Diagonalize Hamiltonians at a set of random k-points:
				std::vector< vector3<> > kTmp(bunchSize);
				for(vector3<>& k: kTmp)
					for(int j=0; j<3; j++)
						k[j] = Random::uniform();
				std::vector<diagMatrix> Etmp = bs.getStates(kTmp, Emax);
				//Add k-points with appropriate states:
				int nFound = 0, nAdded = 0;
				for(int ik=0; ik<bunchSize; ik++)
				{	bool worthwhile = false;
					for(int b=0; b<Etmp[ik].nRows(); b++)
						if(fabs(Etmp[ik][b]) < Emax)
						{	worthwhile = true;
							break;
						}
					if(worthwhile)
					{	nFound++;
						if(kArr.size() < size_t(bunchSize))
						{	kArr.push_back(kTmp[ik]);
							nAdded++;
						}
					}
				}
				nKpts += bunchSize * (nFound ? nAdded * (1./nFound) : 1.); //number of k-points examined to get the relevant ones (needed for normalization)
			}
			nBunches++;
			
			//Get energies and velocities for selected bunch:
			std::vector<diagMatrix> Earr = bs.getStates(kArr, Emax);
			std::vector< vector3<> > vArr[bunchSize];
			for(int ik=0; ik<bunchSize; ik++)
				vArr[ik] = bs.getVelocity(kArr[ik], Emax);
			
			diagMatrix omegaPh[bunchSize];
			std::vector<matrix> gePh[bunchSize];
			for(int ik1=0; ik1<bunchSize; ik1++)
			{	//Calculate phonon stuff for each pair of k-points involving ik1
				bs.setPhononMatElemArray(kArr[ik1], kArr, gePh);
				for(int ik2=0; ik2<bunchSize; ik2++)
					omegaPh[ik2] = bs.getPhononModes(kArr[ik1] - kArr[ik2]);
				
				for(int v=0; v<Earr[ik1].nRows(); v++)
				{	matrix3<> viDotvi = outer(vArr[ik1][v], vArr[ik1][v]);
					std::vector<double> dFdEi(dmuCount);
					for(int iMu=0; iMu<dmuCount; iMu++)
					{	double dFdEi = -1/(T*std::pow(2*cosh((Earr[ik1][v]-dmu[iMu])/(2*T)),2));
						Tcur[iMu] += viDotvi*(-dFdEi);
						g[iMu] += (-dFdEi);
						for(int ik2=0; ik2<bunchSize; ik2++)
							if(ik2 != ik1)
								for(int c=0; c<Earr[ik2].nRows(); c++)
								{	matrix3<> viDotvj = outer(vArr[ik1][v], vArr[ik2][c]);
									double fj = 1./(exp((Earr[ik2][c]-dmu[iMu])/T)+1);
									for(int alpha=0; alpha<omegaPh[ik2].nRows(); alpha++)
									{	double nPh = 1./(exp(omegaPh[ik2][alpha]/T) - 1.);
										double gePhSq = gePh[ik2][alpha](c,v).norm();
										for(int ae=-1; ae<=+1; ae+=2)
										{	double deltaExp = EconserveExpFac * std::pow(Earr[ik2][c]-Earr[ik1][v] - ae*omegaPh[ik2][alpha],2);
											if(deltaExp < -15.) continue; //delta will be negligible
											double delta = EconservePrefac * exp(deltaExp);
											double occFactors = (-dFdEi) * (nPh+0.5 - ae*(0.5-fj));
											Gamma[iMu] += (viDotvi -  viDotvj) * (occFactors * delta * gePhSq);
											tauInv[iMu] += occFactors * delta * gePhSq;
										}
									}
								}
					}
				}
			}
		}
		
		//Accumulate between processes:
		for(int iMu=0; iMu<dmuCount; iMu++)
		{	mpiUtil->allReduce(&Tcur[iMu](0,0), 3*3, MPIUtil::ReduceSum);
			mpiUtil->allReduce(&Gamma[iMu](0,0), 3*3, MPIUtil::ReduceSum);
			mpiUtil->allReduce(g[iMu], MPIUtil::ReduceSum);
			mpiUtil->allReduce(tauInv[iMu], MPIUtil::ReduceSum);
		}
		mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum);
		mpiUtil->allReduce(nBunches, MPIUtil::ReduceSum);
		logPrintf("useFraction: %lg\n", (bunchSize*nBunches)/nKpts); logFlush();
		
		double prefacT = bs.spinWeight/(nKpts);
		double prefacGamma = bs.spinWeight*(2*M_PI)/(nKpts*nKpts*1./nBunches);
		//Apply normalizing factors:
		for(int iMu=0; iMu<dmuCount; iMu++)
		{	Tcur[iMu] *= prefacT;
			g[iMu] *= prefacT;
			Gamma[iMu] *= prefacGamma; Gamma[iMu] = 0.5*(Gamma[iMu] + (~Gamma[iMu])); //symmetrize
			tauInv[iMu] *= prefacGamma;
			slabConstrain(Tcur[iMu], slabDir); //eliminate out-of-plane components if necessary
			slabConstrain(Gamma[iMu], slabDir); //eliminate out-of-plane components if necessary
			//Store relevant quantities:
			Tarr[iMu][block] = Tcur[iMu];
			GammaArr[iMu][block] = Gamma[iMu];
			rhoArr[iMu][block] = Omega * (inv(Tcur[iMu]) * Gamma[iMu] * inv(Tcur[iMu]));
			rhoBarArr[iMu][block] = trace(rhoArr[iMu][block], slabDir) / (slabDir>=0 ? 2. : 3.);
			tauArr[iMu][block] = g[iMu] / tauInv[iMu];
			tauDrudeArr[iMu][block] = trace(Tcur[iMu], slabDir) / trace(Gamma[iMu], slabDir);
			vFarr[iMu][block] = sqrt(trace(Tcur[iMu], slabDir)/g[iMu]);
			gArr[iMu][block] = g[iMu];
			//Fix slab direction values physically:
			if(slabDir>=0.)
			{	Tarr[iMu][block](slabDir,slabDir) = 0.;
				GammaArr[iMu][block](slabDir,slabDir) = 0.;
				rhoArr[iMu][block](slabDir,slabDir) = INFINITY;
			}
		}
	}
	logPrintf("Done.\n\n");

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
	
	finalizeSystem();
}
