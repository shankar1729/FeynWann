#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"
#include "Units.h"

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


int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of resistivity", inputFilename, dryRun, printDefaults);

	//Read input file:
	InputMap inputMap(inputFilename);
	const int nKpts = inputMap.get("nKpts");
	const int nBlocks = inputMap.get("nBlocks"); assert(nBlocks>0);
	const double T = inputMap.get("T") * Kelvin;
	const double EconserveWidth = inputMap.get("EconserveWidth", T/eV) * eV; //energy conservation width (default to T)
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKpts = %d\n", nKpts);
	logPrintf("nBlocks = %d\n", nBlocks);
	logPrintf("T = %lg\n", T);
	logPrintf("EconserveWidth = %lg\n", EconserveWidth);
	
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

	/* //HACK: check translation invariance of e-ph matrix elements:
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
	
	// Compute T and Gamma
	std::vector<matrix3<>> Tarr(nBlocks), GammaArr(nBlocks), rhoArr(nBlocks); //results per block
	std::vector<double> rhoBarArr(nBlocks), tauArr(nBlocks), tauDrudeArr(nBlocks), vFarr(nBlocks), gArr(nBlocks);
	int nKptsMin = ceildiv(nKpts, nBlocks*mpiUtil->nProcesses()); //number of k points per block per process
	double Omega = fabs(det(bs.R));
	const double Emax = 10*T; //max energy from Fermi level to consider
	double EconserveExpFac = -0.5/(T*T), EconservePrefac = 1./(sqrt(2*M_PI)*T); //energy conserving Gaussian parameters
	for(int block=0; block<nBlocks; block++)
	{	logPrintf("Working on block %d of %d\n", block+1, nBlocks); logFlush();
		matrix3<> Tcur, Gamma;
		double g = 0., tauInv=0.;
		double nKpts = 0.; int nBunches = 0;
		while(nKpts < nKptsMin)
		{	//Get a bunch of k-points with states near the Fermi level:
			std::vector< vector3<> > kArr; kArr.reserve(bunchSize);
			while(kArr.size() < bunchSize)
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
						if(kArr.size() < bunchSize)
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
				{	double dFdEi = -1/(T*std::pow(2*cosh(Earr[ik1][v]/(2*T)),2));
					matrix3<> viDotvi = outer(vArr[ik1][v], vArr[ik1][v]);
					Tcur += viDotvi*(-dFdEi);
					g += (-dFdEi);
					for(int ik2=0; ik2<bunchSize; ik2++)
						if(ik2 != ik1)
							for(int c=0; c<Earr[ik2].nRows(); c++)
							{	matrix3<> viDotvj = outer(vArr[ik1][v], vArr[ik2][c]);
								double fj = 1./(exp(Earr[ik2][c]/T)+1);
								for(int alpha=0; alpha<omegaPh[ik2].nRows(); alpha++)
								{	double nPh = 1./(exp(omegaPh[ik2][alpha]/T) - 1.);
									double gePhSq = gePh[ik2][alpha](c,v).norm();
									for(int ae=-1; ae<=+1; ae+=2)
									{	double delta = EconservePrefac * exp(EconserveExpFac * std::pow(Earr[ik2][c]-Earr[ik1][v] - ae*omegaPh[ik2][alpha],2));
										double occFactors = (-dFdEi) * (nPh+0.5 - ae*(0.5-fj));
										Gamma += (viDotvi -  viDotvj) * (occFactors * delta * gePhSq);
										tauInv += occFactors * delta * gePhSq;
									}
								}
							}
				}
			}
		}
		
		//Accumulate between processes:
		mpiUtil->allReduce(&Tcur(0,0), 3*3, MPIUtil::ReduceSum);
		mpiUtil->allReduce(&Gamma(0,0), 3*3, MPIUtil::ReduceSum);
		mpiUtil->allReduce(g, MPIUtil::ReduceSum);
		mpiUtil->allReduce(tauInv, MPIUtil::ReduceSum);
		mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum);
		mpiUtil->allReduce(nBunches, MPIUtil::ReduceSum);
		
		//Apply normalizing factors:
		double prefacT = bs.spinWeight/(nKpts);
		double prefacGamma = bs.spinWeight*(2*M_PI)/(nKpts*nKpts*1./nBunches);
		Tcur *= prefacT;
		g *= prefacT;
		Gamma *= prefacGamma; Gamma = 0.5*(Gamma + (~Gamma)); //symmetrize
		tauInv *= prefacGamma;
		//Store relevant quantities:
		Tarr[block] = Tcur;
		GammaArr[block] = Gamma;
		rhoArr[block] = Omega * (inv(Tcur) * Gamma * inv(Tcur));
		rhoBarArr[block] = trace(rhoArr[block])/3.;
		tauArr[block] = g / tauInv;
		tauDrudeArr[block] = trace(Tcur) / trace(Gamma);
		vFarr[block] = sqrt(trace(Tcur)/g);
		gArr[block] = g;
	}
	logPrintf("Done.\n\n");

	reportResult(Tarr, "T", 1, "");
	reportResult(GammaArr, "Gamma", 1, "");
	reportResult(rhoArr, "Resistivity", 1e-9*Ohm*meter, "nOhm-m");
	reportResult(rhoBarArr, "Resistivity", 1e-9*Ohm*meter, "nOhm-m");
	reportResult(tauDrudeArr, "tauDrude", fs, "fs");
	reportResult(tauArr, "tau", fs, "fs");
	reportResult(vFarr, "vF", 1, "");
	reportResult(gArr, "g(eF)", 1, "");
	
	finalizeSystem();
}
