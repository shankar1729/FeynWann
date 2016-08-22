#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"
#include "Units.h"

void reportResult(const std::vector<matrix3<>>& result, string resultName, double unit, string unitName, int blockStart, int blockStop)
{	matrix3<> resultMean, resultStd;
	for(int i=0; i<3; i++)
	{	for(int j=0; j<3; j++)
		{	double sum = 0., sumSq = 0.; int N = 0;
			for(int block=blockStart; block<blockStop; block++)
			{	N++;
				sum += result[block](i,j);
				sumSq += std::pow(result[block](i,j), 2);
			}
			mpiUtil->allReduce(N, MPIUtil::ReduceSum);
			mpiUtil->allReduce(sum, MPIUtil::ReduceSum);
			mpiUtil->allReduce(sumSq, MPIUtil::ReduceSum);
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
void reportResult(const std::vector<double>& result, string resultName, double unit, string unitName, int blockStart, int blockStop)
{	double sum = 0., sumSq = 0.; int N = 0;
	for(int block=blockStart; block<blockStop; block++)
	{	N++;
		sum += result[block];
		sumSq += std::pow(result[block], 2);
	}
	mpiUtil->allReduce(N, MPIUtil::ReduceSum);
	mpiUtil->allReduce(sum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(sumSq, MPIUtil::ReduceSum);
	double resultMean = sum/N;
	double resultStd = sqrt(sumSq/N - std::pow(sum/N,2));
	logPrintf("%17s = %12lg +/- %12lg %s\n", resultName.c_str(), resultMean/unit, resultStd/unit, unitName.c_str());
}


int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of mobility", inputFilename, dryRun, printDefaults);

	//Read input file:
	InputMap inputMap(inputFilename);
	const int nKpts = inputMap.get("nKpts");
	const int totalBlocks = inputMap.get("totalBlocks"); assert(totalBlocks>0);
	const double T = inputMap.get("T") * Kelvin;

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKpts = %d\n", nKpts);
	logPrintf("totalBlocks = %d\n", totalBlocks);
	logPrintf("T = %lg\n", T);
	
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

	
	//Determine band edges:
	int nValence = int(round(bs.nElectrons/bs.spinWeight));
	if(fabs(nValence*bs.spinWeight-bs.nElectrons > 1e-6))
		die("Number of electrons incompatible with semiconductor / insulator.\n");
	if(nValence >= bs.nBands)
		die("Could not find Wannier bands for empty states: needed to calculate electron mobility.\n");
	double EvMax = -DBL_MAX, EcMin = +DBL_MAX;
	logPrintf("Calculating band edges ... "); logFlush();
	int nBunches = nKpts/(bunchSize*mpiUtil->nProcesses());
	int iBunchInterval = std::max(1, int(round(nBunches/50.))); //interval for reporting progress
	for(int iBunch=0;iBunch<nBunches; iBunch++)
	{	//Random block of kpoints:
		std::vector< vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		//Find HOMO and LUMO at these k points:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		for(const diagMatrix& E: Earr)
		{	EvMax = std::max(EvMax, E[nValence-1]);
			EcMin = std::min(EcMin, E[nValence]);
		}
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunches)));
			logFlush();
		}
	}
	mpiUtil->allReduce(EvMax, MPIUtil::ReduceMax);
	mpiUtil->allReduce(EcMin, MPIUtil::ReduceMin);
	logPrintf("done.\n"); logFlush();
	logPrintf("Band edges:  EvMax: %lg  EcMin: %lg\n\n", bs.mu+EvMax, bs.mu+EcMin);
	
	// Compute mobility
	std::vector<matrix3<>> mob_hArr(totalBlocks),  mob_eArr(totalBlocks); //results per block
	std::vector<double> mobBar_hArr(totalBlocks), mobBar_eArr(totalBlocks);
	logPrintf("Calculating mobility ... "); logFlush();
	int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
	int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nKptsMin = nKpts/totalBlocks;
	int nKmine = nKptsMin * (blockStop - blockStart);
	int iKinterval = std::max(1, int(round(nKmine/50.))), iKprev = 0; //interval for reporting progress
	double Omega = fabs(det(bs.R));
	const double Emax = 10*T; //max energy from band edges to consider
	double EconserveExpFac = -0.5/(T*T), EconservePrefac = 1./(sqrt(2*M_PI)*T); //energy conserving Gaussian parameters
	for(int block=blockStart; block<blockStop; block++)
	{	Random::seed(block);
		matrix3<> T_h, T_e, Gamma_h, Gamma_e;
		double g_h = 0., g_e = 0.;
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
					{	const double& E = Etmp[ik][b];
						if( (E>EvMax-Emax && E<EvMax) || (E>EcMin && E<EcMin+Emax) )
						{	worthwhile = true;
							break;
						}
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
				
				for(int b1=0; b1<Earr[ik1].nRows(); b1++)
				{	const double& E1 = Earr[ik1][b1];
					bool is_e = (E1>EcMin && E1<EcMin+Emax);
					bool is_h = (E1>EvMax-Emax && E1<EvMax);
					if(!(is_e || is_h)) continue; //not near either band edge
					
					//Select appropriate band edge to contribute to:
					matrix3<>& Tcur = is_e ? T_e : T_h;
					matrix3<>& Gamma = is_e ? Gamma_e : Gamma_h;
					double& g = is_e ? g_e : g_h;
					double E0 = is_e ? EcMin : EvMax; //relevant band edge position
					
					double w1 = exp(-fabs(E1-E0)/T)/sqrt(fabs(E1-E0));
					matrix3<> v1dotv1 = outer(vArr[ik1][b1], vArr[ik1][b1]);
					Tcur += v1dotv1*w1;
					g += w1;
					for(int ik2=0; ik2<bunchSize; ik2++)
						if(ik2 != ik1)
							for(int b2=0; b2<Earr[ik2].nRows(); b2++)
							{	const double E2 = Earr[ik2][b2];
								if(is_e && !(E2>EcMin && E2<EcMin+Emax)) continue;
								if(is_h && !(E2>EvMax-Emax && E2<EvMax)) continue;
								
								matrix3<> v1dotv2 = outer(vArr[ik1][b1], vArr[ik2][b2]);
								double f2 = is_e ? 0. : 1.;
								for(int alpha=0; alpha<omegaPh[ik2].nRows(); alpha++)
								{	double nPh = 1./(exp(omegaPh[ik2][alpha]/T) - 1.);
									double gePhSq = gePh[ik2][alpha](b2,b1).norm();
									for(int ae=-1; ae<=+1; ae+=2)
									{	double delta = EconservePrefac * exp(EconserveExpFac * std::pow(E2-E1 - ae*omegaPh[ik2][alpha],2));
										double occFactors = w1 * (nPh+0.5 - ae*(0.5-f2));
										Gamma += (v1dotv1 -  v1dotv2) * (occFactors * delta * gePhSq);
									}
								}
							}
				}
			}
			
			//Print progress:
			int iKmine = (block-blockStart)*nKptsMin + int(round(nKpts));
			if(iKmine > iKprev + iKinterval)
			{	logPrintf("%d%% ", int(round((iKmine+1)*100./nKmine)));
				logFlush();
				iKprev = iKmine;
			}
		}
		//Apply normalizing factors:
		double prefacT = bs.spinWeight/(nKpts);
		double prefacGamma = bs.spinWeight*(2*M_PI)/(nKpts*nKpts*1./nBunches);
		T_h *= prefacT; T_e *= prefacT;
		g_h *= prefacT; g_e *= prefacT;
		Gamma_h *= prefacGamma; Gamma_h = 0.5*(Gamma_h + (~Gamma_h)); //symmetrize
		Gamma_e *= prefacGamma; Gamma_e = 0.5*(Gamma_e + (~Gamma_e)); //symmetrize
		//Store relevant quantities:
		mob_hArr[block] = (T_h * inv(Gamma_h) * T_h) / (Omega * g_h);  mobBar_hArr[block] = trace(mob_hArr[block])/3.;
		mob_eArr[block] = (T_e * inv(Gamma_e) * T_e) / (Omega * g_e);  mobBar_eArr[block] = trace(mob_eArr[block])/3.;
	}
	logPrintf("done.\n\n");

	double mobUnit = std::pow(1e-2*meter,2)*invSeconds/Volt;
	reportResult(mob_hArr, "hMobility", mobUnit, "cm^2/(V.s)", blockStart, blockStop);
	reportResult(mobBar_hArr, "hMobility", mobUnit, "cm^2/(V.s)", blockStart, blockStop);
	logPrintf("\n");
	reportResult(mob_eArr, "eMobility", mobUnit, "cm^2/(V.s)", blockStart, blockStop);
	reportResult(mobBar_eArr, "eMobility", mobUnit, "cm^2/(V.s)", blockStart, blockStop);
	logPrintf("\n");
	
	finalizeSystem();
}
