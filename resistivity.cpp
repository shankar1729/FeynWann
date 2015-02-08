#include <core/Util.h>
#include <electronic/matrix.h>
#include <fstream>
#include <iterator>
#include <sstream>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"
#include "Units.h"

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of resistivity", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const int totalBlocks = inputMap.get("totalBlocks"); assert(totalBlocks>0);
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const double vl = inputMap.get("vl")* meter *2.41888e-17;// m/s in atomic units
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("totalBlocks = %d\n", totalBlocks);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("vl = %lg\n", vl);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Initialize Wannier bandstructure:
	const int bunchSize = 32;
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE");
	bs.setCacheSize(2*bunchSize);
	
	// Compute T and Gamma
	double Tsum = 0., TsumSq = 0., GammaSum = 0., GammaSumSq = 0.;
	double gSum = 0., gSumSq = 0., tauInvSum = 0., tauInvSumSq = 0.;
	logPrintf("Calculating T and Gamma... "); logFlush();
	int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
	int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nKptsMin = nKptsN1/totalBlocks;
	double Omega = fabs(det(R));
	const double Emax = 10*T; //max energy from Fermi level to consider
	double EconserveExpFac = -0.5/(T*T), EconservePrefac = 1./(sqrt(2*M_PI)*T); //energy conserving Gaussian parameters
	for(int block=blockStart; block<blockStop; block++)
	{	Random::seed(block);
		double Tblock = 0., GammaBlock=0.;
		double gBlock = 0., tauInvBlock=0.;
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
				vArr[ik] = bs.getVelocity(kArr[ik], R, Emax);
			
			diagMatrix omegaPh[bunchSize];
			std::vector<matrix> MePh[bunchSize];
			for(int ik1=0; ik1<bunchSize; ik1++)
			{	//Calculate phonon stuff for each pair of k-points involving ik1
				bs.setPhononMatElemArray(kArr[ik1], kArr, MePh);
				for(int ik2=0; ik2<bunchSize; ik2++)
					omegaPh[ik2] = bs.getPhononModes(kArr[ik1] - kArr[ik2]);
				
				for(int v=0; v<Earr[ik1].nRows(); v++)
				{	double dFdEi = -1/(T*std::pow(2*cosh(Earr[ik1][v]/(2*T)),2));
					double viDotvi = vArr[ik1][v].length_squared();
					Tblock += viDotvi*(-dFdEi);
					gBlock += (-dFdEi);
					for(int ik2=0; ik2<bunchSize; ik2++)
						if(ik2 != ik1)
							for(int c=0; c<Earr[ik2].nRows(); c++)
							{	double viDotvj = dot(vArr[ik1][v], vArr[ik2][c]);
								double fj = 1./(exp(Earr[ik2][c]/T)+1);
								for(int alpha=0; alpha<omegaPh[ik2].nRows(); alpha++)
								{	double gk = 1./(exp(omegaPh[ik2][alpha]/T) - 1.);
									double Msq_by_omega = MePh[ik2][alpha](c,v).norm() / omegaPh[ik2][alpha];
									for(int ae=-1; ae<=+1; ae+=2)
									{	double delta = EconservePrefac * exp(EconserveExpFac * std::pow(Earr[ik2][c]-Earr[ik1][v] - ae*omegaPh[ik2][alpha],2));
										double occFactors = (-dFdEi) * (gk+0.5 - ae*(0.5-fj));
										GammaBlock += (viDotvi -  viDotvj) * occFactors * delta * Msq_by_omega;
										tauInvBlock += occFactors * delta * Msq_by_omega;
									}
								}
							}
				}
			}
		}
		double prefacT = spinWeight/(3.*nKpts);
		double prefacGamma = spinWeight*M_PI/(3*nKpts*nKpts*1./nBunches);
		Tblock *= prefacT; Tsum += Tblock; TsumSq += std::pow(Tblock,2);
		gBlock *= prefacT; gSum += gBlock; gSumSq += std::pow(gBlock,2);
		GammaBlock *= prefacGamma; GammaSum += GammaBlock; GammaSumSq += std::pow(GammaBlock,2);
		tauInvBlock *= prefacGamma; tauInvSum += tauInvBlock; tauInvSumSq += std::pow(tauInvBlock,2);
	}

	mpiUtil->allReduce(Tsum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(TsumSq, MPIUtil::ReduceSum);
	double Tt = Tsum / totalBlocks;
	double Tstd = sqrt(TsumSq/totalBlocks - Tt*Tt)/sqrt(totalBlocks);
	logPrintf("T = %lg +/- %lg\n", Tt, Tstd);
	
	//Decay rate:
	mpiUtil->allReduce(GammaSum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(GammaSumSq, MPIUtil::ReduceSum);
	double Gamma = GammaSum / totalBlocks;
	double GammaStd = sqrt(GammaSumSq/totalBlocks - Gamma*Gamma)/sqrt(totalBlocks);
	logPrintf("Gamma = %lg +/- %lg\n", Gamma, GammaStd);
	
	// Calculate Resistivity
	double rho = Omega*Gamma/(Tt*Tt);
	double rhoStd = rho * hypot(GammaStd/Gamma, sqrt(2.)*Tstd/Tt);
	logPrintf("Resistivity = %lg +/- %lg ohm-m\n", rho/(Ohm*meter), rhoStd/(Ohm*meter));
	
	//Drude relaxation time
	double tauDrude = Tt / Gamma;
	double tauDrudeStd = tauDrude * hypot(Tstd/Tt, GammaStd/Gamma);
	logPrintf("tauDrude = %lg +/- %lg fs\n", tauDrude/fs, tauDrudeStd/fs);
	
	//Calculate lifetime:
	mpiUtil->allReduce(gSum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(gSumSq, MPIUtil::ReduceSum);
	mpiUtil->allReduce(tauInvSum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(tauInvSumSq, MPIUtil::ReduceSum);
	double gMean = gSum / totalBlocks;
	double gStd = sqrt(gSumSq/totalBlocks - gMean*gMean)/sqrt(totalBlocks);
	double tauInv = tauInvSum / totalBlocks;
	double tauInvStd = sqrt(tauInvSumSq/totalBlocks - tauInv*tauInv)/sqrt(totalBlocks);
	double tau = gMean / tauInv;
	double tauStd = tau * hypot(tauInvStd/tauInv, gStd/gMean);
	logPrintf("tau = %lg +/- %lg fs\n", tau/fs, tauStd/fs);
	
	double vF = sqrt(Tt/gMean);
	double vFstd = vF * 0.5 * hypot(Tstd/Tt, gStd/gMean);
	logPrintf("vF = %lg +/- %lg\n", vF, vFstd);
	
	logPrintf("g(eF) = %lg +/- %lg\n", 3.*gMean, 3.*gStd);
	finalizeSystem();
}
