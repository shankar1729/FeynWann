#include <core/Util.h>
#include <electronic/matrix.h>
#include <fstream>
#include <iterator>
#include <sstream>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/Units.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"

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
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE");

	//DEBUG
// 	{
// 		int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
// 		int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
// 		int nKpts = nKptsN1/totalBlocks;
// 		
// 		matrix3<> GT = 2*M_PI*inv(~R);
// 		WignerSeitz BZ(GT);
// 		double kappa = 0.9;
// 		double Omega = fabs(det(R));
// 		
// 		char fname[256]; sprintf(fname, "tempMdebug.%d", mpiUtil->iProcess());
// 		FILE* fp = fopen(fname, "w");
// 		for(int block=blockStart; block<blockStop; block++)
// 		{	Random::seed(block);
// 			for(int nk1 =0; nk1<nKpts; nk1++)
// 			{	vector3<> kpnt1, kpnt2;
// 				for(int j=0; j<3; j++)
// 				{	kpnt1[j] = Random::uniform();
// 					kpnt2[j] = Random::uniform();
// 				}
// 				kpnt2 = kpnt1 + (Random::uniform()/kpnt2.length()) * kpnt2;
// 				{	diagMatrix omegaPh = bs.getPhononModes(kpnt1-kpnt2);
// 					std::vector<matrix> HePh = bs.getPhononMatElem(kpnt1, kpnt2);
// 					double nPairs = 0;
// 					double Msq = 0;
// 					for(size_t alpha=0; alpha<HePh.size(); alpha++)
// 						for(int b=0; b<HePh[alpha].nRows(); b++)
// 						{	Msq += HePh[alpha](b,b).norm();
// 							nPairs += (0.5*spinWeight);
// 						}
// 					Msq *= (HePh.size())/nPairs;
// 					vector3<> kDiff = BZ.restrict(kpnt1-kpnt2);
// 					double k = (GT * kDiff).length();
// 					double MsqOld = std::pow(vl*k,2) * M_PI / (4*Omega*(k*k + kappa*kappa));
// 					fprintf(fp, "%lg %lg %lg\n", k, sqrt(Msq), sqrt(MsqOld));
// 				}
// 			}
// 		}
// 		fclose(fp);
// 		double temp = 0.;
// 		mpiUtil->bcast(temp);
// 		if(mpiUtil->isHead())
// 		{	system("cat tempMdebug.* > tempMdebug");
// 			system("rm tempMdebug.*");
// 		}
// 		finalizeSystem();
// 		return 0;
// 	}
	
	// Compute T and Gamma
	double Tsum = 0., TsumSq = 0., GammaSum = 0., GammaSumSq = 0.;
	double gSum = 0., gSumSq = 0., tauInvSum = 0., tauInvSumSq = 0.;
	logPrintf("Calculating T and Gamma... "); logFlush();
	int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
	int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nKpts = nKptsN1/totalBlocks;
	double Omega = fabs(det(R));
	double prefacT = spinWeight/(3.*nKpts);
	double prefacGamma = spinWeight*M_PI/(3*nKpts);
	double EconserveExpFac = -0.5/(T*T), EconservePrefac = 1./(sqrt(2*M_PI)*T); //energy conserving Gaussian parameters
	for(int block=blockStart; block<blockStop; block++)
	{	Random::seed(block);
		double Tblock = 0., GammaBlock=0.;
		double gBlock = 0., tauInvBlock=0.;
		for(int nk1 =0; nk1<nKpts; nk1++)
		{	vector3<> kpnti, kpntj;
			for(int j=0; j<3; j++) 
			{	kpnti[j] = Random::uniform();
				kpntj[j] = Random::uniform();
			}
			
			//Calculate energies and filter kpoint pairs:
			diagMatrix Ei = bs.getStates(kpnti);
			diagMatrix Ej = bs.getStates(kpntj);
			bool worthwhileSingle = false, worthwhileDouble = false;
			for(int v=0; v<Ei.nRows(); v++) if(fabs(Ei[v])<10*T)
			{	worthwhileSingle = true;
				for(int c=0; c<Ej.nRows(); c++) if(fabs(Ej[c])<10*T)
				{	worthwhileDouble = true;
					break;
				}
			}
			if(!worthwhileSingle) continue;
			
			//Calculate remaining (more expensive) quantities for k-point pair:
			std::vector<vector3<>> vi = bs.getVelocity(kpnti, R);
			std::vector<vector3<>> vj;
			diagMatrix omegaPh;
			std::vector<matrix> MePh;
			if(worthwhileDouble)
			{	vj = bs.getVelocity(kpntj, R);
				omegaPh = bs.getPhononModes(kpnti-kpntj);
				MePh = bs.getPhononMatElem(kpnti,kpntj);
			}
			
			for(int v=0; v<Ei.nRows(); v++)
			{	double dFdEi = -1/(T*std::pow(2*cosh(Ei[v]/(2*T)),2));
				double viDotvi = vi[v].length_squared();
				Tblock += prefacT * viDotvi*(-dFdEi);
				gBlock += prefacT * (-dFdEi);
				if(!worthwhileDouble) continue;
				for(int c=0; c<Ej.nRows(); c++)
				{	double viDotvj = dot(vi[v], vj[c]);
					double fj = 1./(exp(Ej[c]/T)+1);
					for(int alpha=0; alpha<omegaPh.nRows(); alpha ++)
					{	double gk = 1./(exp(omegaPh[alpha]/T) - 1.);
						double Msq_by_omega = MePh[alpha](c,v).norm() / omegaPh[alpha];
						for(int ae=-1; ae<=+1; ae+=2)
						{	double delta = EconservePrefac * exp(EconserveExpFac * std::pow(Ej[c]-Ei[v] - ae*omegaPh[alpha],2));
							double occFactors = (-dFdEi) * (gk+0.5 - ae*(0.5-fj));
							GammaBlock += prefacGamma * (viDotvi -  viDotvj) * occFactors * delta * Msq_by_omega;
							tauInvBlock += prefacGamma * (-dFdEi) * (gk+0.5-ae*0.5) * delta * Msq_by_omega;
						}
					}
				}
			}
		}
		Tsum += Tblock; TsumSq += std::pow(Tblock,2);
		GammaSum += GammaBlock; GammaSumSq +=std::pow(GammaBlock,2);
		gSum += gBlock; gSumSq += std::pow(gBlock,2);
		tauInvSum += tauInvBlock; tauInvSumSq +=std::pow(tauInvBlock,2);
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
	
	const double invSeconds = 2.418884326505e-17;
	const double Coulomb = Joule/eV;
	const double Volt = Joule/Coulomb;
	const double Ampere = Coulomb*invSeconds;
	const double Ohm = Volt/Ampere;

	// Calculate Resistivity
	double rho = Omega*Gamma/(Tt*Tt);
	double rhoStd = rho * hypot(GammaStd/Gamma, sqrt(2.)*Tstd/Tt);
	logPrintf("Resistivity = %lg +/- %lg ohm-m\n", rho/(Ohm*meter), rhoStd/(Ohm*meter));
	
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
	double fs = 1e-15/invSeconds;
	logPrintf("tau = %lg +/- %lg fs\n", tau/fs, tauStd/fs);
	
	double vF = sqrt(Tt/gMean);
	double vFstd = vF * 0.5 * hypot(Tstd/Tt, gStd/gMean);
	logPrintf("vF = %lg +/- %lg\n", vF, vFstd);
	
	logPrintf("g(eF) = %lg +/- %lg\n", 3.*gMean, 3.*gStd);
	finalizeSystem();
}
