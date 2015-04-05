#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include "BandStruct.h"
#include "LineWidth.h"
#include "Epsilon.h"
#include "InputMap.h"
#include "Units.h"
#include "Histogram.h"

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Calibrated jellium estimates of intraband processes", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const double EplasmonMax = inputMap.get("EplasmonMax") * eV;
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("EplasmonMax = %lg\n", EplasmonMax);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Initialize dielectric model:
	Epsilon eps("Wannier/epsilon.dat");

	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > AhatArr(1, vector3<complex>(1,0,0));
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE", AhatArr);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);

	//Initalize line width of intermediate electronic states
	LineWidth lineWidth("Wannier/wannier", bs);

	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	long nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	int nBands = bs.getStates(vector3<>()).nRows();
	
	//Initialize histograms:
	double gaussMargin = 5*T;
	Histogram ImEpsPhonon(gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram ImEps2eh(gaussMargin, T, EplasmonMax-gaussMargin);

	//Calculate average linewidths near Fermi level:
	double Esigma = 0.5*eV; //Use a Gaussian energy window near the Fermi level
	double gaussPrefac = 1./(Esigma*sqrt(2*M_PI)), gaussExpfac = -0.5/(Esigma*Esigma);
	double EconserveScaleFac = 1./T, EconservePrefac = 1./(M_PI*T); //energy conserving Lorentzian parameters
	double ImEpsPrefac = (2*M_PI*spinWeight)/(nKpts * fabs(det(R)));
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	double eeNum = 0., eeNumSim = 0., eeDen = 0., ePhNum = 0., ePhNumSim = 0., ePhDen = 0.;
	logPrintf("\nProgress: "); logFlush();
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{	//Generate a bunch of k-points:
		std::vector< vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		//Calculate electronic energies, fillings, momentum matrix elements and lifetimes for bunch:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector<std::vector<matrix>> Parr = bs.getDipoleMatElem(kArr);
		std::vector<diagMatrix> eeImEarr  = lineWidth(kArr, 1., 0.);
		std::vector<diagMatrix> ePhImEarr = lineWidth(kArr, 0., 1.);
		std::vector<diagMatrix> Farr = Earr; //convert to fillings:
		for(diagMatrix& F: Farr)
			for(double& f: F)
			{	double e = f/T; //E/T actually
				f = (e>30 ? exp(-e) : 1./(1. + exp(e))); //avoid overflow issues
			}
		//Calculate weighted sums of lifetimes:
		for(int ik=0; ik<bunchSize; ik++)
		{	const diagMatrix& E = Earr[ik];
			const diagMatrix& eeImE  = eeImEarr[ik];
			const diagMatrix& ePhImE = ePhImEarr[ik];
			for(int b=0; b<nBands; b++)
			{	double EbSq = E[b]*E[b];
				double w = gaussPrefac * exp(gaussExpfac*EbSq);
				eeNum += w * eeImE[b];
				eeDen += w * EbSq;
				ePhNum += w * ePhImE[b];
				ePhDen += w;
			}
		}
		//Calculate sums for intraband estimates:
		for(int ik1=0; ik1<bunchSize; ik1++)
		{	const diagMatrix& E1 = Earr[ik1];
			const diagMatrix& F1 = Farr[ik1];
			const matrix& AdotP1 = Parr[ik1][0];
			//Extra k-points for e-e process:
			std::vector< vector3<> > kArrPrime(bunchSize);
			vector3<>& k3 = kArrPrime[ik1];
			for(int j=0; j<3; j++)
				k3[j] = Random::uniform();
			for(int ik2=0; ik2<bunchSize; ik2++) if(ik2!=ik1)
				kArrPrime[ik2] = kArr[ik2] - (kArr[ik1] + k3);
			std::vector<diagMatrix> EarrPrime = bs.getStates(kArrPrime);
			std::vector<std::vector<matrix>> ParrPrime = bs.getDipoleMatElem(kArrPrime);
			std::vector<diagMatrix> FarrPrime = EarrPrime; //convert to fillings:
			for(diagMatrix& F: FarrPrime)
				for(double& f: F)
				{	double e = f/T; //E/T actually
					f = (e>30 ? exp(-e) : 1./(1. + exp(e))); //avoid overflow issues
				}
			const diagMatrix& E3 = EarrPrime[ik1];
			const diagMatrix& F3 = FarrPrime[ik1];
			const matrix& AdotP3 = ParrPrime[ik1][0];
			//Loop over second k-point:
			for(int ik2=0; ik2<bunchSize; ik2++) if(ik2!=ik1) //avoid gamma-point phonon singularity
			{	const diagMatrix& E2 = Earr[ik2];
				const diagMatrix& F2 = Farr[ik2];
				const matrix& AdotP2 = Parr[ik2][0];
				//
				const diagMatrix& E4 = EarrPrime[ik2];
				const diagMatrix& F4 = FarrPrime[ik2];
				const matrix& AdotP4 = ParrPrime[ik2][0];
				//
				for(int b1=0; b1<nBands; b1++)
				{	double w = gaussPrefac * exp(gaussExpfac*E1[b1]*E1[b1]);
					//Phonon-assisted:
					for(int b2=0; b2<nBands; b2++)
					{	//Phonon-assisted contribution:
						double omega = E2[b2] - E1[b1]; //energy conservation (neglecting phonon energy)
						if(omega > 0 && omega < EplasmonMax)
						{	double weight = ImEpsPrefac * (F1[b1]-F2[b2]) * (AdotP1(b1,b1)-AdotP2(b2,b2)).norm();
							ImEpsPhonon.addEvent(omega, weight);
						}
						//e-ph scattering constribution:
						double delta = EconservePrefac / (1. + std::pow(EconserveScaleFac*omega,2));
						ePhNumSim += w * delta;
					}
					//Two e-h pair:
					for(int b2=0; b2<nBands; b2++)
					for(int b3=0; b3<nBands; b3++)
					for(int b4=0; b4<nBands; b4++)
					{	//Two e-h pair contribution:
						double omega = E2[b2] + E4[b4] - E1[b1] - E3[b3];
						double Fprod1 = F1[b1] * (1.-F2[b2]) * F3[b3] * (1.-F4[b4]);
						double Fprod2 = (1.-F1[b1]) * F2[b2] * (1.-F3[b3]) * F4[b4];
						if(omega > 0 && omega < EplasmonMax)
						{	double weight = ImEpsPrefac * (Fprod1 - Fprod2) * (AdotP1(b1,b1) - AdotP2(b2,b2) + AdotP3(b3,b3) - AdotP4(b4,b4)).norm();
							ImEps2eh.addEvent(omega, weight);
						}
						//e-e scattering constribution:
						double delta = EconservePrefac / (1. + std::pow(EconserveScaleFac*omega,2));
						eeNumSim += w * delta * (Fprod1 + Fprod2);
					}
				}
			}
		}
		
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunchesMine)));
			logFlush();
		}
	}
	logPrintf("done.\n"); logFlush();
	//Collect averages:
	mpiUtil->allReduce(eeNum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(eeDen, MPIUtil::ReduceSum);
	mpiUtil->allReduce(ePhNum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(ePhDen, MPIUtil::ReduceSum);
	double eeGamma0 = 2 * eeNum / eeDen; //factor of 2 to go from ImSigma to Gamma
	double ePhGamma0 = 2 * ePhNum / ePhDen;
	logPrintf("Weight sum: %lg\n", ePhDen); //should be >> 1 for a reliable average
	logPrintf("tau0_ee = %lg fs-eV^2\n", 1./(eeGamma0 * fs * eV*eV));
	logPrintf("tau0_ePh = %lg fs\n", 1./(ePhGamma0 * fs));
	
	//Output plasmon Gamma contributions from ImEps contributions:
	ImEpsPhonon.allReduce(MPIUtil::ReduceSum);
	ImEps2eh.allReduce(MPIUtil::ReduceSum);
	mpiUtil->allReduce(eeNumSim, MPIUtil::ReduceSum);
	mpiUtil->allReduce(ePhNumSim, MPIUtil::ReduceSum);
	if(mpiUtil->isHead())
	{	ofstream ofs("GammaAll-IntrabandEstimate.dat");
		for(size_t i=0; i<ImEpsPhonon.out.size(); i++)
		{	double omega = ImEpsPhonon.Emin + i * ImEpsPhonon.dE;
			eps.setFrequency(omega, false);
			double prefac = eps.exptLinewidth()/eps.epsilon.imag(); //ratio between plasmon linewidth and imaginary part of epsilon
			double eeImEps = ImEps2eh.out[i] * eeGamma0 / (std::pow(omega, 4) * eeNumSim / eeDen);
			double ePhImEps = ImEpsPhonon.out[i] * ePhGamma0 / (std::pow(omega, 4) * ePhNumSim / ePhDen);
			ofs << omega/eV << '\t' << prefac*eeImEps/eV << '\t' << prefac*ePhImEps/eV << '\n';
		}
	}
	
	finalizeSystem();
}