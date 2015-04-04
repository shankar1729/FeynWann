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
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE");
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);

	//Initalize line width of intermediate electronic states
	LineWidth lineWidth("Wannier/wannier", bs);

	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	int nBands = bs.getStates(vector3<>()).nRows();
	
	//Calculate average linewidths near Fermi level:
	double Esigma = 0.5*eV; //Use a Gaussian energy window near the Fermi level
	double gaussPrefac = 1./(Esigma*sqrt(2*M_PI)), gaussExpfac = -0.5/(Esigma*Esigma);
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	double eeNum = 0., eeDen = 0., ePhNum = 0., ePhDen = 0.;
	logPrintf("\nProgress: "); logFlush();
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{	//Generate a bunch of k-points:
		std::vector< vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		//Calculate electronic energies and lifetimes for bunch:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector<diagMatrix> eeImEarr  = lineWidth(kArr, 1., 0.);
		std::vector<diagMatrix> ePhImEarr = lineWidth(kArr, 0., 1.);
		//Calculate weighted sums:
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
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunchesMine)));
			logFlush();
		}
	}
	logPrintf("done.\n"); logFlush();
	//Collect averages:
	mpiUtil->allReduce(eeNum, MPIUtil::MPIUtil::ReduceSum);  mpiUtil->allReduce(eeDen, MPIUtil::MPIUtil::ReduceSum);
	mpiUtil->allReduce(ePhNum, MPIUtil::MPIUtil::ReduceSum); mpiUtil->allReduce(ePhDen, MPIUtil::MPIUtil::ReduceSum);
	double eeGamma0 = 2 * eeNum / eeDen; //factor of 2 to go from ImSigma to Gamma
	double ePhGamma0 = 2 * ePhNum / ePhDen;
	logPrintf("Weight sum: %lg\n", ePhDen); //should be >> 1 for a reliable average
	logPrintf("tau0_ee = %lg fs-eV^2\n", 1./(eeGamma0 * fs * eV*eV));
	logPrintf("tau0_ePh = %lg fs\n", 1./(ePhGamma0 * fs));
	
	//Output linewidth estimates:
	double Zjellium = (spinWeight==1 ? 1. : 3.); //HACK: currently only Al nonrelativistic and rest relativistic
	double nJellium = Zjellium / fabs(det(R));
	double omegaPsq = 4*M_PI*nJellium;
	double kF = pow(3*M_PI*M_PI*nJellium, 1./3);
	double gaussMargin = 5*T;
	if(mpiUtil->isHead())
	{	ofstream ofs("GammaAll-IntrabandEstimate.dat");
		for(double omega = gaussMargin; omega <= EplasmonMax-gaussMargin; omega += T)
		{	eps.setFrequency(omega, false);
			double prefac = eps.exptLinewidth()/eps.epsilon.imag(); //ratio between plasmon linewidth and imaginary part of epsilon
			double ePhImEps = omegaPsq * ePhGamma0 / (std::pow(omega, 3) * std::pow(4*Zjellium, 2./3));
			double eeImEps = 0.;
			ofs << omega/eV << '\t' << prefac*eeImEps/eV << '\t' << prefac*ePhImEps/eV << '\n';
		}
	}
	
	finalizeSystem();
}