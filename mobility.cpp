#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"
#include "Units.h"
#include "LineWidth.h"
#include "Histogram.h"

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of mobility", inputFilename, dryRun, printDefaults);

	//Read input file:
	InputMap inputMap(inputFilename);
	const int nKpts = inputMap.get("nKpts");
	const double T = inputMap.get("T") * Kelvin;

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKpts = %d\n", nKpts);
	logPrintf("T = %lg\n", T);
	
	//Initialize Wannier bandstructure:
	const int bunchSize = 32;
	BandStruct bs("Wannier/totalE", "Wannier/wannier", false);
	bs.setCacheSize(2*bunchSize);
	
	//Initalize line width of electronic states
	LineWidth lineWidth("Wannier/wannier", bs);

	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Energy grid:
	diagMatrix Egamma = bs.getStates(vector3<>());
	double Emin = Egamma.front(), Emax = Egamma.back();
	for(int i=0; i<10; i++)
	{	//Random block of kpoints:
		std::vector<vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		//Find energy range:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		for(const diagMatrix& E: Earr)
		{	Emin = std::min(Emin, E.front());
			Emax = std::max(Emax, E.back());
		}
	}
	mpiUtil->allReduce(Emin, MPIUtil::ReduceMin);
	mpiUtil->allReduce(Emax, MPIUtil::ReduceMax);
	double dE = 0.1*T;
	
	//Collect mobility integrand on energy grid:
	if(bs.nValence >= bs.nBands)
		die("Could not find Wannier bands for empty states: needed to calculate electron mobility.\n");
	logPrintf("Collectinging mobility integrands ... "); logFlush();
	int nBunches = nKpts/(bunchSize*mpiUtil->nProcesses());
	int iBunchInterval = std::max(1, int(round(nBunches/50.))); //interval for reporting progress
	Histogram vSqTau(Emin, dE, Emax), g(Emin, dE, Emax); //collect v^2*tau and DOS by energy
	double EvMax = -DBL_MAX, EcMin = +DBL_MAX; //band edges
	for(int iBunch=0; iBunch<nBunches; iBunch++)
	{	//Random block of kpoints:
		std::vector< vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		//Calculate properties for block:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector<diagMatrix> ImSigma_ePhArr = lineWidth(kArr, 0., 1.); //e-ph linewidth
		for(int ik=0; ik<bunchSize; ik++)
		{	const diagMatrix& E = Earr[ik];
			std::vector<vector3<>> v = bs.getVelocity(kArr[ik]);
			const diagMatrix& ImSigma_ePh = ImSigma_ePhArr[ik];
			//Update band edges:
			EvMax = std::max(EvMax, E[bs.nValence-1]);
			EcMin = std::min(EcMin, E[bs.nValence]);
			//Collect vSqTau and DOS:
			for(int b=0; b<bs.nBands; b++)
			{	double tau_ePh = 0.5/ImSigma_ePh[b];
				vSqTau.addEvent(E[b], v[b].length_squared() * tau_ePh);
				g.addEvent(E[b], 1.);
			}
		}
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunches)));
			logFlush();
		}
	}
	mpiUtil->allReduce(EvMax, MPIUtil::ReduceMax);
	mpiUtil->allReduce(EcMin, MPIUtil::ReduceMin);
	vSqTau.allReduce(MPIUtil::ReduceSum);
	g.allReduce(MPIUtil::ReduceSum);
	logPrintf("done.\n\n"); logFlush();
	logPrintf("Band edges:  EvMax: %lg  EcMin: %lg\n\n", EvMax, EcMin);
	
	//Calculate and report mobilities:
	double hMobilityNum = 0., hMobilityDen = 0.;
	double eMobilityNum = 0., eMobilityDen = 0.;
	for(size_t ie=0; ie<g.out.size(); ie++)
	{	double E = Emin + ie*dE;
		if(E<EvMax) //hole:
		{	double denWeight = exp((E-EvMax)/T); //limit of (1-f) with scale factor
			double numWeight = (1./T)*denWeight; //limit of -(1-f)' with scale factor
			hMobilityNum += numWeight * vSqTau.out[ie] * dE;
			hMobilityDen += denWeight * g.out[ie] * dE;
		}
		if(E>EcMin) //electron:
		{	double denWeight = exp((EcMin-E)/T); //limit of f with scale factor
			double numWeight = (1./T)*denWeight; //limit of -f' with scale factor
			eMobilityNum += numWeight * vSqTau.out[ie] * dE;
			eMobilityDen += denWeight * g.out[ie] * dE;
		}
	}
	double mobUnit = std::pow(1e-2*meter,2)*invSeconds/Volt;
	logPrintf("hMobility = %lg cm^2/(V.s)\n", (hMobilityNum/hMobilityDen)/mobUnit);
	logPrintf("eMobility = %lg cm^2/(V.s)\n", (eMobilityNum/eMobilityDen)/mobUnit);

	finalizeSystem();
}
