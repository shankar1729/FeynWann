#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"
#include <core/Units.h>
#include "LineWidth.h"
#include "Histogram.h"

int main(int argc, char** argv)
{	
	InitParams ip = BandStruct::initialize(argc, argv, "Monte Carlo estimate of mobility");

	//Read input file:
	InputMap inputMap(ip.inputFilename);
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

	if(ip.dryRun)
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
	mpiWorld->allReduce(Emin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(Emax, MPIUtil::ReduceMax);
	double dE = 0.01*T;
	
	//Collect mobility integrand on energy grid:
	if(bs.nValence >= bs.nBands)
		die("Could not find Wannier bands for empty states: needed to calculate electron mobility.\n");
	logPrintf("Collecting mobility integrands ... "); logFlush();
	int nBunches = nKpts/(bunchSize*mpiWorld->nProcesses());
	int iBunchInterval = std::max(1, int(round(nBunches/50.))); //interval for reporting progress
	Histogram vSqTauBy3(Emin, dE, Emax), vSq(Emin, dE, Emax), tau(Emin, dE, Emax), g(Emin, dE, Emax); //collect v^2*tau/3 and DOS by energy
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
			//Collect vSqTauBy3 and DOS:
			for(int b=0; b<bs.nBands; b++)
			{	double tau_ePh = 0.5/ImSigma_ePh[b];
				vSqTauBy3.addEvent(E[b], (1./3) * v[b].length_squared() * tau_ePh);
				vSq.addEvent(E[b],  v[b].length_squared());
				tau.addEvent(E[b],  tau_ePh);
				g.addEvent(E[b], 1.);
			}
		}
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunches)));
			logFlush();
		}
	}
	mpiWorld->allReduce(EvMax, MPIUtil::ReduceMax);
	mpiWorld->allReduce(EcMin, MPIUtil::ReduceMin);
	vSqTauBy3.allReduce(MPIUtil::ReduceSum);
	vSq.allReduce(MPIUtil::ReduceSum);
	tau.allReduce(MPIUtil::ReduceSum);
	g.allReduce(MPIUtil::ReduceSum);
	logPrintf("done.\n\n"); logFlush();
	logPrintf("Band edges:  EvMax: %lg  EcMin: %lg\n\n", EvMax, EcMin);
	
	//Calculate and report mobilities:
	double hMobNum = 0., hVsqNum = 0., hTauNum = 0., hDen = 0.;
	double eMobNum = 0., eVsqNum = 0., eTauNum = 0., eDen = 0.;
	for(size_t ie=0; ie<g.out.size(); ie++)
	{	double E = Emin + ie*dE;
		if(E<=EvMax) //hole:
		{	double denWeight = exp((E-EvMax)/T); //limit of (1-f) with scale factor
			double numWeight = (1./T)*denWeight; //limit of -(1-f)' with scale factor
			hMobNum += numWeight * vSqTauBy3.out[ie] * dE;
			hVsqNum += denWeight * vSq.out[ie] * dE;
			hTauNum += denWeight * tau.out[ie] * dE;
			hDen += denWeight * g.out[ie] * dE;
		}
		if(E>=EcMin) //electron:
		{	double denWeight = exp((EcMin-E)/T); //limit of f with scale factor
			double numWeight = (1./T)*denWeight; //limit of -f' with scale factor
			eMobNum += numWeight * vSqTauBy3.out[ie] * dE;
			eVsqNum += denWeight * vSq.out[ie] * dE;
			eTauNum += denWeight * tau.out[ie] * dE;
			eDen += denWeight * g.out[ie] * dE;
		}
	}
	double mobUnit = std::pow(1e-2*meter,2)/(Volt*sec);
	logPrintf("hMobility = %lg cm^2/(V.s)\n", (hMobNum/hDen)/mobUnit);
	logPrintf("eMobility = %lg cm^2/(V.s)\n", (eMobNum/eDen)/mobUnit);
	logPrintf("vF_h = %lg\n", sqrt(hVsqNum/hDen));
	logPrintf("vF_e = %lg\n", sqrt(eVsqNum/eDen));
	logPrintf("tau_h = %lg fs\n", (hTauNum/hDen)/fs);
	logPrintf("tau_e = %lg fs\n", (eTauNum/eDen)/fs);
	logPrintf("tauDrude_h = %lg fs\n", (3.*T*hMobNum/hVsqNum)/fs);
	logPrintf("tauDrude_e = %lg fs\n", (3.*T*eMobNum/eVsqNum)/fs);
	
	finalizeSystem();
}
