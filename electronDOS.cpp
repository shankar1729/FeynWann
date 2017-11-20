#include "WannierMC.h"
#include "InputMap.h"
#include "Histogram.h"
#include <core/Units.h>
#include <core/Random.h>

//Get energy range from an eLoop call:
struct EnergyRange
{	double Emin;
	double Emax;
	
	static void eProcess(const WannierMC::StateE& state, void* params)
	{	EnergyRange& er = *((EnergyRange*)params);
		er.Emin = std::min(er.Emin, state.E.front()); //E is in ascending order
		er.Emax = std::max(er.Emax, state.E.back()); //E is in ascending order
	}
};

struct CollectDOS
{	Histogram* dos;
	double weight;
	
	static void eProcess(const WannierMC::StateE& state, void* params)
	{	CollectDOS& cd = *((CollectDOS*)params);
		for(const double& Ei: state.E)
			cd.dos->addEvent(Ei, cd.weight);
	}
};

inline double fermi(double x) { return x>30. ? exp(-x) : 1./(1.+exp(x)); } //avoid overflow issues
inline double fermiPrime(double x) { return 0.25*(std::pow(tanh(0.5*x), 2) - 1.); } //avoid overflow issues

int main(int argc, char** argv)
{	InitParams ip = WannierMC::initialize(argc, argv, "Electronic DOS and heat capacity");
	
	InputMap inputMap(ip.inputFilename);
	size_t nOffsets = inputMap.get("nOffsets");
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	const double Tmin = inputMap.get("Tmin") * Kelvin; //electron temperature grid start
	const double Tmax = inputMap.get("Tmax") * Kelvin; //electron temperature grid stop
	const double Tstep = inputMap.get("Tstep") * Kelvin; //electron temperature grid spacing

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %lu\n", nOffsets);
	logPrintf("dE = %lg\n", dE);
	logPrintf("Tmin = %lg\n", Tmin);
	logPrintf("Tmax = %lg\n", Tmax);
	logPrintf("Tstep = %lg\n", Tstep);

	//Initialize Wannier bandstructure:
	WannierMCParams wmcp; //default parametres suffice
	WannierMC wmc(wmcp);
	size_t nKpts = nOffsets * wmc.eCountPerOffset();  
	logPrintf("Effectively sampled nKpts: %lu\n", nKpts);
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		wmc.free();
		WannierMC::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Initialize temperature grid:
	std::vector<double> Tarr(int(ceil((Tmax-Tmin)/Tstep)));
	for(size_t iT=0; iT<Tarr.size(); iT++)
		Tarr[iT] = Tmin + Tstep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", Tarr.front()/Kelvin, Tarr.back()/Kelvin, Tarr.size());
	
	//Initialize energy grid:
	EnergyRange er = { DBL_MAX, -DBL_MAX };
	wmc.eLoop(vector3<>(), EnergyRange::eProcess, &er);
	mpiWorld->allReduce(er.Emin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(er.Emax, MPIUtil::ReduceMax);
	er.Emin -= 10*dE; //add some margin
	er.Emax += 10*dE;
	Histogram dos(er.Emin, dE, er.Emax); //density of states
	logPrintf("Initialized energy grid: %lg to %lg eV with %lu points.\n", er.Emin/eV, (er.Emin+dE*(dos.out.size()-1))/eV, dos.out.size());
	
	//Initialize sampling parameters:
	int oStart=0, oStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsets, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	
	logPrintf("\nCollecting DOS: "); logFlush();
	CollectDOS cd;
	cd.dos = &dos;
	cd.weight = wmc.spinWeight*(1./nKpts);
	for(int o=0; o<noMine; o++)
	{	Random::seed(o+oStart); //to make results independent of MPI division
		//Process with a random offset:
		vector3<> k0 = wmc.randomVector(mpiGroup); //must be constant across group
		wmc.eLoop(k0, CollectDOS::eProcess, &cd);
		//Print progress:
		if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
	}
	logPrintf("done.\n"); logFlush();
	dos.allReduce(MPIUtil::ReduceSum);
	dos.print("eDOS.dat", 1./eV, eV);
	
	//Calculate mu and Ce at each temperature:
	diagMatrix dmu(Tarr.size(), 0.), Ce(Tarr.size(), 0.);
	//--- check enough bands to contain Z:
	double Zmax = 0.;
	for(const double& g: dos.out)
		Zmax += dE * g;
	if(Zmax < wmc.nElectrons)
		die("Current DOS can only support %lg electrons > %lg electrons specified.\n", Zmax, wmc.nElectrons);
	int iTstart, iTstop; TaskDivision(Tarr.size(), mpiWorld).myRange(iTstart, iTstop);
	for(int iT=iTstart; iT<iTstop; iT++)
	{	const double T = Tarr[iT], invT = 1./T;
		//Bisect for chemical potential:
		double& dmuCur = dmu[iT];
		double dmuMin = er.Emin - 10*T;
		double dmuMax = er.Emax + 10*T;
		dmuCur = 0.5*(dmuMin + dmuMax);
		const double tol = 1e-9*T;
		while(dmuMax-dmuMin > tol)
		{	//calculate number of electrons at current Z:
			double nElectrons = 0.;
			for(size_t ie=0; ie<dos.out.size(); ie++)
			{	double Ei = er.Emin + ie*dE;
				double fi = fermi(invT*(Ei - dmuCur));
				nElectrons += dE * dos.out[ie] * fi;
			}
			((nElectrons>wmc.nElectrons) ? dmuMax : dmuMin) = dmuCur;
			dmuCur = 0.5*(dmuMin + dmuMax);
		}
		//Calculate electronic specific heat:
		double& CeCur = Ce[iT];
		CeCur = 0.;
		for(size_t ie=0; ie<dos.out.size(); ie++)
		{	double Ei = er.Emin + ie*dE;
			double x = invT*(Ei-dmuCur);
			double dfdT = fermiPrime(x) * (-x*invT);
			CeCur += dE * Ei * dos.out[ie] * dfdT;
		}
	}
	dmu.allReduce(MPIUtil::ReduceSum);
	Ce.allReduce(MPIUtil::ReduceSum);
	
	//Write to file
	if(mpiWorld->isHead())
	{	const double CeSI = Joule/(Kelvin*pow(meter,3));
		ofstream ofs("Ce.dat");
		ofs << "#T[K] Ce[J/m^3K] dmu[eV]\n";
		for(size_t iT=0; iT<Tarr.size(); iT++)
			ofs << Tarr[iT]/Kelvin << '\t'
				<< Ce[iT]/(wmc.Omega*CeSI) << '\t'
				<< dmu[iT]/eV << '\n';
	}
	
	wmc.free();
	WannierMC::finalize();
}
