#include <core/Util.h>
#include <core/Units.h>
#include <core/Random.h>
#include "BandStruct.h"
#include "InputMap.h"
#include "Histogram.h"

inline double fermi(double x) { return x>30. ? exp(-x) : 1./(1.+exp(x)); } //avoid overflow issues
inline double fermiPrime(double x) { return 0.25*(std::pow(tanh(0.5*x), 2) - 1.); } //avoid overflow issues

int main(int argc, char** argv)
{	
	InitParams ip = BandStruct::initialize(argc, argv, "Ab initio parameters for Transient Absorption analysis");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	long nKpts = inputMap.get("nKpts");
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	const double Tmin = inputMap.get("Tmin") * Kelvin; //electron temperature grid start
	const double Tmax = inputMap.get("Tmax") * Kelvin; //electron temperature grid stop
	const double Tstep = inputMap.get("Tstep") * Kelvin; //electron temperature grid spacing

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKpts = %ld\n", nKpts);
	logPrintf("dE = %lg\n", dE);
	logPrintf("Tmin = %lg\n", Tmin);
	logPrintf("Tmax = %lg\n", Tmax);
	logPrintf("Tstep = %lg\n", Tstep);
	
	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(1); //assume cubic symmetry and only calculate x-axis
	Ahat[0] = vector3<complex>(1., 0., 0.);
	BandStruct bs("Wannier/totalE", "Wannier/wannier", true, Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);

	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");
	
	//Initialize temperature grid:
	std::vector<double> Tarr(int(ceil((Tmax-Tmin)/Tstep)));
	for(size_t iT=0; iT<Tarr.size(); iT++)
		Tarr[iT] = Tmin + Tstep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", Tarr.front()/Kelvin, Tarr.back()/Kelvin, Tarr.size());
	
	//Initialize energy grid:
	diagMatrix Egamma = bs.getStates(vector3<>());
	double Emin = Egamma.front(), Emax = Egamma.back(); //eigenvalues are sorted
	for(int i=0; i<10; i++)
	{	std::vector<vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		for(const diagMatrix& E: Earr)
		{	Emin = std::min(Emin, E.front());
			Emax = std::max(Emax, E.back());
		}
	}
	mpiWorld->allReduce(Emin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(Emax, MPIUtil::ReduceMax);
	Emin -= 10*dE; //add some margin
	Emax += 10*dE;
	Histogram dos(Emin, dE, Emax); //density of states
	logPrintf("Initialized energy grid: %lg to %lg eV with %lu points.\n", Emin/eV, (Emin+dE*(dos.out.size()-1))/eV, dos.out.size());
	
	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKpts, mpiWorld).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	nKpts = nBunchesMine * bunchSize; mpiWorld->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	
	//-------- Pass 1: collect density of states, calculate mu(T) and Ce(T) ---------
	
	logPrintf("\nCollecting DOS: "); logFlush();
	std::vector< std::vector< vector3<> > > kArrArr(nBunchesMine); //use exact same set of MC k-points in the two passes for consistency
	const double dosWeight = bs.spinWeight*(1./nKpts);
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{
		//Generate a bunch of k-points:
		std::vector< vector3<> >& kArr = kArrArr[iBunch];
		kArr.resize(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		
		//Collect DOS:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		for(const diagMatrix& E: Earr)
			for(const double& Ei: E)
				dos.addEvent(Ei, dosWeight);
		
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunchesMine)));
			logFlush();
		}
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
	if(Zmax < bs.nElectrons)
		die("Current DOS can only support %lg electrons > %lg electrons specified.\n", Zmax, bs.nElectrons);
	int iTstart, iTstop; TaskDivision(Tarr.size(), mpiWorld).myRange(iTstart, iTstop);
	for(int iT=iTstart; iT<iTstop; iT++)
	{	const double T = Tarr[iT], invT = 1./T;
		//Bisect for chemical potential:
		double& dmuCur = dmu[iT];
		double dmuMin = Emin - 10*T;
		double dmuMax = Emax + 10*T;
		dmuCur = 0.5*(dmuMin + dmuMax);
		const double tol = 1e-9*T;
		while(dmuMax-dmuMin > tol)
		{	//calculate number of electrons at current Z:
			double nElectrons = 0.;
			for(size_t ie=0; ie<dos.out.size(); ie++)
			{	double Ei = Emin + ie*dE;
				double fi = fermi(invT*(Ei - dmuCur));
				nElectrons += dE * dos.out[ie] * fi;
			}
			((nElectrons>bs.nElectrons) ? dmuMax : dmuMin) = dmuCur;
			dmuCur = 0.5*(dmuMin + dmuMax);
		}
		//Calculate electronic specific heat:
		double& CeCur = Ce[iT];
		CeCur = 0.;
		for(size_t ie=0; ie<dos.out.size(); ie++)
		{	double Ei = Emin + ie*dE;
			double x = invT*(Ei-dmuCur);
			double dfdT = fermiPrime(x) * (-x*invT);
			CeCur += dE * Ei * dos.out[ie] * dfdT;
		}
	}
	dmu.allReduce(MPIUtil::ReduceSum);
	Ce.allReduce(MPIUtil::ReduceSum);
	
	//Write to file
	if(mpiWorld->isHead())
	{	const double Omega = fabs(det(bs.R));
		const double CeSI = Joule/(Kelvin*pow(meter,3));
		ofstream ofs("Ce.dat");
		ofs << "#T[K] Ce[J/m^3K] dmu[eV]\n";
		for(size_t iT=0; iT<Tarr.size(); iT++)
			ofs << Tarr[iT]/Kelvin << '\t'
				<< Ce[iT]/(Omega*CeSI) << '\t'
				<< dmu[iT]/eV << '\n';
	}
	
	finalizeSystem();
};
