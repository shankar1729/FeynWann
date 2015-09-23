#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "InputMap.h"
#include "Units.h"

inline double fermi(double x) { return x>30. ? exp(-x) : 1./(1.+exp(x)); } //avoid overflow issues
inline double argLW(double E,double Es) { return sqrt(E)/(E+Es) + atan(sqrt(E/Es))/sqrt(Es); }

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Calculation of q_TF and E_S from Vallee paper", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const double Zjellium = inputMap.get("Zjellium");
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	const double beta = inputMap.get("beta"); // from vallee paper, q_s = beta * q_TF
	const double epsB = inputMap.get("epsilonB"); // epsilon_b, as defined in Vallee paper
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
//	const double Es = inputMap.get("Es"); //from fitting
	const double TeMin = inputMap.get("TeMin") * Kelvin; //electron temperature grid start
	const double TeMax = inputMap.get("TeMax") * Kelvin; //electron temperature grid stop
	const double TeStep = inputMap.get("TeStep") * Kelvin; //electron temperature grid spacing

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("Zjellium = %lg\n", Zjellium);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	logPrintf("beta = %lg\n", beta);
 	logPrintf("dE = %lg\n", dE);
	logPrintf("epsilonB = %lg\n", epsB);
	logPrintf("TeMin = %lg\n", TeMin);
	logPrintf("TeMax = %lg\n", TeMax);
	logPrintf("TeStep = %lg\n", TeStep);

	const double Omega = fabs(det(R)); //unit cell volume
	const double nJellium = Zjellium / Omega;
	logPrintf("nJellium = %lg per bohr^3, %lg per m^3\n", nJellium, nJellium*meter*meter*meter);
	const double kF = std::pow(3 * M_PI * M_PI * nJellium,1./3);
	const double Ef = 0.5 * kF * kF;
	logPrintf("EfJellium = %lf hartrees, %lg eV, %lg Joules\n",Ef, Ef/eV, Ef/Joule);

	double qTF = (4*M_PI/epsB) * sqrt(kF/(M_PI*M_PI));
	double qs = beta * qTF;
	double Es = qs*qs/2;
	logPrintf("Es = %lg\n", Es);

	//Initialize temperature grid:
	std::vector<double> TeArr(int(ceil((TeMax-TeMin)/TeStep)));
	for(size_t iT=0; iT<TeArr.size(); iT++)
		TeArr[iT] = TeMin + TeStep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", TeArr.front()/Kelvin, TeArr.back()/Kelvin, TeArr.size());

	// ============================ calculate mu(Te) ========================================================
	diagMatrix dmu(TeArr.size(), 0.);
	int iTstart, iTstop; TaskDivision(TeArr.size(), mpiUtil).myRange(iTstart, iTstop);
	for(int iT=iTstart; iT<iTstop; iT++)
	{	const double Te = TeArr[iT], invTe = 1./Te;
		//Initialize energy grid:
		double Emin = 0;
		double Emax = Ef + 5*Te;
		double dE = 0.25*Te; //different from dE for carruer energies used later
		//Bisect for chemical potential:
		double& dmuCur = dmu[iT];
		double dmuMin = Emin - 10*Te;
		double dmuMax = Emax + 10*Te;
		dmuCur = 0.5*(dmuMin + dmuMax);
		const double tol = 1e-9*Te;
		while(dmuMax-dmuMin > tol)
		{	//calculate number of electrons at current Z:
			double nElectrons = 0.;
			for(double E=Emin; E<Emax; E+=dE)
			{	double f = fermi(invTe*(E - dmuCur));
				nElectrons += dE * f * sqrt(2*E)*Omega/(M_PI*M_PI);
			}
			((nElectrons>Zjellium) ? dmuMax : dmuMin) = dmuCur;
			dmuCur = 0.5*(dmuMin + dmuMax);
		}
	}
	dmu.allReduce(MPIUtil::ReduceSum);

	if(mpiUtil->isHead())
	{	ofstream ofs("mu_Te_Jellium.dat");
		ofs << "#T[K] dmu[eV] \n";
		for(size_t iT=0; iT<TeArr.size(); iT++)
			ofs << TeArr[iT]/Kelvin << '\t' << dmu[iT]/eV << '\n';
	}




	// ============================ calculate tau(Te) =======================================================
	//Initialize energy grid:
	std::vector<double> EArr(int(ceil(2*eV/dE)));
	for(size_t iE=0; iE<EArr.size(); iE++)
		EArr[iE] = Ef - 1*eV + dE*iE;

	std::vector< std::vector<double> > invTauTe(TeArr.size());
	
	for(size_t iT=0; iT<TeArr.size(); iT++)
	{	std::vector<double> invTauTeE(EArr.size());
		double T  = TeArr[iT];
		double invT = 1./T;
		double dE12 = 0.25*T;
		double E12min = std::max(0.5*dE12, EArr.front() - 10*T); //no states below E=0
		double E12max = EArr.back() + 10*T;
		logPrintf("E12 in [ %lg , %lg ] for Te = %lg K\n", E12min, E12max, T/Kelvin);
		for(size_t iE=0; iE<EArr.size(); iE++)
		{	double E = EArr[iE];
			double lPrefac = 1. / (32 * std::pow(M_PI,3) * std::pow(epsB/(4*M_PI),2) * Es * sqrt(E));
			for(double E1=E12min; E1<E12max; E1+=dE12)
			{	double f1 = fermi(invT*(E1 - dmu[iT]));
				for(double E2=E12min; E2<E12max; E2+=dE12)
				{	double f2 = fermi(invT*(E2 - dmu[iT]));
					double E3 = E + E1 - E2;
					if(E3 <= 0.) continue;
					double f3 = fermi(invT*(E3 - dmu[iT]));
					double occFactor = f1*(1-f2)*(1-f3) + (1-f1)*f2*f3;
					if(occFactor < 1e-6) continue;
					double EtildeMax = std::min(std::pow(sqrt(E1)+sqrt(E3),2),std::pow(sqrt(E)+sqrt(E2),2));
					double EtildeMin = std::max(std::pow(sqrt(E1)-sqrt(E3),2),std::pow(sqrt(E)-sqrt(E2),2));
					double arg = argLW(EtildeMax,Es) - argLW(EtildeMin,Es);
					invTauTeE[iE] += lPrefac*arg*occFactor*dE12*dE12;
				}
			}
			invTauTe[iT].push_back(invTauTeE[iE]);
		}
	}


	ofstream ofs("invTauTe.dat");
	ofs << 0;
	for(size_t iE=0; iE<EArr.size(); iE++)
		ofs << '\t' <<EArr[iE]/eV;;
	ofs << '\n';
	for(size_t iT=0; iT<TeArr.size(); iT++)
	{	ofs << TeArr[iT]/Kelvin;
		for(size_t iE=0; iE<EArr.size(); iE++)
			ofs << '\t' << invTauTe[iT][iE]/invSeconds;
		ofs << '\n';
	}

	ofstream of("invTauT0.dat");
	for(size_t iE=0; iE<EArr.size(); iE++)
	{	double E = EArr[iE];
		double invTauT0 = std::pow(E-Ef, 2) / (64*std::pow(M_PI,3)*std::pow(epsB/(4*M_PI),2)*std::pow(Es,1.5)*sqrt(Ef))
			* (2.*sqrt(Ef*Es)/(4*Ef+Es) + atan(sqrt(4*Ef/Es)));
		of << E/eV << '\t' << invTauT0/invSeconds << '\n';
	}

	finalizeSystem();
}
