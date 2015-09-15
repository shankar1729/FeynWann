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
	const double T = inputMap.get("T")*eV;
	const double Zjellium = inputMap.get("Zjellium");
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	const double beta = inputMap.get("beta"); // from vallee paper, q_s = beta * q_TF
	const double epsB = inputMap.get("epsilonB"); // epsilon_b, as defined in Vallee paper
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("T = %lg\n", T);
	logPrintf("Zjellium = %lg\n", Zjellium);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	logPrintf("beta = %lg\n", beta);
 	logPrintf("dE = %lg\n", dE);
	logPrintf("epsilonB = %lg\n", epsB);

	const double nJellium = Zjellium / fabs(det(R));
	logPrintf("nJellium = %lg per bohr^3, %lg per m^3\n", nJellium, nJellium*meter*meter*meter);
	const double kF = std::pow(3 * M_PI * M_PI * nJellium,1./3);
	const double Ef = 0.5 * kF * kF;
	logPrintf("EfJellium = %lf hartrees, %lg eV, %lg Joules\n",Ef, Ef/eV, Ef/Joule);

	double qTF = (4*M_PI/epsB) * sqrt(kF/(M_PI*M_PI));
	double qs = beta * qTF;
	double Es = qs*qs/2;
	logPrintf("Es = %lg\n", Es);

	//Initialize energy grid:
	std::vector<double> EArr(int(ceil((2*eV+10*T)/dE)));
	for(size_t iE=0; iE<EArr.size(); iE++)
		EArr[iE] = Ef - 1*eV - 5*T + dE*iE;

	std::vector<double> invTauTe(EArr.size());
	//std::vector< std::vector<double> > invTauTe(EArr.size());
	double invT = 1./T;
	for(size_t iE=0; iE<EArr.size(); iE++)
	{	double E = EArr[iE], f = fermi(invT*(E - Ef));
		double lPrefac = 1. / (32 * std::pow(M_PI,3) * std::pow(epsB/(4*M_PI),2) * Es * sqrt(E));
		for(size_t iE1=0; iE1<EArr.size(); iE1++)
		{	double E1 = EArr[iE1], f1 = fermi(invT*(E1 - Ef));
			for(size_t iE2=0; iE2<EArr.size(); iE2++)
			{	double E2 = EArr[iE2], f2 = fermi(invT*(E2 - Ef));
				double E3 = E + E1 - E2, f3 = fermi(invT*(E3 - Ef));
				double occFactor = f1*(1-f2)*(1-f3) + (1-f1)*f2*f3;
				double EtildeMax = std::min(std::pow(sqrt(E1)+sqrt(E3),2),std::pow(sqrt(E)+sqrt(E2),2));
				double EtildeMin = std::max(std::pow(sqrt(E1)-sqrt(E3),2),std::pow(sqrt(E)-sqrt(E2),2));
				double arg = argLW(EtildeMax,Es) - argLW(EtildeMin,Es);
				invTauTe[iE] += lPrefac*arg*occFactor*dE*dE;
				//invTauTe[iE][iT] += lPrefac*arg*occFactor*dE*dE;
			}
		}
	}
	ofstream ofs("invTauTe.dat");
	ofs << "#E[eV] invTauTe[invSeconds]i\n";
	for(size_t iE=0; iE<EArr.size(); iE++)
	{	double E = EArr[iE];
		double invTauT0 = std::pow(E-Ef, 2) / (64*std::pow(M_PI,3)*std::pow(epsB/(4*M_PI),2)*std::pow(Es,1.5)*sqrt(Ef))
			* (2.*sqrt(Ef*Es)/(4*Ef+Es) + atan(sqrt(4*Ef/Es)));
		ofs << E/eV
			<< '\t' << invTauTe[iE]/invSeconds
			<< '\t' << invTauT0/invSeconds
			<< '\n';
	}

	finalizeSystem();
}
