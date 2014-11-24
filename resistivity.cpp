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

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Metropolis calculation of plasmon decay rate", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	struct InputMap : std::map<string,double> //map class with a safe accessor that quits with error if key not found
	{	double get(string key) const
		{	auto iter = find(key);
			if(iter == end()) die("\nCould not find required entry '%s' in input.\n", key.c_str());
			return iter->second;
		}
	}
	inputMap;
	std::ifstream systemFile(inputFilename.c_str());
	if(!systemFile.is_open())
		die("Could not open system file '%s' for reading.\n", inputFilename.c_str());
	while(!systemFile.eof())
	{	string line; getline(systemFile, line); //line-by-line processing (comments can now be inline)
		trim(line);
		istringstream iss(line);
		string name; double val;
		if(iss >> name >> val)
			inputMap[name] = val;
	}
	systemFile.close();    
	
	const int nKptsN1 = inputMap.get("nKptsN1");
	const int totalBlocks = inputMap.get("totalBlocks"); assert(totalBlocks>0);
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const double spinWeight = inputMap.get("spinWeight");
	const double vl = inputMap.get("vl")* meter *2.41888e-17;// m/s in atomic units
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("totalBlocks = %d\n", totalBlocks);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("spinWeight = %lg\n", spinWeight);
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
	BandStruct bs("wannier", mu);

	// Calculate kappa
	int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
        int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
        int nKpts = nKptsN1/totalBlocks;
        double kappaSum = 0., kappaSumSq = 0.;
        logPrintf("Calculating kappa ... "); logFlush();
        for(int block=blockStart; block<blockStop; block++)
        {       Random::seed(block);
                double kappaSqrdBlock = 0.;
                for(int nk1 =0; nk1<nKpts; nk1++)
                {       vector3<> kpnt1;
                        for(int j=0; j<3; j++) kpnt1[j] = Random::uniform();
                        diagMatrix Ek = bs.getStates(kpnt1);
                        for(int n = 0; n<Ek.nRows(); n++)
                        {       double dFdE =1/(T*std::pow(2*cosh(Ek[n]/(2*T)),2));
                                kappaSqrdBlock += 4*M_PI*spinWeight*dFdE/fabs(det(R));
                        }
                }
                kappaSqrdBlock /= nKpts;
                kappaSum += sqrt(kappaSqrdBlock);
                kappaSumSq +=kappaSqrdBlock;
        }
        double kappa = kappaSum / totalBlocks;
        double kappaStd = sqrt(kappaSumSq/totalBlocks - kappa*kappa);
        logPrintf("kappa = %lg +/- %lg\n", kappa, kappaStd);

	// Intalize Brillouin zone
	matrix3<> GT = (2*M_PI)*inv(~R); //reciprocal lattice vectors
	matrix3<> GGT = (~GT)*GT; //reciprocal space metric
	WignerSeitz BZ(GT); //Wigner-Seitz cell on the reciprocal lattice vectors

	// Compute T and Gamma
	double Tsum = 0., TsumSq = 0., Gamma = 0.;
	logPrintf("Calculating T and Gamma... "); logFlush();
	double prefac = 8*M_PI*M_PI*vl/(3*fabs(det(R))*nKpts);
	for(int block=blockStart; block<blockStop; block++)
	{	Random::seed(block);
		double Tblock = 0.;;
		for(int nk1 =0; nk1<nKpts; nk1++)
		{	vector3<> kpnti, kpntj;
			for(int j=0; j<3; j++) 
			{	kpnti[j] = Random::uniform();
				kpntj[j] = Random::uniform();
			}
			double viDotvi = 1., viDotvj = 1.;

			// Calculate transitions at current k-point:
			double kPh = sqrt(GGT.metric_length_squared(BZ.restrict(kpntj - kpnti)));
			kPh = std::max(1e-7, kPh); //regularize phonon wavevector to avoid 0/0 in phonon factor
			double g_kPh = 1./(exp(vl*kPh/T) - 1.);
			double phononFactor = kPh/(kPh*kPh + kappa*kappa);
			diagMatrix Ei = bs.getStates(kpnti);
			std::vector<matrix> Pki = bs.getTransitions(kpnti);
			diagMatrix Ej = bs.getStates(kpntj);
			std::vector<matrix> Pkj = bs.getTransitions(kpntj);
			for(int v=0; v<Ei.nRows(); v++)
			{	double dFdEi = -1/(T*std::pow(2*cosh(Ei[v]/(2*T)),2));
				double fi  = 1/(exp(Ei[v]/T)+1);
				Tblock += (2/3)*spinWeight*viDotvi*(-dFdEi); // should spineWeight be included here?
				for(int c=0; c<Ej.nRows(); c++)
				{	double fj = 1/(exp(Ej[c]/T)+1);
					double dFdEj = -1/(T*std::pow(2*cosh(Ej[c]/(2*T)),2));
					double deltam = 1/(1-exp(Ej[c]-Ei[v]-vl*kPh));
					double deltap = 1/(1-exp(Ej[c]-Ei[v]+vl*kPh));
					double term1 = prefac * (viDotvi*(-dFdEi)*(g_kPh+fj) -  viDotvj*(-dFdEj)*(g_kPh+1-fi)) * phononFactor * deltam;
					double term2 = prefac * (viDotvi*(-dFdEi)*(g_kPh+1-fj) -  viDotvj*(-dFdEj)*(g_kPh+fi)) * phononFactor * deltap;
					Gamma += term1 + term2;
				}
			}		

		}
		Tblock /=  nKpts;
		Tsum += Tblock;
		TsumSq += std::pow(Tblock,2);
	}

	mpiUtil->allReduce(Tsum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(TsumSq, MPIUtil::ReduceSum);
	double Tt = Tsum / totalBlocks;
	double Tstd = sqrt(TsumSq/totalBlocks - Tt*Tt);
	logPrintf("T = %lg +/- %lg\n", Tt, Tstd);
	
	//Decay rate:
	mpiUtil->allReduce(Gamma, MPIUtil::ReduceSum);
	logPrintf("Linewidth = %lg eV\n", Gamma/eV);
	
	// Calculate Resistivity
	double row = fabs(det(R))*Gamma/(Tt*Tt);
	logPrintf("Resistivity = %lg\n", row);
}
