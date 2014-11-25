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
			{	double dFdE =1/(T*std::pow(2*cosh(Ek[n]/(2*T)),2));
				kappaSqrdBlock += 4*M_PI*spinWeight*dFdE/fabs(det(R));
			}
		}
		kappaSqrdBlock /= nKpts;
		kappaSum += sqrt(kappaSqrdBlock);
		kappaSumSq +=kappaSqrdBlock;
	}
	mpiUtil->allReduce(kappaSum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(kappaSumSq, MPIUtil::ReduceSum);
	double kappa = kappaSum / totalBlocks;
	double kappaStd = sqrt(kappaSumSq/totalBlocks - kappa*kappa)/sqrt(totalBlocks);
	logPrintf("kappa = %lg +/- %lg\n", kappa, kappaStd);

	// Intalize Brillouin zone
	matrix3<> GT = (2*M_PI)*inv(~R); //reciprocal lattice vectors
	matrix3<> GGT = (~GT)*GT; //reciprocal space metric
	WignerSeitz BZ(GT); //Wigner-Seitz cell on the reciprocal lattice vectors

	// Compute T and Gamma
	double Tsum = 0., TsumSq = 0., GammaSum = 0., GammaSumSq = 0.;
	logPrintf("Calculating T and Gamma... "); logFlush();
	double prefacT = spinWeight/(3*nKpts);
	double prefacGamma = spinWeight*std::pow(2*M_PI,2)*vl/(3*nKpts);
	double EconserveExpFac = -0.5/(T*T), EconservePrefac = 1./(sqrt(2*M_PI)*T); //energy conserving Gaussian parameters
	for(int block=blockStart; block<blockStop; block++)
	{	Random::seed(block);
		double Tblock = 0., GammaBlock=0.;
		for(int nk1 =0; nk1<nKpts; nk1++)
		{	vector3<> kpnti, kpntj;
			for(int j=0; j<3; j++) 
			{	kpnti[j] = Random::uniform();
				kpntj[j] = Random::uniform();
			}
			#define getStatesVelocities(suffix) \
				diagMatrix E##suffix = bs.getStates(kpnt##suffix); \
				std::vector<vector3<>> v##suffix = bs.getVelocity(kpnt##suffix); \
				for(vector3<>& v: v##suffix) v = R * v; //Convert to Cartesian
			getStatesVelocities(i)
			getStatesVelocities(j)
			#undef getStatesVelocities
			
			double kPh = sqrt(GGT.metric_length_squared(BZ.restrict(kpntj - kpnti)));
			kPh = std::max(1e-7, kPh); //regularize phonon wavevector to avoid 0/0 in phonon factor
			double g_kPh = 1./(exp(vl*kPh/T) - 1.);
			double phononFactor = kPh/(kPh*kPh + kappa*kappa);
			
			for(int v=0; v<Ei.nRows(); v++)
			{	double fi  = 1./(exp(Ei[v]/T)+1);
				double dFdEi = -1/(T*std::pow(2*cosh(Ei[v]/(2*T)),2));
				double viDotvi = vi[v].length_squared();
				Tblock += prefacT * viDotvi*(-dFdEi);
				for(int c=0; c<Ej.nRows(); c++)
				{	double viDotvj = dot(vi[v], vj[c]);
					double fj = 1./(exp(Ej[c]/T)+1);
					double dFdEj = -1./(T*std::pow(2*cosh(Ej[c]/(2*T)),2));
					double deltam = EconservePrefac * exp(EconserveExpFac * std::pow(Ej[c]-Ei[v]-vl*kPh,2));
					double deltap = EconservePrefac * exp(EconserveExpFac * std::pow(Ej[c]-Ei[v]+vl*kPh,2));
					double term1 = prefacGamma * (viDotvi*(-dFdEi)*(g_kPh+fj) -  viDotvj*(-dFdEj)*(g_kPh+1-fi)) * phononFactor * deltam;
					double term2 = prefacGamma * (viDotvi*(-dFdEi)*(g_kPh+1-fj) -  viDotvj*(-dFdEj)*(g_kPh+fi)) * phononFactor * deltap;
					GammaBlock += term1 + term2;
				}
			}
		}
		Tsum += Tblock;
		TsumSq += std::pow(Tblock,2);
		GammaSum += GammaBlock;
		GammaSumSq +=std::pow(GammaBlock,2);
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
	double rho = Gamma/(Tt*Tt);
	logPrintf("Resistivity = %lg ohm-m\n", rho/(Ohm*meter));
	
	finalizeSystem();
}
