#include <core/Util.h>
#include <electronic/matrix.h>
#include <fstream>
#include <sstream>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/Units.h>
#include "bandStruct.h"
#include "histogram.h"
#include "epsilon.h"

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
	const int totalBlocks = inputMap.get("totalBlocks");
	const int nKptsMetro = inputMap.get("nKptsMetro");
	const double dk = inputMap.get("dk");
	const int totalWalkers = inputMap.get("totalWalkers");
	const double kPhi = inputMap.get("kPhi");
	const double Eplasmon = inputMap.get("Eplasmon") * eV;
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const double Eplasmon2 = inputMap.get("Eplasmon2") * eV;
	const double spinWeight = inputMap.get("spinWeight");
	matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("totalBlocks = %d\n", totalBlocks);
	logPrintf("nKptsMetro = %d\n", nKptsMetro);
	logPrintf("dK = %lg\n", dk);
	logPrintf("totalWalkers = %d\n", totalWalkers);
	logPrintf("kPhi = %lg\n", kPhi);
	logPrintf("Eplasmon = %lg\n", Eplasmon);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("Eplamson2 = %lg\n", Eplasmon2);
	logPrintf("spinWeight = %lg\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	
	const double weightCut = 1e-6; // Ignore states with filling weight below this threshold

	bandStruct bs("wannier");

	//Compute the normalization factor
	int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
	int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nKpts = nKptsN1/totalBlocks;
	double N1sum = 0., N1sumSq = 0.;
	StopWatch watchNorm("normalization"); watchNorm.start();
	logPrintf("Calculating normalization factor ... "); logFlush();
	for(int block=blockStart; block<blockStop; block++)
	{	double N1block = 0.;
		for(int nk =0; nk<nKpts; nk++)
		{	vector3<> kpnt; for(int j=0; j<3; j++) kpnt[j] = Random::uniform();
			diagMatrix eigs = bs.getStates(kpnt);
			double mk = INFINITY;
			for(int indV = 0; indV < eigs.nCols(); indV++)
			{	double Ev = eigs[indV] - mu;
				if (Ev<0) // for every Ev<0
				{	for(int indC = 0; indC < eigs.nRows(); indC++)
					{	double Ec = eigs[indC] - mu;
						if (Ec>0) // for every Ec>0
							mk = std::min(mk, std::pow((Ec - Ev - Eplasmon),2));
					}
				}
			}
			N1block += exp(-0.5*mk/(T*T));
		}
		N1block /=  nKpts;
		N1sum += N1block;
		N1sumSq += std::pow(N1block,2);
	}
	watchNorm.stop();
	mpiUtil->allReduce(N1sum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(N1sumSq, MPIUtil::ReduceSum);
	double N1 = N1sum / totalBlocks;
	double N1std = sqrt(N1sumSq/totalBlocks - N1*N1);
	logPrintf("N1 = %lg +/- %lg\n", N1, N1std);

	// Dielectric values
	epsilon eps("epsilon.txt", Eplasmon);
	double Lquant = eps.getLquant();
	double modGammaMinus = eps.getModGammaMinus();
	double k = eps.getK();

	// For plasmon collect, Plasmon direction
	vector3<complex> kHat(cos(kPhi), sin(kPhi), 0.0);

// Metropolis sampling of BZ:
	logPrintf("Starting Metropolis sampling of BZ\n");
	int numWalkers = floor((totalWalkers*(mpiUtil->iProcess()+1.0))/mpiUtil->nProcesses()) - floor ((totalWalkers*mpiUtil->iProcess()*1.0)/mpiUtil->nProcesses());
	std::vector<double> Econserve_rate;
	//FILE * eigsTxt = fopen("WannierBandstruct.eigenvals","w+");
	nKpts = nKptsMetro/numWalkers;
	
	// Compute effective mode vector
	complex one(1.0,0.0);
	vector3<complex> oneVec(0.0, 0.0, one);
	complex I(0.0,1.0);
	double omega = Eplasmon;
	double absdetR = abs(det(R));
	logPrintf("nKpts = %d abs(det(R)) = %lg modGammaMinus = %lg omega = %lg Lquant = %lg\n", nKpts, absdetR, modGammaMinus, omega, Lquant);
	vector3<complex> sqrtGammaPrefac = (M_PI/(sqrt((nKpts*abs(det(R)))*modGammaMinus*omega*Lquant))) * (kHat - I*(k/modGammaMinus)*oneVec);
	logPrintf("sqrtGammaPrefac Real = %lg %lg %lg\n",  real(sqrtGammaPrefac[0]), real(sqrtGammaPrefac[1]), real(sqrtGammaPrefac[2]));
	logPrintf("sqrtGammaPrefac Imag = %lg %lg %lg\n",  imag(sqrtGammaPrefac[0]), imag(sqrtGammaPrefac[1]), imag(sqrtGammaPrefac[2]));
	
	vector3<double> kpnt, kpntPrev;
	diagMatrix eigs;
	double acceptProb, acceptBar, LineWidth_in_eV_from_1_Proc_soFar, weight, Ev, Ec , Econserve_rateSingle, Econserve_rateSum = 0, Esigma = T;
	int ik, nKptsTot, equib, totalMetroSteps=0;
	histogram EcHist(-10*Esigma, 0.5*Esigma, Eplasmon+5*Esigma);
	histogram EvHist(-Eplasmon-5*Esigma, 0.5*Esigma, 10*Esigma);
	StopWatch watchMet("metropolis"); watchMet.start();
	for(int iw = 0; iw<numWalkers; iw++)
	{	logPrintf("... metropolis sampling for one walker ...");
		ik = 0; nKptsTot = 0; equib=0;
		srand(mpiUtil->iProcess() + iw);
		for(int j=0; j<3; j++)
			kpntPrev[j] = Random::uniform();
		kpnt = kpntPrev;
		double mkPrev = INFINITY, mk;
		std::vector<double> acceptRatio;
		std::vector<matrix> Pk;
		matrix px, py, pz;
		while(ik<nKpts)
		{	// Calculate mk:
			mk = INFINITY;
			eigs = bs.getStates(kpnt);
			//eigs.print(eigsTxt);
			for( int indV = 0; indV < eigs.nCols(); indV++)
			{	Ev = eigs[indV] - mu;
				if (Ev<0) // for every Ev<0
				{	for (int indC = 0; indC < eigs.nRows(); indC++)
					{	Ec = eigs[indC] - mu;
						if (Ec>0) // for every Ec>0
						{	mk = std::min( mk, std::pow((Ec - Ev - Eplasmon),2));
							//logPrintf("mk = %lg\n", mk);
						}
					}
				}
			}

			// Metropolis accept - reject:
			acceptProb = exp(0.5*(mkPrev - mk)/(T*T));
			acceptBar = Random::uniform();
			//logPrintf("mk = %lg   mkPrev = %lg   acceptProb = %lg   acceptBar = %lg\n", mk, mkPrev, acceptProb, acceptBar);
			if (acceptProb > acceptBar)
			{	//logPrintf("loop entered\n");
				mkPrev = mk;
				kpntPrev = kpnt;
				
				if( mk < 2*T*T)
					equib = 1;
			
				if(equib==1)
				{	ik++;
					// Calculate transitions at current k-point:
					for( int indV = 0; indV < eigs.nCols(); indV++)
					{	Ev = eigs[indV] - mu;
						if (Ev<0) // for every Ev<0
						{	for( int indC = 0; indC < eigs.nRows(); indC++){
								Ec = eigs[indC] - mu;
								if ( Ec > 0) // for every Ec > 0
								{	weight=exp(0.5*(mk-std::pow((Ec-Ev-Eplasmon),2))/(T*T))/(T*sqrt(2*M_PI));
									if ( weight > weightCut )
									{	// Effective matrix elements
										Pk = bs.getTransitions(kpnt);
										px = Pk[0]; py = Pk[1]; pz = Pk[2];
										Econserve_rateSingle = (0.5*spinWeight) * weight * std::pow(abs( px(indC,indV) * sqrtGammaPrefac[0] + py(indC,indV) * sqrtGammaPrefac[1] + pz(indC,indV) * sqrtGammaPrefac[2] ),2);
										Econserve_rateSum += Econserve_rateSingle;
										//logPrintf("Econserve_rate = %lg\n", Econserve_rateSingle);
										//logPrintf("Econserve_rateSum = %lg\n", Econserve_rateSum);
									
										// Histogram energies, weights
										EcHist.addEvent(Ec, weight);
										EvHist.addEvent(Ev, weight);
									}
								}
							}
						}
					}
				}
			}
			// Generate next kpoint
			for(int j=0; j<3; j++)
				kpnt[j] = kpntPrev[j] + dk * Random::normal();
			if( equib == 1)
				nKptsTot++;

			totalMetroSteps++;
		}
		acceptRatio.push_back( (double)nKpts/nKptsTot );
		logPrintf("\nacceptRatio = %lg  nKptsTot = %d total Metro Steps = %d\n", (double)nKpts/nKptsTot, nKptsTot, totalMetroSteps);
		LineWidth_in_eV_from_1_Proc_soFar = N1/numWalkers * Econserve_rateSum/eV;
		logPrintf("LineWidth so far =  %lg\n", LineWidth_in_eV_from_1_Proc_soFar);
	}
	watchMet.stop();
	watchMet.print();
	//fclose(eigsTxt);
	
	// For plasmon collect
	double LineWidth_in_eV_from_1_Proc = N1/numWalkers * Econserve_rateSum/eV;
	logPrintf("Econserve_rateSum = %lg\n", Econserve_rateSum);
	logPrintf("LineWidth_in_eV_from_1_Proc = %lg\n", LineWidth_in_eV_from_1_Proc);
	std::vector<double> EcProbDensity = EcHist.getHist();
	std::vector<double> EcGrid = EcHist.getEgrid();
	std::vector<double> EvProbDensity = EvHist.getHist();
	std::vector<double> EvGrid = EvHist.getEgrid();
	
	finalizeSystem();
}
