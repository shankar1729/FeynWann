#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/Units.h>
#include "bandStruct.h"
#include "histogram.h"

int main(int argc, char** argv)
{   initSystem(argc, argv);
	const double c = 1./7.29735257e-3;

	//Get the system parameters (mu, T, lattice vectors etc.)
	struct InputMap : std::map<string,double> //map class with a safe accessor that quits with error if key not found
	{	double get(string key) const
		{	auto iter = find(key);
			if(iter == end()) die("\nCould not find required entry '%s' in input.\n", key.c_str());
			return iter->second;
		}
	}
	inputMap;
	std::ifstream systemFile("system.txt");
	while(!systemFile.eof())
	{	string line; getline(systemFile, line); //line-by-line processing (comments can now be inline)
		trim(line);
		istringstream iss(line);
		string name; double val;
		if(iss >> name >> val)
			inputMap[name] = val;
	}
	systemFile.close();    
	
	const double nKptsN1 = inputMap.get("nKptsN1");
	const double totalBlocks = inputMap.get("totalBlocks");
	const double nKptsMetro = inputMap.get("nKptsMetro");
	const double dk = inputMap.get("dk");
	const double totalWalkers = inputMap.get("totalWalkers");
	const double kPhi = inputMap.get("kPhi");
	const double Eplasmon = inputMap.get("Eplasmon") * eV;
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const double Eplasmon2 = inputMap.get("Eplasmon2") * eV;
	const double spinWeight = inputMap.get("spinWeight");
	matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	std::cout << "nKptsN1 = " << nKptsN1 << std::endl;
	std::cout << "totalBlocks = " << totalBlocks << std::endl;
	std::cout << "nKptsMetro = " << nKptsMetro << std::endl;
	std::cout << "dK = " << dk << std::endl;
	std::cout << "totalWalkers = " << totalWalkers << std::endl;
	std::cout << "kPhi = " << kPhi << std::endl;
	std::cout << "Eplasmon = " << Eplasmon << std::endl;
	std::cout << "mu = " << mu << std::endl;
	std::cout << "T = " << T << std::endl;
	std::cout << "Eplamson2 = " << Eplasmon2 << std::endl;
	std::cout << "spinWeight = " << spinWeight << std::endl;
	std::cout << "R:\n";
	R.print(globalLog, " %lg ");
	
	const double weightCut = 1e-6; // Ignore states with filling weight below this threshold

	bandStruct bs("wannier");

	const int iProc = 1;
	const int nProcs = 1;

	//Compute the normalization factor
	const int numBlocks = floor((totalBlocks*(iProc+1.0))/nProcs) - floor((totalBlocks*iProc*1.0)/nProcs);
	std::vector<double> N1blocks(numBlocks);
	int nKpts = floor(nKptsN1 *1.0/numBlocks);
	double N1blocksSum = 0.0;
	for( int nb = 0; nb < numBlocks; nb++)
	{	logPrintf("Calculating normalization factor ... "); logFlush();
		N1blocks[nb] = 0;
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
			N1blocks[nb] += exp(-0.5*mk/(T*T));
		}
		N1blocks[nb] /=  nKpts;
		logPrintf("N1block = %le\n", N1blocks[nb]);
		N1blocksSum += N1blocks[nb]; // used to calculate linewidth later
	}
	double N1blocksAverage = N1blocksSum/numBlocks;
	std::cout << "N1blocksSum  = " << N1blocksSum << " N1blocksAverage = " << N1blocksAverage <<std::endl;
	//double N1blocksSum = std::accumulate(N1blocks.begin(),N1blocks.end(),0); //used to calculate linewidth

	// Dielectric values
	// Lorentz-drude model parameters: f Gamma omega
	std::vector<vector3<double>> epsParams;
	vector3<double> holdr(0.760, 0.053*eV, 0.000*eV);
	epsParams.push_back(holdr);
	holdr[0] = 0.024; holdr[1] = 0.241*eV; holdr[2] = 0.415*eV;
	epsParams.push_back(holdr);
	holdr[0] = 0.010; holdr[1] = 0.345*eV; holdr[2] = 0.830*eV;
	epsParams.push_back(holdr);
	holdr[0] = 0.071; holdr[1] = 0.870*eV; holdr[2] = 2.969*eV;
	epsParams.push_back(holdr);
	holdr[0] = 0.601; holdr[1] = 2.494*eV; holdr[2] = 4.304*eV;
	epsParams.push_back(holdr);
	holdr[0] = 4.384; holdr[1] = 2.214*eV; holdr[2] = 13.32*eV;
 	epsParams.push_back(holdr);
	double omega_p = 9.03*eV;
	// Calculate the dielectric at omega = Eplasmon
	double omega = Eplasmon;
	complex epsilon(1.0,0.0), omegaEpsilonPrime(1.0,0.0), one(1.0,0.0), den;
	double num;
	vector3<double> epsParam;
	complex I(0.0,1.0);
	for( int iPole = 0; iPole < epsParams.size(); iPole++)
	{	epsParam = epsParams[iPole];
		num = epsParam[0]*(std::pow(omega_p,2));
		den = one/(std::pow(epsParam[2],2) - omega*omega - I * omega * epsParam[1]);
		epsilon = epsilon + num *den;
		omegaEpsilonPrime = omegaEpsilonPrime + num * den*den * (std::pow(epsParam[2],2) + omega*omega);
	}
	// Plasmon mode deatils
	double realEpsilon = real(epsilon);
	double k = (omega/c) * sqrt(realEpsilon/(realEpsilon+1));
	double modGammaPlus = sqrt(k*k - (omega/c)*(omega/c));
	double modGammaMinus = sqrt(k*k - (omega/c)*(omega/c));
	double Lquant = (1/(4*std::pow(modGammaPlus,3))) * (std::pow(modGammaPlus,2) + k*k + std::pow((omega/c),2)) + (1/(4*std::pow(modGammaPlus,3))) * ((std::pow(modGammaMinus,2)+k*k)*real(omegaEpsilonPrime)+(std::pow((real(epsilon*omega/c)),2)));

	// For plasmon collect, Plasmon direction
	vector3<complex> kHat(cos(kPhi), sin(kPhi), 0.0);

	// Compute effective mode vector on the frequency grid
	vector3<complex> oneVec(0.0, 0.0, one);
	vector3<complex> sqrtGammaPrefac = (M_PI/(sqrt((nKpts*abs(det(R)))*modGammaMinus*omega*Lquant))) * (kHat - I*(k/modGammaMinus)*oneVec);
	std::cout << "sqrtGammaPrefac Real = " <<  real(sqrtGammaPrefac[0]) << " " << real(sqrtGammaPrefac[1]) << " " << real(sqrtGammaPrefac[2]) <<std::endl;
	std::cout << "sqrtGammaPrefac Imag = " <<  imag(sqrtGammaPrefac[0]) << " " << imag(sqrtGammaPrefac[1]) << " " << imag(sqrtGammaPrefac[2]) <<std::endl;
	std::cout << "sqrt part 1 = " <<  (M_PI/(sqrt((nKpts*abs(det(R)))*modGammaMinus*omega*Lquant))) << std::endl;
	std::cout << "parts of 1 = " << det(R) << " " << modGammaMinus << " " << omega << " " << Lquant << std::endl;
	std::cout << "sqrt part 2 Real = " << real(kHat[0] - I*(k/modGammaMinus)*oneVec[0]) << " "  <<  real(kHat[1] - I*(k/modGammaMinus)*oneVec[1])  << " "  << real(kHat[2] - I*(k/modGammaMinus)*oneVec[2])  <<std::endl;
	std::cout << "sqrt part 2 Imag = " << imag(kHat[0] - I*(k/modGammaMinus)*oneVec[0]) << " "  <<  imag(kHat[1] - I*(k/modGammaMinus)*oneVec[1])  << " "  << imag(kHat[2] - I*(k/modGammaMinus)*oneVec[2])  <<std::endl;

	// Metropolis sampling of BZ:
	std::cout << "Starting Metropolis sampling of BZ" << std::endl;
	int numWalkers = floor((totalWalkers*(iProc+1.0))/nProcs) - floor ((totalWalkers*iProc*1.0)/nProcs);
	std::vector<double> Econserve_Ec, Econserve_Ev, Econserve_rate;
	std::vector< vector3<complex> > Econserve_pcv; // matrix element (complex 3-vector)
	std::vector< vector3<double> > Econserve_evcw, Econserve_pc, Econserve_pv; // energies and weights, final & initial momentum (real 3-vector)
	FILE * eigsTxt;
        eigsTxt = fopen("WannierBandstruct.eigenvals","w+");
	nKpts = floor(nKptsMetro/numWalkers);
	vector3<double> kpnt, kpntPrev, holder, holderpc, holderpv;
	vector3<complex> holderc;
	diagMatrix eigs;
	double acceptProb, acceptBar, LineWidth_in_eV_from_1_Proc_soFar, weight, Ev, Ec , Econserve_rateSingle, Econserve_rateSum = 0, Esigma = T;
	int ik, nKptsTot, equib;
	histogram EcHist(-10*Esigma, 0.5*Esigma, Eplasmon+5*Esigma);
	histogram EvHist(-Eplasmon-5*Esigma, 0.5*Esigma, 10*Esigma);
	for( int iw = 0 ; iw<numWalkers; iw++)
	{	std::cout << "... metropolis sampling for one walker ...";
		ik = 0; nKptsTot = 0; equib=0;
		srand(iProc + iw);
		//for(int j=0; j<3; j++) kpntPrev[j] = Random::uniform();
		kpntPrev[0] = ((double) rand() / (RAND_MAX));
                kpntPrev[1] = ((double) rand() / (RAND_MAX));
                kpntPrev[2] = ((double) rand() / (RAND_MAX));
		kpnt = kpntPrev;
		double mkPrev = INFINITY, mk;
		std::vector<double> Evs, Ecs, acceptRatio;
		std::vector<matrix> Pk;
		matrix px, py, pz;
		while(ik<nKpts)
		{	// Calculate mk:
			mk = INFINITY;
			eigs = bs.getStates(kpnt);
			eigs.print(eigsTxt);
			for( int indV = 0; indV < eigs.nCols(); indV++)
			{	Ev = eigs[indV] - mu;
				if (Ev<0) // for every Ev<0
				{	Evs.push_back(Ev);
					for (int indC = 0; indC < eigs.nRows(); indC++)
					{	Ec = eigs[indC] - mu;
						if (Ec>0) // for every Ec>0
						{	Ecs.push_back(Ec);
							mk = std::min( mk, std::pow((Ec - Ev - Eplasmon),2));
							//std::cout << "mk = " << mk << std::endl;
                                                 }
                                         }
                                 }
                         }

			// Metropolis accept - reject:
			acceptProb = exp(0.5*(mkPrev - mk)/(T*T));
			acceptBar = ((double) rand() / (RAND_MAX));
			//std::cout << "mk = " << mk << " mkPrev = " << mkPrev << " acceptProb = " << acceptProb << " acceptBar = " << acceptBar << " kpnt = " << kpnt[0] << " " << kpnt[1] << "  " << kpnt[2]  << std::endl;
			if (acceptProb > acceptBar)
			{	//std:: cout << "loop enetered" << std::endl;
				mkPrev = mk;
				kpntPrev = kpnt;
				
				if( mk < 2*T*T)
					equib = 1;
			
				if(equib==1)
				{	ik++;
					
					// Calculate transitions at current k-point:
					weight = exp(0.5*(mk - std::pow((Ec - Ev - Eplasmon),2))/(T*T))/(T*sqrt(2*M_PI));
					Pk = bs.getTransitions(kpnt);
					
					for( int indV = 0; indV < eigs.nCols(); indV++)
					{	Ev = eigs[indV] - mu;
						if (Ev<0) // for every Ev<0
						{	for( int indC = 0; indC < eigs.nRows(); indC++){
								Ec = eigs[indC] - mu;
								if ( Ec > 0)
								{	weight=exp(0.5*(mk-std::pow((Ec-Ev-Eplasmon),2))/(T*T))/(T*sqrt(2*M_PI));
									if ( weight > weightCut )
									{	// Scalars
										holder[0] = Ev; holder[1] = Ec; holder[2] = weight;
										Econserve_evcw.push_back(holder);
										Econserve_Ev.push_back(Ev);
										Econserve_Ec.push_back(Ec);
										// Effective matrix elements
										px = Pk[0]; py = Pk[1]; pz = Pk[2];
										holderc[0] = px(indC,indV); holderc[1] = py(indC,indV); holderc[2] = pz(indC,indV);
										Econserve_pcv.push_back(holderc);
										Econserve_rateSingle = (0.5*spinWeight) * weight * std::pow(abs( holderc[0] * sqrtGammaPrefac[0] + holderc[1] * sqrtGammaPrefac[1] + holderc[2] * sqrtGammaPrefac[2] ),2);
										Econserve_rate.push_back(Econserve_rateSingle);
										Econserve_rateSum += Econserve_rateSingle;
										std::cout << "Econserve_rate = " << Econserve_rateSingle << std::endl;
										//std::cout << "Econserve_rateSum = " << Econserve_rateSum << std::endl;
										// Momenta
										holderpc[0] = real(px(indC,indC));
										holderpc[1] = real(px(indC,indC));
                                                                                holderpc[2] = real(px(indC,indC));
                                                                                holderpv[0] = real(px(indV,indV));
                                                                                holderpv[1] = real(px(indV,indV));
                                                                                holderpv[2] = real(px(indV,indV));
										Econserve_pc.push_back(holderpc);
										Econserve_pv.push_back(holderpv);
									
										// Histogram energys, weights
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
			//for(int j=0; j<3; j++) kpnt[j] = Random::uniform();
			kpnt[0] = kpntPrev[0] + dk * ((double) rand() / (RAND_MAX));
			kpnt[1] = kpntPrev[1] + dk * ((double) rand() / (RAND_MAX));
			kpnt[2] = kpntPrev[2] + dk * ((double) rand() / (RAND_MAX));
			if( equib == 1)
				nKptsTot++;
		}
		acceptRatio.push_back( (double)nKpts/nKptsTot );
		std::cout << std::endl << "acceptRatio = " << (double)nKpts/nKptsTot << " nKptsTot = " << nKptsTot <<std::endl;
		LineWidth_in_eV_from_1_Proc_soFar = N1blocksAverage/numWalkers * Econserve_rateSum/eV;
		std::cout << "LineWidth so far =  " << LineWidth_in_eV_from_1_Proc_soFar << std::endl;
	}

// For plasmon collect
//double N1blocksSum = std::accumulate(N1blocks.begin(),N1blocks.end(),0);
double LineWidth_in_eV_from_1_Proc = N1blocksAverage/numWalkers * Econserve_rateSum/eV;
std::cout << "Econserve_rateSum = " << Econserve_rateSum << std::endl;
std::cout << "LineWidth_in_eV_from_1_Proc = " << LineWidth_in_eV_from_1_Proc << std::endl;
//double Esigma = T;
std::vector<double> EcProbDensity = EcHist.getHist();
std::vector<double> EcGrid = EcHist.getEgrid();
std::vector<double> EvProbDensity = EvHist.getHist();
std::vector<double> EvGrid = EvHist.getEgrid();
std::cout << "done" << std::endl;
}
