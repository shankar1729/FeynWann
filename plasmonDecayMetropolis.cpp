#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include "bandStruct.h"
#include "histogram.h"
#include <sstream>
#include <string>
#include <cmath>
#include <numeric>
#include <core/scalar.h>

int main(int argc, char** argv)
{       initSystem(argc, argv);
	const double Angstrom = 1./0.5291772192;	
	const double nanometer = 10*Angstrom;
	const double eV = 1./27.21138386;
	const double invSeconds = 2.418884326505e-17;
	const double c = 1./7.29735257e-3;

	//Get the system parameters (mu, T, lattice vectors etc.)
	std::ifstream systemFile("system.txt");
	string name;
	double val;
	std::vector<string> names;
	std::vector<double> vals;
	while (systemFile >> name >> val){
		std:: cout << name << " " << val << std::endl;	
		names.push_back(name);
		vals.push_back(val);
	}
		
	systemFile.close();    
	
	const double nKptsN1 = vals.at(0);
	const double totalBlocks = vals.at(1);
	const double nKptsMetro = vals.at(2);
	const double dk = vals.at(3);
	const double totalWalkers = vals.at(4);
	const double kPhi = vals.at(5);
	const double Eplasmon = vals.at(6)*eV;
	const double mu = 0.57938;//vals.at(7);
	const double T = vals.at(8)*eV;
	const double Eplasmon2 = vals.at(9);
	const double spinWeight = vals.at(10);
	//RRRRRR  not sure how to read this in.... so just do it explicitly for now......
	//std::vector< vector3<double> > R;
	matrix R(3,3);
	double llen = (4.080*Angstrom)*0.5;
	//R[1] = {0,llen,llen};
	//R[2] = {llen,0,llen};
	//R[3] = {llen,llen,0};
	R(0,0) = 0; R(0,1) = llen; R(0,2) = llen;
	R(1,0) = llen; R(1,1) = 0; R(1,2) = llen;
	R(2,0) = llen; R(2,1) = llen; R(2,2) = 0;

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
	
	const double weightCut = 1e-6; // Ignore states with filling weight below this threshold

	bandStruct bs("wannier");

	const int iProc = 1;
	const int nProcs = 1;

	//Compute the normalization factor
	const int numBlocks = floor((totalBlocks*(iProc+1.0))/nProcs) - floor((totalBlocks*iProc*1.0)/nProcs);
	std::vector<double> N1blocks(numBlocks);
	int nKpts = floor(nKptsN1 *1.0/numBlocks);
	std::vector<double> expVec;
	for( int nb = 0; nb < numBlocks; nb++){
		std::cout << "Calculating normalization factor ... " << std::endl;
		std::vector<double> mk(nKpts,INFINITY);
		expVec.clear();
		srand(iProc+nb);
		std::vector< vector3<double> > kpoints;
		diagMatrix eigs;
		vector3<double> kpnt;
		double Ev, Ec;
		//std::cout << "nb = " << nb << "   nKpts = " << nKpts << std::endl;
		for( int nk =0; nk<nKpts; nk++){
			kpnt[0] = ((double) rand() / (RAND_MAX));
			kpnt[1] = ((double) rand() / (RAND_MAX));
			kpnt[2] = ((double) rand() / (RAND_MAX));
			kpoints.push_back(kpnt);
			//std::cout << "rands = " << kpnt[1] << " " << kpnt[2] << "  " << kpnt[3]  << std::endl; // for debugging
			eigs = bs.getStates(kpnt);
			for(int indV = 0; indV < eigs.nCols(); indV++){
				Ev = eigs[indV] - mu;
				if (Ev<0){ // for every Ev<0
                        		for (int indC = 0; indC < eigs.nRows(); indC++){ 	
						Ec = eigs[indC] - mu;
						if (Ec>0){ // for every Ec>0
							mk[nk] = std::min( mk[nk], std::pow((Ec - Ev - Eplasmon),2));
							//std::cout << "mk = " << mk[nk] << " Ev = " << Ev << " Ec = " <<Ec << std::endl;
						}
					}					
				}				
			}
			expVec.push_back(exp(-0.5*mk[nk]/(T*T)));
			//std::cout << "mk min= " << mk[nk] <<" exp= " << expVec[nk] << std::endl;
		}
		N1blocks[nb] = std::accumulate(expVec.begin(), expVec.end(), 0) / expVec.size();
		std::cout << "N1block = " << N1blocks[nb] << std::endl;
		//for(int ii = 0; ii < nKpts; ii++)
		//	std::cout << expVec[ii] << " ";
		mk.clear();
	}

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
	for( int iPole = 0; iPole < epsParams.size(); iPole++){
		epsParam = epsParams[iPole];
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

	// For plasmon collect
	// Plasmon direction
	vector3<complex> kHat(cos(kPhi), sin(kPhi), 0.0);

	// Compute effective mode vector on the frequency grid
	vector3<complex> oneVec(0.0, 0.0, one);
	vector3<complex> sqrtGammaPrefac = (M_PI/(sqrt((nKpts*abs(det(R)))*modGammaMinus*omega*Lquant))) * (kHat - I*(k/modGammaMinus)*oneVec);

	// Metropolis sampling of BZ:
	int numWalkers = floor((totalWalkers*(iProc+1.0))/nProcs) - floor ((totalWalkers*iProc*1.0)/nProcs);
	std::vector<double> Econserve_Ec, Econserve_Ev;
	std::vector< vector3<complex> > Econserve_pcv; // matrix element (complex 3-vector)
	std::vector< vector3<double> > Econserve_evcw, Econserve_pc, Econserve_pv, Econserve_rate; // energies and weights, final & initial momentum (real 3-vector)
	double Econserve_rateSum = 0;
	FILE * eigsTxt;
        eigsTxt = fopen("WannierBandstruct.eigenvals","w+");
	nKpts = floor(nKptsMetro/numWalkers);
	vector3<double> kpnt, kpntPrev, holder, holderpc, holderpv, holderr;
	vector3<complex> holderc;
	diagMatrix eigs;
	double acceptProb, weight, ikTot, Ev, Ec;
	for( int iw = 0 ; iw<numWalkers; iw++){
		int ik = 0, nKptsTot = 0, equib=0;
		srand(iProc + iw);
		kpntPrev[1] = ((double) rand() / (RAND_MAX));
                kpntPrev[2] = ((double) rand() / (RAND_MAX));
                kpntPrev[3] = ((double) rand() / (RAND_MAX));
		kpnt = kpntPrev;
		double mkPrev = INFINITY, mk;
		std::vector<double> Evs, Ecs, acceptRatio;
		std::vector<matrix> Pk;
		matrix px, py, pz;
		while(ik<nKpts){
			// Calculate mk:
			mk = INFINITY;
			eigs = bs.getStates(kpnt);
			eigs.print(eigsTxt);
			for( int indV = 0; indV < eigs.nCols(); indV++){
				Ev = eigs[indV] - mu;
				if (Ev<0){ // for every Ev<0
					Evs.push_back(Ev);
					for (int indC = 0; indC < eigs.nRows(); indC++){
						Ec = eigs[indC] - mu;
						if (Ec>0){ // for every Ec>0
							Ecs.push_back(Ec);
							mk = std::min( mk, std::pow((Ec - Ev - Eplasmon),2));
                                                 }
                                         }
                                 }
                         }

			// Metropolis accept - reject:
			acceptProb = exp(0.5*(mkPrev - mk)/(T*T));
			//std::cout << "mk = " << mk << " mkPrev = " << mkPrev << " acceptProb = " << acceptProb << " kpnt = " << kpnt[1] << " " << kpnt[2] << "  " << kpnt[3]  << std::endl;
			if (acceptProb > ((double) rand() / (RAND_MAX))){
				//std:: cout << "loop enetered" << std::endl;
				mkPrev = mk;
				kpntPrev = kpnt;
				
				if( mk < 2*T*T)
					equib = 1;
			
				if(equib==1){
					ik++;
					
					// Calculate transitions at current k-point:
					weight = exp(0.5*(mk - std::pow((Ec - Ev - Eplasmon),2))/(T*T))/(T*sqrt(2*M_PI));
					Pk = bs.getTransitions(kpnt);
					ikTot = (iw)*nKpts + ik + 1;
					
					for( int indV = 0; indV < eigs.nCols(); indV++){
						Ev = eigs[indV] - mu;
						if (Ev<0){ // for every Ev<0
							for( int indC = 0; indC < eigs.nRows(); indC++){
								Ec = eigs[indC] - mu;
								if ( Ec > 0){
									weight=exp(0.5*(mk-std::pow((Ec-Ev-Eplasmon),2))/(T*T))/(T*sqrt(2*M_PI));
									if ( weight > weightCut ){
										// Scalars
										holder[0] = Ev; holder[1] = Ec; holder[2] = weight;
										Econserve_evcw.push_back(holder);
										Econserve_Ev.push_back(Ev);
										Econserve_Ec.push_back(Ec);
										// Effective matrix elements
										px = Pk[0]; py = Pk[1]; pz = Pk[2];
										holderc[0] = px(indC,indV); holderc[1] = py(indC,indV); holderc[2] = pz(indC,indV);
										Econserve_pcv.push_back(holderc);
										holderr[0] = std::pow(abs(holderc[0] * sqrtGammaPrefac[0]),2);
										holderr[1] = std::pow(abs(holderc[1] * sqrtGammaPrefac[1]),2);
										holderr[2] = std::pow(abs(holderc[2] * sqrtGammaPrefac[2]),2);
										Econserve_rate.push_back((0.5*spinWeight) * weight * holderr);
										Econserve_rateSum += (0.5*spinWeight) * weight * (holderr[0] + holderr[1] + holderr[2]);
										// Momenta
										holderpc[0] = real(px(indC,indC));
										holderpc[1] = real(px(indC,indC));
                                                                                holderpc[2] = real(px(indC,indC));
                                                                                holderpv[0] = real(px(indV,indV));
                                                                                holderpv[1] = real(px(indV,indV));
                                                                                holderpv[2] = real(px(indV,indV));
										Econserve_pc.push_back(holderpc);
										Econserve_pv.push_back(holderpv);
									}
								}
							}
						}
					}
				}
			}
			// Generate next kpoint
			kpnt[1] = kpntPrev[1] + dk * ((double) rand() / (RAND_MAX));
			kpnt[2] = kpntPrev[2] + dk * ((double) rand() / (RAND_MAX));
			kpnt[3] = kpntPrev[3] + dk * ((double) rand() / (RAND_MAX));
			if( equib == 1)
				nKptsTot++;
		}
		acceptRatio.push_back( nKpts/nKptsTot );
		std::cout << "acceptRatio = " << nKpts/nKptsTot << std::endl;
	}

// For plasmon collect
double N1blocksSum = std::accumulate(N1blocks.begin(),N1blocks.end(),0);
double LineWidth_in_eV_from_1_Proc = N1blocksSum/N1blocks.size()/numWalkers * Econserve_rateSum/eV;
std::cout<< "LineWidth_in_eV_from_1_Proc = " << LineWidth_in_eV_from_1_Proc << std::endl;
double Esigma = T;

}	
