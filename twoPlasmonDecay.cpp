#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include "BandStruct.h"
#include "Histogram.h"
#include "Epsilon.h"
#include "LineWidth.h"
#include "InputMap.h"
#include "Units.h"

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of two-plasmon decay rate", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const double EplasmonMax = inputMap.get("EplasmonMax") * eV;
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("EplasmonMax = %lg\n", EplasmonMax);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Initialize dielectric model:
	Epsilon eps("Wannier/epsilon.dat");
	
	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(1);
	Ahat[0] = vector3<complex>(1.,0.,0.); //cubic crystal, one polarization sufficient
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE", Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);

	//Initalize line width of intermediate electronic states
	LineWidth lineWidth("Wannier/wannier", bs);

	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	long nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	long nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	int nBands = bs.getStates(vector3<>()).nRows();
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	complex I(0,1);
	double prefac0 = spinWeight * M_PI / (nKpts * fabs(det(R))); //frequency independent part of prefac
	double prefac0ph = spinWeight * M_PI / (nKpairs * fabs(det(R))); //frequency independent part of prefac
	
	//Singularity extrapolation parameters
	double extrapCoeff[] = {-19./12, 13./3, -7./4 }; //account for constant, 1/eta and eta^2 dependence
	//double extrapCoeff[] = { -1, 2.}; //account for constant and 1/eta dependence
	//double extrapCoeff[] = { 1. }; //no extrapolation
	const int nExtrap = sizeof(extrapCoeff)/sizeof(double);
	const double eta = 0.1*eV;

	//Initialize histograms
	double gaussMargin = 5*T;
	double fermiMargin = 10*T;
	double EplasmonTotMax = 2*EplasmonMax; //max on sum of two plasmon energies
	Histogram2D Ev1(-EplasmonTotMax-gaussMargin, T, fermiMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D Ec1(-fermiMargin, T, EplasmonTotMax+gaussMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D Ev3(-EplasmonTotMax-gaussMargin, T, fermiMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D Ec3(-fermiMargin, T, EplasmonTotMax+gaussMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D Ev1ph(-EplasmonTotMax-gaussMargin, T, fermiMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D Ec1ph(-fermiMargin, T, EplasmonTotMax+gaussMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D Ev3ph(-EplasmonTotMax-gaussMargin, T, fermiMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D Ec3ph(-fermiMargin, T, EplasmonTotMax+gaussMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram chi1(gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram chi3(gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram chi1ph(gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram chi3ph(gaussMargin, T, EplasmonMax-gaussMargin);
	std::vector<Histogram> conv(nBands,  chi3); //empty-state convergence

	//Monte Carlo loop:
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	logPrintf("\nProgress: "); logFlush();
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{
		//Generate a bunch of k-points:
		std::vector< vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		
		//Calculate electronic states and matrix elements for bunch:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector<diagMatrix> ImEarr = lineWidth(kArr);
		std::vector< std::vector<matrix> > Parr = bs.getDipoleMatElem(kArr);
		std::vector<diagMatrix> Farr = Earr; //convert to fillings:
		for(diagMatrix& F: Farr)
			for(double& f: F)
			{	double e = f/T; //E/T actually
				f = (e>30 ? exp(-e) : 1./(1. + exp(e))); //avoid overflow issues
			}

		//Direct transitions:
		for(int ik=0; ik<bunchSize; ik++)
		{	const diagMatrix& E = Earr[ik];
			const diagMatrix& ImE = ImEarr[ik];
			const matrix& P = Parr[ik][0];
			const diagMatrix& F = Farr[ik];
			
			//One-plasmon process (chi1)
			for(int v=0; v<nBands; v++) if(E[v]<10.*T)
			{	for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
				{	double omega = E[c] - E[v]; //energy conservation
					if(omega<=0 || omega>=EplasmonMax) continue; //irrelevant event
					double prefac = (F[v]-F[c]) * prefac0; //note omega^-2 factor added later for best histogramming
					complex Meff = P(c,v);
					double weight = prefac * Meff.norm(); //norm=abs^2
					Ev1.addEvent(E[v], omega, weight);
					Ec1.addEvent(E[c], omega, weight);
					chi1.addEvent(omega, weight);
				}
			}
			
			//Two-plasmon process (chi3)
			for(int v=0; v<nBands; v++) if(E[v]<10.*T)
			{	for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
				{	double omegaTot = E[c] - E[v]; //energy conservation
					double omega = 0.5*omegaTot; //only considering processes with equal plasmon energies
					if(omega<=0 || omega>=EplasmonMax) continue; //irrelevant event
					double prefac = (F[v]-F[c]) * prefac0; //note omega^-4 factor added later for best histogramming
					//Effective matrix element
					complex Meff = 0.; double weight = 0.;
					for(int i=0; i<nBands; i++) // sum over the intermediate states
					{	complex Ei(E[i], ImE[i]);
						Meff += 2. * P(c,i) * P(i,v) / (Ei-E[v]-omega); //factor of 2 from exchanged term (identical for two plasmons with same mode)
						weight = prefac * Meff.norm(); //norm = abs^2;
						conv[i].addEvent(omega, weight); //estimate based on truncating to i bands
					}
					//Include in statistics:
					Ev3.addEvent(E[v], omega, weight);
					Ec3.addEvent(E[c], omega, weight);
					chi3.addEvent(omega, weight);
				}
			}
		}
		
		//Phonon-assisted transitions:
		for(int ik1=0; ik1<bunchSize; ik1++)
		{	const diagMatrix& E1 = Earr[ik1];
			const diagMatrix& ImE1 = ImEarr[ik1];
			const diagMatrix& F1 = Farr[ik1];
			const matrix& P1 = Parr[ik1][0];
			//phonon matrix elements for ik1 with rest of bunch:
			std::vector<matrix> gePhArr[bunchSize];
			bs.setPhononMatElemArray(kArr[ik1], kArr, gePhArr);
			//Loop over second k-point:
			for(int ik2=0; ik2<bunchSize; ik2++) if(ik2!=ik1) //avoid gamma-point phonon singularity
			{	const std::vector<matrix>& gePh = gePhArr[ik2];
				diagMatrix omegaPh = bs.getPhononModes(kArr[ik1] - kArr[ik2]);
				const diagMatrix& E2 = Earr[ik2];
				const diagMatrix& ImE2 = ImEarr[ik2];
				const diagMatrix& F2 = Farr[ik2];
				const matrix& P2 = Parr[ik2][0];
				//Loops over bands and phonon modes:
				for(int v=0; v<nBands; v++) if(E1[v]<10.*T)
				{	for(int c=0; c<nBands; c++) if(E2[c]>-10.*T)
					{	for(int alpha=0; alpha<nModes; alpha++)
						{	double nPh = 1./(exp(omegaPh[alpha]/T) - 1.);
							for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
							{	double omegaTot = E2[c] - E1[v] - ae*omegaPh[alpha]; //energy conservation
								if(omegaTot<=0. || omegaTot>=2.*EplasmonMax) continue;
								double prefac = prefac0ph * (F1[v]-F2[c]) * (nPh + 0.5*(1.-ae)); //note omega^-2 or omega^-4 factor added later for best histogramming
								
								//One-plasmon process (chi1ph)
								double omega = omegaTot;
								if(omega < EplasmonMax)
								{	std::vector<complex> Meff(nExtrap, 0.);
									for(int i=0; i<nBands; i++) // sum over the intermediate states
									{	for(int z=0; z<nExtrap; z++)
										{	complex iEta(0, (z+1)*eta);
											Meff[z] += 
												( P2(c,i) * gePh[alpha](i,v) / (E2[i]+iEta - (E2[c] - omega))
												+ gePh[alpha](c,i) * P1(i,v) / (E1[i]+iEta - (E1[v] + omega)) );
										}
									}
									double weight = 0.;
									for(int z=0; z<nExtrap; z++)
										weight += prefac * extrapCoeff[z] * Meff[z].norm();
									Ev1ph.addEvent(E1[v], omega, weight);
									Ec1ph.addEvent(E2[c], omega, weight);
									chi1ph.addEvent(omega, weight);
								}
								
								//Two-plasmon process (chi3ph)
								omega = 0.5*omegaTot;
								if(omega < EplasmonMax)
								{	std::vector<complex> Meff(nExtrap, 0.);
									for(int a=0; a<nBands; a++) // sum over the first intermediate state
									for(int b=0; b<nBands; b++) // sum over the second intermediate state
									{	complex imE1a = complex(0,ImE1[a]);
										complex imE2b = complex(0,ImE2[b]);
										for(int z=0; z<nExtrap; z++)
										{	complex iEta(0, (z+1)*eta);
											Meff[z] += 
												( P2(c,b) * P2(b,a) * gePh[alpha](a,v) / ((E2[a]+iEta - (E2[c] - omegaTot)) * (E2[b]+iEta - (E2[c] - omega)))
												+ P2(c,b) * gePh[alpha](b,a) * P1(a,v) / ((E1[a]+imE1a - (E1[v] + omega)) * (E2[b]+imE2b - (E2[c] - omega))) //use actual linewidths to not extrapolate
												+ gePh[alpha](c,b) * P1(b,a) * P1(a,v) / ((E1[a]+iEta - (E1[v] + omega)) * (E1[b]+iEta - (E1[v] + omegaTot))) );
										}
									}
									double weight = 0.;
									for(int z=0; z<nExtrap; z++)
										weight += prefac * extrapCoeff[z] * Meff[z].norm();
									Ev3ph.addEvent(E1[v], omega, weight);
									Ec3ph.addEvent(E2[c], omega, weight);
									chi3ph.addEvent(omega, weight);
								}
							}
						}
					}
				}
			}
		}
		
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunchesMine)));
			logFlush();
		}
	}
	logPrintf("done.\n"); logFlush();
	
	Ev1.allReduce(MPIUtil::ReduceSum);
	Ec1.allReduce(MPIUtil::ReduceSum);
	Ev3.allReduce(MPIUtil::ReduceSum);
	Ec3.allReduce(MPIUtil::ReduceSum);
	Ev1ph.allReduce(MPIUtil::ReduceSum);
	Ec1ph.allReduce(MPIUtil::ReduceSum);
	Ev3ph.allReduce(MPIUtil::ReduceSum);
	Ec3ph.allReduce(MPIUtil::ReduceSum);
	chi1.allReduce(MPIUtil::ReduceSum);
	chi3.allReduce(MPIUtil::ReduceSum);
	chi1ph.allReduce(MPIUtil::ReduceSum);
	chi3ph.allReduce(MPIUtil::ReduceSum);
	
	//Apply frequency factors:
	for(size_t iOmega=0; iOmega<chi1.out.size(); iOmega++)
	{	double omega = chi1.Emin + chi1.dE * iOmega;
		double invOmega2 = 1./(omega*omega);
		double invOmega4 = invOmega2 * invOmega2;
		chi1.out[iOmega] *= invOmega2;
		chi3.out[iOmega] *= invOmega4;
		chi1ph.out[iOmega] *= invOmega2;
		chi3ph.out[iOmega] *= invOmega4;
		for(int iE=0; iE<Ec1.nE; iE++)
		{	int i = iOmega*Ec1.nE + iE;
			Ec1.out[i] *= invOmega2;
			Ev1.out[i] *= invOmega2;
			Ec3.out[i] *= invOmega4;
			Ev3.out[i] *= invOmega4;
			Ec1ph.out[i] *= invOmega2;
			Ev1ph.out[i] *= invOmega2;
			Ec3ph.out[i] *= invOmega4;
			Ev3ph.out[i] *= invOmega4;
		}
	}
	
	Ev1.print("hDistribAll-chi1.dat", 1./eV, 1./eV, 1.);
	Ec1.print("eDistribAll-chi1.dat", 1./eV, 1./eV, 1.);
	Ev3.print("hDistribAll-chi3.dat", 1./eV, 1./eV, 1.);
	Ec3.print("eDistribAll-chi3.dat", 1./eV, 1./eV, 1.);
	Ev1ph.print("hDistribAll-chi1ph.dat", 1./eV, 1./eV, 1.);
	Ec1ph.print("eDistribAll-chi1ph.dat", 1./eV, 1./eV, 1.);
	Ev3ph.print("hDistribAll-chi3ph.dat", 1./eV, 1./eV, 1.);
	Ec3ph.print("eDistribAll-chi3ph.dat", 1./eV, 1./eV, 1.);
	
	//Output chi1 and chi3:
	if(mpiUtil->isHead())
	{	ofstream ofs("chi13.dat");
		ofs << "#omega[eV] chi1 chi3[au] chi1ph chi3ph[au] ReEps ImEps modGammaMinus[au]\n";
		for(size_t iOmega=0; iOmega<chi1.out.size(); iOmega++)
		{	double omega = chi1.Emin + chi1.dE * iOmega;
			eps.setFrequency(omega, false);
			ofs << omega/eV << '\t'
				<< chi1.out[iOmega] << '\t'
				<< chi3.out[iOmega] << '\t'
				<< chi1ph.out[iOmega] << '\t'
				<< chi3ph.out[iOmega] << '\t'
				<< eps.epsilon.real() << '\t'
				<< eps.epsilon.imag() << '\t'
				<< eps.modGammaMinus << '\n';
		}
	}
	
	//Print convergence:
	for(Histogram& h: conv) h.allReduce(MPIUtil::ReduceSum);
	if(mpiUtil->isHead())
	{	ofstream ofs("bandConvergenceAll-chi3.dat");
		ofs << "#omega[eV]";
		for(size_t i=0; i<conv[0].out.size(); i++)
			ofs << '\t' << (conv[0].Emin + i*conv[0].dE)/eV;
		ofs << '\n';
		for(int b=0; b<nBands; b++)
		{	ofs << (b+1);
			for(double chi3cur: conv[b].out)
				ofs << '\t' << chi3cur;
			ofs << '\n';
		}
	}
	
	finalizeSystem();
	return 0;
}
