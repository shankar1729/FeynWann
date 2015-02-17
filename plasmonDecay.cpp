#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "Histogram.h"
#include "Epsilon.h"
#include "LineWidth.h"
#include "InputMap.h"
#include "Units.h"

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of all single-plasmon decay processes", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const double kPhi = inputMap.get("kPhi");
	const double EplasmonMax = inputMap.get("EplasmonMax") * eV;
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("kPhi = %lg\n", kPhi);
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
	Epsilon eps("epsilon.txt");

	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(2); //contract for zHat and kHat, which will be combined ina frequency dependent way
	Ahat[0] = vector3<complex>(cos(kPhi), sin(kPhi), 0.); //kHat
	Ahat[1] = vector3<complex>(0., 0., 1.); //zHat
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE", Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);
	
	//Initalize line width of intermediate electronic states
	LineWidth lineWidth("Wannier/wannier", bs);

	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	int nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	int nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	int nBands = bs.getStates(vector3<>()).nRows();
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	complex I(0,1);
	double directPrefac0 = (0.5*spinWeight) * std::pow(M_PI,2) / (nKpts * fabs(det(R))); //frequency independent part of prefac
	double phononPrefac0 = (0.5*spinWeight) * std::pow(M_PI,2) / (nKpairs*4*fabs(det(R))); //frequency independent part of prefac

	//Initialize histograms
	double gaussMargin = 5*T;
	double fermiMargin = 10*T;
	Histogram2D EvDirect(-EplasmonMax-gaussMargin, T, fermiMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D EvPhonon(-EplasmonMax-gaussMargin, T, fermiMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D EcDirect(-fermiMargin, T, EplasmonMax+gaussMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D EcPhonon(-fermiMargin, T, EplasmonMax+gaussMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram GammaDirect(gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram GammaPhonon(gaussMargin, T, EplasmonMax-gaussMargin);
	std::vector<Histogram> convPhonon(nBands,  GammaPhonon); //empty-state convergence for phonon
	
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
			const diagMatrix& F = Farr[ik];
			const std::vector<matrix>& P = Parr[ik];
			for(int v=0; v<nBands; v++) if(E[v]<10.*T)
			{	for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
				{	double omega = E[c] - E[v]; //energy conservation
					if(omega<=0 || omega>=EplasmonMax) continue; //irrelevant event
					eps.setFrequency(omega, false);
					complex AdotPcv = P[0](c,v) - I*(eps.k/eps.modGammaMinus)*P[1](c,v);
					double directPrefac = directPrefac0 / (eps.modGammaMinus * omega * eps.Lquant);
					if(!std::isfinite(directPrefac) || directPrefac<0.) continue; //avoid over-damped region
					double weight = directPrefac * F[v] * (1.-F[c]) * AdotPcv.norm(); //norm=abs^2
					EvDirect.addEvent(E[v], omega, weight);
					EcDirect.addEvent(E[c], omega, weight);
					GammaDirect.addEvent(omega, weight);
				}
			}
		}
		
		//Phonon-assisted transitions:
		for(int ik1=0; ik1<bunchSize; ik1++)
		{	const diagMatrix& E1 = Earr[ik1];
			const diagMatrix& ImE1 = ImEarr[ik1];
			const diagMatrix& F1 = Farr[ik1];
			const std::vector<matrix>& P1 = Parr[ik1];
			//phonon matrix elements for ik1 with rest of bunch:
			std::vector<matrix> MePhArr[bunchSize];
			bs.setPhononMatElemArray(kArr[ik1], kArr, MePhArr);
			//Loop over second k-point:
			for(int ik2=0; ik2<bunchSize; ik2++) if(ik2!=ik1) //avoid gamma-point phonon singularity
			{	const std::vector<matrix>& MePh = MePhArr[ik2];
				diagMatrix omegaPh = bs.getPhononModes(kArr[ik1] - kArr[ik2]);
				const diagMatrix& E2 = Earr[ik2];
				const diagMatrix& ImE2 = ImEarr[ik2];
				const diagMatrix& F2 = Farr[ik2];
				const std::vector<matrix>& P2 = Parr[ik2];
				//Loops over bands and phonon modes:
				for(int v=0; v<nBands; v++) if(E1[v]<10.*T)
				{	for(int c=0; c<nBands; c++) if(E2[c]>-10.*T)
					{	for(int alpha=0; alpha<nModes; alpha++)
						{	for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
							{	double omega = E2[c] - E1[v] - ae*omegaPh[alpha]; //energy conservation
								if(omega<=0 || omega>=EplasmonMax) continue; //irrelevant event
								eps.setFrequency(omega, false);
								double phononPrefac = phononPrefac0 / (eps.modGammaMinus * omega * eps.Lquant);
								if(!std::isfinite(phononPrefac) || phononPrefac<0.) continue; //avoid over-damped region
								double g_kPh = 1./(exp(omegaPh[alpha]/T) - 1.);
								double weightPrefac = phononPrefac * F1[v] * (1.-F2[c]) * (g_kPh + 0.5*(1.-ae))/omegaPh[alpha];
								// Effective matrix elements
								complex Meff = 0.; double weight = 0.;
								for(int i=0; i<nBands; i++) // sum over the intermediate states
								{	complex E1i(E1[i], ImE1[i]);
									complex E2i(E2[i], ImE2[i]);
									complex AdotP1iv = P1[0](i,v) - I*(eps.k/eps.modGammaMinus)*P1[1](i,v);
									complex AdotP2ci = P2[0](c,i) - I*(eps.k/eps.modGammaMinus)*P2[1](c,i);
									Meff += 
										( AdotP2ci * (1.-F2[i]) * MePh[alpha](i,v) / (E2i - (E2[c] - omega))
										+ MePh[alpha](c,i) * (1.-F1[i]) * AdotP1iv / (E1i - (E1[v] + omega)) );
									weight =  weightPrefac * Meff.norm();  //norm = abs^2
									convPhonon[i].addEvent(omega, weight); //estimate based on truncating to i bands
								}
								//Results using all available bands:
								EvPhonon.addEvent(E1[v], omega, weight);
								EcPhonon.addEvent(E2[c], omega, weight);
								GammaPhonon.addEvent(omega, weight);
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
	
	EvDirect.allReduce(MPIUtil::ReduceSum); EvDirect.print("hDistribAll-direct.dat", 1./eV, 1./eV, 1.);
	EcDirect.allReduce(MPIUtil::ReduceSum); EcDirect.print("eDistribAll-direct.dat", 1./eV, 1./eV, 1.);
	GammaDirect.allReduce(MPIUtil::ReduceSum); GammaDirect.print("GammaAll-direct.dat", 1./eV, 1./eV);
	
	EvPhonon.allReduce(MPIUtil::ReduceSum); EvPhonon.print("hDistribAll-phonon.dat", 1./eV, 1./eV, 1.);
	EcPhonon.allReduce(MPIUtil::ReduceSum); EcPhonon.print("eDistribAll-phonon.dat", 1./eV, 1./eV, 1.);
	GammaPhonon.allReduce(MPIUtil::ReduceSum); GammaPhonon.print("GammaAll-phonon.dat", 1./eV, 1./eV);
	
	//Print experimental linewidth:
	if(mpiUtil->isHead())
	{	ofstream ofs("GammaAll-expt.dat");
		for(double omega = gaussMargin; omega <= EplasmonMax-gaussMargin; omega += T)
		{	eps.setFrequency(omega, false);
			complex arg = eps.epsilon / (eps.epsilon + 1);
			double omegaIm = fabs(sin(0.5*atan2(imag(arg),real(arg)))) * omega;
			ofs << omega/eV << '\t' << omegaIm/eV << '\n';
		}
	}
	
	//Print phonon convergence:
	for(Histogram& h: convPhonon) h.allReduce(MPIUtil::ReduceSum);
	if(mpiUtil->isHead())
	{	ofstream ofs("bandConvergenceAll-phonon.dat");
		ofs << "#omega[eV]";
		for(size_t i=0; i<convPhonon[0].out.size(); i++)
			ofs << '\t' << (convPhonon[0].Emin + i*convPhonon[0].dE)/eV;
		ofs << '\n';
		for(int b=0; b<nBands; b++)
		{	ofs << (b+1);
			for(double Gamma: convPhonon[b].out)
				ofs << '\t' << Gamma/eV;
			ofs << '\n';
		}
	}
	
	finalizeSystem();
	return 0;
}
