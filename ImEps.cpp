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

//#define PHONON_ENABLED //comment out to only calculate direct (faster)

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of Im(eps)", inputFilename, dryRun, printDefaults);

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
	std::vector< vector3<complex> > Ahat(1); //assume cubic symmetry and only calculate x-axis
	Ahat[0] = vector3<complex>(1., 0., 0.);
	#ifdef PHONON_ENABLED
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE", Ahat);
	#else
	BandStruct bs("Wannier/wannier", mu, spinWeight, string(), Ahat);
	#endif
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);
	
	//Initalize line width of intermediate electronic states
	#ifdef PHONON_ENABLED
	LineWidth lineWidth("Wannier/wannier", bs);
	#endif
	
	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	int nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	int nBands = bs.getStates(vector3<>()).nRows();
	#ifdef PHONON_ENABLED
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	int nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	double phononPrefac0 = 4 * std::pow(M_PI,2) * spinWeight / (nKpairs*fabs(det(R))); //frequency independent part of prefac
	#endif
	double directPrefac0 = 4 * std::pow(M_PI,2) * spinWeight / (nKpts*fabs(det(R))); //frequency independent part of prefac

	//Initialize histograms
	double gaussMargin = 5*T;
	Histogram ImEpsDirect(gaussMargin, T, EplasmonMax-gaussMargin);
	#ifdef PHONON_ENABLED
	Histogram ImEpsPhonon(gaussMargin, T, EplasmonMax-gaussMargin);
	#endif
	
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
		#ifdef PHONON_ENABLED
		std::vector<diagMatrix> ImEarr = lineWidth(kArr);
		#endif
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
					double weight = (directPrefac0/(omega*omega)) * F[v] * (1.-F[c]) * P[0](c,v).norm(); //norm=abs^2
					ImEpsDirect.addEvent(omega, weight);
				}
			}
		}
		
		//Phonon-assisted transitions:
		#ifdef PHONON_ENABLED
		for(int ik1=0; ik1<bunchSize; ik1++)
		{	const diagMatrix& E1 = Earr[ik1];
			const diagMatrix& ImE1 = ImEarr[ik1];
			const diagMatrix& F1 = Farr[ik1];
			const std::vector<matrix>& P1 = Parr[ik1];
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
				const std::vector<matrix>& P2 = Parr[ik2];
				//Loops over bands and phonon modes:
				for(int v=0; v<nBands; v++) if(E1[v]<10.*T)
				{	for(int c=0; c<nBands; c++) if(E2[c]>-10.*T)
					{	for(int alpha=0; alpha<nModes; alpha++)
						{	for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
							{	double omega = E2[c] - E1[v] - ae*omegaPh[alpha]; //energy conservation
								if(omega<=0 || omega>=EplasmonMax) continue; //irrelevant event
								double nPh = 1./(exp(omegaPh[alpha]/T) - 1.);
								double weightPrefac = (phononPrefac/(omega*omega)) * F1[v] * (1.-F2[c]) * (nPh + 0.5*(1.-ae));
								// Effective matrix elements
								complex Meff = 0.; double weight = 0.;
								for(int i=0; i<nBands; i++) // sum over the intermediate states
								{	complex E1i(E1[i], ImE1[i]);
									complex E2i(E2[i], ImE2[i]);
									Meff += 
										( P2[0](c,i) * gePh[alpha](i,v) / (E2i - (E2[c] - omega))
										+ gePh[alpha](c,i) * P1[0](i,v) / (E1i - (E1[v] + omega)) );
									weight =  weightPrefac * Meff.norm();  //norm = abs^2
								}
								ImEpsPhonon.addEvent(omega, weight);
							}
						}
					}
				}
			}
		}
		#endif
		
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunchesMine)));
			logFlush();
		}
	}
	logPrintf("done.\n"); logFlush();
	
	ImEpsDirect.allReduce(MPIUtil::ReduceSum); ImEpsDirect.print("ImEpsAll-direct.dat", 1./eV, 1.);
	#ifdef PHONON_ENABLED
	ImEpsPhonon.allReduce(MPIUtil::ReduceSum); ImEpsPhonon.print("ImEpsAll-phonon.dat", 1./eV, 1.);
	#endif
	
	//Print experimental linewidth:
	if(mpiUtil->isHead())
	{	ofstream ofs("ImEpsAll-expt.dat");
		for(double omega = gaussMargin; omega <= EplasmonMax-gaussMargin; omega += T)
		{	eps.setFrequency(omega, false);
			ofs << omega/eV << '\t' << imag(eps.epsilon) << '\n';
		}
	}
	
	finalizeSystem();
	return 0;
}
