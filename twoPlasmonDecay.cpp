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
	const double kPhi = inputMap.get("kPhi");
	const double EplasmonMax = inputMap.get("EplasmonMax") * eV;
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	const double Squant = std::pow(10*Angstrom, 2); //1 nm^2
	
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
	Epsilon eps("Wannier/epsilon.dat");
	
	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(2); //contract for zHat and kHat, which will be combined in a frequency dependent way
	Ahat[0] = vector3<complex>(cos(kPhi), sin(kPhi), 0.); //kHat
	Ahat[1] = vector3<complex>(0., 0., 1.); //zHat
	BandStruct bs("Wannier/wannier", mu, spinWeight, string(), Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(bunchSize);

	//Initalize line width of intermediate electronic states
	LineWidth lineWidth("Wannier/wannier", bs);

	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	int nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	int nBands = bs.getStates(vector3<>()).nRows();
	complex I(0,1);
	double gammaPrefac0 = (0.5*spinWeight) * std::pow(M_PI,3) / (nKpts * 2*fabs(det(R)) * Squant); //frequency independent part of prefac
	
	//Initialize histograms
	double gaussMargin = 5*T;
	double fermiMargin = 10*T;
	double EplasmonTotMax = 2*EplasmonMax; //max on sum of two plasmon energies
	Histogram2D Ev(-EplasmonTotMax-gaussMargin, T, fermiMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D Ec(-fermiMargin, T, EplasmonTotMax+gaussMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram Gamma(gaussMargin, T, EplasmonMax-gaussMargin);
	std::vector<Histogram> conv(nBands,  Gamma); //empty-state convergence

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
		
		//Direct transitions:
		for(int ik=0; ik<bunchSize; ik++)
		{	const diagMatrix& E = Earr[ik];
			const diagMatrix& ImE = ImEarr[ik];
			const std::vector<matrix>& P = Parr[ik];
			diagMatrix F = E; //convert to fillings:
			for(double& f: F)
			{	double e = f/T; //E/T actually
				f = (e>30 ? exp(-e) : 1./(1. + exp(e))); //avoid overflow issues
			}
			
			for(int v=0; v<nBands; v++) if(E[v]<10.*T)
			{	for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
				{	double omegaTot = E[c] - E[v]; //energy conservation
					double omega = 0.5*omegaTot; //only considering processes with equal plasmon energies
					if(omega<=0 || omega>=EplasmonMax) continue; //irrelevant event
					eps.setFrequency(omega, false);
					double gammaPrefac = gammaPrefac0 / ( (2.*eps.modGammaMinus) * std::pow(omega*eps.Lquant,2) ); //linking two plasmon energies together
					if(!std::isfinite(gammaPrefac) || gammaPrefac<0.) continue; //avoid over-damped region
					double weightPrefac = gammaPrefac * F[v] * (1.-F[c]); //include occupation factors
					//Effective matrix element
					complex Meff = 0.; double weight = 0.;
					for(int i=0; i<nBands; i++) // sum over the intermediate states
					{	complex Ei(E[i], ImE[i]);
						complex AdotPiv = P[0](i,v) - I*(eps.k/eps.modGammaMinus)*P[1](i,v);
						complex AdotPci = P[0](c,i) - I*(eps.k/eps.modGammaMinus)*P[1](c,i);
						Meff += (2. * (1.-F[i]) * AdotPci * AdotPiv) / (Ei-E[v]-omega); //factor of 2 from exchanged term (identical for two plasmons with same mode)
						weight = weightPrefac * Meff.norm(); //norm = abs^2;
						conv[i].addEvent(omega, weight); //estimate based on truncating to i bands
					}
					//Include in statistics:
					Ev.addEvent(E[v], omega, weight);
					Ec.addEvent(E[c], omega, weight);
					Gamma.addEvent(omega, weight);
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
	
	Ev.allReduce(MPIUtil::ReduceSum); Ev.print("hDistribAll-twoplasmon.dat", 1./eV, 1./eV, 1.);
	Ec.allReduce(MPIUtil::ReduceSum); Ec.print("eDistribAll-twoplasmon.dat", 1./eV, 1./eV, 1.);
	Gamma.allReduce(MPIUtil::ReduceSum); Gamma.print("GammaAll-twoplasmon.dat", 1./eV, 1./eV);
	
	//Print phonon convergence:
	for(Histogram& h: conv) h.allReduce(MPIUtil::ReduceSum);
	if(mpiUtil->isHead())
	{	ofstream ofs("bandConvergenceAll-twoplasmon.dat");
		ofs << "#omega[eV]";
		for(size_t i=0; i<conv[0].out.size(); i++)
			ofs << '\t' << (conv[0].Emin + i*conv[0].dE)/eV;
		ofs << '\n';
		for(int b=0; b<nBands; b++)
		{	ofs << (b+1);
			for(double Gamma: conv[b].out)
				ofs << '\t' << Gamma/eV;
			ofs << '\n';
		}
	}
	
	finalizeSystem();
	return 0;
}
