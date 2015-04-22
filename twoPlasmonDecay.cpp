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
	double prefac0 = spinWeight * M_PI / (nKpts * fabs(det(R))); //frequency independent part of prefac
	
	//Initialize histograms
	double gaussMargin = 5*T;
	double fermiMargin = 10*T;
	double EplasmonTotMax = 2*EplasmonMax; //max on sum of two plasmon energies
	Histogram2D Ev1(-EplasmonTotMax-gaussMargin, T, fermiMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D Ec1(-fermiMargin, T, EplasmonTotMax+gaussMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D Ev3(-EplasmonTotMax-gaussMargin, T, fermiMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram2D Ec3(-fermiMargin, T, EplasmonTotMax+gaussMargin,  gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram chi1(gaussMargin, T, EplasmonMax-gaussMargin);
	Histogram chi3(gaussMargin, T, EplasmonMax-gaussMargin);
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
			
			//One-plasmon process (chi1)
			for(int v=0; v<nBands; v++) if(E[v]<10.*T)
			{	for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
				{	double omega = E[c] - E[v]; //energy conservation
					if(omega<=0 || omega>=EplasmonMax) continue; //irrelevant event
					double prefac = (F[v]-F[c]) * prefac0; //note omega^-2 factor added later for best histogramming
					complex Pcv = P[0](c,v);
					double weight = prefac * Pcv.norm(); //norm=abs^2
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
						complex Piv = P[0](i,v);
						complex Pci = P[0](c,i);
						Meff += 2. * Pci * Piv / (Ei-E[v]-omega); //factor of 2 from exchanged term (identical for two plasmons with same mode)
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
	chi1.allReduce(MPIUtil::ReduceSum);
	chi3.allReduce(MPIUtil::ReduceSum);
	
	//Apply frequency factors:
	for(size_t iOmega=0; iOmega<chi1.out.size(); iOmega++)
	{	double omega = chi1.Emin + chi1.dE * iOmega;
		double invOmega2 = 1./(omega*omega);
		double invOmega4 = invOmega2 * invOmega2;
		chi1.out[iOmega] *= invOmega2;
		chi3.out[iOmega] *= invOmega4;
		for(int iE=0; iE<Ec1.nE; iE++)
		{	int i = iOmega*Ec1.nE + iE;
			Ec1.out[i] *= invOmega2;
			Ev1.out[i] *= invOmega2;
			Ec3.out[i] *= invOmega4;
			Ev3.out[i] *= invOmega4;
		}
	}
	
	Ev1.print("hDistribAll-chi1.dat", 1./eV, 1./eV, 1.);
	Ec1.print("eDistribAll-chi1.dat", 1./eV, 1./eV, 1.);
	Ev3.print("hDistribAll-chi3.dat", 1./eV, 1./eV, 1.);
	Ec3.print("eDistribAll-chi3.dat", 1./eV, 1./eV, 1.);
	
	//Output chi1 and chi3:
	if(mpiUtil->isHead())
	{	ofstream ofs("chi13.dat");
		ofs << "#omega[eV] chi1 chi3[au] ReEps ImEps modGammaMinus[au]\n";
		for(size_t iOmega=0; iOmega<chi1.out.size(); iOmega++)
		{	double omega = chi1.Emin + chi1.dE * iOmega;
			eps.setFrequency(omega, false);
			ofs << omega/eV << '\t'
				<< chi1.out[iOmega] << '\t'
				<< chi3.out[iOmega] << '\t'
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
