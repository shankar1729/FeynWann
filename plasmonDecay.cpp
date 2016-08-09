#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include "BandStruct.h"
#include "LineWidth.h"
#include "Histogram.h"
#include "Epsilon.h"
#include "InputMap.h"
#include "Units.h"

//Lorentzian kernel for an odd function stored on postive frequencies alone:
inline double lorentzianOdd(double omega, double omega0, double breadth)
{	double breadthSq = std::pow(breadth,2);
	return (breadth/M_PI) *
		( 1./(breadthSq + std::pow(omega-omega0, 2))
		- 1./(breadthSq + std::pow(omega+omega0, 2)) );
}

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of all single-plasmon decay processes", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	long nKpts = inputMap.get("nKpts");
	const double EplasmonMax = inputMap.get("EplasmonMax") * eV;
	const double T = inputMap.get("T") * Kelvin;
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKpts = %ld\n", nKpts);
	logPrintf("EplasmonMax = %lg\n", EplasmonMax);
	logPrintf("T = %lg\n", T);

	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(1);
	Ahat[0] = vector3<complex>(1., 0., 0.); //cubic symmetry => one projection sufficient (also since not looking at momentum distribution)
	BandStruct bs("Wannier/totalE", "Wannier/wannier", true, Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);
	
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Initialize dielectric model:
	Epsilon eps("Wannier/epsilon.dat");

	//Initalize line width of electronic states
	LineWidth lineWidth("Wannier/wannier", bs);

	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKpts, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	long nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	int nBands = bs.getStates(vector3<>()).nRows();
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	double phononPrefac0 = 4 * std::pow(M_PI,2) * bs.spinWeight / (nKpairs*fabs(det(bs.R))); //frequency independent part of prefac
	double directPrefac0 = 4 * std::pow(M_PI,2) * bs.spinWeight / (nKpts*fabs(det(bs.R))); //frequency independent part of prefac

	//Singularity extrapolation parameters
	double extrapCoeff[] = {-19./12, 13./3, -7./4 }; //account for constant, 1/eta and eta^2 dependence
	//double extrapCoeff[] = { -1, 2.}; //account for constant and 1/eta dependence
	const int nExtrap = sizeof(extrapCoeff)/sizeof(double);
	const double eta = 0.1*eV;
	
	//Initialize frequency grid:
	double omegaMax = 0.;
	for(int i=0; i<10; i++)
	{	std::vector<vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		for(const diagMatrix& E: Earr)
			omegaMax = std::max(omegaMax, E.back()-E.front());
	}
	mpiUtil->allReduce(omegaMax, MPIUtil::ReduceMax);
	
	//Initialize unbroadened histograms:
	const double domega = T;
	Histogram ImEpsDirect(0, domega, omegaMax), breadthDirect(0, domega, omegaMax);
	Histogram ImEpsPhonon(0, domega, omegaMax), breadthPhonon(0, domega, omegaMax), weightPhonon(0, domega, omegaMax);
	Histogram2D ImEpsDirect_E(-EplasmonMax, domega, EplasmonMax,  0, domega, omegaMax); //ImEps resolved by carrier density
	Histogram2D ImEpsPhonon_E(-EplasmonMax, domega, EplasmonMax,  0, domega, omegaMax);
	int nomega = ImEpsDirect.out.size();
	logPrintf("Initialized frequency grid: 0 to %lg eV with %d points.\n", (domega*(nomega-1))/eV, nomega);
	
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
			const diagMatrix& F = Farr[ik];
			const std::vector<matrix>& P = Parr[ik];
			for(int v=0; v<nBands; v++) if(E[v]<10.*T)
			{	for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
				{	double omega = E[c] - E[v]; //energy conservation
					if(omega<domega || omega>=omegaMax) continue; //irrelevant event
					double weight = (directPrefac0/(omega*omega)) * (F[v]-F[c]) * P[0](c,v).norm(); //norm=abs^2
					ImEpsDirect.addEvent(omega, weight);
					ImEpsDirect_E.addEvent(E[v], omega, -weight); //hole
					ImEpsDirect_E.addEvent(E[c], omega, +weight); //electron
					breadthDirect.addEvent(omega, weight*(ImE[c]+ImE[v]));
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
								if(omega<domega || omega>=omegaMax) continue; //irrelevant event
								double nPh = 1./(exp(omegaPh[alpha]/T) - 1.);
								//Effective matrix elements
								std::vector<complex> Meff(nExtrap, 0.);
								for(int i=0; i<nBands; i++) // sum over the intermediate states
								{	for(int z=0; z<nExtrap; z++)
									{	complex iEta(0, (z+1)*eta);
										Meff[z] += 
											( P2[0](c,i) * gePh[alpha](i,v) / (E2[i]+iEta - (E2[c] - omega))
											+ gePh[alpha](c,i) * P1[0](i,v) / (E1[i]+iEta - (E1[v] + omega)) );
									}
								}
								//Singularity extrapolation:
								double MeffSqExtrap = 0.;
								for(int z=0; z<nExtrap; z++)
									MeffSqExtrap += extrapCoeff[z] * Meff[z].norm();
								double weight = (phononPrefac0/(omega*omega)) * (F1[v]-F2[c]) * (nPh + 0.5*(1.-ae)) * MeffSqExtrap;
								//Histogram:
								ImEpsPhonon.addEvent(omega, weight);
								ImEpsPhonon_E.addEvent(E1[v], omega, -weight); //hole
								ImEpsPhonon_E.addEvent(E2[c], omega, +weight); //electron
								breadthPhonon.addEvent(omega, fabs(weight)*(ImE2[c]+ImE1[v]));
								weightPhonon.addEvent(omega, fabs(weight)); //different from ImEpsPhonon, since weight can be negative due to singularity extrapolation
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
	ImEpsDirect.allReduce(MPIUtil::ReduceSum);
	ImEpsPhonon.allReduce(MPIUtil::ReduceSum);
	ImEpsDirect_E.allReduce(MPIUtil::ReduceSum);
	ImEpsPhonon_E.allReduce(MPIUtil::ReduceSum);
	breadthDirect.allReduce(MPIUtil::ReduceSum);
	breadthPhonon.allReduce(MPIUtil::ReduceSum);
	weightPhonon.allReduce(MPIUtil::ReduceSum);
	logPrintf("done.\n"); logFlush();
	
	//Normalize the breadths:
	for(int iomega=0; iomega<nomega; iomega++)
	{	breadthDirect.out[iomega] = std::max(T, ImEpsDirect.out[iomega] ? breadthDirect.out[iomega]/ImEpsDirect.out[iomega] : 0.);
		breadthPhonon.out[iomega] = std::max(T, weightPhonon.out[iomega] ? breadthPhonon.out[iomega]/weightPhonon.out[iomega] : 0.);
	}
	breadthDirect.print("breadth-direct.dat", 1./eV, 1./eV);
	breadthPhonon.print("breadth-phonon.dat", 1./eV, 1./eV);
	
	//Print experimental linewidth:
	Histogram imEpsToGamma(0, domega, EplasmonMax);
	if(mpiUtil->isHead())
	{	ofstream ofs("GammaAll-expt.dat");
		for(size_t jomega=0; jomega<imEpsToGamma.out.size(); jomega++)
		{	double omega = jomega ? jomega*domega : 1e-3*domega; //avoid exact zero in calculation
			eps.setFrequency(omega, false);
			double Gamma = eps.exptLinewidth();
			imEpsToGamma.out[jomega] = Gamma / eps.epsilon.imag();
			omega = jomega*domega; //output exact zero
			ofs << omega/eV << '\t' << Gamma/eV << '\n';
		}
	}
	mpiUtil->bcast(imEpsToGamma.out.data(), imEpsToGamma.out.size());
	
	//Convert ImEps to Gamma with broadening:
	Histogram GammaDirect(0, domega, EplasmonMax);
	Histogram GammaPhonon(0, domega, EplasmonMax);
	Histogram2D GammaDirect_E(-EplasmonMax, domega, EplasmonMax,  0, domega, EplasmonMax);
	Histogram2D GammaPhonon_E(-EplasmonMax, domega, EplasmonMax,  0, domega, EplasmonMax);
	int iomegaStart, iomegaStop; TaskDivision(nomega, mpiUtil).myRange(iomegaStart, iomegaStop);
	logPrintf("Applying broadening ... "); logFlush();
	for(int iomega=iomegaStart; iomega<iomegaStop; iomega++) //input frequency grid split over MPI
	{	double omegaCur = iomega*domega;
		double bDirect = breadthDirect.out[iomega];
		double bPhonon = breadthPhonon.out[iomega];
		for(size_t jomega=0; jomega<GammaDirect.out.size(); jomega++) //output frequency grid
		{	double omega = jomega*domega;
			double ieToGamma = imEpsToGamma.out[jomega];
			double kernelDirect = ieToGamma * lorentzianOdd(omega, omegaCur, bDirect) * domega;
			double kernelPhonon = ieToGamma * lorentzianOdd(omega, omegaCur, bPhonon) * domega;
			GammaDirect.out[jomega] += kernelDirect * ImEpsDirect.out[iomega];
			GammaPhonon.out[jomega] += kernelPhonon * ImEpsPhonon.out[iomega];
			//Carrier distributions:
			const int nE = GammaDirect_E.nE; assert(nE == ImEpsDirect_E.nE);
			for(int iE=0; iE<nE; iE++)
			{	int iOE = iomega*nE + iE;
				int jOE = jomega*nE + iE;
				GammaDirect_E.out[jOE] += kernelDirect * ImEpsDirect_E.out[iOE];
				GammaPhonon_E.out[jOE] += kernelPhonon * ImEpsPhonon_E.out[iOE];
			}
		}
	}
	GammaDirect.allReduce(MPIUtil::ReduceSum); GammaDirect.print("GammaAll-direct.dat", 1./eV, 1./eV);
	GammaPhonon.allReduce(MPIUtil::ReduceSum); GammaPhonon.print("GammaAll-phonon.dat", 1./eV, 1./eV);
	GammaDirect_E.allReduce(MPIUtil::ReduceSum); GammaDirect_E.print("carrierDistribAll-direct.dat", 1./eV, 1./eV, 1.);
	GammaPhonon_E.allReduce(MPIUtil::ReduceSum); GammaPhonon_E.print("carrierDistribAll-phonon.dat", 1./eV, 1./eV, 1.);
	logPrintf("done.\n"); logFlush();
	
	finalizeSystem();
	return 0;
}
