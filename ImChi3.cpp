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

//Lorentzian kernel for an odd function stored on postive frequencies alone:
inline double lorentzianOdd(double omega, double omega0, double breadth)
{	double breadthSq = std::pow(breadth,2);
	if(omega0 < 0.01) return 0.;
	return (breadth/M_PI) *
		( 1./(breadthSq + std::pow(omega-omega0, 2))
		- 1./(breadthSq + std::pow(omega+omega0, 2)) );
}

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of imaginary cubic susceptibility (ImChi3)", inputFilename, dryRun, printDefaults);

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
	Ahat[0] = vector3<complex>(1.,0.,0.); //cubic crystal, one polarization sufficient
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
	
	//Initalize line width of intermediate electronic states
	LineWidth lineWidth("Wannier/wannier", bs);

	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKpts, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	long nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	int nBands = bs.getStates(vector3<>()).nRows();
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	complex I(0,1);
	double prefac0 = bs.spinWeight * M_PI / (nKpts * fabs(det(bs.R))); //frequency independent part of prefac
	double prefac0ph = bs.spinWeight * M_PI / (nKpairs * fabs(det(bs.R))); //frequency independent part of prefac
	
	//Singularity extrapolation parameters
	double extrapCoeff[] = {-19./12, 13./3, -7./4 }; //account for constant, 1/eta and eta^2 dependence
	//double extrapCoeff[] = { -1, 2.}; //account for constant and 1/eta dependence
	//double extrapCoeff[] = { 1. }; //no extrapolation
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
	
	//Initialize unbroadened histograms
	double EplasmonTotMax = 2*EplasmonMax; //max on sum of two plasmon energies
	const double domega = T;
	Histogram chi1(0., domega, omegaMax), breadth1(0., domega, omegaMax);
	Histogram chi3(0., domega, omegaMax), breadth3(0., domega, omegaMax);
	Histogram chi1ph(0., domega, omegaMax), breadth1ph(0., domega, omegaMax), weight1ph(0., domega, omegaMax);
	Histogram chi3ph(0., domega, omegaMax), breadth3ph(0., domega, omegaMax), weight3ph(0., domega, omegaMax);
	Histogram2D chi1_E(-EplasmonTotMax, domega, EplasmonTotMax,  0., domega, omegaMax);
	Histogram2D chi3_E(-EplasmonTotMax, domega, EplasmonTotMax,  0., domega, omegaMax);
	Histogram2D chi1ph_E(-EplasmonTotMax, domega, EplasmonTotMax,  0., domega, omegaMax);
	Histogram2D chi3ph_E(-EplasmonTotMax, domega, EplasmonTotMax,  0., domega, omegaMax);
	int nomega = chi1.out.size();
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
			const matrix& P = Parr[ik][0];
			const diagMatrix& F = Farr[ik];
			
			//One-plasmon process (chi1)
			for(int v=0; v<nBands; v++) if(E[v]<10.*T)
			{	for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
				{	double omega = E[c] - E[v]; //energy conservation
					if(omega<domega || omega>=omegaMax) continue; //irrelevant event
					complex Meff = P(c,v);
					double weight = (prefac0/(omega*omega)) * (F[v]-F[c]) * Meff.norm(); //norm=abs^2
					chi1.addEvent(omega, weight);
					chi1_E.addEvent(E[v], omega, -weight);
					chi1_E.addEvent(E[c], omega, +weight);
					breadth1.addEvent(omega, weight*(ImE[c]+ImE[v]));
				}
			}
			
			//Two-plasmon process (chi3)
			for(int v=0; v<nBands; v++) if(E[v]<10.*T)
			{	for(int c=0; c<nBands; c++) if(E[c]>-10.*T)
				{	double omegaTot = E[c] - E[v]; //energy conservation
					double omega = 0.5*omegaTot; //only considering processes with equal plasmon energies
					if(omega<domega || omega>=omegaMax) continue; //irrelevant event
					//Effective matrix element
					complex Meff = 0.;
					for(int i=0; i<nBands; i++) // sum over the intermediate states
					{	complex Ei(E[i], ImE[i]);
						Meff += 2. * P(c,i) * P(i,v) / (Ei-E[v]-omega); //factor of 2 from exchanged term (identical for two plasmons with same mode)
					}
					double weight = (prefac0/(omega*omega)) * (F[v]-F[c]) * Meff.norm();
					//Include in statistics:
					chi3.addEvent(omega, weight);
					chi3_E.addEvent(E[v], omega, -weight);
					chi3_E.addEvent(E[c], omega, +weight);
					breadth3.addEvent(omega, weight*(ImE[c]+ImE[v])*0.5); //0.5 since energy conservation is on 2 omega
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
								if(omegaTot<domega || omegaTot>=omegaMax) continue; //irrelevant event
								double prefac = prefac0ph * (F1[v]-F2[c]) * (nPh + 0.5*(1.-ae)); //factors of omega added below
								
								//One-plasmon process (chi1ph)
								double omega = omegaTot;
								if(omega < omegaMax)
								{	std::vector<complex> Meff(nExtrap, 0.);
									for(int i=0; i<nBands; i++) // sum over the intermediate states
									{	for(int z=0; z<nExtrap; z++)
										{	complex iEta(0, (z+1)*eta);
											Meff[z] += 
												( P2(c,i) * gePh[alpha](i,v) / (E2[i]+iEta - (E2[c] - omega))
												+ gePh[alpha](c,i) * P1(i,v) / (E1[i]+iEta - (E1[v] + omega)) );
										}
									}
									double MeffSqExtrap = 0.;
									for(int z=0; z<nExtrap; z++)
										MeffSqExtrap += extrapCoeff[z] * Meff[z].norm();
									double weight = (prefac/(omega*omega)) * MeffSqExtrap;
									chi1ph.addEvent(omega, weight);
									chi1ph_E.addEvent(E1[v], omega, -weight);
									chi1ph_E.addEvent(E2[c], omega, +weight);
									breadth1ph.addEvent(omega, fabs(weight)*(ImE2[c]+ImE1[v]));
									weight1ph.addEvent(omega, fabs(weight));
								}
								
								//Two-plasmon process (chi3ph)
								omega = 0.5*omegaTot;
								if(omega < omegaMax)
								{	std::vector<complex> Meff(nExtrap, 0.);
									for(int a=0; a<nBands; a++) // sum over the first intermediate state
									for(int b=0; b<nBands; b++) // sum over the second intermediate state
									{	for(int z=0; z<nExtrap; z++)
										{	complex iEta(0, (z+1)*eta);
											complex iImE1a = complex(0,ImE1[a]);
											complex iImE2b = complex(0,ImE2[b]);
											Meff[z] += 
												( P2(c,b) * P2(b,a) * gePh[alpha](a,v) / ((E2[a]+iEta - (E2[c] - omegaTot)) * (E2[b]+iEta - (E2[c] - omega)))
												+ P2(c,b) * gePh[alpha](b,a) * P1(a,v) / ((E1[a]+iImE1a - (E1[v] + omega)) * (E2[b]+iImE2b - (E2[c] - omega)))
												+ gePh[alpha](c,b) * P1(b,a) * P1(a,v) / ((E1[a]+iEta - (E1[v] + omega)) * (E1[b]+iEta - (E1[v] + omegaTot))) );
										}
									}
									double MeffSqExtrap = 0.;
									for(int z=0; z<nExtrap; z++)
										MeffSqExtrap += extrapCoeff[z] * Meff[z].norm();
									double weight = (prefac/(omega*omega)) * MeffSqExtrap;
									chi3ph.addEvent(omega, weight);
									chi3ph_E.addEvent(E1[v], omega, -weight);
									chi3ph_E.addEvent(E2[c], omega, +weight);
									breadth3ph.addEvent(omega, fabs(weight)*(ImE2[c]+ImE1[v])*0.5); //0.5 since energy conservation is on 2 omega
									weight3ph.addEvent(omega, fabs(weight));
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
	
	chi1.allReduce(MPIUtil::ReduceSum);
	chi3.allReduce(MPIUtil::ReduceSum);
	chi1ph.allReduce(MPIUtil::ReduceSum);
	chi3ph.allReduce(MPIUtil::ReduceSum);
	chi1_E.allReduce(MPIUtil::ReduceSum);
	chi3_E.allReduce(MPIUtil::ReduceSum);
	chi1ph_E.allReduce(MPIUtil::ReduceSum);
	chi3ph_E.allReduce(MPIUtil::ReduceSum);
	breadth1.allReduce(MPIUtil::ReduceSum);
	breadth3.allReduce(MPIUtil::ReduceSum);
	breadth1ph.allReduce(MPIUtil::ReduceSum);
	breadth3ph.allReduce(MPIUtil::ReduceSum);
	weight1ph.allReduce(MPIUtil::ReduceSum);
	weight3ph.allReduce(MPIUtil::ReduceSum);
	logPrintf("done.\n"); logFlush();
	
	//Normalize the breadths:
	for(int iomega=0; iomega<nomega; iomega++)
	{	breadth1.out[iomega] = std::max(T, chi1.out[iomega] ? breadth1.out[iomega]/chi1.out[iomega] : 0.);
		breadth3.out[iomega] = std::max(T, chi3.out[iomega] ? breadth3.out[iomega]/chi3.out[iomega] : 0.);
		breadth1ph.out[iomega] = std::max(T, weight1ph.out[iomega] ? breadth1ph.out[iomega]/weight1ph.out[iomega] : 0.);
		breadth3ph.out[iomega] = std::max(T, weight3ph.out[iomega] ? breadth3ph.out[iomega]/weight3ph.out[iomega] : 0.);
	}
	breadth1.print("breadth1.dat", 1./eV, 1./eV);
	breadth3.print("breadth3.dat", 1./eV, 1./eV);
	breadth1ph.print("breadth1ph.dat", 1./eV, 1./eV);
	breadth3ph.print("breadth3ph.dat", 1./eV, 1./eV);

	//Apply broadening (chi* is unbroadened, Chi* is broadened):
	Histogram Chi1(0., domega, EplasmonMax);
	Histogram Chi3(0., domega, EplasmonMax);
	Histogram Chi1ph(0., domega, EplasmonMax);
	Histogram Chi3ph(0., domega, EplasmonMax);
	Histogram2D Chi1_E(-EplasmonTotMax, domega, EplasmonTotMax,  0., domega, EplasmonMax);
	Histogram2D Chi3_E(-EplasmonTotMax, domega, EplasmonTotMax,  0., domega, EplasmonMax);
	Histogram2D Chi1ph_E(-EplasmonTotMax, domega, EplasmonTotMax,  0., domega, EplasmonMax);
	Histogram2D Chi3ph_E(-EplasmonTotMax, domega, EplasmonTotMax,  0., domega, EplasmonMax);
	int iomegaStart, iomegaStop; TaskDivision(nomega, mpiUtil).myRange(iomegaStart, iomegaStop);
	logPrintf("Applying broadening ... "); logFlush();
	for(int iomega=iomegaStart; iomega<iomegaStop; iomega++) //input frequency grid split over MPI
	{	double omegaCur = iomega*domega;
		double b1 = breadth1.out[iomega];
		double b3 = breadth3.out[iomega];
		double b1ph = breadth1ph.out[iomega];
		double b3ph = breadth3ph.out[iomega];
		for(size_t jomega=0; jomega<Chi1.out.size(); jomega++) //output frequency grid
		{	double omega = jomega*domega, invOmegaSq = std::pow(std::max(omega,0.5*domega), -2);
			double kernel1 = lorentzianOdd(omega, omegaCur, b1) * domega;
			double kernel3 = lorentzianOdd(omega, omegaCur, b3) * domega * invOmegaSq; //omega^4 instead of omega^2 in chi3
			double kernel1ph = lorentzianOdd(omega, omegaCur, b1ph) * domega;
			double kernel3ph = lorentzianOdd(omega, omegaCur, b3ph) * domega * invOmegaSq; //omega^4 instead of omega^2 in chi3
			Chi1.out[jomega] += kernel1 * chi1.out[iomega];
			Chi3.out[jomega] += kernel3 * chi3.out[iomega];
			Chi1ph.out[jomega] += kernel1ph * chi1ph.out[iomega];
			Chi3ph.out[jomega] += kernel3ph * chi3ph.out[iomega];
			//Carrier distributions:
			const int nE = Chi1_E.nE; assert(nE == chi1_E.nE);
			for(int iE=0; iE<nE; iE++)
			{	int iOE = iomega*nE + iE;
				int jOE = jomega*nE + iE;
				Chi1_E.out[jOE] += kernel1 * chi1_E.out[iOE];
				Chi3_E.out[jOE] += kernel3 * chi3_E.out[iOE];
				Chi1ph_E.out[jOE] += kernel1ph * chi1ph_E.out[iOE];
				Chi3ph_E.out[jOE] += kernel3ph * chi3ph_E.out[iOE];
			}
		}
	}
	Chi1.allReduce(MPIUtil::ReduceSum);
	Chi3.allReduce(MPIUtil::ReduceSum);
	Chi1ph.allReduce(MPIUtil::ReduceSum);
	Chi3ph.allReduce(MPIUtil::ReduceSum);
	Chi1_E.allReduce(MPIUtil::ReduceSum); Chi1_E.print("carrierDistribAll-chi1.dat", 1./eV, 1./eV, 1.);
	Chi3_E.allReduce(MPIUtil::ReduceSum); Chi3_E.print("carrierDistribAll-chi3.dat", 1./eV, 1./eV, 1.);
	Chi1ph_E.allReduce(MPIUtil::ReduceSum); Chi1ph_E.print("carrierDistribAll-chi1ph.dat", 1./eV, 1./eV, 1.);
	Chi3ph_E.allReduce(MPIUtil::ReduceSum); Chi3ph_E.print("carrierDistribAll-chi3ph.dat", 1./eV, 1./eV, 1.);
	logPrintf("done.\n"); logFlush();
	
	//Output chi1 and chi3:
	if(mpiUtil->isHead())
	{	ofstream ofs("chi13.dat");
		ofs << "#omega[eV] chi1 chi3[au] chi1ph chi3ph[au] ReEps ImEps modGammaMinus[au]\n";
		for(size_t iOmega=0; iOmega<Chi1.out.size(); iOmega++)
		{	double omega = Chi1.Emin + Chi1.dE * iOmega;
			eps.setFrequency(omega, false);
			ofs << omega/eV << '\t'
				<< Chi1.out[iOmega] << '\t'
				<< Chi3.out[iOmega] << '\t'
				<< Chi1ph.out[iOmega] << '\t'
				<< Chi3ph.out[iOmega] << '\t'
				<< eps.epsilon.real() << '\t'
				<< eps.epsilon.imag() << '\t'
				<< eps.modGammaMinus << '\n';
		}
	}
	
	finalizeSystem();
	return 0;
}
