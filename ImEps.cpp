#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include "BandStruct.h"
#include "LineWidth.h"
#include "Histogram.h"
#include "Epsilon.h"
#include "InputMap.h"
#include <core/Units.h>

//Lorentzian kernel for an odd function stored on postive frequencies alone:
inline double lorentzianOdd(double omega, double omega0, double breadth)
{	double breadthSq = std::pow(breadth,2);
	return (breadth/M_PI) *
		( 1./(breadthSq + std::pow(omega-omega0, 2))
		- 1./(breadthSq + std::pow(omega+omega0, 2)) );
}

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of imaginary dielectric tensor (ImEps)", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	long nKpts = inputMap.get("nKpts");
	const double EplasmonMax = inputMap.get("EplasmonMax") * eV;
	const double T = inputMap.get("T") * Kelvin;
	const double polTheta = inputMap.get("polTheta") * (M_PI/180);
	const double polPhi = inputMap.get("polPhi") * (M_PI/180);
	const double eta = inputMap.get("eta", 0.1) * eV; //on-shell extrapolation width (default to 0.1 eV)
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional shift in chemical potential from neutral value; start of range (default to 0)
	const double dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional shift in chemical potential from neutral value; end of range (default to 0)
	const int dmuCount = inputMap.get("dmuCount", 1); assert(dmuCount>0); //number of chemical potential shifts

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKpts = %ld\n", nKpts);
	logPrintf("EplasmonMax = %lg\n", EplasmonMax);
	logPrintf("T = %lg\n", T);
	logPrintf("polTheta = %lg\n", polTheta);
	logPrintf("polPhi = %lg\n", polPhi);
	logPrintf("eta = %lg\n", eta);
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("dmuCount = %d\n", dmuCount);

	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(1);
	Ahat[0] = vector3<complex>(sin(polTheta)*cos(polPhi), sin(polTheta)*sin(polPhi), cos(polTheta));
	BandStruct bs("Wannier/totalE", "Wannier/wannier", true, Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);
	
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Initalize line width of electronic states
	LineWidth lineWidth("Wannier/wannier", bs);

	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKpts, mpiWorld).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	nKpts = nBunchesMine * bunchSize; mpiWorld->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	long nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	int nBands = bs.getStates(vector3<>()).nRows();
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	double phononPrefac0 = 4 * std::pow(M_PI,2) * bs.spinWeight / (nKpairs*fabs(det(bs.R))); //frequency independent part of prefac
	double directPrefac0 = 4 * std::pow(M_PI,2) * bs.spinWeight / (nKpts*fabs(det(bs.R))); //frequency independent part of prefac

	//Singularity extrapolation parameters
	double extrapCoeff[] = {-19./12, 13./3, -7./4 }; //account for constant, 1/eta and eta^2 dependence
	//double extrapCoeff[] = { -1, 2.}; //account for constant and 1/eta dependence
	const int nExtrap = sizeof(extrapCoeff)/sizeof(double);
	
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
	mpiWorld->allReduce(omegaMax, MPIUtil::ReduceMax);
	
	//dmu array:
	std::vector<double> dmu(dmuCount, dmuMin); //set first value here
	for(int iMu=1; iMu<dmuCount; iMu++) //set remaining values (if any)
		dmu[iMu] = dmuMin + iMu*(dmuMax-dmuMin)/(dmuCount-1);
	double EvMax = *std::max_element(dmu.begin(), dmu.end()) + 10*T;
	double EcMin = *std::min_element(dmu.begin(), dmu.end()) - 10*T;
	
	//Initialize unbroadened histograms:
	const double domega = T;
	std::vector<Histogram> ImEpsDeltaDirect(dmuCount, Histogram(0, domega, omegaMax));
	std::vector<Histogram> ImEpsDeltaPhonon(dmuCount, Histogram(0, domega, omegaMax));
	std::vector<Histogram> breadthDirect(dmuCount, Histogram(0, domega, omegaMax));
	std::vector<Histogram> breadthPhonon(dmuCount, Histogram(0, domega, omegaMax));
	std::vector<Histogram> weightPhonon(dmuCount, Histogram(0, domega, omegaMax));
	Histogram2D ImEpsDeltaDirect_E(-EplasmonMax, domega, EplasmonMax,  0, domega, omegaMax); //ImEpsDelta resolved by carrier density; collected only for first mu
	Histogram2D ImEpsDeltaPhonon_E(-EplasmonMax, domega, EplasmonMax,  0, domega, omegaMax);
	int nomega = ImEpsDeltaDirect[0].out.size();
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
		std::vector<std::vector<diagMatrix>> Farr(bunchSize, std::vector<diagMatrix>(dmuCount, diagMatrix(nBands)));
		for(int ik=0; ik<bunchSize; ik++)
			for(int imu=0; imu<dmuCount; imu++)
				for(int b=0; b<nBands; b++)
				{	double e = (Earr[ik][b]-dmu[imu])/T;
					Farr[ik][imu][b] = (e>30 ? exp(-e) : 1./(1. + exp(e))); //avoid overflow issues
				}
		
		//Direct transitions:
		for(int ik=0; ik<bunchSize; ik++)
		{	const diagMatrix& E = Earr[ik];
			const diagMatrix& ImE = ImEarr[ik];
			const std::vector<diagMatrix>& F = Farr[ik];
			const std::vector<matrix>& P = Parr[ik];
			for(int v=0; v<nBands; v++) if(E[v]<EvMax)
			{	for(int c=0; c<nBands; c++) if(E[c]>EcMin)
				{	double omega = E[c] - E[v]; //energy conservation
					if(omega<domega || omega>=omegaMax) continue; //irrelevant event
					for(int imu=0; imu<dmuCount; imu++)
					{	double weight = (directPrefac0/(omega*omega)) * (F[imu][v]-F[imu][c]) * P[0](c,v).norm(); //norm=abs^2
						ImEpsDeltaDirect[imu].addEvent(omega, weight);
						breadthDirect[imu].addEvent(omega, weight*(ImE[c]+ImE[v]));
						if(imu==0)
						{	ImEpsDeltaDirect_E.addEvent(E[v], omega, -weight); //hole
							ImEpsDeltaDirect_E.addEvent(E[c], omega, +weight); //electron
						}
					}
				}
			}
		}
		
		//Phonon-assisted transitions:
		for(int ik1=0; ik1<bunchSize; ik1++)
		{	const diagMatrix& E1 = Earr[ik1];
			const diagMatrix& ImE1 = ImEarr[ik1];
			const std::vector<diagMatrix>& F1 = Farr[ik1];
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
				const std::vector<diagMatrix>& F2 = Farr[ik2];
				const std::vector<matrix>& P2 = Parr[ik2];
				//Loops over bands and phonon modes:
				for(int v=0; v<nBands; v++) if(E1[v]<EvMax)
				{	for(int c=0; c<nBands; c++) if(E2[c]>EcMin)
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
								for(int imu=0; imu<dmuCount; imu++)
								{	double weight = (phononPrefac0/(omega*omega)) * (F1[imu][v]-F2[imu][c]) * (nPh + 0.5*(1.-ae)) * MeffSqExtrap;
									ImEpsDeltaPhonon[imu].addEvent(omega, weight);
									breadthPhonon[imu].addEvent(omega, fabs(weight)*(ImE2[c]+ImE1[v]));
									weightPhonon[imu].addEvent(omega, fabs(weight)); //different from ImEpsDeltaPhonon, since weight can be negative due to singularity extrapolation
									if(imu==0)
									{	ImEpsDeltaPhonon_E.addEvent(E1[v], omega, -weight); //hole
										ImEpsDeltaPhonon_E.addEvent(E2[c], omega, +weight); //electron
									}
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
	for(int imu=0; imu<dmuCount; imu++)
	{	ImEpsDeltaDirect[imu].allReduce(MPIUtil::ReduceSum);
		ImEpsDeltaPhonon[imu].allReduce(MPIUtil::ReduceSum);
		breadthDirect[imu].allReduce(MPIUtil::ReduceSum);
		breadthPhonon[imu].allReduce(MPIUtil::ReduceSum);
		weightPhonon[imu].allReduce(MPIUtil::ReduceSum);
	}
	ImEpsDeltaDirect_E.allReduce(MPIUtil::ReduceSum);
	ImEpsDeltaPhonon_E.allReduce(MPIUtil::ReduceSum);
	logPrintf("done.\n"); logFlush();
	
	//Normalize the breadths:
	for(int iomega=0; iomega<nomega; iomega++)
		for(int imu=0; imu<dmuCount; imu++)
		{	breadthDirect[imu].out[iomega] = std::max(T, ImEpsDeltaDirect[imu].out[iomega] ? breadthDirect[imu].out[iomega]/ImEpsDeltaDirect[imu].out[iomega] : 0.);
			breadthPhonon[imu].out[iomega] = std::max(T, weightPhonon[imu].out[iomega] ? breadthPhonon[imu].out[iomega]/weightPhonon[imu].out[iomega] : 0.);
		}
	breadthDirect[0].print("breadth-direct.dat", 1./eV, 1./eV);
	breadthPhonon[0].print("breadth-phonon.dat", 1./eV, 1./eV);
	
	//Apply broadening:
	std::vector<Histogram> ImEpsDirect(dmuCount, Histogram(0, domega, omegaMax));
	std::vector<Histogram> ImEpsPhonon(dmuCount, Histogram(0, domega, omegaMax));
	Histogram2D ImEpsDirect_E(-EplasmonMax, domega, EplasmonMax,  0, domega, EplasmonMax);
	Histogram2D ImEpsPhonon_E(-EplasmonMax, domega, EplasmonMax,  0, domega, EplasmonMax);
	int iomegaStart, iomegaStop; TaskDivision(nomega, mpiWorld).myRange(iomegaStart, iomegaStop);
	logPrintf("Applying broadening ... "); logFlush();
	for(int imu=0; imu<dmuCount; imu++)
	{	for(int iomega=iomegaStart; iomega<iomegaStop; iomega++) //input frequency grid split over MPI
		{	double omegaCur = iomega*domega;
			double bDirect = breadthDirect[imu].out[iomega];
			double bPhonon = breadthPhonon[imu].out[iomega];
			for(size_t jomega=0; jomega<ImEpsDirect[imu].out.size(); jomega++) //output frequency grid
			{	double omega = jomega*domega;
				double kernelDirect = lorentzianOdd(omega, omegaCur, bDirect) * domega;
				double kernelPhonon = lorentzianOdd(omega, omegaCur, bPhonon) * domega;
				ImEpsDirect[imu].out[jomega] += kernelDirect * ImEpsDeltaDirect[imu].out[iomega];
				ImEpsPhonon[imu].out[jomega] += kernelPhonon * ImEpsDeltaPhonon[imu].out[iomega];
				//Carrier distributions:
				if(imu==0 && int(jomega)<ImEpsDirect_E.nomega)
				{	const int nE = ImEpsDirect_E.nE; assert(nE == ImEpsDeltaDirect_E.nE);
					for(int iE=0; iE<nE; iE++)
					{	int iOE = iomega*nE + iE;
						int jOE = jomega*nE + iE;
						ImEpsDirect_E.out[jOE] += kernelDirect * ImEpsDeltaDirect_E.out[iOE];
						ImEpsPhonon_E.out[jOE] += kernelPhonon * ImEpsDeltaPhonon_E.out[iOE];
					}
				}
			}
		}
		ImEpsDirect[imu].allReduce(MPIUtil::ReduceSum);
		ImEpsPhonon[imu].allReduce(MPIUtil::ReduceSum);
	}
	ImEpsDirect_E.allReduce(MPIUtil::ReduceSum); ImEpsDirect_E.print("carrierDistribAll-direct.dat", 1./eV, 1./eV, 1.);
	ImEpsPhonon_E.allReduce(MPIUtil::ReduceSum); ImEpsPhonon_E.print("carrierDistribAll-phonon.dat", 1./eV, 1./eV, 1.);
	logPrintf("done.\n"); logFlush();
	
	//Output ImEps:
	if(mpiWorld->isHead())
	{	ofstream ofs("ImEps.dat");
		ofs << "#omega[eV]";
		for(int imu=0; imu<dmuCount; imu++)
			ofs << " direct[mu=" << dmu[imu]/eV << "eV] phonon[mu=" << dmu[imu]/eV << "eV]";
		ofs << "\n";
		for(size_t iOmega=0; iOmega<ImEpsDirect[0].out.size(); iOmega++)
		{	double omega = ImEpsDirect[0].Emin + ImEpsDirect[0].dE * iOmega;
			ofs << omega/eV;
			for(int imu=0; imu<dmuCount; imu++)
			{	ofs << '\t' << ImEpsDirect[imu].out[iOmega];
				ofs << '\t' << ImEpsPhonon[imu].out[iOmega];
			}
			ofs << '\n';
		}
	}
	
	finalizeSystem();
	return 0;
}

