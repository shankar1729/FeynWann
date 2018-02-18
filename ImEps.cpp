#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include "WannierMC.h"
#include "Histogram.h"
#include "InputMap.h"
#include <core/Units.h>

//Get energy range from an eLoop call:
struct EnergyRange
{	double Emin;
	double Emax;
	
	static void eProcess(const WannierMC::StateE& state, void* params)
	{	EnergyRange& er = *((EnergyRange*)params);
		er.Emin = std::min(er.Emin, state.E.front()); //E is in ascending order
		er.Emax = std::max(er.Emax, state.E.back()); //E is in ascending order
	}
};

//Singularity extrapolation for phonon-assisted:
double extrapCoeff[] = {-19./12, 13./3, -7./4 }; //account for constant, 1/eta and eta^2 dependence
//double extrapCoeff[] = { -1, 2.}; //account for constant and 1/eta dependence
const int nExtrap = sizeof(extrapCoeff)/sizeof(double);

//Collect ImEps contibutions using WannierMC callbacks:
struct CollectImEps
{	const std::vector<double>& dmu;
	double T, invT;
	double domega, omegaFull, omegaMax;
	std::vector<Histogram> ImEps, breadth, breadthDen;
	Histogram2D ImEps_E; //ImEpsDelta resolved by carrier density; collected only for first mu
	double prefac;
	double eta; //singularity extrapolation width
	vector3<> Ehat;
	double EvMax, EcMin;
	
	
	CollectImEps(const std::vector<double>& dmu, double T, double domega, double omegaFull, double omegaMax)
	: dmu(dmu), T(T), invT(1./T), domega(domega), omegaFull(omegaFull), omegaMax(omegaMax),
		ImEps(dmu.size(), Histogram(0, domega, omegaFull)),
		breadth(dmu.size(), Histogram(0, domega, omegaFull)),
		breadthDen(dmu.size(), Histogram(0, domega, omegaFull)),
		ImEps_E(-omegaMax, domega, omegaMax,  0, domega, omegaFull) //ImEpsDelta resolved by carrier density; collected only for first mu
	{	logPrintf("Initialized frequency grid: 0 to %lg eV with %d points.\n", ImEps[0].Emax()/eV, ImEps[0].nE);
		EvMax = *std::max_element(dmu.begin(), dmu.end()) + 10*T;
		EcMin = *std::min_element(dmu.begin(), dmu.end()) - 10*T;
	}
	
	void calcStateRelated(const WannierMC::StateE& state, std::vector<diagMatrix>& F, std::vector<diagMatrix>& ImE)
	{	int nBands = state.E.nRows();
		F.assign(dmu.size(), diagMatrix(nBands));
		ImE.assign(dmu.size(), state.ImSigma_ee); //e-e part
		for(unsigned iMu=0; iMu<dmu.size(); iMu++)
		{	for(int b=0; b<nBands; b++)
			{	double expArg = (state.E[b]-dmu[iMu])*invT;
				F[iMu][b] = (expArg < -30.) ? 1.
					: ((expArg > +30.) ? 0.
					: 1./(1.+exp(expArg)) );
				ImE[iMu][b] += state.ImSigma_ePh(b, F[iMu][b]);
			}
		}
	}
	
	//---- Direct transitions ----
	void collectDirect(const WannierMC::StateE& state)
	{	int nBands = state.E.nRows();
		//Calculate Fermi fillings and linewidths:
		const diagMatrix& E = state.E;
		std::vector<diagMatrix> F, ImE;
		calcStateRelated(state, F, ImE);
		//Project dipole matrix elements on field:
		matrix P;
		for(int iDir=0; iDir<3; iDir++)
			P += Ehat[iDir] * state.v[iDir];
		//Collect 
		for(int v=0; v<nBands; v++) if(E[v]<EvMax)
		{	for(int c=0; c<nBands; c++) if(E[c]>EcMin)
			{	double omega = E[c] - E[v]; //energy conservation
				if(omega<domega || omega>=omegaFull) continue; //irrelevant event
				double weight_F = (prefac/(omega*omega)) * P(c,v).norm(); //event weight except for occupation factors
				for(unsigned iMu=0; iMu<dmu.size(); iMu++)
				{	double weight = weight_F * (F[iMu][v]-F[iMu][c]);
					ImEps[iMu].addEvent(omega, weight);
					breadth[iMu].addEvent(omega, weight*(ImE[iMu][c]+ImE[iMu][v]));
					if(iMu==0)
					{	ImEps_E.addEvent(E[v], omega, -weight); //hole
						ImEps_E.addEvent(E[c], omega, +weight); //electron
					}
				}
			}
		}
	}
	static void direct(const WannierMC::StateE& state, void* params)
	{	((CollectImEps*)params)->collectDirect(state);
	}
	
	//---- Phonon-assisted transitions ----
	void collectPhonon(const WannierMC::MatrixEph& mat)
	{	int nBands = mat.e1->E.nRows();
		//Calculate Fermi fillings and linewidths:
		const diagMatrix& E1 = mat.e1->E;
		const diagMatrix& E2 = mat.e2->E;
		std::vector<diagMatrix> F1, F2, ImE1, ImE2;
		calcStateRelated(*mat.e1, F1, ImE1);
		calcStateRelated(*mat.e2, F2, ImE2);
		//Project dipole matrix elements on field:
		matrix P1, P2;
		for(int iDir=0; iDir<3; iDir++)
		{	P1 += Ehat[iDir] * mat.e1->v[iDir];
			P2 += Ehat[iDir] * mat.e2->v[iDir];
		}
		//Bose occupations:
		const diagMatrix& omegaPh = mat.ph->omega;
		int nModes = omegaPh.nRows();
		diagMatrix nPh(nModes);
		for(int iMode=0; iMode<nModes; iMode++)
		{	double omegaPhByT = omegaPh[iMode]/T;
			nPh[iMode] = omegaPhByT>36
				? 0. //avoid overflow
				: 1./(exp(std::max(1e-3, omegaPhByT)) - 1.); //avoid 0/0 for zero phonon frequencies
		}
		//Collect
		for(int v=0; v<nBands; v++) if(E1[v]<EvMax)
		{	for(int c=0; c<nBands; c++) if(E2[c]>EcMin)
			{	for(int alpha=0; alpha<nModes; alpha++)
				{	for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
					{	double omega = E2[c] - E1[v] - ae*omegaPh[alpha]; //energy conservation
						if(omega<domega || omega>=omegaFull) continue; //irrelevant event
						//Effective matrix elements
						std::vector<complex> Meff(nExtrap, 0.);
						for(int i=0; i<nBands; i++) // sum over the intermediate states
						{	complex numA = P2(c,i) * mat.M[alpha](i,v); double denA = E2[i] - (E2[c] - omega);
							complex numB = mat.M[alpha](c,i) * P1(i,v); double denB = E1[i] - (E1[v] + omega);
							double zEta = eta;
							for(int z=0; z<nExtrap; z++)
							{	Meff[z] += ( numA / complex(denA,zEta) + numB / complex(denB,zEta) );
								zEta += eta; //contains (z+1)*eta when evaluating above
							}
						}
						//Singularity extrapolation:
						double MeffSqExtrap = 0.;
						for(int z=0; z<nExtrap; z++)
							MeffSqExtrap += extrapCoeff[z] * Meff[z].norm();
						double weight_F = (prefac/(omega*omega)) * (nPh[alpha] + 0.5*(1.-ae)) * MeffSqExtrap;
						for(unsigned iMu=0; iMu<dmu.size(); iMu++)
						{	double weight = weight_F * (F1[iMu][v]-F2[iMu][c]);
							ImEps[iMu].addEvent(omega, weight);
							breadth[iMu].addEvent(omega, fabs(weight)*(ImE2[iMu][c]+ImE1[iMu][v]));
							breadthDen[iMu].addEvent(omega, fabs(weight));
							if(iMu==0)
							{	ImEps_E.addEvent(E1[v], omega, -weight); //hole
								ImEps_E.addEvent(E2[c], omega, +weight); //electron
							}
						}
					}
				}
			}
		}
	}
	static void phonon(const WannierMC::MatrixEph& mat, void* params)
	{	((CollectImEps*)params)->collectPhonon(mat);
	}
};

//Lorentzian kernel for an odd function stored on postive frequencies alone:
inline double lorentzianOdd(double omega, double omega0, double breadth)
{	double breadthSq = std::pow(breadth,2);
	return (breadth/M_PI) *
		( 1./(breadthSq + std::pow(omega-omega0, 2))
		- 1./(breadthSq + std::pow(omega+omega0, 2)) );
}

int main(int argc, char** argv)
{	
	InitParams ip = WannierMC::initialize(argc, argv, "Wannier calculation of imaginary dielectric tensor (ImEps)");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const int nOffsets = inputMap.get("nOffsets"); assert(nOffsets>0);
	const double omegaMax = inputMap.get("omegaMax") * eV;
	const double T = inputMap.get("T") * Kelvin;
	const double polTheta = inputMap.get("polTheta") * (M_PI/180);
	const double polPhi = inputMap.get("polPhi") * (M_PI/180);
	const double eta = inputMap.get("eta", 0.1) * eV; //on-shell extrapolation width (default to 0.1 eV)
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional shift in chemical potential from neutral value; start of range (default to 0)
	const double dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional shift in chemical potential from neutral value; end of range (default to 0)
	const int dmuCount = inputMap.get("dmuCount", 1); assert(dmuCount>0); //number of chemical potential shifts
	string contribution = inputMap.getString("contribution"); //direct / phonon

	//Check contribution:
	enum ContribType { Direct, Phonon };
	EnumStringMap<ContribType> contribMap(Direct, "Direct", Phonon, "Phonon");
	ContribType contribType;
	if(!contribMap.getEnum(contribution.c_str(), contribType))
		die("Input parameter 'contribution' must be one of %s.\n\n", contribMap.optionList().c_str());
	string fileSuffix = contribMap.getString(contribType);
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("omegaMax = %lg\n", omegaMax);
	logPrintf("T = %lg\n", T);
	logPrintf("polTheta = %lg\n", polTheta);
	logPrintf("polPhi = %lg\n", polPhi);
	logPrintf("eta = %lg\n", eta);
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("dmuCount = %d\n", dmuCount);
	logPrintf("contribution = %s\n", contribMap.getString(contribType));

	//Initialize WannierMC:
	WannierMCParams wmcp;
	wmcp.needPhonons = (contribType==Phonon);
	wmcp.needVelocity = true;
	wmcp.needLinewidth_ee = true;
	wmcp.needLinewidth_ePh = true;
	std::shared_ptr<WannierMC> wmc = std::make_shared<WannierMC>(wmcp);
	size_t nKeff = nOffsets * (contribType==Direct ? wmc->eCountPerOffset() : wmc->ePhCountPerOffset());
	logPrintf("Effectively sampled %s: %lu\n", (contribType==Direct ? "nKpts" : "nKpairs"), nKeff);

	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		wmc = 0;
		WannierMC::finalize();
		return 0;
	}
	logPrintf("\n");

	//Initialize sampling parameters:
	int oStart=0, oStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsets, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	
	//Initialize frequency grid:
	const double domega = T;
	EnergyRange er = { DBL_MAX, -DBL_MAX };
	wmc->eLoop(vector3<>(), EnergyRange::eProcess, &er);
	mpiWorld->allReduce(er.Emin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(er.Emax, MPIUtil::ReduceMax);
	double omegaFull = er.Emax - er.Emin;
	
	//dmu array:
	std::vector<double> dmu(dmuCount, dmuMin); //set first value here
	for(int iMu=1; iMu<dmuCount; iMu++) //set remaining values (if any)
		dmu[iMu] = dmuMin + iMu*(dmuMax-dmuMin)/(dmuCount-1);
	
	//Calculate delta-function resolved versions (no broadening yet):
	CollectImEps cie(dmu, T, domega, omegaFull, omegaMax);
	cie.prefac = 4. * std::pow(M_PI,2) * wmc->spinWeight / (nKeff*fabs(det(wmc->R))); //frequency independent part of prefactor
	cie.Ehat = vector3<>(sin(polTheta)*cos(polPhi), sin(polTheta)*sin(polPhi), cos(polTheta)); //Efield direction
	cie.eta = eta;
	
	for(int iSpin=0; iSpin<wmc->nSpins; iSpin++)
	{	//Update WannierMC for spin channel if necessary:
		if(iSpin>0)
		{	wmc = 0; //free memory from previous spin
			wmcp.iSpin = iSpin;
			wmc = std::make_shared<WannierMC>(wmcp);
		}
		logPrintf("\nCollecting ImEps: "); logFlush();
		for(int o=0; o<noMine; o++)
		{	Random::seed(o+oStart); //to make results independent of MPI division
			//Process with a random offset:
			switch(contribType)
			{	case Direct:
				{	vector3<> k0 = wmc->randomVector(mpiGroup); //must be constant across group
					wmc->eLoop(k0, CollectImEps::direct, &cie);
					break;
				}
				case Phonon:
				{	vector3<> k01 = wmc->randomVector(mpiGroup); //must be constant across group
					vector3<> k02 = wmc->randomVector(mpiGroup); //must be constant across group
					wmc->ePhLoop(k01, k02, CollectImEps::phonon, &cie);
					break;
				}
			}
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
	}
	for(int iMu=0; iMu<dmuCount; iMu++)
	{	cie.ImEps[iMu].allReduce(MPIUtil::ReduceSum);
		cie.breadth[iMu].allReduce(MPIUtil::ReduceSum);
		if(contribType==Direct)
			cie.breadthDen[iMu] = cie.ImEps[iMu]; //normalization weight is just ImEps
		else
			cie.breadthDen[iMu].allReduce(MPIUtil::ReduceSum); //collected separately due to extrapolation sign
	}
	cie.ImEps_E.allReduce(MPIUtil::ReduceSum);
	logPrintf("done.\n"); logFlush();
	
	//Normalize the breadths:
	int nomega = cie.breadth[0].nE;
	for(int iomega=0; iomega<nomega; iomega++)
		for(int iMu=0; iMu<dmuCount; iMu++)
		{	cie.breadth[iMu].out[iomega] = std::max(T, 
				cie.breadthDen[iMu].out[iomega]
					? cie.breadth[iMu].out[iomega]/cie.breadthDen[iMu].out[iomega]
					: 0.);
		}
	cie.breadth[0].print("breadth"+fileSuffix+".dat", 1./eV, 1./eV);
	
	//Apply broadening:
	std::vector<Histogram> ImEps(dmuCount, Histogram(0, domega, omegaFull));
	Histogram2D ImEps_E(-omegaMax, domega, omegaMax,  0, domega, omegaMax);
	int iomegaStart, iomegaStop; TaskDivision(nomega, mpiWorld).myRange(iomegaStart, iomegaStop);
	logPrintf("Applying broadening ... "); logFlush();
	for(int iMu=0; iMu<dmuCount; iMu++)
	{	for(int iomega=iomegaStart; iomega<iomegaStop; iomega++) //input frequency grid split over MPI
		{	double omegaCur = iomega*domega;
			double b = cie.breadth[iMu].out[iomega];
			for(size_t jomega=0; jomega<ImEps[iMu].out.size(); jomega++) //output frequency grid
			{	double omega = jomega*domega;
				double kernel = lorentzianOdd(omega, omegaCur, b) * domega;
				ImEps[iMu].out[jomega] += kernel * cie.ImEps[iMu].out[iomega];
				//Carrier distributions:
				if(iMu==0 && int(jomega)<ImEps_E.nomega)
				{	const int nE = ImEps_E.nE; assert(nE == cie.ImEps_E.nE);
					for(int iE=0; iE<nE; iE++)
					{	int iOE = iomega*nE + iE;
						int jOE = jomega*nE + iE;
						ImEps_E.out[jOE] += kernel * cie.ImEps_E.out[iOE];
					}
				}
			}
		}
		ImEps[iMu].allReduce(MPIUtil::ReduceSum);
	}
	ImEps_E.allReduce(MPIUtil::ReduceSum); ImEps_E.print("carrierDistrib"+fileSuffix+".dat", 1./eV, 1./eV, 1.);
	logPrintf("done.\n"); logFlush();
	
	//Output ImEps:
	if(mpiWorld->isHead())
	{	ofstream ofs("ImEps"+fileSuffix+".dat");
		ofs << "#omega[eV]";
		for(int iMu=0; iMu<dmuCount; iMu++)
			ofs << " ImEps[mu=" << dmu[iMu]/eV << "eV]";
		ofs << "\n";
		for(size_t iOmega=0; iOmega<ImEps[0].out.size(); iOmega++)
		{	double omega = ImEps[0].Emin + ImEps[0].dE * iOmega;
			ofs << omega/eV;
			for(int iMu=0; iMu<dmuCount; iMu++)
				ofs << '\t' << ImEps[iMu].out[iOmega];
			ofs << '\n';
		}
	}
	
	wmc = 0;
	WannierMC::finalize();
	return 0;
}
