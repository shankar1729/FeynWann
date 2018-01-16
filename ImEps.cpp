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
	double omegaDirectMax;
	
	static void eProcess(const WannierMC::StateE& state, void* params)
	{	EnergyRange& er = *((EnergyRange*)params);
		er.Emin = std::min(er.Emin, state.E.front()); //E is in ascending order
		er.Emax = std::max(er.Emax, state.E.back()); //E is in ascending order
		er.omegaDirectMax = std::max(er.omegaDirectMax, state.E.back()-state.E.front()); //max direct transition frequency
	}
};

//Collect ImEps contibutions using WannierMC callbacks:
struct CollectImEps
{	const std::vector<double>& dmu;
	double T, invT;
	double domega, omegaFull, omegaMax;
	std::vector<Histogram> ImEps, breadth;
	Histogram2D ImEps_E; //ImEpsDelta resolved by carrier density; collected only for first mu
	double prefac;
	vector3<> Ehat;
	double EvMax, EcMin;
	
	CollectImEps(const std::vector<double>& dmu, double T, double domega, double omegaFull, double omegaMax)
	: dmu(dmu), T(T), invT(1./T), domega(domega), omegaFull(omegaFull), omegaMax(omegaMax),
		ImEps(dmu.size(), Histogram(0, domega, omegaFull)),
		breadth(dmu.size(), Histogram(0, domega, omegaFull)),
		ImEps_E(-omegaMax, domega, omegaMax,  0, domega, omegaFull) //ImEpsDelta resolved by carrier density; collected only for first mu
	{	logPrintf("Initialized frequency grid: 0 to %lg eV with %d points.\n", ImEps[0].Emax()/eV, ImEps[0].nE);
		EvMax = *std::max_element(dmu.begin(), dmu.end()) + 10*T;
		EcMin = *std::min_element(dmu.begin(), dmu.end()) - 10*T;
	}
	
	void collect(const WannierMC::StateE& state)
	{	int nBands = state.E.nRows();
		//Calculate Fermi fillings and linewidths:
		std::vector<diagMatrix> F(dmu.size(), diagMatrix(nBands));
		std::vector<diagMatrix> ImE(dmu.size(), state.ImSigma_ee); //e-e part
		for(unsigned iMu=0; iMu<dmu.size(); iMu++)
		{	for(int b=0; b<nBands; b++)
			{	double expArg = (state.E[b]-dmu[iMu])*invT;
				F[iMu][b] = (expArg < -30.) ? 1.
					: ((expArg > +30.) ? 0.
					: 1./(1.+exp(expArg)) );
				ImE[iMu][b] += state.ImSigma_ePh(b, F[iMu][b]);
			}
		}
		//Collect 
		for(int v=0; v<nBands; v++) if(state.E[v]<EvMax)
		{	for(int c=0; c<nBands; c++) if(state.E[c]>EcMin)
			{	double omega = state.E[c] - state.E[v]; //energy conservation
				if(omega<domega || omega>=omegaFull) continue; //irrelevant event
				complex Pcv;
				for(int iDir=0; iDir<3; iDir++)
					Pcv += state.v[iDir](c,v) * Ehat[iDir];
				double weight_F = (prefac/(omega*omega)) * Pcv.norm(); //event weight except for occupation factors
				for(unsigned iMu=0; iMu<dmu.size(); iMu++)
				{	double weight = weight_F * (F[iMu][v]-F[iMu][c]);
					ImEps[iMu].addEvent(omega, weight);
					breadth[iMu].addEvent(omega, weight*(ImE[iMu][c]+ImE[iMu][v]));
					if(iMu==0)
					{	ImEps_E.addEvent(state.E[v], omega, -weight); //hole
						ImEps_E.addEvent(state.E[c], omega, +weight); //electron
					}
				}
			}
		}
	}
	static void eProcess(const WannierMC::StateE& state, void* params)
	{	((CollectImEps*)params)->collect(state);
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
	InitParams ip = WannierMC::initialize(argc, argv, "Monte Carlo estimate of imaginary dielectric tensor (ImEps)");

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
	

	//Initialize WannierMC:
	WannierMCParams wmcp;
	wmcp.needPhonons = true;
	wmcp.needVelocity = true;
	wmcp.needLinewidth_ee = true;
	wmcp.needLinewidth_ePh = true;
	std::shared_ptr<WannierMC> wmc = std::make_shared<WannierMC>(wmcp);
	size_t nKpts = nOffsets * wmc->eCountPerOffset();  
	logPrintf("Effectively sampled nKpts: %lu\n", nKpts);

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
	mpiWorld->allReduce(er.omegaDirectMax, MPIUtil::ReduceMax);
	double omegaFull = er.omegaDirectMax;
	
	//dmu array:
	std::vector<double> dmu(dmuCount, dmuMin); //set first value here
	for(int iMu=1; iMu<dmuCount; iMu++) //set remaining values (if any)
		dmu[iMu] = dmuMin + iMu*(dmuMax-dmuMin)/(dmuCount-1);
	
	//Calculate delta-function resolved versions (no broadening yet):
	CollectImEps cie(dmu, T, domega, omegaFull, omegaMax);
	cie.prefac = 4. * std::pow(M_PI,2) * wmc->spinWeight / (nKpts*fabs(det(wmc->R))); //frequency independent part of prefactor
	cie.Ehat = vector3<>(sin(polTheta)*cos(polPhi), sin(polTheta)*sin(polPhi), cos(polTheta)); //Efield direction
	
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
			vector3<> k0 = wmc->randomVector(mpiGroup); //must be constant across group
			wmc->eLoop(k0, CollectImEps::eProcess, &cie);
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
	}
	for(int imu=0; imu<dmuCount; imu++)
	{	cie.ImEps[imu].allReduce(MPIUtil::ReduceSum);
		cie.breadth[imu].allReduce(MPIUtil::ReduceSum);
	}
	cie.ImEps_E.allReduce(MPIUtil::ReduceSum);
	logPrintf("done.\n"); logFlush();
	
	//Normalize the breadths:
	int nomega = cie.breadth[0].nE;
	for(int iomega=0; iomega<nomega; iomega++)
		for(int imu=0; imu<dmuCount; imu++)
		{	cie.breadth[imu].out[iomega] = std::max(T, 
				cie.ImEps[imu].out[iomega]
					? cie.breadth[imu].out[iomega]/cie.ImEps[imu].out[iomega]
					: 0.);
		}
	cie.breadth[0].print("breadth-direct.dat", 1./eV, 1./eV);
	
	//Apply broadening:
	std::vector<Histogram> ImEps(dmuCount, Histogram(0, domega, omegaFull));
	Histogram2D ImEps_E(-omegaMax, domega, omegaMax,  0, domega, omegaMax);
	int iomegaStart, iomegaStop; TaskDivision(nomega, mpiWorld).myRange(iomegaStart, iomegaStop);
	logPrintf("Applying broadening ... "); logFlush();
	for(int imu=0; imu<dmuCount; imu++)
	{	for(int iomega=iomegaStart; iomega<iomegaStop; iomega++) //input frequency grid split over MPI
		{	double omegaCur = iomega*domega;
			double b = cie.breadth[imu].out[iomega];
			for(size_t jomega=0; jomega<ImEps[imu].out.size(); jomega++) //output frequency grid
			{	double omega = jomega*domega;
				double kernel = lorentzianOdd(omega, omegaCur, b) * domega;
				ImEps[imu].out[jomega] += kernel * cie.ImEps[imu].out[iomega];
				//Carrier distributions:
				if(imu==0 && int(jomega)<ImEps_E.nomega)
				{	const int nE = ImEps_E.nE; assert(nE == cie.ImEps_E.nE);
					for(int iE=0; iE<nE; iE++)
					{	int iOE = iomega*nE + iE;
						int jOE = jomega*nE + iE;
						ImEps_E.out[jOE] += kernel * cie.ImEps_E.out[iOE];
					}
				}
			}
		}
		ImEps[imu].allReduce(MPIUtil::ReduceSum);
	}
	ImEps_E.allReduce(MPIUtil::ReduceSum); ImEps_E.print("carrierDistrib-direct.dat", 1./eV, 1./eV, 1.);
	logPrintf("done.\n"); logFlush();
	
	//Output ImEps:
	if(mpiWorld->isHead())
	{	ofstream ofs("ImEps-direct.dat");
		ofs << "#omega[eV]";
		for(int imu=0; imu<dmuCount; imu++)
			ofs << " ImEps[mu=" << dmu[imu]/eV << "eV]";
		ofs << "\n";
		for(size_t iOmega=0; iOmega<ImEps[0].out.size(); iOmega++)
		{	double omega = ImEps[0].Emin + ImEps[0].dE * iOmega;
			ofs << omega/eV;
			for(int imu=0; imu<dmuCount; imu++)
				ofs << '\t' << ImEps[imu].out[iOmega];
			ofs << '\n';
		}
	}
	
	wmc = 0;
	WannierMC::finalize();
	return 0;
}
