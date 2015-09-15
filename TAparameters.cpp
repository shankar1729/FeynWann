#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "LineWidth.h"
#include "InputMap.h"
#include "Units.h"
#include "Histogram.h"
#include "Epsilon.h"

//Lorentzian kernel for an odd function stored on postive frequencies alone:
inline double lorentzianOdd(double omega, double omega0, double breadth)
{       double breadthSq = std::pow(breadth,2);
        return (breadth/M_PI) *
                ( 1./(breadthSq + std::pow(omega-omega0, 2))
                - 1./(breadthSq + std::pow(omega+omega0, 2)) );
}


inline double fermi(double x) { return x>30. ? exp(-x) : 1./(1.+exp(x)); } //avoid overflow issues
inline double fermiPrime(double x) { return 0.25*(std::pow(tanh(0.5*x), 2) - 1.); } //avoid overflow issues
inline double Ejellium(vector3<> k) { return (k[0]*k[0]+k[1]*k[1]+k[2]*k[2])/2; } //avoid overflow issues
inline double argLW(double E,double Es) { return sqrt(E)/(E+Es) + 1/sqrt(Es) * atan(sqrt(E/Es)); }

inline void writeImEps(const char* fname, const std::vector<Histogram>& ImEps, const std::vector<double> TeArr)
{	std::ofstream ofs(fname);
	//Header:
	ofs << "#omega";
	for(const double& Te: TeArr)
		ofs << " ImEps[T=" << Te/Kelvin << "K]";
	ofs << '\n';
	//Data:
	for(size_t iomega=0; iomega<ImEps[0].out.size(); iomega++)
	{	double omega = ImEps[0].dE * iomega;
		ofs << omega/eV;
		for(const Histogram& h: ImEps)
			ofs << ' ' << h.out[iomega];
		ofs << '\n';
	}
}

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Ab initio parameters for Transient Absorption analysis", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	double mu = inputMap.get("mu"); //initial guess only - will be calculated self-consistently in this executable
	const double Z = inputMap.get("Z"); //number of electrons per unit cell
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	const double TeMin = inputMap.get("TeMin") * Kelvin; //electron temperature grid start
	const double TeMax = inputMap.get("TeMax") * Kelvin; //electron temperature grid stop
	const double TeStep = inputMap.get("TeStep") * Kelvin; //electron temperature grid spacing
	const double Tl = inputMap.get("Tl") * Kelvin; //lattice temperature
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	const double Es = inputMap.get("Es"); // Es in hartrees, as defined in Vallee paper
	const double epsB = inputMap.get("epsilonB"); // epsilon_b, as defined in Vallee paper

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("mu = %lg\n", mu);
	logPrintf("Z = %lg\n", Z);
	logPrintf("dE = %lg\n", dE);
	logPrintf("TeMin = %lg\n", TeMin);
	logPrintf("TeMax = %lg\n", TeMax);
	logPrintf("TeStep = %lg\n", TeStep);
	logPrintf("Tl = %lg\n", Tl);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	logPrintf("Es: = %lg\n", Es);
	logPrintf("epsilon_b = %lg\n", epsB);
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");
	
	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(1); //assume cubic symmetry and only calculate x-axis
	Ahat[0] = vector3<complex>(1., 0., 0.);
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE", Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);

	//Initialize temperature grid:
	std::vector<double> TeArr(int(ceil((TeMax-TeMin)/TeStep)));
	for(size_t iT=0; iT<TeArr.size(); iT++)
		TeArr[iT] = TeMin + TeStep*iT;
	logPrintf("Initialized temperature grid: %lg to %lg K with %lu points.\n", TeArr.front()/Kelvin, TeArr.back()/Kelvin, TeArr.size());
	
	//Initialize energy grid:
	diagMatrix Egamma = bs.getStates(vector3<>());
	double Emin = Egamma.front(), Emax = Egamma.back(); //eigenvalues are sorted
	for(int i=0; i<10; i++)
	{	std::vector<vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		for(const diagMatrix& E: Earr)
		{	Emin = std::min(Emin, E.front());
			Emax = std::max(Emax, E.back());
		}
	}
	mpiUtil->allReduce(Emin, MPIUtil::ReduceMin);
	mpiUtil->allReduce(Emax, MPIUtil::ReduceMax);
	Emin -= 10*dE; //add some margin
	Emax += 10*dE;
	Histogram dos(Emin, dE, Emax); //density of states
	logPrintf("Initialized energy grid: %lg to %lg eV with %lu points.\n", Emin/eV, (Emin+dE*(dos.out.size()-1))/eV, dos.out.size());
	
	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	long nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	long nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	int nBands = Egamma.nRows();
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	double phononPrefac0 = 4 * std::pow(M_PI,2) * spinWeight / (nKpairs*fabs(det(R))); //frequency independent part of prefac
	double directPrefac0 = 4 * std::pow(M_PI,2) * spinWeight / (nKpts*fabs(det(R))); //frequency independent part of prefac
	double GePhPrefac = spinWeight * (2*M_PI) / nKpairs;
	
	//Singularity extrapolation parameters
	double extrapCoeff[] = {-19./12, 13./3, -7./4 }; //account for constant, 1/eta and eta^2 dependence
	//double extrapCoeff[] = { -1, 2.}; //account for constant and 1/eta dependence
	const int nExtrap = sizeof(extrapCoeff)/sizeof(double);
	const double eta = 0.1*eV;

	//-------- Pass 1: collect density of states, calculate mu(Te) and Ce(Te) ---------
	
	logPrintf("\nCollecting DOS: "); logFlush();
	std::vector< std::vector< vector3<> > > kArrArr(nBunchesMine); //use exact same set of MC k-points in the two passes for consistency
	const double dosWeight = spinWeight*(1./nKpts);
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{
		//Generate a bunch of k-points:
		std::vector< vector3<> >& kArr = kArrArr[iBunch];
		kArr.resize(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		
		//Collect DOS:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		for(const diagMatrix& E: Earr)
			for(const double& Ei: E)
				dos.addEvent(Ei, dosWeight);
		
		//Print progress:
		if((iBunch+1) % iBunchInterval == 0)
		{	logPrintf("%d%% ", int(round((iBunch+1)*100./nBunchesMine)));
			logFlush();
		}
	}
	logPrintf("done.\n"); logFlush();
	dos.allReduce(MPIUtil::ReduceSum);
	dos.print("dos.dat", 1./eV, eV);
	
	//Calculate mu and Ce at each temperature:
	diagMatrix dmu(TeArr.size(), 0.), Ce(TeArr.size(), 0.);
	//--- check enough bands to contain Z:
	double Zmax = 0.;
	for(const double& g: dos.out)
		Zmax += dE * g;
	if(Zmax < Z)
		die("Current DOS can only support %lg electrons > %lg electrons specified.\n", Zmax, Z);
	int iTstart, iTstop; TaskDivision(TeArr.size(), mpiUtil).myRange(iTstart, iTstop);
	for(int iT=iTstart; iT<iTstop; iT++)
	{	const double Te = TeArr[iT], invTe = 1./Te;
		//Bisect for chemical potential:
		double& dmuCur = dmu[iT];
		double dmuMin = Emin - 10*Te;
		double dmuMax = Emax + 10*Te;
		dmuCur = 0.5*(dmuMin + dmuMax);
		const double tol = 1e-9*Te;
		while(dmuMax-dmuMin > tol)
		{	//calculate number of electrons at current Z:
			double nElectrons = 0.;
			for(size_t ie=0; ie<dos.out.size(); ie++)
			{	double Ei = Emin + ie*dE;
				double fi = fermi(invTe*(Ei - dmuCur));
				nElectrons += dE * dos.out[ie] * fi;
			}
			((nElectrons>Z) ? dmuMax : dmuMin) = dmuCur;
			dmuCur = 0.5*(dmuMin + dmuMax);
		}
		//Calculate electronic specific heat:
		double& CeCur = Ce[iT];
		CeCur = 0.;
		for(size_t ie=0; ie<dos.out.size(); ie++)
		{	double Ei = Emin + ie*dE;
			double x = invTe*(Ei-dmuCur);
			double dfdT = fermiPrime(x) * (-x*invTe);
			CeCur += dE * Ei * dos.out[ie] * dfdT;
		}
	}
	dmu.allReduce(MPIUtil::ReduceSum);
	Ce.allReduce(MPIUtil::ReduceSum);
	


	// -------------------------------------  Setup for Pass 2 --------------------------------------
	//Initalize line width of electronic states
	LineWidth lineWidth("Wannier/wannier", bs);


	//Initialize frequency grid:
	double omegaMax = 0.;
	for(size_t iT=0; iT<TeArr.size(); iT++)
	{	double hEmax = dmu[iT] - Emin; //max hole energy
		double eEmax = Emax - dmu[iT]; //max electron energy
		double Emax = std::max(hEmax, eEmax) + 10*TeArr[iT]; //with margin for partially occupied excitations
		omegaMax = std::max(omegaMax, Emax);
	}

	//Initialize unbroadened histograms:
	std::vector<Histogram> ImEpsDirect(TeArr.size(), Histogram(0, dE, omegaMax)), breadthDirect(TeArr.size(), Histogram(0, dE, omegaMax));
	std::vector<Histogram> ImEpsPhonon(TeArr.size(), Histogram(0, dE, omegaMax)), breadthPhonon(TeArr.size(), Histogram(0, dE, omegaMax)),  weightPhonon(TeArr.size(), Histogram(0, dE, omegaMax));
	int nomega = ImEpsDirect[0].out.size();
	logPrintf("Initialized frequency grid: 0 to %lg eV with %d points.\n", (dE*(nomega-1))/eV, nomega);
	
	//-------- Pass 2: electron-phonon coupling and dielectric response ---------
	diagMatrix GePh(TeArr.size());
	Histogram MepNum(Emin, dE, Emax);
	Histogram MepDen(Emin, dE, Emax);
	const double EconserveScaleFac = 1./dE, EconservePrefac = 1./(M_PI*dE); //energy conserving Lorentzian parameters
	logPrintf("\nePhCoupling and ImEps: "); logFlush();
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{
		//Retrieve k-point bunch:
		const std::vector< vector3<> >& kArr = kArrArr[iBunch];
		
		//Calculate electronic states and matrix elements and T_e contribution to lifetime for bunch:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector<diagMatrix> ImEarr = lineWidth(kArr);
		std::vector< std::vector<matrix> > Parr = bs.getDipoleMatElem(kArr);
		std::vector< std::vector<diagMatrix> > Farr(bunchSize); //fillings by k-point, temperature and band
		for(int ik=0; ik<bunchSize; ik++)
		{	Farr[ik].resize(TeArr.size());
			double Ejel = Ejellium(kArr[ik]);
                        double lPrefac = -1 / (32 * std::pow(M_PI,3) * std::pow(4*M_PI*epsB,2) * Es * sqrt(Ejel));
			for(size_t iT=0; iT<TeArr.size(); iT++)
			{	double invTe = 1./TeArr[iT];
				Farr[ik][iT] = Earr[ik];
				for(double& f: Farr[ik][iT]) //convert to fillings:
					f = fermi(invTe*(f-dmu[iT]));
			}
		}

		diagMatrix omegaPhArr[bunchSize];
		std::vector<matrix> gePhArr[bunchSize];
		for(int ik1=0; ik1<bunchSize; ik1++)
		{	const diagMatrix& E1 = Earr[ik1];
			const diagMatrix& ImE1 = ImEarr[ik1];
			const std::vector<diagMatrix>& F1 = Farr[ik1];
			const matrix& P1 = Parr[ik1][0];
			
			//Direct transition contributions to ImEps:
			for(int v=0; v<nBands; v++)
			{	for(int c=0; c<nBands; c++)
				{	double omega = E1[c] - E1[v]; //energy conservation
					if(omega<dE || omega>=omegaMax) continue; //irrelevant event
					double weight = (directPrefac0/(omega*omega)) * P1(c,v).norm(); //upto Te-dependent electron occupation factors
					for(size_t iT=0; iT<TeArr.size(); iT++)
					{	ImEpsDirect[iT].addEvent(omega, weight * (F1[iT][v] - F1[iT][c]));
						breadthDirect[iT].addEvent(omega, weight * (F1[iT][v] - F1[iT][c]) * (ImE1[c]+ImE1[v]+invTauTe[ik1][iT]/2));
					}	
				}
			}
			
			//Calculate phonon stuff for each pair of k-points involving ik1
			bs.setPhononMatElemArray(kArr[ik1], kArr, gePhArr);
			for(int ik2=0; ik2<bunchSize; ik2++)
				omegaPhArr[ik2] = bs.getPhononModes(kArr[ik1] - kArr[ik2]);

			for(int ik2=0; ik2<bunchSize; ik2++) if(ik2 != ik1)
			{	const diagMatrix& E2 = Earr[ik2];
				const diagMatrix& ImE2 = ImEarr[ik2];
				const std::vector<diagMatrix>& F2 = Farr[ik2];
				const matrix& P2 = Parr[ik2][0];
			
				for(int alpha=0; alpha<nModes; alpha++)
				{	const matrix& gePh = gePhArr[ik2][alpha];
					double omegaPh = omegaPhArr[ik2][alpha];
					double nPh = 1./(exp(omegaPh/Tl) - 1.);
					std::vector<double> nPh_T(TeArr.size()); //phonon occupation finite difference ratio between Tl and Te's
					for(size_t iT=0; iT<TeArr.size(); iT++)
					{	const double& Te = TeArr[iT];
						double nPhTe = 1./(exp(omegaPh/TeArr[iT]) - 1.); //phonon occupation at Te
						nPh_T[iT] = (fabs(Tl-Te) > 1e-3*Tl)
							? (nPh - nPhTe) / (Tl - Te)
							: nPh*(nPh+1)*omegaPh/(Tl*Tl); //dnPh/dTl (limit Te->Tl of above)
					}
					for(int v=0; v<nBands; v++)
					for(int c=0; c<nBands; c++)
					{	double gePhSq = gePh(c,v).norm();
						double delta = EconservePrefac/(1. + std::pow(EconserveScaleFac*(E1[v]-E2[c] + omegaPh),2));
						
						//Matrix element squared (weighted by energy conservation)
						MepNum.addEvent(E1[v], delta * gePhSq);
						MepNum.addEvent(E2[c], delta * gePhSq);
						MepDen.addEvent(E1[v], delta);
						MepDen.addEvent(E2[c], delta);
						
						//Electron-phonon heat baths coupling GePh:
						for(size_t iT=0; iT<TeArr.size(); iT++)
						{	//Note occFactors = ((f1-f2)*nPh - f2*(1-f1)) / (Tl - Te)
							//Equlibrium => f1*(1-f2)*nPhTe = (1-f1)*f2*(nPhTe+1), where nPhTe = phonon occupation at Te
							//  => 0 = (f1-f2)*nPhTe - f2*(1-f1)
							//  => occFactors = (f1-f2) * (nPh - nPhTe)/(Tl - Te)
							double occFactors = (F1[iT][v] - F2[iT][c]) * nPh_T[iT];
							GePh[iT] += GePhPrefac * omegaPh * gePhSq * occFactors * delta;
						}
						
						//Phonon-assisted transition contribution to ImEps:
						for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
						{	double omega = E2[c] - E1[v] - ae*omegaPh; //energy conservation
							if(omega<dE || omega>=omegaMax) continue; //irrelevant event
							//Effective matrix elements
							std::vector<complex> Meff(nExtrap, 0.);
							for(int i=0; i<nBands; i++) // sum over the intermediate states
							{	complex P1iv = P1(i,v);
								complex P2ci = P2(c,i);
								for(int z=0; z<nExtrap; z++)
								{	complex iEta(0, (z+1)*eta);
									Meff[z] += 
										( P2ci * gePh(i,v) / (E2[i]+iEta - (E2[c] - omega))
										+ gePh(c,i) * P1iv / (E1[i]+iEta - (E1[v] + omega)) );
								}
							}
							//Singularity extrapolation:
							double MeffSqExtrap = 0.;
							for(int z=0; z<nExtrap; z++)
								MeffSqExtrap += extrapCoeff[z] * Meff[z].norm();
							double weight = (phononPrefac0/(omega*omega)) * (nPh + 0.5*(1.-ae)) * MeffSqExtrap;
							//Include T dependent electron occupations:
							for(size_t iT=0; iT<TeArr.size(); iT++)
							{	ImEpsPhonon[iT].addEvent(omega, weight * (F1[iT][v] - F2[iT][c]));
								breadthPhonon[iT].addEvent(omega, fabs(weight * (F1[iT][v] - F2[iT][c]))*(ImE2[c]+ImE1[v]));
								weightPhonon[iT].addEvent(omega, fabs(weight * (F1[iT][v] - F2[iT][c]))); //different from ImEpsPhonon, since weight can be negative due to singularity extrapolation
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

	//Matrix element statistics:
	MepNum.allReduce(MPIUtil::ReduceSum);
	MepDen.allReduce(MPIUtil::ReduceSum);
	if(mpiUtil->isHead())
	{	ofstream ofs("Mep.dat");
		for(size_t i=0; i<MepNum.out.size(); i++)
			ofs << (MepNum.Emin + i*MepNum.dE)/eV << '\t' << MepNum.out[i]/MepDen.out[i] << '\n';
	}

	//e-ph coupling:
	GePh.allReduce(MPIUtil::ReduceSum);
	
	//ImEps:
	for(Histogram& h: ImEpsDirect) h.allReduce(MPIUtil::ReduceSum);
	for(Histogram& h: ImEpsPhonon) h.allReduce(MPIUtil::ReduceSum);
	for(Histogram& h: breadthDirect) h.allReduce(MPIUtil::ReduceSum);
	for(Histogram& h: breadthPhonon) h.allReduce(MPIUtil::ReduceSum);
	for(Histogram& h: weightPhonon) h.allReduce(MPIUtil::ReduceSum);

	//Normalize the breadths
	for(size_t iT=0; iT<TeArr.size(); iT++)
	{	for(int iomega=0; iomega<nomega; iomega++)
		{	breadthDirect[iT].out[iomega] = std::max(dE, ImEpsDirect[iT].out[iomega] ? breadthDirect[iT].out[iomega]/ImEpsDirect[iT].out[iomega] : 0.);
			breadthPhonon[iT].out[iomega] = std::max(dE, weightPhonon[iT].out[iomega] ? breadthPhonon[iT].out[iomega]/weightPhonon[iT].out[iomega] : 0.);
		}
	}

	//Apply Broadening
	std::vector<Histogram> ImEpsDirectBroad(TeArr.size(), Histogram(0, dE, omegaMax));
	std::vector<Histogram> ImEpsPhononBroad(TeArr.size(), Histogram(0, dE, omegaMax));
	int iomegaStart, iomegaStop; TaskDivision(nomega, mpiUtil).myRange(iomegaStart, iomegaStop);
	logPrintf("Applying broadening ... "); logFlush();
	for(size_t iT=0; iT<TeArr.size(); iT++)
	{	for(int iomega=iomegaStart; iomega<iomegaStop; iomega++) //input frequency grid split over MPI
		{	double omegaCur = iomega*dE;
			double bDirect = breadthDirect[iT].out[iomega];
			double bPhonon = breadthPhonon[iT].out[iomega];
			for(size_t jomega=0; jomega<ImEpsDirectBroad[iT].out.size(); jomega++) //output frequency grid
			{	double omega = jomega*dE;
				double kernelDirect = lorentzianOdd(omega, omegaCur, bDirect) * dE;
				double kernelPhonon = lorentzianOdd(omega, omegaCur, bPhonon) * dE;
				ImEpsDirectBroad[iT].out[jomega] += kernelDirect * ImEpsDirect[iT].out[iomega];
				ImEpsPhononBroad[iT].out[jomega] += kernelPhonon * ImEpsPhonon[iT].out[iomega];
			}
		}
	}
        
        for(Histogram& h: ImEpsDirectBroad) h.allReduce(MPIUtil::ReduceSum);
        for(Histogram& h: ImEpsPhononBroad) h.allReduce(MPIUtil::ReduceSum);

	if(mpiUtil->isHead())
	{	const double Omega = fabs(det(R));
		const double CeSI = Joule/(Kelvin*pow(meter,3));
		const double GePhSI = Joule*invSeconds/(Kelvin*pow(meter,3));
		ofstream ofs("TAparameters.dat");
		ofs << "#T[K] dmu[eV] Ce[J/m^3K] GePh[W/m^3K]\n";
		for(size_t iT=0; iT<TeArr.size(); iT++)
			ofs << TeArr[iT]/Kelvin << '\t'
				<< dmu[iT]/eV << '\t'
				<< Ce[iT]/(Omega*CeSI) << '\t'
				<< GePh[iT]/(Omega*GePhSI) << '\n';
		
		//Print calculated ImEps contributions:
		writeImEps("ImEps_direct.dat", ImEpsDirectBroad, TeArr);
		writeImEps("ImEps_phonon.dat", ImEpsPhononBroad, TeArr);
		
		//Print experimental dielectric function (at room temperature):
		ofstream ofsExpt("ImEps_expt.dat");
		ofsExpt << "#omega[eV] ImEpsExpt\n";
		ofstream ofsReExpt("ReEps_expt.dat");
		ofsExpt << "#omega[eV] ReEpsExpt\n";
		Epsilon eps("Wannier/epsilon.dat");
		for(size_t iomega=0; iomega<ImEpsDirect[0].out.size(); iomega++)
		{	double omega = dE * iomega;
			eps.setFrequency(omega, false);
			ofsExpt << omega/eV << '\t' << imag(eps.epsilon) << '\n';
			ofsReExpt << omega/eV << '\t' << real(eps.epsilon) << '\n';
		}
	}
	
	finalizeSystem();
}
