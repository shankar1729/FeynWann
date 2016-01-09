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
inline double ImSigmaTe(double x,double invTauTePrefac) { return 0.5*invTauTePrefac*std::pow(x, 2); }

inline int findNearestNeighbourIndex(double value,std::vector<double> &x)
{	double dist = DBL_MAX;
	int idx = -1;
	for(size_t i=0; i<x.size(); ++i)
	{	double newDist = value-x[i];
		if(newDist>0 && newDist<dist)
		{	dist = newDist;
			idx=i;
		}
	}
	return idx;
}

inline std::vector<double> interp(std::vector<double> &x, std::vector<double> &y, std::vector<double> &newX)
{	std::vector<double> newY, dx, dy, slope, intercept;
	for(size_t i=1; i<x.size(); i++)
	{	if(i<x.size()-1)
		{	dx.push_back(x[i+1]-x[i]);
			dy.push_back(y[i+1]-y[i]);
			slope.push_back(dy[i]/dx[i]);
			intercept.push_back(y[i]-x[i]*slope[i]);
		}
		else
		{	dx.push_back( dx[i-1]);
			dy.push_back( dy[i-1]);
			slope.push_back( slope[i-1]);
			intercept.push_back(intercept[i-1]);
		}
	}

	for(size_t i=1; i<newX.size(); i++)
	{	int idx = findNearestNeighbourIndex(newX[i],x);
		newY.push_back(slope[idx]*newX[i]+intercept[idx]);
	}
	return newY;
}


inline double interp1(std::vector<double> &x, std::vector<double> &y, double newX)
{	int idx = findNearestNeighbourIndex(newX,x);
	double dx = x[idx+1]-x[idx];
	double dy = y[idx+1]-y[idx];
	double slope = dy/dx;
	double intercept = y[idx]-x[idx]*slope;
	double newY = slope * newX + intercept;
	return newY;
}


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
	const double Tl = inputMap.get("Tl") * Kelvin; //lattice temperature
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
	const double invTauTePrefac = inputMap.get("invTauTePrefac"); // prefactor A as in invTau(Te)=A*T^2
	const double EfJellium = inputMap.get("EfJellium") * eV; // jellium fermi energy in eV, converted to hartrees
	const int numEnergies = inputMap.get("eeRelaxNumRows");
	const int numTimes = inputMap.get("eeRelaxNumCols");

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("mu = %lg\n", mu);
	logPrintf("Z = %lg\n", Z);
	logPrintf("dE = %lg\n", dE);
	logPrintf("Tl = %lg\n", Tl);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	logPrintf("invTauTePrefac = %lg\n", invTauTePrefac);
	logPrintf("EfJellium = %lg\n", EfJellium);
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Read in nonthermal electron distribution function from eeRleax code:
	ifstream inFile1;
	inFile1.open("invTau.dat");
	std::vector<diagMatrix> distFunctArr;
	double inputEnergySI;
	std::vector<double> inputEnergy;
	for (int a = 0; a < numEnergies; a++)
	{	for (int iT = 0; iT < numTimes; iT++)
		{	if (iT==0)
			{	inFile1 >> inputEnergySI;
				inputEnergy.push_back(inputEnergySI*eV); // convert to atomic units
			}
			if (iT>0) inFile1 >> distFunctArr[iT-1][a];
		}
	}
	//The rest of the code assumes that the matrix distFunctArr has the distribution function for each time in each column, first column is energy

	//Read in electron linewidth correction from eeRleax dode:
	ifstream inFile2;
	inFile2.open("invTau.dat");
	std::vector<diagMatrix> LWcorrection;
	double trash;
	for (int a = 0; a < numEnergies; a++)
	{	for (int b = 0; b < numTimes; b++)
		{       if (b==0) inFile1 >> trash;
			if (b>0) inFile1 >> LWcorrection[a][b-1];
		}
	}


	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(1); //assume cubic symmetry and only calculate x-axis
	Ahat[0] = vector3<complex>(1., 0., 0.);
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE", Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);

	
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
	
	//Singularity extrapolation parameters
	double extrapCoeff[] = {-19./12, 13./3, -7./4 }; //account for constant, 1/eta and eta^2 dependence
	//double extrapCoeff[] = { -1, 2.}; //account for constant and 1/eta dependence
	const int nExtrap = sizeof(extrapCoeff)/sizeof(double);
	const double eta = 0.1*eV;

	// -------------------------------------  Setup --------------------------------------
	
	std::vector< std::vector< vector3<> > > kArrArr(nBunchesMine); //use exact same set of MC k-points in the two passes for consistency
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{	//Generate a bunch of k-points:
		std::vector< vector3<> >& kArr = kArrArr[iBunch];
		kArr.resize(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();

        }

	//Initalize line width of electronic states
	LineWidth lineWidth("Wannier/wannier", bs);

	//Initialize frequency grid:
	double omegaMax = 77.210*eV;

	//Initialize unbroadened histograms:
	std::vector<Histogram> ImEpsDirect(numTimes, Histogram(0, dE, omegaMax)), breadthDirect(numTimes, Histogram(0, dE, omegaMax));
	std::vector<Histogram> ImEpsPhonon(numTimes, Histogram(0, dE, omegaMax)), breadthPhonon(numTimes, Histogram(0, dE, omegaMax)),  weightPhonon(numTimes, Histogram(0, dE, omegaMax));
	int nomega = ImEpsDirect[0].out.size();
	logPrintf("Initialized frequency grid: 0 to %lg eV with %d points.\n", (dE*(nomega-1))/eV, nomega);
	
	//-------- Pass 2: electro n-phonon coupling and dielectric response ---------
	const double EconserveScaleFac = 1./dE, EconservePrefac = 1./(M_PI*dE); //energy conserving Lorentzian parameters
	logPrintf("\nePhCoupling and ImEps: "); logFlush();
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{
		//Retrieve k-point bunch:
		const std::vector< vector3<> >& kArr = kArrArr[iBunch];
		
		//Calculate electronic states and matrix elements for bunch:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector<diagMatrix> ImEarr = lineWidth(kArr);
		std::vector< std::vector<matrix> > Parr = bs.getDipoleMatElem(kArr);
		std::vector< std::vector<diagMatrix> > Farr(bunchSize); //fillings by k-point, temperature and band
		for(int ik=0; ik<bunchSize; ik++)
		{	Farr[ik].resize(numTimes);
			for(int iT=0; iT<numTimes; iT++)
			{	Farr[ik][iT] = Earr[ik];
				for(double& f: Farr[ik][iT]) //convert to fillings:
					f = interp1(inputEnergy,distFunctArr[iT],f);
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
					for(int iT=0; iT<numTimes; iT++)
					{	ImEpsDirect[iT].addEvent(omega, weight * (F1[iT][v] - F1[iT][c]));
						breadthDirect[iT].addEvent(omega, weight * (F1[iT][v] - F1[iT][c]) * (ImE1[c]+ImE1[v]));
						//breadthDirect[iT].addEvent(omega, weight * (F1[iT][v] - F1[iT][c]) * (ImE1[c]+ImE1[v]+invTauTe[ik1][iT]/2));
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
					for(int v=0; v<nBands; v++)
					for(int c=0; c<nBands; c++)
					{	
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
							for(int iT=0; iT<numTimes; iT++)
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

	//ImEps:
	for(Histogram& h: ImEpsDirect) h.allReduce(MPIUtil::ReduceSum);
	for(Histogram& h: ImEpsPhonon) h.allReduce(MPIUtil::ReduceSum);
	for(Histogram& h: breadthDirect) h.allReduce(MPIUtil::ReduceSum);
	for(Histogram& h: breadthPhonon) h.allReduce(MPIUtil::ReduceSum);
	for(Histogram& h: weightPhonon) h.allReduce(MPIUtil::ReduceSum);

	//Normalize the breadths
	for(int iT=0; iT<numTimes; iT++)
	{	for(int iomega=0; iomega<nomega; iomega++)
		{	breadthDirect[iT].out[iomega] = std::max(dE, ImEpsDirect[iT].out[iomega] ? breadthDirect[iT].out[iomega]/ImEpsDirect[iT].out[iomega] : 0.);
			breadthPhonon[iT].out[iomega] = std::max(dE, weightPhonon[iT].out[iomega] ? breadthPhonon[iT].out[iomega]/weightPhonon[iT].out[iomega] : 0.);
		}
	}

	//Apply Broadening
	std::vector<Histogram> ImEpsDirectBroad(numTimes, Histogram(0, dE, omegaMax));
	std::vector<Histogram> ImEpsPhononBroad(numTimes, Histogram(0, dE, omegaMax));
	int iomegaStart, iomegaStop; TaskDivision(nomega, mpiUtil).myRange(iomegaStart, iomegaStop);
	logPrintf("Applying broadening ... "); logFlush();
	for(int iT=0; iT<numTimes; iT++)
	{	for(int iomega=iomegaStart; iomega<iomegaStop; iomega++) //input frequency grid split over MPI
		{	double omegaCur = iomega*dE;
			double bDirect = breadthDirect[iT].out[iomega] + 2*ImSigmaTe(TeArr[iT],invTauTePrefac);//recal breadth and add Te dependence of lifetime
			double bPhonon = breadthPhonon[iT].out[iomega] + 2*ImSigmaTe(TeArr[iT],invTauTePrefac);//recal breadth and add Te dependence of lifetime
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
	{	//Print calculated ImEps contributions:
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
