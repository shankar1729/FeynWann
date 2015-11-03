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


inline double fermi(double x) { return x>30. ? exp(-x) : 1./(1.+exp(x)); } //avoid overflow issues
inline double fermiPrime(double x) { return 0.25*(std::pow(tanh(0.5*x), 2) - 1.); } //avoid overflow issues
inline double ImSigmaTe(double x,double invTauTePrefac) { return 0.5*invTauTePrefac*std::pow(x, 2); }

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
	const double invTauTePrefac = inputMap.get("invTauTePrefac"); // prefactor A as in invTau(Te)=A*T^2
	const double EfJellium = inputMap.get("EfJellium") * eV; // jellium fermi energy in eV, converted to hartrees

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
	logPrintf("invTauTePrefac = %lg\n", invTauTePrefac);
	logPrintf("EfJellium = %lg\n", EfJellium);
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
                {       logPrintf("%d%% ", int(round((iBunch+1)*100./nBunchesMine)));
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
        {       const double Te = TeArr[iT], invTe = 1./Te;
                //Bisect for chemical potential:
                double& dmuCur = dmu[iT];
                double dmuMin = Emin - 10*Te;
                double dmuMax = Emax + 10*Te;
                dmuCur = 0.5*(dmuMin + dmuMax);
                const double tol = 1e-9*Te;
                while(dmuMax-dmuMin > tol)
                {       //calculate number of electrons at current Z:
                        double nElectrons = 0.;
                        for(size_t ie=0; ie<dos.out.size(); ie++)
                        {       double Ei = Emin + ie*dE;
                                double fi = fermi(invTe*(Ei - dmuCur));
                                nElectrons += dE * dos.out[ie] * fi;
                        }
                        ((nElectrons>Z) ? dmuMax : dmuMin) = dmuCur;
                        dmuCur = 0.5*(dmuMin + dmuMax);
                }
        }
        dmu.allReduce(MPIUtil::ReduceSum);

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

	
	//-------- Pass 2: electron-phonon coupling and dielectric response ---------
	Histogram MepNum(Emin, dE, Emax);
	Histogram MepDen(Emin, dE, Emax);
	Histogram MepAshcroft(Emin, dE, Emax);
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{
		//Retrieve k-point bunch:
		const std::vector< vector3<> >& kArr = kArrArr[iBunch];
		
		//Calculate electronic states and matrix elements for bunch:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector<diagMatrix> ImEarr = lineWidth(kArr);
		std::vector< std::vector<diagMatrix> > Farr(bunchSize); //fillings by k-point, temperature and band
		for(int ik=0; ik<bunchSize; ik++)
		{	Farr[ik].resize(TeArr.size());
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
			//Calculate phonon stuff for each pair of k-points involving ik1
			bs.setPhononMatElemArray(kArr[ik1], kArr, gePhArr);
			for(int ik2=0; ik2<bunchSize; ik2++)
				omegaPhArr[ik2] = bs.getPhononModes(kArr[ik1] - kArr[ik2]);

			for(int ik2=0; ik2<bunchSize; ik2++) if(ik2 != ik1)
			{	const diagMatrix& E2 = Earr[ik2];
				for(int alpha=0; alpha<nModes; alpha++)
				{	double omegaPh = omegaPhArr[ik2][alpha];
						
					for(int v=0; v<nBands; v++)
                                        for(int c=0; c<nBands; c++)
					{	//Approxiimate matrix element squared from Ashcroft & Mermin p.523
						const double Omega = fabs(det(R));
						MepAshcroft.addEvent(E1[v], omegaPh * EfJellium / (3*Z));
						MepAshcroft.addEvent(E2[c], omegaPh * EfJellium / (3*Z));

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

	//Ashcroft Matrix element statistics:
	MepAshcroft.allReduce(MPIUtil::ReduceSum);
	if(mpiUtil->isHead())
	{	ofstream ofss("MepAshcroft.dat");
		for(size_t i=0; i<MepAshcroft.out.size(); i++)
			ofss << (MepAshcroft.Emin + i*MepAshcroft.dE)/eV << '\t' << MepAshcroft.out[i] << '\n';
	}
	
	finalizeSystem();
}
