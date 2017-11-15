#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "LineWidth.h"
#include "InputMap.h"
#include <core/Units.h>
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
	long nKpts = inputMap.get("nKpts");
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKpts = %ld\n", nKpts);
	logPrintf("dE = %lg\n", dE);
	
	//Initialize Wannier bandstructure:
	BandStruct bs("Wannier/totalE", "Wannier/wannier", true);
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
	mpiWorld->allReduce(Emin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(Emax, MPIUtil::ReduceMax);
	Emin -= 10*dE; //add some margin
	Emax += 10*dE;
	Histogram dos(Emin, dE, Emax); //density of states
	Histogram hInt(Emin, dE, Emax); //main integral in energy-resolved coupling
	logPrintf("Initialized energy grid: %lg to %lg eV with %lu points.\n", Emin/eV, (Emin+dE*(dos.out.size()-1))/eV, dos.out.size());
	
	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKpts, mpiWorld).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	nKpts = nBunchesMine * bunchSize; mpiWorld->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	long nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	int nBands = Egamma.nRows();
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	
	logPrintf("\nCollecting g and h: "); logFlush();
	std::vector< std::vector< vector3<> > > kArrArr(nBunchesMine); //use exact same set of MC k-points in the two passes for consistency
	const double dosWeight = bs.spinWeight/(nKpts*fabs(det(bs.R)));
	const double hIntWeight = bs.spinWeight/(nKpairs*fabs(det(bs.R)));
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
		
		//Collect double integral in h(epsilon):
		std::vector<diagMatrix> ImEarr = lineWidth(kArr);
		diagMatrix omegaPhArr[bunchSize];
		std::vector<matrix> gePhArr[bunchSize];
		for(int ik1=0; ik1<bunchSize; ik1++)
		{	const diagMatrix& E1 = Earr[ik1];
			const diagMatrix& ImE1 = ImEarr[ik1];
			
			//Calculate phonon stuff for each pair of k-points involving ik1
			bs.setPhononMatElemArray(kArr[ik1], kArr, gePhArr);
			for(int ik2=0; ik2<bunchSize; ik2++)
				omegaPhArr[ik2] = bs.getPhononModes(kArr[ik1] - kArr[ik2]);
			
			//Loop over all pairs of electron and phonon states:
			for(int ik2=0; ik2<bunchSize; ik2++) if(ik2 != ik1)
			{	const diagMatrix& E2 = Earr[ik2];
				const diagMatrix& ImE2 = ImEarr[ik2];
				for(int alpha=0; alpha<nModes; alpha++)
				{	const matrix& gePh = gePhArr[ik2][alpha];
					double omegaPh = omegaPhArr[ik2][alpha];
					for(int v=0; v<nBands; v++)
					for(int c=0; c<nBands; c++)
					{	double gePhSq = gePh(c,v).norm();
						double breadth = ImE1[v] + ImE2[c];
						double delta = 1./(M_PI*breadth*(1. + std::pow((E1[v]-E2[c] + omegaPh)/breadth,2)));
						hInt.addEvent(E1[v], hIntWeight * delta * omegaPh * gePhSq);
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
	dos.allReduce(MPIUtil::ReduceSum);
	hInt.allReduce(MPIUtil::ReduceSum);
	
	//Output:
	if(mpiWorld->isHead())
	{	//Fermi level DOS:
		double gF = 0.;
		for(size_t i=0; i<dos.out.size(); i++)
		{	double E = dos.Emin + i*dos.dE;
			gF += dos.out[i]/(M_PI*(1.+std::pow(E/dos.dE,2)));
		}
		//Apply smoothing:
		Histogram dosBar(Emin, dE, Emax);
		Histogram hIntBar(Emin, dE, Emax);
		for(size_t i=0; i<dos.out.size(); i++)
		{	double E = dos.Emin + i*dos.dE;
			for(size_t iOut=0; iOut<dos.out.size(); iOut++)
			{	double Eout = dos.Emin + iOut*dos.dE;
				double smoothKernel = 1./(M_PI*(1.+std::pow((Eout-E)/dos.dE,2)));
				dosBar.out[iOut] += smoothKernel * dos.out[i];
				hIntBar.out[iOut] += smoothKernel * hInt.out[i];
			}
		}
		//Output:
		ofstream ofs("h.dat");
		for(size_t i=0; i<dos.out.size(); i++)
		{	double E = dos.Emin + i*dos.dE;
			double g = dosBar.out[i];
			double h = (2*gF/(g*g))*hIntBar.out[i];
			ofs << E/eV << '\t' << h/std::pow(1e-3*eV,2) << '\n';
		}
		ofs.close();
		ofs.open("hInt.dat");
		for(size_t i=0; i<dos.out.size(); i++)
		{	double E = dos.Emin + i*dos.dE;
			double hInt = hIntBar.out[i]; //dimensions of h * g = energy/volume
			ofs << E/eV << '\t' << hInt/(eV/pow(Angstrom,3)) << '\n';
		}
	}
	
	finalizeSystem();
}
