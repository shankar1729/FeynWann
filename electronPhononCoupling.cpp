#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include <core/Units.h>
#include "InputMap.h"
#include "Histogram.h"
#include "FeynWann.h"


//Get energy range from an eLoop call:
struct EnergyRange
{	double Emin;
	double Emax;
	
	static void eProcess(const FeynWann::StateE& state, void* params)
	{	EnergyRange& er = *((EnergyRange*)params);
		er.Emin = std::min(er.Emin, state.E.front()); //E is in ascending order
		er.Emax = std::max(er.Emax, state.E.back()); //E is in ascending order
	}
};

//Collect ImEps contibutions using FeynWann callbacks:
struct CollectHePh
{	Histogram hEph;
	double prefac;
	
	CollectHePh(double Emin, double dE, double Emax)
	: hEph(Emin, dE, Emax)
	{	logPrintf("Initialized energy grid: %lg to %lg eV with %d points.\n", hEph.Emin/eV, hEph.Emax()/eV, hEph.nE);
	}
	
	void calcLinewidth(const FeynWann::StateE& state, diagMatrix& ImE)
	{	int nBands = state.E.nRows();
		ImE = state.ImSigma_ee; //e-e part
		for(int b=0; b<nBands; b++)
			ImE[b] += state.ImSigma_ePh(b, state.E[b]<0. ? 1. : 0.); //e-ph part
	}
	
    void collect(const FeynWann::MatrixEph& mat)
	{	int nBands = mat.e1->E.nRows();
		//Get energies and linewidths:
		const diagMatrix& E1 = mat.e1->E;
		const diagMatrix& E2 = mat.e2->E;
		diagMatrix ImE1, ImE2;
		calcLinewidth(*mat.e1, ImE1);
		calcLinewidth(*mat.e2, ImE2);
		const diagMatrix& omegaPh = mat.ph->omega;
		int nModes = omegaPh.nRows();
		//Collect
		for(int v=0; v<nBands; v++)
		{	for(int c=0; c<nBands; c++)
			{	for(int alpha=0; alpha<nModes; alpha++)
				{	double gePhSq = mat.M[alpha](c,v).norm();
					double breadth = ImE1[v] + ImE2[c];
					double delta = 1./(M_PI*breadth*(1. + std::pow((E1[v]-E2[c] + omegaPh[alpha])/breadth,2)));
					hEph.addEvent(0.5*(E1[v]+E2[c]), prefac * delta * omegaPh[alpha] * gePhSq);
				}
			}
		}
	}
	static void ePhProcess(const FeynWann::MatrixEph& mat, void* params)
	{	((CollectHePh*)params)->collect(mat);
	}
};

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Energy-resolved electron-phonon coupling strength");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const int nOffsets = inputMap.get("nOffsets"); assert(nOffsets>0);
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("dE = %lg\n", dE);
	
	//Initialize FeynWann:
	FeynWannParams fwp;
	fwp.needPhonons = true;
	fwp.needLinewidth_ee = true;
	fwp.needLinewidth_ePh = true;
	std::shared_ptr<FeynWann> fw = std::make_shared<FeynWann>(fwp);
	size_t nKeff = nOffsets * fw->ePhCountPerOffset();
	logPrintf("Effectively sampled nKpairs: %lu\n", nKeff);

	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
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
	
	//Initialize energy grid:
	EnergyRange er = { DBL_MAX, -DBL_MAX };
	fw->eLoop(vector3<>(), EnergyRange::eProcess, &er);
	mpiWorld->allReduce(er.Emin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(er.Emax, MPIUtil::ReduceMax);
	er.Emin = dE * (floor(er.Emin/dE) - 10); //add some margin and ensure grid contains 0
	er.Emax = dE * (ceil(er.Emax/dE) + 10);
	
	//Collect e-ph coupling resolved by energy:
	CollectHePh ch(er.Emin, dE, er.Emax);
	ch.prefac = fw->spinWeight / (nKeff*fabs(det(fw->R)));
	for(int iSpin=0; iSpin<fw->nSpins; iSpin++)
	{	//Update FeynWann for spin channel if necessary:
		if(iSpin>0)
		{	fw = 0; //free memory from previous spin
			fwp.iSpin = iSpin;
			fw = std::make_shared<FeynWann>(fwp);
		}
		logPrintf("\nCollecting hEph: "); logFlush();
		for(int o=0; o<noMine; o++)
		{	Random::seed(o+oStart); //to make results independent of MPI division
			vector3<> k01 = fw->randomVector(mpiGroup); //must be constant across group
			vector3<> k02 = fw->randomVector(mpiGroup); //must be constant across group
			fw->ePhLoop(k01, k02, CollectHePh::ePhProcess, &ch);
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
	}
	ch.hEph.allReduce(MPIUtil::ReduceSum);
	ch.hEph.print("hEph.dat", 1./eV, 1./(eV/pow(Angstrom,3)));
	
	fw = 0;
	FeynWann::finalize();
}
