#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"
#include <core/Units.h>
#include "Histogram.h"

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of resistivity", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	long nKpts = inputMap.get("nKpts");
	const double Zjellium = inputMap.get("Zjellium");

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKpts = %ld\n", nKpts);
	logPrintf("Zjellium = %lg\n", Zjellium);

	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(1); //assume cubic symmetry and only calculate x-axis
	Ahat[0] = vector3<complex>(1., 0., 0.);
	BandStruct bs("Wannier/totalE", "Wannier/wannier", false, Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);

	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");
	
	//Initialize energy grid:
	double dE = 0.1*eV;
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
	Histogram dos(Emin, dE, Emax); //density of states
	Histogram KEeff(Emin, dE, Emax); //effective KE (calculated as p^2/2)
	logPrintf("Initialized energy grid: %lg to %lg eV with %lu points.\n", Emin/eV, (Emin+dE*(dos.out.size()-1))/eV, dos.out.size());
	
	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKpts, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	int nBands = Egamma.nRows();

	logPrintf("\nCollecting DOS and KE: "); logFlush();
	const double dosWeight = bs.spinWeight*(1./nKpts);
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{
		//Generate a bunch of k-points:
		std::vector<vector3<>> kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		
		//Collect DOS:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector< std::vector<matrix> > Parr = bs.getDipoleMatElem(kArr);
		for(int ik=0; ik<bunchSize; ik++)
		{	const diagMatrix& E = Earr[ik];
			const matrix& Px = Parr[ik][0];
			for(int b=0; b<nBands; b++)
			{	dos.addEvent(E[b], dosWeight);
				KEeff.addEvent(E[b], dosWeight * ((3./2)*Px(b,b).norm())); //factor of 3 for x,y,z (cubic symmetry)
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
	KEeff.allReduce(MPIUtil::ReduceSum);

	//Jellium parameters:
	const double nJellium = Zjellium / fabs(det(bs.R));
	const double kF = std::pow(3*M_PI*M_PI*nJellium, 1./3);
	const double eF = 0.5*kF*kF;

	if(mpiUtil->isHead())
	{	ofstream ofs("EffectiveKE.dat");
		ofs << "#E-Ef[eV] KEff[eV] KEfree[eV] DOS[1/eV]\n";
		for(size_t iE=0; iE<KEeff.out.size(); iE++)
		{	double E = (KEeff.Emin+KEeff.dE*iE);
			ofs << E/eV << '\t'
				<< (KEeff.out[iE]/dos.out[iE])/eV << '\t'
				<< std::max(0.,eF+E)/eV << '\t'
				<< dos.out[iE]*eV << '\n';
		}
	}
	
	finalizeSystem();
}