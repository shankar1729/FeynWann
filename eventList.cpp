#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include "BandStruct.h"
#include "Histogram.h"
#include "Epsilon.h"
#include "InputMap.h"
#include "Units.h"

int main(int argc, char** argv)
{   string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Generate event list for transport modules", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const double Eplasmon = inputMap.get("Eplasmon") * eV;
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("Eplasmon = %lg\n", Eplasmon);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Initialize dielectric model:
	Epsilon eps("Wannier/epsilon.dat");
	double omega = Eplasmon;
	eps.setFrequency(omega);
	
	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(3); //use Cartesian basis unlike other executables so that event lists can have the general uncontracted matrix element
	for(int iDir=0; iDir<3; iDir++)
		Ahat[iDir][iDir] = 1.;
	BandStruct bs("Wannier/wannier", mu, spinWeight, string(), Ahat);

	//Prepare for event collection:
	double eventPrefac = 1./(nKptsN1*fabs(det(R)));
	struct Event
	{	double Ev, Ec;
		vector3<> vv, vc;
		vector3<complex> Pcv;
	};
	std::vector<Event> events;
	events.reserve(nKptsN1/10);

	//Monte-Carlo loop over k-points:
	const double weightCut = 1e-4;
	const double mhlfByTsq = -0.5/(T*T), EconservePrefac = (0.5*spinWeight)/(T*sqrt(2*M_PI));
	int ikStart, ikStop; TaskDivision(nKptsN1, mpiUtil).myRange(ikStart, ikStop);
	int nkMine = ikStop-ikStart;
	int ikInterval = std::max(1, int(round(nkMine/50.))); //interval for reporting progress
	logPrintf("\nProgress: "); logFlush();
	for(int ik=0; ik<nkMine; ik++)
	{	//Generate random k-point:
		vector3<> kpnt;
		for(int j=0; j<3; j++)
			kpnt[j] = Random::uniform();
		//Filter k-point:
		double mk = bs.get_mk(kpnt, omega, T);
		double weightEconserveMax = EconservePrefac * exp(mhlfByTsq * mk);
		//Loop over potential events:
		if(weightEconserveMax > weightCut)
		{	diagMatrix E = bs.getStates(kpnt);
			std::vector<matrix> Pk = bs.getDipoleMatElem(kpnt);
			std::vector<vector3<>> vk = bs.getVelocity(kpnt, R, Eplasmon);
			for(int v=0; v<E.nRows(); v++) if(E[v]<10.*T)
			{	for(int c=0; c<E.nRows(); c++) if(E[c]>-10.*T)
				{	//Filter events by energy conservation:
					double mk_cv = BandStruct::mk_sub(E[c], E[v], Eplasmon, T);
					double weightEconserve = EconservePrefac * exp(mhlfByTsq * mk_cv);
					if(weightEconserve < weightCut) continue;
					//Collect momentum matrix element:
					vector3<complex> Pk_cv;
					for(int j=0; j<3; j++)
						Pk_cv[j] = Pk[j](c,v);
					//Add event:
					Event event = { E[v], E[c], vk[v], vk[c], sqrt(eventPrefac * weightEconserve) * Pk_cv };
					events.push_back(event);
				}
			}
		}
		
		//Print progress:
		if((ik+1) % ikInterval == 0)
		{	logPrintf("%d%% ", int(round((ik+1)*100./nkMine)));
			logFlush();
		}
	}
	logPrintf("done.\n"); logFlush();
	
	//Determine offsets for events from each process:
	unsigned long nEventsPrev = 0; //number of events from previous processes
	for(int jProcess=0; jProcess<mpiUtil->nProcesses(); jProcess++)
	{	unsigned long nEvents = events.size();
		mpiUtil->bcast(nEvents, jProcess); //nEvents is now the number of events on jProcess
		if(jProcess < mpiUtil->iProcess()) nEventsPrev += nEvents;
	}
	
	//Write events:
	char fname[256];
	sprintf(fname, "events-%.1lfeV.dat", Eplasmon/eV);
	MPIUtil::File fpEvent;
	mpiUtil->fopenWrite(fpEvent, fname);
	mpiUtil->fseek(fpEvent, nEventsPrev*sizeof(Event), SEEK_SET);
	mpiUtil->fwrite(events.data(), sizeof(Event), events.size(), fpEvent);
	mpiUtil->fclose(fpEvent);
	
	finalizeSystem();
}
