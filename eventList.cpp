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

inline vector3<complex> operator*(const vector3<complex>& v, complex s)
{	return vector3<complex>(v[0]*s, v[1]*s, v[2]*s);
}
inline vector3<complex> conj(const vector3<complex>& v)
{	return vector3<complex>(v[0].conj(), v[1].conj(), v[2].conj());
}

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
	BandStruct bs("Wannier/wannier", mu, spinWeight, "Wannier/totalE", Ahat);
	bs.setCacheSize(4);

	//Prepare for event collection:
	struct Event
	{	double Ev, Ec;
		vector3<> vv, vc;
		matrix3<complex> PcvSq;
	};
	std::vector<Event> events, eventsPh;
	
	//Singularity extrapolation parameters
	double extrapCoeff[] = {-19./12, 13./3, -7./4 }; //account for constant, 1/eta and eta^2 dependence
	//double extrapCoeff[] = { -1, 2.}; //account for constant and 1/eta dependence
	const int nExtrap = sizeof(extrapCoeff)/sizeof(double);
	const double eta = 0.1*eV;

	//Monte-Carlo loop over k-points:
	const double mhlfByTsq = -0.5/(T*T), EconservePrefac = (0.5*spinWeight)/(T*sqrt(2*M_PI));
	const double weightCut = 1e-2 * EconservePrefac;
	size_t ieStart, ieStop; TaskDivision(nKptsN1, mpiUtil).myRange(ieStart, ieStop);
	size_t neMine = ieStop-ieStart;
	size_t ieInterval = std::max(1, int(round(neMine/50.))); //interval for reporting progress
	events.reserve(neMine);
	eventsPh.reserve(neMine);
	logPrintf("\nProgress: "); logFlush();
	size_t nkDirect = 0;
	size_t nkPhonon = 0;
	size_t neCur = 0, nePrev = 0;
	for(int ik=0; neCur<neMine; ik++)
	{	nkDirect += 2;
		nkPhonon++;
		//Generate a pair of random k-points:
		std::vector< vector3<> > kArr(2);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		//Filter k1,k2 for direct and (k1,k2) for phonon:
		double weightEconserveMax[2];
		for(int iPair=0; iPair<2; iPair++)
			weightEconserveMax[iPair] = EconservePrefac * exp(mhlfByTsq * bs.get_mk(kArr[iPair], omega, T));
		double weightEconserveMax12 = EconservePrefac * exp(mhlfByTsq * bs.get_mk1k2(kArr[0], kArr[1], omega, T));
		if((weightEconserveMax[0] > weightCut) || (weightEconserveMax[1] > weightCut) || (weightEconserveMax12 > weightCut))
		{	std::vector<std::vector<matrix> > Parr = bs.getDipoleMatElem(kArr);
			//Direct contributions:
			for(int iPair=0; iPair<2; iPair++) if(weightEconserveMax[iPair] > weightCut)
			{	const diagMatrix& E = Earr[iPair];
				const std::vector<matrix>& P = Parr[iPair];
				for(int v=0; v<E.nRows(); v++) if(E[v]<10.*T)
				{	for(int c=0; c<E.nRows(); c++) if(E[c]>-10.*T)
					{	//Filter events by energy conservation:
						double mk_cv = BandStruct::mk_sub(E[c], E[v], Eplasmon, T);
						double weightEconserve = EconservePrefac * exp(mhlfByTsq * mk_cv);
						if(weightEconserve < weightCut) continue;
						//Collect momentum matrix element and velocities:
						vector3<complex> P_cv; vector3<> vv, vc;
						for(int j=0; j<3; j++)
						{	P_cv[j] = P[j](c,v);
							vv[j] = -P[j](v,v).imag(); //(P is calculated without an i to make things real when possible)
							vc[j] = -P[j](c,c).imag();
						}
						//Add event:
						Event event = { E[v], E[c], vv, vc, complex(weightEconserve) * outer(P_cv, conj(P_cv)) };
						events.push_back(event);
					}
				}
			}
			//Phonon contributions:
			if(weightEconserveMax12 > weightCut)
			{	const diagMatrix& E1 = Earr[0];
				const diagMatrix& E2 = Earr[1];
				const std::vector<matrix>& P1 = Parr[0];
				const std::vector<matrix>& P2 = Parr[1];
				diagMatrix omegaPh = bs.getPhononModes(kArr[0]-kArr[1]);
				std::vector<matrix> gePh = bs.getPhononMatElem(kArr[0], kArr[1]);
				for(int v=0; v<E1.nRows(); v++) if(E1[v]<10.*T)
				{	for(int c=0; c<E2.nRows(); c++) if(E2[c]>-10.*T)
					{	Event e;
						e.Ev = E1[v];
						e.Ec = E2[c];
						for(int j=0; j<3; j++)
						{	e.vv[j] = -P1[j](v,v).imag(); //(P is calculated without an i to make things real when possible)
							e.vc[j] = -P2[j](c,c).imag();
						}
						double weightEconserveTot = 0.;
						for(int alpha=0; alpha<omegaPh.nRows(); alpha ++)
						{	for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
							{	double mk_cv = BandStruct::mk_sub(E2[c], E1[v], Eplasmon + ae*omegaPh[alpha], T);
								double nPh = 1./(exp(omegaPh[alpha]/T) - 1.);
								double weightEconserve = EconservePrefac * exp(mhlfByTsq * mk_cv) * (nPh+0.5*(1.-ae)); //weight contribution (including phonon and electron occupation factors) due to energy conservation
								if(weightEconserve < weightCut) continue;
								weightEconserveTot += weightEconserve;
								// Effective matrix elements
								std::vector< vector3<complex> > Pcv_eff(nExtrap);
								for(int i=0; i<E1.nRows(); i++) // sum over the intermediate states
								{	vector3<complex> P1_iv, P2_ci;
									for(int j=0; j<3; j++)
									{	P1_iv[j] = P1[j](i,v);
										P2_ci[j] = P2[j](c,i);
									}
									for(int z=0; z<nExtrap; z++)
									{	complex iEta(0, (z+1)*eta);
										Pcv_eff[z] +=
											( P1_iv * (gePh[alpha](c,i) / (E1[i]+iEta - (E1[v] + Eplasmon)))
											+ P2_ci * (gePh[alpha](i,v) / (E2[i]+iEta - (E2[c] - Eplasmon))) );
									}
								}
								//Singularity extrapolation:
								for(int z=0; z<nExtrap; z++)
									e.PcvSq += complex(weightEconserve * extrapCoeff[z]) * outer(Pcv_eff[z], conj(Pcv_eff[z]));
							}
						}
						if(weightEconserveTot < weightCut) continue;
						eventsPh.push_back(e);
					}
				}
			}
		}
		
		//Print progress:
		neCur = events.size() + eventsPh.size();
		if(neCur - nePrev >= ieInterval)
		{	logPrintf("%d%% ", int(round(neCur*100./neMine)));
			logFlush();
			nePrev = neCur;
		}
	}
	logPrintf("done.\n"); logFlush();
	
	//Normalize events based on the total k-points explored:
	mpiUtil->allReduce(nkDirect, MPIUtil::ReduceSum);
	mpiUtil->allReduce(nkPhonon, MPIUtil::ReduceSum);
	double directPrefac = 4 * std::pow(M_PI,2) * spinWeight / (nkDirect*fabs(det(R)) * omega*omega); //prefactor that yields Im(eps)
	double phononPrefac = 4 * std::pow(M_PI,2) * spinWeight / (nkPhonon*fabs(det(R)) * omega*omega); //prefactor that yields Im(eps)
	for(Event& e: events) e.PcvSq *= directPrefac;
	for(Event& e: eventsPh) e.PcvSq *= phononPrefac;
	
	//Determine offsets for events from each process:
	unsigned long nEventsPrev = 0, nEventsPhPrev = 0; //number of events from previous processes
	for(int jProcess=0; jProcess<mpiUtil->nProcesses(); jProcess++)
	{	unsigned long nEvents = events.size(), nEventsPh = eventsPh.size();
		mpiUtil->bcast(nEvents, jProcess); //nEvents is now the number of events on jProcess
		mpiUtil->bcast(nEventsPh, jProcess); //nEventsPh is now the number of eventsPh on jProcess
		if(jProcess < mpiUtil->iProcess())
		{	nEventsPrev += nEvents;
			nEventsPhPrev += nEventsPh;
		}
	}
	
	//Write events:
	char fname[256];
	MPIUtil::File fpEvent;
	//--- direct
	sprintf(fname, "events-%.1lfeV.dat", Eplasmon/eV);
	mpiUtil->fopenWrite(fpEvent, fname);
	mpiUtil->fseek(fpEvent, nEventsPrev*sizeof(Event), SEEK_SET);
	mpiUtil->fwrite(events.data(), sizeof(Event), events.size(), fpEvent);
	mpiUtil->fclose(fpEvent);
	//--- phonon
	sprintf(fname, "eventsPh-%.1lfeV.dat", Eplasmon/eV);
	mpiUtil->fopenWrite(fpEvent, fname);
	mpiUtil->fseek(fpEvent, nEventsPhPrev*sizeof(Event), SEEK_SET);
	mpiUtil->fwrite(eventsPh.data(), sizeof(Event), eventsPh.size(), fpEvent);
	mpiUtil->fclose(fpEvent);
	
	finalizeSystem();
}
