#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include "BandStruct.h"
#include "InputMap.h"
#include <core/Units.h>
#include "Histogram.h"

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "e-phonon scattering properties", inputFilename, dryRun, printDefaults);
	
	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	long nKpts = inputMap.get("nKpts");
	const double T = inputMap.get("T") * Kelvin;
	const double EplasmonMax = inputMap.get("EplasmonMax") * eV;

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKpts = %ld\n", nKpts);
	logPrintf("T = %lg\n", T);
	logPrintf("EplasmonMax = %lg\n", EplasmonMax);
	
	//Initialize Wannier bandstructure:
	const int bunchSize = 32;
	BandStruct bs("Wannier/totalE", "Wannier/wannier", true);
	bs.setCacheSize(2*bunchSize);

	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");
	
	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKpts, mpiUtil).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	nKpts = nBunchesMine * bunchSize; mpiUtil->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	long nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-scattering events
	int nBands = bs.getStates(vector3<>()).nRows();
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	
	//Determine phonon energy scale:
	double EphMax = 0.;
	for(int i=0; i<10*bunchSize; i++)
	{	vector3<> k;
		for(int j=0; j<3; j++)
			k[j] = Random::uniform();
		EphMax = std::max(EphMax, bs.getPhononModes(k).back());
	}
	mpiUtil->allReduce(EphMax, MPIUtil::ReduceMax);
	logPrintf("Maximum phonon energy = %lg eV\n", EphMax/eV);
	
	//Binning:
	const int nBins = 20; //logical number of bins before margins
	double db = 2./nBins;
	double bStart = -1. - db;
	double bStop = 1. + 2.*db;
	
	//Initialize histograms
	const double dE = 0.1*eV;
	Histogram dos(-EplasmonMax,dE,EplasmonMax);
	Histogram lbdaInv(-EplasmonMax,dE,EplasmonMax); //inverse mean free path by energy
	Histogram2D deltaE(-EplasmonMax,dE,EplasmonMax, bStart*EphMax,db*EphMax,bStop*EphMax); //scattered energy change by initial energy
	Histogram2D cosTheta(-EplasmonMax,dE,EplasmonMax, bStart,db,bStop); //scattering angle by initial energy
	
	//Energy conservation parameters:
	double eta = 0.5*T;
	double EconserveExpFac = -0.5/(eta*eta);
	double EconservePrefac = 1./(sqrt(2*M_PI)*eta);
	double prefacTauInv = bs.spinWeight*(2*M_PI)/nKpairs;
	double prefacDos = bs.spinWeight*1./nKpts;
	
	//Monte Carlo loop:
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	logPrintf("\nProgress: "); logFlush();
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{
		//Generate a bunch of k-points:
		std::vector< vector3<> > kArr(bunchSize);
		for(vector3<>& k: kArr)
			for(int j=0; j<3; j++)
				k[j] = Random::uniform();
		
		//Calculate electronic states, occupations and velocities for bunch:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector<diagMatrix> Farr = Earr; //convert to fillings:
		for(diagMatrix& F: Farr)
			for(double& f: F)
			{	double e = f/T; //E/T actually
				f = (e>30 ? exp(-e) : 1./(1. + exp(e))); //avoid overflow issues
			}
		std::vector< vector3<> > vArr[bunchSize];
		for(int ik=0; ik<bunchSize; ik++)
			vArr[ik] = bs.getVelocity(kArr[ik], EplasmonMax);
	
		//Loop over k-pairs for e-ph interactions:
		for(int ik1=0; ik1<bunchSize; ik1++)
		{	const diagMatrix& E1 = Earr[ik1];
			const std::vector< vector3<> >& v1 = vArr[ik1];
			//collect dos for normalization:
			for(int b1=0; b1<nBands; b1++)
				dos.addEvent(E1[b1], prefacDos);
			//phonon matrix elements for ik1 with rest of bunch:
			std::vector<matrix> gePhArr[bunchSize];
			bs.setPhononMatElemArray(kArr[ik1], kArr, gePhArr);
			//Loop over second k-point:
			for(int ik2=0; ik2<bunchSize; ik2++) if(ik2!=ik1) //avoid gamma-point phonon singularity
			{	const std::vector<matrix>& gePh = gePhArr[ik2];
				diagMatrix omegaPh = bs.getPhononModes(kArr[ik1] - kArr[ik2]);
				const diagMatrix& E2 = Earr[ik2];
				const diagMatrix& F2 = Farr[ik2];
				const std::vector< vector3<> >& v2 = vArr[ik2];
				//Loops over bands and phonon modes:
				for(int b1=0; b1<nBands; b1++) if(fabs(E1[b1])<EplasmonMax)
				{	double prefacLbdaInv = prefacTauInv / v1[b1].length();
					for(int b2=0; b2<nBands; b2++) if(fabs(E2[b2])<EplasmonMax)
					{	for(int alpha=0; alpha<nModes; alpha++)
						{	for(int ae=-1; ae<=+1; ae+=2) // +/- for phonon absorption or emmision
							{	//Energy conservation factor:
								double EconserveExponent = EconserveExpFac * std::pow(E2[b2] - E1[b1] - ae*omegaPh[alpha], 2);
								if(EconserveExponent < -10.) continue; //irrelevant event
								double delta = EconservePrefac * exp(EconserveExponent);
								//Scattering event properties:
								double nPh = 1./(exp(omegaPh[alpha]/T) - 1.);
								double occFactors = (nPh+0.5 - ae*(0.5-F2[b2]));
								double gePhSq = gePh[alpha](b2,b1).norm(); //norm=abs^2
								double weight = prefacLbdaInv * delta * occFactors * gePhSq;
								//Histogram:
								double deltaEcur = ae*omegaPh[alpha];
								double cosThetaCur = dot(v1[b1],v2[b2]) / sqrt(v1[b1].length_squared() * v2[b2].length_squared());
								deltaE.addEvent(E1[b1], deltaEcur, weight);
								cosTheta.addEvent(E1[b1], cosThetaCur, weight);
								lbdaInv.addEvent(E1[b1], weight);
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
	
	dos.allReduce(MPIUtil::ReduceSum);
	lbdaInv.allReduce(MPIUtil::ReduceSum);
	deltaE.allReduce(MPIUtil::ReduceSum);
	cosTheta.allReduce(MPIUtil::ReduceSum);
	logPrintf("done.\n"); logFlush();
	
	//Normalize histograms:
	for(int e=0; e<deltaE.nE; e++)
	{	double scale2d = 1./lbdaInv.out[e]; //scale the 2D histograms by the total weight in each energy bin
		for(int o=0; o<deltaE.nomega; o++)
		{	deltaE.out[o*deltaE.nE+e] *= scale2d;
			cosTheta.out[o*deltaE.nE+e] *= scale2d;
		}
		lbdaInv.out[e] /= dos.out[e]; //scale lbdaInv to be per particle (from per energy bin)
	}
	
	//Output histograms:
	dos.print("ePhDos.dat", 1./eV, eV); //dos in eV^-1
	lbdaInv.print("ePhLbdaInv.dat", 1./eV, 1e-9*meter); //lbdaInv in nm^-1
	deltaE.print("ePhDeltaE.dat", 1./eV, 1./eV, eV); //deltaE probability distribution in eV^-1
	cosTheta.print("ePhCosTheta.dat", 1./eV, 1., 1.); //cosTheta probability distribution is dimensionless
	
	finalizeSystem();
}