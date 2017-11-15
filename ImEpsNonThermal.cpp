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
#include "Interp1.h"

//Lorentzian kernel for an odd function stored on postive frequencies alone:
inline double lorentzianOdd(double omega, double omega0, double breadth)
{       double breadthSq = std::pow(breadth,2);
        return (breadth/M_PI) *
                ( 1./(breadthSq + std::pow(omega-omega0, 2))
                - 1./(breadthSq + std::pow(omega+omega0, 2)) );
}


inline void writeImEps(const char* fname, const std::vector<Histogram>& ImEps, const std::vector<string>& headerVals)
{	std::ofstream ofs(fname);
	//Header:
	ofs << "#omega[eV]";
	for(const string& headerVal: headerVals)
		ofs << ' ' << headerVal;
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
	long nKpts = inputMap.get("nKpts");
	const double Z = inputMap.get("Z"); //number of electrons per unit cell
	const double dE = inputMap.get("dE") * eV; //energy resolution used for output and energy conservation
	string runName = inputMap.getString("runName");
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKpts = %ld\n", nKpts);
	logPrintf("Z = %lg\n", Z);
	logPrintf("dE = %lg\n", dE);
	
	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(1); //assume cubic symmetry and only calculate x-axis
	Ahat[0] = vector3<complex>(1., 0., 0.);
	BandStruct bs("Wannier/totalE", "Wannier/wannier", true, Ahat);
	const int bunchSize = 32;
	bs.setCacheSize(2*bunchSize);

	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");
	
	Interp1 fInterp, lwInterp;
	fInterp.init(runName + ".f", eV, 1.);
	lwInterp.init(runName + ".lwDelta", eV, eV);
	int numTimes = fInterp.headerVals.size();
	
	//Read lattice temperatures:
	Interp1 TlInterp; TlInterp.init(runName + ".Tl", fs, Kelvin);
	const std::vector<double>& Tl = TlInterp.yGrid[0];
	assert(int(Tl.size()) == numTimes);
		
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
	int steps = (Emax-Emin)/dE;
	logPrintf("Initialized energy grid: %lg to %lg eV with %d points.\n", Emin/eV, (Emin+dE*(steps))/eV, steps);
	
	//Initialize sampling parameters:
	int ikStart, ikStop; TaskDivision(nKpts, mpiWorld).myRange(ikStart, ikStop);
	int nBunchesMine = ceil((ikStop-ikStart)*1./bunchSize); //number of bunches on current process
	int iBunchInterval = std::max(1, int(round(nBunchesMine/50.))); //interval for reporting progress
	nKpts = nBunchesMine * bunchSize; mpiWorld->allReduce(nKpts, MPIUtil::ReduceSum); //total number of sampled k-points
	long nKpairs = nKpts * (bunchSize-1); //total number of sampled k-point pairs for phonon-assisted transitions
	int nBands = Egamma.nRows();
	int nModes = bs.getPhononModes(vector3<>()).nRows();
	double phononPrefac0 = 4 * std::pow(M_PI,2) * bs.spinWeight / (nKpairs*fabs(det(bs.R))); //frequency independent part of prefac
	double directPrefac0 = 4 * std::pow(M_PI,2) * bs.spinWeight / (nKpts*fabs(det(bs.R))); //frequency independent part of prefac
	double Tl0lw = 0.026*eV; //Tl at which e-ph linewidths were calculated

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
	//double omegaMax = 77.210*eV;
	double omegaMax = 19.92*eV;

	//Initialize unbroadened histograms:
	std::vector<Histogram> ImEpsDirect(numTimes, Histogram(0, dE, omegaMax)), breadthDirect(numTimes, Histogram(0, dE, omegaMax));
	std::vector<Histogram> ImEpsPhonon(numTimes, Histogram(0, dE, omegaMax)), breadthPhonon(numTimes, Histogram(0, dE, omegaMax)),  weightPhonon(numTimes, Histogram(0, dE, omegaMax));
	int nomega = ImEpsDirect[0].out.size();
	logPrintf("Initialized frequency grid: 0 to %lg eV with %d points.\n", (dE*(nomega-1))/eV, nomega);
	
	//-------- Calculate dielectric response ------------------------------------------------
	logPrintf("\nePhCoupling and ImEps: "); logFlush();
	for(int iBunch=0; iBunch<nBunchesMine; iBunch++)
	{
		//Retrieve k-point bunch:
		const std::vector< vector3<> >& kArr = kArrArr[iBunch];
		
		//Calculate electronic states and matrix elements for bunch:
		std::vector<diagMatrix> Earr = bs.getStates(kArr);
		std::vector<diagMatrix> ImEarr_ee = lineWidth(kArr, 1., 0.);
		std::vector<diagMatrix> ImEarr_ePh = lineWidth(kArr, 0., 1.);
		//account for linear Tl dependence of e-ph linewidths
		std::vector< std::vector<matrix> > Parr = bs.getDipoleMatElem(kArr);
		std::vector< std::vector<diagMatrix> > Farr(bunchSize), ImEarr(bunchSize); //fillings and linewidths by k-point, temperature and band
		for(int ik=0; ik<bunchSize; ik++)
		{	Farr[ik].assign(numTimes, diagMatrix(nBands));
			ImEarr[ik].assign(numTimes, diagMatrix(nBands));
			for(int iT=0; iT<numTimes; iT++)
			{	double TlRatio = Tl[iT] / Tl0lw;
				for(int b=0; b<nBands; b++)
				{	double E = Earr[ik][b];
					Farr[ik][iT][b] = fInterp(iT,E); //interpolate electron occupations (not necessarily a Fermi distribution)
					ImEarr[ik][iT][b] = ImEarr_ee[ik][b] //e-e contribution at low temperature
						+ ImEarr_ePh[ik][b] * TlRatio //e-ph contribution with linear Tl dependence
						+ lwInterp(iT,E); //add linewidth correction (interpolated as a function of carrier energy)
				}
			}
		}

		diagMatrix omegaPhArr[bunchSize];
		std::vector<matrix> gePhArr[bunchSize];
		for(int ik1=0; ik1<bunchSize; ik1++)
		{	const diagMatrix& E1 = Earr[ik1];
			const std::vector<diagMatrix>& ImE1 = ImEarr[ik1];
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
						breadthDirect[iT].addEvent(omega, weight * (F1[iT][v] - F1[iT][c]) * (ImE1[iT][c]+ImE1[iT][v]));
					}	
				}
			}
			
			//Calculate phonon stuff for each pair of k-points involving ik1
			bs.setPhononMatElemArray(kArr[ik1], kArr, gePhArr);
			for(int ik2=0; ik2<bunchSize; ik2++)
				omegaPhArr[ik2] = bs.getPhononModes(kArr[ik1] - kArr[ik2]);

			for(int ik2=0; ik2<bunchSize; ik2++) if(ik2 != ik1)
			{	const diagMatrix& E2 = Earr[ik2];
				const std::vector<diagMatrix>& ImE2 = ImEarr[ik2];
				const std::vector<diagMatrix>& F2 = Farr[ik2];
				const matrix& P2 = Parr[ik2][0];
			
				for(int alpha=0; alpha<nModes; alpha++)
				{	const matrix& gePh = gePhArr[ik2][alpha];
					double omegaPh = omegaPhArr[ik2][alpha];
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
							//Include T dependent electron occupations:
							for(int iT=0; iT<numTimes; iT++)
							{	double nPh = 1./(exp(omegaPh/Tl[iT]) - 1.);
								double weight = (phononPrefac0/(omega*omega)) * (nPh + 0.5*(1.-ae)) * MeffSqExtrap;
								ImEpsPhonon[iT].addEvent(omega, weight * (F1[iT][v] - F2[iT][c]));
								breadthPhonon[iT].addEvent(omega, fabs(weight * (F1[iT][v] - F2[iT][c]))*(ImE2[iT][c]+ImE1[iT][v]));
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
	int iomegaStart, iomegaStop; TaskDivision(nomega, mpiWorld).myRange(iomegaStart, iomegaStop);
	logPrintf("Applying broadening ... "); logFlush();
	for(int iT=0; iT<numTimes; iT++)
	{	for(int iomega=iomegaStart; iomega<iomegaStop; iomega++) //input frequency grid split over MPI
		{	double omegaCur = iomega*dE;
			double bDirect = breadthDirect[iT].out[iomega]; //non-thermal carrier distribution corrections already included above
			double bPhonon = breadthPhonon[iT].out[iomega]; //non-thermal carrier distribution corrections already included above
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

	if(mpiWorld->isHead())
	{	//Print calculated ImEps contributions:
		writeImEps("ImEps_directNontherm.dat", ImEpsDirectBroad, fInterp.headerVals);
		writeImEps("ImEps_phononNontherm.dat", ImEpsPhononBroad, fInterp.headerVals);
		
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

		//Print Linewidth correction at each time:
		ofstream ofs("LWcorrection.dat");
		ofs << "#LinewidthCorrection[eV]\n";
		for(int iT=0; iT<numTimes; iT++)
		{	ofs << lwInterp(iT,0)/eV << '\n';
		}

	}
	
	finalizeSystem();
}
