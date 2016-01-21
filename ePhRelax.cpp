#include "Units.h"
#include "InputMap.h"
#include "Interp1.h"
#include "Histogram.h"
#include <core/Util.h>
#include <core/Operators.h>
#include <electronic/matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <fstream>

struct ePhRelax
{
	Interp1 dos, dosPh;
	diagMatrix f0, fPert; //Initial Fermi and photon-perturbed distributions
	double T, dos0; //initial temperature and density of states at the Fermi level
	double scaledDe; //De scaled by g(eF)**-3
	diagMatrix hInt; //energy resolved electron-phonon coupling
	
	//Energy grid:
	int nE; double Emin, dE;
	inline double Egrid(int i) const { return dos.xGrid[i]; }
	int ieMin, ieMax; //min and max active energy grid indices (that evolve with time)
	int ieStart, ieStop; //min and max energy grid indices to deal with on current MPI process
	
	ePhRelax(int argc, char** argv)
	{
		//Parse the command line:
		string inputFilename; bool dryRun, printDefaults;
		initSystemCmdline(argc, argv, "Electron-phonon relaxation using Boltzmann equation", inputFilename, dryRun, printDefaults);

		//Get the system parameters (mu, T, lattice vectors etc.)
		InputMap inputMap(inputFilename);	
		const double Z = inputMap.get("Z"); //number of electrons per unit cell
		T = inputMap.get("T") * Kelvin; //initial temperature in Kelvin (electron and lattice)
		const double Uabs = inputMap.get("Uabs") * Joule/std::pow(meter,3); //absorbed laser energy per unit volume in Joule/meter^3
		const double Eplasmon = inputMap.get("Eplasmon") * eV; //incident photon energy in eV
		const double De = inputMap.get("De") / eV; //quadratic e-e lifetime coefficient in eV^-1
		const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);
		double detR = fabs(det(R));
		
		logPrintf("\nInputs after conversion to atomic units:\n");
		logPrintf("Z = %lg\n", Z);
		logPrintf("T = %lg\n", T);
		logPrintf("Uabs = %lg\n", Uabs);
		logPrintf("Eplasmon = %lg\n", Eplasmon);
		logPrintf("De = %lg\n", De);
		logPrintf("R:\n");
		R.print(globalLog, " %lg ");
		if(dryRun)
		{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
			finalizeSystem();
			exit(0);
		}
		logPrintf("\n");
		
		//Read electron and phonon DOS (and convert to atomic units and per-unit volume):
		dos.init("dos.dat", eV, 1./(detR*eV));
		dosPh.init("phononDOS.dat", eV, 1./(detR*eV));
		nE = dos.xGrid.size();
		dE = dos.dx;
		
		//Determine initial Fermi distribution:
		f0.resize(nE);
		//--- Bisect for chemical potential:
		double dmuMin = dos.xGrid.front() - 10*T;
		double dmuMax = dos.xGrid.back() + 10*T;
		double dmu = 0.5*(dmuMin + dmuMax);
		const double tol = 1e-9*T;
		while(dmuMax-dmuMin > tol)
		{	//calculate number of electrons at current Z:
			double nElectrons = 0.;
			for(int ie=0; ie<nE; ie++)
			{	double& fi = f0[ie];
				fi = fermi((Egrid(ie) - dmu)/T);
				nElectrons += dE * dos.yGrid[0][ie] * fi * detR;
			}
			((nElectrons>Z) ? dmuMax : dmuMin) = dmu;
			dmu = 0.5*(dmuMin + dmuMax);
		}
		logPrintf("Initial Fermi distribution: dmu = %le eV\n", dmu/eV);
		//--- calculate density of states at the Fermi level:
		dos0 = 0.;
		for(int ie=0; ie<nE; ie++)
			dos0 += dE * dos.yGrid[0][ie] * fermiPrime((Egrid(ie) - dmu)/T) * (-1./T);
		logPrintf("Density of states at Fermi level = %le /eV-cell\n", dos0*(eV*detR));
		scaledDe = De / std::pow(dos0,3);
		
		//Perturb by photon-induced carrier density:
		//--- read carrier distributions from plasmonDecay:
		Histogram2D distribDirect("carrierDistribAll-direct.dat", 1./eV, 1./eV, 1.);
		Histogram2D distribPhonon("carrierDistribAll-phonon.dat", 1./eV, 1./eV, 1.);
		if(Eplasmon < distribDirect.omegaMin || Eplasmon > distribDirect.omegaMin + (distribDirect.nomega-1)*distribDirect.domega)
			die("Plasmon energy is out of the range available in carrierDistribAll-direct.dat")
		if(Eplasmon < distribPhonon.omegaMin || Eplasmon > distribPhonon.omegaMin + (distribPhonon.nomega-1)*distribPhonon.domega)
			die("Plasmon energy is out of the range available in carrierDistribAll-phonon.dat")
		//--- interpolate to required photon energy and carrier eenergy grid:
		fPert.resize(nE);
		double Upert = 0.;
		for(int ie=0; ie<nE; ie++)
		{	const double& Ei = Egrid(ie);
			double dni = distribDirect.interp1(Ei, Eplasmon) + distribPhonon.interp1(Ei, Eplasmon); //induced carrier number change at given energy
			fPert[ie] = dni / std::max(dos.yGrid[0][ie], 1e-3*dos0); //divide by DOS to get the effective filling change (regularize to avoid Infs)
			Upert += dni * Ei * dE; //calculate energy of perturbation
		}
		fPert *= Uabs / Upert; //normalize to match absorbed laser energy per unit volume
		fPert += f0; //add initial Fermi distribution
		
		//Electron-phonon coupling:
		Interp1 hIntInterp; hIntInterp.init("hInt.dat", eV, eV/pow(Angstrom,3));
		//--- interpolate to all the interval midpoints of energy grid:
		hInt.resize(nE-1);
		for(int ie=0; ie<nE-1; ie++)
			hInt[ie] = hIntInterp(Egrid(ie)+0.5*dE);
		
		//Determine active energy grid:
		ieMin = std::max(0, int(floor((-Eplasmon-10*T-dos.xMin)/dE)));
		ieMax = std::min(nE, int(ceil((Eplasmon+10*T-dos.xMin)/dE)));
		int neActive = ieMax - ieMin;
		TaskDivision(neActive, mpiUtil).myRange(ieStart, ieStop);
		ieStart += ieMin;
		ieStop += ieMin;
		logPrintf("Active energy grid: [%d,%d) of total %d points, with [%d,%d) on current process.\n", ieMin, ieMax, nE, ieStart, ieStop);
	}
	
	//Evaluate e-e and e-Ph collision integrals (nonlinear):
	diagMatrix fdot(const diagMatrix& f) const
	{	diagMatrix results(nE+1); //last entry is TlDot
		double& TlDot = results.back();
		const double Tl = f.back();
		const double* g = dos.yGrid[0].data(); //DOS data pointer
		//e-e collisions:
		for(int i=ieStart; i<ieStop; i++)
		{	double rateSum = 0.;
			for(int i1=ieMin; i1<ieMax; i1++)
			{	double inOcc = f[i]*f[i1];
				double inUnocc = (1.-f[i])*(1.-f[i1]);
				//i2 range set by both i2 and i3 in [ieMin,ieMax)
				int i2min = std::max(i+i1+1-ieMax, ieMin);
				int i2max = std::min(i+i1+1-ieMin, ieMax);
				for(int i2=i2min; i2<i2max; i2++)
				{	int i3 = i+i1-i2; //energy conservation
					double outOcc = f[i2]*f[i3];
					double outUnocc = (1.-f[i2])*(1.-f[i3]);
					rateSum += (inUnocc*outOcc - inOcc*outUnocc) * g[i1]*g[i2]*g[i3];
				}
			}
			results[i] = (2*scaledDe) * (dE*dE) * rateSum;
		}
		results.allReduce(MPIUtil::ReduceSum, true);
		//e-ph collisions:
		double ElDot = 0.; //rate of energy transfer to lattice
		for(int i=0; i<nE-1; i++)
		{	if(std::max(g[i-1],g[i]) < 1e-3*dos0) continue; //ignore intervals with no electrons to avoid division by zero below
			double fPrime = (f[i+1]-f[i])/dE;
			double fMean = 0.5*(f[i+1]+f[i]);
			double ElDot_i = (2*M_PI*dE) * hInt[i] * (fMean*(1.-fMean) + fPrime*Tl); //rate of energy transfer to lattice from this interval
			double nDot = -ElDot_i / dE; //number of electrons that move up in energy due to energy transfer to lattice
			results[i] += nDot / g[i];
			results[i-1] -= nDot / g[i-1];
			ElDot += ElDot_i;
		}
		TlDot = ElDot / Cl(Tl);
		return results;
	}
	
	//Calculate lattice specific heat
	inline double Cl(double Tl) const
	{	assert(dosPh.xMin==0.);
		const double& domegaPh = dosPh.dx;
		double result = 0.;
		for(size_t ie=1; ie<dosPh.xGrid.size(); ie++) //omit zero energy phonons to avoid 0/0 error
		{	double omegaPh = ie*domegaPh;
			double x = omegaPh/Tl;
			double g = 1./(exp(x)-1.);
			double g_Tl = g*(g+1)*x/Tl; //dg/dTl
			result += domegaPh * omegaPh * g_Tl  * dosPh.yGrid[0][ie];
		}
		return result;
	}
	
	inline double fermi(double x) { return x>30. ? exp(-x) : 1./(1.+exp(x)); } //avoid overflow issues
	inline double fermiPrime(double x) { return 0.25*(std::pow(tanh(0.5*x), 2) - 1.); } //avoid overflow issues
};

//Wrapper function for GSL integrator:
int fdot_wrapper(double t, const double* f, double* fdot, void* params)
{	const ePhRelax& e = *((ePhRelax*)params);
	diagMatrix fMat; fMat.assign(f, f+e.nE+1); //copy input to diagMatrix
	diagMatrix fdotMat = e.fdot(fMat); //calculate result in diagMatrix form
	fdotMat.bcast(); //make sure results are consistent across processes to numerical precision
	std::copy(fdotMat.begin(), fdotMat.end(), fdot); //copy output to pointer
	return GSL_SUCCESS;
}

int main(int argc, char** argv)
{	ePhRelax e(argc, argv);
	
	//Solve time dependence:
	StopWatch watchSolve("Solve"); watchSolve.start();
	gsl_odeiv2_system odeSystem = {fdot_wrapper, NULL, size_t(e.nE), &e };
	gsl_odeiv2_driver* odeDriver = gsl_odeiv2_driver_alloc_y_new(&odeSystem, gsl_odeiv2_step_msadams, 1e-6, 1e-6, 0.0);
	double tMax = 1000.*fs, dt = 50.*fs;
	int itInterval = std::max(1, int(round((tMax/dt)/50.))); //interval for reporting progress
	double t = 0.;
	diagMatrix f = e.fPert; f.push_back(e.T);
	std::vector<diagMatrix> fArr;
	fArr.push_back(f);
	logPrintf("\nSolving boltzmann eqn: "); logFlush();
	while(t < tMax)
	{	int status = gsl_odeiv2_driver_apply(odeDriver, &t, t+dt, f.data());
		if(status != GSL_SUCCESS) die("Error %d in ODE propagation", status)
		fArr.push_back(f);
		
		//Print progress:
		if((fArr.size()+1) % itInterval == 0)
		{	logPrintf("%d%% ", int(round(100.*t/tMax)));
			logFlush();
		}
	}
	logPrintf("done.\n"); logFlush();
	gsl_odeiv2_driver_free(odeDriver);
	watchSolve.stop();
	
	//File outputs:
	if(mpiUtil->isHead())
	{	std::ofstream ofs;
		
		//Pulse shape and effective T:
// 		ofs.open((e.runName+".pulseShape").c_str());
// 		ofs.precision(10);
// 		ofs << "#t[fs] pulseShape[fs^-1] Teff[K]\n";
// 		for(size_t it=0; it<Teff.size(); it++)
// 			ofs << (it*dt)/fs << '\t' << pulseShape[it]*fs << '\t' << Teff[it]/Kelvin << '\n';
// 		ofs.close();
		
		//Lattice temperature:
		ofs.open("temp.Tl");
		ofs.precision(10);
		ofs << "#t[fs] Tl[K]\n";
		for(size_t it=0; it<fArr.size(); it++)
			ofs << (it*dt)/fs << '\t' << fArr[it].back()/Kelvin << '\n';
		ofs.close();
		
		//Distributions [dimensionless]
		//ofs.open((e.runName+".f").c_str());
		ofs.open("temp.f");
		ofs.precision(10);
		//--- Header
		ofs << "#E[ev]\\t[fs]";
		for(size_t it=0; it<fArr.size(); it++)
			ofs << '\t' << (it*dt)/fs;
		ofs << '\n';
		//--- Data
		for(int ie=0; ie<e.nE; ie++)
		{	ofs << e.Egrid(ie)/eV;
			for(size_t it=0; it<fArr.size(); it++)
				ofs << '\t' << fArr[it][ie];
			ofs << '\n';
		}
		ofs.close();
		
		//Linewidth corrections [eV]
// 		ofs.open((e.runName+".lwDelta").c_str());
// 		ofs.precision(10);
// 		//--- Header
// 		ofs << "#E[ev]\\t[fs]";
// 		for(size_t it=0; it<Teff.size(); it++)
// 			ofs << '\t' << (it*dt)/fs;
// 		ofs << '\n';
// 		//--- Data
// 		for(int ie=0; ie<e.nE; ie++)
// 		{	ofs << e.Egrid(ie)/eV;
// 			for(size_t it=0; it<Teff.size(); it++)
// 				ofs << '\t' << lwDelta[it][ie]/eV;
// 			ofs << '\n';
// 		}
// 		ofs.close();
	}
	
	finalizeSystem();
};
