#include <core/Units.h>
#include <core/Util.h>
#include <core/Operators.h>
#include <core/matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <fstream>

struct eeRelax
{
	//Inputs and constants:
	double Ephoton, Tmax, De;
	double T0;
	string runName;
	
	//Energy grid:
	double Emin, dE, Emax;
	int nEhlf, nE;
	int ieStart, ieStop; //energy grid MPI division
	inline double Egrid(int i) const { return Emin + i*dE; }
	
	//Initial distributions:
	diagMatrix f0, fPert; //Initial Fermi and photon-perturbed distributions
	double pulseEnergy;
	
	eeRelax(int argc, char** argv)
	{	if(argc<3)
			die("Usage: eeRelax <Ephoton> <Tmax> [De=0.016]\n"
				"Ephoton is the energy of the photon in eV.\n"
				"Tmax in Kelvins is the asymptotic max temperature that the electrons should reach\n"
				"De in 1/eV units is the quadratic lifetime prefactor, with the default value set to the ab initio fits for gold.\n");
		
		//Parse inputs:
		Ephoton = atof(argv[1])*eV;
		Tmax = atof(argv[2])*Kelvin;
		De = (argc<4 ? 0.016 : atof(argv[3]))/eV;
		logPrintf("Using De = %lg eV^{-1}\n", De/(1./eV));
		runName = string("eeRelax-") + argv[1] + string("-") + argv[2];
		T0 = 300*Kelvin;
		
		//Energy grid:
		double Eouter = Ephoton + 10*T0;
		nEhlf = ceil(Eouter/T0);
		nE = 2*nEhlf+1;
		dE = 2.*Eouter/nE;
		Emin = -Eouter+0.5*dE;
		Emax = Emin + (nE-1)*dE;
		logPrintf("Energy grid: Emin = %lg eV to Emax = %lg eV with dE = %lg eV and nE = %d\n", Emin/eV, Emax/eV, dE/eV, nE);
		TaskDivision(nE, mpiWorld).myRange(ieStart, ieStop);
		
		//Initial Fermi distribution:
		f0.resize(nE);
		for(int i=0; i<nE; i++)
			f0[i] = 1./(1.+exp(Egrid(i)/T0));
		
		//Photon perturbation:
		pulseEnergy = (std::pow(M_PI,2)/6.)*(std::pow(Tmax,2) - std::pow(T0,2));
		fPert.resize(nE);
		for(int i=0; i<nE; i++)
		{	double E = Egrid(i);
			fPert[i] = f0[i] + (pulseEnergy/std::pow(Ephoton,2)) * (
				1./((1. + exp(-E/T0)) * (1. + exp((E-Ephoton)/T0))) -
				1./((1. + exp(E/T0)) * (1. + exp((-E-Ephoton)/T0)))
			);
		}
	}
	
	//Estimate temperature based on df/dE at the Fermi level:
	double calcT(diagMatrix f)
	{	return 3*dE/(8.*(f[nEhlf-1]-f[nEhlf+1]) - (f[nEhlf-2]-f[nEhlf+2]));
	}

	//Evaluate e-e collision integral (nonlinear):
	diagMatrix fdot(diagMatrix f) const
	{	diagMatrix results(nE);
		for(int i=ieStart; i<ieStop; i++)
		{	double rateSum = 0.;
			for(int i1=0; i1<nE; i1++)
			{	double inOcc = f[i]*f[i1];
				double inUnocc = (1.-f[i])*(1.-f[i1]);
				//i2 range set by both i2 and i3 in [0,nE)
				int i2min = std::max(i+i1+1-nE, 0);
				int i2max = std::min(i+i1+1, nE);
				for(int i2=i2min; i2<i2max; i2++)
				{	int i3 = i+i1-i2; //energy conservation
					double outOcc = f[i2]*f[i3];
					double outUnocc = (1.-f[i2])*(1.-f[i3]);
					rateSum += inUnocc*outOcc - inOcc*outUnocc;
				}
			}
			results[i] = (2*De) * (dE*dE) * rateSum;
		}
		mpiWorld->allReduceData(results, MPIUtil::ReduceSum, true);
		return results;
	}
	
	//Evaluate e-e collision integral (nonlinear):
	diagMatrix linewidthCorrection(diagMatrix f) const
	{	diagMatrix results(nE);
		//Padded fillings array:
		int nEbig = 3*nE;
		diagMatrix fBig(nEbig);
		for(int i=0; i<nE; i++)
		{	fBig[i] = 1.;
			fBig[i+nE] = f[i];
			fBig[i+2*nE] = 0.;
		}
		for(int i=nE+ieStart; i<nE+ieStop; i++)
		{	double rateSum = 0.;
			for(int i1=0; i1<nEbig; i1++)
			{	//i2 range set by both i2 and i3 in [0,nE)
				int i2min = std::max(i+i1+1-nEbig, 0);
				int i2max = std::min(i+i1+1, nEbig);
				for(int i2=i2min; i2<i2max; i2++)
				{	int i3 = i+i1-i2; //energy conservation
					rateSum += (1.-fBig[i1])*fBig[i2]*fBig[i3] + fBig[i1]*(1.-fBig[i2])*(1.-fBig[i3]);
				}
			}
			results[i-nE] = (De*dE*dE)*rateSum - 0.5*De*std::pow(Egrid(i-nE),2);
		};
		mpiWorld->allReduceData(results, MPIUtil::ReduceSum, true);
		return results;
	}
};

//Wrapper function for GSL integrator:
int fdot_wrapper(double t, const double* f, double* fdot, void* params)
{	const eeRelax& ee = *((eeRelax*)params);
	diagMatrix fMat; fMat.assign(f, f+ee.nE); //copy input to diagMatrix
	diagMatrix fdotMat = ee.fdot(fMat); //calculate result in diagMatrix form
	std::copy(fdotMat.begin(), fdotMat.end(), fdot); //copy output to pointer
	return GSL_SUCCESS;
}

int main(int argc, char** argv)
{
	initSystem(argc, argv);
	eeRelax ee(argc, argv);
	
	//Solve time dependence:
	StopWatch watchSolve("Solve"); watchSolve.start();
	gsl_odeiv2_system odeSystem = {fdot_wrapper, NULL, size_t(ee.nE), &ee };
	gsl_odeiv2_driver* odeDriver = gsl_odeiv2_driver_alloc_y_new(&odeSystem, gsl_odeiv2_step_msadams, 1e-6, 1e-6, 0.0);
	double tMax = 10000.*fs, dt = 50.*fs;
	double t = 0.;
	diagMatrix f = ee.fPert;
	std::vector<diagMatrix> fArr;
	fArr.push_back(f);
	while(t < tMax)
	{	int status = gsl_odeiv2_driver_apply(odeDriver, &t, t+dt, f.data());
		if(status != GSL_SUCCESS) die("Error %d in ODE propagation", status)
		fArr.push_back(f);
	}
	gsl_odeiv2_driver_free(odeDriver);
	watchSolve.stop();
	
	//Calculate carrier linewidths and effective temperature:
	StopWatch watchLinewidths("Linewidths"); watchLinewidths.start();
	std::vector<diagMatrix> lwDelta;
	std::vector<double> Teff;
	for(diagMatrix& f: fArr)
	{	lwDelta.push_back(ee.linewidthCorrection(f));
		Teff.push_back(ee.calcT(f));
	}
	watchLinewidths.stop();

	//Calculate effective pulse shape:
	std::vector<double> pulseShape;
	for(size_t it=0; it<Teff.size(); it++)
	{	size_t itPrev = it ? it-1 : it;
		size_t itNext = it+1<Teff.size() ? it+1 : it;
		double Tdot = (Teff[itNext] - Teff[itPrev]) / (dt*(itNext - itPrev));
		pulseShape.push_back( 2.*Teff[it]*Tdot/(std::pow(ee.Tmax,2) - std::pow(ee.T0,2)) ); //normalized to unit pulse energy
	}
	
	//File outputs:
	if(mpiWorld->isHead())
	{	//Pulse shape and effective T:
		std::ofstream ofs((ee.runName+".pulseShape").c_str());
		ofs.precision(10);
		ofs << "#t[fs] pulseShape[fs^-1] Teff[K]\n";
		for(size_t it=0; it<Teff.size(); it++)
			ofs << (it*dt)/fs << '\t' << pulseShape[it]*fs << '\t' << Teff[it]/Kelvin << '\n';
		ofs.close();
		
		//Distributions [dimensionless]
		ofs.open((ee.runName+".f").c_str());
		ofs.precision(10);
		//--- Header
		ofs << "#E[ev]\\t[fs]";
		for(size_t it=0; it<Teff.size(); it++)
			ofs << '\t' << (it*dt)/fs;
		ofs << '\n';
		//--- Data
		for(int ie=0; ie<ee.nE; ie++)
		{	ofs << ee.Egrid(ie)/eV;
			for(size_t it=0; it<Teff.size(); it++)
				ofs << '\t' << fArr[it][ie];
			ofs << '\n';
		}
		ofs.close();
		
		//Linewidth corrections [eV]
		ofs.open((ee.runName+".lwDelta").c_str());
		ofs.precision(10);
		//--- Header
		ofs << "#E[ev]\\t[fs]";
		for(size_t it=0; it<Teff.size(); it++)
			ofs << '\t' << (it*dt)/fs;
		ofs << '\n';
		//--- Data
		for(int ie=0; ie<ee.nE; ie++)
		{	ofs << ee.Egrid(ie)/eV;
			for(size_t it=0; it<Teff.size(); it++)
				ofs << '\t' << lwDelta[it][ie]/eV;
			ofs << '\n';
		}
		ofs.close();
	}
	
	finalizeSystem();
	return 0.;
}
