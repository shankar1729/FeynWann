/*-------------------------------------------------------------------
Copyright 2019 Ravishankar Sundararaman, Adela Habib

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------*/

#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include "FeynWann.h"
#include "Histogram.h"
#include "InputMap.h"
#include <core/Units.h>

inline matrix dot(const matrix* P, vector3<complex> pol)
{	return pol[0]*P[0] + pol[1]*P[1] + pol[2]*P[2];
}

//Lindblad initialization, time evolution and measurement operators using FeynWann callback
struct Lindblad
{	
	FeynWann& fw;
	const std::vector<vector3<>>& k0; //!< k-point offsets
	const int oStart, oStop, noMine; //!< range of offsets handled by this process group
	const int ikStart, ikStop, nkMine; //!< range of k-points within offset handled by this process

	int o; //!< current offset being worked on by this process group (used as an outer loop variable)
	std::vector<matrix> rho; //!< current density matrices (indexed by offset and ik)
	inline int index(int ik) { return ik-ikStart + nkMine*(o-oStart); } //!< index to density matrix at ik and current o
	
	const double dmu, T, invT; //!< Fermi level position relative to neutral value / VBM, and temperature
	const double pumpOmega, pumpA0, pumpTau; const vector3<complex> pumpPol; //!< pump parameters
	const double omegaMin, domega; const int nomega; //!< probe frequency grid
	const double tau; const std::vector<vector3<complex>> pol; //!< probe parameters
	
	Lindblad(FeynWann& fw, const std::vector<vector3<>>& k0, int oStart, int oStop,
		double dmu, double T, double pumpOmega, double pumpA0, double pumpTau, vector3<complex> pumpPol,
		double omegaMin, double omegaMax, double domega, double tau, std::vector<vector3<complex>> pol)
	: fw(fw), k0(k0), oStart(oStart), oStop(oStop), noMine(oStop-oStart),
		ikStart(fw.Hw->ikStart), ikStop(ikStart+fw.Hw->nk), nkMine(ikStop-ikStart),
		rho(noMine * nkMine), dmu(dmu), T(T), invT(1./T),
		pumpOmega(pumpOmega), pumpA0(pumpA0), pumpTau(pumpTau), pumpPol(pumpPol),
		omegaMin(omegaMin), domega(domega), nomega(1+int(round((omegaMax-omegaMin)/domega))),
		tau(tau), pol(pol)
	{
		
	}
	
	//--------- Initialize -------------
	
	//Cache required properties per state
	struct State
	{	diagMatrix E; //energy eigenvalues (i.e. H0)
		std::vector<matrix> P; //P matrix elements for each probe polarization (energy conservation delta (D) not included)
		matrix pumpPD; //P matrix elements at pump polarization x energy conservation delta (D), but without A0 and time factor
	};
	std::vector<State> state;
	
	inline void initializeE(const FeynWann::StateE& stateE)
	{	
		//Identify destination for results:
		int rhoIndex = index(stateE.ik);
		State& s = state[rhoIndex];
		
		//Cache required properties to state:
		//--- Energies
		s.E = stateE.E;
		//--- Probe matrix elements (without energy conservation)
		s.P.clear(); s.P.reserve(pol.size());
		for(const vector3<complex>& pol_i: pol)
			s.P.push_back(dot(stateE.v, pol_i));
		//--- Pump matrix elements
		s.pumpPD = dot(stateE.v, pumpPol);
		double normFac = sqrt(pumpTau/sqrt(M_PI));
		complex* PDdata = s.pumpPD.data();
		for(int b2=0; b2<fw.nBands; b2++)
			for(int b1=0; b1<fw.nBands; b1++)
			{	//Multiply energy conservation:
				double tauDeltaE = tau*(s.E[b1] - s.E[b2] - pumpOmega);
				*(PDdata++) *= normFac * exp(-0.5*tauDeltaE*tauDeltaE);
			}
		
		//Set rho to initial occupations:
		diagMatrix rho0(fw.nBands);
		for(int b=0; b<fw.nBands; b++)
		{	double expArg = (s.E[b]-dmu)*invT;
			rho0[b] = (expArg < -30.) ? 1.
				: ((expArg > +30.) ? 0.
				: 1./(1.+exp(expArg)) );
		}
		rho[rhoIndex] = rho0;
	}
	static void initializeE(const FeynWann::StateE& stateE, void* params)
	{	((Lindblad*)params)->initializeE(stateE);
	}
	
	void initialize()
	{	state.resize(rho.size());
		//Initialize eLoop:
		int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
		logPrintf("\nInitializing electronic quantities: "); logFlush();
		for(int o=oStart; o<oStop; o++)
		{	fw.eLoop(k0[o], Lindblad::initializeE, this);
			//Print progress:
			if((o-oStart+1)%oInterval==0) { logPrintf("%d%% ", int(round((o-oStart+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
	}
	
	
	//Stage 1: FGR pump eLoop (optional): rho0 += rho2 (TODO)
	
	//Stage 2: time evolution operator eLoop  (TODO)
	
	//Stage 3: time evolution operator ePhLoop (TODO)
	
	//DeltaRho = P * RHo) * +h.c.
		//ImEps += Tr(DeltaRho)
	/*
	//---- Direct transitions ----
	void collectDirect(const FeynWann::StateE& state)
	{	int nBands = state.E.nRows();
		const diagMatrix& E = state.E;
		//rho0 
		matrix rho0 = zeros(nBands, nBands);
		for(int v=0; v<nV; v++)
			rho0 += 1;
		//Project dipole matrix elements on field:
		matrix P;
		for(int iDir=0; iDir<3; iDir++)
			P += A0 * Ehat[iDir] * state.v[iDir];
		
		matrix deltaRho(nBands,nBands) = zeros(nBands, nBands);
		diagMatrix Id(nBands) = eye(nBands);
		
			//Collect 
		for(int v=0; v<nBands; v++) //if(E[v]<EvMax)
		{	for(int c=0; c<nBands; c++) //if(E[c]>EcMin)
			{	double deltaE = E[c] - E[v] - omega; //energy conservation
				double deltaPrefac = tau/std::sqrt(M_PI);
				double eConserv = std::sqrt(deltaPrefac*exp(-std::pow(deltaE*tau, 2)));
				P += eConserv;
			}
						
		}
		for (int s=0; s<2; s++)
		{	//TODO deltarho
			deltaRho += M_PI*((Id-rho0)*P*rho0*daggar(P) - dagger(P)*(Id-rho0)*P*rho0);
			P = daggerP;					
		}
	}
	
	static void direct(const FeynWann::StateE& state, void* params)
	{	((CollectImEps*)params)->collectDirect(state);
	}
	*/
};

inline void print(FILE* fp, const vector3<complex>& v, const char* format="%lg ")
{	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].real()); std::fprintf(fp, "] + 1j*");
	std::fprintf(fp, "[ "); for(int k=0; k<3; k++) fprintf(fp, format, v[k].imag()); std::fprintf(fp, "]\n");
}
inline vector3<complex> normalize(const vector3<complex>& v) { return v * (1./sqrt(v[0].norm() + v[1].norm() + v[2].norm())); }

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Lindblad dynamics in an ab initio Wannier basis");

	//Get the system parameters:
	InputMap inputMap(ip.inputFilename);
	//--- kpoints
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	vector3<int> NkMult;
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	//--- doping / temperature
	const double dmu = inputMap.get("dmu", 0.) * eV; //optional: shift in fermi level from neutral value / VBM in eV (default: 0)
	const double T = inputMap.get("T") * Kelvin; //temperature in Kelvin (ambient phonon T = initial electron T)
	//--- pump
	const double pumpOmega = inputMap.get("pumpOmega") * eV; //pump frequency in eV
	const double pumpA0 = inputMap.get("pumpA0"); //pump pulse amplitude / intensity (Units TBD)
	const double pumpTau = inputMap.get("pumpTau")*fs; //Gaussian pump pulse width in fs
	const vector3<complex> pumpPol = normalize(
		complex(1,0)*inputMap.getVector("pumpPolRe", vector3<>(1.,0.,0.)) +  //Real part of polarization
		complex(0,1)*inputMap.getVector("pumpPolIm", vector3<>(0.,0.,0.)) ); //Imag part of polarization
	//--- probes
	const double omegaMin = inputMap.get("omegaMin") * eV; //start of frequency grid for probe response
	const double omegaMax = inputMap.get("omegaMax") * eV; //end of frequency grid for probe response
	const double domega = inputMap.get("domega") * eV; //frequency resolution for probe calculation
	const double tau = inputMap.get("tau") * fs; //Gaussian probe pulse width in fs
	std::vector<vector3<complex>> pol;
	while(true)
	{	int iPol = int(pol.size())+1;
		ostringstream oss; oss << iPol;
		string polName = oss.str();
		vector3<> polRe = inputMap.getVector("polRe"+polName, vector3<>(0.,0.,0.)); //Real part of polarization
		vector3<> polIm = inputMap.getVector("polIm"+polName, vector3<>(0.,0.,0.)); //Imag part of polarization
		if(polRe.length_squared() + polIm.length_squared() == 0.) break; //End of probe polarizations
		pol.push_back(normalize(complex(1,0)*polRe + complex(0,1)*polIm));
	}

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	logPrintf("dmu = %lg\n", dmu);
	logPrintf("T = %lg\n", T);
	logPrintf("pumpOmega = %lg\n", pumpOmega);
	logPrintf("pumpA0 = %lg\n", pumpA0);
	logPrintf("pumpTau = %lg\n", pumpTau);
	logPrintf("pumpPol = "); print(globalLog, pumpPol);
	logPrintf("omegaMin = %lg\n", omegaMin);
	logPrintf("omegaMax = %lg\n", omegaMax);
	logPrintf("domega = %lg\n", domega);
	logPrintf("tau = %lg\n", tau);
	for(int iPol=0; iPol<int(pol.size()); iPol++)
	{	logPrintf("pol%d = ", iPol+1);
		print(globalLog, pol[iPol]);
	}

	//Initialize FeynWann:
	FeynWannParams fwp;
	fwp.needVelocity = true;
	fwp.needSpin = true;
	FeynWann fw(fwp);
	
	//Construct mesh of k-offsets:
	std::vector<vector3<>> k0;
	vector3<int> NkFine;
	for(int iDir=0; iDir<3; iDir++)
	{	if(fw.isTruncated[iDir] && NkMult[iDir]!=1)
		{	logPrintf("Setting NkMult = 1 along truncated direction %d.\n", iDir+1);
			NkMult[iDir] = 1; //no multiplication in truncated directions
		}
		NkFine[iDir] = fw.kfold[iDir] * NkMult[iDir];
	}
	matrix3<> NkFineInv = inv(Diag(vector3<>(NkFine)));
	vector3<int> ikMult;
	for(ikMult[0]=0; ikMult[0]<NkMult[0]; ikMult[0]++)
	for(ikMult[1]=0; ikMult[1]<NkMult[1]; ikMult[1]++)
	for(ikMult[2]=0; ikMult[2]<NkMult[2]; ikMult[2]++)
		k0.push_back(NkFineInv * ikMult);
	logPrintf("Effective interpolated k-mesh dimensions: ");
	NkFine.print(globalLog, " %d ");
	size_t nOffsets = k0.size();
	size_t nKeff = nOffsets * fw.eCountPerOffset();
	logPrintf("Effectively sampled %s: %lu\n", "nKpts", nKeff);
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Initialize sampling parameters:
	int oStart=0, oStop=0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsets, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	
	//Create and initialize lindblad calculator:
	Lindblad lb(fw, k0, oStart, oStop, dmu, T,
		pumpOmega, pumpA0, pumpTau, pumpPol,
		omegaMin, omegaMax, domega, tau, pol);
	lb.initialize();
	
	//Cleanup:
	fw.free();
	FeynWann::finalize();
	return 0;
}
