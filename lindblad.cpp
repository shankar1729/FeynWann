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

//Lindblad initialization, time evolution and measurement operators using FeynWann callback
struct Lindblad
{	
	const FeynWann& fw;
	std::vector<matrix> rho; //current density matrices (indexed by offset and ik)
	//TODO rho0
	
	Lindblad(const FeynWann& fw)
	: fw(fw)
	{
	}
	
	
	//Stage 0: initialize eLoop: set rho0 (TODO)
	
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
	//--- pump
	const double pumpOmega = inputMap.get("pumpOmega") * eV; //pump frequency in eV
	const double pumpA0 = inputMap.get("pumpA0"); //pump pulse amplitude / intensity (Units TBD)
	const double pumpTau = inputMap.get("pumpTau")*fs; //Gaussian pump pulse width in fs
	const vector3<complex> pumpPol = normalize(
		complex(1,0)*inputMap.getVector("pumpPolRe", vector3<>(1.,0.,0.)) +  //Real part of polarization
		complex(0,1)*inputMap.getVector("pumpPolIm", vector3<>(0.,0.,0.)) ); //Imag part of polarization
	//--- probes
	const double omegaMin = inputMap.get("omegaMin", 0.) * eV; //optional start of frequency grid for probe response (default: 0)
	double omegaMax = inputMap.get("omegaMax", 0.) * eV; //optional end of frequency grid for probe response (default 0 => max available DeltaE)
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
	int noMine = oStop-oStart; //number of offsets handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
	
	//Initialize frequency grid:
	EnergyRange er = { DBL_MAX, -DBL_MAX };
	fw.eLoop(vector3<>(), EnergyRange::eProcess, &er);
	mpiWorld->allReduce(er.Emin, MPIUtil::ReduceMin);
	mpiWorld->allReduce(er.Emax, MPIUtil::ReduceMax);
	if(!omegaMax) omegaMax = er.Emax - er.Emin;

	/*
	//Calculate delta-function resolved versions (no broadening yet):
	CollectImEps cie(tau, A0, domega, omegaFull, omegaMax);
	cie.prefac = 2. * M_PI * fw->spinWeight / (std::pow(A0,2)*nKeff*fabs(det(fw->R))); //frequency independent part of prefactor
	cie.Ehat = Ehat;
	
	for(int iSpin=0; iSpin<fw->nSpins; iSpin++)
	{	//Update FeynWann for spin channel if necessary:
		if(iSpin>0)
		{	fw = 0; //free memory from previous spin
			fwp.iSpin = iSpin;
			fw = std::make_shared<FeynWann>(fwp);
		}
		logPrintf("\nCollecting ImEps: "); logFlush();
		for(int o=0; o<noMine; o++)
		{	Random::seed(o+oStart); //to make results independent of MPI division
			//Process with a random offset:
			vector3<> k0 = fw->randomVector(mpiGroup); //must be constant across group
			fw->eLoop(k0, CollectImEps::direct, &cie);
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
	}
	//current 
	cie.ImEps.allReduce(MPIUtil::ReduceSum);
	int iomegaStart, iomegaStop; TaskDivision(nomega, mpiWorld).myRange(iomegaStart, iomegaStop);
	*/
}
