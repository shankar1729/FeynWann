/*-------------------------------------------------------------------
Copyright 2019 Ravishankar Sundararaman

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
#include <core/tensor3.h>
#include <core/Random.h>
#include <core/string.h>
#include "FeynWann.h"
#include "Histogram.h"
#include "InputMap.h"
#include <core/Units.h>

//Levi-civita operators:
template<typename scalar> matrix3<scalar> epsDot(const vector3<scalar>& v)
{	matrix3<scalar> M;
	M(2,1) = -(M(1,2) = v[0]);
	M(0,2) = -(M(2,0) = v[1]);
	M(1,0) = -(M(0,1) = v[2]);
	return M;
}
template<typename scalar> vector3<scalar> epsDot(const matrix3<scalar>& M)
{	return vector3<scalar>(
		M(1,2) - M(2,1),
		M(2,0) - M(0,2),
		M(0,1) - M(1,0) );
}
template<typename scalar> matrix3<scalar> Sym(const matrix3<scalar>& M)
{	return scalar(0.5)*(M + (~M));
}
inline matrix3<> Real(const matrix3<complex>& M)
{	matrix3<> ret;
	for(int i=0; i<3; i++)
		for(int j=0; j<3; j++)
			ret(i,j) = M(i,j).real();
	return ret;
}

inline vector3<complex> getVectorElement(const matrix M[3], int b1, int b2)
{	vector3<complex> result;
	for(int iDir=0; iDir<3; iDir++)
		result[iDir] = M[iDir](b1, b2);
	return result;
}

inline tensor3<complex> getTensorElement(const matrix M[5], int b1, int b2)
{	tensor3<complex> result;
	for(int iComp=0; iComp<5; iComp++)
		result[iComp] = M[iComp](b1, b2);
	return result;
}

//Collect circular dichroism contibutions using FeynWann callbacks:
struct CollectCD
{	double dmu, T, invT;
	double domega, omegaMax;
	bool spinAvailable;
	std::vector<Histogram> CD, CDmd; //Total circular dichorism and magnetic momentum contributions alone (xx,yy,zz,yz,zx,xy components)
	std::vector<Histogram> CDspin; //Spin contributions, if present
	double prefac;
	
	CollectCD(double dmu, double T, double domega, double omegaMax, bool spinAvailable)
	: dmu(dmu), T(T), invT(1./T), domega(domega), omegaMax(omegaMax), spinAvailable(spinAvailable),
		CD(6, Histogram(0, domega, omegaMax)),
		CDmd(6, Histogram(0, domega, omegaMax))
	{	logPrintf("Initialized frequency grid: 0 to %lg eV with %d points.\n", CD[0].Emax()/eV, CD[0].nE);
		if(spinAvailable) CDspin.assign(6, Histogram(0, domega, omegaMax));
	}
	
	void collectE(const FeynWann::StateE& state)
	{	int nBands = state.E.nRows();
		matrix3<> Id(1.,1.,1.); //3x3 identity
		//Calculate Fermi fillings and linewidths:
		const diagMatrix& E = state.E;
		diagMatrix F(nBands);
		for(int b=0; b<nBands; b++)
			F[b] = fermi((state.E[b]-dmu)*invT);
		//Collect 
		for(int b2=0; b2<nBands; b2++)
		{	for(int b1=0; b1<nBands; b1++)
			{	double omega = E[b1] - E[b2]; //energy conservation
				if(omega<domega || omega>=omegaMax) continue; //irrelevant event
				if(fabs(F[b1] - F[b2]) < 1E-6) continue; //negligible weight below
				//Collect relevant matrix elements:
				vector3<complex> Pconj = getVectorElement(state.v, b2, b1); //b1 <-> b2 to get conjugate (P is Hermitian)
				vector3<complex> L = getVectorElement(state.L, b1, b2), S2;
				if(spinAvailable) S2 = getVectorElement(state.S, b1, b2); //2*S
				matrix3<complex> Q(getTensorElement(state.Q, b1, b2));
				//Compute contributions:
				matrix3<> Feq = Sym(Real(epsDot(Pconj) * Q));
				matrix3<> Fmd = Id*dot(Pconj, L).real() - Sym(Real(outer(Pconj, L))), Fspin;
				if(spinAvailable) Fspin = Id*dot(Pconj, S2).real() - Sym(Real(outer(Pconj, S2)));
				matrix3<> Ftot = Feq + Fmd + Fspin;
				double weight = prefac * (F[b1] - F[b2]);
				//Save contribution to appropriate frequency:
				int iOmega; double tOmega; //coordinates of frequency on frequency grid
				bool useEvent = CD[0].eventPrecalc(omega, iOmega, tOmega); //all histograms on same frequency grid
				if(useEvent)
				{
					#define addEventTensor(H, G) \
						H[0].addEventPrecalc(iOmega, tOmega, weight*G(0,0)); \
						H[1].addEventPrecalc(iOmega, tOmega, weight*G(1,1)); \
						H[2].addEventPrecalc(iOmega, tOmega, weight*G(2,2)); \
						H[3].addEventPrecalc(iOmega, tOmega, weight*G(1,2)); \
						H[4].addEventPrecalc(iOmega, tOmega, weight*G(2,0)); \
						H[5].addEventPrecalc(iOmega, tOmega, weight*G(0,1));
					addEventTensor(CD, Ftot);
					addEventTensor(CDmd, Fmd);
					if(spinAvailable) { addEventTensor(CDspin, Fspin); }
					#undef addEventTensor
				}
			}
		}
	}
	static void collect(const FeynWann::StateE& state, void* params)
	{	((CollectCD*)params)->collectE(state);
	}
	
	void allReduce()
	{	for(Histogram& h: CD) h.allReduce(MPIUtil::ReduceSum);
		for(Histogram& h: CDmd) h.allReduce(MPIUtil::ReduceSum);
		if(spinAvailable) for(Histogram& h: CDspin) h.allReduce(MPIUtil::ReduceSum);
	}
	
	void saveTensor(const std::vector<Histogram>& hArr, string fname, const FeynWann& fw)
	{	if(mpiWorld->isHead())
		{	ofstream ofs(fname.c_str());
			//Header:
			ofs << "#omega[eV]";
			const char* comps[6] = { "xx", "yy", "zz", "yz", "zx", "xy" };
			for(int iComp=0; iComp<6; iComp++)
				ofs << " dAlpha_" << comps[iComp] << "[cm^-1]";
			ofs << "\n";
			//Result for each frequency in a row:
			for(size_t iOmega=0; iOmega<hArr[0].out.size(); iOmega++)
			{	double omega = hArr[0].Emin + hArr[0].dE * iOmega;
				ofs << omega/eV;
				//Collect and symmetrize tensor:
				matrix3<> M;
				M(0,0) = hArr[0].out[iOmega];
				M(1,1) = hArr[1].out[iOmega];
				M(2,2) = hArr[2].out[iOmega];
				M(1,2) = (M(2,1) = hArr[3].out[iOmega]);
				M(2,0) = (M(0,2) = hArr[4].out[iOmega]);
				M(0,1) = (M(1,0) = hArr[5].out[iOmega]);
				fw.symmetrize(M);
				//Switch units:
				M *= (1e8*Angstrom); //switch from atomic units to cm^-1
				//Write components:
				ofs << '\t' << M(0,0) << '\t' << M(1,1) << '\t' << M(2,2)
					<< '\t' << M(1,2) << '\t' << M(2,0) << '\t' << M(0,1) << '\n';
			}
		}
	}
	void save(const FeynWann& fw)
	{	saveTensor(CD, "CD.dat", fw);
		saveTensor(CDmd, "CDmd.dat", fw);
		if(spinAvailable) saveTensor(CDspin, "CDspin.dat", fw);
	}
};

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Wannier calculation of circular dichroism");

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const int nOffsets = inputMap.get("nOffsets"); assert(nOffsets>0);
	const double omegaMax = inputMap.get("omegaMax") * eV; assert(omegaMax>0.); //maximum photon frequency to collect results for
	const double domega = inputMap.get("domega") * eV; assert(domega>0.); //photon energy grid resolution
	const double T = inputMap.get("T") * Kelvin;
	const double dmu = inputMap.get("dmu", 0.) * eV; //optional shift in chemical potential from neutral value/ VBM; (default to 0)
	FeynWannParams fwp(&inputMap);

	//Check contribution:
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("omegaMax = %lg\n", omegaMax);
	logPrintf("domega = %lg\n", domega);
	logPrintf("T = %lg\n", T);
	logPrintf("dmu = %lg\n", dmu);
	fwp.printParams();
	
	//Initialize FeynWann:
	fwp.needSymmetries = true;
	fwp.needVelocity = true;
	fwp.needQ = true;
	fwp.needL = true;
	fwp.needSpin = true; //for spin contribution, if available
	std::shared_ptr<FeynWann> fw = std::make_shared<FeynWann>(fwp);
	size_t nKeff = nOffsets * fw->eCountPerOffset();
	logPrintf("Effectively sampled nKpts: %lu\n", nKeff);
	if(mpiWorld->isHead()) logPrintf("%d electron k-mesh offsets parallelized over %d process groups.\n", nOffsets, mpiGroupHead->nProcesses());

	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw = 0;
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
	
	//Collect results:
	CollectCD ccd(dmu, T, domega, omegaMax, fwp.needSpin);
	const double c = 137.035999084; //speed of light in atomic units = 1/(fine structure constant)
	ccd.prefac = 4.*std::pow(M_PI/c,2) * fw->spinWeight / (nKeff*fabs(det(fw->R))); //frequency independent part of prefactor
	
	for(int iSpin=0; iSpin<fw->nSpins; iSpin++)
	{	//Update FeynWann for spin channel if necessary:
		if(iSpin>0)
		{	fw = 0; //free memory from previous spin
			fwp.iSpin = iSpin;
			fw = std::make_shared<FeynWann>(fwp);
		}
		logPrintf("\nCollecting CD spectrum: "); logFlush();
		for(int o=0; o<noMine; o++)
		{	Random::seed(o+oStart); //to make results independent of MPI division
			vector3<> k0 = fw->randomVector(mpiGroup); //must be constant across group
			fw->eLoop(k0, CollectCD::collect, &ccd);
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
	}
	ccd.allReduce();
	logPrintf("done.\n"); logFlush();
	
	//Output results:
	ccd.save(*fw);
	
	fw = 0;
	FeynWann::finalize();
	return 0;
}
