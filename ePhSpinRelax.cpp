/*-------------------------------------------------------------------
Copyright 2018 Ravishankar Sundararaman

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

#include "FeynWann.h"
#include "InputMap.h"
#include "SparseMatrix.h"
#include <core/Units.h>
#include <core/Random.h>
#include <algorithm>

struct SpinRelaxCollect
{	const std::vector<double>& dmu; //doping levels
	const double T; //temperature
	const double omegaPhMinByT; //lower cutoff in phonon frequency / T
	const int nModes; //number of phonon modes to include in calculation (override if any, applied below already)
	
	const double EconserveExpFac; //exponential factor in Gaussian delta for energy conservation
	const double prefacGamma, prefacChi; //prefactors for numerator and denominator of T1
	
	std::vector<matrix3<>> Gamma, chi; //numerator and denominator in T1^-1, for each dmu
	
	SpinRelaxCollect(const std::vector<double>& dmu, double T, double omegaPhMin, int nModes, double EconserveWidth, size_t nKpairs)
	: dmu(dmu), T(T), omegaPhMinByT(std::max(1e-3,omegaPhMin/T)), nModes(nModes),
		EconserveExpFac(-0.5/std::pow(EconserveWidth, 2)),
		prefacGamma(2*M_PI/ (nKpairs * sqrt(2.*M_PI)*EconserveWidth)), //include prefactor of Gaussian energy conservation
		prefacChi(0.5/nKpairs), //collected over both k in each k-pair for consistency
		Gamma(dmu.size()), chi(dmu.size())
	{
	}
	
	inline SparseMatrix degenerateProject(const matrix& M, const diagMatrix& E)
	{	static const double degeneracyThreshold = 1e-6;
		const int nBands = E.nRows();
		SparseMatrix result; result.reserve(nBands); //typically diagonal (Rashba) or block diagonal with size 2 (Kramer-degenerate)
		const complex* Mdata = M.data();
		for(int b2=0; b2<nBands; b2++)
		for(int b1=0; b1<nBands; b1++)
		{	if(fabs(E[b1] - E[b2]) < degeneracyThreshold)
			{	SparseEntry sr;
				sr.i = b1;
				sr.j = b2;
				sr.val = *(Mdata);
				result.push_back(sr);
			}
			Mdata++;
		}
		return result;
	}
	
	void process(const FeynWann::MatrixEph& mEph)
	{	const FeynWann::StateE& e1 = *(mEph.e1);
		const FeynWann::StateE& e2 = *(mEph.e2);
		const FeynWann::StatePh& ph = *(mEph.ph);
		const int nBands = e1.E.nRows();
		const double invT = 1./T;
		
		//Degenerate spin projections:
		std::vector<SparseMatrix> Sdeg1(3), Sdeg2(3);
		for(int iDir=0; iDir<3; iDir++)
		{	Sdeg1[iDir] = degenerateProject(e1.S[iDir], e1.E);
			Sdeg2[iDir] = degenerateProject(e2.S[iDir], e2.E);
		}
		
		//Compute chi contributions by band except for electron occupation factors:
		#define CONTRIB_chi(s) \
			std::vector<matrix3<>> contribChi##s(nBands); \
			for(int iDir=0; iDir<3; iDir++) \
			for(int jDir=0; jDir<3; jDir++) \
			{	diagMatrix SiSj = diagSS(Sdeg##s[iDir], Sdeg##s[jDir], nBands); \
				for(int b=0; b<nBands; b++) \
					contribChi##s[b](iDir,jDir) = prefacChi * SiSj[b]; \
			}
		CONTRIB_chi(1)
		CONTRIB_chi(2)
		#undef CONTRIB_chi
		
		//Compute Gamma contributions by band pair except for electron occupation factors:
		std::vector<matrix3<>> contribGamma(nBands*nBands);
		for(int alpha=0; alpha<nModes; alpha++)
		{	//Phonon occupation:
			const double& omegaPh = ph.omega[alpha];
			const double omegaPhByT = invT*omegaPh;
			if(omegaPhByT < omegaPhMinByT) continue; //avoid 0./0. below
			const double prefac_nPhByT =  prefacGamma * invT * bose(omegaPhByT);
			//Energy conservation factor and prefactor (including nPh/T):
			std::vector<double> prefacEconserve(nBands*nBands);
			int bIndex = 0;
			bool contrib = false;
			for(int b2=0; b2<nBands; b2++)
			for(int b1=0; b1<nBands; b1++)
			{	double expTerm = EconserveExpFac * std::pow(e1.E[b1] - e2.E[b2] - omegaPh, 2);
				if(expTerm > -15.)
				{	prefacEconserve[bIndex] = prefac_nPhByT * exp(expTerm); //compute exponential only when needed
					contrib = true;
				}
				bIndex++;
			}
			if(not contrib) continue; //no energy conserving combination for this phonon mode at present k-pair
			//Commutator contributions:
			const matrix& G = mEph.M[alpha];
			matrix GSdeg2[3];
			for(int jDir=0; jDir<3; jDir++)
				GSdeg2[jDir] = G * Sdeg2[jDir];
			for(int iDir=0; iDir<3; iDir++)
			{	matrix Sdeg1Gi = Sdeg1[iDir] * G; 
				for(int jDir=0; jDir<3; jDir++)
				{	const matrix SGcomm = Sdeg1Gi - GSdeg2[jDir];
					const complex* SGcommData = SGcomm.data();
					for(int bIndex=0; bIndex<nBands*nBands; bIndex++) //loop over b2 and b1
						contribGamma[bIndex](iDir,jDir) += prefacEconserve[bIndex] * SGcommData[bIndex].norm();
				}
			}
		}
		
		//Collect results for various dmu values:
		for(unsigned iMu=0; iMu<dmu.size(); iMu++)
		{	//Compute Fermi occupations and accumulate chi contributions:
			#define CALC_F_ACCUM_CHI(s) \
				diagMatrix F##s(nBands); \
				for(int b=0; b<nBands; b++) \
				{	double f = fermi(invT*(e##s.E[b] - dmu[iMu])); \
					F##s[b] = f; \
					chi[iMu] += (invT * f*(1.-f)) * contribChi##s[b]; \
				}
			CALC_F_ACCUM_CHI(1)
			CALC_F_ACCUM_CHI(2)
			#undef CALC_F_ACCUM_CHI
			
			//Accumulate Gamma contributions:
			int bIndex = 0;
			for(int b2=0; b2<nBands; b2++)
			for(int b1=0; b1<nBands; b1++)
			{	Gamma[iMu] += contribGamma[bIndex] * (F2[b2] * (1 - F1[b1]));
				bIndex++;
			}
		}
	}
	static void ePhProcess(const FeynWann::MatrixEph& mEph, void* params)
	{	((SpinRelaxCollect*)params)->process(mEph);
	}
};


int main(int argc, char** argv)
{	InitParams ip = FeynWann::initialize(argc, argv, "Electron-phonon scattering contribution to spin relaxation.");

	//Read input file:
	InputMap inputMap(ip.inputFilename);
	const int nOffsets = inputMap.get("nOffsets"); assert(nOffsets>0);
	const int nBlocks = inputMap.get("nBlocks"); assert(nBlocks>0);
	const double T = inputMap.get("T") * Kelvin;
	const double EconserveWidth = inputMap.get("EconserveWidth") * eV;
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional shift in chemical potential from neutral value; start of range (default to 0)
	const double dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional shift in chemical potential from neutral value; end of range (default to 0)
	const int dmuCount = inputMap.get("dmuCount", 1); assert(dmuCount>0); //number of chemical potential shifts
	const double omegaPhMin = inputMap.get("omegaPhMin", 0.0) * eV; //lower cutoff in phonon frequency
	const int nModesOverride = inputMap.get("nModesOverride", 0); //if non-zero, use only these many lowest phonon modes (eg. set to 3 for acoustic only in 3D)
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nOffsets = %d\n", nOffsets);
	logPrintf("nBlocks = %d\n", nBlocks);
	logPrintf("T = %lg\n", T);
	logPrintf("EconserveWidth = %lg\n", EconserveWidth);
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("dmuCount = %d\n", dmuCount);
	logPrintf("omegaPhMin = %lg\n", omegaPhMin);
	logPrintf("nModesOverride = %d\n", nModesOverride);
	
	//Initialize FeynWann:
	FeynWannParams fwp;
	fwp.needSymmetries = true;
	fwp.needPhonons = true;
	fwp.needSpin = true;
	FeynWann fw(fwp);

	//dmu array:
	std::vector<double> dmu(dmuCount, dmuMin); //set first value here
	for(int iMu=1; iMu<dmuCount; iMu++) //set remaining values (if any)
		dmu[iMu] = dmuMin + iMu*(dmuMax-dmuMin)/(dmuCount-1);
	int nModes = nModesOverride ? std::min(nModesOverride, fw.nModes) : fw.nModes;
	
	//Initialize sampling parameters:
	int nOffsetsPerBlock = ceildiv(nOffsets, nBlocks);
	size_t nKpairsPerBlock = fw.ePhCountPerOffset() * nOffsetsPerBlock;
	logPrintf("Effectively sampled nKpairs: %lu\n", nKpairsPerBlock * nBlocks);
	int oStart = 0, oStop = 0;
	if(mpiGroup->isHead())
		TaskDivision(nOffsetsPerBlock, mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop-oStart; //number of offsets (per block) handled by current group
	int oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress

	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");

	//Collect integrals involved in T1 calculation:
	std::vector<std::shared_ptr<SpinRelaxCollect>> srcArr(nBlocks);
	for(int block=0; block<nBlocks; block++)
	{	logPrintf("Working on block %d of %d: ", block+1, nBlocks); logFlush();
		srcArr[block] = std::make_shared<SpinRelaxCollect>(dmu, T, omegaPhMin, nModes, EconserveWidth, nKpairsPerBlock);
		SpinRelaxCollect& src = *(srcArr[block]);
		for(int o=0; o<noMine; o++)
		{	Random::seed(block*nOffsetsPerBlock+o+oStart); //to make results independent of MPI division
			//Process with a random offset pair:
			vector3<> k01 = fw.randomVector(mpiGroup); //must be constant across group
			vector3<> k02 = fw.randomVector(mpiGroup); //must be constant across group
			fw.ePhLoop(k01, k02, SpinRelaxCollect::ePhProcess, &src);
			//Print progress:
			if((o+1)%oInterval==0) { logPrintf("%d%% ", int(round((o+1)*100./noMine))); logFlush(); }
		}
		//Accumulate over MPI:
		mpiWorld->allReduceData(src.Gamma, MPIUtil::ReduceSum);
		mpiWorld->allReduceData(src.chi, MPIUtil::ReduceSum);
		logPrintf("done.\n"); logFlush();
	}
	
	//Report results with statistics:
	const double ps = 1e3*fs; //picosecond
	for(int iMu=0; iMu<dmuCount; iMu++)
	{	logPrintf("\nResults for dmu = %lg eV:\n", dmu[iMu]/eV);
		std::vector<matrix3<>> Gamma(nBlocks), chi(nBlocks), T1bar(nBlocks);
		std::vector<double> T1(nBlocks);
		for(int block=0; block<nBlocks; block++)
		{	SpinRelaxCollect& src = *(srcArr[block]);
			fw.symmetrize(src.Gamma[iMu]);
			fw.symmetrize(src.chi[iMu]);
			Gamma[block] = src.Gamma[iMu];
			chi[block] = src.chi[iMu];
			T1bar[block] = chi[block] * inv(Gamma[block]);
			T1[block] = (1./3)*trace(T1bar[block]);
		}
		reportResult(Gamma, "Gamma", 1./(eV*ps), "1/(eV.ps)");
		reportResult(chi, "chi", 1./eV, "1/eV");
		reportResult(T1bar, "T1", ps, "ps"); //tensor version
		reportResult(T1, "T1", ps, "ps"); //tensor version
	}
	
	fw.free();
	FeynWann::finalize();
	return 0;
}
