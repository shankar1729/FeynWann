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
#include <core/Units.h>
#include <core/LatticeUtils.h>
#include <electronic/TetrahedralDOS.h>
#include <algorithm>
#include <gsl/gsl_cblas.h>

template<typename T> T prod(const vector3<T>& v) { return v[0]*v[1]*v[2]; }

complex c0(0., 0.);
complex c1(1., 0.);
complex cm1(-1., 0.);

struct T1params
{
	unsigned nSpinTime; //number of spin relaxation/dephasing times; currently, only T1 is computed and it must be 1
	double T;
	double dmu;
	double EconserveWidth;
	double EconserveExpFac, EconservePrefac; //energy conserving Gaussian exponential and pre-factor
	// phDOS(omega close to 0) should be proportional to omega^2
	// but due to numeric errors or the existence of maginary phonon frequencies, it is not alway true, so we can suppress tiny phonon frequencies
	double omegaStart; // in unit of Kelvin
	vector3<> kshift;
	bool acousticonly;

	T1params(const FeynWann& fw, InputMap &map){
		nSpinTime = fw.isRelativistic() ? 1 : 0;
		if (nSpinTime == 0) throw std::invalid_argument("is not relativistic");
		T = map.get("T", 300.) * Kelvin;
		dmu = map.get("dmu", 0.0) * eV;
		EconserveWidth = map.get("EconserveWidth", 0.01) * eV;
		EconserveExpFac = -0.5 / std::pow(EconserveWidth, 2); EconservePrefac = 1. / (sqrt(2.*M_PI)*EconserveWidth); // two sqrt of Gaussian
		omegaStart = map.get("omegaStart", 0.0) * Kelvin;
		kshift = map.getVector("kshift", vector3<>(0.5, 0.5, 0.5));
		acousticonly = map.get("acousticonly", false);
		logPrintf("\nInputs after conversion to atomic units:\n");
		logPrintf("T = %g\n", T);
		logPrintf("dmu= %lg\n", dmu);
		logPrintf("EconserveWidth = %lg\n", EconserveWidth);
		logPrintf("omegaStart = %g\n", omegaStart);
		logPrintf("kshift = %12.8lf %12.8lf %12.8lf\n", kshift[0], kshift[1], kshift[2]);
		logPrintf("acousticonly = %d\n", acousticonly);
	}
} *T1p;

struct CollectEph
{
	const FeynWann& fw;
	int nBands, nk, nOffsets1, nOffsets2, o1, o2;
	std::vector<double> wko1, wko2;
	std::vector<std::vector<diagMatrix>> E; //electron energies on k2 meshes
	std::vector<vector3<>> kmesh; //DFT k-point mesh (full version i.e. unreduced)
	double T1, NS, prefacT1kn; // actually T1 inverse; prefacT1kn is 2pi/hbar*EconservePrefac/prod(kFine)
	std::vector<std::vector<diagMatrix>> T1kn; // actually T1 inverse, nOffsets2*nkfold*nBands
	double ***SzdegSq;
	bool **b_skip; // storing which k2 should be skipped due to tiny dfde
	int **band0_used, **nband_used; // not used
	double dfdemax, thr_skip, thr_deg; // thrhold for skipping a particular k2 where dfde are all tiny, set in CollectEph

	CollectEph(const FeynWann& fw, InputMap &map, const vector3<int> NkMult[], const int nOffsets[], const std::vector<double> wko[])
		: fw(fw), nBands(fw.nBands), nk(prod(fw.kfold)),
		nOffsets1(nOffsets[0]), nOffsets2(nOffsets[1]),
		wko1(nOffsets1), wko2(nOffsets2),
		E(nOffsets2, std::vector<diagMatrix>(nk)), kmesh(nk),
		T1kn(nOffsets2, std::vector<diagMatrix>(nk, diagMatrix(nBands)))
	{
		for (int io = 0; io < nOffsets1; io++) { wko1[io] = wko[0][io]; }
		for (int io = 0; io < nOffsets2; io++) { wko2[io] = wko[1][io]; }
		T1 = 0.0; NS=0.0;
		prefacT1kn = 2 * M_PI / nk / prod(NkMult[0]) * T1p->EconservePrefac; // including Gaussian prefactor
		b_skip = new bool*[nOffsets2]; for (int io = 0; io < nOffsets2; io++) { b_skip[io] = new bool[nk]; }
		band0_used = new int*[nOffsets2]; for (int io = 0; io < nOffsets2; io++) { band0_used[io] = new int[nk]; }
		nband_used = new int*[nOffsets2]; for (int io = 0; io < nOffsets2; io++) { nband_used[io] = new int[nk]; }
		dfdemax = 0.; thr_skip = 1e-8; thr_deg = 1e-6;
		SzdegSq = new double**[nOffsets2]; 
		for (int io = 0; io < nOffsets2; io++){
			SzdegSq[io] = new double*[nk];
			for (int ik = 0; ik < nk; ik++)
				SzdegSq[io][ik] = new double[nBands];
		}
	}

	//---- Collect energies and kmesh ----
	static void collectE(const FeynWann::StateE& state, void* params)
	{
		CollectEph& cEph = *((CollectEph*)params);
		cEph.E[cEph.o2][state.ik] = state.E;
		if (state.E.size() > 0)
		for (int b1 = 0; b1 < cEph.nBands; b1++){
			double dtmp = cEph.FD(state.E[b1])*(1. - cEph.FD(state.E[b1]));
			cEph.SzdegSq[cEph.o2][state.ik][b1] = 0;
			for (int b2 = 0; b2 < cEph.nBands; b2++){
				if (fabs(state.E[b1] - state.E[b2]) < cEph.thr_deg)
					cEph.SzdegSq[cEph.o2][state.ik][b1] += state.S[2](b1, b2).norm();
			}
			cEph.NS += cEph.SzdegSq[cEph.o2][state.ik][b1] * dtmp * cEph.wko2[cEph.o2];
			if (dtmp > cEph.dfdemax)
				cEph.dfdemax = dtmp;
		}
		if (cEph.o2 == 0)
			cEph.kmesh[state.ik] = state.k;
	}
	
	void get_skip()
	{
		for (int io = 0; io < nOffsets2; io++)
		for (int ik = 0; ik < nk; ik++){
			b_skip[io][ik] = true;
			band0_used[io][ik] = nBands;
			nband_used[io][ik] = 0;
			for (int b1 = 0; b1 < nBands; b1++)
			if (FD(E[io][ik][b1])*(1 - FD(E[io][ik][b1])) / dfdemax > thr_skip){
				b_skip[io][ik] = false;
				band0_used[io][ik] = b1;
				nband_used[io][ik]++;
				for (int b2 = b1 + 1; b2 < nBands; b2++)
				if (FD(E[io][ik][b1])*(1 - FD(E[io][ik][b1])) / dfdemax > thr_skip)
					nband_used[io][ik]++;
				else
					break;
				break;
			}
		}
	}
	
	inline matrix degenerateProject(const matrix& in, const diagMatrix& E)
	{	matrix out = in;
		complex* outData = out.data();
		for(int b2=0; b2<nBands; b2++)
		for(int b1=0; b1<nBands; b1++)
		{	if(fabs(E[b1] - E[b2]) > thr_deg)
				*(outData) = 0.; //outside degenerate subspace
			outData++;
		}
		return out;
	}
	
	void process(const FeynWann::MatrixEph& mEph)
	{
		const FeynWann::StateE& e1 = *(mEph.e1);
		const FeynWann::StateE& e2 = *(mEph.e2);
		const FeynWann::StatePh& ph = *(mEph.ph);
		const int nModes = ph.omega.nRows();
		
		//Calculate electron fillings:
		diagMatrix f(nBands), f2(nBands);
		for(int b=0; b<nBands; b++)
		{	f[b] = FD(e1.E[b]);
			f2[b] = FD(e2.E[b]);
		}
		
		//Degenerate spin projections:
		matrix skdeg = degenerateProject(e1.S[2], e1.E);
		matrix sk2deg = degenerateProject(e2.S[2], e2.E);
		
		// 1/T1kn = prefac / NS sum_k k2 |sk^deg g_kk2 - g_kk2 sk2^deg|^2 delta n_q f' (1-f)
		int nModes_ = T1p->acousticonly ? 3 : nModes;
		for(int alpha = 0; alpha < nModes_; alpha++) //Loop over phonon modes
		{	//Phonon occupation
			const double& omegaPh = ph.omega[alpha];
			if (omegaPh < T1p->omegaStart) continue;
			double nPh = BoseEinstein(omegaPh);
			
			//Spin-phonon commutator:
			const matrix& g = mEph.M[alpha];
			matrix sg_commute = skdeg * g - g * sk2deg;
			
			//Accumulate contributions:
			const complex* sg_commuteData = sg_commute.data();
			for(int b2=0; b2<nBands; b2++)
			for(int b1=0; b1<nBands; b1++)
				T1kn[o2][e2.ik][b2] += wko1[o1] //prefactor
					* (sg_commuteData++)->norm() //matrix elements
					* Gaussian(e1.E[b1] - e2.E[b2] - omegaPh) //energy conservation
					* nPh * f2[b2] * (1 - f[b1]); //occupation factors
		}
	}
	
	void get_T1()
	{
		for (int io = 0; io < nOffsets2; io++){
			double dtmp = 0.;
			for (int ik = 0; ik < nk; ik++)
			for (int b = 0; b < nBands; b++){
				T1kn[io][ik][b] *= prefacT1kn;
				dtmp += T1kn[io][ik][b];
			}
			T1 += dtmp * wko2[io];
		}
		T1 /= NS;
	}
	
	double Gaussian(double E) { return exp(T1p->EconserveExpFac * E * E); }
	
	double BoseEinstein(const double omegaPh)
	{
		double omegaPhByT = omegaPh / T1p->T;
		if (omegaPhByT < 1e-3) return 0.; //avoid 0./0. below
		return omegaPhByT>36 ? 0. : 1. / (exp(omegaPhByT) - 1.); //avoid overflow
	}
	
	double FD(double E) { return 1. / (exp((E - T1p->dmu) / T1p->T) + 1.); }
	
	static void ePhProcess(const FeynWann::MatrixEph& mEph, void* params){
		((CollectEph*)params)->process(mEph);
	}
};

void find_equiv_k(std::vector<vector3<>> &kmesh, FeynWann &fw, std::vector<int> &invertList, std::vector<int> &equiv_k, std::vector<int> &kReduced){
	matrix3<> G = 2 * M_PI*inv(fw.R), GGT = G*(~G);
	PeriodicLookup<vector3<>> plook(kmesh, GGT);
	for (size_t i = 0; i < kmesh.size(); i++){ equiv_k[i] = i; }
	for (size_t i0 = 0; i0 < kmesh.size(); i0++){
		if (equiv_k[i0] == i0){	//Find orbit of this k-points under symmetries:
			equiv_k[i0] = -1;
			for (int invert : invertList)
			for (const SpaceGroupOp& op : fw.sym){
				size_t i = plook.find(invert * kmesh[i0] * op.rot);
				if (i != string::npos && equiv_k[i] == i){
					equiv_k[i] = i0; //i will be covered in i0's orbit
					equiv_k[i0]--;
				}
			}
			kReduced.push_back(i0);
		}
	}
}

void write_equiv_k(std::vector<vector3<>> &kmesh, std::vector<int> &equiv_k, std::vector<int> &kReduced){
	string fname = "equiv_k.dat";
	FILE* fp = fopen(fname.c_str(), "w");
	fprintf(fp, "reduced k points %lu:\n", kReduced.size());
	for (int i : kReduced)
		fprintf(fp, "%4d %12.8lf %12.8lf %12.8lf\n", i, kmesh[i][0], kmesh[i][1], kmesh[i][2]);
	fprintf(fp, "other k points:\n");
	for (size_t i = 0; i < kmesh.size(); i++)
	if (equiv_k[i]>0)
		fprintf(fp, "%4lu %4d %12.8lf %12.8lf %12.8lf\n", i, equiv_k[i], kmesh[i][0], kmesh[i][1], kmesh[i][2]);
	fclose(fp);
}

void get_kMult(InputMap &inputMap, FeynWann &fw, vector3<int> NkMult[], std::vector<vector3<>> kMult[], vector3<> &kshift, vector3<int> &NkFine){
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	NkMult[1][0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1][1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[1][2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	logPrintf("NkMult2 = "); NkMult[1].print(globalLog, " %d ");
	const int NkMult1All = inputMap.get("NkMult1", 1); //increase in number of k-points for phonon mesh
	NkMult[0][0] = inputMap.get("NkxMult1", NkMult1All); //override increase in x direction
	NkMult[0][1] = inputMap.get("NkyMult1", NkMult1All); //override increase in y direction
	NkMult[0][2] = inputMap.get("NkzMult1", NkMult1All); //override increase in z direction
	logPrintf("NkMult1 = "); NkMult[0].print(globalLog, " %d ");

	//Check NkMult compatibility with symmetries:
	for (int kork2 = 0; kork2 < 2; kork2++)
	for (const SpaceGroupOp& op : fw.sym)
	{	//Similar to Symmetries::checkFFTbox in JDFTx
		matrix3<int> mMesh = Diag(NkMult[kork2]) * op.rot;
		for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		if (mMesh(i, j) % NkMult[kork2][j] == 0)
			mMesh(i, j) /= NkMult[kork2][j];
		else
		{
			logPrintf("NkMult%d not commensurate with symmetry matrix:\n", kork2 + 1);
			op.rot.print(globalLog, " %2d ");
			op.a.print(globalLog, " %lg ");
			die("NkMult%d not commensurate with symmetries.\n", kork2 + 1);
		}
	}

	//Construct NkMult mesh:
	vector3<> kOffset[2];
	for (int iDir = 0; iDir < 3; iDir++){
		kOffset[0][iDir] = fw.isTruncated[iDir] ? 0. : kshift[iDir]; //offset from Gamma in periodic directions
		kOffset[1][iDir] = 0.; // shifting only one of them
		for (int kork2 = 0; kork2 < 2; kork2++)
		if (fw.isTruncated[iDir] && NkMult[kork2][iDir] != 1){
			logPrintf("Setting NkMult = 1 along truncated direction %d.\n", iDir + 1);
			NkMult[kork2][iDir] = 1; //no multiplication in truncated directions
		}
		NkFine[iDir] = fw.kfold[iDir] * NkMult[1][iDir];
	}
	vector3<int> ikMult;
	for (int kork2 = 0; kork2 < 2; kork2++){
		matrix3<> NkMultInv = inv(Diag(vector3<>(NkMult[kork2])));
		for (ikMult[0] = 0; ikMult[0] < NkMult[kork2][0]; ikMult[0]++)
		for (ikMult[1] = 0; ikMult[1] < NkMult[kork2][1]; ikMult[1]++)
		for (ikMult[2] = 0; ikMult[2] < NkMult[kork2][2]; ikMult[2]++)
			kMult[kork2].push_back(NkMultInv * (ikMult + kOffset[kork2]));
	}
	logPrintf("Effective interpolated k-mesh dimensions: ");
	NkFine.print(globalLog, " %d ");
}

void get_ko(FeynWann &fw, std::vector<int> &invertList, std::vector<vector3<>> kMult[], std::vector<vector3<>> ko[], std::vector<double> wko[], int nOffsets[]){
	matrix3<> G = 2 * M_PI*inv(fw.R), GGT = G*(~G);
	//--- Compile list of inversions to check:
	invertList.push_back(+1);
	invertList.push_back(-1);
	for (const SpaceGroupOp& op : fw.sym)
	if (op.rot == matrix3<int>(-1, -1, -1)){
		invertList.resize(1); //inversion explicitly found in symmetry list, so remove from invertList
		break;
	}
	//Reduce under symmetries (simplified version of Symmetries::reduceKmesh from JDFTx):
	matrix3<> kfoldInv = inv(Diag(vector3<>(fw.kfold)));
	logPrintf("kfoldInv: %12.8lf %12.8lf %12.8lf\n", kfoldInv(0, 0), kfoldInv(1, 1), kfoldInv(2, 2));

	bool apply_sym = ( kMult[0].size() > 1 && kMult[1].size() == 1 ) || ( kMult[0].size() == 1 && kMult[1].size() > 1 );
	for (int kork2 = 0; kork2 < 2; kork2++){
		if (mpiWorld->isHead())
		{	//compile kpoint map:
			PeriodicLookup<vector3<>> plook(kMult[kork2], GGT);
			std::vector<bool> kDone(kMult[kork2].size(), false);
			for (size_t iSrc = 0; iSrc < kMult[kork2].size(); iSrc++)
			if (!kDone[iSrc])
			{
				double w = 0.; //weight of current point
				if (apply_sym)
					for (int invert : invertList)
					for (const SpaceGroupOp& op : fw.sym)
					{
						size_t iDest = plook.find(invert * kMult[kork2][iSrc] * op.rot);
						if (iDest != string::npos && (!kDone[iDest]))
						{
							kDone[iDest] = true; //iDest in iSrc's orbit
							w += 1.; //increase weight of iSrc
						}
					}
				else
					w = 1;
				//add corresponding offset:
				ko[kork2].push_back(kfoldInv * kMult[kork2][iSrc]);
				logPrintf("kMult[%lu] = %12.8lf %12.8lf %12.8lf\n", iSrc, kMult[kork2][iSrc][0], kMult[kork2][iSrc][1], kMult[kork2][iSrc][2]);
				wko[kork2].push_back(w);
			}
		}
		//--- make available on all processes
		nOffsets[kork2] = ko[kork2].size(); mpiWorld->bcast(&nOffsets[kork2], 1);
		ko[kork2].resize(nOffsets[kork2]); mpiWorld->bcast(&ko[kork2][0][0], 3 * nOffsets[kork2]);
		wko[kork2].resize(nOffsets[kork2]); mpiWorld->bcast(&wko[kork2][0], nOffsets[kork2]);
		logPrintf("\n%lu offsets in NkMult mesh reduced to %d under symmetries.\n", kMult[kork2].size(), nOffsets[kork2]);
		for (int o = 0; o < nOffsets[kork2]; o++)
			logPrintf("offsets: %d, ko: %12.8lf %12.8lf %12.8lf, wko: %12.8lf\n", o, ko[kork2][o][0], ko[kork2][o][1], ko[kork2][o][2], wko[kork2][o]);
	}
}

int main(int argc, char** argv){
	//Initialize FeynWann:
	InitParams ip = FeynWann::initialize(argc, argv, "Electron-phonon scattering contribution to electron linewidth.");
	FeynWannParams fwp;
	fwp.needSymmetries = true;
	fwp.needCellWeights = true;
	fwp.needPhonons = true;
	fwp.needSpin = true;
	FeynWann fw(fwp);

	//Read input including NkMult; Get ko:
	InputMap inputMap(ip.inputFilename);
	T1params T1par(fw, inputMap); T1p = &T1par; // get input parameters except NkMult

	vector3<int> NkMult[2], NkFine; // NkMult[1] for k2, NkMult[0] for k1
	std::vector<vector3<>> kMult[2];
	get_kMult(inputMap, fw, NkMult, kMult, T1p->kshift, NkFine);

	std::vector<int> invertList;
	std::vector<vector3<>> ko[2]; std::vector<double> wko[2]; int nOffsets[2];
	get_ko(fw, invertList, kMult, ko, wko, nOffsets);

	//Initialize cEph and Collect energies, k-point mesh, NS and skip
	CollectEph cEph(fw, inputMap, NkMult, nOffsets, wko);
	for (cEph.o2 = 0; cEph.o2 < cEph.nOffsets2; cEph.o2++){
		fw.eLoop(ko[1][cEph.o2], CollectEph::collectE, &cEph);
		//--- make available on all processes:
		for (int i = 0; i < cEph.nk; i++){
			int root = cEph.E[cEph.o2][i].size() ? mpiGroup->iProcess() : mpiGroup->nProcesses(); //my process ID or N, depending on whether I have E[i]
			mpiGroup->allReduce(root, MPIUtil::ReduceMin); //lowest process number which has E[i] available
			cEph.E[cEph.o2][i].resize(fw.nBands);
			mpiGroup->bcast(cEph.E[cEph.o2][i].data(), fw.nBands, root);
		}
	}
	mpiGroup->allReduce(&cEph.kmesh[0][0], 3 * cEph.kmesh.size(), MPIUtil::ReduceSum);
	mpiGroup->allReduce(&cEph.NS, 1, MPIUtil::ReduceSum);
	mpiGroup->allReduce(&cEph.dfdemax, 1, MPIUtil::ReduceMax);

	cEph.get_skip();
	std::vector<int> equiv_k(cEph.kmesh.size()), kReduced;
	find_equiv_k(cEph.kmesh, fw, invertList, equiv_k, kReduced);

	logPrintf("\n");
	if (ip.dryRun){
		logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}

	//Initialize sampling parameters:
	int oStart = 0, oStop = 0;
	if (mpiGroup->isHead())
		TaskDivision(nOffsets[0]*nOffsets[1], mpiGroupHead).myRange(oStart, oStop);
	mpiGroup->bcast(oStart);
	mpiGroup->bcast(oStop);
	int noMine = oStop - oStart; //number of offsets handled by current group
	int oInterval = std::max(1, int(round(noMine / 50.))); //interval for reporting progress

	//For T1kn, collect results for each offset
	logPrintf("Collecting 1/T1kn: "); logFlush();
	for (int o = oStart; o < oStop; o++){
		cEph.o1 = o / cEph.nOffsets2;
		cEph.o2 = o % cEph.nOffsets2;
		fw.ePhLoop(ko[0][cEph.o1], ko[1][cEph.o2], CollectEph::ePhProcess, &cEph);
		//Print progress:
		if ((o - oStart + 1) % oInterval == 0) { logPrintf("%d%%(%2d,%2d) ", int(round((o - oStart + 1)*100. / noMine)), cEph.o1, cEph.o2); logFlush(); }
	}
	//Collect results from all processes:
	for (std::vector<diagMatrix>& dArr : cEph.T1kn)
	for (diagMatrix& d : dArr)
		mpiWorld->allReduceData(d, MPIUtil::ReduceSum);

	logPrintf("done.\n"); logFlush();
	cEph.get_T1();
	logPrintf("T1 = %19.12le Eh^-1 =  %19.12le ps\n", 1./cEph.T1, 1./cEph.T1/(1e3*fs));
	
	fw.free();
	FeynWann::finalize();
	return 0;
}
