/*-------------------------------------------------------------------
Copyright 2019 Adela Habib, Ravishankar Sundararaman

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
#include "FeynWann.h"
#include "InputMap.h"
#include <core/Units.h>

//Read a list of k-points from a file
std::vector<vector3<>> readKpointsFile(string fname)
{	std::vector<vector3<>> kArr;
	logPrintf("Reading '%s' ... ", fname.c_str()); logFlush();
	ifstream ifs(fname); if(!ifs.is_open()) die("could not open file.\n");
	while(!ifs.eof())
	{	string line; getline(ifs, line);
		trim(line);
		if(!line.length()) continue;
		//Parse line
		istringstream iss(line);
		string key; iss >> key;
		if(key == "kpoint")
		{	vector3<> k;
			iss >> k[0] >> k[1] >> k[2];
			kArr.push_back(k);
		}
	}
	ifs.close();
	logPrintf("done.\n");
	return kArr;
}

//Write debug code within process() to examine arbitrary e-ph properties along a k- or q-path
struct DebugEph
{	int bandStart, bandStop; //optional band range read in from input
	int modeStart, modeStop; //optional mode range read in from input
	
	DebugEph(int bandStart, int bandStop, int modeStart, int modeStop)
	: bandStart(bandStart), bandStop(bandStop), modeStart(modeStart), modeStop(modeStop)
	{
	}
	void process(const FeynWann::MatrixEph& mEph)
	{	const diagMatrix& E1 = mEph.e1->E;
		const diagMatrix& E2 = mEph.e2->E;
		const diagMatrix& omegaPh = mEph.ph->omega;
		int nBands = E1.nRows();
		int nModes= omegaPh.nRows();
		
		/*
		//---- Phonon frequency debug ----
		logPrintf("OMEGAPH(%lf,%lf,%lf):", mEph.ph->q[0], mEph.ph->q[1], mEph.ph->q[2]);
		for(const double omega: omegaPh)
			logPrintf(" %11.8lf", omega);
		logPrintf("\n");
		logFlush();
		*/
		
		/*
		//---- e-ph matrix element debug ----
		logPrintf("|g|(%lf,%lf,%lf):", mEph.ph->q[0], mEph.ph->q[1], mEph.ph->q[2]);
		double gNormCur = 0.0;
		for(int b1=0; b1<nBands; b1++) 
		{	for(int b2=0; b2<nBands; b2++)
			{	if ( (fabs(E1[b1] - E1[bandStart]) < 1e-5) 
						&&	 (fabs(E2[b2] - E2[bandStart]) < 1e-5) )
							gNormCur += mEph.M[modeStart](b1,b2).norm();	
			}
		}
		logPrintf(" %11.8lf", sqrt(gNormCur)); 
		logPrintf("\n");
		logFlush();
		*/
		
		//---- Spin commutator debug ---
		const double degeneracyThreshold = 1e-6;
		matrix S1z = degenerateProject(mEph.e1->S[2], E1);
		matrix S2z = degenerateProject(mEph.e2->S[2], E2);
		double Gsq = 0., SGsq = 0.; //traced over degenerate subspaces to specified band range
		const matrix& G = mEph.M[modeStart];
		matrix SGcomm = S1z * G - G * S2z;
		double E1min = E1[bandStart] - degeneracyThreshold;
		double E1max = E1[bandStop-1] + degeneracyThreshold;
		double E2min = E2[bandStart] - degeneracyThreshold;
		double E2max = E2[bandStop-1] + degeneracyThreshold;
		for(int mode=modeStart; mode<modeStop; mode++)
		{	const matrix& G = mEph.M[mode];
			matrix SGcomm = S1z * G - G * S2z;
			for(int b1=0; b1<nBands; b1++) if(E1[b1]>E1min and E1[b1]<E1max)
			for(int b2=0; b2<nBands; b2++) if(E2[b2]>E2min and E2[b2]<E2max)
			{	Gsq += G(b1,b2).norm();
				SGsq += SGcomm(b1,b2).norm();
			}
		}
		logPrintf("SGcommDEBUG: %le %le\n", sqrt(Gsq), sqrt(SGsq));
	}
	static void ePhProcess(const FeynWann::MatrixEph& mEph, void* params)
	{	((DebugEph*)params)->process(mEph);
	}
	
	inline matrix degenerateProject(const matrix& M, const diagMatrix& E)
	{	static const double degeneracyThreshold = 1e-6;
		matrix out = M;
		complex* outData = out.data();
		for(int b2=0; b2<out.nCols(); b2++)
			for(int b1=0; b1<out.nRows(); b1++)
			{	if(fabs(E[b1] - E[b2]) > degeneracyThreshold) (*outData) = 0;
				outData++;
			}
		return out;
	}
};

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Print electron phonon matrix element, |g_q|.");
	
	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(ip.inputFilename);
	const vector3<> k1 = inputMap.getVector("k1");
	string k2file = inputMap.getString("k2file"); //file containing list of k2 points (like a bandstruct.kpoints file)
	int bandStart = inputMap.get("bandStart", 0);
	int bandStop = inputMap.get("bandStop", 0); //replaced with nBands below if 0
	int modeStart = inputMap.get("modeStart", 0);
	int modeStop = inputMap.get("modeStop", 0); //replaced with nModes below if 0
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("k1 = "); k1.print(globalLog, " %lg ");
	logPrintf("k2file = %s\n", k2file.c_str());
	logPrintf("bandStart = %d\n", bandStart);
	logPrintf("bandStop = %d\n", bandStop);
	logPrintf("modeStart = %d\n", modeStart);
	logPrintf("modeStop = %d\n", modeStop);
	
	//Read k-points:
	std::vector<vector3<>> k2arr = readKpointsFile(k2file);
	logPrintf("Read %lu k-points from '%s'\n", k2arr.size(), k2file.c_str());
	
	//Initialize FeynWann:
	FeynWannParams fwp;
	fwp.needPhonons = true;
	fwp.ePhHeadOnly = true; //so as to debug k-path alone
	fwp.needSpin = true;
	FeynWann fw(fwp);
	if(!bandStop) bandStop = fw.nBands;
	if(!modeStop) modeStop = fw.nModes;
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	DebugEph src(bandStart, bandStop, modeStart, modeStop);
	
	for(vector3<> k2: k2arr)
		fw.ePhLoop(k1, k2, DebugEph::ePhProcess, &src);
	
	fw.free();
	FeynWann::finalize();
	return 0;
}
