#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <InputMap.h>
#include <lindblad/LindbladFile.h>


//Mean absolute value of matrix
double mean_abs(const matrix& X)
{	double sum_abs = 0.0;
	const complex* Xdata = X.data();
	for(size_t i=0; i<X.nData(); i++)
		sum_abs += Xdata[i].abs();
	return sum_abs / X.nData();
}

//Elementwise multiply X.conj * Y (used for phase matching)
matrix elementwise_multiply_conj(const matrix& X, const matrix& Y)
{	assert(X.nData() == Y.nData());
	matrix result(Y); //modifiable copy
	complex* resultData = result.data();
	const complex* Xdata = X.data();
	for(size_t i=0; i<X.nData(); i++)
		resultData[i] *= Xdata[i].conj();
	return result;
}


struct LindbladModify
{
	bool dryRun;
	string prefix;
	string inFile;
	string outFile;
	
	LindbladFile::Header h;
	TaskDivision kDivision;
	size_t ikStart, ikStop, nkMine; //!< range and number of selected k-points on this process
	std::vector<LindbladFile::Kpoint> kpoints;
	FeynWannParams fwp;
	double mu;
	
	LindbladModify(int argc, char** argv)
	{
		InitParams ip = FeynWann::initialize(argc, argv, "Modify Lindblad sparse matrices to include DFT matrix elements");
		dryRun = ip.dryRun;
		
		//Get the system parameters:
		InputMap inputMap(ip.inputFilename);
		prefix = inputMap.getString("prefix"); //file prefix to read/write DFT information from
		inFile = inputMap.getString("inFile"); //input lindblad data file name
		outFile = inputMap.getString("outFile"); //output lindblad data file name
		fwp = FeynWannParams(&inputMap); //need settings like scissor to match energies
		
		logPrintf("\nInputs after conversion to atomic units:\n");
		logPrintf("prefix = %s\n", prefix.c_str());
		logPrintf("inFile = %s\n", inFile.c_str());
		logPrintf("outFile = %s\n", outFile.c_str());
		fwp.printParams();
		
		//Get key DFT parameters using FeynWann:
		FeynWann fw(fwp);
		mu = fw.mu;
	}
	
	void run()
	{	read();
		if(dryRun)
		{	printKpointList();
			logPrintf("Use JDFTx to generate eigenvals, momenta, L and S (if applicable) for the above kpoints.\n\n");
			logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
			return;
		}
		updateMatrixElements();
		write();
	}
	
	void read()
	{	logPrintf("Reading %s ... ", inFile.c_str()); logFlush();
		
		//Read header:
		MPIUtil::File fp;
		mpiWorld->fopenRead(fp, inFile.c_str());
		h.read(fp, mpiWorld);
		
		//Read k-point offsets:
		std::vector<size_t> byteOffsets(h.nk);
		mpiWorld->freadData(byteOffsets, fp);

		//Divide and allocate k-point data:
		kDivision.init(h.nk, mpiWorld);
		kDivision.myRange(ikStart, ikStop);
		nkMine = ikStop-ikStart;
		kpoints.resize(nkMine);
		
		//Read k-point data belonging to current process:
		mpiWorld->fseek(fp, byteOffsets[ikStart], SEEK_SET);
		for(size_t ikMine=0; ikMine<nkMine; ikMine++)
		{	LindbladFile::Kpoint& kpoint = kpoints[ikMine];
			kpoint.read(fp, mpiWorld, h);
		}
		logPrintf("done.\n");
	}
	
	void printKpointList()
	{	//Compile list of all k-points:
		std::vector<vector3<>> k(h.nk);
		for(size_t ik=ikStart; ik<ikStop; ik++)
			k[ik] = kpoints[ik - ikStart].k;
		mpiWorld->allReduceData(k, MPIUtil::ReduceSum);
		
		//Write file in bandstruct.kpoints format:
		string fname = prefix + ".kpoints";
		logPrintf("Writing %s ... ", fname.c_str()); logFlush();
		if(mpiWorld->isHead())
		{	FILE* fp = fopen(fname.c_str(), "w");
			fprintf(fp, "kpoint-folding 1 1 1\n");
			fprintf(fp, "symmetries none\n");
			double wk = 1.0 / h.nk;
			for(const vector3<>& ki: k)
				fprintf(fp, "kpoint %+15.12lf %+15.12lf %+15.12lf %.15le\n",
					ki[0], ki[1], ki[2], wk);
			fclose(fp);
		}
		logPrintf("done.\n");
	}
	
	void updateMatrixElements()
	{	//Get number of DFT bands from eigenvals size:
		int nBands = int(fileSize((prefix + ".eigenvals").c_str()) / (h.nk * sizeof(double)));
		size_t sizeE = nBands * sizeof(double); //size per k-point
		size_t sizeP = 3 * nBands * nBands * sizeof(complex); //size per k-point, same for L and S
		
		//Open all relevant matrix element files:
		MPIUtil::File fpE, fpP, fpL, fpS;
		mpiWorld->fopenRead(fpE, (prefix + ".eigenvals").c_str(), h.nk * sizeE);
		mpiWorld->fopenRead(fpP, (prefix + ".momenta").c_str(), h.nk * sizeP);
		mpiWorld->fopenRead(fpL, (prefix + ".L").c_str(), h.nk * sizeP);
		if(h.spinorial) mpiWorld->fopenRead(fpS, (prefix + ".S").c_str(), h.nk * sizeP);
		
		//Process each k-point separately:
		mpiWorld->fseek(fpE, ikStart * sizeE, SEEK_SET);
		mpiWorld->fseek(fpP, ikStart * sizeP, SEEK_SET);
		mpiWorld->fseek(fpL, ikStart * sizeP, SEEK_SET);
		if(h.spinorial) mpiWorld->fseek(fpS, ikStart * sizeP, SEEK_SET);
		//--- allocate common variables for all DFT input
		diagMatrix E(nBands);
		std::vector<matrix> P(3, zeroes(nBands, nBands)), L(3, zeroes(nBands, nBands)), S;
		if(h.spinorial) S.assign(3, zeroes(nBands, nBands));
		//--- collect error statistics in matrix elements
		double maeE = 0.0, maeP = 0.0, maeS = 0.0, maeL = 0.0;
		logPrintf("Updating matrix elements: "); logFlush();
		size_t ikInterval = std::max(1, int(round(nkMine/50.))); //interval for reporting progress
		for(size_t ik=0; ik<nkMine; ik++)
		{	LindbladFile::Kpoint& kpoint = kpoints[ik];
			
			//Read DFT matrix elements:
			mpiWorld->freadData(E, fpE);
			for(matrix& Pi: P) mpiWorld->freadData(Pi, fpP);
			for(matrix& Li: L) mpiWorld->freadData(Li, fpL);
			if(h.spinorial) for(matrix& Si: S) mpiWorld->freadData(Si, fpS);
			
			//Shift DFT energies to same scale as FeynWann:
			for(double& Ei: E) Ei -= mu;
			if(fwp.scissor) fwp.applyScissor(E);
			
			//Align DFT energies with those in data file (find window offsets):
			int outerStart = 0;
			double maeEcur = DBL_MAX;
			for(int bStart=0; bStart<(nBands - kpoint.nOuter); bStart++)
			{	double maeEtest = 0.0;
				for(int b=0; b<kpoint.nOuter; b++)
					maeEtest += fabs(E[bStart + b] - kpoint.E[b]);
				maeEtest *= (1.0 / kpoint.nOuter);
				if(maeEtest < maeEcur)
				{	maeEcur = maeEtest;
					outerStart = bStart;
				}
			}
			maeE += maeEcur;
			int outerStop = outerStart + kpoint.nOuter;
			int innerStart = outerStart + kpoint.innerStart; //start of inner window in DFT set
			int innerStop = innerStart + kpoint.nInner;
			
			//Select active energy window matrix elements from DFT:
			diagMatrix Eouter = E(outerStart, outerStop);
			diagMatrix Einner = E(innerStart, innerStop);
			std::vector<matrix> PinnerW(3), Pinner(3), Linner(3), Sinner(3);
			for(int iDir=0; iDir<3; iDir++)
			{	PinnerW[iDir] = kpoint.P[iDir](
					0, kpoint.nInner, kpoint.innerStart, kpoint.innerStart + kpoint.nInner
				); //Wannier version in data file has optional outer components dropped here
				Pinner[iDir] = P[iDir](innerStart, innerStop, innerStart, innerStop);
				Linner[iDir] = L[iDir](innerStart, innerStop, innerStart, innerStop);
				if(h.spinorial)
					Sinner[iDir] = S[iDir](innerStart, innerStop, innerStart, innerStop);
			}
			
			//Resolve degeneracies within inner window using all available matrix elements:
			matrix Vinner(eye(kpoint.nInner)); //dgenerate-subspace rotations of DFT to match Wannier
			for(int bStart=0; bStart<kpoint.nInner;)
			{	int bStop = bStart + 1;
				while(
					(bStop < kpoint.nInner) and
					(Einner[bStop] < Einner[bStart] + fwp.degeneracyThreshold)
				)
					bStop++;
				int bSize = bStop - bStart; //size of current degenerate subspace
				if(bSize > + 1)
				{	//Find best of several random perturbations at resolving degeneracy:
					double deig_best = 0.0; //highest eigenvalue separation introduced by perturbation
					matrix VdftBest, VwBest;
					for(int iRepeat=0; iRepeat<10; iRepeat++)
					{	matrix Hdft = zeroes(bSize, bSize), Hw = zeroes(bSize, bSize);
						double wSqSum = 0.0;
						for(int iDir=0; iDir<3; iDir++)
						{	//Use P matrix elements:
							double w = Random::normal();
							Hdft += w * Pinner[iDir](bStart, bStop, bStart, bStop);
							Hw += w * PinnerW[iDir](bStart, bStop, bStart, bStop);
							wSqSum += w * w;
							//Use S matrix elements if available:
							if(h.spinorial)
							{	w = Random::normal();
								Hdft += w * Sinner[iDir](bStart, bStop, bStart, bStop);
								Hw += w * kpoint.S[iDir](bStart, bStop, bStart, bStop);
								wSqSum += w * w;
							}
						}
						double normFac = 1.0 / sqrt(wSqSum);
						Hdft *= normFac;
						Hw *= normFac;
						
						//Diagonalize and check extent of degeneracy resolution:
						diagMatrix Edft, Ew; matrix Vdft;
						Hdft.diagonalize(Vdft, Edft);
						double deig = DBL_MAX;
						for(int b=0; b<bSize-1; b++)
							deig = std::min(deig, Edft[b+1] - Edft[b]);
						if(deig > deig_best)
						{	deig_best = deig;
							VdftBest = Vdft;
							Hw.diagonalize(VwBest, Ew);
						}
					}
					Vinner.set(bStart, bStop, bStart, bStop, VdftBest * dagger(VwBest));
				}
				bStart = bStop;
			}
			for(matrix& Pi: Pinner) Pi = dagger(Vinner) * Pi * Vinner;
			for(matrix& Li: Linner) Li = dagger(Vinner) * Li * Vinner;
			if(h.spinorial) for(matrix& Si: Sinner) Si = dagger(Vinner) * Si * Vinner;
			
			//Fix relative phases of states (for off-diagonal matrix elements):
			matrix conjProduct = zeroes(kpoint.nInner, kpoint.nInner);
			for(int iDir=0; iDir<3; iDir++)
			{	// conjProduct += elementwise_multiply_conj(Pinner[iDir], PinnerW[iDir]);
				if(h.spinorial) conjProduct += elementwise_multiply_conj(Sinner[iDir], kpoint.S[iDir]);
			}
			//--- The phase of each off-diagonal element in conjProduct is now the best fit relative phase for that state pair.
			//--- Now find a consistent phase for each state that best matches these pair relative phases.
			std::vector<complex> phase(kpoint.nInner, complex(1.0)); //WLOG let first state have zero phase (storing cis(phase))
			for(int b1=1; b1<kpoint.nInner; b1++)
			{	//Reference phase to all previous cases:
				complex conjProductSum;
				for(int b2=0; b2<b1; b2++)
					conjProductSum += phase[b2].conj() * conjProduct(b2, b1);
				phase[b1] = conjProductSum / conjProductSum.abs(); //make unit complex number
			}
			//--- apply phase
			matrix phaseMat(phase);
			for(matrix& Pi: Pinner) Pi = dagger(phaseMat) * Pi * phaseMat;
			for(matrix& Li: Linner) Li = dagger(phaseMat) * Li * phaseMat;
			if(h.spinorial) for(matrix& Si: Sinner) Si = dagger(phaseMat) * Si * phaseMat;
			
			//Replace matrix elements with DFT versions wherever possible:
			kpoint.E = Eouter;
			for(int iDir=0; iDir<3; iDir++)
			{	maeP += mean_abs(Pinner[iDir] - PinnerW[iDir]) / 3;
				if(kpoint.nInner == kpoint.nOuter) //only replace P when no outer bands
					kpoint.P[iDir] = Pinner[iDir];
				if(h.spinorial)
				{	maeS += mean_abs(Sinner[iDir] - kpoint.S[iDir]) / 3;
					kpoint.S[iDir] = Sinner[iDir];
				}
				if(h.haveL) maeL += mean_abs(Linner[iDir] - kpoint.L[iDir]) / 3;
				kpoint.L[iDir] = Linner[iDir];
			}
			
			//Print progress:
			if((ik+1)%ikInterval==0) { logPrintf("%d%% ", int(round((ik+1)*100./nkMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		
		//Report matrix element error statistics:
		mpiWorld->allReduce(maeE, MPIUtil::ReduceSum); maeE /= h.nk; logPrintf("MAE(E): %le\n", maeE);
		mpiWorld->allReduce(maeP, MPIUtil::ReduceSum); maeP /= h.nk; logPrintf("MAE(P): %le\n", maeP);
		if(h.spinorial) { mpiWorld->allReduce(maeS, MPIUtil::ReduceSum); maeS /= h.nk; logPrintf("MAE(S): %le\n", maeS); }
		if(h.haveL) { mpiWorld->allReduce(maeL, MPIUtil::ReduceSum); maeL /= h.nk; logPrintf("MAE(L): %le\n", maeL); }
		
		//Close matrix element files:
		mpiWorld->fclose(fpE);
		mpiWorld->fclose(fpP);
		mpiWorld->fclose(fpL);
		if(h.spinorial) mpiWorld->fclose(fpS);
		h.haveL = true; //L matrix elements have been set now, even if originally not present
	}
	
	void write()
	{	
		//Synchronize kpoint sizes and ownership:
		std::vector<LindbladFile::Kpoint> kpAll(h.nk); //array of kpoint data for all active k-points
		std::vector<int> kpWhose(h.nk); //index of process in mpiWorld that owns each entry in kpAll
		std::vector<size_t> kpSize(h.nk); //size in bytes of each entry in kpAll when written to file
		for(size_t ik=ikStart; ik<ikStop; ik++)
		{	std::swap(kpAll[ik], kpoints[ik - ikStart]);
			kpWhose[ik] = mpiWorld->iProcess();
			kpSize[ik] = kpAll[ik].nBytes(h);
		}
		mpiWorld->allReduceData(kpWhose, MPIUtil::ReduceMax); //now each process knows who owns a specific k-point data
		mpiWorld->allReduceData(kpSize, MPIUtil::ReduceMax); //... and its size when written to file
		
		//Compute offsets to each k-point within file:
		std::vector<size_t> byteOffsets(h.nk);
		byteOffsets[0] = h.nBytes() + h.nk*sizeof(size_t); //offset to first k-point (header + byteOffsets array)
		for(size_t ik=0; ik+1<h.nk; ik++)
			byteOffsets[ik+1] = byteOffsets[ik] + kpSize[ik];
		
		//Open file
		#ifdef MPI_SAFE_WRITE
		FILE* fp = NULL;
		if(mpiWorld->isHead()) fp = fopen(outFile.c_str(), "w"); //I/O from world head alone
		#else
		MPIUtil::File fp;
		mpiWorld->fopenWrite(fp, outFile.c_str()); //I/O collectively from all processes
		#endif
		
		//Header and byte offsets to each k
		if(mpiWorld->isHead())
		{	std::ostringstream oss;
			h.write(oss);
			#ifdef MPI_SAFE_WRITE
			fwrite(oss.str().data(), 1, h.nBytes(), fp);
			fwrite(byteOffsets.data(), sizeof(size_t), byteOffsets.size(), fp);
			#else
			mpiWorld->fwrite(oss.str().data(), 1, h.nBytes(), fp);
			mpiWorld->fwriteData(byteOffsets, fp);
			#endif
		}
		
		//Write data:
		logPrintf("Writing %s: ", outFile.c_str()); logFlush();
		size_t ikInterval = std::max(1, int(round(h.nk/50.))); //interval for reporting progress
		for(size_t ik=0; ik<h.nk; ik++)
		{	const LindbladFile::Kpoint& kp = kpAll[ik];
			std::vector<char> buf(kpSize[ik]); //buffer containing serialization of kp
			bool isMine = (kpWhose[ik] == mpiWorld->iProcess());
			if(isMine)
			{	membuf mbuf(buf);
				std::ostream os(&mbuf);
				kp.write(os, h);
				#ifdef MPI_SAFE_WRITE
				if(not mpiWorld->isHead()) //send to head to write:
					mpiWorld->sendData(buf, 0, ik);
				#else
				//Write from each process in parallel:
				mpiWorld->fseek(fp, byteOffsets[ik], SEEK_SET);
				mpiWorld->fwrite(buf.data(), 1, kpSize[ik], fp);
				#endif
			}
			#ifdef MPI_SAFE_WRITE
			//Write data from head:
			if(mpiWorld->isHead())
			{	if(not isMine) //Recv data to write
					mpiWorld->recvData(buf, kpWhose[ik], ik);
				fseek(fp, byteOffsets[ik], SEEK_SET);
				fwrite(buf.data(), 1, kpSize[ik], fp);
			}
			#endif
			//Print progress:
			if((ik+1)%ikInterval==0) { logPrintf("%d%% ", int(round((ik+1)*100./h.nk))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		
		//Close file:
		#ifdef MPI_SAFE_WRITE
		if(mpiWorld->isHead()) fclose(fp);
		#else
		mpiWorld->fclose(fp);
		#endif
	}
};


int main(int argc, char** argv)
{	LindbladModify(argc, argv).run();
	FeynWann::finalize();
}
