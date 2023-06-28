#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <InputMap.h>
#include <lindblad/LindbladFile.h>

struct LindbladModify
{
	string prefix;
	string inFile;
	string outFile;
	
	LindbladFile::Header h;
	TaskDivision kDivision;
	size_t ikStart, ikStop, nkMine; //!< range and number of selected k-points on this process
	std::vector<LindbladFile::Kpoint> kpoints;
	
	LindbladModify(int argc, char** argv)
	{
		InitParams ip = FeynWann::initialize(argc, argv, "Modify Lindblad sparse matrices to include DFT matrix elements");

		//Get the system parameters:
		InputMap inputMap(ip.inputFilename);
		prefix = inputMap.getString("prefix"); //file prefix to read/write DFT information from
		inFile = inputMap.getString("inFile"); //input lindblad data file name
		outFile = inputMap.getString("outFile"); //output lindblad data file name
		
		logPrintf("\nInputs after conversion to atomic units:\n");
		logPrintf("prefix = %s\n", prefix.c_str());
		logPrintf("inFile = %s\n", inFile.c_str());
		logPrintf("outFile = %s\n", outFile.c_str());
	}
	
	void run()
	{	read();
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
