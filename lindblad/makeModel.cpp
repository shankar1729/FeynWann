#include <core/matrix.h>
#include <core/Random.h>
#include <core/Units.h>
#include <InputMap.h>
#include <lindblad/LindbladFile.h>

int main(int argc, char** argv)
{
	InitParams ip = FeynWann::initialize(argc, argv, "Create a (2-band) spin model system");

	//Get input parameters:
	InputMap inputMap(ip.inputFilename);
	const int nK = int(inputMap.get("nK")); //number of k-points
	const double Tesla = Joule/(Ampere*meter*meter);
	const vector3<> sigmaB = inputMap.getVector("sigmaB") * Tesla; //magnitude of internal magnetic field fluctuations per direction
	const int nBands = 2;
	
	//Print back input parameters (converted):
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nK = %d\n", nK);
	logPrintf("sigmaB = "); sigmaB.print(globalLog, " %lg ");
	logPrintf("\n");
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Create Pauli matrices:
	vector3<matrix> S;
	// Sx
	S[0] = zeroes(nBands, nBands);
	S[0].set(0,1, 1);
	S[0].set(1,0, 1);
	// Sy
	S[1] = zeroes(nBands, nBands);
	S[1].set(0,1, complex(0,-1));
	S[1].set(1,0, complex(0, 1));
	// Sz
	S[2] = zeroes(nBands, nBands);
	S[2].set(0,0,  1);
	S[2].set(1,1, -1);
	
	//Create model Hamiltonian:
	Random::seed(0);
	std::vector<LindbladFile::Kpoint> kArray(nK);
	for(LindbladFile::Kpoint& k: kArray)
	{	k.nInner = k.nOuter = nBands;
		k.innerStart = 0;

		//Create a random magnetic field hamiltonian:
		matrix H0 = zeroes(nBands, nBands);
		for(int iDir=0; iDir<3; iDir++)
			H0 += (sigmaB[iDir] * Random::normal()) * S[iDir];
		
		//Diagonalize:
		matrix V;
		H0.diagonalize(V, k.E);
		for (int i=0; i < 3; i++)
		{   k.S[i] = dagger(V) * S[i] * V;
			k.P[i] = zeroes(nBands, nBands);
		}
	}

	//Prepare the file header:
	LindbladFile::Header h;
	h.dmuMin = 0;
	h.dmuMax = 0;
	h.Tmax = DBL_MAX;
	h.pumpOmegaMax = 0;
	h.probeOmegaMax = 0;
	h.nk = nK;
	h.nkTot = nK;
	h.ePhEnabled = true;
	h.spinorial = true;
	h.spinWeight = 1;
	h.R = matrix3<>(1, 1, 1);

	//Compute offsets to each k-point within file:
	std::vector<size_t> byteOffsets(h.nk);
	byteOffsets[0] = h.nBytes() + h.nk*sizeof(size_t); //offset to first k-point (header + byteOffsets array)
	for(size_t ik=0; ik+1<h.nk; ik++)
		byteOffsets[ik+1] = byteOffsets[ik] + kArray[ik].nBytes(h);

	//Write file:
	if(mpiWorld->isHead())
	{	FILE* fp = fopen("ldbd.dat", "w");
		// --- header
		std::ostringstream oss;
		h.write(oss);
		fwrite(oss.str().data(), 1, h.nBytes(), fp);
		// --- byte offsets
		fwrite(byteOffsets.data(), sizeof(size_t), byteOffsets.size(), fp);
		// --- data for each k-point
		for(const LindbladFile::Kpoint& k: kArray)
		{	oss.str(std::string());
			k.write(oss, h);
			fwrite(oss.str().data(), 1, k.nBytes(h), fp);
		}
		fclose(fp);
	}
	FeynWann::finalize();
	return 0;
}
