#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <commands/command.h>
#include <lindblad/LindbladFile.h>
#include <core/Units.h>

int main()
{
	int nBands   = 2;
	int nKpoints = 100;
	double larmorFreq = 0.001; // time unit is 1/140 fs
	double deltaOmegaX = 0*larmorFreq/10;
	double deltaOmegaY = 0*larmorFreq/5;
	std::cout << "omega in ps = " << larmorFreq*1000*fs << std::endl;

	Random::seed(0);

	std::vector<LindbladFile::Kpoint> kArray(nKpoints);
	for (int ik=0; ik<nKpoints; ik++) 
	{	LindbladFile::Kpoint& k0 = kArray[ik];
		double randomNumber = Random::normal();
		double randomPerturbation = randomNumber/100*larmorFreq;
		
		k0.k[0] = ik;
		k0.nInner = k0.nOuter = nBands;
		k0.innerStart = 0;
		k0.E.resize(nBands);
		k0.E[0] =  (0.*larmorFreq + randomPerturbation)/2;
		k0.E[1] = -(0.*larmorFreq + randomPerturbation)/2;

		for (int i=0; i < 3; i++)
		{   k0.S[i] = zeroes(nBands, nBands);
			k0.P[i] = zeroes(nBands, nBands);
		}

		// Sx
		k0.S[0].set(0,1, 1);
		k0.S[0].set(1,0, 1);

		// Sy
		k0.S[1].set(0,1, complex(0,-1));
		k0.S[1].set(1,0, complex(0, 1));

		// Sz
		k0.S[2].set(0,0, 1);
		k0.S[2].set(1,1,-1);

		matrix H = k0.E + 0.5*deltaOmegaX*k0.S[0] + 0.5*deltaOmegaY*k0.S[1];

		// Transform Sx, Sy, Sz into eigenbasis of H
		matrix V;
		H.diagonalize(V, k0.E);
		for (int i=0; i < 3; i++)
			k0.S[i] = dagger(V) * k0.S[i] * V;
	}

	//Prepare the file header:
	LindbladFile::Header h;
	h.dmuMin = 0;
	h.dmuMax = 0;
	h.Tmax = 1e-1; // in Hartrees; 1e-3 ~ 320 K.
	h.pumpOmegaMax = 0;
	h.probeOmegaMax = 0;
	h.nk = nKpoints;
	h.nkTot = nKpoints;
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
	FILE* fp = fopen("ldbd.dat", "w");
	// --- header
	std::ostringstream oss;
	h.write(oss);
	fwrite(oss.str().data(), 1, h.nBytes(), fp);
	// --- byte offsets
	fwrite(byteOffsets.data(), sizeof(size_t), byteOffsets.size(), fp);
	// --- data for each k-point
	for (int ik=0; ik < nKpoints; ik++)
	{   LindbladFile::Kpoint& k0 = kArray[ik];
		oss.str(std::string());
		k0.write(oss, h);
		fwrite(oss.str().data(), 1, k0.nBytes(h), fp);
	}
	fclose(fp);
	return 0;
}
