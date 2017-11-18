#include <core/Util.h>
#include <core/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include "BandStruct.h"
#include "InputMap.h"
#include <core/Units.h>

int main(int argc, char** argv)
{	
	InitParams ip = BandStruct::initialize(argc, argv, "Test Wannier bandstructure");

	
	//Read k-points from input:
	std::vector< vector3<> > kpoints;
	while(!std::cin.eof())
	{	string line; getline(std::cin, line);
		istringstream iss(line);
		string cmd; iss >> cmd;
		if(cmd != "kpoint") continue;
		vector3<> k; iss >> k[0] >> k[1] >> k[2];
		kpoints.push_back(k);
	}
	std::cout << "Read " << kpoints.size() << " kpoints\n";

	//Initialize Wannier band structure:
	std::vector< vector3<complex> > Ahat(3);
	Ahat[0] = vector3<complex>(1., 0., 0.);
	Ahat[1] = vector3<complex>(0., 1., 0.);
	Ahat[2] = vector3<complex>(0., 0., 1.);
	BandStruct bs("Wannier/totalE", "Wannier/wannier", false, Ahat);

	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");

	//Band eigenvalues:
	FILE* fp = fopen("WannierBandstruct.eigenvals","w");
	for(vector3<> k: kpoints)
		bs.getStates(k).print(fp);
	fclose(fp);
	
	//Velocities vs momentum matrix elements:
	std::ofstream ofs("WannierBandstruct.VvsP");
	for(vector3<> k: kpoints)
	{	diagMatrix E = bs.getStates(k);
		std::vector<vector3<> > v = bs.getVelocity(k);
		std::vector<matrix> P = bs.getDipoleMatElem(k);
		for(unsigned b=0; b<v.size(); b++)
		{	for(int dir=0; dir<3; dir++)
				ofs << E[b] << '\t' << v[b][dir] << '\t' << -P[dir](b,b).imag() << '\n';
			ofs << '\n';
		}
	}
	ofs.close();
	
	finalizeSystem();
	return 0;
}
