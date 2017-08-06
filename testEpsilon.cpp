#include <sstream>
#include <core/scalar.h>
#include <core/string.h>
#include <core/Util.h>
#include <electronic/matrix.h>
#include <fstream>
#include <vector>
#include <math.h>
#include "Epsilon.h"
#include <core/Units.h>

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Test dielectric function parametrization", inputFilename, dryRun, printDefaults);

	Epsilon eps("Wannier/epsilon.dat");
	
	ofstream ofs("testEpsilon.out");
	for(double omega=0.01*eV; omega<10*eV; omega+=0.01*eV)
	{	eps.setFrequency(omega, false);
		ofs << eps.omega << '\t' << eps.epsilon.real() << '\t' << eps.epsilon.imag() << '\t' << eps.exptLinewidth() << '\n';
	}
	ofs.close();

	finalizeSystem();
	return 0;
}
