#include <sstream>
#include <core/scalar.h>
#include <core/string.h>
#include <core/Util.h>
#include <electronic/matrix.h>
#include <fstream>
#include <vector>
#include <math.h>
#include "Epsilon.h"
#include "Units.h"

int main(int argc, char** argv)
{        initSystem(argc, argv);
	
	double Eplasmon = 2.8*eV;
	Epsilon eps("epsilon.txt");
	double omega_p = 9.03*eV;
	std::vector<complex> epsilons;
        std::vector<double> omegaExp, omegaIm;
	char fname[256];
        sprintf(fname, "Distrib-%.1lfeV-phonon.dat", Eplasmon/eV);
	string ffname = string("expWidthAndEpsilon")+fname;
        ofstream ofs(ffname.c_str());
        for (int ii = 0; ii<1000; ii++)
        {       double omegaE = 0.1*eV + ii*(omega_p - 0.1*eV)/1000;
                eps.setFrequency(omegaE);
                omegaExp.push_back(omegaE);
                complex arg = eps.epsilon / (eps.epsilon + 1);
                double argg = fabs(sin(0.5*atan2(imag(arg),real(arg)))) * omegaE;
                omegaIm.push_back(argg);
		epsilons.push_back(eps.epsilon);
                ofs << omegaExp[ii]/eV  << "\t" << omegaIm[ii]/eV  << "\t" << real(epsilons[ii]) << "\t" << imag(epsilons[ii]) << '\n';
        }

        logPrintf("\nHello world from process %d of %d using JDFTx!\n\n", mpiUtil->iProcess(), mpiUtil->nProcesses());
	finalizeSystem();
        return 0;
}
