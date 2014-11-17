#include <sstream>
#include <core/scalar.h>
#include <core/string.h>
#include <core/Units.h>
#include <core/Util.h>
#include <electronic/matrix.h>
#include <fstream>
#include <vector>
#include <math.h>
#include "Epsilon.h"

int main(int argc, char** argv)
{        initSystem(argc, argv);
	
	Epsilon eps("epsilon.txt");
	double omega_p = 9.03*eV;
        std::vector<double> omegaExp, omegaIm;
        for (int ii = 0; ii<1000; ii++)
        {       double omegaE = 0.1*eV + ii*(omega_p - 0.1*eV)/1000;
                eps.setFrequency(omegaE);
                omegaExp.push_back(omegaE);
                complex arg = eps.epsilon / (eps.epsilon + 1);
                double argg = fabs(sin(0.5*atan2(real(arg),imag(arg)))) * omegaE;
                omegaIm.push_back(argg);
        }
        string ffname = string("expWidthAndEpsilon")+fname;
        ofstream ofs(ffname.c_str());
        for(size_t i=0; i<omegaExp.size(); i++)
                ofs << omegaExp[i]/eV  << "\t" << omegaIm[i]/eV  << "\t" << fabs(real(eps.epsilon)) << "\t" << imag(eps.epsilon)<< '\n';

        logPrintf("\nHello world from process %d of %d using JDFTx!\n\n", mpiUtil->iProcess(), mpiUtil->nProcesses());
	finalizeSystem();
        return 0;
}
