#include "histogram.h"
#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>


//-------------------------- class histogram ------------------------------------------------

histogram::histogram(double Emin, double dE, double Emax)
: Emin(Emin), dE(dE), normFac(1./(dE*sqrt(2*M_PI))), out(ceil((Emax-Emin)/dE), 0.)
{
}

void histogram::addEvent(double E, double weight)
{	//Gauss smoothed histogram (analagous to whist.oct)
	double iCenter = (E-Emin)/dE;
	int iStart = std::max(0, int(floor(iCenter-5)));
	int iStop = std::min(int(out.size())-1, int(ceil(iCenter+5)));
	for(int i=iStart; i<iStop; i++)
		out[i] += weight * normFac * exp(-0.5*std::pow(i-iCenter,2));
}

void histogram::allReduce(MPIUtil::ReduceOp op, bool safeMode)
{	if(mpiUtil->nProcesses()>1)
		mpiUtil->allReduce(out.data(), out.size(), op, safeMode);
}

void histogram::print(string fname, double Eunit) const
{	ofstream ofs(fname.c_str());
	for(size_t i=0; i<out.size(); i++)
		ofs << (Emin+i*dE)/Eunit << "\t" << out[i]*Eunit << '\n';
}
