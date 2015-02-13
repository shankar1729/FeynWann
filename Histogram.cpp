#include "Histogram.h"
#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

//------------- class Histogram --------------

Histogram::Histogram(double Emin, double dE, double Emax)
: Emin(Emin), dE(dE), normFac(1./(sqrt(2*M_PI)*dE)), out(int(ceil((Emax-Emin)/dE)), 0.)
{
}

void Histogram::addEvent(double E, double weight)
{	//Gauss smoothed Histogram (analagous to whist.oct)
	double iCenter = (E-Emin)/dE;
	int iStart = std::max(0, int(floor(iCenter-5)));
	int iStop = std::min(int(out.size()), 1+int(ceil(iCenter+5)));
	for(int i=iStart; i<iStop; i++)
		out[i] += weight * normFac * exp(-0.5*std::pow(i-iCenter,2));
}

void Histogram::allReduce(MPIUtil::ReduceOp op, bool safeMode)
{	if(mpiUtil->nProcesses()>1)
		mpiUtil->allReduce(out.data(), out.size(), op, safeMode);
}

void Histogram::print(string fname, double Escale, double histScale) const
{	if(!mpiUtil->isHead()) return;
	ofstream ofs(fname.c_str());
	for(size_t i=0; i<out.size(); i++)
		ofs << (Emin+i*dE)*Escale << "\t" << out[i]*histScale << '\n';
}

//------------- class Histogram2D --------------

Histogram2D::Histogram2D(double Emin, double dE, double Emax, double omegaMin, double domega, double omegaMax)
: Emin(Emin), dE(dE), omegaMin(omegaMin), domega(domega), normFac(1./(2*M_PI*dE*domega)),
nE(int(ceil((Emax-Emin)/dE))), nomega(int(ceil((omegaMax-omegaMin)/domega))), out(nE*nomega, 0.)
{
}

void Histogram2D::addEvent(double E, double omega, double weight)
{	//Gauss smoothed 2D Histogram
	//--- E coordinate:
	double eCenter = (E-Emin)/dE;
	int eStart = std::max(0, int(floor(eCenter-5)));
	int eStop = std::min(nE, 1+int(ceil(eCenter+5)));
	//--- omega coordinate:
	double oCenter = (omega-omegaMin)/domega;
	int oStart = std::max(0, int(floor(oCenter-5)));
	int oStop = std::min(nomega, 1+int(ceil(oCenter+5)));
	//--- accumulate normalized Gaussian:
	for(int o=oStart; o<oStop; o++)
		for(int e=eStart; e<eStop; e++)
			out[o*nE+e] += weight * normFac
				* exp(-0.5 * (std::pow(e-eCenter,2) + std::pow(o-oCenter,2)));
}

void Histogram2D::allReduce(MPIUtil::ReduceOp op, bool safeMode)
{	if(mpiUtil->nProcesses()>1)
		mpiUtil->allReduce(out.data(), out.size(), op, safeMode);
}

void Histogram2D::print(string fname, double Escale, double omegaScale, double histScale) const
{	if(!mpiUtil->isHead()) return;
	ofstream ofs(fname.c_str());
	//Print in octave/matlab mat format for ease:
	//--- E grid:
	ofs << "# name: E\n";
	ofs << "# type: matrix\n";
	ofs << "# rows: " << nE << "\n";
	ofs << "# columns: 1\n";
	for(int e=0; e<nE; e++)
		ofs << (Emin+e*dE)*Escale << "\n";
	ofs <<"\n";
	//--- omega grid:
	ofs << "# name: omega\n";
	ofs << "# type: matrix\n";
	ofs << "# rows: " << nomega << "\n";
	ofs << "# columns: 1\n";
	for(int o=0; o<nomega; o++)
		ofs << (omegaMin+o*domega)*omegaScale << "\n";
	ofs <<"\n";
	//--- Histogram data:
	ofs << "# name: histData\n";
	ofs << "# type: matrix\n";
	ofs << "# rows: " << nE << "\n";
	ofs << "# columns: " << nomega << "\n";
	for(int e=0; e<nE; e++)
	{	for(int o=0; o<nomega; o++)
		{	if(o) ofs << ' ';
			ofs << out[o*nE+e] * histScale;
		}
		ofs <<"\n";
	}
}
