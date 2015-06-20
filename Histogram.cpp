#include "Histogram.h"
#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>

//------------- class Histogram --------------

Histogram::Histogram(double Emin, double dE, double Emax)
: Emin(Emin), dE(dE), dEinv(1./dE), nE(int(ceil((Emax-Emin)/dE))), out(nE, 0.)
{
}

void Histogram::addEvent(double E, double weight)
{	//Linear splined histogram
	//--- E coordinate:
	double eCenter = (E-Emin)*dEinv;
	int ie = floor(eCenter);
	if(ie<0 || ie+1>=nE) return;
	double te = eCenter - ie;
	//--- accumulate normalized linear spline:
	double prefac = dEinv * weight;
	out[ ie ] += prefac * (1.-te);
	out[ie+1] += prefac * te;
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
: Emin(Emin), dE(dE), dEinv(1./dE), omegaMin(omegaMin), domega(domega), domegaInv(1./domega),
nE(int(ceil((Emax-Emin)/dE))), nomega(int(ceil((omegaMax-omegaMin)/domega))), out(nE*nomega, 0.)
{
}

void Histogram2D::addEvent(double E, double omega, double weight)
{	//Linear splined 2D Histogram
	//--- E coordinate:
	double eCenter = (E-Emin)*dEinv;
	int ie = floor(eCenter);
	if(ie<0 || ie+1>=nE) return;
	double te = eCenter - ie;
	//--- omega coordinate:
	double oCenter = (omega-omegaMin)*domegaInv;
	int io = floor(oCenter);
	if(io<0 || io+1>=nomega) return;
	double to = oCenter - io;
	//--- accumulate normalized linear spline:
	double prefac = dEinv * domegaInv * weight;
	double eContrib0 = prefac * (1.-te);
	double eContrib1 = prefac * te;
	out[( io )*nE+( ie )] += eContrib0 * (1.-to);
	out[( io )*nE+(ie+1)] += eContrib1 * (1.-to);
	out[(io+1)*nE+( ie )] += eContrib0 * to;
	out[(io+1)*nE+(ie+1)] += eContrib1 * to;
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
