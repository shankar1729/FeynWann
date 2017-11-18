#ifndef WANNIERMC_HISTOGRAM_H
#define WANNIERMC_HISTOGRAM_H

#include <core/Util.h>
#include <core/matrix.h>
#include <vector>
#include <math.h>
#include <algorithm>

struct Histogram
{
	double Emin, dE, dEinv;
	int nE;
	std::vector<double> out;

	Histogram(double Emin, double dE, double Emax);
	void addEvent(double E, double weight);
	void allReduce(MPIUtil::ReduceOp op, bool safeMode=false); //collect over MPI
	void print(string fname, double Escale, double histScale) const; //write to file
};

struct Histogram2D
{
	double Emin, dE, dEinv, omegaMin, domega, domegaInv;
	int nE, nomega;
	std::vector<double> out; //nE by nomega with E inner dimension and omega outer

	Histogram2D(double Emin, double dE, double Emax, double omegaMin, double domega, double omegaMax);
	void addEvent(double E, double omega, double weight);
	void allReduce(MPIUtil::ReduceOp op, bool safeMode=false); //collect over MPI
	void print(string fname, double Escale, double omegaScale, double histScale) const;
	
	Histogram2D(string fname, double Escale, double omegaScale, double histScale); //read back histogram written using print
	double interp1(double E, double omega) const; //return interpolated value
};

#endif //WANNIERMC_HISTOGRAM_H
