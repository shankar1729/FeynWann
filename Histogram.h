#ifndef WANNIERMETROPOLIS_HISTOGRAM_H
#define WANNIERMETROPOLIS_HISTOGRAM_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <vector>
#include <math.h>
#include <algorithm>

struct Histogram
{
	double Emin, dE, normFac;
	std::vector<double> out;

	Histogram(double Emin, double dE, double Emax);
	void addEvent(double E, double weight);
	void allReduce(MPIUtil::ReduceOp op, bool safeMode=false); //collect over MPI
	void print(string fname, double Escale, double histScale) const; //write to file
};

struct Histogram2D
{
	double Emin, dE, omegaMin, domega, normFac;
	int nE, nomega;
	std::vector<double> out; //nE by nomega with E inner dimension and omega outer

	Histogram2D(double Emin, double dE, double Emax, double omegaMin, double domega, double omegaMax);
	void addEvent(double E, double omega, double weight);
	void allReduce(MPIUtil::ReduceOp op, bool safeMode=false); //collect over MPI
	void print(string fname, double Escale, double omegaScale, double histScale) const;
};

#endif //WANNIERMETROPOLIS_HISTOGRAM_H
