#ifndef WANNIERMETROPOLIS_HISTOGRAM_H
#define WANNIERMETROPOLIS_HISTOGRAM_H

#include <core/Util.h>
#include <electronic/matrix.h>
#include <vector>
#include <math.h>
#include <algorithm>

class Histogram
{
private:
	double Emin, dE, normFac;
	std::vector<double> out;
public:
	Histogram(double Emin, double dE, double Emax);
	void addEvent(double E, double weight);
	void allReduce(MPIUtil::ReduceOp op, bool safeMode=false); //collect over MPI
	void print(string fname, double Eunit=1.) const; //write to file (with optional unit conversion)
};
#endif //WANNIERMETROPOLIS_HISTOGRAM_H
