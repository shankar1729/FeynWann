#ifndef WANNIERMETROPOLIS_INTERP1_H
#define WANNIERMETROPOLIS_INTERP1_H

#include <core/Util.h>
#include <cmath>

struct Interp1
{
	std::vector<double> headerVals; //header values for each columns (read from file, if available, but not used by this class)
	std::vector<double> xGrid; //common x values (uniform grid)
	std::vector<std::vector<double> > yGrid; //y values per column
	
	//Read from file which has a single line header, interpolate along columns
	//xScale and yScale allow for unit conversions in the input data
	void init(string fname, double xScale, double yScale);
	
	inline double operator()(int iColumn, double x) const
	{	assert(iColumn<int(yGrid.size()));
		const std::vector<double>& yCol = yGrid[iColumn];
		double fx = dxInv * (x - xMin);
		if(fx <= 0.) return yCol.front();
		if(fx >= yCol.size()-1) return yCol.back();
		int ix = floor(fx); double tx = fx - ix; //find integer and fractional coordinate
		return (1.-tx)*yCol[ix] + tx*yCol[ix+1];
	}
	
	//Single y-column version
	inline double operator()(double x) const
	{	assert(yGrid.size()==1);
		return (*this)(0, x);
	}
	
private:
	double xMin, dxInv; //for speeding up interpolation
};



#endif //WANNIERMETROPOLIS_INTERP1_H