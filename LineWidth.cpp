#include "LineWidth.h"
#include <core/Util.h>
#include <core/Units.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>

LineWidth::LineWidth(string inputFilename)
{	std::ifstream ltFile(inputFilename.c_str());
	if(!ltFile.is_open())
		die("Could not open system file '%s' for reading.\n", inputFilename.c_str());
	while(!ltFile.eof())
	{	string line; getline(ltFile, line); //line-by-line processing (comments can now be inline)
		trim(line);
		if(line[0]=='#' || !line.length()) continue; //ignore comments and blank lines
		istringstream iss(line);
		double e, imSig; iss >> e >> imSig;
		switch(imSigma.size())
		{	case 0: Emin = e; break;
			case 1: dE = e-Emin; break;
			default:
				if(round((e-Emin)/dE) != imSigma.size())
					die("Energy grid in '%s' is not uniform\n", inputFilename.c_str());
		}
		imSigma.push_back(imSig);
	}
}

double LineWidth::operator()(double energy) const
{	double x = (energy - Emin) / dE;
	if(x <= 0.) return imSigma.front();
	if(x >= imSigma.size()-1) return imSigma.back();
	int i = floor(x);
	double t = x - i;
	return imSigma[i]*(1-t) + imSigma[i+1]*t;
}
