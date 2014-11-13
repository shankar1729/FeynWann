#include "lifeTime.h"
#include <core/Util.h>
#include <core/Units.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>
// -------------------------------------------- class lifeTime --------------------------------------------

// Constructor
lifeTime::lifeTime(string inputFilename)
{	//Get the epsilon parameters
	std::ifstream ltFile(inputFilename.c_str());
	if(!ltFile.is_open())
		die("Could not open system file '%s' for reading.\n", inputFilename.c_str());
	while(!ltFile.eof())
	{	string line; getline(ltFile, line); //line-by-line processing (comments can now be inline)
		trim(line);
		if(line[0]=='#' || !line.length()) continue; //ignore comments and blank lines
		istringstream iss(line);
		double e, imSig; iss >> e >> imSig;
		E.push_back(e);
		imSigma.push_back(imSig);
	}
	logPrintf("\n");
	ltFile.close();
}

double lifeTime::get_lifeTime(double energy)
{	int up = std::upper_bound (E.begin(), E.end(), energy) - E.begin();
	double E1 = E[up-1], E2 = E[up], imSig1 = imSigma[up-1], imSig2 = imSigma[up];
	double a = (imSig2 - imSig1) / (E2 - E1);
	double b = -a*E1 + imSig1;
	double y = a * energy + b;
	return y;
}
