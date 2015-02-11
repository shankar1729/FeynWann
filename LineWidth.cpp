#include "LineWidth.h"
#include <core/Util.h>
#include <electronic/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>
#include "Units.h"

void readMatrix(matrix& m, string fname, int spinWeight); //declared in BandStruct.cpp

LineWidth::LineWidth(string prefix, const BandStruct& bs) : bs(bs)
{
	//Read e-e scattering:
	eeWannier.init(bs.nBands*bs.nBands, bs.cellMap.size());
	readMatrix(eeWannier, prefix + ".mlwfImSigma_ee", bs.spinWeight);
	
	//Read e-ph scattering:
	ePhWannier.init(bs.nBands*bs.nBands, bs.cellMap.size());
	readMatrix(ePhWannier, prefix + ".mlwfImSigma_ePh", bs.spinWeight);
}

diagMatrix LineWidth::operator()(vector3< double > k) const
{	static StopWatch watch("LineWidth::operator()"); watch.start();
	std::shared_ptr<const BandStruct::CacheEntry> ce = bs.getElectronCache(k);
	matrix ee_k = eeWannier * ce->phase; 
	matrix ePh_k = ePhWannier * ce->phase;
	ee_k.reshape(bs.nBands, bs.nBands);
	ePh_k.reshape(bs.nBands, bs.nBands);
	ee_k = dagger(ce->evecs) * ee_k * ce->evecs; //switch to eigenbasis of Hk
	ePh_k = dagger(ce->evecs) * ePh_k * ce->evecs; //switch to eigenbasis of Hk
	diagMatrix out(bs.nBands);
	for(int b=0; b<bs.nBands; b++)
		out[b] = std::max(0.,ee_k(b,b).real()) + exp(ePh_k(b,b).real()); //e-ph is interpolated logarithmically
	watch.stop();
	return out;
}
