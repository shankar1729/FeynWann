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
	nBandsSq = bs.nBands*bs.nBands;
	nCells = bs.cellMap.size();
	
	//Read e-e scattering:
	matrix eeWannier(nBandsSq, nCells);
	readMatrix(eeWannier, prefix + ".mlwfImSigma_ee", bs.spinWeight);
	
	//Read e-ph scattering:
	matrix ePhWannier(nBandsSq, nCells);
	readMatrix(ePhWannier, prefix + ".mlwfImSigma_ePh", bs.spinWeight);
	
	//Combine into longer oclumsn for efficient matrix multiply:
	ImSigmaWannier.init(2*nBandsSq, nCells);
	ImSigmaWannier.set(0,nBandsSq, 0,nCells, eeWannier);
	ImSigmaWannier.set(nBandsSq,2*nBandsSq, 0,nCells, ePhWannier);
}

diagMatrix LineWidth::operator()(vector3< double > k) const
{	static StopWatch watch("LineWidth::operator()"); watch.start();
	std::shared_ptr<const BandStruct::CacheEntry> ce = bs.getElectronCache(k);
	matrix ImSigma_k = ImSigmaWannier * ce->phase;
	matrix ee_k = ImSigma_k(0,nBandsSq, 0,1); 
	matrix ePh_k = ImSigma_k(nBandsSq,2*nBandsSq, 0,1);
	ee_k.reshape(bs.nBands, bs.nBands); ee_k = dagger(ce->evecs) * ee_k * ce->evecs; //switch to eigenbasis of Hk
	ePh_k.reshape(bs.nBands, bs.nBands); ePh_k = dagger(ce->evecs) * ePh_k * ce->evecs; //switch to eigenbasis of Hk
	diagMatrix out(bs.nBands);
	for(int b=0; b<bs.nBands; b++)
		out[b] = std::max(0.,ee_k(b,b).real()) + exp(ePh_k(b,b).real()); //e-ph is interpolated logarithmically
	watch.stop();
	return out;
}
