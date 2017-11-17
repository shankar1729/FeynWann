#include "LineWidth.h"
#include <core/Util.h>
#include <core/matrix.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <algorithm>
#include <core/Units.h>

LineWidth::LineWidth(string prefix, const WannierMC& wmc) : wmc(wmc)
{
/*
	nBandsSq = wmc.nBands*wmc.nBands;
	nCells = wmc.cellMap.size();
	
	//Read e-e scattering:
	matrix eeWannier(nBandsSq, nCells);
	readMatrix(eeWannier, prefix + ".mlwfImSigma_ee", wmc.spinWeight);
	
	//Read e-ph scattering:
	matrix ePhWannier(nBandsSq, nCells);
	readMatrix(ePhWannier, prefix + ".mlwfImSigma_ePh", wmc.spinWeight);
	
	//Combine into longer oclumsn for efficient matrix multiply:
	ImSigmaWannier.init(2*nBandsSq, nCells);
	ImSigmaWannier.set(0,nBandsSq, 0,nCells, eeWannier);
	ImSigmaWannier.set(nBandsSq,2*nBandsSq, 0,nCells, ePhWannier);
*/
}

diagMatrix LineWidth::operator()(vector3< double > k, double eeWeight, double ePhWeight) const
{	return (*this)(std::vector<vector3<>>(1, k), eeWeight, ePhWeight)[0];
}

std::vector< diagMatrix > LineWidth::operator()(const std::vector< vector3<> >& kArr, double eeWeight, double ePhWeight) const
{	static StopWatch watch("LineWidth::operator()"); watch.start();
/*
	std::vector< std::shared_ptr<const BandStruct::CacheEntry> > ceArr = wmc.getElectronCache(kArr);
	//Collect phases:
	matrix phase(nCells, kArr.size());
	for(size_t ik=0; ik<kArr.size(); ik++)
		phase.set(0,nCells, ik,ik+1, ceArr[ik]->phase);
	matrix ImSigma_k = ImSigmaWannier * phase;
*/
	std::vector<diagMatrix> out(kArr.size());
/*
	for(size_t ik=0; ik<kArr.size(); ik++)
	{	const matrix& evecs = ceArr[ik]->evecs;
		matrix ee_k = ImSigma_k(0,nBandsSq, ik,ik+1); 
		matrix ePh_k = ImSigma_k(nBandsSq,2*nBandsSq, ik,ik+1);
		ee_k.reshape(wmc.nBands, wmc.nBands); ee_k = dagger(evecs) * ee_k * evecs; //switch to eigenbasis of Hk
		ePh_k.reshape(wmc.nBands, wmc.nBands); ePh_k = dagger(evecs) * ePh_k * evecs; //switch to eigenbasis of Hk
		out[ik].resize(wmc.nBands);
		for(int b=0; b<wmc.nBands; b++)
			out[ik][b] = std::max(0.,eeWeight * ee_k(b,b).real()) + ePhWeight * exp(ePh_k(b,b).real()); //e-ph is interpolated logarithmically
	}
*/
	watch.stop();
	return out;
}
