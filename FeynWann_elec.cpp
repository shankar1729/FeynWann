/*-------------------------------------------------------------------
Copyright 2022 Ravishankar Sundararaman

This file is part of JDFTx.

JDFTx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

JDFTx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with JDFTx.  If not, see <http://www.gnu.org/licenses/>.
-------------------------------------------------------------------*/

#include "FeynWann.h"
#include "FeynWann_internal.h"
#include <core/LatticeUtils.h>


void FeynWann::eLoop(const vector3<>& k0, FeynWann::eProcessFunc eProcess, void* params, const std::vector<bool>* mask)
{	static StopWatch watchCallback("FeynWann::eLoop:callback");
	//Run Fourier transforms with this offset:
	eTransformNeeded(k0);
	//Call eProcess for k-points on present process:
	int ik = Hw->ikStart;
	int ikStop = ik + Hw->nk;
	StateE state;
	PartialLoop3D(offsetDim, ik, ikStop, state.k, k0,
		state.ik = ik;
		state.withinRange = mask ? mask->at(ik) : true;
		setState(state);
		if(state.withinRange)
		{	watchCallback.start();
			eProcess(state, params);
			watchCallback.stop();
		}
	)
}


void FeynWann::eCalc(const vector3<>& k, FeynWann::StateE& e)
{	//Compute Fourier versions for this k:
	eComputeNeeded(k);
	if(fwp.needPhonons) //prepare sum rule quantities
	{	inEphLoop = true;
		Dw->compute(k);
		HePhSumW->compute(k);
	}
	//Prepare state on group head:
	e.ik = 0;
	e.k = k;
	e.withinRange = true;
	if(mpiGroup->isHead()) setState(e);
	inEphLoop = false;
}


void FeynWann::eTransformNeeded(const vector3<>& k0)
{	Hw->transform(k0);
	if(fwp.needVelocity or (fwp.needL or fwp.needQ)) Pw->transform(k0);
	if(fwp.needSpin) Sw->transform(k0);
	if(fwp.needL or fwp.needQ) { RPw->transform(k0); for(int iDir=0; iDir<3; iDir++) HprimeW[iDir]->transform(k0); }
	if(fwp.EzExt) Zw->transform(k0);
	if(fwp.needLinewidth_ee) ImSigma_eeW->transform(k0);
	if(fwp.needLinewidth_ePh) ImSigma_ePhW->transform(k0);
	if(fwp.needLinewidthP_ePh) ImSigmaP_ePhW->transform(k0);
	if(fwp.needLinewidth_D.length()) ImSigma_DW->transform(k0);
	if(fwp.needLinewidthP_D.length()) ImSigmaP_DW->transform(k0);
}


void FeynWann::eComputeNeeded(const vector3<>& k)
{	Hw->compute(k);
	if(fwp.needVelocity or (fwp.needL or fwp.needQ)) Pw->compute(k);
	if(fwp.needSpin) Sw->compute(k);
	if(fwp.needL or fwp.needQ) { RPw->compute(k); for(int iDir=0; iDir<3; iDir++) HprimeW[iDir]->compute(k); }
	if(fwp.EzExt) Zw->compute(k);
	if(fwp.needLinewidth_ee) ImSigma_eeW->compute(k);
	if(fwp.needLinewidth_ePh) ImSigma_ePhW->compute(k);
	if(fwp.needLinewidthP_ePh) ImSigmaP_ePhW->compute(k);
	if(fwp.needLinewidth_D.length()) ImSigma_DW->compute(k);
	if(fwp.needLinewidthP_D.length()) ImSigmaP_DW->compute(k);
}


void FeynWann::setState(FeynWann::StateE& state)
{	static StopWatch watchRotations("FeynWann::setState:rotations");
	//Get and diagonalize Hamiltonian:
	matrix Hk = getMatrix(Hw->getResult(state.ik), nBands, nBands);
	if(fwp.needSpin and fwp.Bext.length_squared())
	{	//Add Zeeman perturbation:
		for(int iDir=0; iDir<3; iDir++)
			if(fwp.Bext[iDir]) Hk += fwp.Bext[iDir] * getMatrix(Sw->getResult(state.ik), nBands, nBands, iDir);
	}
	Hk.diagonalize(state.U, state.E);
	if(fwp.enforceKramerDeg) //enforce Kramers degeneracy in the inversion symmetric cases
	{	int bIn = 0;
		while(bIn < nBands)
		{       double Etemp = state.E[bIn];
				state.E[bIn+1] = Etemp;
				bIn += 2;
		}
	}
	matrix HE = state.U * state.E * dagger(state.U);
	//Add Stark perturbation:
	if(fwp.EzExt) 
	{	HE += fwp.EzExt * getMatrix(Zw->getResult(state.ik), nBands, nBands);
		HE.diagonalize(state.U, state.E);	
	}
	for(double& E: state.E) E -= mu; //reference to Fermi level
	if(fwp.scissor)
	{	//Apply scissor operator (move up unoccupied states):
		for(double& E: state.E)
			if(E > symmThreshold)  // TODO: not use symm threshold here
				E += fwp.scissor;
	}
	//Check whether any states in range (only if not already masked out by initial value of withinRange):
	if(state.withinRange and inEphLoop and (ePhEstart<ePhEstop))
	{	state.withinRange = false;
		for (double& E : state.E)
			if(E >= ePhEstart and E <= ePhEstop)
			{	state.withinRange = true;
				break;
			}
	}
	if(not state.withinRange) return; //Remaining quantities will never be used
	watchRotations.start();
	//Velocity matrix, if needed:
	if(fwp.needVelocity or (fwp.needL or fwp.needQ))
	{	state.vVec.resize(nBands);
		for(int iDir=0; iDir<3; iDir++)
		{	state.v[iDir] = complex(0,-1) //Since P was stored with -i omitted (to make it real when possible)
				* state.getMatrixRotated(Pw, iDir);
			//Extract diagonal parts for convenience:
			for(int b=0; b<nBands; b++)
				state.vVec[b][iDir] = state.v[iDir](b,b).real();
		}
	}
	//Spin matrix, if needed:
	if(fwp.needSpin)
	{	state.Svec.resize(nBands);
		for(int iDir=0; iDir<3; iDir++)
		{	state.S[iDir] = state.getMatrixRotated(Sw, iDir);
			//Extract diagonal parts for convenience:
			for(int b=0; b<nBands; b++)
				state.Svec[b][iDir] = state.S[iDir](b,b).real();
		}
	}
	//R*P matrix elements for angular momentum and/or electric quadrupole:
	if(fwp.needL or fwp.needQ)
	{	matrix3<matrix> RP;
		for(int iDir=0; iDir<3; iDir++)
		{	for(int jDir=0; jDir<3; jDir++)
				RP(iDir, jDir) = complex(0,-1) //Since RP was stored with -i omitted (to make it real when possible)
					* state.getMatrixRotated(RPw, 3*iDir+jDir);
			//Long range correction:
			//--- fetch dH/dk
			matrix iDi = state.getMatrixRotated(HprimeW[iDir]);
			//--- convert to i*D := i dU/dk in place:
			{	complex* iDiData = iDi.data();
				for(int bCol=0; bCol<nBands; bCol++) //note: column major storage
					for(int bRow=0; bRow<nBands; bRow++)
					{	double Ediff = state.E[bCol] - state.E[bRow];
						*(iDiData++) *= complex(0., (fabs(Ediff) < fwp.degeneracyThreshold) ? 0. : 1./Ediff);
					}
			}
			//--- add correction
			for(int jDir=0; jDir<3; jDir++)
				RP(iDir, jDir) += iDi * state.v[jDir];
		}
		//Extract L if needed:
		if(fwp.needL)
		{	for(int kDir=0; kDir<3; kDir++)
			{	int iDir = (kDir + 1) % 3;
				int jDir = (kDir + 2) % 3;
				state.L[kDir] = dagger_symmetrize(RP(iDir, jDir) - RP(jDir, iDir));
			}
		}
		//Extract Q if needed:
		if(fwp.needQ)
		{	//xy, yz and zx components:
			for(int iDir=0; iDir<3; iDir++)
			{	int jDir = (iDir + 1) % 3;
				state.Q[iDir] = dagger_symmetrize(RP(iDir, jDir) + RP(jDir, iDir));
			}
			//xx - r^2/3 and yy - r^2/3 components:
			matrix traceTerm = (1./3) * trace(RP);
			for(int iDir=0; iDir<2; iDir++)
				state.Q[iDir+3] = 2.*dagger_symmetrize(RP(iDir, iDir) - traceTerm);
		}
	}
	//Linewidths, as needed:
	if(fwp.needLinewidth_ee)
		state.ImSigma_ee = diag(state.getMatrixRotated(ImSigma_eeW));
	if(fwp.needLinewidth_ePh)
	{	state.logImSigma_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for(unsigned iMat=0; iMat<state.logImSigma_ePhArr.size(); iMat++)
			state.logImSigma_ePhArr[iMat] = diag(state.getMatrixRotated(ImSigma_ePhW, iMat));
	}
	if(fwp.needLinewidthP_ePh)
	{	state.logImSigmaP_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for(unsigned iMat=0; iMat<state.logImSigmaP_ePhArr.size(); iMat++)
			state.logImSigmaP_ePhArr[iMat] = diag(state.getMatrixRotated(ImSigmaP_ePhW, iMat));
	}
	if(fwp.needLinewidth_D.length())
	{	state.ImSigma_D = diag(state.getMatrixRotated(ImSigma_DW));
		for(double& ImSigma: state.ImSigma_D) ImSigma = exp(ImSigma); //ImSigma_D is interpolated logarithmically
	}
	if(fwp.needLinewidthP_D.length())
	{	state.ImSigmaP_D = diag(state.getMatrixRotated(ImSigmaP_DW));
		for(double& ImSigma: state.ImSigmaP_D) ImSigma = exp(ImSigma); //ImSigmaP_D is interpolated logarithmically
	}
	
	//e-ph sum rule if needed:
	if(inEphLoop)
	{	state.dHePhSum.init(nBands*nBands, 3);
		complex* dHsumData = state.dHePhSum.dataPref();
		for(int iDir=0; iDir<3; iDir++)
		{	matrix D = state.getMatrixRotated(Dw, iDir);
			matrix H = state.getMatrixRotated(HePhSumW, iDir);
			//Compute error in the sum rule:
			const double Emag = 1e-3; //damp correction for energy differences >> Emag (to handle fringes of Wannier window)
			const double expFac = -1./(Emag*Emag);
			complex* Hdata = H.data();
			const complex* Ddata = D.data();
			for(int b2=0; b2<nBands; b2++)
				for(int b1=0; b1<nBands; b1++)
				{	double E12 = state.E[b1] - state.E[b2];
					*Hdata -= (*(Ddata++)) * E12;
					*Hdata *= exp(expFac*E12*E12); //damp correction based on energy difference
					Hdata++;
				}
			//Rotate back to Wannier basis and store to HePhSum
			H = state.U * H * dagger(state.U);
			callPref(eblas_copy)(dHsumData, H.data(), H.nData());
			dHsumData += H.nData();
		}
	}
	watchRotations.stop();
}


void FeynWann::bcastState(FeynWann::StateE& state, MPIUtil* mpiUtil, int root)
{	if(mpiUtil->nProcesses()==1) return; //no communictaion needed
	mpiUtil->bcast(state.ik, root);
	mpiUtil->bcast(&state.k[0], 3, root);
	//Energy and eigenvectors:
	bcast(state.E, nBands, mpiUtil, root);
	mpiUtil->bcast(state.withinRange, root);
	if(not state.withinRange) return; //Remaining quantities will never be used
	bcast(state.U, nBands, nBands, mpiUtil, root);
	//Velocity matrix, if needed:
	if(fwp.needVelocity)
	{	for(int iDir=0; iDir<3; iDir++)
			bcast(state.v[iDir], nBands, nBands, mpiUtil, root);
		state.vVec.resize(nBands);
		mpiUtil->bcastData(state.vVec, root);
	}
	//Spin matrix, if needed:
	if(fwp.needSpin)
	{	for(int iDir=0; iDir<3; iDir++)
			bcast(state.S[iDir], nBands, nBands, mpiUtil, root);
		state.Svec.resize(nBands);
		mpiUtil->bcastData(state.Svec, root);
	}
	//Angular momentum matrix, if needed:
	if(fwp.needL)
	{	for(int iDir=0; iDir<3; iDir++)
			bcast(state.L[iDir], nBands, nBands, mpiUtil, root);
	}
	//Electric quadrupole r*p matrix, if needed:
	if(fwp.needQ)
	{	for(int iComp=0; iComp<3; iComp++)
			bcast(state.Q[iComp], nBands, nBands, mpiUtil, root);
	}
	//Linewidths, if needed:
	if(fwp.needLinewidth_ee) bcast(state.ImSigma_ee, nBands, mpiUtil, root);
	if(fwp.needLinewidth_ePh)
	{	state.logImSigma_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for(diagMatrix& d: state.logImSigma_ePhArr) bcast(d, nBands, mpiUtil, root);
	}
	if(fwp.needLinewidthP_ePh)
	{	state.logImSigmaP_ePhArr.resize(FeynWannParams::fGrid_ePh.size());
		for(diagMatrix& d: state.logImSigmaP_ePhArr) bcast(d, nBands, mpiUtil, root);
	}
	if(fwp.needLinewidth_D.length()) bcast(state.ImSigma_D, nBands, mpiUtil, root);
	if(fwp.needLinewidthP_D.length()) bcast(state.ImSigmaP_D, nBands, mpiUtil, root);
	//e-ph sum rule if needed
	if(inEphLoop)
		bcast(state.dHePhSum, nBands*nBands, 3, mpiUtil, root);
}


//----------- class FeynWann::StateE -------------

inline double interpQuartic(const std::vector<diagMatrix>& Y, int n, double f)
{	//Get bernstein coeffs
	double a0 = Y[0][n];
	double a4 = Y[4][n];
	double a1 = (1./12)*(-13.*Y[0][n]+48.*Y[1][n]-36.*Y[2][n]+16.*Y[3][n]-3.*Y[4][n]);
	double a3 = (1./12)*(-13.*Y[4][n]+48.*Y[3][n]-36.*Y[2][n]+16.*Y[1][n]-3.*Y[0][n]);
	double a2 = (1./18)*(13.*(Y[0][n]+Y[4][n])-64.*(Y[1][n]+Y[3][n])+120.*Y[2][n]);
	//Evaluate bernstein polynomial
	//--- 1
	double b0 = a0+f*(a1-a0);
	double b1 = a1+f*(a2-a1);
	double b2 = a2+f*(a3-a2);
	double b3 = a3+f*(a4-a3);
	//--- 2
	double c0 = b0+f*(b1-b0);
	double c1 = b1+f*(b2-b1);
	double c2 = b2+f*(b3-b2);
	//--- 3
	double d0 = c0+f*(c1-c0);
	double d1 = c1+f*(c2-c1);
	//--- 4
	return d0+f*(d1-d0);
}


double FeynWann::StateE::ImSigma_ePh(int n, double f) const
{	return exp(interpQuartic(logImSigma_ePhArr, n, f));
}


double FeynWann::StateE::ImSigmaP_ePh(int n, double f) const
{	return exp(interpQuartic(logImSigmaP_ePhArr, n, f));
}

matrix FeynWann::StateE::getMatrixRotated(const std::shared_ptr<DistributedMatrix>& mat, int iMat) const
{	int nBands = U.nRows();
	return dagger(U) * getMatrix(mat->getResult(ik), nBands, nBands, iMat) * U;
}
