/*-------------------------------------------------------------------
Copyright 2021 Ravishankar Sundararaman

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

#ifndef FEYNWANN_LINDBLAD_H
#define FEYNWANN_LINDBLAD_H

#include <core/matrix.h>
#include <lindblad/LindbladFile.h>
#include <Integrator.h>
#include <BlockCyclicMatrix.h>

#ifdef PETSC_ENABLED
	#include <petsc.h>
	
	//Slightly more graceful wrapper to CHKERRQ() macro from Petsc:
	#define CHECKERR(codeLine) \
		{	PetscInt iErr = codeLine; \
			CHKERRQ(iErr); \
		}
#else
	#define PetscErrorCode int
	#define CHECKERR(codeLine) codeLine;
#endif


//! Handling of valley degrees of freedom
enum ValleyMode
{	ValleyNone, //!< no special handling of valley
	ValleyInter, //!< only include inter-valley processes
	ValleyIntra //!< only include intra-valley processes
};


//Input parameters controlling Lindblad dynamics
struct LindbladParams
{
	double dmu; //!< Fermi level position relative to neutral value / VBM
	double T; //!< Temperature
	
	double dt, tStop; //!< Time evolution reprtin interval and end time
	double tStep; //!< Explicit time evolution step size (fixed integrator)
	double tolAdaptive; //!< Adaptive integrator relative tolerance

	double pumpOmega, pumpA0, pumpTau; //!< pump frequency, amplitude and width
	vector3<complex> pumpPol; //!< pump polarization
	bool pumpEvolve; //!< whether pump is explicitly evolved in time

	bool pumpBfield; //!< whether the "pump" is a magnetic field initialization
	vector3<> pumpB; //!< initialization magnetic field

	double omegaMin, domega, omegaMax; //!< probe frequency grid
	double tau; //!< probe width
	std::vector<vector3<complex>> pol; //!< probe polarizations
	double dE; //!< energy resolution for distribution functions

	bool linearized; //!< Whether dynamics is linearized
	bool spectrumMode; //!< Time-evolution spectrum if yes, dynamics otherwise
	int blockSize; //!< block size in ScaLAPACK matrix for spectrum mode only
	#ifdef SCALAPACK_ENABLED
	BlockCyclicMatrix::DiagMethod diagMethod;
	#endif

	bool ePhEnabled; //!< whether e-ph coupling is enabled
	double defectFraction; //!< defect fraction if present
	ValleyMode valleyMode; //!< whether all k-pairs (None) or only those corresponding to Intra/Inter-valley scattering are included
	bool verbose; //!< whether to print more detailed stats during evolution
	string inFile; //!< file name to get lindblad data from
	string checkpointFile; //!< file name to save checkpoint data to
	string evecFile; //!< filename to write eigenvectors to in spectrum mode

	//---- Dependent variables computed from above ----
	double invT; //inverse temperature
	double nomega; //number of probe frequencies

	//! Set dependent variables
	void initialize() 
	{	invT = 1./T;
		nomega = 1 + int(round((omegaMax-omegaMin)/domega));
	}
};


// Base class of all lindblad dynamics (implemented in Lindblad.cpp)
class Lindblad : public Integrator<DM1>
{
protected:
	const LindbladParams& lp;
	int stepID; //current time and reporting step number
	
	bool spinorial; //!< whether spin is available
	int spinWeight; //!< weight of spin in BZ integration
	matrix3<> R; double Omega; //!< lattice vectors and unit cell volume

	size_t nk, nkTot; //!< number of selected k-points overall and original total k-points effectively used in BZ sampling
	size_t ikStart, ikStop, nkMine; //!< range and number of selected k-points on this process
	TaskDivision kDivision;
	inline bool isMine(size_t ik) const { return kDivision.isMine(ik); } //!< check if k-point index is local
	inline int whose(size_t ik) const { return kDivision.whose(ik); } //!< find out which process (in mpiWorld) this k-point belongs to

	struct State : LindbladFile::Kpoint
	{	int innerStop; //end of active inner window range (relative to outer window)
		diagMatrix rho0; //equilibrium / initial density matrix (diagonal)
		matrix pumpPD; //P matrix elements at pump polarization x energy conservation delta (D), but without A0 and time factor
	};
	std::vector<State> state; //!< all information read from lindbladInit output (e and e-ph properties) + extra local variables above
	std::vector<int> nInnerAll; //!< nInner for all k-points on all processes
	double Emin, Emax; //!< energy range of active space across all k (for spin and number density output)
	
	std::vector<double> Eall; //!< inner window energies for all k (only needed and initialized when ePhEnabled)
	std::vector<size_t> nInnerPrev; //!< cumulative nInner for each k, which is the offset into the Eall array for each k
	double tPrev; //last time at which compute() was called; used internally to update e-ph operator phases
	
	const vector3<> K, Kp; //!< K and K' valley in reciprocal lattice coordinates
	static inline vector3<> wrap(const vector3<>& x); //!< Wrap fratcional coordinates to fundamental interval
	inline bool isKvalley(vector3<> k) const { return (wrap(K-k)).length_squared() < (wrap(Kp-k)).length_squared(); }

public:
	Lindblad(const LindbladParams& lp);
	virtual ~Lindblad();
	DM1 compute(double t, const DM1& v); //specify differential equation for time evolution
	void report(double t, const DM1& v) const; //called by integrator for periodic reporting
};


//Full nonlinear implementation of real-time Lindblad dynamics (in LindbladNonlinear.cpp)
class LindbladNonlinear : public Lindblad
{
public:
	LindbladNonlinear(const LindbladParams& lp);
	virtual ~LindbladNonlinear();
};


//Base class of lindblad implementations with explicit time-evolution matrix (in LindbladMatrix.cpp)
class LindbladMatrix : public Lindblad
{
	PetscErrorCode initialize();
public:
	LindbladMatrix(const LindbladParams& lp);
	virtual ~LindbladMatrix();
};


//Linearized real-time Lindblad dynamics (in LindbladLinear.cpp)
class LindbladLinear : public LindbladMatrix
{
public:
	LindbladLinear(const LindbladParams& lp);
	virtual ~LindbladLinear();
};


//Diagonalization of Lindblad superoperator to get time evolution spectrum (in LindbladSpectrum.cpp)
class LindbladSpectrum : public LindbladMatrix
{
public:
	LindbladSpectrum(const LindbladParams& lp);
	virtual ~LindbladSpectrum();
};



//----- Inline function implementations -----

inline vector3<> Lindblad::wrap(const vector3<>& x)
{	vector3<> result = x;
	for(int dir=0; dir<3; dir++)
		result[dir] -= floor(0.5 + result[dir]);
	return result;
}

#endif

