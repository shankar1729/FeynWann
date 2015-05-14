#include <core/Util.h>
#include <electronic/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/WignerSeitz.h>
#include "BandStruct.h"
#include "InputMap.h"
#include "Units.h"

int main(int argc, char** argv)
{	string inputFilename; bool dryRun, printDefaults;
	initSystemCmdline(argc, argv, "Monte Carlo estimate of resistivity", inputFilename, dryRun, printDefaults);

	//Get the system parameters (mu, T, lattice vectors etc.)
	InputMap inputMap(inputFilename);
	const int nKptsN1 = inputMap.get("nKptsN1");
	const int totalBlocks = inputMap.get("totalBlocks"); assert(totalBlocks>0);
	const double mu = inputMap.get("mu");
	const double T = inputMap.get("T") * eV;
	const int spinWeight = round(inputMap.get("spinWeight"));
	const matrix3<> R = matrix3<>(0,1,1, 1,0,1, 1,1,0) * (0.5*inputMap.get("aCubic")*Angstrom);

	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("nKptsN1 = %d\n", nKptsN1);
	logPrintf("totalBlocks = %d\n", totalBlocks);
	logPrintf("mu = %lg\n", mu);
	logPrintf("T = %lg\n", T);
	logPrintf("spinWeight = %d\n", spinWeight);
	logPrintf("R:\n");
	R.print(globalLog, " %lg ");
	if(dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		finalizeSystem();
		return 0;
	}
	logPrintf("\n");
	
	//Initialize Wannier bandstructure:
	std::vector< vector3<complex> > Ahat(3); //use Cartesian basis
	for(int iDir=0; iDir<3; iDir++)
		Ahat[iDir][iDir] = 1.;
	BandStruct bs("Wannier/wannier", mu, spinWeight, string(), Ahat);
	
	//finite-difference for secodn derivatibe
	const double dkMag = 1e-4;
	std::vector< vector3<> > dk(3);
	for(int iDir=0; iDir<3; iDir++)
	{	dk[iDir][iDir] = dkMag;
		dk[iDir] = (1./(2*M_PI)) * dk[iDir] * R; //convert to reciprocal lattice coordinates
	}
	
	//Monte Carlo sample BZ
	double m1sum = 0., m1sumSq = 0;
	double m2sum = 0., m2sumSq = 0;
	int blockStart = (totalBlocks * (mpiUtil->iProcess())) / mpiUtil->nProcesses(); //MPI division
	int blockStop = (totalBlocks * (mpiUtil->iProcess()+1)) / mpiUtil->nProcesses();
	int nKpts = nKptsN1/totalBlocks;
	const double Emax = 10*T; //max energy from Fermi level to consider
	for(int block=blockStart; block<blockStop; block++)
	{	Random::seed(block);
		double wSum = 0., wm1sum=0., wm2sum=0.;
		for(int ik=0; ik<nKpts; ik++)
		{	//Generate a random k-point with a relevant state:
			vector3<> k;
			diagMatrix E;
			while(true)
			{	for(int j=0; j<3; j++)
					k[j] = Random::uniform();
				E = bs.getStates(k, Emax);
				bool worthwhile = false;
				for(int b=0; b<E.nRows(); b++)
					if(fabs(E[b]) < Emax)
					{	worthwhile = true;
						break;
					}
				if(worthwhile) break;
			}
			std::vector< vector3<> > v = bs.getVelocity(k, R, Emax);
			std::vector<matrix> Pmat = bs.getDipoleMatElem(k);
			//Calculate inertial mass tensor trace:
			std::vector< matrix3<> > m2inv(v.size());
// 			for(int j=0; j<3; j++)
// 			{	std::vector< vector3<> > vp = bs.getVelocity(k+dk[j], R, Emax);
// 				std::vector< vector3<> > vm = bs.getVelocity(k-dk[j], R, Emax);
// 				for(int b=0; b<E.nRows(); b++)
// 					m2inv[b].set_col(j, (1./(2*dkMag)) * (vp[b] - vm[b]));
// 			}
			std::vector< vector3<> > kArr(12, k);
			for(int j=0; j<3; j++)
			{	int j2 = (j+1)%3;
				int j3 = (j+2)%3;
				kArr[0+4*j] += dk[j];
				kArr[1+4*j] -= dk[j];
				kArr[2+4*j] += dk[j2] + dk[j3];
				kArr[3+4*j] -= dk[j2] + dk[j3];
			}
			std::vector<diagMatrix> Earr = bs.getStates(kArr, Emax);
			double denFac = 1./(dkMag*dkMag);
			for(int j=0; j<3; j++)
				for(int b=0; b<E.nRows(); b++)
					m2inv[b](j,j) = denFac * (Earr[0+4*j][b] + Earr[1+4*j][b] - 2*E[b]);
			for(int j=0; j<3; j++)
			{	int j2 = (j+1)%3;
				int j3 = (j+2)%3;
				for(int b=0; b<E.nRows(); b++)
				{	double offDiag = 0.5*(denFac*(Earr[2+4*j][b] + Earr[3+4*j][b] - 2*E[b]) - (m2inv[b](j2,j2)+ m2inv[b](j3,j3)));
					m2inv[b](j2,j3) = offDiag;
					m2inv[b](j3,j2) = offDiag;
				}
			}
			
			for(int b=0; b<E.nRows(); b++) if(fabs(E[b])<Emax)
			{	double w = -1/(T*std::pow(2*cosh(E[b]/(2*T)),2));
				wSum += w;
				//m1: p/v ratio
				vector3<> p; for(int j=0; j<3; j++) p[j] = Pmat[j](b,b).imag();
				double m1 = p.length() / v[b].length();
				wm1sum += w * m1;
				//m1: inertial mass tensor d^2E/dk^2
				wm2sum += w * trace(inv(m2inv[b]))/3;
			}
		}
		double m1 = wm1sum / wSum;
		m1sum += m1;
		m1sumSq += m1*m1;
		double m2 = wm2sum / wSum;
		m2sum += m2;
		m2sumSq += m2*m2;
	}
	logPrintf("\n");
	
	mpiUtil->allReduce(m1sum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(m1sumSq, MPIUtil::ReduceSum);
	double m1 = m1sum / totalBlocks;
	double m1std = sqrt(m1sumSq/totalBlocks - m1*m1)/sqrt(totalBlocks);
	logPrintf("p/v ratio mass: %le +/- %le\n", m1, m1std);
	
	mpiUtil->allReduce(m2sum, MPIUtil::ReduceSum);
	mpiUtil->allReduce(m2sumSq, MPIUtil::ReduceSum);
	double m2 = m2sum / totalBlocks;
	double m2std = sqrt(m2sumSq/totalBlocks - m2*m2)/sqrt(totalBlocks);
	logPrintf("inertial mass:  %le +/- %le\n", m2, m2std);
	
	logPrintf("\n");
	finalizeSystem();
}