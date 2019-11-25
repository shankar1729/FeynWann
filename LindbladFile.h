/*-------------------------------------------------------------------
Copyright 2019 Ravishankar Sundararaman

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

#ifndef FEYNWANN_LINDBLADFILE_H
#define FEYNWANN_LINDBLADFILE_H

#include <core/MPIUtil.h>
#include "SparseMatrix.h"

//! Structures stored into sparse lindblad files
namespace LindbladFile
{
	static const size_t markerLen = 4; //Length of section markers in file
	
	//! Global file header
	struct Header
	{	static constexpr const char* marker = "LDBD";
		double dmuMin, dmuMax, Tmax, pumpOmegaMax, probeOmegaMax; //mu, T and pump/probe frequency range accounted for
		size_t nk, nkTot; //number of selected k-points and original total k-points (1/nkTot is BZ integration weight)
		bool ePhEnabled, spinorial; //whether e-ph and spinorial info are available
		int spinWeight; //spin factor in BZ integration
		matrix3<> R; //unit cell lattice vectors
		
		size_t nBytes() const
		{	return sizeof(char)*markerLen + sizeof(double)*5 + sizeof(size_t)*2 + sizeof(bool)*2 + sizeof(int) + sizeof(matrix3<>);
		}
		void write(MPIUtil::File fp, const MPIUtil* mpiUtil) const
		{	mpiUtil->fwrite(marker, sizeof(char), markerLen, fp);
			mpiUtil->fwrite(&dmuMin, sizeof(double), 5, fp);
			mpiUtil->fwrite(&nk, sizeof(size_t), 2, fp);
			mpiUtil->fwrite(&ePhEnabled, sizeof(bool), 2, fp);
			mpiUtil->fwrite(&spinWeight, sizeof(int), 1, fp);
			mpiUtil->fwrite(&R, sizeof(matrix3<>), 1, fp);
		}
		void read(MPIUtil::File fp, const MPIUtil* mpiUtil)
		{	//Read and check marker:
			char markerIn[markerLen];
			mpiUtil->fread(markerIn, sizeof(char), markerLen, fp);
			if(strncmp(markerIn, marker, markerLen) != 0)
			{	fprintf(stderr, "File format error: could not find LDBD header.\n");
				mpiUtil->exit(1);
			}
			//Read data:
			mpiUtil->fread(&dmuMin, sizeof(double), 5, fp);
			mpiUtil->fread(&nk, sizeof(size_t), 2, fp);
			mpiUtil->fread(&ePhEnabled, sizeof(bool), 2, fp);
			mpiUtil->fread(&spinWeight, sizeof(int), 1, fp);
			mpiUtil->fread(&R, sizeof(matrix3<>), 1, fp);
		}
	};
	
	//! E-ph coupling to a specific k and for a phonon mode
	struct GePhEntry
	{	static constexpr const char* marker = "GEPH";
		size_t jk; //index of second k-point
		double omegaPh; //phonon frequency
		SparseMatrix G; //e-ph matrix elements
		
		size_t nBytes() const
		{	return sizeof(char)*markerLen + sizeof(size_t) + sizeof(double)
				+ sizeof(size_t)+sizeof(SparseEntry)*G.size(); //storage for G.size() and then its entries
		}
		void write(MPIUtil::File fp, const MPIUtil* mpiUtil) const
		{	mpiUtil->fwrite(marker, sizeof(char), markerLen, fp);
			mpiUtil->fwrite(&jk, sizeof(size_t), 1, fp);
			mpiUtil->fwrite(&omegaPh, sizeof(double), 1, fp);
			size_t Gsize = G.size();
			mpiUtil->fwrite(&Gsize, sizeof(size_t), 1, fp);
			mpiUtil->fwriteData(G, fp);
		}
		void read(MPIUtil::File fp, const MPIUtil* mpiUtil)
		{	//Read and check marker:
			char markerIn[markerLen];
			mpiUtil->fread(markerIn, sizeof(char), markerLen, fp);
			if(strncmp(markerIn, marker, markerLen) != 0)
			{	fprintf(stderr, "File format error: could not find GEPH header.\n");
				mpiUtil->exit(1);
			}
			//Read data:
			mpiUtil->fread(&jk, sizeof(size_t), 1, fp);
			mpiUtil->fread(&omegaPh, sizeof(double), 1, fp);
			size_t Gsize;
			mpiUtil->fread(&Gsize, sizeof(size_t), 1, fp);
			G.resize(Gsize);
			mpiUtil->freadData(G, fp);
		}
		
		//For searching partner lists:
		inline bool operator<(const size_t jk2) const
		{	return jk < jk2;
		}
	};
	
	//! K-point header
	struct Kpoint
	{	static constexpr const char* marker = "\nKPT";
		vector3<> k; //k-point in reciprocal lattice coordinates
		int nInner; //number of bands in the inner pump-active window
		int nOuter; //number of bands in the outer probe-active window
		int innerStart; //start of inner window relative to outer window
		
		diagMatrix E; //energies (dim: nOuter)
		matrix P[3]; //momentum matrix elements (dim: nInner x nOuter each)
		matrix S[3]; //spin matrix elements (dim: nInner x nInner each, only if spinorial)
		std::vector<GePhEntry> GePh; //e-ph matrix elements (only if ePhEnabled)
		
		size_t nBytes(const Header& h) const
		{	size_t dataSize = sizeof(char)*markerLen + sizeof(vector3<>) + sizeof(int)*3
				+ nOuter*sizeof(double) //E
				+ 3*nInner*nOuter*sizeof(complex); //P
			if(h.spinorial)
				dataSize += 3*nInner*nInner*sizeof(complex); //S
			if(h.ePhEnabled)
			{	dataSize += sizeof(size_t); //to store number of g
				for(const GePhEntry& g: GePh)
					dataSize += g.nBytes();
			}
			return dataSize;
		}
		void write(MPIUtil::File fp, const MPIUtil* mpiUtil, const Header& h) const
		{	mpiUtil->fwrite(marker, sizeof(char), markerLen, fp);
			mpiUtil->fwrite(&k, sizeof(vector3<>), 1, fp);
			mpiUtil->fwrite(&nInner, sizeof(int), 3, fp);
			mpiUtil->fwriteData(E, fp);
			for(int iDir=0; iDir<3; iDir++)
				mpiUtil->fwriteData(P[iDir], fp);
			if(h.spinorial)
			{	for(int iDir=0; iDir<3; iDir++)
					mpiUtil->fwriteData(S[iDir], fp);
			}
			if(h.ePhEnabled)
			{	size_t Gsize = GePh.size();
				mpiUtil->fwrite(&Gsize, sizeof(size_t), 1, fp);
				for(const GePhEntry& g: GePh)
					g.write(fp, mpiUtil);
			}
		}
		void read(MPIUtil::File fp, const MPIUtil* mpiUtil, const Header& h)
		{	//Read and check marker:
			char markerIn[markerLen];
			mpiUtil->fread(markerIn, sizeof(char), markerLen, fp);
			if(strncmp(markerIn, marker, markerLen) != 0)
			{	fprintf(stderr, "File format error: could not find KPT header (%s).\n", markerIn);
				mpiUtil->exit(1);
			}
			//Read data:
			mpiUtil->fread(&k, sizeof(vector3<>), 1, fp);
			mpiUtil->fread(&nInner, sizeof(int), 3, fp);
			E.resize(nOuter);
			mpiUtil->freadData(E, fp);
			for(int iDir=0; iDir<3; iDir++)
			{	P[iDir] = zeroes(nInner, nOuter);
				mpiUtil->freadData(P[iDir], fp);
			}
			if(h.spinorial)
			{	for(int iDir=0; iDir<3; iDir++)
				{	S[iDir] = zeroes(nInner, nInner);
					mpiUtil->freadData(S[iDir], fp);
				}
			}
			if(h.ePhEnabled)
			{	size_t Gsize;
				mpiUtil->fread(&Gsize, sizeof(size_t), 1, fp);
				GePh.resize(Gsize);
				for(GePhEntry& g: GePh)
					g.read(fp, mpiUtil);
			}

// 			if(mpiUtil->iProcess()==1)
// 			{	MPI_Offset offset; MPI_File_get_position(fp, &offset);
// 				printf("At location: %lld (dataSize = %lu)\n", offset, dataSize);
// 			}
		}
	};
}

#endif //FEYNWANN_LINDBLADFILE_H
