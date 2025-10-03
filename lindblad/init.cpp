/*-------------------------------------------------------------------
Copyright 2019 Ravishankar Sundararaman, Adela Habib

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

#include <core/Util.h>
#include <core/matrix.h>
#include <core/scalar.h>
#include <core/Random.h>
#include <core/string.h>
#include <core/Units.h>
#include <FeynWann.h>
#include <Histogram.h>
#include <InputMap.h>
#include <lindblad/LindbladFile.h>

#ifdef HDF5_ENABLED
#include <hdf5.h>
#include <core/H5io.h>
hid_t h5_complex;
hid_t h5_bool;

template<> struct h5type<complex> { static hid_t get() { return h5_complex; } };
template<> struct h5type<bool> { static hid_t get() { return h5_bool; } };


hsize_t* hsizes(std::initializer_list<hsize_t> list)
{	return (hsize_t*)list.begin();
}

//Open HDF5 file for collective access:
hid_t openHDF5(string fname)
{       logPrintf("Writing %s: ", fname.c_str()); logFlush();
        //Create MPI file access across all processes:
        hid_t plid = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plid, MPI_COMM_WORLD, MPI_INFO_NULL);
        //Open file with MPI access:
        hid_t fid = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plid);
        if(fid<0) die("Could not open/create output HDF5 file '%s'\n", fname.c_str());
        H5Pclose(plid);
        return fid;
}

std::vector<complex> unravelMatrix(matrix M)
{
	std::vector<complex> result; result.reserve(M.nRows()*M.nCols());
	for (int i = 0; i < M.nRows(); i++)
	{
		for (int j = 0; j < M.nCols(); j++)
		{
			result.push_back(M(i, j));
		}
	}
	return result;
}	

//Create dataset collectively across all processes
template<typename T> hid_t h5createVectorDataset(hid_t fid, const char* dname, hsize_t* dims, int rank)
{
	hid_t dataType = h5type<T>::get();
	hid_t sidGlobal = H5Screate_simple(rank, dims, NULL);
	hid_t did = H5Dcreate(fid, dname, dataType, sidGlobal, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	H5Sclose(sidGlobal);
	if(did<0){ fprintf(stderr, "Could not create dataset '%s' in output file.\n", dname); MPI_Abort(MPI_COMM_WORLD, 1); }
	return did;
}

//Write slice to existing dataset independently form each process
template<typename T> void h5writeVectorSlice(hid_t did, const T* data, hsize_t* offset, hsize_t* count, int rank)
{	
	//Select hyperslab:
	hid_t sid = H5Screate_simple(rank, count, NULL);
	hid_t sidGlobal = H5Dget_space(did);
	H5Sselect_hyperslab(sidGlobal, H5S_SELECT_SET, offset, NULL, count, NULL);
	
	//Write data:
	hid_t dataType = h5type<T>::get();
	H5Dwrite(did, dataType, sid, sidGlobal, H5P_DEFAULT, data);
	
	//Cleanup:
	H5Sclose(sidGlobal);
	H5Sclose(sid);
}

template<typename T> void h5writeVectorAttr(hid_t fid, const char* dname, const T* data, const hsize_t* dims, hsize_t rank)
{	hid_t dataType = h5type<T>::get();
	hid_t sid = H5Screate_simple(rank, dims, NULL);
	hid_t aid = H5Acreate2(fid, dname, dataType, sid, H5P_DEFAULT, H5P_DEFAULT);
	H5Sclose(sid);
	if(aid<0) die("Could not create dataset '%s' in HDF5 file.\n", dname);
	//Create collective write property:
	hid_t plid = H5Pcreate(H5P_DATASET_XFER);
	H5Pset_dxpl_mpio(plid, H5FD_MPIO_INDEPENDENT);
	//Write data:
	H5Awrite(aid, dataType, data);
	//Cleanup:
	H5Aclose(aid);
	H5Pclose(plid);
}

template<typename T> void h5writeScalarAttr(hid_t fid, const char* dname, const T& data)
{	hid_t dataType = h5type<T>::get();
	//Create dataset:
	hid_t sid = H5Screate(H5S_SCALAR);
	hid_t aid = H5Acreate2(fid, dname, dataType, sid, H5P_DEFAULT, H5P_DEFAULT);
	H5Sclose(sid);
	if(aid<0) die("Could not create dataset '%s' in HDF5 file.\n", dname);
	//Write data:
	H5Awrite(aid, dataType, &data);
	H5Aclose(aid);
}
#endif

//Reverse iterator for pointers:
template<class T> constexpr std::reverse_iterator<T*> reverse(T* i) { return std::reverse_iterator<T*>(i); }

static const double omegaPhCut = 1e-6;
static const double nEphDelta = 5.; //number of ePhDelta to include in output

//Helper class to "argsort" an array i.e. determine the indices that sort it
template<typename ArrayType> struct IndexCompare
{	const ArrayType& array;
	IndexCompare(const ArrayType& array) : array(array) {}
	template<typename Integer> bool operator()(Integer i1, Integer i2) const { return array[i1] < array[i2]; }
};


enum BandSelection {ElectronsOnly, HolesOnly, AllBands};
EnumStringMap<BandSelection> bandSelectionMap(
	ElectronsOnly, "electrons-only",
	HolesOnly, "holes-only",
	AllBands, "all-bands"
);



std::vector<complex> getContiguousData(const matrix X[3])
{	std::vector<complex> result;
	int nBands = X[0].nRows();
	result.reserve(3 * nBands * nBands);
	for(int iDir=0; iDir<3; iDir++)
		for(int i=0; i<nBands; i++)
			for(int j=0; j<nBands; j++)
				result.push_back(X[iDir](i, j));
	return result;
}


//Lindblad initialization using FeynWann callback
struct LindbladInit
{	
	FeynWann& fw;
	const vector3<int>& NkFine; //!< effective k-point mesh sampled
	const size_t nkTot; //!< total k-points effectively used in BZ sampling
	
	const double dmuMin, dmuMax, Tmax;
	const BandSelection bandSelection; //!< which sets of bands to include (e, h or all)
	const int bandCountFixed; //!< if non-zero, fix nInner = bandCountFixed with no nOuter for all energies
	const double pumpOmegaMax, probeOmegaMax;
	
	const bool ePhEnabled; //!< whether e-ph coupling is enabled
	const double ePhDelta; //!< Gaussian energy conservation width
	
	const bool defectEnabled; //!< if defect scattering is enabled with phonons
	
	LindbladInit(FeynWann& fw, const vector3<int>& NkFine,
		double dmuMin, double dmuMax, double Tmax,
		BandSelection bandSelection, int bandCountFixed,
		double pumpOmegaMax, double probeOmegaMax,
		bool ePhEnabled, double ePhDelta, bool defectEnabled)
	: fw(fw), NkFine(NkFine), nkTot(NkFine[0]*NkFine[1]*NkFine[2]),
		dmuMin(dmuMin), dmuMax(dmuMax), Tmax(Tmax),
		bandSelection(bandSelection), bandCountFixed(bandCountFixed),
		pumpOmegaMax(pumpOmegaMax), probeOmegaMax(probeOmegaMax),
		ePhEnabled(ePhEnabled), ePhDelta(ePhDelta), defectEnabled(defectEnabled)
	{
	}
	
	//--------- k-point selection -------------
	
	double EvMax, EcMin; //VBM and CBM estimates
	inline void eRange(const FeynWann::StateE& state)
	{	for(const double& E: state.E)
		{	if(E<dmuMin and E>EvMax) EvMax = E;
			if(E>dmuMax and E<EcMin) EcMin = E;
		}
	}
	static void eRange(const FeynWann::StateE& state, void* params)
	{	((LindbladInit*)params)->eRange(state);
	}
	
	double Estart, Estop; //energy range for k selection
	std::vector<vector3<>> k; //selected k-points
	std::vector<double> E; //all band energies for selected k-points
	size_t nActiveTot; //total number of active states

	std::vector<vector3<int>> offK; //index of each k by offsets: [ kOffset, qOffset, ik ]
	vector3<int> offKcur; //offset indices of current eLoop call in first two components
	std::vector<vector3<int>> offKuniq; //unique offsets in offK
	std::vector<size_t> ikStartOff; //starting k index for each offset in offKuniq (length: offKuniq.size()+1)
	
	size_t iGroup; //index of current group amongst groups
	size_t nGroups; //number of process groups
	std::vector<size_t> offStartGroup; //starting offset index for each process group (length: nGroups+1)
	std::vector<size_t> ikStartGroup; //starting k index for each process group (length: nGroups+1)
	
	std::map<size_t,size_t> kIndexMap; //map from k-point mesh index to index in selected set
	inline size_t kIndex(vector3<> k)
	{	size_t index=0;
		for(int iDir=0; iDir<3; iDir++)
		{	double ki = k[iDir] - floor(k[iDir]); //wrapped to [0,1)
			index = (size_t)round(NkFine[iDir]*(index+ki));
		}
		return index;
	}
	//Search for k using kIndexMap; return false if not found
	inline bool findK(vector3<> k, size_t&ik)
	{	const std::map<size_t,size_t>::iterator iter = kIndexMap.find(kIndex(k));
		if(iter != kIndexMap.end())
		{	ik = iter->second;
			return true;
		}
		else return false;
	}
	
	inline void kSelect(const FeynWann::StateE& state)
	{	int nActiveCur = 0;
		for(double E: state.E)
			if(E>=Estart and E<=Estop)
				nActiveCur++;
		if(nActiveCur)
		{	nActiveTot += (bandCountFixed ? bandCountFixed : nActiveCur);
			k.push_back(state.k);
			E.insert(E.end(), state.E.begin(), state.E.end());
			offKcur[2] = state.ik;
			offK.push_back(offKcur);
		}
	}
	static void kSelect(const FeynWann::StateE& state, void* params)
	{	((LindbladInit*)params)->kSelect(state);
	}
	void kpointSelect(const std::vector<vector3<>>& k0)
	{
		//Initialize sampling parameters:
		size_t oStart, oStop; //range of offsets handled by this process group
		if(mpiGroup->isHead())
			TaskDivision(k0.size(), mpiGroupHead).myRange(oStart, oStop);
		mpiGroup->bcast(oStart);
		mpiGroup->bcast(oStop);
		size_t noMine = oStop-oStart;
		size_t oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress
		
		//Determine energy range:
		EvMax = -DBL_MAX;
		EcMin = +DBL_MAX;
		logPrintf("Determining energy range: "); logFlush();
		for(size_t o=oStart; o<oStop; o++)
		{	for(vector3<> qOff: fw.qOffset) fw.eLoop(k0[o]+qOff, LindbladInit::eRange, this);
			//Print progress:
			if((o-oStart+1)%oInterval==0) { logPrintf("%d%% ", int(round((o-oStart+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		mpiWorld->allReduce(EvMax, MPIUtil::ReduceMax);
		mpiWorld->allReduce(EcMin, MPIUtil::ReduceMin);
		logPrintf("Max occupied energy across dmu range: %.3lf eV\n", EvMax/eV);
		logPrintf("Min unoccupied energy across dmu range: %.3lf eV\n", EcMin/eV);
		//--- add margins of max phonon energy, energy conservation width and fermiPrime width
		double Emargin =7.*Tmax; //neglect below 10^-3 occupation deviation from equilibrium
		Estart = std::min(EcMin - pumpOmegaMax, EvMax) - Emargin;
		Estop = std::max(EvMax + pumpOmegaMax, EcMin) + Emargin;
		//--- apply band selections if any
		double Emid = 0.5 * (EvMax + EcMin);
		switch(bandSelection)
		{	case ElectronsOnly:
				Estart = Emid;
				logPrintf("Restricting lower end of energy range to only include electrons.\n");
				break;
			case HolesOnly:
				Estop = Emid;
				logPrintf("Restricting upper end of energy range to only include holes.\n");
				break;
			default:;
		}
		logPrintf("Active energy range: %.3lf to %.3lf eV\n", Estart/eV, Estop/eV);
		
		//Select k-points:
		nActiveTot = 0;
		logPrintf("Scanning k-points with active states: "); logFlush();
		for(size_t o=oStart; o<oStop; o++)
		{	offKcur[0] = o;
			offKcur[1] = 0;
			for(vector3<> qOff: fw.qOffset)
			{	fw.eLoop(k0[o]+qOff, LindbladInit::kSelect, this);
				offKcur[1]++; //increment qOffset index
			}
			//Print progress:
			if((o-oStart+1)%oInterval==0) { logPrintf("%d%% ", int(round((o-oStart+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		mpiWorld->allReduce(nActiveTot, MPIUtil::ReduceSum);
		
		//Synchronize selected k and E across all processes:
		//--- determine nk on each process and compute cumulative counts
		std::vector<size_t> nkPrev(mpiWorld->nProcesses()+1);
		for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
		{	size_t nkCur = k.size();
			mpiWorld->bcast(nkCur, jProc); //nkCur = k.size() on jProc in all processes
			nkPrev[jProc+1] = nkPrev[jProc] + nkCur; //cumulative count
		}
		size_t nkSelected = nkPrev.back();
		//--- broadcast k and E:
		{	//Set k and E in position in global arrays:
			std::vector<vector3<>> k(nkSelected);
			std::vector<double> E(nkSelected*fw.nBands);
			std::vector<vector3<int>> offK(nkSelected);
			std::copy(this->k.begin(), this->k.end(), k.begin()+nkPrev[mpiWorld->iProcess()]);
			std::copy(this->E.begin(), this->E.end(), E.begin()+nkPrev[mpiWorld->iProcess()]*fw.nBands);
			std::copy(this->offK.begin(), this->offK.end(), offK.begin()+nkPrev[mpiWorld->iProcess()]);
			//Broadcast:
			for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
			{	size_t ikStart = nkPrev[jProc], nk = nkPrev[jProc+1]-ikStart;
				mpiWorld->bcast(k.data()+ikStart, nk, jProc);
				mpiWorld->bcast(E.data()+ikStart*fw.nBands, nk*fw.nBands, jProc);
				mpiWorld->bcast(offK.data()+ikStart, nk, jProc);
			}
			//Sort by offset:
			std::vector<size_t> sortIndex(nkSelected);
			for(size_t i=0; i<nkSelected; i++) sortIndex[i] = i;
			std::sort(sortIndex.begin(), sortIndex.end(), IndexCompare<std::vector<vector3<int>>>(offK));
			//Store to class variables in sorted order:
			this->k.resize(nkSelected);
			this->E.resize(nkSelected*fw.nBands);
			this->offK.resize(nkSelected);
			vector3<int> offKprev(-1,-1,0);
			for(size_t i=0; i<nkSelected; i++)
			{	size_t iSrc = sortIndex[i];
				this->k[i] = k[iSrc];
				eblas_copy(this->E.data()+i*fw.nBands, E.data()+iSrc*fw.nBands, fw.nBands);
				this->offK[i] = offKcur = offK[iSrc];
				//Update unique offset list:
				offKcur[2] = 0; //ignore k in comparing for unique list
				if(not (offKcur == offKprev))
				{	offKuniq.push_back(offKcur);
					ikStartOff.push_back(i);
					offKprev = offKcur;
				}
			}
			ikStartOff.push_back(nkSelected);
		}
		logPrintf("Found k-points with active states in %lu of %lu q-mesh offsets (%.0fx reduction)\n",
			offKuniq.size(), k0.size()*fw.qOffset.size(), round(k0.size()*fw.qOffset.size()*1./offKuniq.size()));
		logPrintf("Found %lu k-points with active states from %lu total k-points (%.0fx reduction)\n",
			nkSelected, nkTot, round(nkTot*1./nkSelected));
		logPrintf("Selected %lu active states from %lu total electronic states (%.0fx reduction)\n\n",
			nActiveTot, nkTot*fw.nBands, round((nkTot*fw.nBands)*1./nActiveTot));
		
		//Make group index, count, offset division and k division available on all processes of each group:
		if(mpiGroup->isHead())
		{	iGroup = mpiGroupHead->iProcess();
			nGroups = mpiGroupHead->nProcesses();
			offStartGroup.assign(nGroups+1, 0);
			ikStartGroup.assign(nGroups+1, 0);
			TaskDivision groupDiv(offKuniq.size(), mpiGroupHead);
			for(size_t jGroup=0; jGroup<nGroups; jGroup++)
			{	offStartGroup[jGroup+1] = groupDiv.stop(jGroup);
				ikStartGroup[jGroup+1] = ikStartOff[offStartGroup[jGroup+1]];
			}
		}
		mpiGroup->bcast(iGroup);
		mpiGroup->bcast(nGroups);
		offStartGroup.resize(nGroups+1);
		ikStartGroup.resize(nGroups+1);
		mpiGroup->bcastData(offStartGroup);
		mpiGroup->bcastData(ikStartGroup);
		
		//Initialize kIndexMap for searching selected k-points:
		for(size_t ik=0; ik<k.size(); ik++)
			kIndexMap[kIndex(k[ik])] = ik;
	}
	
	//--------- k-pair selection -------------
	std::vector<std::vector<size_t>> kpartners; //list of e-ph coupled k2 for each k1
	std::vector<double> kpairWeight; //Econserve weight factor for all k1 pairs due to downsampling (1 if no downsmapling)
	size_t nActivePairs; //total number of active state pairs
	inline void selectActive(const double*& Ebegin, const double*& Eend, double Elo, double Ehi) //narrow pointer range to data within [Estart,Estop]
	{	if(bandCountFixed)
		{	//New logic for fixed band count targeting qimpy.transport
			if(bandSelection == ElectronsOnly)
			{	Ebegin = std::lower_bound(Ebegin, Eend, Elo);
				const double* EendNew = Ebegin + bandCountFixed;
				assert(EendNew <= Eend); //Ensure fixed band count doesn't cross upper range of available bands
				Eend = EendNew;
			}
			else //bandSelection == HolesOnly (Ensured during startup)
			{	Eend = &(*std::lower_bound(reverse(Eend), reverse(Ebegin), Ehi, std::greater<double>()))+1;
				const double* EbeginNew = Eend - bandCountFixed;
				assert(EbeginNew >= Ebegin);
				Ebegin = EbeginNew; //Ensure fixed band count doesn't cross lower range of available bands
			}
		}
		else
		{	//Select based on energies only:
			Ebegin = std::lower_bound(Ebegin, Eend, Elo);
			Eend = &(*std::lower_bound(reverse(Eend), reverse(Ebegin), Ehi, std::greater<double>()))+1;
		}
	}
	inline void kpSelect(const FeynWann::StatePh& state)
	{	//Find pairs of momentum conserving electron states with this q:
		for(size_t ik1=0; ik1<k.size(); ik1++)
		{	const vector3<>& k1 = k[ik1];
			vector3<> k2 = k1 - state.q; //momentum conservation
			size_t ik2; if(not findK(k2, ik2)) continue;
			//Check energy conservation for pair of bands within active range:
			//--- determine ranges of all E1 and E2:
			const double *E1begin = E.data()+ik1*fw.nBands, *E1end = E1begin+fw.nBands;
			const double *E2begin = E.data()+ik2*fw.nBands, *E2end = E2begin+fw.nBands;
			//--- narrow to active energy ranges:
			selectActive(E1begin, E1end, Estart, Estop);
			selectActive(E2begin, E2end, Estart, Estop);
			//--- check energy ranges:
			bool Econserve = false;
			for(const double* E1=E1begin; E1<E1end; E1++) //E1 in active range
			{	for(const double* E2=E2begin; E2<E2end; E2++) //E2 in active range
				{	for(const double omegaPh: state.omega) if(omegaPh>omegaPhCut) //loop over non-zero phonon frequencies
					{	double deltaE = (*E1) - (*E2) - omegaPh; //energy conservation violation
						if(fabs(deltaE) < nEphDelta*ePhDelta) //else negligible
						{	Econserve = true;
							nActivePairs++;
						}
					} 
					if(defectEnabled) //Check for energy conservation for elastic scattering
					{   double deltaE = (*E1) - (*E2); //energy conservation violation
						if(fabs(deltaE) < nEphDelta*ePhDelta) //else negligible
						{	Econserve = true;
							nActivePairs++;
						}
                    }
				}
			}
			if(Econserve) kpartners[ik1].push_back(ik2);
		}
	}
	static void kpSelect(const FeynWann::StatePh& state, void* params)
	{	((LindbladInit*)params)->kpSelect(state);
	}
	void kpairSelect(const std::vector<vector3<>>& q0, size_t maxNeighbors)
	{	
		//Initialize sampling parameters:
		size_t oStart, oStop; //!< range of offstes handled by this process groups
		if(mpiGroup->isHead())
			TaskDivision(q0.size(), mpiGroupHead).myRange(oStart, oStop);
		mpiGroup->bcast(oStart);
		mpiGroup->bcast(oStop);
		size_t noMine = oStop-oStart;
		size_t oInterval = std::max(1, int(round(noMine/50.))); //interval for reporting progress

		//Find momentum-conserving k-pairs for which energy conservation is also possible for some bands:
		nActivePairs = 0;
		kpartners.resize(k.size());
		logPrintf("Scanning k-pairs with e-ph coupling: "); logFlush();
		for(size_t o=oStart; o<oStop; o++)
		{	fw.phLoop(q0[o], LindbladInit::kpSelect, this);
			//Print progress:
			if((o-oStart+1)%oInterval==0) { logPrintf("%d%% ", int(round((o-oStart+1)*100./noMine))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		mpiWorld->allReduce(nActivePairs, MPIUtil::ReduceSum);
		
		//Redistribute kpartners by the processes that will deal with each ik1:
		size_t nPartnersMin = k.size(), nPartnersMax = 0, nPartnersSum = 0;
		size_t nPartDownMin = k.size(), nPartDownMax = 0, nPartDownSum = 0; //down-selected versions
		kpairWeight.assign(k.size(), 0.);
		for(size_t ik1=0; ik1<k.size(); ik1++)
		{	//Determine which group head process is responsible for this k:
			const bool isMyGroups = (ik1>=ikStartGroup[iGroup]) and (ik1<ikStartGroup[iGroup+1]);
			const bool isMine = mpiGroup->isHead() and isMyGroups;
			int whose = isMine ? mpiWorld->iProcess() : 0;
			mpiWorld->allReduce(whose, MPIUtil::ReduceMax); //whose now points to the process which had isMine=1
			//Transfer kpartner data to the responsible process:
			std::vector<size_t>& kp = kpartners[ik1];
			if(isMine)
			{	//Get data from every other process
				for(int jProc=0; jProc<mpiWorld->nProcesses(); jProc++)
					if(jProc != mpiWorld->iProcess())
					{	//Recv into a temporary buffer:
						size_t size;
						mpiWorld->recv(size, jProc, 0);
						std::vector<size_t> buf(size);
						mpiWorld->recvData(buf, jProc, 1);
						//Append to partners list:
						kp.insert(kp.end(), buf.begin(), buf.end());
					}
				//Sort partners (automatically by offset order since k's sorted that way):
				size_t nPartners = kp.size();
				std::sort(kp.begin(), kp.end());
				//Sort partners and optionally downselect:
				double pSel = maxNeighbors ? std::min(1., maxNeighbors*1./nPartners) : 1.;
				kpairWeight[ik1] = sqrt(1./pSel);
				if(pSel < 1.)
				{	std::vector<size_t> kpNew; kpNew.reserve(int(ceil(pSel * nPartners)));
					Random::seed(ik1);
					for(size_t ik: kp) if(Random::uniform()<pSel) kpNew.push_back(ik);
					std::swap(kp, kpNew);
				}
				size_t nPartDown = kp.size();
				//Update stats:
				if(nPartners < nPartnersMin) nPartnersMin = nPartners;
				if(nPartners > nPartnersMax) nPartnersMax = nPartners;
				nPartnersSum += nPartners;
				if(nPartDown < nPartDownMin) nPartDownMin = nPartDown;
				if(nPartDown > nPartDownMax) nPartDownMax = nPartDown;
				nPartDownSum += nPartDown;
			}
			else
			{	//Send to the responsible process:
				mpiWorld->send(kp.size(), whose, 0);
				mpiWorld->sendData(kp, whose, 1);
				kp.clear(); //no longer needed on this process
			}
			//Synchronize within group:
			if(isMyGroups)
			{	size_t size = kp.size();
				mpiGroup->bcast(size);
				kp.resize(size);
				mpiGroup->bcastData(kp);
			}
		}
		mpiWorld->allReduceData(kpairWeight, MPIUtil::ReduceMax); //only set on group head that owns this k
		mpiWorld->allReduce(nPartnersMin, MPIUtil::ReduceMin);
		mpiWorld->allReduce(nPartnersMax, MPIUtil::ReduceMax);
		mpiWorld->allReduce(nPartnersSum, MPIUtil::ReduceSum);
		mpiWorld->allReduce(nPartDownMin, MPIUtil::ReduceMin);
		mpiWorld->allReduce(nPartDownMax, MPIUtil::ReduceMax);
		mpiWorld->allReduce(nPartDownSum, MPIUtil::ReduceSum);
		size_t nkpairsTot = k.size()*k.size();
		logPrintf("Found %lu k-pairs with e-ph coupling from %lu total pairs of selected k-points (%.0fx reduction)\n",
			nPartnersSum, nkpairsTot, round(nkpairsTot*1./nPartnersSum));
		size_t nStatePairsTot = std::pow(nkTot*fw.nBands, 2);
		logPrintf("Selected %lu active state pairs from %lu total electronic state pairs (%.0fx reduction)\n",
			nActivePairs, nStatePairsTot, round(nStatePairsTot*1./nActivePairs));
		logPrintf("Number of partners per k-point:  min: %lu  max: %lu  mean: %.1lf\n", nPartnersMin, nPartnersMax, nPartnersSum*1./k.size());
		if(maxNeighbors) logPrintf("%9s After down-selection:  min: %lu  max: %lu  mean: %.1lf\n", "", nPartDownMin, nPartDownMax, nPartDownSum*1./k.size());
	}

	//--------- Save data -------------
	std::vector<LindbladFile::Kpoint> kpAll; //array of kpoint data for all active k-points
	std::vector<LindbladFile::Kpoint> kpOffset; //array of kpoint data for current offset
	std::vector<int> kpWhose; //index of process in mpiWorld that owns each entry in kpAll
	std::vector<size_t> kpSize; //size in bytes of each entry in kpAll when written to file
	std::vector<size_t> kpGcount; //number of G entries at each kp
	size_t GcountTot; //total number of G entries
	
	//Wrapper to selectActive that computes offset and length of the active range (rather than narrowing the iterator range)
	inline void activeOffsets(const double* Ebegin, const double* Eend, double Estart, double Estop, int& offset, int& length)
	{	const double *EactiveBegin = Ebegin, *EactiveEnd = Eend;
		selectActive(EactiveBegin, EactiveEnd, Estart, Estop);
		offset = EactiveBegin - Ebegin;
		length = EactiveEnd - EactiveBegin;
	}
	
	//Initialize k-point data:
	void initKpoint(const FeynWann::StateE& state)
	{	LindbladFile::Kpoint& kp = kpOffset[state.ik];
		if(kp.E.size()) return; //already initialized
		
		//Find k-point index in global list:
		kp.k = state.k;
		size_t ik; if(not findK(kp.k, ik)) return;

		//Determine energy ranges:
		const double *Ebegin = state.E.data(), *Eend = Ebegin+fw.nBands;
		//--- pump-active (inner) energy range:
		int innerOffset = 0; //offset from original bands to inner window
		activeOffsets(Ebegin, Eend, Estart, Estop, innerOffset, kp.nInner);
		double EinnerMin = Ebegin[innerOffset];
		double EinnerMax = Ebegin[innerOffset+kp.nInner-1];
		//--- probe-active (outer) energy range:
		int outerOffset = 0; //offset from original bands to outer window
		if(bandCountFixed)
		{	//Set outer window to be same as inner window
			outerOffset = innerOffset;
			kp.nOuter = kp.nInner;
		}
		else
		{	//Set outer window based on probe energies from inner window
			activeOffsets(Ebegin, Eend,
				EinnerMin-probeOmegaMax, //lowest occupied energy accessible from bottom of active window
				EinnerMax+probeOmegaMax,  //highest unoccupied energy accessible from top of active window
				outerOffset, kp.nOuter);
		}
		kp.innerStart = innerOffset - outerOffset;
		kp.innerOffset = innerOffset;
		
		//Save energy and matrix elements to kp:
		//Energies:
		kp.E.assign(Ebegin+outerOffset, Ebegin+outerOffset+kp.nOuter);
		//Momenta:
		for(int iDir=0; iDir<3; iDir++)
			kp.P[iDir] = state.v[iDir](innerOffset,innerOffset+kp.nInner, outerOffset,outerOffset+kp.nOuter);
		//Spin:
		if(fw.nSpinor == 2)
			for(int iDir=0; iDir<3; iDir++)
				kp.S[iDir] = state.S[iDir](innerOffset,innerOffset+kp.nInner, innerOffset,innerOffset+kp.nInner);
		if(fw.fwp.needL)
			for(int iDir=0; iDir<3; iDir++)
				kp.L[iDir] = state.L[iDir](innerOffset,innerOffset+kp.nInner, innerOffset,innerOffset+kp.nInner);

		// Set U to a submatrix of state.U (nBands by nInner)
		kp.U = state.U(0, fw.nBands, innerOffset, innerOffset+kp.nInner);
		
	}
	static void initKpoint(const FeynWann::StateE& state, void* params)
	{	((LindbladInit*)params)->initKpoint(state);
	}
	
	//Add e-ph nmatrix element to k-point data:
	void addEph(const FeynWann::MatrixEph& mEph)
	{	const FeynWann::StateE& e1 = *(mEph.e1);
		const FeynWann::StateE& e2 = *(mEph.e2);
		const FeynWann::StatePh& ph = *(mEph.ph);
		LindbladFile::Kpoint& kp1 = kpOffset[e1.ik];
		//Get global index and active ranges of each point:
		#define PREP(i) \
			size_t ik##i; \
			int innerOffset##i, nInner##i; \
			{	if(not findK(e##i.k, ik##i)) return; \
				const double *Ebegin = E.data()+ik##i*fw.nBands; \
				const double *Eend= Ebegin + fw.nBands; \
				activeOffsets(Ebegin, Eend, Estart, Estop, innerOffset##i, nInner##i); \
			}
		PREP(1)
		PREP(2)
		#undef PREP
		//Collect energy-conserving matrix elements within active window:
		for(int alpha=0; alpha<fw.nModes; alpha++)
		{	LindbladFile::GePhEntry g;
			g.jk = ik2;
			g.omegaPh = ph.omega[alpha];
			if(g.omegaPh < omegaPhCut) continue; //avoid zero frequency phonons
			double sigmaInv = 1./ePhDelta;
			double deltaPrefac = sqrt(sigmaInv/sqrt(2.*M_PI)) * kpairWeight[ik1]; //account for down-sampling weight (1 if no down-sampling)
			const matrix& M = mEph.M[alpha];
			for(int n2=innerOffset2; n2<innerOffset2+nInner2; n2++)
				for(int n1=innerOffset1; n1<innerOffset1+nInner1; n1++)
				{	double deltaEbySigma = sigmaInv*(e1.E[n1] - e2.E[n2] - g.omegaPh);
					if(fabs(deltaEbySigma) < nEphDelta)
					{	SparseEntry s;
						s.i = n1 - innerOffset1;
						s.j = n2 - innerOffset2;
						s.val = M(n1,n2) * (deltaPrefac*exp(-0.25*deltaEbySigma*deltaEbySigma)); //apply e-conservation factor (sqrt(normalized gaussian))
						g.G.push_back(s);
					}
				}
			if(g.G.size()) kp1.GePh.push_back(g);
		}
	}
	static void addEph(const FeynWann::MatrixEph& mEph, void* params)
	{	((LindbladInit*)params)->addEph(mEph);
	}
	
	//Add e-defect matrix element to k-point data
	void addDefect(const FeynWann::MatrixDefect& mD)
	{	const FeynWann::StateE& e1 = *(mD.e1);
		const FeynWann::StateE& e2 = *(mD.e2);
		LindbladFile::Kpoint& kp1 = kpOffset[e1.ik];
		//Get global index and active ranges of each point:
		#define PREP(i) \
			size_t ik##i; \
			int innerOffset##i, nInner##i; \
			{	if(not findK(e##i.k, ik##i)) return; \
				const double *Ebegin = E.data()+ik##i*fw.nBands; \
				const double *Eend= Ebegin + fw.nBands; \
				activeOffsets(Ebegin, Eend, Estart, Estop, innerOffset##i, nInner##i); \
			}
		PREP(1)
		PREP(2)
		#undef PREP
		//Collect energy-conserving matrix elements within active window:
		LindbladFile::GePhEntry g;
		g.jk = ik2;
		g.omegaPh = 0.;
		double sigmaInv = 1./ePhDelta;
		double deltaPrefac = sqrt(sigmaInv/sqrt(2.*M_PI)) * kpairWeight[ik1]; //account for down-sampling weight (1 if no down-sampling)
		const matrix& M = mD.M;
		for(int n2=innerOffset2; n2<innerOffset2+nInner2; n2++)
			for(int n1=innerOffset1; n1<innerOffset1+nInner1; n1++)
			{	double deltaEbySigma = sigmaInv*(e1.E[n1] - e2.E[n2]);
				if(fabs(deltaEbySigma) < nEphDelta)
				{	SparseEntry s;
					s.i = n1 - innerOffset1;
					s.j = n2 - innerOffset2;
					s.val = M(n1,n2) * (deltaPrefac*exp(-0.25*deltaEbySigma*deltaEbySigma)); //apply e-conservation factor (sqrt(normalized gaussian))
					g.G.push_back(s);
				}
			}
		if(g.G.size()) kp1.GePh.push_back(g);
	}
    static void addDefect(const FeynWann::MatrixDefect& mD, void* params)
	{	((LindbladInit*)params)->addDefect(mD);
	}

        matrix pivotPermute(matrix M)
	{
		std::vector<int> perm;
		for (int i = 0; i < M.nCols(); i++)
		{
			int argmax = -1;
			double max = 0;
			for (int j = 0; j < M.nRows(); j++)
			{
				if (M(i, j).abs() > max)
				{
					max = M(i, j).abs();
					argmax = j;
				}
			}
			// Ensure all rows will be permuted
			for (const int& ind : perm) if (ind == argmax)
					die("Overlap matrix is malformed.");
			perm.push_back(argmax);
		}
		matrix result = zeroes(M.nRows(), M.nCols());
		for (int row = 0; row < M.nRows(); row++)
		{
			result.set(row, row+1, 0, M.nCols(), M(perm[row], perm[row]+1, 0, M.nCols()));
		}
		return result;
	}

    	matrix getRotation(matrix C, matrix CRef, std::vector<double> eigRef)
	{
		matrix O = dagger(C)*CRef;
		O = pivotPermute(O);
		// Prevent rotation between non-degenerate states
		for (int i = 0; i < O.nRows(); i++)
		{
			for (int j = 0; j < O.nCols(); j++)
			{
				if (eigRef[i] - eigRef[j] > fw.fwp.degeneracyThreshold)
				{
					O(i, j) = 0.0;
				}
			}
		}

		matrix result = O*invsqrt(dagger(O)*O);

		return result;
	}
	
	void saveData(const std::vector<vector3<>>& k0, string outFile, bool enableH5Output, string outFileH5)
	{
		//Prepare the file header:
		LindbladFile::Header h;
		h.dmuMin = dmuMin;
		h.dmuMax = dmuMax;
		h.Tmax = Tmax;
		h.pumpOmegaMax = pumpOmegaMax;
		h.probeOmegaMax = probeOmegaMax;
		h.nk = k.size();
		h.nkTot = nkTot;
		h.ePhEnabled = ePhEnabled;
		h.spinorial = (fw.nSpinor==2);
		h.spinWeight = fw.spinWeight;
		h.R = fw.R;
		h.haveL = fw.fwp.orbitalZeeman;
		
		//Loop over offsets in current group:
		logPrintf("\nGenerating matrix elements: "); logFlush();
		kpAll.clear(); kpAll.resize(k.size());
		kpWhose.assign(k.size(), -1);
		kpSize.assign(k.size(), 0);
		kpGcount.assign(k.size(), 0);
		size_t nOffMine = offStartGroup[iGroup+1] - offStartGroup[iGroup];
		size_t offInterval = std::max(1, int(round(nOffMine/50.))); //interval for reporting progress
		for(size_t iOffMine=0; iOffMine<nOffMine; iOffMine++)
		{	size_t iOff = offStartGroup[iGroup] + iOffMine; //current offset index
			size_t ikStart = ikStartOff[iOff]; //range start of k in this offset
			size_t ikStop = ikStartOff[iOff+1]; //range end of k in this offset
			offKcur = offKuniq[iOff];
			vector3<> k01 = k0[offKcur[0]] + fw.qOffset[offKcur[1]];
			
			//Initialize mask of active states:
			size_t nkOff = fw.eCountPerOffset(); //number of k in an offset
			std::vector<bool> mask(nkOff, false);
			for(size_t ik=ikStart; ik<ikStop; ik++) mask[offK[ik][2]] = true;
			
			kpOffset.clear();
			kpOffset.resize(nkOff);
			
			if(ePhEnabled)
			{	//Create list of offsets contained in partners of all k in current offset:
				std::map<vector3<int>, std::vector<bool>> maskMap; //(ik1,ik2) mask indexed by partner offset
				for(size_t ik=ikStart; ik<ikStop; ik++)
				{	int ikOff = offK[ik][2]; //index of ik within its offset
					for(size_t jk: kpartners[ik])
					{	vector3<int> offKpartner = offK[jk];
						int jkOff = offKpartner[2]; //index of jk within its offset
						//Locate entry in maskMap or create one:
						offKpartner[2] = 0; //unique partner offset index to maskMap
						auto iter = maskMap.find(offKpartner);
						if(iter == maskMap.end())
							iter = maskMap.insert(iter, std::make_pair(offKpartner, std::vector<bool>(nkOff*nkOff, false)));
						//Update mask:
						iter->second[ikOff*nkOff+jkOff] = true;
					}
				}
				//Loop over partner offsets:
				bool initDone = false; //whether k-point has already been initialized
				for(auto entry: maskMap)
				{	//Initialize the masks:
					const std::vector<bool>& maskPair =  entry.second; //mask of which pairs of k are active in current offset pair
					std::vector<bool> mask1(nkOff, false); //which ikOffs are encountered in maskPair
					std::vector<bool> mask2(nkOff, false); //which jkOffs are encountered in maskPair
					if(not initDone) mask1 = mask; //make sure all k-points are inited in first case (not just ones with neighbors in that offset)
					auto maskPairIter = maskPair.begin();
					for(size_t ikOff=0; ikOff<nkOff; ikOff++)
						for(size_t jkOff=0; jkOff<nkOff; jkOff++)
							if(*(maskPairIter++))
							{	mask1[ikOff] = true;
								mask2[jkOff] = true;
							}
					//Compute matrix elements:
					vector3<int> offKpartner = entry.first;
					vector3<> k02 = k0[offKpartner[0]] + fw.qOffset[offKpartner[1]];
					FeynWann::eProcessFunc initFunc = 0; //after first pass, only need to invoke addEph(),
					if(not initDone) initFunc = initKpoint; //... but in first pass, also invoke initKpoint()
					fw.ePhLoop(k01, k02, addEph, this, initFunc, 0, 0, &mask1, &mask2, &maskPair);
                    if(defectEnabled)
						fw.defectLoop(k01, k02, addDefect, this, initFunc, 0, &mask1, &mask2, &maskPair);
					initDone = true;
				}
				if(not initDone) fw.eLoop(k01, initKpoint, this, &mask); //corner case: entire offset has no partners (make sure init still happens)
				//Move e-ph matrix elements to process that owns each ik:
				if(mpiGroup->nProcesses() > 1)
				{	for(size_t ik=ikStart; ik<ikStop; ik++)
					{	LindbladFile::Kpoint& kp = kpOffset[offK[ik][2]]; //current k-point data
						int whose = kp.E.size() ? mpiGroup->iProcess() : -1;
						mpiGroup->allReduce(whose, MPIUtil::ReduceMax); //now whose points to process within group that owns ik
						bool isMine = (whose == mpiGroup->iProcess());
						//Determine number of entries on each process:
						std::vector<size_t> nGePh(mpiGroup->nProcesses(), 0);
						nGePh[mpiGroup->iProcess()] = kp.GePh.size();
						mpiGroup->reduceData(nGePh, MPIUtil::ReduceMax, whose);
						if(isMine)
						{	size_t nGePhTot = 0;  for(size_t n: nGePh) nGePhTot += n; //total number of entries
							std::vector<LindbladFile::GePhEntry> GePh(nGePhTot);
							auto iter = GePh.begin();
							for(int jProc=0; jProc<mpiGroup->nProcesses(); jProc++)
							{	if(jProc == mpiGroup->iProcess())
								{	for(const LindbladFile::GePhEntry& g: kp.GePh)
										*(iter++) = g;
								}
								else
								{	for(size_t i=0; i<nGePh[jProc]; i++)
										(iter++)->recv(mpiGroup, jProc, ik);
								}
							}
							std::swap(kp.GePh, GePh);
						}
						else
						{	for(LindbladFile::GePhEntry g: kp.GePh)
								g.send(mpiGroup, whose, ik);
							kp.GePh.clear();
						}
					}
				}
			}
			else
			{	fw.eLoop(k01, initKpoint, this, &mask);
			}
			
			//Save active k-points to global array:
			for(size_t ik=ikStart; ik<ikStop; ik++)
			{	std::swap(kpAll[ik], kpOffset[offK[ik][2]]);
				if(kpAll[ik].E.size())
				{	kpWhose[ik] = mpiWorld->iProcess();
					kpSize[ik] = kpAll[ik].nBytes(h);
					kpGcount[ik] = kpAll[ik].GePh.size();
					std::sort(kpAll[ik].GePh.begin(), kpAll[ik].GePh.end()); //sort e-ph matrix elements by partner index
				}
			}
			
			//Print progress:
			if((iOffMine+1)%offInterval==0) { logPrintf("%d%% ", int(round((iOffMine+1)*100./nOffMine))); logFlush(); }
		}
		mpiWorld->allReduceData(kpWhose, MPIUtil::ReduceMax); //now each process knows who owns a specific k-point data
		mpiWorld->allReduceData(kpSize, MPIUtil::ReduceMax); //... and its size when written to file
		mpiWorld->allReduceData(kpGcount, MPIUtil::ReduceMax);
		logPrintf("done.\n"); logFlush();
		GcountTot = 0;
		for(size_t Gcount: kpGcount) GcountTot += Gcount;

		//Compute offsets to each k-point within file:
		std::vector<size_t> byteOffsets(h.nk);
		byteOffsets[0] = h.nBytes() + h.nk*sizeof(size_t); //offset to first k-point (header + byteOffsets array)
		for(size_t ik=0; ik+1<h.nk; ik++)
			byteOffsets[ik+1] = byteOffsets[ik] + kpSize[ik];
		
		//Write file:
		//--- Open file
		#ifdef MPI_SAFE_WRITE
		FILE* fp = NULL;
		if(mpiWorld->isHead()) fp = fopen(outFile.c_str(), "w"); //I/O from world head alone
		#else
		MPIUtil::File fp;
		mpiWorld->fopenWrite(fp, outFile.c_str()); //I/O collectively from all processes
		#endif
		//--- Header and byte offsets to each k
		if(mpiWorld->isHead())
		{	std::ostringstream oss;
			h.write(oss);
			#ifdef MPI_SAFE_WRITE
			fwrite(oss.str().data(), 1, h.nBytes(), fp);
			fwrite(byteOffsets.data(), sizeof(size_t), byteOffsets.size(), fp);
			#else
			mpiWorld->fwrite(oss.str().data(), 1, h.nBytes(), fp);
			mpiWorld->fwriteData(byteOffsets, fp);
			#endif
		}
		//--- Write data:

		logPrintf("Writing %s: ", outFile.c_str()); logFlush();
		size_t ikInterval = std::max(1, int(round(h.nk/50.))); //interval for reporting progress
		for(size_t ik=0; ik<h.nk; ik++)
		{	const LindbladFile::Kpoint& kp = kpAll[ik];
			std::vector<char> buf(kpSize[ik]); //buffer containing serialization of kp
			bool isMine = (kpWhose[ik] == mpiWorld->iProcess());
			if(isMine)
			{	membuf mbuf(buf);
				std::ostream os(&mbuf);
				kp.write(os, h);
				#ifdef MPI_SAFE_WRITE
				if(not mpiWorld->isHead()) //send to head to write:
					mpiWorld->sendData(buf, 0, ik);
				#else
				//Write from each process in parallel:
				mpiWorld->fseek(fp, byteOffsets[ik], SEEK_SET);
				mpiWorld->fwrite(buf.data(), 1, kpSize[ik], fp);
				#endif
			}
			#ifdef MPI_SAFE_WRITE
			//Write data from head:
			if(mpiWorld->isHead())
			{	if(not isMine) //Recv data to write
					mpiWorld->recvData(buf, kpWhose[ik], ik);
				fseek(fp, byteOffsets[ik], SEEK_SET);
				fwrite(buf.data(), 1, kpSize[ik], fp);
			}
			#endif
			//Print progress:
			if((ik+1)%ikInterval==0) { logPrintf("%d%% ", int(round((ik+1)*100./h.nk))); logFlush(); }
		}
		logPrintf("done.\n"); logFlush();
		///--- Close file:
		#ifdef MPI_SAFE_WRITE
		if(mpiWorld->isHead()) fclose(fp);
		#else
		mpiWorld->fclose(fp);
		#endif

#ifdef HDF5_ENABLED
		//---------- HDF5 output ---------
		if (enableH5Output)
		{
			hid_t fid = openHDF5(outFileH5);
			
			//Prepate h5py compatible datatype for complex
			h5_complex = H5Tcreate(H5T_COMPOUND, sizeof(complex));
			H5Tinsert(h5_complex, "r", HOFFSET(complex, x), H5T_NATIVE_DOUBLE);
			H5Tinsert(h5_complex, "i", HOFFSET(complex, y), H5T_NATIVE_DOUBLE);

			//Prepate h5py compatible datatype for bool
			h5_bool = H5Tenum_create (H5T_STD_I8LE);
			bool trueVal = true;
			bool falseVal = false;
			H5Tenum_insert(h5_bool, "FALSE", &falseVal);
			H5Tenum_insert(h5_bool, "TRUE", &trueVal);
			
			//--- Write scalars
			h5writeScalarAttr(fid, "Tmax", h.Tmax);
			h5writeScalarAttr(fid, "dmuMax", h.dmuMax);
			h5writeScalarAttr(fid, "dmuMin", h.dmuMin);
			h5writeScalarAttr(fid, "nk", (int)h.nk);
			h5writeScalarAttr(fid, "nkTot", (int)h.nkTot);
			h5writeScalarAttr(fid, "probeOmegaMax", h.probeOmegaMax);
			h5writeScalarAttr(fid, "pumpOmegaMax", h.pumpOmegaMax);
			h5writeScalarAttr(fid, "spinWeight", h.spinWeight);
			h5writeScalarAttr(fid, "ePhEnabled", h.ePhEnabled);
			h5writeScalarAttr(fid, "haveL", h.haveL);
			h5writeScalarAttr(fid, "spinorial", h.spinorial);

			//Write lattice vectors:
			{	double R[9] = { 
					h.R(0, 0), h.R(0, 1), h.R(0, 2), 
					h.R(1, 0), h.R(1, 1), h.R(1, 2),
					h.R(2, 0), h.R(2, 1), h.R(2, 2)
				};
				hsize_t dimsR[2] = { 3, 3 };
				h5writeVectorAttr(fid, "R", R, dimsR, 2);
			}
			
			//Write k resolution:
			{	hsize_t dims_nk[1] = {3};
				h5writeVectorAttr(fid, "nk_grid", &NkFine[0], dims_nk, 1);
			}
			
			//Create datasets collectively:
			hsize_t nBands = bandCountFixed;
			hid_t did_k = h5createVectorDataset<double>(fid, "k", hsizes({h.nk, 3}), 2);
			hid_t did_E = h5createVectorDataset<double>(fid, "E", hsizes({h.nk, nBands}), 2);
			hid_t did_k_adj = h5createVectorDataset<int>(fid, "k_adj", hsizes({h.nk, 3, 5}), 3);
			hid_t did_U = h5createVectorDataset<complex>(fid, "U", hsizes({h.nk, 3, 5, nBands, nBands}), 5);
			hid_t did_P = h5createVectorDataset<complex>(fid, "P", hsizes({h.nk, 3, nBands, nBands}), 4);		
			hid_t did_S=0, did_L=0, did_G=0, did_omegaPh=0, did_ikpair=0;
			if(h.spinorial) did_S = h5createVectorDataset<complex>(fid, "S", hsizes({h.nk, 3, nBands, nBands}), 4);
			if(h.haveL) did_L = h5createVectorDataset<complex>(fid, "L", hsizes({h.nk, 3, nBands, nBands}), 4);
			if(h.ePhEnabled)
			{	did_G = h5createVectorDataset<complex>(fid, "G", hsizes({GcountTot, nBands, nBands}), 3);
				did_omegaPh = h5createVectorDataset<double>(fid, "omega_ph", hsizes({GcountTot}), 1);
				did_ikpair = h5createVectorDataset<int>(fid, "ikpair", hsizes({GcountTot, 2}), 2);
			}
			
			size_t GcountPrev = 0;
			for(size_t ik=0; ik<h.nk; ik++)
			{	if(kpWhose[ik] == mpiWorld->iProcess())
				{	const LindbladFile::Kpoint& kp = kpAll[ik];
	

					// Calculate adjacency and phase matching
					std::vector<int> k_adj; k_adj.reserve(15);
					std::vector<complex> U; U.reserve(15*nBands*nBands);
					const LindbladFile::Kpoint& kp_self = kpAll[ik];
					for (int dir=0; dir<3; dir++)
					{	std::vector<int> adj_arr(5);
						vector3<> offset; offset[dir] = 1.0 / NkFine[dir];
						
						int k_ind_prev = ik;
						size_t jk_ind;

						matrix kp_j_U;
						LindbladFile::Kpoint kp_j;
						
						// Get cells to the "left"
						for(int cell=1; cell>=0; cell--)
						{	vector3<double> jk = k[ik] + (cell - 2) * offset;
							bool exists = findK(jk, jk_ind);
							if (exists)
							{	adj_arr[cell] = jk_ind;
								k_ind_prev = jk_ind;
							}
							else
							{	// If no adjacent point exists, insert the previous point (moving outward)
								adj_arr[cell] = k_ind_prev;
							}
							kp_j = kpAll[adj_arr[cell]];

							// Get rotations, using selected energy levels at kp_j (for calculating degeneracy)
							kp_j_U = getRotation(kp_self.U, kp_j.U, 
									std::vector<double>(kp_j.E.begin() + kp_j.innerOffset, kp_j.E.begin() + kp_j.innerOffset + kp_j.nInner));
							std::vector<complex> kp_j_U_flat = unravelMatrix(kp_j_U);
							U.insert(U.end(), kp_j_U_flat.begin(), kp_j_U_flat.end());
						}
						
						adj_arr[2] = ik;
						k_ind_prev = ik;
						kp_j_U = eye(nBands);

						std::vector<complex> kp_j_U_flat = unravelMatrix(kp_j_U);
						U.insert(U.end(), kp_j_U_flat.begin(), kp_j_U_flat.end());
						
						//Get cells to the "right"
						for(int cell=3; cell<5; cell++)
						{	vector3<double> jk = k[ik] + (cell - 2) * offset;
							bool exists = findK(jk, jk_ind);
							if (exists)
							{	adj_arr[cell] = jk_ind;
								k_ind_prev = jk_ind;
							}
							else
							{	// If no adjacent point exists, insert the previous point (moving outward)
								adj_arr[cell] = k_ind_prev;
							}
							kp_j = kpAll[adj_arr[cell]];

							// Get rotations, using selected energy levels at kp_j (for calculating degeneracy)
							kp_j_U = getRotation(kp_self.U, kp_j.U, 
									std::vector<double>(kp_j.E.begin() + kp_j.innerOffset, kp_j.E.begin() + kp_j.innerOffset + kp_j.nInner));
							std::vector<complex> kp_j_U_flat = unravelMatrix(kp_j_U);
							U.insert(U.end(), kp_j_U_flat.begin(), kp_j_U_flat.end());
						}
						k_adj.insert(k_adj.end(), adj_arr.begin(), adj_arr.end());
					}
					
					h5writeVectorSlice(did_k, &kp.k[0], hsizes({ik, 0}), hsizes({1, 3}), 2);
					h5writeVectorSlice(did_E, kp.E.data(), hsizes({ik, 0}), hsizes({1, nBands}), 2);
					h5writeVectorSlice(did_k_adj, k_adj.data(), hsizes({ik, 0, 0}), hsizes({1, 3, 5}), 3);
					h5writeVectorSlice(did_U, U.data(), hsizes({ik, 0, 0, 0, 0}), hsizes({1, 3, 5, nBands, nBands}), 5);
					h5writeVectorSlice(did_P, getContiguousData(kp.P).data(), hsizes({ik, 0, 0, 0}), hsizes({1, 3, nBands, nBands}), 4);
					if(h.spinorial) h5writeVectorSlice(did_S, getContiguousData(kp.S).data(), hsizes({ik, 0, 0, 0}), hsizes({1, 3, nBands, nBands}), 4);
					if(h.haveL) h5writeVectorSlice(did_L, getContiguousData(kp.L).data(), hsizes({ik, 0, 0, 0}), hsizes({1, 3, nBands, nBands}), 4);
					
					if(h.ePhEnabled)
					{	std::vector<double> omegaPh; omegaPh.reserve(kp.GePh.size());
						std::vector<int> ikpair; ikpair.reserve(kp.GePh.size() * 2);
						std::vector<complex> G; G.reserve(kp.GePh.size() * nBands * nBands);
						for(const LindbladFile::GePhEntry& g: kp.GePh)
						{	omegaPh.push_back(g.omegaPh);
							ikpair.push_back(ik);
							ikpair.push_back(g.jk);

							matrix Gdense = zeroes(bandCountFixed, bandCountFixed);
							for(const SparseEntry& Gentry: g.G)
								Gdense.set(Gentry.i, Gentry.j, Gentry.val);
						
							for(unsigned iBand=0; iBand<nBands; iBand++)
								for(unsigned jBand=0; jBand<nBands; jBand++)
									G.push_back(Gdense(iBand, jBand));
						}
						h5writeVectorSlice(did_omegaPh, omegaPh.data(), hsizes({GcountPrev}), hsizes({kpGcount[ik]}), 1);
						h5writeVectorSlice(did_ikpair, ikpair.data(), hsizes({GcountPrev, 0}), hsizes({kpGcount[ik], 2}), 2);
						h5writeVectorSlice(did_G, G.data(), hsizes({GcountPrev, 0, 0}), hsizes({kpGcount[ik], nBands, nBands}), 3);					
					}
				}
				GcountPrev += kpGcount[ik];
				if((ik+1)%ikInterval==0) { logPrintf("%d%% ", int(round((ik+1)*100./h.nk))); logFlush(); }
			}
			H5Dclose(did_k);
			H5Dclose(did_E);
			H5Dclose(did_k_adj);
			H5Dclose(did_U);
			H5Dclose(did_P);
			if(h.spinorial) H5Dclose(did_S);
			if(h.haveL) H5Dclose(did_L);
			if(h.ePhEnabled)
			{	H5Dclose(did_omegaPh);
				H5Dclose(did_ikpair);
				H5Dclose(did_G);
			}
			H5Fclose(fid);
			logPrintf("done.\n"); logFlush();
		}
#endif
	}
};

int main(int argc, char** argv)
{	
	InitParams ip = FeynWann::initialize(argc, argv, "Initialize sparse matrices for Lindblad dynamics");

	//Get the system parameters:
	InputMap inputMap(ip.inputFilename);
	//--- kpoints
	const int NkMultAll = int(round(inputMap.get("NkMult"))); //increase in number of k-points for phonon mesh
	vector3<int> NkMult;
	NkMult[0] = inputMap.get("NkxMult", NkMultAll); //override increase in x direction
	NkMult[1] = inputMap.get("NkyMult", NkMultAll); //override increase in y direction
	NkMult[2] = inputMap.get("NkzMult", NkMultAll); //override increase in z direction
	//--- doping / temperature
	const double dmuMin = inputMap.get("dmuMin", 0.) * eV; //optional: lowest shift in fermi level from neutral value / VBM in eV (default: 0)
	const double dmuMax = inputMap.get("dmuMax", 0.) * eV; //optional: highest shift in fermi level from neutral value / VBM in eV (default: 0)
	const double Tmax = inputMap.get("Tmax") * Kelvin; //maximum temperature in Kelvin (ambient phonon T = initial electron T)
	//--- which bands to use
	const int bandCountFixed = inputMap.get("bandCountFixed", 0); //if nonzero, fix number of inner bands (no separate outer)
	const string bandSelectionStr = inputMap.has("bandSelection") ? inputMap.getString("bandSelection") : "all-bands";
	BandSelection bandSelection;
	if(not bandSelectionMap.getEnum(bandSelectionStr.c_str(), bandSelection))
		die("bandSelection must be one of %s\n", bandSelectionMap.optionList().c_str());
	if(bandCountFixed and (bandSelection == AllBands))
		die("bandSelection must be ElectronsOnly or HolesOnly to use fixed band count.\n");
	//--- pump
	const double pumpOmegaMax = inputMap.get("pumpOmegaMax") * eV; //maximum pump frequency in eV
	const double probeOmegaMax = inputMap.get("probeOmegaMax") * eV; //maximum probe frequency in eV
	const string ePhMode = inputMap.getString("ePhMode"); //must be Off or DiagK (add FullK in future)
	const string defectName = inputMap.has("defectName") ? inputMap.getString("defectName") : "Off"; //optional defect contribution
	const bool ePhEnabled = (ePhMode != "Off");
	const bool defectEnabled = (defectName != "Off");
	const double ePhDelta = inputMap.get("ePhDelta") * eV; //energy conservation width for e-ph coupling
	const size_t maxNeighbors = inputMap.get("maxNeighbors", 0); //if non-zero: limit neighbors per k by stochastic down-sampling and amplifying the Econserve weights
	const string outFile = inputMap.has("outFile") ? inputMap.getString("outFile") : "ldbd.dat"; //output file name

        // H5 output options
	const string h5OutputOption = inputMap.has("enableH5Output") ? inputMap.getString("enableH5Output") : "no"; //optional defect contribution
	const bool enableH5Output = (h5OutputOption == "yes");
	if(enableH5Output & (bandCountFixed <= 0))
		die("bandCountFixed must be set to a non-zero value to enable HDF5 output.\n");
	const string outFileH5 = inputMap.has("outFileH5") ? inputMap.getString("outFileH5") : "ldbd.h5"; //output file name

#ifndef HDF5_ENABLED
	if (enableH5Output)
		die("HDF5 output is enabled, however FeynWann has not been compiled with HDF5 support.\n");
#endif

	FeynWannParams fwp(&inputMap);
	
	logPrintf("\nInputs after conversion to atomic units:\n");
	logPrintf("NkMult = "); NkMult.print(globalLog, " %d ");
	logPrintf("dmuMin = %lg\n", dmuMin);
	logPrintf("dmuMax = %lg\n", dmuMax);
	logPrintf("Tmax = %lg\n", Tmax);
	logPrintf("bandSelection = %s\n", bandSelectionMap.getString(bandSelection));
	logPrintf("bandCountFixed = %d\n", bandCountFixed);
	logPrintf("pumpOmegaMax = %lg\n", pumpOmegaMax);
	logPrintf("probeOmegaMax = %lg\n", probeOmegaMax);
	logPrintf("ePhMode = %s\n", ePhMode.c_str());
	logPrintf("defectName = %s\n", defectName.c_str());
	logPrintf("ePhDelta = %lg\n", ePhDelta);
	logPrintf("maxNeighbors = %lu\n", maxNeighbors);
	logPrintf("outFile = %s\n", outFile.c_str());
	logPrintf("outFileH5 = %s\n", outFileH5.c_str());
	logPrintf("enableH5Output = %s\n", h5OutputOption.c_str());
	fwp.printParams();
	
	//Initialize FeynWann:
	fwp.needVelocity = true;
	fwp.needSpin = true;
	fwp.needL = fwp.orbitalZeeman;
	fwp.needPhonons = ePhEnabled;
    if(defectEnabled)
		fwp.needDefect = defectName;
	fwp.maskOptimize = true;
	FeynWann fw(fwp);
	
	//Construct mesh of k-offsets:
	std::vector<vector3<>> k0;
	vector3<int> NkFine;
	for(int iDir=0; iDir<3; iDir++)
	{	if(fw.isTruncated[iDir] && NkMult[iDir]!=1)
		{	logPrintf("Setting NkMult = 1 along truncated direction %d.\n", iDir+1);
			NkMult[iDir] = 1; //no multiplication in truncated directions
		}
		NkFine[iDir] = fw.kfold[iDir] * NkMult[iDir];
	}
	matrix3<> NkFineInv = inv(Diag(vector3<>(NkFine)));
	vector3<int> ikMult;
	for(ikMult[0]=0; ikMult[0]<NkMult[0]; ikMult[0]++)
	for(ikMult[1]=0; ikMult[1]<NkMult[1]; ikMult[1]++)
	for(ikMult[2]=0; ikMult[2]<NkMult[2]; ikMult[2]++)
		k0.push_back(NkFineInv * ikMult);
	logPrintf("Effective interpolated k-mesh dimensions: ");
	NkFine.print(globalLog, " %d ");
	size_t nKeff = k0.size() * fw.eCountPerOffset() * fw.qOffset.size();
	logPrintf("Effectively sampled nKpts: %lu\n", nKeff);
	
	//Construct mesh of q-offsets:
	std::vector<vector3<>> q0;
	if(ePhEnabled)
	{	vector3<int> NqMult;
		for(int iDir=0; iDir<3; iDir++)
			NqMult[iDir] = NkFine[iDir] / fw.phononSup[iDir];
		vector3<int> iqMult;
		for(iqMult[0]=0; iqMult[0]<NqMult[0]; iqMult[0]++)
		for(iqMult[1]=0; iqMult[1]<NqMult[1]; iqMult[1]++)
		for(iqMult[2]=0; iqMult[2]<NqMult[2]; iqMult[2]++)
			q0.push_back(NkFineInv * iqMult);
	}
	
	//Create and initialize lindblad calculator:
	LindbladInit lb(fw, NkFine, dmuMin, dmuMax, Tmax, bandSelection, bandCountFixed, pumpOmegaMax, probeOmegaMax, ePhEnabled, ePhDelta, defectEnabled);
	
	//First pass (e only): select k-points
	fw.energyOnly = true;
	lb.kpointSelect(k0); //only screens energies (coarsely, without E, B perturbations included)
	fw.energyOnly = false;
	if(mpiWorld->isHead()) logPrintf("%lu active q-mesh offsets parallelized over %d process groups.\n", lb.offKuniq.size(), mpiGroupHead->nProcesses());
	
	if(ip.dryRun)
	{	logPrintf("Dry run successful: commands are valid and initialization succeeded.\n");
		fw.free();
		FeynWann::finalize();
		return 0;
	}
	logPrintf("\n");
	
	//Second pass (ph only): select k pairs
	if(ePhEnabled)
		lb.kpairSelect(q0, maxNeighbors);
	
	//Final pass: output electronic and e-ph quantities
	lb.saveData(k0, outFile, enableH5Output, outFileH5);
	
	//Cleanup:
	fw.free();
	FeynWann::finalize();
	return 0;
}
