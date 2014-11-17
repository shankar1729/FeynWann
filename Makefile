JDFTX_BUILD_DIR=/home/shankar/DFT/Code/JDFTx/build_testing
JDFTX_SRC_DIR=/home/shankar/DFT/Code/JDFTx/jdftx

CXX_FLAGS=-g -Wall -O3 -std=c++0x -I$(JDFTX_SRC_DIR) -DENABLE_PROFILING
LINK_FLAGS=-L$(JDFTX_BUILD_DIR) -Wl,-rpath,$(JDFTX_BUILD_DIR) -ljdftx

all: testBandStruct testEpsilon plasmonDecayMetropolis plasmonPhononDecay

testBandStruct: testBandStruct.cpp BandStruct.h BandStruct.cpp
	g++ -o testBandStruct testBandStruct.cpp BandStruct.cpp $(CXX_FLAGS) $(LINK_FLAGS)

testEpsilon: testEpsilon.cpp Epsilon.h Epsilon.cpp
	g++ -o testEpsilon testEpsilon.cpp Epsilon.cpp $(CXX_FLAGS) $(LINK_FLAGS)

plasmonDecayMetropolis: plasmonDecayMetropolis.cpp BandStruct.h BandStruct.cpp Histogram.h Histogram.cpp Epsilon.h Epsilon.cpp
	mpicxx -o plasmonDecayMetropolis plasmonDecayMetropolis.cpp BandStruct.cpp Histogram.cpp Epsilon.cpp $(CXX_FLAGS) $(LINK_FLAGS) -DMPI_ENABLED

plasmonPhononDecay: plasmonPhononDecay.cpp BandStruct.h BandStruct.cpp Histogram.h Histogram.cpp Epsilon.h Epsilon.cpp LineWidth.h LineWidth.cpp
	mpicxx -o plasmonPhononDecay plasmonPhononDecay.cpp BandStruct.cpp Histogram.cpp Epsilon.cpp LineWidth.cpp $(CXX_FLAGS) $(LINK_FLAGS) -DMPI_ENABLED
