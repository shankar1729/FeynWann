JDFTX_BUILD_DIR=/home/shankar/DFT/Code/JDFTx/build_testing
JDFTX_SRC_DIR=/home/shankar/DFT/Code/JDFTx/jdftx

CXX_FLAGS=-g -Wall -O3 -std=c++0x -I$(JDFTX_SRC_DIR) -DENABLE_PROFILING
LINK_FLAGS=-L$(JDFTX_BUILD_DIR) -Wl,-rpath,$(JDFTX_BUILD_DIR) -ljdftx

all: WannierBandstruct test testBandStructClass plasmonDecayMetropolis


WannierBandstruct: WannierBandstruct.cpp
	g++ -o WannierBandstruct WannierBandstruct.cpp $(CXX_FLAGS) $(LINK_FLAGS)

test: test.cpp histogram.h histogram.cpp
	g++ -o test test.cpp histogram.cpp $(CXX_FLAGS) $(LINK_FLAGS)

testBandStructClass: testBandStructClass.cpp bandStruct.h bandStruct.cpp
	g++ -o testBandStructClass testBandStructClass.cpp bandStruct.cpp $(CXX_FLAGS) $(LINK_FLAGS)

plasmonDecayMetropolis: plasmonDecayMetropolis.cpp bandStruct.h bandStruct.cpp histogram.h histogram.cpp epsilon.h epsilon.cpp
	mpicxx -o plasmonDecayMetropolis plasmonDecayMetropolis.cpp bandStruct.cpp histogram.cpp epsilon.cpp $(CXX_FLAGS) $(LINK_FLAGS) -DMPI_ENABLED
