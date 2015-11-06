project(WannierMetropolis)
cmake_minimum_required(VERSION 2.8)

set(JDFTX_SRC "<full-path-to-jdftx-src>" CACHE STRING "Full path to the jdftx source root")
set(JDFTX_BUILD "<full-path-to-jdftx-build>" CACHE STRING "Full path to the jdftx build directory")

set(CMAKE_MODULE_PATH "${JDFTX_SRC}/CMake-Modules/")

#Whether Profiling is on or off should match the corresponding JDFTx build:
option(EnableProfiling "Enable timers and print profiling output" ON)
if(EnableProfiling)
	add_definitions("-DENABLE_PROFILING")
endif()

#Whether MPI is on or off should match the corresponding JDFTx build:
option(EnableMPI "Use MPI parallelization (in addition to threads / gpu)" ON)
if(EnableMPI)
	find_package(MPI REQUIRED)
	include_directories(${MPI_INCLUDE_PATH})
	add_definitions("-DMPI_ENABLED")
endif()

find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIR})

option(EnableLibXC "Use LibXC to provide additional exchange-correlation functionals")
if(EnableLibXC)
	add_definitions("-DLIBXC_ENABLED")
endif()

include_directories("${JDFTX_SRC}")
include_directories(${CMAKE_SOURCE_DIR})
add_definitions(-Wall -g -O3 -std=c++0x ${MPI_CXX_COMPILE_FLAGS})

#Main library for WannierMetropolis containing all the auxiliary functionality
add_library(WannierMetropolis SHARED BandStruct.cpp Epsilon.cpp Histogram.cpp LineWidth.cpp InputMap.cpp)
target_link_libraries(WannierMetropolis ${JDFTX_BUILD}/libjdftx.so ${MPI_CXX_LIBRARIES})

#Executables linked to above library
macro(add_WannierMetropolis_executable targetName)
	add_executable(${targetName} ${targetName}.cpp)
	target_link_libraries(${targetName} WannierMetropolis)
	set_target_properties(${targetName} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endmacro()

add_WannierMetropolis_executable(testBandStruct)
add_WannierMetropolis_executable(testEpsilon)
add_WannierMetropolis_executable(resistivity)
add_WannierMetropolis_executable(resistivityFull)
add_WannierMetropolis_executable(eventList)
add_WannierMetropolis_executable(plasmonDecayMetropolis)
add_WannierMetropolis_executable(twoPlasmonDecay)
add_WannierMetropolis_executable(plasmonPhononDecay)
add_WannierMetropolis_executable(plasmonDecay)
add_WannierMetropolis_executable(electronPhononLinewidth)
add_WannierMetropolis_executable(drudeGamma)
add_WannierMetropolis_executable(surfaceAssisted)
add_WannierMetropolis_executable(EffectiveMass)
add_WannierMetropolis_executable(TAparameters)
add_WannierMetropolis_executable(phononDOS)
add_WannierMetropolis_executable(calcEsTeTau)
add_WannierMetropolis_executable(phononEnergy)
add_WannierMetropolis_executable(MepAshcroft)
add_WannierMetropolis_executable(FermiSurface)
