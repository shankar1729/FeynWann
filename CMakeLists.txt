project(WannierMC)
cmake_minimum_required(VERSION 2.8)

set(JDFTX_SRC "<full-path-to-jdftx-src>" CACHE STRING "Full path to the jdftx source root")
set(JDFTX_BUILD "<full-path-to-jdftx-build>" CACHE STRING "Full path to the jdftx build directory")

set(CMAKE_MODULE_PATH "${JDFTX_SRC}/CMake-Modules/")

#Static linking:
if(StaticLinking)
        if(WIN32)
                set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
        else()
                set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
        endif()
        set(CMAKE_EXE_LINKER_FLAGS -static)
        set(CMAKE_EXE_LINK_DYNAMIC_C_FLAGS)       # remove -Wl,-Bdynamic
        set(CMAKE_EXE_LINK_DYNAMIC_CXX_FLAGS)
        set(CMAKE_SHARED_LIBRARY_C_FLAGS)         # remove -fPIC
        set(CMAKE_SHARED_LIBRARY_CXX_FLAGS)
        set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS)    # remove -rdynamic
        set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS)
endif()

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

find_package(FFTW3 REQUIRED)
include_directories(${FFTW3_INCLUDE_DIR})

option(EnableLibXC "Use LibXC to provide additional exchange-correlation functionals")
if(EnableLibXC)
	add_definitions("-DLIBXC_ENABLED")
endif()

include_directories("${JDFTX_SRC}")
include_directories(${CMAKE_SOURCE_DIR})
add_definitions(-Wall -g -O3 -std=c++0x ${MPI_CXX_COMPILE_FLAGS})

#Link options:
if(StaticLinking)
        set(LINK_TYPE STATIC)
        set(LINK_PREFIX -Wl,--whole-archive)
        set(LINK_SUFFIX -Wl,--no-whole-archive)
else()
        set(LINK_TYPE SHARED)
endif()

#Main library for WannierMC containing all the auxiliary functionality
add_library(WannierMC ${LINK_TYPE} BandStruct.cpp LineWidth.cpp InputMap.cpp)
find_library(JDFTX_LIBRARY jdftx PATHS ${JDFTX_BUILD})
target_link_libraries(WannierMC ${JDFTX_LIBRARY} ${MPI_CXX_LIBRARIES} ${EXTRA_LIBRARIES})

#Executables linked to above library
macro(add_WannierMC_executable targetName)
	add_executable(${targetName} ${targetName}.cpp)
	target_link_libraries(${targetName} ${LINK_PREFIX} WannierMC ${LINK_SUFFIX})
	set_target_properties(${targetName} PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endmacro()

add_WannierMC_executable(resistivity)
add_WannierMC_executable(electronPhononLinewidth)
