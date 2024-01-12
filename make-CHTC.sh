#!/bin/bash

module purge
module load openmpi/4.1.3-gcc-11.3.0
module load cmake
module list
CC=mpicc CXX=mpicxx FC=mpifort cmake \
	-D MKL_PATH="/software/groups/ping_group/shared/libs/mkl-2023.2.0/mkl/2023.2.0" \
	-D EnableScaLAPACK=yes \
	-D FFTW3_PATH="/software/groups/ping_group/shared/libs/fftw-3.3.10/build" \
	-D GSL_PATH="/software/groups/ping_group/shared/libs/gsl-2.7.1/build" \
	-D EnableLibXC=yes \
	-D LIBXC_PATH="/software/groups/ping_group/shared/libs/libxc-6.2.2/build" \
	-D EnableProfiling=yes \
        -D JDFTX_BUILD="/software/groups/ping_group/shared/apps/jdftx-1.7.0/build" \
        -D JDFTX_SRC="/software/groups/ping_group/shared/apps/jdftx-1.7.0/jdftx" \
.

make -j8
