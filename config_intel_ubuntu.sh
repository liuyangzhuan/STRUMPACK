#!/bin/bash

mkdir build
cd build

export CRAYPE_LINK_TYPE="dynamic"
export PARMETIS_INSTALL="/home/administrator/Desktop/software/parmetis-4.0.3"
cmake .. \
-DCMAKE_BUILD_TYPE=Debug \
-DCMAKE_INSTALL_PREFIX=. \
-DCMAKE_CXX_COMPILER=mpiicpc \
-DCMAKE_CXX_FLAGS="-I/home/administrator/Desktop/research/STRUMPACK_H_RBF/hod-lr-bf/SRC_DOUBLE" \
-DMPI_CXX_COMPILER=/opt/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/bin/mpiicpc \
-DCMAKE_C_COMPILER=mpiicc \
-DMPI_C_COMPILER=/opt/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/bin/mpiicc \
-DCMAKE_EXE_LINKER_FLAGS="-qopt-matmul -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64" \
-DCMAKE_Fortran_COMPILER=mpiifort \
-DMPI_Fortran_COMPILER=/opt/intel/compilers_and_libraries_2018.1.163/linux/mpi/intel64/bin/mpiifort \
-DMETIS_INCLUDES=$PARMETIS_INSTALL/metis/include \
-DMETIS_LIBRARIES=$PARMETIS_INSTALL/build/Linux-x86_64/libmetis/libmetis.a \
-DHMATRIX_LIBRARIES="/home/administrator/Desktop/research/STRUMPACK_H_RBF/h_matrix_rbf_randomization/build/SRC/libhmatrix.a;/home/administrator/Desktop/research/STRUMPACK_H_RBF/hod-lr-bf/build/SRC_DOUBLE/libdhodlrbf.a" \
-DSCALAPACK_LIBRARIES="/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64/libmkl_blacs_intelmpi_lp64.a;/opt/intel/compilers_and_libraries_2018.1.163/linux/mkl/lib/intel64/libmkl_scalapack_lp64.a"

make install VERBOSE=1
cd examples
make


