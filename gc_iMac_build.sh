#!/bin/bash
module swap intel/18.0.1.163 intel/17.0.3.191
mkdir build
cd build

export CRAYPE_LINK_TYPE="dynamic"
export PARMETIS_INSTALL="~/Cori/my_software/parmetis-4.0.3"
cmake .. \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_INSTALL_PREFIX=. \
-DCMAKE_CXX_COMPILER=CC \
-DCMAKE_C_COMPILER=cc \
-DCMAKE_Fortran_COMPILER=ftn \
-DCMAKE_EXE_LINKER_FLAGS="" \
-DCMAKE_CXX_FLAGS="-std=c++11 " \
-DMETIS_INCLUDES=$PARMETIS_INSTALL/metis/include \
-DMETIS_LIBRARIES=$PARMETIS_INSTALL/build/Linux-x86_64/libmetis/libmetis.a \
-DHMATRIX_LIBRARIES=~/Cori/my_research/STRUMPACK_H_RBF/h_matrix_rbf_randomization/build/SRC/libhmatrix.a
# -DHMATRIX_INCLUDES=$PARMETIS_INSTALL/include \

make install VERBOSE=1
cd examples
make
