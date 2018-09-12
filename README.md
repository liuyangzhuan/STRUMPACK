# STRUMPACK
STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014-2017, The
Regents of the University of California, through Lawrence Berkeley
National Laboratory (subject to receipt of any required approvals from
the U.S. Dept. of Energy).  All rights reserved.

[![Build Status](https://travis-ci.org/pghysels/STRUMPACK.svg?branch=master)](https://travis-ci.org/pghysels/STRUMPACK)

## Installation instructions
   See INSTALL.txt


## Website
   [http://portal.nersc.gov/project/sparse/strumpack/](http://portal.nersc.gov/project/sparse/strumpack/)


## Current developers
 - Pieter Ghysels - pghysels@lbl.gov (Lawrence Berkeley National Laboratory)
 - Xiaoye S. Li - xsli@lbl.gov (Lawrence Berkeley National Laboratory)
 - Gustavo Chávez - gichavez@lbl.gov (Lawrence Berkeley National Laboratory)
 - Yang Liu - liuyangzhuan@lbl.gov (Lawrence Berkeley National Laboratory)

## Other contributors
 - Lucy Guo - lcguo@lbl.gov
 - Liza Rebrova - erebrova@umich.edu (University of Michigan)
 - François-Henry Rouet - fhrouet@lbl.gov,fhrouet@lstc.com (Livermore
   Software Technology Corp., Lawrence Berkeley National Laboratory)
 - Theo Mary - theo.mary@manchester.ac.uk (University of Manchester)
 - Christopher Gorman - (UC Santa Barbara)
 - Jonas Actor - (Rice University)

## Overview
STRUMPACK - STRUctured Matrix PACKage - is a software library
providing linear algebra routines for sparse matrices and for dense
rank-structured matrices, i.e., matrices that exhibit some kind of
low-rank property. In particular, STRUMPACK uses the Hierarchically
Semi-Separable matrix format (HSS).  Such matrices appear in many
applications, e.g., Finite Element Methods, Boundary Element Methods
... In sparse matrix factorization, the fill-in in the triangular
factors often has a low-rank structure. Hence, the sparse linear
solve in STRUMPACK exploits the HSS matrix format to compress the
fill-in. Exploiting this structure using a compression algorithm
allows for fast solution of linear systems and/or fast computation of
matrix-vector products, which are two of the main building blocks of
matrix computations. STRUMPACK has two main components: a
distributed-memory dense matrix computations package (for dense
matrices that have the HSS structure) and a distributed memory fully
algebraic sparse general solver and preconditioner. The preconditioner
is mostly aimed at large sparse linear systems which result from the
discretization of a partial differential equation, but is not limited
to any particular type of problem. STRUMPACK also provides
preconditioned GMRES and BiCGStab iterative solvers.

##  Components
 - The sparse solver is documented in doc/manual.pdf. STRUMPACK-sparse
   can be used as a direct solver for sparse linear systems or as a
   preconditioner. It also includes GMRes and BiCGStab iterative
   solvers that can use the preconditioner. The preconditioning
   strategy is based on applying low-rank approximations to the
   fill-in of a sparse multifrontal LU factorization.  The code uses
   MPI+OpenMP for hybrid distributed and shared memory parallelism.
   The main point of contact is: Pieter Ghysels (pghysels@lbl.gov).
 - The dense distributed-memory package can be found in the src/HSS
   directory. This is currently not well documented.

If you have questions about your rights to use or distribute this
software, please contact Berkeley Lab's Technology Transfer Department
at TTD@lbl.gov.

## NOTICE
This software is owned by the U.S. Department of Energy.  As
such, the U.S. Government has been granted for itself and others
acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide
license in the Software to reproduce, prepare derivative works, and
perform publicly and display publicly.  Beginning five (5) years after
the date permission to assert copyright is obtained from the
U.S. Department of Energy, and subject to any subsequent five (5) year
renewals, the U.S. Government is granted for itself and others acting
on its behalf a paid-up, nonexclusive, irrevocable, worldwide license
in the Software to reproduce, prepare derivative works, distribute
copies to the public, perform publicly and display publicly, and to
permit others to do so.
