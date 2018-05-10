/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
 * Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals
 * from the U.S. Dept. of Energy).  All rights reserved.
 *
 * If you have questions about your rights to use or distribute this
 * software, please contact Berkeley Lab's Technology Transfer
 * Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As
 * such, the U.S. Government has been granted for itself and others
 * acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, prepare derivative
 * works, and perform publicly and display publicly. Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li,
               Gustavo Chavez.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <atomic>

#include "HSS/HSSMatrixMPI.hpp"
#include "misc/TaskTimer.hpp"

#define ERROR_TOLERANCE 1e1
#define SOLVE_TOLERANCE 1e-11
#define myscalar double

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

int run(int argc, char *argv[]) {

  int n = 8;

  // Initialize timer
  TaskTimer::t_begin = GET_TIME_NOW();
  TaskTimer timer(string("compression"), 1);

  // Setting command line arguments
  if (argc > 1) n = stoi(argv[1]);

  HSSOptions<double> hss_opts;
  hss_opts.set_from_command_line(argc, argv);

  auto np = mpi_nprocs(MPI_COMM_WORLD);
  if (!mpi_rank()) {
    cout << "# usage: ./DenseToeplitzQChem n (problem size)" << endl;
    cout << "# matrix size: n = " << n << endl;
  }

  int npcol = floor(sqrt((float)np));
  int nprow = np / npcol;
  int ctxt, dummy, prow, pcol;
  scalapack::Cblacs_get(0, 0, &ctxt);
  scalapack::Cblacs_gridinit(&ctxt, "C", nprow, npcol);
  scalapack::Cblacs_gridinfo(ctxt, &dummy, &dummy, &prow, &pcol);
  int ctxt_all = scalapack::Csys2blacs_handle(MPI_COMM_WORLD);
  scalapack::Cblacs_gridinit(&ctxt_all, "R", 1, np);

  //========================================
  //=== Build dense (distributed) matrix ===
  //========================================
  if (!mpi_rank())
    cout << "# Building dense matrix A..." << endl;

  timer.start();
  DistributedMatrix<double> A(ctxt, n, n); // Creates descriptor
  
  // Toeplitz matrix from Quantum Chemistry.
  myscalar d = 0.1;
  myscalar d2 = d * d;

  // TODO diagonal element is different from the description in the TOMS paper
  myscalar diag = M_PI * M_PI / (6.0 * d2);
  int rlo, rhi, clo, chi;
  A.lranges(rlo, rhi, clo, chi);
  for (int lc=clo; lc<chi; ++lc) {
    auto c = A.coll2g_fixed(lc);
    for (int lr=rlo; lr<rhi; ++lr) {
      auto r = A.rowl2g_fixed(lr);
      myscalar rc = r - c;
      A(lr, lc) = (r==c) ? diag : pow(-1.0, rc) / (rc*rc * d2);
    }
  }

  if (!mpi_rank()) {
    cout << "## Dense matrix construction time = " << timer.elapsed() << endl;
    cout << "# A.total_memory() = " << (double)A.total_memory()/(1000.0*1000.0) << " MB" << endl;
  }

  if ( hss_opts.verbose() == 1 && n < 8 ){
    cout << "n = " << n << endl;
    A.print("A");
  }

  //===================
  //=== Compression ===
  //===================
  if (!mpi_rank())
    cout << "# Creating HSS matrix H..." << endl;

  if (!mpi_rank()) cout << "# rel_tol = " << hss_opts.rel_tol() << endl;

  // Simple compression
  timer.start();
    HSSMatrixMPI<double> H(A, hss_opts, MPI_COMM_WORLD);
  if (!mpi_rank())
    cout << "## Compression time = " << timer.elapsed() << endl;

  if (H.is_compressed()) {
    if (!mpi_rank()) {
      cout << "# created H matrix of dimension "
           << H.rows() << " x " << H.cols()
           << " with " << H.levels() << " levels" << endl;
      cout << "# compression succeeded!" << endl;
    }
  } else {
    if (!mpi_rank()) cout << "# compression failed!!!!!!!!" << endl;
    // MPI_Abort(MPI_COMM_WORLD, 1);
  }

  auto Hrank = H.max_rank();
  auto Hmem = H.total_memory();
  auto Amem = A.total_memory();
  if (!mpi_rank()) {
    cout << "# rank(H) = " << Hrank << endl;
    cout << "# memory(H) = " << Hmem/1e6 << " MB, " << endl;
    cout << "# mem percentage = " << 100. * Hmem / Amem << "% (of dense)" << endl;
  }

  // Checking error against dense matrix
  if ( n <= 1000 ) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto Hdense = H.dense(A.ctxt());
    MPI_Barrier(MPI_COMM_WORLD);

    Hdense.scaled_add(-1., A);
    auto HnormF = Hdense.normF();
    auto AnormF = A.normF();
    if (!mpi_rank())
      cout << "# relative error = ||A-H*I||_F/||A||_F = "
           << HnormF / AnormF << endl;
    if (A.active() && HnormF / AnormF >
        ERROR_TOLERANCE * max(hss_opts.rel_tol(),hss_opts.abs_tol())) {
      if (!mpi_rank()) cout << "ERROR: compression error too big!!" << endl;
      // MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

// # =====================
// # === Factorization ===
// # =====================
  if (!mpi_rank())
    cout << "# Factorization..." << endl;
  timer.start();

  MPI_Barrier(MPI_COMM_WORLD);
  if (!mpi_rank()) cout << "# computing ULV factorization of HSS matrix .. " << endl;
  
    auto ULV = H.factor();
  if (!mpi_rank()){
    cout << "## Factorization time = " << timer.elapsed() << endl;
    cout << "# ULV.memory() = " << ULV.memory()/(1000.0*1000.0) << "MB" << endl;
  }

// # =============
// # === Solve ===
// # =============
  if (!mpi_rank()) cout << "# Solve..." << endl;

  DistributedMatrix<double> B(ctxt, n, 1);
  B.random();
  DistributedMatrix<double> X(B);

  timer.start();
    H.solve(ULV, X);
  if (!mpi_rank())
    cout << "## Solve time = " << timer.elapsed() << endl;

// # ==================================
// # === Checking relative residual ===
// # ==================================

  auto Bnorm = B.normF();

  DistributedMatrix<double> R(B);
  strumpack::gemm(strumpack::Trans::N, strumpack::Trans::N, 1., A, X, 0., R);
  DistributedMatrix<double> R2 = H.apply(X);
  R.scaled_add(-1., B);
  R2.scaled_add(-1., B);
  double resnorm = R.normF();
  double resnorm2 = R2.normF();

  if (!mpi_rank()){
      cout << "# relative residual = ||A*X-B||_F/||B||_F = "
           << resnorm / Bnorm << endl;
      cout << "# relative residual = ||H*X-B||_F/||B||_F = "
           << resnorm2 / Bnorm << endl;
  }

  return 0;
}

void print_flop_breakdown
  (float random_flops, float ID_flops, float QR_flops, float ortho_flops,
   float reduce_sample_flops, float update_sample_flops,
   float extraction_flops, float CB_sample_flops, float sparse_sample_flops,
   float ULV_factor_flops, float schur_flops, float full_rank_flops) {
    
    // Just root process continues
    if (mpi_rank() != 0) return;

    float sample_flops = CB_sample_flops
      + sparse_sample_flops;
    float compression_flops = random_flops
      + ID_flops + QR_flops + ortho_flops
      + reduce_sample_flops + update_sample_flops
      + extraction_flops + sample_flops;
    std::cout << std::endl;
    std::cout << "# ----- FLOP BREAKDOWN ---------------------"
              << std::endl;
    std::cout << "# compression           = "
              << compression_flops << std::endl;
    std::cout << "#    random             = "
              << random_flops << std::endl;
    std::cout << "#    ID                 = "
              << ID_flops << std::endl;
    std::cout << "#    QR                 = "
              << QR_flops << std::endl;
    std::cout << "#    ortho              = "
              << ortho_flops << std::endl;
    std::cout << "#    reduce_samples     = "
              << reduce_sample_flops << std::endl;
    std::cout << "#    update_samples     = "
              << update_sample_flops << std::endl;
    std::cout << "#    extraction         = "
              << extraction_flops << std::endl;
    std::cout << "#    sampling           = "
              << sample_flops << std::endl;
    std::cout << "#       CB_sample       = "
              << CB_sample_flops << std::endl;
    std::cout << "#       sparse_sampling = "
              << sparse_sample_flops << std::endl;
    std::cout << "# ULV_factor            = "
              << ULV_factor_flops << std::endl;
    std::cout << "# Schur                 = "
              << schur_flops << std::endl;
    std::cout << "# full_rank             = "
              << full_rank_flops << std::endl;
    std::cout << "# --------------------------------------------"
              << std::endl;
    std::cout << "# total                 = "
              << (compression_flops + ULV_factor_flops +
                  schur_flops + full_rank_flops) << std::endl;
    std::cout << "# --------------------------------------------";
    std::cout << std::endl;
}


int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // Main program execution  
  int ierr = run(argc, argv);

  // Reducing flop counters
  float flops[12] = {
    float(params::random_flops.load()),
    float(params::ID_flops.load()),
    float(params::QR_flops.load()),
    float(params::ortho_flops.load()),
    float(params::reduce_sample_flops.load()),
    float(params::update_sample_flops.load()),
    float(params::extraction_flops.load()),
    float(params::CB_sample_flops.load()),
    float(params::sparse_sample_flops.load()),
    float(params::ULV_factor_flops.load()),
    float(params::schur_flops.load()),
    float(params::full_rank_flops.load())
  };

  float rflops[12];
  MPI_Reduce(flops, rflops, 12, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    
  print_flop_breakdown (rflops[0], rflops[1], rflops[2], rflops[3],
                        rflops[4], rflops[5], rflops[6], rflops[7], 
                        rflops[8], rflops[9], rflops[10], rflops[11]);

  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return ierr;
}