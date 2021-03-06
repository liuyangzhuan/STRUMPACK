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
 * works, and perform publicly and display publicly.  Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 */
#ifndef HSS_MATRIX_MPI_APPLY_HPP
#define HSS_MATRIX_MPI_APPLY_HPP

namespace strumpack {
  namespace HSS {

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HSSMatrixMPI<scalar_t>::apply(const DistM_t& b) const {
      assert(this->cols() == std::size_t(b.rows()));
      DistM_t c(b.grid(), this->rows(), b.cols());
#if !defined(NDEBUG)
      c.zero(); // silence valgrind
#endif
      apply_HSS(Trans::N, *this, b, scalar_t(0.), c);
      return c;
    }

    template<typename scalar_t> DistributedMatrix<scalar_t>
    HSSMatrixMPI<scalar_t>::applyC(const DistM_t& b) const {
      assert(this->rows() == std::size_t(b.rows()));
      DistM_t c(b.grid(), this->cols(), b.cols());
#if !defined(NDEBUG)
      c.zero(); // silence valgrind
#endif
      apply_HSS(Trans::C, *this, b, scalar_t(0.), c);
      return c;
    }

    template<typename scalar_t> void apply_HSS
    (Trans ta, const HSSMatrixMPI<scalar_t>& a,
     const DistributedMatrix<scalar_t>& b, scalar_t beta,
     DistributedMatrix<scalar_t>& c) {
      DistSubLeaf<scalar_t> B(b.cols(), &a, a.grid_local(), b),
        C(b.cols(), &a, a.grid_local());
      WorkApplyMPI<scalar_t> w;
      long long int flops = 0;
      if (ta == Trans::N) {
        a.apply_fwd(B, w, true, flops);
        a.apply_bwd(B, beta, C, w, true, flops);
      } else {
        a.applyT_fwd(B, w, true, flops);
        a.applyT_bwd(B, beta, C, w, true, flops);
      }
      C.from_block_row(c);
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::apply_fwd
    (const DistSubLeaf<scalar_t>& B, WorkApplyMPI<scalar_t>& w,
     bool isroot, long long int flops) const {
      if (!this->active()) return;
      if (this->leaf()) {
        if (!isroot) {
          w.tmp1 = _V.applyC(B.leaf);
          flops += _V.applyC_flops(B.leaf.cols());
        }
      } else {
        w.c.resize(2);
        w.c[0].offset = w.offset;
        w.c[1].offset = w.offset + this->_ch[0]->dims();
        this->_ch[0]->apply_fwd(B, w.c[0], Comm().size()==1, flops);
        this->_ch[1]->apply_fwd(B, w.c[1], Comm().size()==1, flops);
        if (!isroot) {
          w.tmp1 = _V.applyC
            (vconcat(B.cols(), this->_B10.cols(), this->_B01.cols(),
                     w.c[0].tmp1, w.c[1].tmp1, grid(), grid()->ctxt_all()));
          flops += _V.applyC_flops(B.cols());
        }
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::apply_bwd
    (const DistSubLeaf<scalar_t>& B, scalar_t beta, DistSubLeaf<scalar_t>& C,
     WorkApplyMPI<scalar_t>& w, bool isroot, long long int flops) const {
      if (!this->active()) return;
      if (this->leaf()) {
        if (this->U_rank() && !isroot) {  // c = D*b + beta*c + U*w.tmp2
          gemm(Trans::N, Trans::N, scalar_t(1.), _D, B.leaf, beta, C.leaf);
          C.leaf.add(_U.apply(w.tmp2));
          flops += C.leaf.rows() * C.leaf.cols() +
            gemm_flops(Trans::N, Trans::N, scalar_t(1.), _D, B.leaf, beta);
        } else {
          gemm(Trans::N, Trans::N, scalar_t(1.), _D, B.leaf, beta, C.leaf);
          flops += gemm_flops(Trans::N, Trans::N, scalar_t(1.), _D, B.leaf, beta);
        }
      } else {
        auto n = B.cols();
        DistM_t c0tmp1(grid(), _B10.cols(), n, w.c[0].tmp1, grid()->ctxt_all());
        DistM_t c1tmp1(grid(), _B01.cols(), n, w.c[1].tmp1, grid()->ctxt_all());
        DistM_t c0tmp2(grid(), _B01.rows(), n);
        DistM_t c1tmp2(grid(), _B10.rows(), n);
        if (isroot || !this->U_rank()) {
          gemm(Trans::N, Trans::N, scalar_t(1.), _B01, c1tmp1, scalar_t(0.), c0tmp2);
          gemm(Trans::N, Trans::N, scalar_t(1.), _B10, c0tmp1, scalar_t(0.), c1tmp2);
          flops +=
            gemm_flops(Trans::N, Trans::N, scalar_t(1.), _B01, c1tmp1, scalar_t(0.)) +
            gemm_flops(Trans::N, Trans::N, scalar_t(1.), _B10, c0tmp1, scalar_t(0.));
        } else {
          auto tmp = _U.apply(w.tmp2);
          copy(this->_B01.rows(), n, tmp, 0, 0, c0tmp2, 0, 0, grid()->ctxt_all());
          copy(this->_B10.rows(), n, tmp, this->_B01.rows(), 0, c1tmp2, 0, 0, grid()->ctxt_all());
          gemm(Trans::N, Trans::N, scalar_t(1.), _B01, c1tmp1, scalar_t(1.), c0tmp2);
          gemm(Trans::N, Trans::N, scalar_t(1.), _B10, c0tmp1, scalar_t(1.), c1tmp2);
          flops +=
            gemm_flops(Trans::N, Trans::N, scalar_t(1.), _B01, c1tmp1, scalar_t(1.)) +
            gemm_flops(Trans::N, Trans::N, scalar_t(1.), _B10, c0tmp1, scalar_t(1.));
        }
        w.c[0].tmp2 = DistM_t
          (w.c[0].tmp1.grid(), _B01.rows(), n, c0tmp2, grid()->ctxt_all());
        w.c[1].tmp2 = DistM_t
          (w.c[1].tmp1.grid(), _B10.rows(), n, c1tmp2, grid()->ctxt_all());
        this->_ch[0]->apply_bwd(B, beta, C, w.c[0], false, flops);
        this->_ch[1]->apply_bwd(B, beta, C, w.c[1], false, flops);
      }
    }


    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::applyT_fwd
    (const DistSubLeaf<scalar_t>& B, WorkApplyMPI<scalar_t>& w,
     bool isroot, long long int flops) const {
      if (!this->active()) return;
      if (this->leaf()) {
        if (!isroot) {
          w.tmp1 = _U.applyC(B.leaf);
          flops += _U.applyC_flops(B.leaf.cols());
        }
      } else {
        w.c.resize(2);
        w.c[0].offset = w.offset;
        w.c[1].offset = w.offset + this->_ch[0]->dims();
        this->_ch[0]->applyT_fwd(B, w.c[0], Comm().size()==1, flops);
        this->_ch[1]->applyT_fwd(B, w.c[1], Comm().size()==1, flops);
        if (!isroot) {
          w.tmp1 = _U.applyC
            (vconcat(B.cols(), this->_B01.rows(), this->_B10.rows(),
                     w.c[0].tmp1, w.c[1].tmp1, grid(), grid()->ctxt_all()));
          flops += _U.applyC_flops(B.cols());
        }
      }
    }

    template<typename scalar_t> void HSSMatrixMPI<scalar_t>::applyT_bwd
    (const DistSubLeaf<scalar_t>& B, scalar_t beta, DistSubLeaf<scalar_t>& C,
     WorkApplyMPI<scalar_t>& w, bool isroot, long long int flops) const {
      if (!this->active()) return;
      if (this->leaf()) {
        if (this->V_rank() && !isroot) {  // c = D*b + beta*c + U*w.tmp2
          gemm(Trans::C, Trans::N, scalar_t(1.), _D, B.leaf, beta, C.leaf);
          C.leaf.add(_V.apply(w.tmp2));
          flops += C.leaf.rows() * C.leaf.cols() +
            gemm_flops(Trans::C, Trans::N, scalar_t(1.), _D, B.leaf, beta);
        } else {
          gemm(Trans::C, Trans::N, scalar_t(1.), _D, B.leaf, beta, C.leaf);
          flops += gemm_flops(Trans::C, Trans::N, scalar_t(1.), _D, B.leaf, beta);
        }
      } else {
        auto n = B.cols();
        DistM_t c0tmp1(grid(), _B01.rows(), n, w.c[0].tmp1, grid()->ctxt_all());
        DistM_t c1tmp1(grid(), _B10.rows(), n, w.c[1].tmp1, grid()->ctxt_all());
        DistM_t c0tmp2(grid(), _B10.cols(), n);
        DistM_t c1tmp2(grid(), _B01.cols(), n);
        if (isroot || !this->V_rank()) {
          gemm(Trans::C, Trans::N, scalar_t(1.), _B10, c1tmp1,
               scalar_t(0.), c0tmp2);
          gemm(Trans::C, Trans::N, scalar_t(1.), _B01, c0tmp1,
               scalar_t(0.), c1tmp2);
          flops +=
            gemm_flops(Trans::C, Trans::N, scalar_t(1.), _B10, c1tmp1, scalar_t(0.)) +
            gemm_flops(Trans::C, Trans::N, scalar_t(1.), _B01, c0tmp1, scalar_t(0.));
        } else {
          auto tmp = _V.apply(w.tmp2);
          copy(c0tmp2.rows(), n, tmp, 0, 0, c0tmp2, 0, 0, grid()->ctxt_all());
          copy(c1tmp2.rows(), n, tmp, c0tmp2.rows(), 0, c1tmp2, 0, 0, grid()->ctxt_all());
          gemm(Trans::C, Trans::N, scalar_t(1.), _B10, c1tmp1,
               scalar_t(1.), c0tmp2);
          gemm(Trans::C, Trans::N, scalar_t(1.), _B01, c0tmp1,
               scalar_t(1.), c1tmp2);
          flops +=
            gemm_flops(Trans::C, Trans::N, scalar_t(1.), _B10, c1tmp1, scalar_t(1.)) +
            gemm_flops(Trans::C, Trans::N, scalar_t(1.), _B01, c0tmp1, scalar_t(1.));
        }
        w.c[0].tmp2 = DistM_t
          (w.c[0].tmp1.grid(), c0tmp2.rows(), n, c0tmp2, grid()->ctxt_all());
        w.c[1].tmp2 = DistM_t
          (w.c[1].tmp1.grid(), c1tmp2.rows(), n, c1tmp2, grid()->ctxt_all());
        this->_ch[0]->applyT_bwd(B, beta, C, w.c[0], false, flops);
        this->_ch[1]->applyT_bwd(B, beta, C, w.c[1], false, flops);
      }
    }

  } // end namespace HSS
} // end namespace strumpack

#endif // HSS_MATRIX_MPI_APPLY_HPP
