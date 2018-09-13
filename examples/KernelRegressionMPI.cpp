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
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
#define COUNT_FLOPS 1
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
// #include "dC_HODLR_wrapper.h"

using namespace std;
using namespace strumpack;
using namespace strumpack::HSS;

// FAST_H_SAMPLING= 0: slow sampling 1: H sampling 2: HODLR sampling
#define FAST_H_SAMPLING 2

#if defined(_OPENMP)
#include <omp.h>
#endif

extern "C" {
#define SSYEVX_FC FC_GLOBAL(ssyevx, SSYEVX)
#define DSYEVX_FC FC_GLOBAL(dsyevx, DSYEVX)
void SSYEVX_FC(char *JOBZ, char *RANGE, char *UPLO, int *N, float *A, int *LDA,
               float *VL, float *VU, int *IL, int *IU, float *ABSTOL, int *M,
               float *W, float *Z, int *LDZ, float *WORK, int *LWORK,
               int *IWORK, int *IFAIL, int *INFO);
void DSYEVX_FC(char *JOBZ, char *RANGE, char *UPLO, int *N, double *A, int *LDA,
               double *VL, double *VU, int *IL, int *IU, double *ABSTOL, int *M,
               double *W, double *Z, int *LDZ, double *WORK, int *LWORK,
               int *IWORK, int *IFAIL, int *INFO);
}

inline int syevx(char JOBZ, char RANGE, char UPLO, int N, float *A, int LDA,
                 float VL, float VU, int IL, int IU, float ABSTOL, int &M,
                 float *W, float *Z, int LDZ) {
  int INFO;
  auto IWORK = new int[5 * N + N];
  auto IFAIL = IWORK + 5 * N;
  int LWORK = -1;
  float SWORK;
  SSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU, &ABSTOL, &M,
            W, Z, &LDZ, &SWORK, &LWORK, IWORK, IFAIL, &INFO);
  LWORK = int(SWORK);
  auto WORK = new float[LWORK];
  SSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU, &ABSTOL, &M,
            W, Z, &LDZ, WORK, &LWORK, IWORK, IFAIL, &INFO);
  delete[] WORK;
  delete[] IWORK;
  return INFO;
}
inline int dyevx(char JOBZ, char RANGE, char UPLO, int N, double *A, int LDA,
                 double VL, double VU, int IL, int IU, double ABSTOL, int &M,
                 double *W, double *Z, int LDZ) {
  int INFO;
  auto IWORK = new int[5 * N + N];
  auto IFAIL = IWORK + 5 * N;
  int LWORK = -1;
  double DWORK;
  DSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU, &ABSTOL, &M,
            W, Z, &LDZ, &DWORK, &LWORK, IWORK, IFAIL, &INFO);
  LWORK = int(DWORK);
  auto WORK = new double[LWORK];
  DSYEVX_FC(&JOBZ, &RANGE, &UPLO, &N, A, &LDA, &VL, &VU, &IL, &IU, &ABSTOL, &M,
            W, Z, &LDZ, WORK, &LWORK, IWORK, IFAIL, &INFO);
  delete[] WORK;
  delete[] IWORK;
  return INFO;
}

#define ERROR_TOLERANCE 1e2

const int kmeans_max_it = 100;
random_device rd;
double r;
//mt19937 generator(rd());
mt19937 generator; // make sure to use the same ordering on all processes

inline double dist2(double *x, double *y, int d) {
  double k = 0.;
  for (int i = 0; i < d; i++)
    k += pow(x[i] - y[i], 2.);
  return k;
}

inline double dist(double *x, double *y, int d) { return sqrt(dist2(x, y, d)); }

inline double norm1(double *x, double *y, int d) {
  double k = 0.;
  for (int i = 0; i < d; i++)
    k += fabs(x[i] - y[i]);
  return k;
}

// dot product of two real vectors
inline double dot_product(double* v, double* u, int d)
{
    double result = 0.0;
    for (int i = 0; i < d; i++)
        result += v[i]*u[i];
    return result;
}

inline double Gauss_kernel(double *x, double *y, int d, double h) {
  return exp(-dist2(x, y, d) / (2. * h * h));
}

inline double Laplace_kernel(double *x, double *y, int d, double h) {
  return exp(-norm1(x, y, d) / h);
}


//R^4 kernel
inline double K07_kernel(double *x, double *y, int d) {
  double dists;
  dists = dist2(x, y, d);
  return pow(dists,4);
}

// sqrt(R^2+h) kernel
inline double K08_kernel(double *x, double *y, int d, double h) {
  double dists;
  dists = dist2(x, y, d);
  return sqrt(pow(dists,2)+h);
}

// 1/sqrt(R^2+h) kernel
inline double K09_kernel(double *x, double *y, int d, double h) {
  double dists;
  dists = dist2(x, y, d);
  return 1.0/sqrt(pow(dists,2)+h);
}

// Polynomial kernel (X^tY+h)^2
inline double K10_kernel(double *x, double *y, int d, double h) {
  double dotp;
  dotp = dot_product(x, y, d);
  return pow(dotp+h,2);
}


// get maximum tree level
inline void get_treelevel(HSSPartitionTree &tree, int &level, int level_ind) {
	level = std::max(level,level_ind);
	if (!tree.c.empty()) {
		get_treelevel(tree.c[0], level, level_ind+1);
		get_treelevel(tree.c[1], level, level_ind+1);
	}
}

// get leaf node sizes of a tree
inline void get_leafsizes(HSSPartitionTree &tree, int &leaf_ind, int* leaf_sizes) {
	if (!tree.c.empty()) {
		get_leafsizes(tree.c[0], leaf_ind, leaf_sizes);
		get_leafsizes(tree.c[1], leaf_ind, leaf_sizes);
	}else{
		leaf_sizes[leaf_ind++] = tree.size;
	}
} 


//convert from blacs local indices to global indices
inline int l2g(int il, int p, int n,int np,int nb){
	int ilm1 = il-1;
	return ((floor(ilm1/nb) * np) + p)*nb + ilm1%nb + 1;
} 


inline int *kmeans_start_random(int n, int k) {
  uniform_int_distribution<int> uniform_random(0, n - 1);
  int *ind_centers = new int[k];
  for (int i = 0; i < k; i++) {
    ind_centers[i] = uniform_random(generator);
  }
  return ind_centers;
}

// 3 more start sampling methods for the case k == 2

int *kmeans_start_random_dist_maximized(int n, double *p, int d) {
  constexpr size_t k = 2;

  uniform_int_distribution<int> uniform_random(0, n - 1);
  const auto t = uniform_random(generator);
  // compute probabilities
  double *cur_dist = new double[n];
  for (int i = 0; i < n; i++) {
    cur_dist[i] = dist2(&p[i * d], &p[t * d], d);
  }

  discrete_distribution<int> random_center(&cur_dist[0], &cur_dist[n]);

  delete[] cur_dist;

  int *ind_centers = new int[k];
  ind_centers[0] = t;
  ind_centers[1] = random_center(generator);
  return ind_centers;
}

// for k = 2 only
int *kmeans_start_dist_maximized(int n, double *p, int d) {
  constexpr size_t k = 2;

  // find centroid
  double centroid[d];

  for (int i = 0; i < d; i++) {
    centroid[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      centroid[j] += p[i * d + j];
    }
  }

  for (int j = 0; j < d; j++)
    centroid[j] /= n;

  // find farthest point from centroid
  int first_index = 0;
  double max_dist = -1;

  for (int i = 0; i < n; i++) {
    double dd = dist(&p[i * d], centroid, d);
    if (dd > max_dist) {
      max_dist = dd;
      first_index = i;
    }
  }
  // find fathest point from the firsth point
  int second_index = 0;
  max_dist = -1;
  for (int i = 0; i < n; i++) {
    double dd = dist(&p[i * d], &p[first_index * d], d);
    if (dd > max_dist) {
      max_dist = dd;
      second_index = i;
    }
  }
  int *ind_centers = new int[k];
  ind_centers[0] = first_index;
  ind_centers[1] = second_index;
  return ind_centers;
}

inline int *kmeans_start_fixed(int n, double *p, int d) {
  int *ind_centers = new int[2];
  ind_centers[0] = 0;
  ind_centers[1] = n - 1;
  return ind_centers;
}

void k_means(int k, double *p, int n, int d, int *nc, double *labels) {
  double **center = new double *[k];

  int *ind_centers = NULL;

  constexpr int kmeans_options = 2;
  switch (kmeans_options) {
  case 1:
    ind_centers = kmeans_start_random(n, k);
    break;
  case 2:
    ind_centers = kmeans_start_random_dist_maximized(n, p, d);
    break;
  case 3:
    ind_centers = kmeans_start_dist_maximized(n, p, d);
    break;
  case 4:
    ind_centers = kmeans_start_fixed(n, p, d);
    break;
  }

  for (int c = 0; c < k; c++) {
    center[c] = new double[d];
    for (int j = 0; j < d; j++)
      center[c][j] = p[ind_centers[c] * d + j];
  }

  int iter = 0;
  bool changes = true;
  int *cluster = new int[n];
  for (int i = 0; i < n; i++) {
    cluster[i] = 0;
  }

  while ((changes == true) and (iter < kmeans_max_it)) {
    // for each point, find the closest cluster center
    changes = false;
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
      double min_dist = dist(&p[i * d], center[0], d);
      cluster[i] = 0;
      for (int c = 1; c < k; c++) {
        double dd = dist(&p[i * d], center[c], d);
        if (dd <= min_dist) {
          min_dist = dd;
          if (c != cluster[i]) {
#pragma omp atomic write
            changes = true;
          }
          cluster[i] = c;
        }
      }
    }

    for (int c = 0; c < k; c++) {
      nc[c] = 0;
      for (int j = 0; j < d; j++)
        center[c][j] = 0.;
    }

#pragma omp parallel
    {
      double **my_center = new double*[k];
      for (int i = 0; i < k; i++) {
        my_center[i] = new double[d];
        for (int j = 0; j < d; j++)
          my_center[i][j] = 0.;
      }
#pragma omp for
      for (int i = 0; i < n; i++) {
        auto c = cluster[i];
#pragma omp atomic
        nc[c]++;
        for (int j = 0; j < d; j++)
          my_center[c][j] += p[i * d + j];
      }

#pragma omp critical
      {
        for (int i = 0; i < k; i++) {
          for (int j = 0; j < d; j++)
            center[i][j] += my_center[i][j];
        }
      }
      for (int i = 0; i < k; i++)
        delete[] my_center[i];
      delete[] my_center;
    }

    for (int c = 0; c < k; c++)
      for (int j = 0; j < d; j++)
        center[c][j] /= nc[c];
    iter++;
  }

  int *ci = new int[k];
  for (int c = 0; c < k; c++)
    ci[c] = 0;
  double *p_perm = new double[n * d];
  double *labels_perm = new double[n];
  int row = 0;
  for (int c = 0; c < k; c++) {
    for (int j = 0; j < nc[c]; j++) {
      while (cluster[ci[c]] != c)
        ci[c]++;
      for (int l = 0; l < d; l++)
        p_perm[l + row * d] = p[l + ci[c] * d];
      labels_perm[row] = labels[ci[c]];
      ci[c]++;
      row++;
    }
  }
  copy(p_perm, p_perm + n * d, p);
  copy(labels_perm, labels_perm + n, labels);
  delete[] p_perm;
  delete[] labels_perm;
  delete[] ci;

  for (int i = 0; i < k; i++)
    delete[] center[i];
  delete[] center;
  delete[] cluster;
  delete[] ind_centers;
}

void recursive_2_means(double *p, int n, int d, int cluster_size,
                       HSSPartitionTree &tree, double *labels) {
  if (n < cluster_size)
    return;
  auto nc = new int[2];
  k_means(2, p, n, d, nc, labels);
  if (nc[0] == 0 || nc[1] == 0)
    return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_2_means(p, nc[0], d, cluster_size, tree.c[0], labels);
  recursive_2_means(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1],
                    labels + nc[0]);
  delete[] nc;
}

void kd_partition(double *p, int n, int d, int *nc, double *labels) {
  // find coordinate of the most spread
  double *maxes = new double[d];
  double *mins = new double[d];

  for (int j = 0; j < d; ++j) {
    maxes[j] = p[j];
    mins[j] = p[j];
  }

  for (int i = 1; i < n; ++i) {
    for (int j = 0; j < d; ++j) {
      if (p[i * d + j] > maxes[j]) {
        maxes[j] = p[i * d + j];
      }
      if (p[i * d + j] > mins[j]) {
        mins[j] = p[i * d + j];
      }
    }
  }
  double max_var = maxes[0] - mins[0];
  int dim = 0;
  for (int j = 0; j < d; ++j) {
    if (maxes[j] - mins[j] > max_var) {
      max_var = maxes[j] - mins[j];
      dim = j;
    }
  }

   
  
  std::vector<std::pair<double, int>> dat_MD;
  dat_MD.resize(n);
  for (int i=0;i<n;++i){
	dat_MD.data()[i].first = p[i * d + dim];
	dat_MD.data()[i].second = i;
  } 
  std::sort(dat_MD.begin(), dat_MD.end());  
  double *p_perm = new double[n * d];
  double *labels_perm = new double[n];
  for (int row=0; row<n; ++row){
  for (int l = 0; l < d; l++)
    p_perm[l + row * d] = p[l + dat_MD.data()[row].second * d]; 
	labels_perm[row] = labels[dat_MD.data()[row].second]; 
  }  
  dat_MD.clear();
  
#if 1  // split by median
  nc[0] = floor((double)n/2.0);
  nc[1] = n - nc[0];  
#else // split by mean  
  // find the mean
  double mean_value = 0.;
  for (int i = 0; i < n; ++i) {
    mean_value += p_perm[i * d + dim];
  }
  mean_value /= n;  

  nc[0] = 0;
  nc[1] = 0;
  for (int i = 0; i < n; ++i) {
    if (p_perm[d * i + dim] > mean_value) {
      nc[1] += 1;
    } else {
      nc[0] += 1;
    }
  }

#endif  
  
  
  copy(p_perm, p_perm + n * d, p);
  copy(labels_perm, labels_perm + n, labels);
  delete[] p_perm;
  delete[] labels_perm;
  delete[] maxes;
  delete[] mins;
  
  // // split the data
  // int *cluster = new int[n];
  // for (int i = 0; i < n; i++) {
    // cluster[i] = 0;
  // }
  // nc[0] = 0;
  // nc[1] = 0;
  // for (int i = 0; i < n; ++i) {
    // if (p[d * i + dim] > mean_value) {
      // cluster[i] = 1;
      // nc[1] += 1;
    // } else {
      // nc[0] += 1;
    // }
  // }

  // // permute the data

  // int *ci = new int[2];
  // for (int c = 0; c < 2; c++)
    // ci[c] = 0;
  // double *p_perm = new double[n * d];
  // double *labels_perm = new double[n];
  // int row = 0;
  // for (int c = 0; c < 2; c++) {
    // for (int j = 0; j < nc[c]; j++) {
      // while (cluster[ci[c]] != c)
        // ci[c]++;
      // for (int l = 0; l < d; l++)
        // p_perm[l + row * d] = p[l + ci[c] * d];
      // labels_perm[row] = labels[ci[c]];
      // ci[c]++;
      // row++;
    // }
  // }

  // copy(p_perm, p_perm + n * d, p);
  // copy(labels_perm, labels_perm + n, labels);
  // delete[] p_perm;
  // delete[] labels_perm;
  // delete[] ci;
  // delete[] maxes;
  // delete[] mins;
  // delete[] cluster;
}


void cobble_partition(double *p, int n, int d, int *nc, double *labels) {

  // find centroid
  double centroid[d];

  for (int i = 0; i < d; i++) {
    centroid[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      centroid[j] += p[i * d + j];
    }
  }

  for (int j = 0; j < d; j++)
    centroid[j] /= n;

  // find farthest point from centroid
  int first_index = 0;
  double max_dist = -1;

  for (int i = 0; i < n; i++) {
    double dd = dist(&p[i * d], centroid, d);
    if (dd > max_dist) {
      max_dist = dd;
      first_index = i;
    }
  }

  // compute and sort distance from the firsth point
  std::vector<double> dists;
  dists.resize(n);
  for (int i = 0; i < n; i++) {
    dists[i] = dist(&p[i * d], &p[first_index * d], d);
  }
  
  std::vector<std::pair<double, int>> dat_MD;
  dat_MD.resize(n);
  for (int i=0;i<n;++i){
	dat_MD.data()[i].first = dists[i];
	dat_MD.data()[i].second = i;
  } 
  std::sort(dat_MD.begin(), dat_MD.end());  
  
  double *p_perm = new double[n * d];
  double *labels_perm = new double[n];
  for (int row=0; row<n; ++row){
  // cout<<dat_MD.data()[row].first<<""<<dat_MD.data()[row].second<<endl;
  for (int l = 0; l < d; l++)
    p_perm[l + row * d] = p[l + dat_MD.data()[row].second * d]; 
	labels_perm[row] = labels[dat_MD.data()[row].second]; 
  }  
  dat_MD.clear();
  dists.clear();
  
  nc[0] = floor((double)n/2.0);
  nc[1] = n - nc[0];    
  
  copy(p_perm, p_perm + n * d, p);
  copy(labels_perm, labels_perm + n, labels);
  delete[] p_perm;
  delete[] labels_perm;

}




void recursive_cobble(double *p, int n, int d, int cluster_size,
                  HSSPartitionTree &tree, double *labels) {
  if (n < cluster_size)
    return;
  auto nc = new int[2];
  cobble_partition(p, n, d, nc, labels);
  if (nc[0] == 0 || nc[1] == 0)
    return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_cobble(p, nc[0], d, cluster_size, tree.c[0], labels);
  recursive_cobble(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1],
               labels + nc[0]);
  delete[] nc;
}



void recursive_kd(double *p, int n, int d, int cluster_size,
                  HSSPartitionTree &tree, double *labels) {
  if (n < cluster_size)
    return;
  auto nc = new int[2];
  kd_partition(p, n, d, nc, labels);
  if (nc[0] == 0 || nc[1] == 0)
    return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_kd(p, nc[0], d, cluster_size, tree.c[0], labels);
  recursive_kd(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1],
               labels + nc[0]);
  delete[] nc;
}

void pca_partition(double *p, int n, int d, int *nc, double *labels) {
  // find first pca direction
  int num = 0;
  double *W = new double[d];
  double *Z = new double[d * d];
  DenseMatrixWrapper<double> X(n, d, p, n);
  DenseMatrix<double> XtX(d, d);
  gemm(Trans::T, Trans::N, 1., X, X, 0., XtX);
  double *XtX_data = new double[d * d];
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      XtX_data[d * i + j] = XtX(i, j);
    }
  }
  dyevx('V', 'I', 'U', d, XtX_data, d, 1., 1., d, d, 1e-2, num, W, Z, d);
  // compute pca coordinates
  double *new_x_coord = new double[n];
  for (int i = 0; i < n; i++) {
    new_x_coord[i] = 0.;
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      new_x_coord[i] += p[i * d + j] * Z[j];
    }
  }

  // find the mean
  double mean_value = 0.;
  for (int i = 0; i < n; ++i) {
    mean_value += new_x_coord[i];
  }
  mean_value /= n;

  // split the data
  int *cluster = new int[n];
  for (int i = 0; i < n; i++) {
    cluster[i] = 0;
  }
  nc[0] = 0;
  nc[1] = 0;
  for (int i = 0; i < n; ++i) {
    if (new_x_coord[i] > mean_value) {
      cluster[i] = 1;
      nc[1] += 1;
    } else {
      nc[0] += 1;
    }
  }

  // permute the data

  int *ci = new int[2];
  for (int c = 0; c < 2; c++)
    ci[c] = 0;
  double *p_perm = new double[n * d];
  double *labels_perm = new double[n];
  int row = 0;
  for (int c = 0; c < 2; c++) {
    for (int j = 0; j < nc[c]; j++) {
      while (cluster[ci[c]] != c)
        ci[c]++;
      for (int l = 0; l < d; l++)
        p_perm[l + row * d] = p[l + ci[c] * d];
      labels_perm[row] = labels[ci[c]];
      ci[c]++;
      row++;
    }
  }

  copy(p_perm, p_perm + n * d, p);
  copy(labels_perm, labels_perm + n, labels);
  delete[] p_perm;
  delete[] labels_perm;
  delete[] ci;
  delete[] cluster;
  delete[] new_x_coord;
  delete[] W;
  delete[] Z;
}

void recursive_pca(double *p, int n, int d, int cluster_size,
                   HSSPartitionTree &tree, double *labels) {
  if (n < cluster_size)
    return;
  auto nc = new int[2];
  pca_partition(p, n, d, nc, labels);
  if (nc[0] == 0 || nc[1] == 0)
    return;
  tree.c.resize(2);
  tree.c[0].size = nc[0];
  tree.c[1].size = nc[1];
  recursive_pca(p, nc[0], d, cluster_size, tree.c[0], labels);
  recursive_pca(p + nc[0] * d, nc[1], d, cluster_size, tree.c[1],
                labels + nc[0]);
  delete[] nc;
}

typedef void* F2Cptr;  // pointer passing fortran derived types to c
typedef void* C2Fptr;  // pointer passing c objects to fortran

extern "C" {
  void FC_GLOBAL_(h_matrix_fill,H_MATRIX_FILL)
    (int* Npo, int* Ndim, double* Locations,
     int* Nmin, double* tol, double* h, double* lam,
     int* nth, int* nmpi, int* ninc,
     int* aca, int* perm, int* myseg);
  void FC_GLOBAL_(h_matrix_apply,H_MATRIX_APPLY)
    (int* Npo, int* Ncol, double* Xin, double* Xout);
	
	
    void c_hodlr_construct(int* Npo, int* Ndim, double* Locations, int* nlevel, int* tree, int* perms, int* Npo_loc, F2Cptr* ho_bf_for, F2Cptr* option,F2Cptr* stats,F2Cptr* msh,F2Cptr* ker,F2Cptr* ptree, void (*C_FuncZmn)(int*, int*, double*,C2Fptr), C2Fptr C_QuantZmn, MPI_Fint* MPIcomm);	
 
	void c_hodlr_factor(F2Cptr* ho_bf_for,F2Cptr* option,F2Cptr* stats,F2Cptr* ptree);	

	void c_hodlr_solve(double* x, double* b, int* Nloc, int* Nrhs, F2Cptr* ho_bf_for,F2Cptr* option,F2Cptr* stats,F2Cptr* ptree);	
	
	void c_hodlr_mult(double* xin, double* xout, int* Nloc, int* Ncol, F2Cptr* ho_bf_for,F2Cptr* option,F2Cptr* stats,F2Cptr* ptree);	
	
	void c_hodlr_createptree(int* nmpi, int* groupmembers, MPI_Fint* MPIcomm, F2Cptr* ptree);
	
	void c_hodlr_createstats(F2Cptr* stats);		
	void c_hodlr_printstats(F2Cptr* stats,F2Cptr* ptree);		
	void c_hodlr_createoption(F2Cptr* option);	
	void c_hodlr_setoption(F2Cptr* option, char const * nam, C2Fptr val);	
	
	inline void c_hodlr_set_I_option(F2Cptr* option, char const * nam, int val){
		c_hodlr_setoption(option, nam, (C2Fptr) &val);
	}
	inline void c_hodlr_set_D_option(F2Cptr* option, char const * nam, double val){
		c_hodlr_setoption(option, nam, (C2Fptr) &val);
	}	
}




// The object handling kernel parameters and sampling function
class C_QuantZmn {
public:
  vector<double> _data;
  int _d = 0;
  int _n = 0;
  double _h = 0.;
  double _l = 0.;
  int _ker=1; // 
  
  int _rank_rand;
  int _n_rand;
  std::vector<double> _MatU;
  std::vector<double> _MatV;
  
  std::vector<int> _Hperm;
  std::vector<int> _iHperm;
  int _nloc = 0;

  C_QuantZmn() = default;
  
  C_QuantZmn(vector<double> data, int d, double h, double l, int ker)
    : _data(move(data)), _d(d), _n(_data.size() / _d),
      _h(h), _l(l),_ker(ker){
    assert(size_t(_n * _d) == _data.size());
	}
  
  C_QuantZmn(int n_rand, int rank_rand, int ker, vector<double> MatU, vector<double> MatV)
    : _n_rand(n_rand), _rank_rand(rank_rand), _ker(ker), _MatU(move(MatU)), _MatV(move(MatV)){
	// cout<<_n_rand<<_rank_rand<<_MatU.size()<<endl;
    assert(size_t(_n_rand * _rank_rand) == _MatU.size());
	}  
  
  
  inline void Sample(int m, int n, double* val){
	switch(_ker){
	case 1: //Gaussian kernel
		*val = Gauss_kernel(&_data[m * _d], &_data[n * _d], _d, _h);
		if (m==n)
		*val += _l;
		break;
	case 2: //R^4 kernel
		*val = K07_kernel(&_data[m * _d], &_data[n * _d], _d);	
		break;
	case 3: //sqrt(R^2+h) kernel
		*val = K08_kernel(&_data[m * _d], &_data[n * _d], _d, _h);	
		break;
	case 4: //1/sqrt(R^2+h) kernel
		*val = K09_kernel(&_data[m * _d], &_data[n * _d], _d, _h);
		break;
	case 5: //Polynomial kernel (X^tY+h)^2
		*val = K10_kernel(&_data[m * _d], &_data[n * _d], _d, _h);
		break;
	case 6: //Low-rank product of two random matrices
		*val =0;
		for (int k = 0; k < _rank_rand; k++){
			*val += _MatU[k*_n_rand+m]*_MatV[k*_n_rand+n];
		}
		break;
	}
  } 	
};


// The sampling function wrapper required by the Fortran HODLR code
inline void C_FuncZmn(int *m, int *n, double *val, C2Fptr quant) {
	
  C_QuantZmn* Q = (C_QuantZmn*) quant;	
  Q->Sample(*m,*n,val);
}




class KernelMPI {
  using DenseM_t = DenseMatrix<double>;
  using DenseMW_t = DenseMatrixWrapper<double>;
  using DistM_t = DistributedMatrix<double>;
  using DistMW_t = DistributedMatrixWrapper<double>;

public:
  vector<double> _data;
  int _d = 0;
  int _n = 0;
  double _h = 0.;
  double _l = 0.;
  int _ctxt_all = -1;
  int _prows = -1;
  int _pcols = -1;
  std::vector<int> _Hperm;
  std::vector<int> _iHperm;
  int _Hrows = 0;
  int _cluster_size = 500; // finest leafsize
  int _nlevel = 0; // 0: tree level, nonzero if a tree is provided 
  int* _tree = new int[(int)pow(2,_nlevel)]; //user provided array containing size of each leaf node, not used if _nlevel=0  
  vector<int> _dist;
  C_QuantZmn* quant_ptr;
  F2Cptr ho_bf;  //HODLR returned by Fortran code 
  F2Cptr option;     //option structure returned by Fortran code 
  F2Cptr stats;      //statistics structure returned by Fortran code
  F2Cptr msh;		   //d_mesh structure returned by Fortran code
  F2Cptr kerquant;   //kernel quantities structure returned by Fortran code
  F2Cptr ptree;      //process tree returned by Fortran code
  MPI_Fint Fcomm;  // the fortran MPI communicator
  
  
  KernelMPI() = default;
  KernelMPI(vector<double> data, int d, double h, double l,
            HSSOptions<double>& opts, int ctxt_all, 
	    int nprow, int npcol,  int nmpi, int ninc, int aca, int cluster_size, int nlevel, int *tree)
    : _data(move(data)), _d(d), _n(_data.size() / _d),
      _h(h), _l(l), _ctxt_all(ctxt_all), _prows(nprow), _pcols(npcol),_cluster_size(cluster_size),_nlevel(nlevel) {
    assert(size_t(_n * _d) == _data.size());

// #if FAST_H_SAMPLING == 1
    // const auto P = mpi_nprocs();
    // const auto rank = mpi_rank();
    // if (!rank)
      // cout << "# Called <FAST_H_SAMPLING> Started matrix construction..." << endl;
    // auto starttime = MPI_Wtime();
    // int Nmin = 500;    // finest leafsize
    // double tol = 1e-2; // compression tolerance
    // //double tol = opts.rel_tol(); // compression tolerance
    // int nth = omp_get_max_threads();
    // _Hperm.resize(_n);
    // FC_GLOBAL_(h_matrix_fill,H_MATRIX_FILL)
      // (&_n, &_d, _data.data(), &Nmin, &tol, &h, &l,
       // &nth, &nmpi, &ninc, &aca, _Hperm.data(), &_Hrows);
    // for (auto& i : _Hperm) i--; // Fortran to C
    // MPI_Bcast(_Hperm.data(), _n, MPI_INT, 0, MPI_COMM_WORLD);
    // _iHperm.resize(_n);
    // for (int i=0; i<_n; i++) 
      // _iHperm[_Hperm[i]] = i;
    // _dist.resize(P+1);
    // _dist[rank+1] = _Hrows;
    // MPI_Allgather
      // (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
       // _dist.data()+1, 1, MPI_INT, MPI_COMM_WORLD);
    // for (int p=0; p<P; p++)
      // _dist[p+1] += _dist[p];
    // auto endtime = MPI_Wtime();
    // if (!rank)
      // cout << "# H Matrix construction time " << endtime - starttime
           // << " seconds" << endl;
		   
// #elif FAST_H_SAMPLING == 2	
	   
    int P = mpi_nprocs();
    const auto rank = mpi_rank();
    if (!rank)
      cout << "# Called <FAST_H_SAMPLING> Started matrix construction..." << endl;
    auto starttime = MPI_Wtime();		   
	
	_tree = new int[(int)pow(2,_nlevel)];
	copy(tree, tree + (int)pow(2,_nlevel), _tree);
	
	int* groups;
	int ker = 1;
	int preorder=1;
	// int Nmin = 500;    // finest leafsize
	double tol = 1e-2; // compression tolerance
	int com_opt=2;    //compression option 1:SVD 2:RRQR 3:ACA 4:BACA  
	int sort_opt=3; //1:KD tree 3:balanced 2-means
	int checkerr = 0; //1: check compression quality 
	int batch = 100; //batch size for BACA	
	int myseg=0;     // local number of unknowns
	
	
	quant_ptr=new C_QuantZmn(_data, _d, _h, _l,ker);	

    _Hperm.resize(_n);   
	groups = new int[P];
	Fcomm = MPI_Comm_c2f(MPI_COMM_WORLD); 
	for (int i = 0; i < P; i++)groups[i]=i;

	// create hodlr data structures
	c_hodlr_createptree(&P, groups, &Fcomm, &ptree);
	c_hodlr_createoption(&option);	
	c_hodlr_createstats(&stats);		
	
	// set hodlr options
	c_hodlr_set_D_option(&option, "tol_comp", tol);
	c_hodlr_set_I_option(&option, "preorder", preorder);
	c_hodlr_set_I_option(&option, "Nmin_leaf", _cluster_size); 
	c_hodlr_set_I_option(&option, "RecLR_leaf", com_opt); 
	c_hodlr_set_I_option(&option, "xyzsort", sort_opt); 
	c_hodlr_set_I_option(&option, "ErrFillFull", checkerr); 
	c_hodlr_set_I_option(&option, "BACA_Batch", batch); 
	

    // construct hodlr with geometrical points	
	c_hodlr_construct(&_n, &_d, _data.data(), &_nlevel, _tree, _Hperm.data(), &_Hrows, &ho_bf, &option, &stats, &msh, &kerquant, &ptree, &C_FuncZmn, quant_ptr, &Fcomm);		
	
	
    for (auto& i : _Hperm) i--; // Fortran to C
    MPI_Bcast(_Hperm.data(), _n, MPI_INT, 0, MPI_COMM_WORLD);
    _iHperm.resize(_n);
    for (int i=0; i<_n; i++) 
      _iHperm[_Hperm[i]] = i;
    _dist.resize(P+1);
    _dist[rank+1] = _Hrows;
    MPI_Allgather
      (MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
       _dist.data()+1, 1, MPI_INT, MPI_COMM_WORLD);
    for (int p=0; p<P; p++)
      _dist[p+1] += _dist[p];
    auto endtime = MPI_Wtime();
    if (!rank)
      cout << "# HODLR construction time " << endtime - starttime
           << " seconds" << endl;
      
// #endif

  }

  void operator()(const vector<size_t> &I, const vector<size_t> &J,
                  DistM_t &B) {
    if (!B.active()) return;
    assert(I.size() == size_t(B.rows()) &&
           J.size() == size_t(B.cols()));
    for (size_t j = 0; j < J.size(); j++) {
      if (B.colg2p(j) != B.pcol()) continue;
      for (size_t i = 0; i < I.size(); i++) {
        if (B.rowg2p(i) == B.prow()) {
          assert(B.is_local(i, j));
          B.global(i, j) = Gauss_kernel(&_data[I[i] * _d], &_data[J[j] * _d], _d, _h);
          if (I[i] == J[j])
            B.global(i, j) += _l;
        }
      }
    }
  }

  DenseM_t redistribute_2D_to_1D(DistM_t& R) {
    const auto P = mpi_nprocs();
    const auto rank = mpi_rank();
    const auto cols = R.cols();
    const auto lcols = R.lcols();
    const auto lrows = R.lrows();
    const auto B = DistM_t::default_MB;
    vector<vector<double>> sbuf(P);
    if (R.active()) {
      // global, local, proc
      vector<tuple<int,int,int>> glp(lrows);
      {
	vector<size_t> count(P);
	for (int r=0; r<lrows; r++) {
	  auto gr = _Hperm[R.rowl2g(r)];
	  auto p = -1 + distance
	    (_dist.begin(), upper_bound(_dist.begin(), _dist.end(), gr));
	  glp[r] = tuple<int,int,int>{gr, r, p};
	  count[p] += lcols;
	}
	sort(glp.begin(), glp.end());
	for (int p=0; p<P; p++)
	  sbuf[p].reserve(count[p]);
      }
      for (int r=0; r<lrows; r++)
	for (int c=0, lr=get<1>(glp[r]), p=get<2>(glp[r]); c<lcols; c++)
	  sbuf[p].push_back(R(lr,c));
    }
    double *rbuf = nullptr, **pbuf = nullptr;
    all_to_all_v(sbuf, rbuf, pbuf, MPI_COMM_WORLD);
    DenseM_t R1D(_Hrows, cols);
    if (_Hrows) {
      vector<int> src_c(cols);
      for (int c=0; c<cols; c++)
	src_c[c] = ((c / B) % _pcols) * _prows;
      for (int r=0; r<_Hrows; r++) {
	auto gr = _iHperm[r + _dist[rank]];
	auto src_r = (gr / B) % _prows;
	for (int c=0; c<cols; c++)
	  R1D(r, c) = *(pbuf[src_r + src_c[c]]++);
      }
    }
    delete[] pbuf;
    delete[] rbuf;
    return R1D;
  }

  void redistribute_1D_to_2D(DenseM_t& S1D, DistM_t& S) {
    const auto rank = mpi_rank();
    const auto P = mpi_nprocs();
    const auto B = DistM_t::default_MB;
    const auto cols = S1D.cols();
    const auto lcols = S.lcols();
    const auto lrows = S.lrows();
    vector<vector<double>> sbuf(P);
    if (_Hrows) {
      vector<tuple<int,int,int>> glp(_Hrows);
      for (int r=0; r<_Hrows; r++) {
	auto gr = _iHperm[r + _dist[rank]];
	glp[r] = tuple<int,int,int>{gr,r,(gr / B) % _prows};
      }
      sort(glp.begin(), glp.end());
      vector<int> pc(cols);
      for (int c=0; c<cols; c++)
	pc[c] = ((c / B) % _pcols) * _prows;
      {
	vector<size_t> count(P);
	for (int r=0; r<_Hrows; r++)
	  for (int c=0, pr=get<2>(glp[r]); c<cols; c++)
	    count[pr+pc[c]]++;
	for (int p=0; p<P; p++) 
	  sbuf[p].reserve(count[p]);
      }
      for (int r=0; r<_Hrows; r++)
	for (int c=0, lr=get<1>(glp[r]), pr=get<2>(glp[r]); c<cols; c++)
	  sbuf[pr+pc[c]].push_back(S1D(lr,c));
    }
    double *rbuf = nullptr, **pbuf = nullptr;
    all_to_all_v(sbuf, rbuf, pbuf, MPI_COMM_WORLD);
    if (S.active()) {
      for (int r=0; r<lrows; r++) {
	auto gr = _Hperm[S.rowl2g(r)];
	auto p = -1 + distance
	  (_dist.begin(), upper_bound(_dist.begin(), _dist.end(), gr));
	for (int c=0; c<lcols; c++)
	  S(r,c) = *(pbuf[p]++);
      }
    }
    delete[] pbuf;
    delete[] rbuf;
  }


// #if FAST_H_SAMPLING == 1

  // void operator()(DistM_t &R, DistM_t &Sr, DistM_t &Sc) {
    // auto starttime = MPI_Wtime();
    // int Ncol = R.cols();
    // {
      // DenseM_t S1D(_Hrows, Ncol);
      // {
	// DenseM_t R1D = redistribute_2D_to_1D(R);
	// FC_GLOBAL_(h_matrix_apply,H_MATRIX_APPLY)
	  // (&_n, &Ncol, R1D.data(), S1D.data());
      // }
      // redistribute_1D_to_2D(S1D, Sr);
    // }
    // Sc = Sr;
    // auto endtime = MPI_Wtime();
    // if (!mpi_rank())
      // cout << "# Apply time " << (endtime-starttime)
           // << " seconds, per vector "
           // << ((endtime-starttime)/Ncol)
           // << " seconds" << endl;
  // }		   

// #elif FAST_H_SAMPLING == 2
  void operator()(DistM_t &R, DistM_t &Sr, DistM_t &Sc) {
    auto starttime = MPI_Wtime();
    int Ncol = R.cols();
    {
      DenseM_t S1D(_Hrows, Ncol);
      {
	DenseM_t R1D = redistribute_2D_to_1D(R);
	c_hodlr_mult(R1D.data(), S1D.data(), &_Hrows, &Ncol, &ho_bf, &option, &stats, &ptree);	
	}	
      redistribute_1D_to_2D(S1D, Sr);
    }
    Sc = Sr;
    auto endtime = MPI_Wtime();
    if (!mpi_rank())
      cout << "# Apply time " << (endtime-starttime)
           << " seconds, per vector "
           << ((endtime-starttime)/Ncol)
           << " seconds" << endl;		   
		   
  }

// #else

  // void times(DenseM_t &R, DistM_t &S, int Rprow) {
    // const auto B = S.MB();
    // const auto Bc = S.lcols();
    // DenseM_t Asub(B, B);
// #pragma omp parallel for firstprivate(Asub) schedule(dynamic)
    // for (int lr = 0; lr < S.lrows(); lr += B) {
      // const size_t Br = std::min(B, S.lrows() - lr);
      // const int Ar = S.rowl2g(lr);
      // for (int k = 0, Ac = Rprow*B; Ac < _n; k += B) {
        // const size_t Bk = std::min(B, _n - Ac);
        // // construct a block of A
        // for (size_t j = 0; j < Bk; j++) {
          // for (size_t i = 0; i < Br; i++) {
            // Asub(i, j) = Gauss_kernel
              // (&_data[(Ar + i) * _d], &_data[(Ac + j) * _d], _d, _h);
          // }
          // if (Ar==Ac) Asub(j,j) += _l;
        // }
        // DenseMW_t Ablock(Br, Bk, Asub, 0, 0);
        // DenseMW_t Sblock(Br, Bc, &S(lr, 0), S.ld());
        // DenseMW_t Rblock(Bk, Bc, &R(k, 0), R.ld());
        // // multiply block of A with a row-block of Rr and add result to Sr
        // gemm(Trans::N, Trans::N, 1., Ablock, Rblock, 1., Sblock);
        // Ac += S.prows() * B;
      // }
    // }
  // }

  // void operator()(DistM_t &R, DistM_t &Sr, DistM_t &Sc) {
    // Sr.zero();
    // int maxlocrows = R.MB() * (R.rows() / R.MB());
    // if (R.rows() % R.MB()) maxlocrows += R.MB();
    // int maxloccols = R.MB() * (R.cols() / R.MB());
    // if (R.cols() % R.MB()) maxloccols += R.MB();
    // DenseM_t tmp(maxlocrows, maxloccols);
    // // each processor broadcasts his/her local part of R to all
    // // processes in the same column of the BLACS grid, one after the
    // // other
    // for (int p=0; p<R.prows(); p++) {
      // if (p == R.prow()) {
        // strumpack::scalapack::gebs2d
          // (R.ctxt(), 'C', ' ', R.lrows(), R.lcols(), R.data(), R.ld());
        // DenseMW_t Rdense(R.lrows(), R.lcols(), R.data(), R.ld());
        // strumpack::copy(Rdense, tmp, 0, 0);
      // } else {
        // int recvrows = strumpack::scalapack::numroc
          // (R.rows(), R.MB(), p, 0, R.prows());
        // strumpack::scalapack::gebr2d
          // (R.ctxt(), 'C', ' ', recvrows, R.lcols(),
           // tmp.data(), tmp.ld(), p, R.pcol());
      // }
      // times(tmp, Sr, p);
    // }
    // Sc = Sr;
  // }
// #endif
};

vector<double> write_from_file(string filename) {
  vector<double> data;
  ifstream f(filename);
  string l;
  while (getline(f, l)) {
    istringstream sl(l);
    string s;
    while (getline(sl, s, ','))
      data.push_back(stod(s));
  }
  return data;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  auto P = mpi_nprocs(MPI_COMM_WORLD);
  // initialize the BLACS grid
  int npcol = floor(sqrt((float)P));
  int nprow = P / npcol;
  int ctxt, dummy, prow, pcol;
  scalapack::Cblacs_get(0, 0, &ctxt);
  scalapack::Cblacs_gridinit(&ctxt, "C", nprow, npcol);
  scalapack::Cblacs_gridinfo(ctxt, &dummy, &dummy, &prow, &pcol);
  int ctxt_all = scalapack::Csys2blacs_handle(MPI_COMM_WORLD);
  scalapack::Cblacs_gridinit(&ctxt_all, "R", 1, P);

	BLACSGrid grid(MPI_COMM_WORLD);								 
  string filename("smalltest.dat");
  int d = 8;
  string reorder("natural");
  double h = 1.;
  double lambda = 20.;
  int nmpi = P; // Number of MPI ranks for H compression
  int ninc = 1; // Increment between MPI ranks for H code
  int ACA = 1; //1: aca_basic 2:aca_withrandomtests 3:aca_full
  int kernel = 1; // Gaussian=1, Laplace=2
  double total_time = 0.;
  string mode("test");

  if (!mpi_rank())
    cout << "# usage: ./KernelRegressionMPI file d h kernel(1=Gauss,2=Laplace) "
      "reorder(natural, 2means, kd, pca) lambda nmpi ninc ACA mode(valid, test)"
         << endl;
  if (argc > 1)
    filename = string(argv[1]);
  if (argc > 2)
    d = stoi(argv[2]);
  if (argc > 3)
    h = stof(argv[3]);
  if (argc > 4)
    kernel = stoi(argv[4]);
  if (argc > 5)
    reorder = string(argv[5]);
  if (argc > 6)
    lambda = stof(argv[6]);
  if (argc > 7)
    nmpi = stoi(argv[7]);
  if (argc > 8)
    ninc = stoi(argv[8]);
  if (argc > 9)
    ACA = stoi(argv[9]);
  if (argc > 10)
    mode = string(argv[10]);
  if (!mpi_rank()) {
    cout << "# data dimension = " << d << endl;
    cout << "# kernel h = " << h << endl;
    cout << "# lambda = " << lambda << endl;
    cout << "# kernel type = "
         << ((kernel == 1) ? "Gauss" : "Laplace") << endl;
    cout << "# reordering/clustering = " << reorder << endl;
    cout << "# nmpi = " << nmpi << endl;
    // cout << "# ninc = " << ninc << endl;
    // cout << "# ACA = " << ACA << endl;
    cout << "# validation/test = " << mode << endl;
  }

  TaskTimer::t_begin = GET_TIME_NOW();
  TaskTimer timer(string("compression"), 1);

  HSSOptions<double> hss_opts;
  hss_opts.set_verbose(true);
  hss_opts.set_from_command_line(argc, argv);

  vector<double> data_train = write_from_file(filename + "_train.csv");
  vector<double> data_test = write_from_file(filename + "_" + mode + ".csv");
  vector<double> data_train_label =
      write_from_file(filename + "_train_label.csv");
  vector<double> data_test_label =
      write_from_file(filename + "_" + mode + "_label.csv");

  int n = data_train.size() / d;
  int m = data_test.size() / d;

  if (!mpi_rank())
    cout << "# matrix size = " << n << " x " << d << endl;

  if (!mpi_rank())
    cout << "# Preprocessing data..." << endl;
  timer.start();

  HSSPartitionTree cluster_tree;
  cluster_tree.size = n;
  int cluster_size = hss_opts.leaf_size();

  if (reorder == "2means") {
    recursive_2_means(data_train.data(), n, d, cluster_size, cluster_tree,
                      data_train_label.data());
  } else if (reorder == "cob") {
    recursive_cobble(data_train.data(), n, d, cluster_size, cluster_tree,
                 data_train_label.data());  
  } else if (reorder == "kd") {
    recursive_kd(data_train.data(), n, d, cluster_size, cluster_tree,
                 data_train_label.data());
  } else if (reorder == "pca") {
    recursive_pca(data_train.data(), n, d, cluster_size, cluster_tree,
                  data_train_label.data());
  }

  int nlevel=0;
  get_treelevel(cluster_tree, nlevel, 0); 
  // cout << "# tree level " << nlevel<< endl; 

  int* tree = new int[(int)pow(2,nlevel)]; //user provided array containing size of each leaf node, not used
  int leaf_ind=0;  
  get_leafsizes(cluster_tree, leaf_ind, tree);
  // for (int i=0; i<(int)pow(2,nlevel); ++i)
  // cout  << tree[i]<< endl; 
  
  
  if (!mpi_rank())
    cout << "# Preprocessing took " << timer.elapsed() << endl;
  
  if (!mpi_rank())
    cout << "# HSS compression .. " << endl;
  timer.start();

  HSSMatrixMPI<double>* K = nullptr;

  KernelMPI kernel_matrix
    (data_train, d, h, lambda, hss_opts,
     ctxt_all, nprow, npcol, nmpi, ninc, ACA, cluster_size, nlevel, tree);

  auto f0_compress = strumpack::params::flops.load();


	
#if FAST_H_SAMPLING == 2	
    // factor hodlr 	
	c_hodlr_factor(&kernel_matrix.ho_bf, &kernel_matrix.option, &kernel_matrix.stats, &kernel_matrix.ptree);		
	
#else	

  if (reorder != "natural")
    K = new HSSMatrixMPI<double>
      (cluster_tree, &grid, kernel_matrix, kernel_matrix,
       hss_opts);
  else
    K = new HSSMatrixMPI<double>
      (n, n, &grid, kernel_matrix, kernel_matrix,
       hss_opts);


  if (K->is_compressed()) {
    // reduction over all processors
    const auto max_rank = K->max_rank();
    const auto total_memory = K->total_memory();
    if (!mpi_rank())
      cout << "# created K matrix of dimension "
           << K->rows() << " x " << K->cols()
           << " with " << K->levels() << " levels" << endl
           << "# compression succeeded!" << endl
           << "# rank(K) = " << max_rank << endl
           << "# memory(K) = " << total_memory / 1e6 << " MB " << endl;
  } else {
    if (!mpi_rank())
      cout << "# compression failed!!!!!!!!" << endl;
    return 1;
  }

  if (!mpi_rank())
    cout << "#HSS compression took " << timer.elapsed() << endl;
  total_time += timer.elapsed();

  auto total_flops_compress = Allreduce
    (strumpack::params::flops.load() - f0_compress, MPI_SUM, MPI_COMM_WORLD);
  if (!mpi_rank())
    cout << "# compression flops = " << total_flops_compress << endl;

  // Starting factorization
  if (!mpi_rank())
    cout << "factorization start" << endl;
  timer.start();
  auto f0_factor = strumpack::params::flops.load();
  auto ULV = K->factor();
  if (!mpi_rank())
    cout << "# factorization time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();
  auto total_flops_factor = Allreduce
    (strumpack::params::flops.load() - f0_factor, MPI_SUM, MPI_COMM_WORLD);
  if (!mpi_rank())
    cout << "# factorization flops = " << total_flops_factor << endl;
#endif	
	
	
  // Starting solve	
  DenseMatrix<double> B(n, 1, &data_train_label[0], n);
  DenseMatrix<double> weights(B);
  DistributedMatrix<double> Bdist(&grid, n,1);
  Bdist.scatter(B);
  DistributedMatrix<double> wdist(&grid, n,1);
  wdist.scatter(weights);
  
  if (!mpi_rank())
    cout << "solve start" << endl;
  timer.start();
  auto f0_solve = strumpack::params::flops.load();
  
#if FAST_H_SAMPLING == 2  
	DenseMatrix<double> S1D(kernel_matrix._Hrows, 1);
	DenseMatrix<double> R1D = kernel_matrix.redistribute_2D_to_1D(wdist);
	int rhs=1;      
	c_hodlr_solve(S1D.data(), R1D.data(), &kernel_matrix._Hrows, &rhs, &kernel_matrix.ho_bf,&kernel_matrix.option, &kernel_matrix.stats, &kernel_matrix.ptree);	
    kernel_matrix.redistribute_1D_to_2D(S1D, wdist);
  
  
	c_hodlr_mult(S1D.data(), R1D.data(), &kernel_matrix._Hrows, &rhs, &kernel_matrix.ho_bf,&kernel_matrix.option, &kernel_matrix.stats, &kernel_matrix.ptree);	
	DistributedMatrix<double> Bcheck(&grid, n, 1);
	Bcheck.scatter(B);
    kernel_matrix.redistribute_1D_to_2D(R1D, Bcheck);  
  
#else
  K->solve(ULV, wdist);
  if (!mpi_rank())
    cout << "# solve time = " << timer.elapsed() << endl;
  total_time += timer.elapsed();
  auto total_flops_solve = Allreduce
    (strumpack::params::flops.load() - f0_solve, MPI_SUM, MPI_COMM_WORLD);
  if (!mpi_rank())
    cout << "# solve flops = " << total_flops_solve << endl;
		
  if (!mpi_rank())
    cout << "# total time (comp + fact): " << total_time << endl;
	
	auto Bcheck = K->apply(wdist);
	
#endif

c_hodlr_printstats(&kernel_matrix.stats, &kernel_matrix.ptree);	
  

  Bcheck.scaled_add(-1., Bdist);
  auto Bchecknorm = Bcheck.normF() / Bdist.normF();
  if (!mpi_rank())
    cout << "# relative error = ||B-H*(H\\B)||_F/||B||_F = "
         << Bchecknorm << endl;

		 

//------generate random x vector for test solution accuracy----		 
		 
		 
{


  vector<double> sample_vector(n);
  normal_distribution<double> normal_distr(0.0,1.0);
  for (int i = 0; i < n; i++) 
  {
    sample_vector[i] = normal_distr(generator);
  }
  DenseMatrix<double> sample_v(n, 1, &sample_vector[0], n);
  

	
	DistributedMatrix<double> Kdense_dist(&grid, n,n);		
	int lrows,lcols;
	int r,c;
	lrows = scalapack::numroc(Kdense_dist.desc()[2], Kdense_dist.desc()[4], Kdense_dist.prow(), Kdense_dist.desc()[6], Kdense_dist.nprows());
	lcols = scalapack::numroc(Kdense_dist.desc()[3], Kdense_dist.desc()[5], Kdense_dist.pcol(), Kdense_dist.desc()[7], Kdense_dist.npcols());
	// double tmp=0;
	for( int myi=1; myi<= lrows; myi++){
		r =  l2g(myi,Kdense_dist.prow(),n,Kdense_dist.nprows(),Kdense_dist.desc()[4])-1;
		for( int myj=1; myj<= lcols; myj++){
			c =  l2g(myj,Kdense_dist.pcol(),n,Kdense_dist.npcols(),Kdense_dist.desc()[5])-1;
			// if(!mpi_rank())
				// cout<<c<<endl;		
			Kdense_dist(myi-1, myj-1) = Gauss_kernel(&data_train[r*d], &data_train[c*d], d, h);
			if (r == c)
			{
			  Kdense_dist(myi-1, myj-1) = Kdense_dist(myi-1, myj-1) + lambda;
			}					
			// tmp += Kdense_dist(myi-1, myj-1);
		}
	}	
  
    DistributedMatrix<double> sample_v_dist(&grid, n,1);
	sample_v_dist.scatter(sample_v);
    DistributedMatrix<double> sample_rhs_dist(sample_v_dist);	
	gemm(Trans::N, Trans::N, 1., Kdense_dist, sample_v_dist, 0., sample_rhs_dist);
	

	

#if FAST_H_SAMPLING == 2  
	DenseMatrix<double> S1D(kernel_matrix._Hrows, 1);
	DenseMatrix<double> R1D = kernel_matrix.redistribute_2D_to_1D(sample_rhs_dist);
	int rhs=1;      
	c_hodlr_solve(S1D.data(), R1D.data(), &kernel_matrix._Hrows, &rhs, &kernel_matrix.ho_bf,&kernel_matrix.option, &kernel_matrix.stats, &kernel_matrix.ptree);	
    kernel_matrix.redistribute_1D_to_2D(S1D, sample_rhs_dist);
#else
  K->solve(ULV, sample_rhs_dist);
#endif  
  
  sample_rhs_dist.scaled_add(-1., sample_v_dist);
  double err_sol=sample_rhs_dist.normF()/sample_v_dist.normF();
  if (!mpi_rank()){
  cout << "# solution error = "<<  err_sol<< endl; 
  }  
			
		 
} 

  if (!mpi_rank())
    cout << "# Starting prediction step" << endl;
  timer.start();

  double* prediction = new double[m];
  std::fill(prediction, prediction+m, 0.);

  timer.start();
  if (kernel == 1) {
    if (wdist.active() && wdist.lcols() > 0)
#pragma omp parallel for
      for (int c = 0; c < m; c++) {
        for (int r = 0; r < wdist.lrows(); r++) {
          prediction[c] +=
            Gauss_kernel
            (&data_train[wdist.rowl2g(r) * d], &data_test[c * d], d, h)
            * wdist(r, 0);
        }
      }
  } else {
    if (wdist.active() && wdist.lcols() > 0)
#pragma omp parallel for
      for (int c = 0; c < m; c++) {
        for (int r = 0; r < wdist.lrows(); r++) {
          prediction[c] +=
            Laplace_kernel
            (&data_train[wdist.rowl2g(r) * d], &data_test[c * d], d, h)
            * wdist(r, 0);
        }
      }
  }
  MPI_Allreduce
    (MPI_IN_PLACE, prediction, m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#pragma omp parallel for
  for (int i = 0; i < m; ++i)
    prediction[i] = ((prediction[i] > 0) ? 1. : -1.);

  // compute accuracy score of prediction
  double incorrect_quant = 0;
#pragma omp parallel for reduction(+:incorrect_quant)
  for (int i = 0; i < m; ++i) {
    double a = (prediction[i] - data_test_label[i]) / 2;
    incorrect_quant += (a > 0 ? a : -a);
  }

  if (!mpi_rank())
    cout << "# prediction took " << timer.elapsed() << endl;
  if (!mpi_rank())
    cout << "# prediction score: " << ((m - incorrect_quant) / m) * 100 << "%"
         << endl << endl;

  scalapack::Cblacs_exit(1);
  MPI_Finalize();
  return 0;
}
