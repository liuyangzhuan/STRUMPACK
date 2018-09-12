export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export DATA=/project/projectdirs/sparse/liuyangz/my_research/ML/SUSY_Origin/susy_10Kn
#export DATA=/project/projectdirs/sparse/liuyangz/my_research/ML/SUSY_Origin/susy_4_5Mn

h=0.1
lambda=10
nmpi=2
srun -N 1 -n 2 -c 32 --cpu_bind=cores ./build/examples/KernelRegressionMPI $DATA 8 $h 1 2means $lambda $nmpi --hss_leaf_size 512 --hss_rel_tol 1e-2 --hss_compression_algorithm original --hss_d0 512 --hss_max_rank 2048
