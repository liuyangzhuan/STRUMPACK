#export OMP_NUM_THREADS=1
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread
#export DATA=/project/projectdirs/sparse/liuyangz/my_research/ML/SUSY_Origin/susy_10Kn
##export DATA=/project/projectdirs/sparse/liuyangz/my_research/ML/SUSY_Origin/susy_4_5Mn

#h=0.1
#lambda=10
#nmpi=2
#srun -N 1 -n 2 -c 32 --cpu_bind=cores ./build/examples/KernelRegressionMPI $DATA 8 $h 1 2means $lambda $nmpi --hss_leaf_size 512 --hss_rel_tol 1e-2 --hss_compression_algorithm original --hss_d0 512 --hss_max_rank 204

export EXEC=./build/examples/KernelRegressionMPI
# export DATA=/global/cscratch1/sd/gichavez/kernel/datasets/MNIST/MNIST_2M_train_10K_test/mnist_10Kn
export DATA=/project/projectdirs/sparse/liuyangz/my_research/ML/MNIST_10K_train_1K_test/mnist_10Kn

export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

h=0.3
lambda=1e-10
ninc=1
aca=1

ACC=1e-1
LEAF=128

nmpi=8
echo srun -N 8 -n 8 -c 64 --cpu_bind=cores ${EXEC} ${DATA} 784 ${h} 1 2means ${lambda} ${nmpi} ${ninc} ${aca} \
--hss_leaf_size ${LEAF} --hss_rel_tol ${ACC} --hss_compression_algorithm original --hss_d0 128 --hss_max_rank 1024 2>&1 | tee -a mnist_kernel.out
time srun -N 8 -n 8 -c 64 --cpu_bind=cores ${EXEC} ${DATA} 784 ${h} 1 2means ${lambda} ${nmpi} ${ninc} ${aca}\
--hss_leaf_size ${LEAF} --hss_rel_tol ${ACC} --hss_compression_algorithm original --hss_d0 128 --hss_max_rank 1024 2>&1 | tee -a mnist_kernel.out8
