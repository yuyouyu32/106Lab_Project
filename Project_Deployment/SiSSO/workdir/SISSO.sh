#!/bin/sh
#BSUB -q test
#BSUB -n 32
#BSUB -o SISSO.out
#BSUB -e SISSO.err
#BSUB -J testjob
#BSUB -x

ncpus=`cat $LSB_DJOB_HOSTFILE | wc -l`
source /data/soft/compiler/intel/v2013/2013intel.sh
source /data/soft/intel/impi/4.1.0.024/bin64/mpivars.sh
#source /data/home/xjsjswf/RESCU/RESCUPackage/barc
#mpirun -machine $LSB_DJOB_HOSTFILE  -np ${ncpus} rescu --smi -i scf.input
mpirun -np ${ncpus} /data/home/xjsjswf/bin/SISSO >log
