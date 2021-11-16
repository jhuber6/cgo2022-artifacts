#!/bin/bash

iters=100
if [ $# -lt 2 ]; then
  printf "usage: %s executable kernel_name [iterations]\n" $0
  exit 1
elif [ $# -eq 3 ]; then
  iters=$3
fi
exe=$1
name=$2

set -x
srun nvprof --kernels $name --metrics flop_count_sp,dram_read_transactions,dram_write_transactions ./${exe} -i${iters} |& grep -A3 "Kernel:" | grep -v "Kernel:" | awk {'printf "%d %d %d\n",$7,$8,$9'}
#srun nvprof --kernels $name --metrics sm_efficiency,achieved_occupancy,eligible_warps_per_cycle ./${exe} -i${iters}
#srun nvprof --kernels $name --metrics dram_read_throughput,dram_write_throughput ./${exe} -i${iters}

