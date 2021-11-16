#!/bin/bash
set -x

# Set exe to a specific implementation
exe=./bench_f32_cuda.exe

skip=1
count=1
metrics="--metrics dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum"

srun nv-nsight-cu-cli -s $skip -c $count $metrics $exe -i $count

