#!/bin/bash

set -e
set -x

ROOT=$(realpath $(dirname $0))

cd ${ROOT}/XSBench/openmp-offload
nvprof ./XSBench -m event 2>&1
cd ${ROOT}/XSBench/cuda
nvprof ./XSBench -m event 2>&1

cd ${ROOT}/RSBench/openmp-offload
nvprof ./rsbench -m event -s large 2>&1
cd ${ROOT}/RSBench/cuda
nvprof ./rsbench -m event -s large 2>&1

cd ${ROOT}/su3_bench
nvprof ./bench_f32_openmp.exe 2>&1
nvprof ./bench_f32_cuda.exe 2>&1

cd ${ROOT}/su3_bench
nvprof ./bench_f32_openmp.exe 2>&1
nvprof ./bench_f32_cuda.exe 2>&1

cd $ROOT/miniqmc/build
nvprof env OMP_NUM_THREADS=10 ./bin/check_spo_batched -m 2 -g "2 2 1" -w 80 -n 1
