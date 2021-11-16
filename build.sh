#!/bin/bash

set -e

ROOT=$(realpath $(dirname $0))
echo "XSBench OpenMP Offloading"
cd ${ROOT}/XSBench/openmp-offload
make clean && make -j
echo "XSBench CUDA"
cd ${ROOT}/XSBench/cuda
make clean && make -j

echo "XSBench OpenMP Offloading"
cd ${ROOT}/RSBench/openmp-offload
make clean && make -j
echo "XSBench CUDA"
cd ${ROOT}/RSBench/cuda
make clean && make -j

echo "SU3Bench OpenMP Offloading"
cd ${ROOT}/su3_bench
make --file=Makefile.openmp clean && make --file=Makefile.openmp -j
echo "SU3Bench CUDA"
make --file=Makefile.cuda clean && make --file=Makefile.cuda -j

echo "SU3Bench OpenMP Offloading"
cd ${ROOT}/su3_bench
make --file=Makefile.openmp clean && make --file=Makefile.openmp -j
echo "SU3Bench CUDA"
make --file=Makefile.cuda clean && make --file=Makefile.cuda -j

cd $ROOT/miniqmc
mkdir -p ./build && cd ./build
cmake -D CMAKE_CXX_COMPILER=clang++ -D ENABLE_OFFLOAD=1 -D USE_OBJECT_TARGET=ON \
  -DCMAKE_CXX_FLAGS="-Rpass=openmp-opt -Rpass-analysis=openmp-opt -Rpass-missed=openmp-opt" ..
make clean && make -j
