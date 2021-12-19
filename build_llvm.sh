#!/bin/bash

set -e

ROOT=$(realpath $(dirname $0))

# -- LLVM BUILD REQUIREMENTS --
# CMake     >=3.12.3
# GCC       >=5.1.0
# python    >=2.7
# zlib      >=1.2.3.4
# GNU Make  3.79, 3.79.1

# -- LLVM TARGETS -- 
# AArch64, AMDGPU, ARM, BPF, Hexagon, Lanai, Mips, MSP430, NVPTX,
# PowerPC, RISCV, Sparc, SystemZ, WebAssembly, X86, XCore, all
export TARGETS="X86;NVPTX"

# -- LLVM PROJECTS -- 
# clang, clang-tools-extra, compiler-rt, debuginfo-tests, libc,
# libclc, libcxx, libcxxabi, libunwind, lld, lldb, openmp, parallel-libs,
# polly, pstl
export PROJECTS="clang"

# -- INSTALLATION DIRECTORY --
# where the compiler and libraries will be installed
export PREFIX=${ROOT}/clang

# -- BUILD DIRECTORY --
# where the compiler source will be checked out and built
export BUILD_DIR=${ROOT}
export LLVM_SRC=${BUILD_DIR}/llvm-project/llvm/


export GCC=$(which gcc)
export GCC_DIR=${GCC%/bin/gcc}

export THREADS=96

CMAKE_OPTIONS=" \
    -DLLVM_ENABLE_PROJECTS=${PROJECTS}                              \
    -DLLVM_TARGETS_TO_BUILD=${TARGETS}                              \
    -DLLVM_ENABLE_ASSERTIONS=ON                                     \
    -DLLVM_OPTIMIZED_TABLEGEN=ON                                    \
    -DLLVM_CCACHE_BUILD=OFF                                         \
    -DLLVM_APPEND_VC_REV=OFF                                        \
    -DLLVM_PARALLEL_LINK_JOBS=${THREADS}                            \
    -DGCC_INSTALL_PREFIX=${GCC_DIR}                                 \
    -DCLANG_OPENMP_NVPTX_DEFAULT_ARCH=sm_70                         \
    -DLIBOMPTARGET_NVPTX_COMPUTE_CAPABILITIES=70                    \
    -DLIBOMPTARGET_ENABLE_DEBUG=OFF                                 \
    -DLLVM_ENABLE_RUNTIMES=openmp"

echo "Building LLVM in ${BUILD_DIR} with ${THREADS} threads and installing to ${PREFIX}"

sleep 1

# Checkout clang from git repository or pull if it already exists
mkdir -p ${BUILD_DIR} && cd ${BUILD_DIR}
if [ -d ${BUILD_DIR}/llvm-project ]; then
    cd llvm-project
else
  echo "Archived LLVM 13 source not found"
  return 1
fi

# Create install directory if it doesn't exist
if [ ! -d ${PREFIX} ]; then
    mkdir -p ${PREFIX}
fi

mkdir -p ${BUILD_DIR}/llvm-project/build && cd ${BUILD_DIR}/llvm-project/build
cmake -G "Unix Makefiles"                                                                   \
    -DCMAKE_INSTALL_PREFIX=${PREFIX}                                                        \
    -DCMAKE_BUILD_TYPE=Release                                                              \
    ${CMAKE_OPTIONS}                                                                        \
    ${LLVM_SRC}
    
make -j ${THREADS}
make install

echo Installation Complete. Add the following to your environment.
echo export PATH=${PREFIX}/bin:'$PATH'
echo export LD_LIBRARY_PATH=${PREFIX}/lib:'$LD_LIBRARY_PATH'
