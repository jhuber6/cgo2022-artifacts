# Artifact Appendix

## Abstract

Our artifact provides the benchmarks used to evaluate the inter-procedural
OpenMP optimizations implemented for this work. These benchmarks were evaluated
using LLVM 12.0.1 as the baseline against a development branch of LLVM
containing our changes with CUDA version 11.0. All but one of these patches have
landed upstream, so any build of LLVM containing the commit hash `29a3e3dd7bed`
should be sufficient for general testing. Evaluation was done on a single Nvidia
V100 GPU node, only kernel time was considered for benchmarking to measure the
impact of our optimizations on the GPU.

## Description

This artifact contains the benchmarks and some scripts to build an OpenMP
offloading compatible LLVM compiler. The benchmarks are taken directly from
their repositories and only the build systems have been modified to build for
V100 gpus with LLVM OpenMP offloading.

### Artifact check-list

  - Algorithm: Inter-procedural optimization using OpenMP runtime knowledge.
  - Program: miniQMC, XSBench, RSBench, and SU3Bench (sources included).
  - Compilation: Clang with OpenMP Nvidia offloading post commit `29a3e3dd7bed`.
    Approximately after August 5th 2021.
  - Transformations: OpenMP runtime call and general code transformation.
  - Hardware: Tests were run using an Nvidia V100 GPU, compute capability
    `sm_70` on a Linux system.
  - Software: Tests were run using CUDA 11.0 for the and LLVM release 12.0.1
    with OpenMP offloading as the baseline compiler.
  - Metrics: Results were measured as the total time spent in GPU kernels via
    `nvprof`.
  - How much time is needed to prepare workflow (approximately)?: Building LLVM
    from scratch should take under and hour.
  - Publicly available?: Yes.

### Hardware dependencies

Our benchmarks were run on an Nvidia V100 GPU, whose compute capability is
`sm_70`. Running these tests will require a GPU accelerated system with
functional offloading via the CUDA RTL.

### Software dependencies

Building and running all the benchmarks requires an up-to-date CUDA installation (We
used version 11.0), libelf, at least CMake version 3.17, and a BLAS/LAPACK library in
addition to the standard dependencies for building LLVM.

## Installation

We have provided a script to assist in building an OpenMP offloading compatible
version of the LLVM compiler that contains our contributions. Running the script
will fetch the git repository and build using the `29a3e3dd7bed` hash. Once it's
finished the user can add the library and executables to their path.

```
$ ./build_llvm.sh
```

If the build completed without errors, add the newly installed compiler to your
environment.

```
$ export PATH=${PREFIX}/bin:'$PATH'
$ export LD_LIBRARY_PATH=${PREFIX}/lib:'$LD_LIBRARY_PATH'
```

### Building and running benchmarks

There are scripts called `build.sh` and `run.sh` provided that will attempt to
build and run the OpenMP offloading and CUDA version each benchmark. This
workflow can be modified to perform individual tests. To build the benchmarks,
run.

```
$ ./build.sh
```

And to do a simple run on all the benchmarks using `nvprof` run,

```
$ ./run.sh
```

## Evaluation and expected results

The expected results should show improvements in execution time compared to the
LLVM 12.0.1 release for all applications. Each build should also show remarks
indicating which optimizations were triggered. The optimizations triggered
should match those described in the paper except for SU3Bench.

## Experiment customization

The experiments can be customized as we did using special LLVM flags to disable
certain features, these flags are:

  - openmp-opt-disable-spmdization
  - openmp-opt-disable-deglobalization
  - openmp-opt-disable-state-machine-rewrite
  - openmp-opt-disable-folding

Flags can be added to the makefile for each benchmark, or to the CMake
configuration for miniQMC, as shown in the `build.sh` file.

## Notes

The SU3Bench evaluation done in the paper uses a patch that has not landed yet,
(https://reviews.llvm.org/D102107). This means that the local variables will not
be put in stack memory, and will be placed in shared memory with HeapToStack.
Some results will vary because of the moving nature of LLVM.
