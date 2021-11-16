## SU3_bench: Lattice QCD SU(3) Matrix-Matrix Multiply Microbenchmark
The purpose of this microbenchmark is to provide a means to explore different programming methodologies using a simple, but nontrivial, mathematical kernel. It is most useful in exploring the performance of current and next-generation architectures and their respective programming environments, in particular the maturity of compilers.

The kernel is based on the *mult\_su3\_nn()* SU(3) matrix-matrix multiply routine in the [MILC Lattice Quantum Chromodynamics(LQCD) code](https://github.com/milc-qcd/milc_qcd). Matrix-matrix (and matrix-vector) SU(3) operations are a fundamental building block of LQCD applications. Most LQCD applications use custom implementations of these kernels, and they are usually written in machine specific languages and/or  intrinsics in order to obtain maximum performance on advanced high-performance computing architectures such as highly parallel multi-core CPUs and GPUs.

### Usage
#### Build and Execution
To build for a given programming model, simply invoke the respective compile environment using the Makefile supplied for that model. For example, when building for OpenMP simply invoke Makefile.openmp, and then execute  `bench_f32_openmp.exe`.

```
cgpu01:su3_bench$ make -f Makefile.openmp
clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -DPRECISION=1 -DUSE_OPENMP -DMILC_COMPLEX -o bench_f32_openmp.exe su3_nn_bench.cpp
cgpu01:su3_bench$ srun bench_f32_openmp.exe
Number of sites = 32^4
Executing 100 iterations with 1 warmups
Number of teams = 1600
Threads per team = 128
Number of work items = 37748736
Checksum SUCCESS... though please be diligent and check the following value is not NaN: checksum=36
Total execution time = 0.103701 secs
Total GFLOP/s = 873.636
Total GByte/s (GPU memory)  = 647.138
```

By default, the benchmark is built using 32-bit floating-point arithmetic. To build for 64-bit floating-point, specify the 64-bit binary, `bench_fp64_xxx.exe`.

```
cgpu01:su3_bench$ make -f Makefile.openmp bench_f64_openmp.exe
clang++ -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -DUSE_OPENMP -DMILC_COMPLEX -o bench_f64_openmp.exe su3_nn_bench.cpp
```

Each programming model has its own implementation dependent compile time parameters, including multiple compiler options and multiple target GPU architectures. See the respective Makefile.xxx to understand what these options are as they are too numerous to cover in a README, and given the nature of being an active research code, they are prone to change over time.

#### Using Kokkos version
The Kokkos version uses a CMake build system. The build only requires a path to the Kokkos library install, for example:
```
mkdir build && cd build
cmake -DKokkos_ROOT=$PATH-to-kokkos-install -DCMAKE_CXX_EXTENSIONS=OFF ../
```
The `-DCMAKE_CXX_EXTENSIONS=OFF` is required by Kokkos to avoid build warnings.


#### Runtime parameters
There are several runtime parameters that control execution:

```
cgpu01:su3_bench$ srun bench_f32_openmp.exe --help
Usage: bench_f32_openmp.exe [-i iterations] [-l lattice dimension] [-t threads per workgroup] [-d device] [-v verbosity level [0,1,2,3]] [-w warmups]
```

- The dimensionality of the lattice, *L*, is set with `-l`.  The default is *L=32*, or *32x32x32x32* sites. Note that this parameter has a significant effect on memory footprint and execution time.
- The threads per work group (or block)  is set with `-t`. This is primarily used as a tuning parameter. The default is programming model dependent.
- If there is more than one target device, use `-d` to select the device of interest. For most programming model implementations, the default is the first GPU device. For some programming models, using `-v 3` will list the available devices.
- Use `-v` to control the output verbosity. The higher the number, the more verbose.
- Use `-w` and `-i` to control the number of warmups and iterations respectively. By default, a single warmup and 100 timed iterations are performed.
- Some implementations also have programing model specific flags, you many need to peruse the source code to find them though. For example with OpenMP you can use `-n num_teams` to set the total number of teams at runtime.

#### Metrics
The primary runtime metrics of interest for benchmarking are the *GFLOP/s* and *GByte/s* rates. These values are derived based on the measured time of execution for the computation, not actual based on performance counters. As such, they are also directly proportional to each other by a factor of ~1.35, the theoretical arithmetic intensity of the kernel.  For most architectures, SU3_bench is memory bandwidth bound, hence GByte/s is the most appropriate metric to use and can be compared to the peak bandwidth, or that obtained using a [STREAM benchmark](http://uob-hpc.github.io/BabelStream), for a simple roofline analysis.

### Design

#### Software Architecture
The code is written in a combination of standard C and C++ programming languages. The main driver routine, `su3_nn_bench.cpp`, is used for all programming model implementations, with programming model specifics encapsulated in a C++ include file, for example `mat_nn_openmp.hpp`. This encapsulation is achieved through a single function call interface,

```
double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c,
              size_t total_sites, size_t iterations, size_t threads_per_workgroup, int device);
```

and the return value is the total execution time in seconds.

A given programming model is chosen at compile time and specified with a preprocessor directive in its associated Makefile.xxx. For example Makefile.openmp specifies `-DUSE_OPENMP`. The `.hpp` file is conditionally included in `su3_nn_bench.cpp`.

The high level intent is to keep as much programming model specific implementation details as possible encapsulated within `su3_nn_bench.cpp`. All timing is done within the `.hpp` file, as there may be significant programming model specific initialization required before actually performing the calculation.

#### Using std::complex or MILC complex

When the benchmark was initially being developed, some compilers where unable to handle `std::complex` math in the kernel. For this reason, some implementations have two complex math methods 1) use `std::complex` and let the compiler handle the arithmetic, or 2) define a complex value using a data structure and explicitly handle the arithmetic. The latter is enabled with `-DMILC_COMPLEX` at compile time. This issue may be fully resolved at this point in time, but the different methods are left in the code in case there may be performance issues for std::complex that one may want to investigate with a particular compiler.

### Contact info
SU3_Bench is part of the [NERSC Proxy Application Suite](https://gitlab.com/NERSC/nersc-proxies/info).
