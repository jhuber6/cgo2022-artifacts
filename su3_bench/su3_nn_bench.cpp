#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/resource.h>
#include <math.h>
#include <vector>
#include <iostream>
#include <string>
#include <cassert>
#include <cmath>
#include <complex>
#include <chrono>
typedef std::chrono::system_clock Clock;
#ifdef USE_OPENMP
  #include <omp.h>
#endif

#ifndef ITERATIONS
#  define ITERATIONS 100
#endif
#ifndef LDIM
#  define LDIM 32       // Lattice size = LDIM^4
#endif
#ifndef PRECISION
#  define PRECISION 2  // 1->single, 2->double
#endif

// Global variables
unsigned int verbose=1;
size_t       warmups=1;
// global argc and argv for parsing model specific parameters
int  g_argc;
char **g_argv;

#include "lattice.hpp"

// validation function used by main()
template<class T>
bool almost_equal(T x, T y, double tol)
{
  if (std::isnan(x) || std::isnan(y))
	  return (0);
  return std::abs( x - y ) < tol ;
}

// std::isnan() lacks complex support, so need a complex template
template<class T>
bool almost_equal(std::complex<T> x, std::complex<T> y, double tol)
{
  if (std::isnan(x.real()) || std::isnan(x.imag())
  ||  std::isnan(y.real()) || std::isnan(y.imag()) )
	  return (0);
  return std::abs( x - y ) < tol ;
}

// Thrust version
#ifdef USE_THRUST
template<class T>
bool almost_equal(thrust::complex<T> x, thrust::complex<T> y, double tol)
{
  if (std::isnan(x.real()) || std::isnan(x.imag())
  ||  std::isnan(y.real()) || std::isnan(y.imag()) )
	  return (0);
  return thrust::abs( x - y ) < tol ;
}
#endif

// Kokkos version
#if defined(USE_KOKKOS)
template <class T>
bool almost_equal(Kokkos::complex<T> x, Kokkos::complex<T> y, double tol) {
    if (std::isnan(x.real()) || std::isnan(x.imag()) || std::isnan(y.real()) ||
        std::isnan(y.imag()))
        return (0);
    return Kokkos::abs(x - y) < tol;
}
#endif

#ifdef RANDOM_INIT
#include <random>
std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
std::uniform_real_distribution<> dis(-1.f, 1.f);
#endif

// initializes su3_matrix to a given value
void init_link(su3_matrix *s, Complx val) {
  for(int j=0; j<4; ++j) for(int k=0; k<3; ++k) for(int l=0; l<3; ++l) {
#ifndef RANDOM_INIT
    s[j].e[k][l] = val;
#elif !defined MILC_COMPLEX
    s[j].e[k][l]=dis(gen);
#else
    s[j].e[k][l].real=dis(gen);
    s[j].e[k][l].imag=dis(gen);
#endif
  }
}

// initializes a lattice site
void make_lattice(site *s, size_t n, Complx val) {
  int nx=n;
  int ny=n;
  int nz=n;
  int nt=n;

  #pragma omp parallel for
  for(int t=0;t<nt;t++) {
    int i=t*nz*ny*nx;
    for(int z=0;z<nz;z++)for(int y=0;y<ny;y++)for(int x=0;x<nx;x++,i++){
      s[i].x=x; s[i].y=y; s[i].z=z; s[i].t=t;
      s[i].index = x+nx*(y+ny*(z+nz*t));
      if( (x+y+z+t)%2 == 0)
        s[i].parity=EVEN;
      else
        s[i].parity=ODD;
      init_link(&s[i].link[0], val);
    }
  }
}

// Include the programming model specific function for su3_mat_nn()
#ifdef USE_CUDA
  #include "mat_nn_cuda.hpp"
#elif  USE_OPENMP
  #include "mat_nn_openmp.hpp"
#elif  USE_OPENMP_CPU
  #include "mat_nn_openmp2.hpp"
#elif  USE_OPENACC
  #include "mat_nn_openacc.hpp"
#elif  USE_OPENCL
  // OpenCL 1.2 doesn't support complex data types
  #define MILC_COMPLEX
  #include "mat_nn_opencl.hpp"
#elif USE_SYCL
  #include "mat_nn_sycl.hpp"
#elif USE_DPCPP
  #include "mat_nn_dpcpp.hpp"
#elif USE_HIP
  #include "mat_nn_hip.hpp"
#elif USE_KOKKOS
  #include "mat_nn_kokkos.hpp"
#else
  #error Unknown programming model
#endif

// Main
int main(int argc, char **argv)
{
  size_t iterations = ITERATIONS;
  size_t ldim = LDIM;
  size_t threads_per_group = 128; // nominally works well across implementations
  int device = -1;                // Let implementation choose the device

  int opt;
  g_argc = argc;
  g_argv = argv;
  // parse command line for parameters
	// the options list must include flags used by the various
  //   su3_mat_nn() implementations internally,
  //   as getopt rearrages the order of arguments and
  //   can screw things up for unknown options
  while ((opt=getopt(argc, argv, ":hi:l:t:v:d:w:n:")) != -1) {
    switch (opt) {
    case 'i':
      iterations = atoi(optarg);
      break;
    case 'l':
      ldim = atoi(optarg);
      break;
    case 't':
      threads_per_group = atoi(optarg);
      break;
    case 'v':
      verbose = atoi(optarg);
      break;
    case 'd':
      device = atoi(optarg);
      break;
    case 'w':
      warmups = atoi(optarg);
      break;
    case 'h':
      fprintf(stderr, "Usage: %s [-i iterations] [-l lattice dimension] \
[-t threads per workgroup] [-d device] [-v verbosity level [0,1,2,3]] [-w warmups]\n", argv[0]);
      exit (EXIT_SUCCESS);
    }
  }

  // allocate and initialize the working lattices and B su3 matrices
  size_t total_sites = ldim*ldim*ldim*ldim;
#ifdef USE_KOKKOS
  Kokkos::ScopeGuard scope(argc, argv);
  printf("Kokkos::ExecutionSpace = %s\n", typeid(ExecSpace).name());
  h_site_view a("a", total_sites);
  h_site_view c("c", total_sites);
  h_su3_matrix_view b("b", 4);
#else
  std::vector<site> a(total_sites);
  std::vector<su3_matrix> b(4);
  std::vector<site> c(total_sites);
#endif

#ifdef USE_OPENMP_CPU
  first_touch(a.data(), b.data(), c.data(), total_sites);
#endif

  // initialize the lattices
  make_lattice(a.data(), ldim, Complx{1.0,0.0});
  init_link(b.data(), Complx{1.0/3.0,0.0});

  if (verbose >= 1) {
    printf("Number of sites = %zu^4\n", ldim);
    printf("Executing %zu iterations with %zu warmups\n", iterations, warmups);
  }

  // benchmark call
  const double ttotal = su3_mat_nn(a, b, c, total_sites, iterations, threads_per_group, device);
  if (verbose >= 1)
    printf("Total execution time = %f secs\n", ttotal);
  // calculate flops/s, etc.
  // each matrix multiply is (3*3)*4*(12 mult + 12 add) = 4*(108 mult + 108 add) = 4*216 ops
  const double tflop = (double)total_sites * 864.0;
  printf("Total GFLOP/s = %.3f\n", iterations * tflop / ttotal / 1.0e9);

  const double memory_usage = (double)sizeof(site) * (a.size() + c.size()) + sizeof(su3_matrix) * b.size();
  printf("Total GByte/s (GPU memory)  = %.3f\n", iterations * memory_usage / ttotal / 1.0e9);
  fflush(stdout);

  // Verification of the result
  for (size_t i=0;i<total_sites;++i) for(int j=0;j<4;++j)  for(int k=0;k<3;++k)  for(int l=0;l<3;++l) {
    Complx cc = {0.0, 0.0};
    for(int m=0;m<3;m++) {
      #ifdef MILC_COMPLEX
        CMULSUM( a[i].link[j].e[k][m], b[j].e[m][l], cc)
      #elif USE_KOKKOS
        cc += a(i).link[j].e[k][m] * b[j].e[m][l];
      #else
        cc += a[i].link[j].e[k][m] * b[j].e[m][l];
      #endif
    }

    #ifdef MILC_COMPLEX
      assert(almost_equal(c[i].link[j].e[k][l].real, cc.real, 1E-6));
      assert(almost_equal(c[i].link[j].e[k][l].imag, cc.imag, 1E-6));
    #elif USE_KOKKOS
      assert(almost_equal(c(i).link[j].e[k][l], cc, 1E-6));
    #else
      assert(almost_equal(c[i].link[j].e[k][l], cc, 1E-6));
    #endif
  }

  // check memory usage
  if (verbose >= 2) {
    printf("Total allocation for matrices = %.3f MiB\n", memory_usage / 1048576.0);
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0)
      printf("Approximate memory usage = %.3f MiB\n", (float)usage.ru_maxrss/1024.0);
  }
}

