#ifndef _LATTICE_HPP
#define _LATTICE_HPP
// Adapted from lattice.h in MILC version 7

#if defined(USE_OPENCL) || defined(MILC_COMPLEX)
  #include "su3.h"
#else
#if defined(USE_CUDA) || defined(USE_HIP) && !defined(MILC_COMPLEX)
  #define USE_THRUST
#endif
  #include "su3.hpp"
#endif

#define EVEN 0x02
#define ODD  0x01

// The lattice is an array of sites
typedef struct Site {
    su3_matrix link[4];  // the fundamental gauge field
    int x, y, z, t;      // coordinates of this site
    int index;           // my index in the array
    char parity;         // is it even or odd?
#if (PRECISION == 1)
    int pad[2];  // pad out to 64 byte alignment
#else
    int pad[10];  // pad out to 64 byte alignment
#endif
#ifndef USE_OPENCL
    #ifdef USE_KOKKOS
    KOKKOS_INLINE_FUNCTION
    #endif
    Site() {}  // Use a no-op constructor to avoid NUMA initialization issues
               // The application is responsible for initialization
#endif
} site __attribute__((aligned));

#endif  // _LATTICE_HPP
