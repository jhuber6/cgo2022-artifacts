// OpenMP target offload implementation
#include <omp.h>
#include <unistd.h>

#ifndef USE_VERSION
  #define USE_VERSION 1
#endif

void first_touch(site *a, su3_matrix *b, site *c,
		 size_t total_sites)
{
#if USE_VERSION == 0
# pragma omp parallel for collapse(4)
#elif USE_VERSION == 1
# pragma omp parallel for
#elif USE_VERSION == 2
# pragma omp target teams loop collapse(4)
#elif USE_VERSION == 3
# pragma omp target teams loop
#elif USE_VERSION == 4
# pragma omp target teams distribute parallel for collapse(4)
#elif USE_VERSION == 5
# pragma omp target teams distribute parallel for
#else
    // Nothing
#endif
  for(int i=0;i<total_sites;++i) {
#if USE_VERSION == 3
# pragma omp loop bind(thread)
#endif
    for (int j=0; j<4; ++j) {
#if USE_VERSION == 3
# pragma omp loop bind(thread)
#endif
      for(int k=0;k<3;k++) {
#if USE_VERSION == 3
# pragma omp loop bind(thread)
#endif
	for(int l=0;l<3;l++){
	  Complx cc = {0.0, 0.0};
#ifndef MILC_COMPLEX
# if USE_VERSION == 2 || USE_VERSION == 3
#  pragma omp loop bind(thread)
# endif
	  for(int m=0;m<3;m++) {
	    a[i].link[j].e[k][m] = cc;
	    b[j].e[m][l] = cc;
	  }
	  c[i].link[j].e[k][l] = cc;
#else
# if USE_VERSION == 2 || USE_VERSION == 3
#  pragma omp loop bind(thread)
# endif
	  for(int m=0;m<3;m++) {
	    a[i].link[j].e[k][m].real = cc.real;
	    a[i].link[j].e[k][m].imag = cc.imag;
	    b[j].e[m][l].real = cc.real;
	    b[j].e[m][l].imag = cc.imag;
	  }
	  c[i].link[j].e[k][l].real = cc.real;
	  c[i].link[j].e[k][l].imag = cc.imag;
#endif
	}
      }
    }
  }
}

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c, 
		  size_t total_sites, size_t iterations, size_t threads_per_team, int use_device)
{
  if (verbose > 0)
    std::cout << "Number of threads = " << omp_get_max_threads() << std::endl;

  // benchmark loop
  double ttotal;
  auto tstart = Clock::now();
  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups)
      tstart = Clock::now();

#if USE_VERSION == 0
# pragma omp parallel for collapse(4)
#elif USE_VERSION == 1
# pragma omp parallel for
#elif USE_VERSION == 2
# pragma omp target teams loop collapse(4)
#elif USE_VERSION == 3
# pragma omp target teams loop
#elif USE_VERSION == 4
# pragma omp target teams distribute parallel for collapse(4)
#elif USE_VERSION == 5
# pragma omp target teams distribute parallel for
#else
    // Nothing
#endif
    for(int i=0;i<total_sites;++i) {
#if USE_VERSION == 3
# pragma omp loop bind(thread)
#endif
      for (int j=0; j<4; ++j) {
#if USE_VERSION == 3
# pragma omp loop bind(thread)
#endif
        for(int k=0;k<3;k++) {
#if USE_VERSION == 3
# pragma omp loop bind(thread)
#endif
          for(int l=0;l<3;l++){
            Complx cc = {0.0, 0.0};
#ifndef MILC_COMPLEX
# if USE_VERSION == 2 || USE_VERSION == 3
#  pragma omp loop bind(thread)
# endif
            for(int m=0;m<3;m++) {
               cc += a[i].link[j].e[k][m] * b[j].e[m][l];
            }
            c[i].link[j].e[k][l] = cc;
#else
# if USE_VERSION == 2 || USE_VERSION == 3
#  pragma omp loop bind(thread)
# endif
            for(int m=0;m<3;m++) {
               CMULSUM(a[i].link[j].e[k][m], b[j].e[m][l], cc);
            }
            c[i].link[j].e[k][l].real = cc.real;
            c[i].link[j].e[k][l].imag = cc.imag;
#endif
          }
        }
      }
    }
  }

  ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  // It is not possible to check for NaNs when the application is compiled with -ffast-math
  // Therefore we print out the calculated checksum as a manual check for the user.
  // This is helpful when using LLVM/Clang-10.0 to compile the OpenMP target offload
  // implementation without MILC_COMPLEX (i.e. using std::complex).
  double sum = 0.0;
  for (int i=0;i<total_sites;++i) for(int j=0;j<4;++j)  for(int k=0;k<3;++k)  for(int l=0;l<3;++l) {
    Complx cc = {0.0, 0.0};
    for(int m=0;m<3;m++) {
      #ifdef MILC_COMPLEX
        CMULSUM( a[i].link[j].e[k][m], b[j].e[m][l], cc)
      #else
        cc += a[i].link[j].e[k][m] * b[j].e[m][l];
      #endif
    }

    #ifdef MILC_COMPLEX
      sum += c[i].link[j].e[k][l].real;
    #else
      sum += std::real(c[i].link[j].e[k][l]);
    #endif
  }
  sum /= (double)total_sites;
  if (almost_equal(sum, 4.0*sizeof(su3_matrix)/(sizeof(Complx)), 1E-6)) {
    if (verbose > 0)
      printf("Checksum SUCCESS... though please be diligent and check the "
      "following value is not NaN: checksum=%.0lf\n", sum);
  } else {
    printf("Checksum FAILURE\n");
  }

  return (ttotal /= 1.0e6);
}
