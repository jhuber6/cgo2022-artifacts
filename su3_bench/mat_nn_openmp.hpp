// OpenMP target offload implementation
#include <omp.h>
#include <unistd.h>

#define THREADS_PER_SITE 36
#define NUM_TEAMS 1600
#ifndef USE_VERSION
  #define USE_VERSION 2
#endif

double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c, 
              size_t total_sites, size_t iterations, size_t threads_per_team, int use_device)
{
  size_t num_teams = NUM_TEAMS;

  // Set num_teams from the command line
  int opt;
  optind = 1;
  while ((opt=getopt(g_argc, g_argv, ":n:")) != -1) {
    switch (opt) {
    case 'n':
      num_teams = atoi(optarg);
      break;
    }
  }

  if (threads_per_team == 0)
    threads_per_team = THREADS_PER_SITE;

  site *d_a, *d_c;
  su3_matrix *d_b;
  size_t len_a, len_b, len_c;
  d_a = a.data(); len_a = a.size();
  d_b = b.data(); len_b = b.size();
  d_c = c.data(); len_c = c.size();
 
  // Move A and B data to the device, Allocate C data
  double ttotal;
  #pragma omp target data map(to: d_a[0:len_a], d_b[0:len_b]) map(from: d_c[0:len_c])
  {  // begin OpenMP block

  // benchmark loop
  auto tstart = Clock::now();
#if USE_VERSION == 0
  // Baseline implementation
	// Original intent is to have teams process whole sites, 
	//   hence sites are distributed across the teams
	// However, for the Clang 10.0 OpenMP compiler this has issues in that memory gets
	//   flushed after each parallel region causing excessive global memory traffic
  // See USE_VERSION
  if (verbose >= 1) {
    std::cout << "Number of teams = " << num_teams << std::endl;
    std::cout << "Threads per team = " << threads_per_team << std::endl;
  }

  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups)
      tstart = Clock::now();
    #pragma omp target teams distribute num_teams(num_teams) thread_limit(threads_per_team)
    for(int i=0;i<total_sites;++i) {
      #pragma omp parallel for collapse(3)
      for (int j=0; j<4; ++j) {
        for(int k=0;k<3;k++) {
          for(int l=0;l<3;l++){
            Complx cc = {0.0, 0.0};
#ifndef MILC_COMPLEX
            for(int m=0;m<3;m++) {
               cc += d_a[i].link[j].e[k][m] * d_b[j].e[m][l];
            }
            d_c[i].link[j].e[k][l] = cc;
#else
            for(int m=0;m<3;m++) {
               CMULSUM(d_a[i].link[j].e[k][m], d_b[j].e[m][l], cc);
            }
            d_c[i].link[j].e[k][l].real = cc.real;
            d_c[i].link[j].e[k][l].imag = cc.imag;
#endif
          }
        }
      }
    }
  }

#elif USE_VERSION == 1
  // This version improves performance over the baseline
  // Contributed by Chris Daley, NERSC
  if (verbose >= 1) {
    std::cout << "Number of teams = " << num_teams << std::endl;
    std::cout << "Threads per team = " << threads_per_team << std::endl;
  }

  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups)
      tstart = Clock::now();
    #pragma omp target teams num_teams(num_teams) thread_limit(threads_per_team)
    {
      #pragma omp parallel
      {
        int total_teams = omp_get_num_teams();
        int team_id = omp_get_team_num();
        int sites_per_team = (total_sites + total_teams - 1) / total_teams;
        int istart = team_id * sites_per_team;
        if (istart > total_sites) istart = total_sites;
        int iend = istart + sites_per_team;
        if (iend > total_sites) iend = total_sites;

        for (int i = istart; i < iend; ++i) {
          #pragma omp for collapse(3)
          for (int j=0; j<4; ++j) {
            for(int k=0;k<3;k++) {
              for(int l=0;l<3;l++){
                Complx cc = {0.0, 0.0};
#ifndef MILC_COMPLEX
                for(int m=0;m<3;m++) {
                  cc += d_a[i].link[j].e[k][m] * d_b[j].e[m][l];
                }
                d_c[i].link[j].e[k][l] = cc;
#else
                for(int m=0;m<3;m++) {
                   CMULSUM(d_a[i].link[j].e[k][m], d_b[j].e[m][l], cc);
                }
                d_c[i].link[j].e[k][l].real = cc.real;
                d_c[i].link[j].e[k][l].imag = cc.imag;
#endif
              }
            }
          }
        }  // end of i loop
      }  // end of parallel region
    }  // end of teams region
  }

#elif USE_VERSION == 2
  // This code improves performance over above baseline
  // Similar to Cuda and OpenCL work item approach
  // Initial contribution by Xinmin Tian, Intel
  size_t num_work_items = total_sites * THREADS_PER_SITE;

  if (verbose >= 1) {
    std::cout << "Number of teams = " << num_teams << std::endl;
    std::cout << "Threads per team = " << threads_per_team << std::endl;
    std::cout << "Number of work items = " << num_work_items << std::endl;
  }

  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups)
      tstart = Clock::now();
    #pragma omp target teams distribute parallel for num_teams(num_teams)  thread_limit(threads_per_team)
    for (int id =0; id < num_work_items; id++) {
      int i = id/36;
      if (i < total_sites) {
        int j = (id%36)/9;
        int k = (id%9)/3;
        int l = id%3;

        Complx cc = {0.0, 0.0};
#ifndef MILC_COMPLEX
        for(int m=0;m<3;m++) {
          cc += d_a[i].link[j].e[k][m] * d_b[j].e[m][l];
        }
        d_c[i].link[j].e[k][l] = cc;
#else
        for(int m=0;m<3;m++) {
           CMULSUM(d_a[i].link[j].e[k][m], d_b[j].e[m][l], cc);
        }
        d_c[i].link[j].e[k][l].real = cc.real;
        d_c[i].link[j].e[k][l].imag = cc.imag;
#endif
      }
    }
  }

#else // VERSION == 3 || VERSION == 4
  // Baseline implementation
  // Uses the purest intent of OpenMP
  // Version 3 is a prescriptive approach using OpenMP-4.5 constructs
  // Version 4 is a descriptive approach using the OpenMP-5.0 loop construct and
  // giving the compiler the freedom to choose the number of teams and threads per team
  if (verbose >= 1) {
#if USE_VERSION == 3
  #ifdef NOTARGET
    std::cout << "Number of threads = " << omp_get_max_threads() << std::endl;
  #else
    std::cout << "Number of teams = " << num_teams << std::endl;
    std::cout << "Threads per team = " << threads_per_team << std::endl;
  #endif
#elif USE_VERSION == 4
    std::cout << "Number of teams = " << "Compiler selected" << std::endl;
    std::cout << "Threads per team = " << "Compiler selected" << std::endl;
#endif
  }

  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups)
      tstart = Clock::now();
#if USE_VERSION == 3
  #ifdef NOTARGET
    #pragma omp parallel for schedule(static)
  #else
    #pragma omp target teams distribute parallel for collapse(4) num_teams(num_teams) thread_limit(threads_per_team)
  #endif
#elif USE_VERSION == 4
    #pragma omp target teams loop collapse(4)
#endif
    for(int i=0;i<total_sites;++i) {
      for (int j=0; j<4; ++j) {
        for(int k=0;k<3;k++) {
          for(int l=0;l<3;l++){
            Complx cc = {0.0, 0.0};
#ifndef MILC_COMPLEX
#if USE_VERSION == 4
            #pragma omp loop bind(thread)
#endif
            for(int m=0;m<3;m++) {
               cc += d_a[i].link[j].e[k][m] * d_b[j].e[m][l];
            }
            d_c[i].link[j].e[k][l] = cc;
#else
#if USE_VERSION == 4
            #pragma omp loop bind(thread)
#endif
            for(int m=0;m<3;m++) {
               CMULSUM(d_a[i].link[j].e[k][m], d_b[j].e[m][l], cc);
            }
            d_c[i].link[j].e[k][l].real = cc.real;
            d_c[i].link[j].e[k][l].imag = cc.imag;
#endif
          }
        }
      }
    }
  }

#endif

  ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();
  } // end of OpenMP block, C gets moved back to the host

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
