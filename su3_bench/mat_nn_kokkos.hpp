// Kokkos implementation
#include <Kokkos_Core.hpp>

#define THREADS_PER_SITE 36
#define NUM_TEAMS 1600

using ExecSpace = Kokkos::DefaultExecutionSpace;
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using d_site_view = Kokkos::View<site *, ExecSpace>;
using d_su3_matrix_view = Kokkos::View<su3_matrix *, ExecSpace>;
using h_site_view = Kokkos::View<site *, HostExecSpace>;
using h_su3_matrix_view = Kokkos::View<su3_matrix *, HostExecSpace>;
//
//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints
//  C  <-  A*B

double k_mat_nn(size_t iterations, d_site_view a, d_su3_matrix_view b,
                d_site_view c, int total_sites, int blocksPerGrid,
                int threadsPerBlock) {
    using team_policy =
        Kokkos::TeamPolicy<ExecSpace,
                           Kokkos::IndexType<size_t>>;
    using member_type = team_policy::member_type;
    team_policy policy(blocksPerGrid, threadsPerBlock);

    Kokkos::Timer start;
    for (size_t iters = 0; iters < iterations + warmups; ++iters) {
        if (iters == warmups) {
            Kokkos::fence();
            start.reset();
        }
        Kokkos::parallel_for(
            "k_mat_nn", policy, KOKKOS_LAMBDA(const member_type &team) {
                int myThread =
                    team.team_size() * team.league_rank() + team.team_rank();
                int mySite = myThread / 36;
                if (mySite < total_sites) {
                    int j = (myThread % 36) / 9;
                    int k = (myThread % 9) / 3;
                    int l = myThread % 3;
                    Complx cc = {0.0, 0.0};
                    for (int m = 0; m < 3; m++)
                        cc += a(mySite).link[j].e[k][m] * b(j).e[m][l];

                    c(mySite).link[j].e[k][l] = cc;
                }
            });
        Kokkos::fence();
    }

    return (start.seconds());
}

double su3_mat_nn(h_site_view &a, h_su3_matrix_view &b, h_site_view &c,
                  size_t total_sites, size_t iterations, size_t threadsPerBlock,
                  int use_device) {
    if (threadsPerBlock == 0) threadsPerBlock = THREADS_PER_SITE;
    double sitesPerBlock = (double)threadsPerBlock / THREADS_PER_SITE;
    int blocksPerGrid = total_sites / sitesPerBlock + 0.999999;

    if (verbose >= 1) {
        printf("Number of blocks set to %d\n", blocksPerGrid);
        printf("Threads per block set to %zu\n", threadsPerBlock);
        printf("Device number set to %d\n", use_device);
    }

    d_site_view d_a("d_a", total_sites);
    d_site_view d_c("d_c", total_sites);
    d_su3_matrix_view d_b("d_b", 4);

    Kokkos::deep_copy(d_a, a);
    Kokkos::deep_copy(d_b, b);

    double ttotal = k_mat_nn(iterations, d_a, d_b, d_c, total_sites,
                             blocksPerGrid, threadsPerBlock);

    Kokkos::deep_copy(c, d_c);

    return ttotal;
}
