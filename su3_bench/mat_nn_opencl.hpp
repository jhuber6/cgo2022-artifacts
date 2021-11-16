// OpenCL implementation
// Preferably use the Khronos CL/cl2.hpp C++ definitions
#define CL_HPP_TARGET_OPENCL_VERSION 120 
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#ifndef DEVICE
#  define DEVICE CL_DEVICE_TYPE_ALL
#endif

#define THREADS_PER_SITE 36

//*******************  m_mat_nn.c  (in su3.a) ****************************
//  void mult_su3_nn( su3_matrix *a,*b,*c )
//  matrix multiply, no adjoints
//  C  <-  A*B
static const char kernel_src[] =
"#include <lattice.hpp>\n"
"__kernel void k_mat_nn(\n"
"  __global const site*       restrict a,\n"
"  __global const su3_matrix* restrict b,\n"
"  __global       site*       restrict c,\n"
"           const int         total_sites)\n"
"{\n"
"  int myThread = get_global_id(0);\n"
"  int mySite = myThread/36;\n"
"  if (mySite < total_sites) {\n"
"    int j = (myThread%36)/9;\n"
"    int k = (myThread%9)/3;\n"
"    int l = myThread%3;\n"
"    Complx cc = {0.0, 0.0};\n"
"    for (int m=0;m<3;m++)\n"
"      CMULSUM(a[mySite].link[j].e[k][m], b[j].e[m][l], cc);\n"
"    c[mySite].link[j].e[k][l].real = cc.real;\n"
"    c[mySite].link[j].e[k][l].imag = cc.imag;\n"
"  }\n"
"}\n";

// OpenCL implementation of su3_mat_nn()
double su3_mat_nn(std::vector<site> &a, std::vector<su3_matrix> &b, std::vector<site> &c, 
              size_t total_sites, size_t iterations, size_t wgsize, int use_device)
{ 
  // Setup OpenCL context and devices
  std::vector<cl::Device> devices;
  std::vector<cl::Platform> platforms;
  cl::Platform::get(&platforms);
  for (int i=0, d=0; i< platforms.size(); ++i) {
    std::vector<cl::Device> pdevices;
    platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &pdevices);
    for (int j = 0; j < pdevices.size(); ++j, ++d) {
      if (verbose >= 3) {
        std::string s;
        pdevices[j].getInfo(CL_DEVICE_NAME, &s);
        std::cout << "Located device " << d << ": " << s << std::endl;
      }
      devices.insert(devices.end(), pdevices[j]);
    }
  }
  if (devices.size() == 0) {
    std::cout << "ERROR: No devices found\n" << std::endl;
    exit(1);
  }

  // If a device id isn't specified, if available specify a GPU as a default
  if (use_device < 0) {
    cl_device_type device_type;
    use_device = 0;
    for (int j = 0; j < devices.size(); ++j) {
      devices[j].getInfo(CL_DEVICE_TYPE, &device_type);
      if (device_type == CL_DEVICE_TYPE_GPU) {
        use_device = j;
        break;
      }
    }
  }
  else if (use_device >= devices.size()) {
    std::cout << "ERROR: Device " << use_device << " not found\n" << std::endl;
    exit(1);
  }
  if (verbose >= 2) {
    std::string s;
    devices[use_device].getInfo(CL_DEVICE_NAME, &s);
    std::cout << "Using device: " << s << std::endl;
  }

  // Set up the OpenCl context, queue, program, etc.
  cl::Device device=devices[use_device];
  cl::Context context(device);
  cl::CommandQueue queue(context);

  // build the kernel
  char build_args[80];
  sprintf(build_args, "-I. -DPRECISION=%d -DUSE_OPENCL", PRECISION);
  if (verbose >= 2)
    std::cout << "Building Kernel with: " << build_args << std::endl;
  cl::Program program(context, std::string(kernel_src));
  if (program.build(build_args) != CL_SUCCESS) {
    std::cout << "ERROR: OpenCL kernel failed to build" << std::endl;
    exit(1);
  }

  // Declare target storage and copy A and B
  auto d_a = cl::Buffer(context, begin(a), end(a), true);
  auto d_b = cl::Buffer(context, begin(b), end(b), true);
  auto d_c = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(site)*c.size());

  // Make the kernel and set the arguments
  cl::Kernel k_mat_nn(program, "k_mat_nn");
  k_mat_nn.setArg(0, d_a);
  k_mat_nn.setArg(1, d_b);
  k_mat_nn.setArg(2, d_c);
  k_mat_nn.setArg(3, (int)total_sites);

  size_t total_wi = total_sites * THREADS_PER_SITE;
  if (wgsize == 0)  // check to make sure work group size is set
    wgsize = THREADS_PER_SITE;
  if (verbose >= 1) {
    std::cout << "Setting number of work items " << total_wi << std::endl;
    std::cout << "Setting workgroup size to " << wgsize << std::endl;
  }

  // benchmark loop
  auto tstart = Clock::now();
  for (int iters=0; iters<iterations+warmups; ++iters) {
    if (iters == warmups) {
      queue.finish(); 
      tstart = Clock::now();
	  }
    queue.enqueueNDRangeKernel(k_mat_nn, cl::NullRange, cl::NDRange(total_wi), cl::NDRange(wgsize));
  }
  queue.finish(); 
  double ttotal = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now()-tstart).count();

  // copy data back from device
  cl::copy(queue, d_c, begin(c), end(c));

  return (ttotal /= 1.0e6);
}

