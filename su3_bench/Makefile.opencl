#
# Compiler is gnu | clang
CC = g++
ifeq ($(COMPILER),clang)
  CC = clang++
endif
CFLAGS = -O3 -fopenmp -Wno-ignored-attributes -Wno-deprecated-declarations
LIBS = -lOpenCL
DEFINES = -DUSE_OPENCL
DEPENDS = su3.h lattice.hpp mat_nn_opencl.hpp

bench_f32_opencl.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) -DPRECISION=1 $(CFLAGS) $(DEFINES) -o $@ su3_nn_bench.cpp $(LIBS)

bench_f64_opencl.exe: su3_nn_bench.cpp $(DEPENDS)
	$(CC) $(CFLAGS) $(DEFINES) -o $@ su3_nn_bench.cpp $(LIBS)

all: bench_f64_opencl.exe bench_f32_opencl.exe

clean:
	rm -f *opencl.exe
