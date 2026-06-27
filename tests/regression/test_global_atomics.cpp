// Regression test for integer global atomics on CPU devices.
//
// On AArch64 the "generic" subtarget enables outline-atomics, which lowers
// integer atomicrmw (add/sub/and/or/xor/xchg/cmpxchg) to calls to libgcc
// helpers (__aarch64_ldadd4_acq_rel, ...). When kernels are linked in-process
// with lld, those helpers (static-only in libgcc.a, not exported by any loaded
// library) are left unresolved and the atomics silently no-op -- e.g. an
// atomic_add accumulator stays at its initial value. atomic_max/min have no
// LSE/outline form and are always inlined, so they kept working. This test
// checks that the read-modify-write actually lands in memory.

#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <iostream>

const char *SOURCE = R"RAW(
  __kernel void add_one(__global int *p) { atomic_add(p, 1); }
  __kernel void sub_one(__global int *p) { atomic_sub(p, 1); }
  __kernel void or_bits(__global int *p) { atomic_or(p, 1 << (get_global_id(0) & 31)); }
)RAW";

int main(int, char **) {
  try {
    cl::Device device = cl::Device::getDefault();
    cl::Context context = cl::Context::getDefault();
    cl::CommandQueue queue = cl::CommandQueue::getDefault();

    cl::Program program(context, SOURCE);
    program.build();

    const int N = 4096;
    auto run = [&](const char *name, cl_int init) -> cl_int {
      cl_int v = init;
      cl::Buffer buf(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                     sizeof(cl_int), &v);
      cl::Kernel kernel(program, name);
      kernel.setArg(0, buf);
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(N),
                                 cl::NullRange);
      queue.enqueueReadBuffer(buf, CL_TRUE, 0, sizeof(cl_int), &v);
      return v;
    };

    cl_int add = run("add_one", 0);                 // expect N
    cl_int sub = run("sub_one", N);                 // expect 0
    cl_int orr = run("or_bits", 0);                 // N>=32 sets all 32 bits -> 0xFFFFFFFF
    cl_int orr_expected = -1;

    bool ok = (add == N) && (sub == 0) && (orr == orr_expected);
    std::cout << "atomic_add=" << add << " (expect " << N << "), "
              << "atomic_sub=" << sub << " (expect 0), "
              << "atomic_or=0x" << std::hex << orr << std::dec
              << " (expect 0xffffffff)\n";

    if (ok) {
      std::cout << "OK\n";
      return EXIT_SUCCESS;
    }
    std::cout << "FAIL\n";
    return EXIT_FAILURE;
  } catch (cl::Error &err) {
    std::cout << "FAIL with OpenCL error = " << err.what() << " (" << err.err()
              << ")\n";
    return EXIT_FAILURE;
  }
}
