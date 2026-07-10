// Regression test for CPU kernels whose private stack frame is large enough to
// require a target ABI stack-probe helper (___chkstk_ms on x86_64 MinGW).

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

static const char *Source = R"RAW(
__kernel void large_stack_frame(__global uint *out) {
  const uint gid = get_global_id(0);
  volatile uint scratch[2048];

  for (uint i = 0; i < 2048; ++i)
    scratch[i] = (i * 33u) ^ (gid + 17u);

  uint acc = 0;
  for (uint i = 0; i < 2048; ++i)
    acc += scratch[i];

  out[gid] = acc;
}
)RAW";

static uint32_t expected(uint32_t Gid) {
  uint32_t Acc = 0;
  for (uint32_t I = 0; I < 2048; ++I)
    Acc += (I * 33u) ^ (Gid + 17u);
  return Acc;
}

int main() try {
  cl::Device Device = cl::Device::getDefault();
  cl::Context Context = cl::Context::getDefault();
  cl::CommandQueue Queue = cl::CommandQueue::getDefault();

  cl::Program Program(Context, Source);
  Program.build({Device});

  constexpr size_t N = 16;
  std::vector<cl_uint> Out(N, 0);
  cl::Buffer OutBuf(Context, CL_MEM_WRITE_ONLY, Out.size() * sizeof(cl_uint));

  cl::Kernel Kernel(Program, "large_stack_frame");
  Kernel.setArg(0, OutBuf);
  Queue.enqueueNDRangeKernel(Kernel, cl::NullRange, cl::NDRange(N),
                             cl::NullRange);
  Queue.enqueueReadBuffer(OutBuf, CL_TRUE, 0, Out.size() * sizeof(cl_uint),
                          Out.data());

  for (size_t I = 0; I < Out.size(); ++I) {
    uint32_t Want = expected(static_cast<uint32_t>(I));
    if (Out[I] != Want) {
      std::cout << "FAIL at " << I << ": got " << Out[I] << ", expected "
                << Want << "\n";
      return EXIT_FAILURE;
    }
  }

  std::cout << "OK\n";
  return EXIT_SUCCESS;
} catch (cl::Error &Err) {
  std::cout << "FAIL with OpenCL error = " << Err.what() << " (" << Err.err()
            << ")\n";
  return EXIT_FAILURE;
}
