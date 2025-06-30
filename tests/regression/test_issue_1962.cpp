// A regression test from pocl/pocl#1962
//
// Copyright (c) 2025 Henry Linjamäki / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <iostream>
#include <numeric>
#include <vector>

const char Source[] = R"OCLC(
kernel void k(global uchar *p0, global uchar *p1) {
  size_t tid = get_global_id(0);
  p0[tid] += p1[tid];
}
)OCLC";

int main() try {
  unsigned PlatformIdx = 0;
  unsigned DevIdx = 0;

  std::vector<cl::Platform> Platforms;
  cl::Platform::get(&Platforms);
  cl::Platform Platform = Platforms.at(PlatformIdx);
  std::cout << "Selected platform: " << Platform.getInfo<CL_PLATFORM_NAME>()
            << std::endl;

  std::vector<cl::Device> Devices;
  Platform.getDevices(CL_DEVICE_TYPE_ALL, &Devices);
  cl::Device Dev = Devices.at(DevIdx);
  std::cout << "Selected device: " << Dev.getInfo<CL_DEVICE_NAME>()
            << std::endl;

  auto Ctx = cl::Context(Dev);
  auto CmdQ = cl::CommandQueue(Ctx, 0);
  auto Prog = cl::Program(Ctx, Source);
  Prog.build(Dev, "-cl-std=CL3.0");
  auto Kernel = cl::Kernel(Prog, "k");

  size_t Alignment = Dev.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();

  std::vector<unsigned char> Data(2 * Alignment);
  std::iota(Data.begin(), Data.end(), 1);
  auto Buf = cl::Buffer(Ctx, Data.begin(), Data.end(), /*readOnly=*/false);

  size_t Offset = Alignment;
  size_t SubSize = Alignment - 1;
  {
    cl_buffer_region Region0{0, SubSize};
    auto SubBuf0 =
        Buf.createSubBuffer(0, CL_BUFFER_CREATE_TYPE_REGION, &Region0);
    cl_buffer_region Region1{Offset, SubSize};
    auto SubBuf1 =
        Buf.createSubBuffer(0, CL_BUFFER_CREATE_TYPE_REGION, &Region1);

    Kernel.setArg(0, SubBuf0);
    Kernel.setArg(1, SubBuf1);
    CmdQ.enqueueNDRangeKernel(Kernel, cl::NullRange, cl::NDRange(SubSize),
                              cl::NullRange);
    CmdQ.finish();
    // The sub-buffers are released here.
  }

  CmdQ.enqueueReadBuffer(Buf, CL_TRUE, 0, 2 * Alignment, Data.data());

  for (unsigned I = 0; I < SubSize; I++) {
    unsigned char Expect = (1 + I) + Data.at(I + Offset);
    if (Data.at(I) != Expect) {
      std::cerr << "line " << __LINE__ << ": error at Data.at(" << I
                << "): expected '" << static_cast<unsigned>(Expect)
                << "', Got '" << static_cast<unsigned>(Data.at(I)) << "'\n";
      return 1;
    }
  }

  for (unsigned I = SubSize; I < 2 * Alignment; I++) {
    unsigned char Expect = 1 + I;
    if (Data.at(I) != Expect) {
      std::cerr << "line " << __LINE__ << ": error at Data.at(" << I
                << "): expected '" << static_cast<unsigned>(Expect)
                << "', Got '" << static_cast<unsigned>(Data.at(I)) << "'\n";
      return 1;
    }
  }

  std::cout << "OK" << std::endl;
  return 0;
} catch (cl::Error &Ex) {
  std::cout << "Caught an OpenCL exception: " << Ex.what()
            << "(code=" << Ex.err() << ")" << std::endl;
  return 2;
} catch (std::exception &Ex) {
  std::cout << "Caught an exception: " << Ex.what() << std::endl;
  return 2;
}
