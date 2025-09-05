// A regression test for `pocl-level0.cc:1640: void
// pocl_level0_free(cl_device_id, cl_mem): Assertion
// `Mem->mem_host_ptr != nullptr' failed.`
//
// Copyright (c) 2025 Henry Linjamäki / Intel Finland Oy
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal// in the Software without restriction, including without limitation the
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
// FROM,// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <iostream>
#include <vector>

static cl::Buffer createSubBuffer([[maybe_unused]] cl::Context &Ctx,
                                  cl::Buffer &ParentBuf, size_t Origin,
                                  size_t Size) {
  cl_buffer_region Region{Origin, Size};
  return ParentBuf.createSubBuffer(0, CL_BUFFER_CREATE_TYPE_REGION, &Region);
}

int main() try {
  unsigned PlatformIdx = 0;

  std::vector<cl::Platform> Platforms;
  cl::Platform::get(&Platforms);
  cl::Platform Platform = Platforms.at(PlatformIdx);
  std::cout << "Selected platform: " << Platform.getInfo<CL_PLATFORM_NAME>()
            << std::endl;

  std::vector<cl::Device> Devices;
  Platform.getDevices(CL_DEVICE_TYPE_ALL, &Devices);
  if (Devices.size() < 2) {
    std::cerr << "Need two devices. SKIP.\n";
    return 77;
  }

  Devices.resize(2);
  cl::Device Dev0 = Devices.at(0);
  cl::Device Dev1 = Devices.at(1);

  std::cout << "Selected devices:\n";
  for (unsigned I = 0; I < 2; I++)
    std::cout << I << ") " << Devices[I].getInfo<CL_DEVICE_NAME>() << std::endl;

  auto Ctx = cl::Context(Devices);
  size_t Alignment =
      std::max<size_t>(Dev0.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>(),
                       Dev1.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>());

  std::vector<unsigned char> Data(2 * Alignment, 1);
  {
    auto PB = cl::Buffer(Ctx, Data.begin(), Data.end(), /*readOnly=*/false);
    auto SB = createSubBuffer(Ctx, PB, 0, Alignment);
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
