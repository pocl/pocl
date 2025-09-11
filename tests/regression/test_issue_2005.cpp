// A regression test from pocl/pocl#2005
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

static std::string shortNumber(size_t Val) {
  const char *Units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
  unsigned NumUnits = sizeof(Units) / sizeof(Units[0]);
  for (unsigned I = 0; I < NumUnits; I++) {
    if (Val < 1024)
      return std::to_string(Val) + " " + Units[I];
    Val /= 1024;
  }
  return std::to_string(Val) + " " + Units[NumUnits - 1];
}

static void testAllocationSize(cl::Context &Ctx, size_t AllocSize) {
  std::cout << "Test " << shortNumber(AllocSize) << " allocation." << std::endl;
  cl::Buffer Buf(Ctx, 0, AllocSize);

  // This triggers device allocations for the Buf in PoCL.
  cl_buffer_region Region{0, 1};
  auto SubBuf = Buf.createSubBuffer(0, CL_BUFFER_CREATE_TYPE_REGION, &Region);
}

int main() try {
  unsigned PlatformIdx = 0;

  std::vector<cl::Platform> Platforms;
  cl::Platform::get(&Platforms);
  cl::Platform Platform = Platforms.at(PlatformIdx);
  std::cout << "Selected platform: " << Platform.getInfo<CL_PLATFORM_NAME>()
            << std::endl;

  size_t MaxAllocSize = std::numeric_limits<size_t>::max();
  std::vector<cl::Device> Devices;
  Platform.getDevices(CL_DEVICE_TYPE_ALL, &Devices);
  for (auto &Dev : Devices) {
    std::cout << "Using device: " << Dev.getInfo<CL_DEVICE_NAME>() << std::endl;
    MaxAllocSize = std::min<size_t>(Dev.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(),
                                    MaxAllocSize);
  }
  std::cout << "Max allocation size across devices: "
            << shortNumber(MaxAllocSize) << std::endl;

  auto Ctx = cl::Context(Devices);

  size_t AllocSize = 32 * 1024 * 1024;
  for (; AllocSize < MaxAllocSize; AllocSize <<= 1)
    testAllocationSize(Ctx, AllocSize);

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
