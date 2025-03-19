/* Tests that a kernel with recursion does not result in crash,
   but instead a CL_BUILD_PROGRAM_FAILURE.

   Copyright (c) 2025 Henry Linjamäki / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "pocl_opencl.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <iostream>

static char SourceStr[] = R"clc(
uint fibonacci(uint n) {
  if (n < 2)
    return n;
  return fibonacci(n - 1) + fibonacci(n - 2);
}

kernel void test() { printf("%u\n", fibonacci(6)); }
)clc";

int main() try {
  std::vector<cl::Platform> PlatformList;
  cl::Platform::get(&PlatformList);

  cl_context_properties CProps[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(PlatformList[0])(), 0};
  cl::Context Context(CL_DEVICE_TYPE_ALL, CProps);

  std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Program::Sources Source({SourceStr});
  cl::Program program(Context, Source);
  program.build(Devices);
  cl::Kernel TestKernel(program, "test");
  cl::CommandQueue Queue(Context, Devices.at(0), 0);
  Queue.enqueueNDRangeKernel(TestKernel, cl::NullRange, cl::NDRange(1),
                             cl::NullRange);
  Queue.finish();

  // test should fail to build
  return 2;
} catch (cl::Error &err) {
  std::cerr << "Exception: " << err.what() << "(" << err.err() << ")"
            << std::endl;
  return (err.err() == CL_BUILD_PROGRAM_FAILURE) ? 0 : 1;
}
