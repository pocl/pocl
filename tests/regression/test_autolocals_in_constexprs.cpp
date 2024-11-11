/* AutomaticLocals pass might break the IR if it promotes a local used in a
   constant expression to an argument (which is no longer constant)
   (GitHub issue #467).

   Copyright (c) 2017 pocl developers

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

// Enable OpenCL C++ exceptions
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "poclu.h"

static char
kernelSourceCode[] =
"kernel void test_kernel (global ulong *output)\n"
"{\n"
"   local char  l_int8[3]; \n"
"   local int   l_int32[3]; \n"
"   local float l_float[3]; \n"
"   output[0] = (ulong)l_int8;\n"
"   output[1] = (ulong)l_int32;\n"
"   output[2] = (ulong)l_float;\n"
"}\n";

int
main(void)
{
  uint64_t A[3];

  std::vector<cl::Platform> platformList;
  try {

    // Pick platform
    cl::Platform::get(&platformList);

    // Pick first platform
    cl_context_properties cprops[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};
    cl::Context context(CL_DEVICE_TYPE_ALL, cprops);

    // Query the set of devices attched to the context
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

    // Create and program from source
    cl::Program::Sources sources({kernelSourceCode});
    cl::Program program(context, sources);

    for (int i = 0; i < 3; ++i)
      A[i] = 0;

    // Build program
    program.build(devices);

    cl::Buffer aBuffer = cl::Buffer(
        context,
        CL_MEM_COPY_HOST_PTR,
        3 * sizeof(uint64_t),
        (void *) &A[0]);

    // Create kernel object
    cl::Kernel kernel(program, "test_kernel");

    // Set kernel args
    kernel.setArg(0, aBuffer);

    // Create command queue
    cl::CommandQueue queue(context, devices[0], 0);

    // Do the work
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(1));
    queue.finish();

    // We don't actually care about the result.
  }
  catch (cl::Error &err) {
    std::cerr
      << "ERROR: "
      << err.what()
      << "("
      << err.err()
      << ")"
      << std::endl;
    return EXIT_FAILURE;
  }

  platformList[0].unloadCompiler();

  std::cout << "OK" << std::endl;
  return EXIT_SUCCESS;
}
