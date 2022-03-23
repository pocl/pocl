/* Tests a kernel with an infinite loop.

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#define WORK_ITEMS 1

static char
kernelSourceCode[] =
"void test_arrays_float(global int* data, int const ndata)\n"
"{\n"
"  float s = 0.0f;\n"
"  for (int i=0; i<ndata; ++i) {\n"
"    s += as_float(data[i]);\n"
"  }\n"
"  printf(\"sum=%g\\n\", s);\n"
"  for (;;);\n"
"}\n"
"\n"
"kernel void test_kernel(global int *input, global int* output)\n"
"{\n"
"  test_arrays_float(input, 2);\n"
"}\n";

int
main(void)
{
    int output[2];
    int input[2];
    input[0] = 1;
    input[1] = 2;

    try {
        std::vector<cl::Platform> platformList;

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

        // Build program
        program.build(devices);

        // Create buffer for that uses the host ptr C
        cl::Buffer outBuffer = cl::Buffer(
            context, 
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
            2 * sizeof(int), 
            (void *) &output[0]);

        cl::Buffer inBuffer = cl::Buffer(
            context, 
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
            2 * sizeof(int), 
            (void *) &input[0]);

        // Create kernel object
        cl::Kernel kernel(program, "test_kernel");

        // Set kernel args
        kernel.setArg(0, inBuffer);
        kernel.setArg(1, outBuffer);

        // Create command queue
        cl::CommandQueue queue(context, devices[0], 0);
 
        // Do the work
        queue.enqueueNDRangeKernel(
            kernel, 
            cl::NullRange, 
            cl::NDRange(1),
            cl::NullRange);

        /* The call should return cleanly, the compiler should not crash. */

        std::cout << "OK" << std::endl;

        /* Force exit of the process regardless of the running kernel thread
           by replacing the process with a dummy process. */
        execlp("true", "true", NULL);
    } 
    catch (cl::Error &err) {
         std::cerr
             << "ERROR: "
             << err.what()
             << "("
             << err.err()
             << ")"
             << std::endl;
    }

    return EXIT_FAILURE;
}
