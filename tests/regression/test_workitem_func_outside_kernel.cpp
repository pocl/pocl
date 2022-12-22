/* tests that non-kernel functions which call work-item functions like
 * get_global_id() can be compiled

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   Copyright (c) 2022 Michal Babej / Intel Finland Oy

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

// Enable OpenCL C++ exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

#include "pocl_opencl.h"

#define WORK_ITEMS 128
#define BUFFER_SIZE 128

#define STRINGIFY(X, Y) X #Y
#define BUILD_OPTION(NUM) STRINGIFY("-DBUFFER_SIZE=", NUM)

static char
kernelSourceCode[] = R"raw(

  void test5(__global float *b) {
    size_t gid = get_global_id(0);
    float f = sin(b[gid]);
    printf("calling printf in test5: %u | %f\n", (unsigned)gid, f);
  }

  __kernel void test1 (__global const float *a) {
    size_t gid = get_global_id(0);
    printf("calling printf in test1: %u | %f\n", (unsigned)gid, a[gid]);
    test5(a);
  }

)raw";

int
main(void)
{
    float A[BUFFER_SIZE];
    unsigned errors = 0;
    std::random_device RandomDevice;
    std::mt19937 Mersenne{RandomDevice()};
    std::uniform_real_distribution<float> UniDist{100.0f, 200.0f};

    try {
        std::vector<cl::Platform> platformList;

        cl::Platform::get(&platformList);

        cl_context_properties cprops[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_CPU|CL_DEVICE_TYPE_GPU, cprops);

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        cl::Program::Sources sources({kernelSourceCode});
        cl::Program program(context, sources);

        for (int i = 0; i < BUFFER_SIZE; ++i)
            A[i] = UniDist(Mersenne);

        program.build(devices, BUILD_OPTION(BUFFER_SIZE));

        cl::Buffer aBuffer = cl::Buffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            BUFFER_SIZE * sizeof(float),
            (void *) &A[0]);

        cl::Kernel test1(program, "test1");

        test1.setArg(0, aBuffer);

        cl::CommandQueue queue(context, devices[0], 0);

        queue.enqueueNDRangeKernel(
            test1,
            cl::NullRange,
            cl::NDRange(WORK_ITEMS),
            cl::NullRange);

        queue.finish();
        platformList[0].unloadCompiler();

        if (errors) {
          std::cout << "FAILED\n";
          return EXIT_FAILURE;
        } else {
          std::cout << "OK" << std::endl;
          return EXIT_SUCCESS;
        }
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
