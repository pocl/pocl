/* Tests a kernel with a work-group barrier followed by an infinite loop.

   This deterministically exercises the CBS (sub-group) work-group handler on a
   kernel that has no exit block: the barrier makes the handler chooser pick
   CBS, and the infinite loop leaves the function with no return. CBS used to
   abort such kernels in SubCFGFormation ("Invalid kernel! No kernel exits!");
   it must instead fall back gracefully (to the work-item loops handler) as it
   already does for barrier-free kernels. Unlike regression/infinite_loop, the
   explicit barrier triggers the CBS path on every platform rather than relying
   on an implicit barrier being inserted, which is build/target dependent.

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
                 2026 PoCL developers

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

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include <thread>
#include <chrono>

#define WORK_ITEMS 4

static char
kernelSourceCode[] =
"kernel void test_kernel(global int *input, global int* output)\n"
"{\n"
"  size_t lid = get_local_id(0);\n"
"  output[lid] = input[lid] + 1;\n"
"  barrier(CLK_GLOBAL_MEM_FENCE | CLK_LOCAL_MEM_FENCE);\n"
"  for (;;);                       /* never returns: no kernel exit */\n"
"}\n";

int
main(void)
{
    int output[WORK_ITEMS];
    int input[WORK_ITEMS];
    for (int i = 0; i < WORK_ITEMS; ++i) {
        input[i] = i;
        output[i] = 0;
    }

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

        cl::Buffer outBuffer = cl::Buffer(
            context,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
            WORK_ITEMS * sizeof(int),
            (void *) &output[0]);

        cl::Buffer inBuffer = cl::Buffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            WORK_ITEMS * sizeof(int),
            (void *) &input[0]);

        // Create kernel object
        cl::Kernel kernel(program, "test_kernel");

        // Set kernel args
        kernel.setArg(0, inBuffer);
        kernel.setArg(1, outBuffer);

        // Create command queue
        cl::CommandQueue queue(context, devices[0], 0);

        // Do the work: one work-group of WORK_ITEMS so the barrier is real.
        queue.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(WORK_ITEMS),
            cl::NDRange(WORK_ITEMS));

        /* The kernel build/specialization should not crash the compiler. */

        std::cout << "OK" << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(1200));

        // Force exit of the process regardless of the running kernel thread
        // by replacing the process with a dummy process.
#if __cplusplus >= 201103L && /* C++11 */                                      \
    !defined(__APPLE__) && !(defined(__MINGW64__))
        std::quick_exit(EXIT_SUCCESS);
#else
        execlp("true", "true", NULL);
#endif
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
    std::cerr << "UNREACHABLE. Perhaps there was an uncaught STL exception."
              << std::endl;
    return EXIT_FAILURE;
}
