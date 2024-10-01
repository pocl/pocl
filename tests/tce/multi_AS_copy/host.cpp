/* Tests copying data between different address spaces.

   Copyright (c) 2017 Ville Korhonen / Tampere University of Technology
   
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

/* This test:
   1. Creates initialized buffer and uses kernel
      to check it is ok. (this ensures that buffer ends up in the devices mem)
   2. copies buffer to another
   3. Kernel on the other device is used to check the copy result.
   4. to be sure, result is read back to host and checked again.
*/

// Enable OpenCL C++ exceptions
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <string>

#include "poclu.h"

static char
kernelSourceCode[] =
"kernel \n"
"void test_kernel(__global char *buffer, __global char *reference) {\n"
"    int i;\n"
"    for (i = 0; i < 64; ++i) {\n"
"        if(buffer[i] != reference[i])\n"
"            printf(\"%c\", (buffer[i] + 48));\n"
"    }\n"
"}\n";

int
main(void)
{
    char output[64] = {0};
    char formattedInput[64] = {1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 2, 3, 4, 1, 1, 1,
                               1, 5, 6, 7, 8, 1, 1, 1,
                               1, 9, 10, 11, 12, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1,
                               1, 1, 1, 1, 1, 1, 1, 1};

    char expectedResult[64] = {0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 1, 2, 3, 4, 0, 0,
                               0, 0, 5, 6, 7, 8, 0, 0,
                               0, 0, 9, 10, 11, 12, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0, 0, 0};

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
        assert (devices.size() == 2);

        cl::Device device1 = devices.at(0);
        cl::Device device2 = devices.at(1);

        // Create and program from source
        cl::Program::Sources sources({kernelSourceCode});
        cl::Program program(context, sources);

        // Build program
        program.build(devices);

        cl::Buffer inputBuffer = cl::Buffer(
            context,
            (cl_mem_flags)CL_MEM_COPY_HOST_PTR,
            64, (void*) formattedInput);

        cl::Buffer inputReferenceBuffer = cl::Buffer(
            context,
            (cl_mem_flags)CL_MEM_COPY_HOST_PTR,
            64, (void*) formattedInput);

        cl::Buffer outputBuffer = cl::Buffer(
            context,
            (cl_mem_flags)CL_MEM_COPY_HOST_PTR,
            64, (void *) output);

        cl::Buffer outputReferenceBuffer = cl::Buffer(
            context,
            (cl_mem_flags)CL_MEM_COPY_HOST_PTR,
            64, (void *) expectedResult);

        // Create kernel object
        cl::Kernel kernel(program, "test_kernel");

        // Create command queue
        cl::CommandQueue queue1(context, device1, CL_QUEUE_PROFILING_ENABLE);
        cl::CommandQueue queue2(context, device2, CL_QUEUE_PROFILING_ENABLE);

        // Do the work
        kernel.setArg(0, inputBuffer);
        kernel.setArg(1, inputReferenceBuffer);
        cl::Event firstEvent;
        queue1.enqueueNDRangeKernel(
            kernel, 
            cl::NullRange, 
            cl::NDRange(1),
            cl::NullRange,
            NULL, &firstEvent);

        queue1.finish();

        cl::array<size_t, 3> input_origin;
        input_origin[0] = 1;
        input_origin[1] = 2;
        input_origin[2] = 0;
        cl::array<size_t, 3> output_origin;
        output_origin[0] = 2;
        output_origin[1] = 3;
        output_origin[2] = 0;
        cl::array<size_t, 3> region;
        region[0] = 4;
        region[1] = 3;
        region[2] = 1;

        std::vector<cl::Event> copyWaitList;
        copyWaitList.push_back(firstEvent);
        cl::Event copyEvent;
        queue2.enqueueCopyBufferRect (inputBuffer, outputBuffer,
                                      input_origin, output_origin,
                                      region,
                                      8, 64,
                                      8, 64,
                                      &copyWaitList,
                                      &copyEvent);
        queue2.finish();

        kernel.setArg(0, outputBuffer);
        kernel.setArg(1, outputReferenceBuffer);
        std::vector<cl::Event> kernelWaitList;
        kernelWaitList.push_back(copyEvent);
        cl::Event lastEvent;
        queue2.enqueueNDRangeKernel(
            kernel,
            cl::NullRange,
            cl::NDRange(1),
            cl::NullRange,
            &kernelWaitList, &lastEvent);

        std::vector<cl::Event> readWaitList;
        readWaitList.push_back(lastEvent);
        cl::Event readEvent;
        queue2.enqueueReadBuffer(
            outputBuffer,
            CL_FALSE,
            0, 64,
            (void*)(output),
            &readWaitList,
            &readEvent);

        queue1.flush();
        queue2.finish();

        for (int i = 0; i < 64; ++i) {
            if (output[i] != expectedResult[i]) {
              std::cout << "unexpected result at " << i
                        << ", got " << output[i] - 0
                        << " expected " << expectedResult[i] - 0
                        << std::endl;
            }
        }
        std::cout << "OK" << std::endl;
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

    return EXIT_SUCCESS;
}
