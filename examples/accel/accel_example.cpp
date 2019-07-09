/* Tests the generic accelerator device driver.

   Copyright (c) 2019 Pekka Jääskeläinen / Tampere University

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
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#ifndef _MSC_VER
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

int
main(void)
{
    try {
        std::vector<cl::Platform> platformList;

        // Pick platform
        cl::Platform::get(&platformList);

        // Pick first platform
        cl_context_properties cprops[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_CUSTOM, cprops);

        // Query the set of devices attched to the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        cl::Program program(context, devices, "pocl.add32;pocl.mul32");

        // Build program
        program.build(devices);

        uint32_t in0[64];
        uint32_t in1[64];
        uint32_t in2[64];

        uint32_t result[64];
        for (size_t i = 0; i < 64; ++i) {
            in0[i] = i + 777;
            in1[i] = 2020 - i;
            in2[i] = i + 9999;
        }

        cl::Buffer inBuffer0 = cl::Buffer(
            context,
            CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
            (cl::size_type)64*sizeof(uint32_t),
            (void*)in0);
        cl::Buffer inBuffer1 = cl::Buffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            (cl::size_type)64*sizeof(uint32_t),
            (void*)in1);
        cl::Buffer inBuffer2 = cl::Buffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            (cl::size_type)64*sizeof(uint32_t),
            (void*)in2);

        cl::Buffer outBuffer0 = cl::Buffer(
            context,
            CL_MEM_READ_WRITE,
            (cl::size_type)64*sizeof(uint32_t),
            NULL);
        cl::Buffer outBuffer1 = cl::Buffer(
            context,
            CL_MEM_READ_WRITE,
            (cl::size_type)64*sizeof(uint32_t),
            NULL);

        // Create command queue
        cl::CommandQueue queue(context, devices[0], 0);

        // Create kernel object
        cl::Kernel mul0(program, "pocl.mul32");
        mul0.setArg(0, inBuffer0);
        mul0.setArg(1, inBuffer1);
        mul0.setArg(2, outBuffer0);

        cl::Kernel mul1(program, "pocl.mul32");
        mul1.setArg(0, inBuffer1);
        mul1.setArg(1, inBuffer2);
        mul1.setArg(2, outBuffer1);

        cl::Kernel add(program, "pocl.add32");
        add.setArg(0, outBuffer0);
        add.setArg(1, outBuffer1);
        add.setArg(2, inBuffer0);


        //queue.enqueueWriteBuffer(inBuffer, CL_TRUE, 0, 128, &input[0], NULL, NULL);
        queue.enqueueNDRangeKernel(mul0, cl::NullRange, cl::NDRange(64),
                                   cl::NullRange);
        queue.enqueueNDRangeKernel(mul1, cl::NullRange, cl::NDRange(64),
                                   cl::NullRange);
        queue.enqueueNDRangeKernel(add, cl::NullRange, cl::NDRange(64),
                                   cl::NullRange);

        uint32_t* Res = (uint32_t*) queue.enqueueMapBuffer(inBuffer0, CL_FALSE,
                                                           CL_MAP_READ, 0, 256);

        queue.finish();

        int correct = 1;
        for (size_t i = 0; i < 64; ++i) {
            uint32_t value = Res[i];
            uint32_t expected = (i + 777)*(2020-i) + (2020-i)*(i+9999);
            if (value != expected) {
                printf("at idx %zd expected %i, got %i\n", i, expected, value);
                correct = 0;
            }

        }
        if (correct == 1) {
            printf("OK: Correct result!\n");
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

         return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
