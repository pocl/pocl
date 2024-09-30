/* Tests a case where one of the arguments is NULL

   Copyright (c) 2013 Victor Oliveira <victormatheus@gmail.com>
   
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

#ifdef CL_HPP_TARGET_OPENCL_VERSION
#undef CL_HPP_TARGET_OPENCL_VERSION
#endif
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#define BUFFER_SIZE (1024)

static char
kernelSourceCode[] = 
"__kernel void test_kernel (__global const float      *in,  \n"
"                           __global const float      *aux, \n"
"                           __global       float      *out) \n"
"{                                                          \n"
"  int gid = get_global_id(0);                              \n"
"  float  in_v  = in [gid];                                 \n"
"  float  aux_v = (aux)? aux[gid] : 0.5f;                   \n"
"  float  out_v;                                            \n"
"  out_v = (in_v > aux_v)? 1.0f : 0.0f;                     \n"
"  out[gid]  =  out_v;                                      \n"
"}                                                          \n";

int
main(void)
{
    float in [BUFFER_SIZE];
    float out[BUFFER_SIZE];

    for (int i = 0; i < BUFFER_SIZE; i++) {
        in[i] = (float)rand()/(float)RAND_MAX;;
    }

    std::vector<cl::Platform> platformList;
    bool ok = false;
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

        // Build program
        program.build(devices);

        // Create buffer for A and copy host contents
        cl::Buffer inBuffer = cl::Buffer(
            context, 
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
            BUFFER_SIZE * sizeof(float), 
            (void *) &in[0]);

        // Create buffer for that uses the host ptr C
        cl::Buffer outBuffer = cl::Buffer(
            context, 
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
            BUFFER_SIZE * sizeof(float), 
            (void *) &out[0]);

        // Create kernel object
        cl::Kernel kernel(program, "test_kernel");

        // Set kernel args
        kernel.setArg(0, inBuffer);
        kernel.setArg(1, (cl::Buffer) 0);
        kernel.setArg(2, outBuffer);

        // Create command queue
        cl::CommandQueue queue(context, devices[0], 0);
 
        // Do the work
        queue.enqueueNDRangeKernel(
            kernel, 
            cl::NullRange, 
            cl::NDRange(BUFFER_SIZE),
            cl::NullRange);
 
        // Map to host pointer. This enforces a sync with 
        // the host backing space, remember we choose GPU device.
        float * output = (float *) queue.enqueueMapBuffer(
            outBuffer,
            CL_TRUE, // block 
            CL_MAP_READ,
            0,
            BUFFER_SIZE * sizeof(float));

        ok = true;
        for (int i = 0; i < BUFFER_SIZE; i++) {
            if (   (in[i] >  0.5f && out[i] < 1.0f)
                || (in[i] <= 0.5f && out[i] > 0.0f))
              ok = false;
        }

        // Finally release our hold on accessing the memory
        queue.enqueueUnmapMemObject(
            outBuffer,
            (void *) output);

        queue.finish();
    }
    catch (cl::Error &err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
        return EXIT_FAILURE;
    }

    platformList[0].unloadCompiler();

    if (ok) {
        std::cout << "OK" << std::endl;
        return EXIT_SUCCESS;
    } else {
        std::cout << "FAIL\n";
        return EXIT_FAILURE;
    }
}
