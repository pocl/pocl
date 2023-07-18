/* Tests the ttasim device driver.

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

// Enable OpenCL C++ exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cassert>

#include "poclu.h"

static char
kernelSourceCode[] = 
"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n"
//"#pragma OPENCL EXTENSION cl_khr_fp16 : disable\n"
"kernel \n"
"void test_kernel(float a) {\n"
"   half h1 = (half)(a);\n"
"   half h2 = -4.f;\n"
"   float bar = (float)(h1*h1+h2+h2+h2+5.5f);\n" // 3*3-4-4-4+5.5=2.5
"   half bar1 = 0.0f;\n"
"   for (int i = 0; i < 2; i++) { \n"
"     bar1 += (float)(h1*h1+h2+h2+h2+5.5f);\n" // 3*3-4-4-4+5.5=2.5
"     h1 += a;\n"
//if the assignment is done here, it works:
//"     float bar = bar1;\n"
"     barrier(CLK_LOCAL_MEM_FENCE);\n"
"     float bar = bar1;\n"
"     printf(\"%f\\n\", bar);\n"
"   }\n"
"}\n";

int
main(void)
{
    //float a = 23456.0f;
    float a = 3.f;

    // test the poclu's half conversion functions
    printf("through conversion: %.0f\n", 
           poclu_cl_half_to_float(poclu_float_to_cl_half(42.0f)));
    fflush(stdout);
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
        
        assert (devices.size() == 1);

        cl::Device device = devices.at(0);

        a = poclu_bswap_cl_float (device(), a);

        // Create and program from source
        cl::Program::Sources sources({kernelSourceCode});
        cl::Program program(context, sources);

        // Build program
        program.build(devices);

        // Create kernel object
        cl::Kernel kernel(program, "test_kernel");

        // Set kernel args
        kernel.setArg(0, a);

        // Create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
 
        cl::Event enqEvent;

        // Do the work
        queue.enqueueNDRangeKernel(
            kernel, 
            cl::NullRange, 
            cl::NDRange(8),
            cl::NDRange(8),
            NULL, &enqEvent);
	queue.finish();
 
        assert (enqEvent.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE);

        assert (
            enqEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() <=
            enqEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>());

        assert (
            enqEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() <=
            enqEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());

        assert (
            enqEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() <
            enqEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>());

#if 0
        std::cerr << "exec time: " 
                  << enqEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
            enqEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
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

    return EXIT_SUCCESS;
}
