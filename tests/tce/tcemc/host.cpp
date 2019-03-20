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
#include <CL/cl2.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cassert>

#include "poclu.h"

static char
kernelSourceCode[] = 
"kernel \n"
"void test_kernel(constant char *input,\n"
"                 __global char *output,\n"
"                 float a,\n"
"                 int b) {\n"
"    constant char* pos = input; \n"
"    while (*pos) {\n"
"        printf (\"%c\", *pos);\n"
"        ++pos;\n"
"    }\n"
"#ifdef cl_TCE_ABSF\n"
"    clABSFTCE(input[0], output[0]); \n"
"#else\n"
"#error The machine should have ADDF in the ISA\n"
"#endif\n"
"    printf(\"%f %d\", a, b);\n"
"    output[0] = 'P'; \n"
"    output[1] = 'O'; \n"
"    output[2] = 'N'; \n"
"    output[3] = 'G'; \n"
"    output[4] = '\\0'; \n"
"}\n";

int
main(void)
{
    const size_t OUTPUT_SIZE = 5;
    const char *input = "PING\0";
    char output[OUTPUT_SIZE];
    float a = 23456.0f;
    int b = 2000001;   

    try {
        std::vector<cl::Platform> platformList;

        // Pick platform
        cl::Platform::get(&platformList);

        // Pick first platform
        cl_context_properties cprops[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, cprops);

        // Query the set of devices attched to the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        
        assert (devices.size() == 1);

        cl::Device device = devices.at(0);

        assert (strncmp(device.getInfo<CL_DEVICE_NAME>().c_str(), "tta", 3)==0 );

        a = poclu_bswap_cl_float (device(), a);
        b = poclu_bswap_cl_int (device(), b);

        // Create and program from source
        cl::Program::Sources sources({kernelSourceCode});
        cl::Program program(context, sources);

        // Build program
        program.build(devices);

        cl::Buffer inputBuffer = cl::Buffer(
            context, 
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
            strlen (input), (void *) &input[0]);

        // Create buffer for that uses the host ptr C
        cl::Buffer outputBuffer = cl::Buffer(
            context, 
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
            OUTPUT_SIZE, (void *) &output[0]);

        // Create kernel object
        cl::Kernel kernel(program, "test_kernel");

        // Set kernel args
        kernel.setArg(0, inputBuffer);
        kernel.setArg(1, outputBuffer);
        kernel.setArg(2, a);
        kernel.setArg(3, b);

        // Create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
 
        cl::Event enqEvent;

        // Do the work
        queue.enqueueNDRangeKernel(
            kernel, 
            cl::NullRange, 
            cl::NDRange(1),
            cl::NullRange,
            NULL, &enqEvent);
 
        cl::Event mapEvent;
        (int *) queue.enqueueMapBuffer(
            outputBuffer,
            CL_TRUE, // block 
            CL_MAP_READ,
            0, OUTPUT_SIZE, NULL, &mapEvent);
       
        if (std::string(output) == "PONG") 
            std::cout << "OK\n";
        else
            std::cerr << "FAIL, received: " << output << "\n";

        cl::Event unmapEvent;
        // Finally release our hold on accessing the memory
        queue.enqueueUnmapMemObject(
            outputBuffer,
            (void *) &output[0],
            NULL,
            &unmapEvent);

        queue.finish();

        assert (enqEvent.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE);
        assert (mapEvent.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE);
        assert (unmapEvent.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE);


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

        assert (
            mapEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() <=
            mapEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>());

        assert (
            mapEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() <=
            mapEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());


        assert (
            mapEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() <=
            mapEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>());

        assert (
            unmapEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>() <=
            unmapEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>());

        assert (
            unmapEvent.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() <=
            unmapEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>());

        assert (
            unmapEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() <=
            unmapEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>());

        assert (enqEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() <=
                mapEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>());

        assert (mapEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() <=
                unmapEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>());

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
