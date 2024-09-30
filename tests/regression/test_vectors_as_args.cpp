/* Tests a kernel with a struct scalar argument.

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

#ifdef CL_HPP_TARGET_OPENCL_VERSION
#undef CL_HPP_TARGET_OPENCL_VERSION
#endif
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#define WORK_ITEMS 1

char const kernelSourceCode[] = 
"kernel void test_float2(float2 input, global float* output) {\n"
" output[0] = input.x;\n"
" output[1] = input.y;\n"
"}\n"
"kernel void test_float3(float3 input, global float* output) {\n"
" output[0] = input.x;\n"
" output[1] = input.y;\n"
" output[2] = input.z;\n"
"}\n"
"kernel void test_float4(float4 input, global float* output) {\n"
" output[0] = input.x;\n"
" output[1] = input.y;\n"
" output[2] = input.z;\n"
" output[3] = input.w;\n"
"}\n"
"kernel void test_float8(float8 input, global float* output) {\n"
" output[0] = input.s0;\n"
" output[1] = input.s1;\n"
" output[2] = input.s2;\n"
" output[3] = input.s3;\n"
" output[4] = input.s4;\n"
" output[5] = input.s5;\n"
" output[6] = input.s6;\n"
" output[7] = input.s7;\n"
"}\n"
"kernel void test_float16(float16 input, global float* output) {\n"
" output[0] = input.s0;\n"
" output[1] = input.s1;\n"
" output[2] = input.s2;\n"
" output[3] = input.s3;\n"
" output[4] = input.s4;\n"
" output[5] = input.s5;\n"
" output[6] = input.s6;\n"
" output[7] = input.s7;\n"
" output[8] = input.s8;\n"
" output[9] = input.s9;\n"
" output[10] = input.sa;\n"
" output[11] = input.sb;\n"
" output[12] = input.sc;\n"
" output[13] = input.sd;\n"
" output[14] = input.se;\n"
" output[15] = input.sf;\n"
"}\n";

char const* const kernelName[5] =
{
    "test_float2",
    "test_float3",
    "test_float4",
    "test_float8",
    "test_float16"
};

int const kernelVectorSize[5] = { 2, 3, 4, 8, 16 };

int main() {
    float input[16];
    float output[16];

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

        // Create buffer for that uses the host ptr C, initialized to zero
        cl::Buffer cBuffer = cl::Buffer(
            context, 
            CL_MEM_READ_WRITE, 
            16 * sizeof(float), 
            NULL);

        // Set up input dataset
        for (int i = 0; i < 16; ++i) {
            input[i] = static_cast<float>(i + 1);
        }

        // Create command queue
        cl::CommandQueue queue(context, devices[0], 0);

        ok = true;
        for (int k = 0; k < 5; ++k) {
            // Create kernel object
            cl::Kernel kernel(program, kernelName[k]);

            // Set kernel args
            kernel.setArg(0, kernelVectorSize[k] * sizeof(float), input);
            kernel.setArg(1, cBuffer);

            // Do the work
            queue.enqueueNDRangeKernel(
                kernel, 
                cl::NullRange, 
                cl::NDRange(1),
                cl::NullRange);

            // Read buffer into output, forces kernel to complete.
            memset(output, 0, sizeof(output));
            queue.enqueueReadBuffer(
                cBuffer,
                CL_TRUE,
                0,
                kernelVectorSize[k] * sizeof(float),
                output);

            for (int i = 0; i < kernelVectorSize[k]; i++) {
                const float correct = static_cast<float>(i + 1);
                if (output[i] != correct) {
                    std::cout 
                        << kernelName[k]
                        << "(" << i << ": " << output[i] << " != " << correct 
                        << ") ";
                    ok = false;
                }
            }
        }

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
