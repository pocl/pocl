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
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>

#undef SRCDIR
#include "config.h"

#define WORK_ITEMS 1

// Currently assume these types map the OpenCL types, might be better to use
// the explicitly-sized types in <cstdint> under C++11 for better portability.
// 
// We also assume the same packing rules between host and target, so if this
// regression fails in the future, this might be worth investigating as to
// the origin of the problem.

struct int_single {
    cl_int a;
};

struct int_pair {
    cl_long a;
    cl_long b;
};

// i386 has a default alignment of 4 even for 64-bit types
#ifdef __i386__
#define CL_LONG_ALIGNMENT __attribute__((aligned(8)))
#else
#define CL_LONG_ALIGNMENT
#endif

struct test_struct {
    cl_int elementA;
    cl_int elementB;
    cl_long elementC CL_LONG_ALIGNMENT;
    cl_char elementD;
    cl_long elementE CL_LONG_ALIGNMENT;
    cl_float elementF;
    cl_short elementG;
    cl_long elementH CL_LONG_ALIGNMENT;
};

#undef CL_LONG_ALIGNMENT

static char
kernelSourceCode[] = 
"typedef struct int_single {\n"
"    int a; \n"
"} int_single;\n"
"typedef struct int_pair {\n"
"    long a;\n"
"    long b;\n"
"} int_pair;\n"
"typedef struct test_struct {\n"
"    int elementA;\n"
"    int elementB;\n"
"    long elementC;\n"
"    char elementD;\n"
"    long elementE;\n"
"    float elementF;\n"
"    short elementG;\n"
"    long elementH;\n"
"} test_struct;\n"
"\n"
"kernel void test_single(int_single input, global int* output) {"
" output[0] = input.a;\n"
"}\n"
"kernel void test_pair(int_pair input, global int* output) {"
" output[0] = (int)input.a;\n"
" output[1] = (int)input.b;\n"
"}\n"
"kernel void test_kernel(test_struct input, global int* output) {"
" output[0] = input.elementA;\n"
" output[1] = input.elementB;\n"
" output[2] = (int)input.elementC;\n"
" output[3] = (int)input.elementD;\n"
" output[4] = (int)input.elementE;\n"
" output[5] = (int)input.elementF;\n"
" output[6] = (int)input.elementG;\n"
" output[7] = (int)input.elementH;\n"
"}\n";

int
main(void)
{
    bool ok = true;
    int buffer_storage[8];

    int_pair input_single;
    input_single.a = 1234567;

    int_pair input_pair;
    input_pair.a = -5588;
    input_pair.b = 8855;

    test_struct input;
    input.elementA = 1;
    input.elementB = 2;
    input.elementC = 3;
    input.elementD = 4;
    input.elementE = 5;
    input.elementF = 6;
    input.elementG = 7;
    input.elementH = 8;

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
        cl::Buffer cBuffer = cl::Buffer(
            context, 
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
            8 * sizeof(int), 
            (void *) &buffer_storage[0]);

        // Create command queue
        cl::CommandQueue queue(context, devices[0], 0);

        //
        // int_single
        //

        // Create kernel object
        cl::Kernel kernel_single(program, "test_single");

        // Set kernel args
        kernel_single.setArg(0, sizeof(int_single), &input_single);
        kernel_single.setArg(1, cBuffer);

        // Do the work
        queue.enqueueNDRangeKernel(
            kernel_single, 
            cl::NullRange, 
            cl::NDRange(1),
            cl::NullRange
        );

        // Map cBuffer to host pointer. This enforces a sync with 
        // the host backing space, remember we choose GPU device.
        int* output = (int*)queue.enqueueMapBuffer(
            cBuffer,
            CL_TRUE, // block 
            CL_MAP_READ,
            0,
            1 * sizeof(int));

        if (*output != 1234567) {
           std::cout 
               << "Small struct failure - size: 4 bytes expected: 123456 actual: "
               << *output << std::endl;
           ok = false;
        }

        queue.enqueueUnmapMemObject(cBuffer, output);

        //
        // int_pair
        //

        // Create kernel object
        cl::Kernel kernel_pair(program, "test_pair");

        // Set kernel args
        kernel_pair.setArg(0, sizeof(int_pair), &input_pair);
        kernel_pair.setArg(1, cBuffer);

        // Do the work
        queue.enqueueNDRangeKernel(
            kernel_pair, 
            cl::NullRange, 
            cl::NDRange(1),
            cl::NullRange
        );

        // Map cBuffer to host pointer. This enforces a sync with 
        // the host backing space, remember we choose GPU device.
        output = (int*)queue.enqueueMapBuffer(
            cBuffer,
            CL_TRUE, // block 
            CL_MAP_READ,
            0,
            2 * sizeof(int));

        if ((output[0] != -5588) || (output[1] != 8855)) {
           std::cout 
               << "Small struct failure - size: 8 bytes expected: (-5588, 8855) actual: ("
               << output[0] << ", " << output[1] << ")" << std::endl;
           ok = false;
        }

        queue.enqueueUnmapMemObject(cBuffer, output);

        //
        // test_struct
        //

        // Create kernel object
        cl::Kernel kernel(program, "test_kernel");

        // Set kernel args
        kernel.setArg(0, sizeof(test_struct), &input);
        kernel.setArg(1, cBuffer);

        // Do the work
        queue.enqueueNDRangeKernel(
            kernel, 
            cl::NullRange, 
            cl::NDRange(1),
            cl::NullRange
        );

        // Map cBuffer to host pointer. This enforces a sync with 
        // the host backing space, remember we choose GPU device.
        output = (int*)queue.enqueueMapBuffer(
            cBuffer,
            CL_TRUE, // block 
            CL_MAP_READ,
            0,
            8 * sizeof(int));

        for (int i = 0; i < 8; i++) {
            int correct = i + 1;
            if (output[i] != correct) {
                std::cout 
                    << "F(" << i << ": " << output[i] << " != " << correct 
                    << ") ";
                ok = false;
            }
        }

        queue.finish();
        platformList[0].unloadCompiler();

        if (ok) {
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
