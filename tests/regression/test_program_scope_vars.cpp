/* Issues with __local pointers (lp:918801)

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
#define CL_TARGET_OPENCL_VERSION 300
#define CL_HPP_MINIMUM_OPENCL_VERSION 300
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>

#include "pocl_opencl.h"

#define BUFFER_SIZE_1 128
#define BUFFER_SIZE_2 256

#define STRINGIFY(X, Y) X #Y
#define BUILD_OPTION(NUM) STRINGIFY("-DBUFFER_SIZE=", NUM)

static char
programOneSourceCode[] = R"raw(

  global float testGlobalVar[BUFFER_SIZE];

  constant float testConstantVar[8] = { 1, 2, 3, 4, 5, 6, 7, 8};

  __kernel void test1 (__global const float *a) {
    size_t i = get_global_id(0);
    testGlobalVar[i] += a[i] * 12.0f;
    testGlobalVar[i] += testConstantVar[i % 8];
  }

  __kernel void test2 (__global const float *a) {
    size_t i = get_global_id(0);
    testGlobalVar[i] += a[i] * 7.0f;
    testGlobalVar[i] += testConstantVar[i % 8];
  }

  __kernel void test3 (__global float *out) {
    size_t i = get_global_id(0);
    out[i] = testGlobalVar[i];
  }

)raw";

static char
programTwoSourceCode[] = R"raw(

  global float testGlobalVar[BUFFER_SIZE];

  constant float testConstantVar[8] = { 10,9,8,7,6,5,4,3};

  __kernel void test1 (__global const float *a) {
    size_t i = get_global_id(0);
    testGlobalVar[i] += a[i] * 23.0f;
    testGlobalVar[i] += testConstantVar[i % 8];
  }

  __kernel void test2 (__global const float *a) {
    size_t i = get_global_id(0);
    testGlobalVar[i] += a[i] * 3.0f;
    testGlobalVar[i] += testConstantVar[i % 8];
  }

  __kernel void test3 (__global float *out) {
    size_t i = get_global_id(0);
    out[i] = testGlobalVar[i];
  }

)raw";

static char
programThreeSourceCode[] = R"raw(

  typedef struct { char c; int i; long l; } mystruct_t;
  global mystruct_t testGlobalStruct;

  __kernel void test4w (char c, int i, long l) {
    testGlobalStruct.c = c;
    testGlobalStruct.i = i;
    testGlobalStruct.l = l;
  }

  __kernel void test5r (char c, int i, long l, __global int *out) {
    int res = 0;
    if (testGlobalStruct.c == c) ++res;
    if (testGlobalStruct.i == i) ++res;
    if (testGlobalStruct.l == l) ++res;
    *out = res;
  }

)raw";

static char programFourSourceCode[] = R"raw(

float2 from_buf(float2 a) { return a; }

float2 to_buf(float2 a) { return a; }

#define INIT_VAR(a) ((float2)(a))

float2 var = INIT_VAR(0);

global float2 g_var = INIT_VAR(1);

float2 a_var[2] = { INIT_VAR(1), INIT_VAR(1) };

volatile global float2* p_var = &a_var[1];

kernel void writer( global const float2* src, uint idx ) {
  var = from_buf(src[0]);
  g_var = from_buf(src[1]);
  a_var[0] = from_buf(src[2]);
  a_var[1] = from_buf(src[3]);
  p_var = a_var + idx;
}

kernel void reader( global float2* dest, float2 ptr_write_val ) {
  *p_var = from_buf(ptr_write_val);
  dest[0] = to_buf(var);
  dest[1] = to_buf(g_var);
  dest[2] = to_buf(a_var[0]);
  dest[3] = to_buf(a_var[1]);
}

)raw";

int
main(void)
{
    float OneA[BUFFER_SIZE_1];
    float OneB[BUFFER_SIZE_1];
    float OneConstantVar[8] = { 1, 2, 3, 4, 5, 6, 7, 8};

    float TwoA[BUFFER_SIZE_2];
    float TwoB[BUFFER_SIZE_2];
    float TwoConstantVar[8] = { 10, 9, 8, 7, 6, 5, 4, 3 };

    unsigned errors = 0;
    std::random_device RandomDevice;
    std::mt19937 Mersenne{RandomDevice()};
    std::uniform_real_distribution<float> UniDist{100.0f, 200.0f};

    std::vector<cl::Platform> platformList;
    try {

        cl::Platform::get(&platformList);

        cl_context_properties cprops[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0};
        cl::Context context(CL_DEVICE_TYPE_CPU|CL_DEVICE_TYPE_GPU, cprops);

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

        std::vector<cl::Device> SuitableDevices;

        std::string ProgramScopeFeature("__opencl_c_program_scope_global_variables");
        for (cl::Device &Dev : devices) {
          std::string DevVer = Dev.getInfo<CL_DEVICE_VERSION>();
          if (DevVer.find("OpenCL 3.0") == 0) {
            std::vector<cl_name_version> Features = Dev.getInfo<CL_DEVICE_OPENCL_C_FEATURES>();
            for (auto &Item : Features) {
              if (ProgramScopeFeature == Item.name) {
                SuitableDevices.push_back(Dev);
              }
            }
          }
        }

        if (SuitableDevices.empty()) {
          std::cout << "No devices with OpenCL 3.0 + "
                       "program scope variables found.\n";
          return 77;
        }
        cl::CommandQueue queue(context, SuitableDevices[0], 0);

        cl::Program::Sources sources1({programOneSourceCode});
        cl::Program program1(context, sources1);
        program1.build(devices, "-cl-std=CL3.0 " BUILD_OPTION(BUFFER_SIZE_1));

        cl::Kernel test1_1(program1, "test1");
        cl::Kernel test2_1(program1, "test2");
        cl::Kernel test3_1(program1, "test3");
        size_t numKernels1 = program1.getInfo<CL_PROGRAM_NUM_KERNELS>();
        if (numKernels1 != 3)
          throw std::runtime_error("program1 kernel count incorrect");

        cl::Program::Sources sources2({programTwoSourceCode});
        cl::Program program2(context, sources2);
        program2.build(devices, "-cl-std=CL3.0 " BUILD_OPTION(BUFFER_SIZE_2));

        cl::Kernel test1_2(program2, "test1");
        cl::Kernel test2_2(program2, "test2");
        cl::Kernel test3_2(program2, "test3");
        size_t numKernels2 = program2.getInfo<CL_PROGRAM_NUM_KERNELS>();
        if (numKernels2 != 3)
          throw std::runtime_error("program2 kernel count incorrect");

        for (int i = 0; i < BUFFER_SIZE_1; ++i)
            OneA[i] = UniDist(Mersenne);
        for (int i = 0; i < BUFFER_SIZE_2; ++i)
            TwoA[i] = UniDist(Mersenne);

        cl::Buffer aBuffer1 = cl::Buffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            BUFFER_SIZE_1 * sizeof(float),
            (void *) &OneA[0]);

        cl::Buffer outBuffer1 = cl::Buffer(
            context, CL_MEM_READ_WRITE,
            BUFFER_SIZE_1 * sizeof(float), nullptr);

        test1_1.setArg(0, aBuffer1);
        test2_1.setArg(0, aBuffer1);
        test3_1.setArg(0, outBuffer1);

        cl::Buffer aBuffer2 = cl::Buffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            BUFFER_SIZE_2 * sizeof(float),
            (void *) &TwoA[0]);

        cl::Buffer outBuffer2 = cl::Buffer(
            context, CL_MEM_READ_WRITE,
            BUFFER_SIZE_2 * sizeof(float), nullptr);

        test1_2.setArg(0, aBuffer2);
        test2_2.setArg(0, aBuffer2);
        test3_2.setArg(0, outBuffer2);

        queue.enqueueNDRangeKernel(
            test1_1,
            cl::NullRange,
            cl::NDRange(BUFFER_SIZE_1),
            cl::NullRange);

        queue.enqueueNDRangeKernel(
            test2_1,
            cl::NullRange,
            cl::NDRange(BUFFER_SIZE_1),
            cl::NullRange);

        queue.enqueueNDRangeKernel(
            test1_2,
            cl::NullRange,
            cl::NDRange(BUFFER_SIZE_2),
            cl::NullRange);

        queue.enqueueNDRangeKernel(
            test2_2,
            cl::NullRange,
            cl::NDRange(BUFFER_SIZE_2),
            cl::NullRange);

        queue.finish();

        queue.enqueueNDRangeKernel(
            test3_1,
            cl::NullRange,
            cl::NDRange(BUFFER_SIZE_1),
            cl::NullRange);

        queue.enqueueReadBuffer(
            outBuffer1,
            CL_TRUE, // block
            0,
            BUFFER_SIZE_1 * sizeof(float),
            (void*) &OneB[0]);

        queue.enqueueNDRangeKernel(
            test3_2,
            cl::NullRange,
            cl::NDRange(BUFFER_SIZE_2),
            cl::NullRange);

        queue.enqueueReadBuffer(
            outBuffer2,
            CL_TRUE, // block
            0,
            BUFFER_SIZE_2 * sizeof(float),
            (void*) &TwoB[0]);

        queue.finish();

        for (int i = 0; i < BUFFER_SIZE_1; ++i) {
            float expected = OneA[i] * 19.0f + 2.0f * OneConstantVar[i % 8];
            if (std::abs(OneB[i] - expected) > 1e-3f) {
            std::cout << "ONE N " << i << " expected " << expected << " got "
                      << OneB[i] << "\n";
            ++errors;
            }
        }

        for (int i = 0; i < BUFFER_SIZE_2; ++i) {
            float expected = TwoA[i] * 26.0f + 2.0f * TwoConstantVar[i % 8];
            if (std::abs(TwoB[i] - expected) > 1e-3f) {
            std::cout << "TWO N " << i << " expected " << expected << " got "
                      << TwoB[i] << "\n";
            ++errors;
            }
        }

        cl::Program::Sources sources3({programThreeSourceCode});
        cl::Program program3(context, sources3);
        program3.build(devices, "-cl-std=CL3.0");

        size_t numKernels3 = program3.getInfo<CL_PROGRAM_NUM_KERNELS>();
        if (numKernels3 != 2)
          throw std::runtime_error("program3 kernel count incorrect");
        cl::Kernel test4w(program3, "test4w");
        cl::Kernel test5r(program3, "test5r");

        cl::Buffer outBuffer3 = cl::Buffer(
            context, CL_MEM_READ_WRITE, 128);

        cl_char c = 42;
        cl_int i = 0x3838292;
        cl_long l = static_cast<cl_long>(0x283849239423LL);
        cl_int matching_res = 0;

        test4w.setArg(0, c);
        test4w.setArg(1, i);
        test4w.setArg(2, l);

        test5r.setArg(0, c);
        test5r.setArg(1, i);
        test5r.setArg(2, l);
        test5r.setArg(3, outBuffer3);

        queue.enqueueNDRangeKernel(
            test4w,
            cl::NullRange,
            cl::NDRange(1),
            cl::NullRange);

        queue.enqueueNDRangeKernel(
            test5r,
            cl::NullRange,
            cl::NDRange(1),
            cl::NullRange);

        queue.enqueueReadBuffer(
            outBuffer3,
            CL_TRUE, // block
            0,
            sizeof(int),
            (void*) &matching_res);

        queue.finish();

        std::cout << "TEST_STRUCT matching res: " << matching_res << "\n";
        errors += (matching_res != 3 ? 1 : 0);

        cl::Program::Sources sources4({programFourSourceCode});
        cl::Program program4(context, sources4);
        program4.build(devices, "-cl-std=CL3.0");

        size_t numKernels4 = program4.getInfo<CL_PROGRAM_NUM_KERNELS>();
        if (numKernels4 != 2)
          throw std::runtime_error("program4 kernel count incorrect");
        cl::Kernel reader4(program4, "reader");
        cl::Kernel writer4(program4, "writer");

        for (int i = 0; i < BUFFER_SIZE_1; ++i) {
          OneA[i] = UniDist(Mersenne);
          OneB[i] = 0.0f;
        }

        cl::Buffer inBuffer4 = cl::Buffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            BUFFER_SIZE_1 * sizeof(float),
            (void *) &OneA[0]);

        cl::Buffer outBuffer4 = cl::Buffer(
            context, CL_MEM_READ_WRITE,
            BUFFER_SIZE_1 * sizeof(float), nullptr);

        int in1 = 1;
        writer4.setArg(0, inBuffer4);
        writer4.setArg(1, in1);

        cl_float2 out1;
        out1.s[0] = 0.5f;
        out1.s[1] = 2.0f;
        reader4.setArg(0, outBuffer4);
        reader4.setArg(1, out1);

        queue.enqueueNDRangeKernel(
            writer4,
            cl::NullRange,
            cl::NDRange(1),
            cl::NullRange);

        queue.enqueueNDRangeKernel(
            reader4,
            cl::NullRange,
            cl::NDRange(1),
            cl::NullRange);

        queue.enqueueReadBuffer(
            outBuffer4,
            CL_TRUE, // block
            0,
            BUFFER_SIZE_1 * sizeof(float),
            (void*) &OneB[0]);

        queue.finish();

        OneA[6] = out1.s[0];
        OneA[7] = out1.s[1];
        matching_res = 0;
        for (unsigned i = 0; i < 8; ++i) {
          if (OneA[i] == OneB[i]) {
            ++matching_res;
          } else {
            std::cout << "FAIL at " << i << " OneA: " << OneA[i]
                      << " OneB: " << OneB[i] << "\n";
          }
        }

        std::cout << "TEST_GVAR_PTR matching res: " << matching_res << "\n";
        errors += (matching_res != 8 ? 1 : 0);

    }
    catch (cl::Error &err) {
        std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")"
                  << std::endl;
        return EXIT_FAILURE;
    }

    platformList[0].unloadCompiler();

    if (errors) {
        std::cout << "FAILED, errors: " << errors << "\n";
        return EXIT_FAILURE;
    } else {
        std::cout << "PASSED" << std::endl;
        return EXIT_SUCCESS;
    }

    return EXIT_FAILURE;
}
