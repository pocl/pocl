/* accel_example.cpp - Example program for built-in kernels

   Copyright (c) 2022 Topi Leppänen / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include <CL/opencl.hpp>

#include <string>
#include <vector>
#include <deque>
#include <chrono>
#include <mutex>
#include <cmath>
#include <condition_variable>
#include <map>
#include <set>
#include <thread>
#include <unordered_map>
#include <random>
#include <iostream>

// SGEMM kernel requires 16
#define SGEMM_TILE 16
#define X 16
#define Y 16
#define BUFSIZE (X*Y*4)

#define CHECK_CL_ERROR(EXPR, ...) \
    if (EXPR != CL_SUCCESS) { std::cerr << __VA_ARGS__; return 12; }



int
main(int argc, char** argv)
{
    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Context ClContext;

    cl::Device AccelDev;
    cl::CommandQueue AccelQueue;
    cl::Program AccelProgram;
    cl::Kernel AccelKernel;

    cl::NDRange Offset, Global2D, Local2D, Global, Local;
    int err;

    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(!all_platforms.size()) {
        std::cerr << "No OpenCL platforms available!\n";
        return 1;
    }

    platform = all_platforms[0];
    platform.getDevices(CL_DEVICE_TYPE_CUSTOM, &devices);
    if(devices.size() == 0) {
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if(devices.size() == 0) {
            std::cerr << "No OpenCL devices available!\n";
            return 2;
        }
    }

    const char* kernel_name = "pocl.add.i32";
    const char* known_kernels[] = { "pocl.add.i32", "pocl.mul.i32",
                                    "pocl.abs.f32",
                                    "pocl.sgemm.local.f32",
                                    "pocl.countred",
                                    NULL,
                                  };
    if (argc > 1)
      {
        kernel_name = argv[1];
        std::string k{kernel_name};
        int found = 0;
        unsigned i = 0;
        while (known_kernels[i]) {
            if (k.compare(known_kernels[i]) == 0) found = 1;
            ++i;
          }
        if (!found) {
            std::cerr << "unknown builtin kernel: " << kernel_name << "\n";
            return 3;
          }
      }

    std::string kernel_str{kernel_name};

    for (auto& D : devices) {
      std::string AccelBuiltinKernels = D.getInfo<CL_DEVICE_BUILT_IN_KERNELS>();
      std::cout << "Device " << D.getInfo<CL_DEVICE_NAME>() <<
                   " has builtin kernels: " << AccelBuiltinKernels << "\n";
      if (AccelBuiltinKernels.find(kernel_str) != std::string::npos)
      {
         AccelDev = D;
         break;
      }
    }
    if (AccelDev.get() == nullptr) {
        std::cerr << "no devices which support builtin kernel " << kernel_str << "\n";
        return 4;
      }

    std::cout << "Using device: " << AccelDev.getInfo<CL_DEVICE_NAME>() << "\n";
    std::cout << "Using builtin kernel: " << kernel_str << "\n";

    std::vector<cl::Device> AccelDevs = {AccelDev};
    ClContext = cl::Context(AccelDevs, nullptr, nullptr, nullptr, &err);
    CHECK_CL_ERROR(err, "Context creation failed\n");

    AccelQueue = cl::CommandQueue(ClContext, AccelDev, 0, &err); // , CL_QUEUE_PROFILING_ENABLE
    CHECK_CL_ERROR(err, "CmdQueue creation failed\n");

    AccelProgram = cl::Program{ClContext, AccelDevs, kernel_str.c_str(), &err};
    CHECK_CL_ERROR(err, "Program creation failed\n");

    err = AccelProgram.build(AccelDevs);
    CHECK_CL_ERROR(err, "Program build failed\n");
    cl::Kernel Kernel = cl::Kernel(AccelProgram, kernel_str.c_str(), &err);
    CHECK_CL_ERROR(err, "Kernel creation failed\n");

    cl::Buffer Input1 = cl::Buffer(ClContext, CL_MEM_READ_WRITE, (cl::size_type)BUFSIZE, nullptr, &err);
    CHECK_CL_ERROR(err, "Input1 buffer creation failed\n");
    cl::Buffer Input2 = cl::Buffer(ClContext, CL_MEM_READ_WRITE, (cl::size_type)BUFSIZE, nullptr, &err);
    CHECK_CL_ERROR(err, "Input2 buffer creation failed\n");
    cl::Buffer Out1 = cl::Buffer(ClContext, CL_MEM_READ_WRITE, (cl::size_type)BUFSIZE, nullptr, &err);
    CHECK_CL_ERROR(err, "OUtput1 buffer creation failed\n");

    void *i1, *i2, *o1;
    Offset = cl::NullRange;
    Local = cl::NullRange;
    if (kernel_str.compare("pocl.sgemm.local.f32") == 0)
        Local2D = cl::NDRange(SGEMM_TILE, SGEMM_TILE);
    else
        Local2D = cl::NullRange;
    Global2D = cl::NDRange(X, Y);
    Global = cl::NDRange(X * Y);

    if (kernel_str.compare("pocl.add.i32") == 0 ||
        kernel_str.compare("pocl.mul.i32") == 0)
    {
        Kernel.setArg(0, Input1);
        Kernel.setArg(1, Input2);
        Kernel.setArg(2, Out1);
    }

    if (kernel_str.compare("pocl.abs.f32") == 0 ||
        kernel_str.compare("pocl.countred") == 0)
    {
        Kernel.setArg(0, Input1);
        Kernel.setArg(1, Out1);
    }

    if (kernel_str.compare("pocl.sgemm.local.f32") == 0)
    {
        Kernel.setArg(0, Input1);
        Kernel.setArg(1, Input2);
        Kernel.setArg(2, Out1);
        Kernel.setArg(3, X);
        Kernel.setArg(4, Y);
        Kernel.setArg(5, X);
    }

    if (kernel_str.compare("pocl.sgemm.local.f32") == 0 ||
        kernel_str.compare("pocl.abs.f32") == 0)
    {
        std::mt19937 mt{234545649UL};
        std::uniform_real_distribution<float> dist(15.0, 25.0);

        float* in1 = new float[X*Y];
        float* in2 = new float[X*Y];
        float* out1 = new float[X*Y];

        for (size_t i = 0; i < X*Y; ++i) {
            in1[i] = dist(mt);
            in2[i] = dist(mt);
            out1[i] = 0.0f;
          }
        i1 = in1; i2 = in2; o1 = out1;
    }
    else
    {
        std::mt19937 mt{234545649UL};
        std::uniform_int_distribution<unsigned int> dist(10, 124);

        uint32_t *in1 = new uint32_t[X*Y];
        uint32_t *in2 = new uint32_t[X*Y];
        uint32_t *out1 = new uint32_t[X*Y];

        for (size_t i = 0; i < X*Y; ++i) {
            in1[i] = dist(mt);
            in2[i] = dist(mt);
            out1[i] = 0;
          }
        i1 = in1; i2 = in2; o1 = out1;
      }

    using clock_type = std::chrono::steady_clock;
    using second_type = std::chrono::duration<double, std::ratio<1> >;
    std::chrono::time_point<clock_type> m_beg { clock_type::now() };

    err = AccelQueue.enqueueWriteBuffer(Input1, CL_FALSE, 0, BUFSIZE, i1);
    CHECK_CL_ERROR(err, "en 1");
    err = AccelQueue.enqueueWriteBuffer(Input2, CL_FALSE, 0, BUFSIZE, i2);
    CHECK_CL_ERROR(err, "en 2");

    if (kernel_str.compare("pocl.sgemm.local.f32") == 0 ||
        kernel_str.compare("pocl.countred") == 0)
      err = AccelQueue.enqueueNDRangeKernel(Kernel, Offset, Global2D, Local2D);
    else
      err = AccelQueue.enqueueNDRangeKernel(Kernel, Offset, Global, Local);
    CHECK_CL_ERROR(err, "en 3");
    err = AccelQueue.enqueueReadBuffer(Out1, CL_TRUE, 0, BUFSIZE, o1);
    CHECK_CL_ERROR(err, "en 4");

    std::chrono::time_point<clock_type> m_end { clock_type::now() };
    double diff = std::chrono::duration_cast<second_type>(m_end - m_beg).count();
    std::cout << "Execution time(s): " << diff << "\n\n";

    bool Failed = false;
    bool ResultChecked = false;

    if (kernel_str.compare("pocl.sgemm.local.f32") == 0 ||
        kernel_str.compare("pocl.abs.f32") == 0)
    {
        float* in1 = (float *)i1;
        float* in2 = (float *)i2;
        float* out1 = (float *)o1;
        for (size_t i = 0; i < 10; ++i)
        {
           std::cout << "IN1: " << in1[i] << "  IN2: " << in2[i] <<
                     "  OUT1: " << out1[i] << "\n";
        }

        if (kernel_str.compare("pocl.abs.f32") == 0) {
          ResultChecked = true;
          for (size_t i = 0; i < X*Y; ++i) {
            if (out1[i] != fabsf(in1[i])) { Failed = true ; break; }
          }
        }

        delete [] in1;
        delete [] in2;
        delete [] out1;
    }
    else
    {
        uint32_t *in1 = (uint32_t *)i1;
        uint32_t *in2 = (uint32_t *)i2;
        uint32_t *out1 = (uint32_t *)o1;

        for (size_t i = 0; i < 10; ++i)
        {
           std::cout << "IN1: " << in1[i] << "  IN2: " << in2[i] <<
                     "  OUT1: " << out1[i] << "\n";
        }

        if (kernel_str.compare("pocl.add.i32") == 0) {
          ResultChecked = true;
          for (size_t i = 0; i < X*Y; ++i) {
            if (out1[i] != in1[i] + in2[i]) { Failed = true ; break; }
          }
        }

        if (kernel_str.compare("pocl.mul.i32") == 0) {
          ResultChecked = true;
          for (size_t i = 0; i < X*Y; ++i) {
            if (out1[i] != in1[i] * in2[i]) { Failed = true ; break; }
          }
        }

        if (kernel_str.compare("pocl.countred") == 0) {
          ResultChecked = true;
          unsigned count = 0;
          for (size_t i = 0; i < X*Y; ++i) {
            if (in1[i] > 100) count++;
          }
          if (out1[0] != count)
            Failed = true;
        }

        delete [] in1;
        delete [] in2;
        delete [] out1;
      }

    if (ResultChecked) {
        if (Failed)
          std::cout << "TEST FAILED\n";
        else
          std::cout << "OK; TEST PASSED\n";
      }

    return EXIT_SUCCESS;
}
