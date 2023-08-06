/* OpenCLcontext.cpp - Dual-device Carla example

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


#include <CL/cl2.hpp>

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

#include "OpenCLcontext.h"

#ifdef LIBCARLA_INCLUDED_FROM_UE4
#include "carla/Debug.h"
#include "carla/Logging.h"
#endif

static const char *PX_COUNT_SOURCE = R"(

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

__kernel void count_red_pixels(global const uchar4 *input, global ulong* output) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t width = get_global_size(0);
    size_t height = get_global_size(1);

    uchar4 px = input[y * width + x];
    if (px.x > 100)
        atom_add(output, 1);
}
)";

static const char *DOWNSAMPLE_SOURCE = R"(

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

__kernel void downsample_image(global const uchar4 *input, global uchar4* output) {
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);
    size_t width = get_global_size(0);
    size_t height = get_global_size(1);

/*
    uint4 sum = input[y * width + x]
              + input[y * width + x + 1]
              + input[(y+1) * width + x]
              + input[(y+1) * width + x + 1];
    output[y * (width/2) + x] = convert_uchar4(sum / 4);
*/
   unsigned out_idx = y + x*width;

   const unsigned factor = 2;
   width *= factor;
   height *= factor;
   unsigned y1, y2, x1, x2;
   y1=y*factor;
   y2=y1+1;
   x1=x*factor;
   x2=x1+1;

   uchar4 q11, q12, q21, q22;
   q11 = input[y1 * width + x1];
   q12 = input[y1 * width + x2];
   q21 = input[y2 * width + x1];
   q22 = input[y2 * width + x2];
   uint4 sum = convert_uint4(q11) + convert_uint4(q12) + convert_uint4(q21) + convert_uint4(q22);
   output[out_idx] = convert_uchar4(sum/4);
}
)";


#define OUTPUT_SIZE (cl::size_type)(sizeof(unsigned long)*64)

#ifdef LIBCARLA_INCLUDED_FROM_UE4
#define LOG_I(...) carla::log_info(__VA_ARGS__)
#define LOG_D(...) carla::log_debug(__VA_ARGS__)
#define LOG_E(...) carla::log_error(__VA_ARGS__)
#else
#define LOG_I(...)
#define LOG_D(...)
#define LOG_E(...)
#endif


#define CHECK_CL_ERROR(EXPR, ...) \
    if (EXPR != CL_SUCCESS) { LOG_E(__VA_ARGS__); return false; }

class OpenCL_Context {
    cl::Platform platform;
    std::vector<cl::Device> devices;
    cl::Context ClContext;

    cl::Device GpuDev;
    cl::CommandQueue GpuQueue;
    cl::Program GpuProgram;
    cl::Kernel GpuKernel;

    cl::Device FpgaDev;
    cl::CommandQueue FpgaQueue;
    cl::Program FpgaProgram;
    cl::Kernel FpgaKernel;

    cl::Buffer GpuInputBuffer, InterBuffer, FpgaOutputBuffer;
    unsigned long InputBufferSize;

    std::mutex OclMutex;
    cl::NDRange Offset, Global, Local;
    bool initialized = false;
    bool UsingFpgaBuiltinKernel = false;
    unsigned imgID = 0;

public:
    OpenCL_Context() {};

    bool isAvailable() {
      std::unique_lock<std::mutex> lock(OclMutex);
      if (!initialized)
          return false;
      if (GpuDev() == nullptr || FpgaDev() == nullptr)
          return false;
//      bool avail = GpuDev.getInfo<CL_DEVICE_AVAILABLE>() != CL_FALSE;
//      return avail;
      return true;
    }

    ~OpenCL_Context() {
        if (isAvailable())
            shutdown();
    }

    bool initialize(unsigned width, unsigned height, unsigned bpp);
    bool processCameraFrame(unsigned char* input, unsigned long *output);

private:
    void shutdown();
};

bool OpenCL_Manager::initialize(unsigned width, unsigned height, unsigned bpp) {
    if (Context->initialize(width, height, bpp))
        isValid = true;
    else {
        LOG_E("Error in OpenCL->initialize\n");
        isValid = false;
    }
    return isValid;
}

bool OpenCL_Manager::processCameraFrame(unsigned char* input, unsigned long *output) {

    if (isValid) {
        if (!Context->processCameraFrame(input, output)) {
            LOG_E("Error in OpenCL->processCameraFrame\n");
            isValid = false;
        }
    }
    if (!isValid)
        *output = 11223344;
    return isValid;
}

OpenCL_Manager::OpenCL_Manager()
    : Context{std::unique_ptr<OpenCL_Context>(new OpenCL_Context())} {}
OpenCL_Manager::~OpenCL_Manager() {}



bool OpenCL_Context::initialize(unsigned width, unsigned height, unsigned bpp)
{
    cl_int err;
    InputBufferSize = width * height * bpp / 8;
    // Take first platform and create a context for it.
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if(!all_platforms.size()) {
        LOG_E("No OpenCL platforms available!\n");
        return false;
    }

    platform = all_platforms[0];
    Offset = cl::NullRange;

    unsigned local_h = (height % 2) ? 1 : 2;
    local_h = (height % 4) ? local_h : 4;
    local_h = (height % 8) ? local_h : 8;
    local_h = (height % 16) ? local_h : 16;
    unsigned local_w = (width % 2) ? 1 : 2;
    local_w = (width % 4) ? local_w : 4;
    local_w = (width % 8) ? local_w : 8;
    local_w = (width % 16) ? local_w : 16;

    Local = cl::NDRange(local_w/2, local_h/2);
    Global = cl::NDRange(width/2, height/2);

    // Find all devices.
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if(devices.size() == 0) {
        LOG_E("No OpenCL devices available!\n");
        return false;
    }

    LOG_I("OpenCL platform name: %s\n",  platform.getInfo<CL_PLATFORM_NAME>().c_str());
    LOG_I("OpenCL platform version: %s\n", platform.getInfo<CL_PLATFORM_VERSION>().c_str());
    LOG_I("OpenCL platform vendor: %s\n",  platform.getInfo<CL_PLATFORM_VENDOR>().c_str());
    LOG_I("Found %lu OpenCL devices.\n", devices.size());

    if (devices.size() == 1) {
        GpuDev = devices[0];
        FpgaDev = devices[0];
    } else if (devices[1].getInfo<CL_DEVICE_TYPE>() & CL_DEVICE_TYPE_CUSTOM) {
        GpuDev = devices[0];
        FpgaDev = devices[1];
    } else {
        GpuDev = devices[1];
        FpgaDev = devices[0];
    }

    ClContext = cl::Context(devices, nullptr, nullptr, nullptr, &err);
    CHECK_CL_ERROR(err, "Context creation failed\n");

    GpuQueue = cl::CommandQueue(ClContext, GpuDev, 0, &err); // , CL_QUEUE_PROFILING_ENABLE
    CHECK_CL_ERROR(err, "CmdQueue creation failed\n");
    if (devices.size() == 1)
        FpgaQueue = GpuQueue;
    else {
        FpgaQueue = cl::CommandQueue(ClContext, FpgaDev, 0, &err); // , CL_QUEUE_PROFILING_ENABLE
        CHECK_CL_ERROR(err, "CmdQueue creation failed\n");
    }

    std::vector<cl::Device> GpuDevs = {GpuDev};
    GpuProgram = cl::Program{ClContext, DOWNSAMPLE_SOURCE, false, &err};
    CHECK_CL_ERROR(err, "Program creation failed\n");
    err = GpuProgram.build(GpuDevs);
    CHECK_CL_ERROR(err, "Program build failed\n");
    GpuKernel = cl::Kernel(GpuProgram, "downsample_image", &err);
    CHECK_CL_ERROR(err, "Kernel creation failed\n");

    std::string FpgaBuiltinKernels = FpgaDev.getInfo<CL_DEVICE_BUILT_IN_KERNELS>();
    std::string RedPixelKernelName{"pocl.countred"};
    std::vector<cl::Device> FpgaDevs = {FpgaDev};
    if (FpgaBuiltinKernels.find(RedPixelKernelName) != std::string::npos)
      {
        UsingFpgaBuiltinKernel = true;
        FpgaProgram = cl::Program{ClContext, FpgaDevs, RedPixelKernelName, &err};
        CHECK_CL_ERROR(err, "Program creation failed\n");
      }
    else
      {
        FpgaProgram = cl::Program{ClContext, PX_COUNT_SOURCE, false, &err};
        RedPixelKernelName = "count_red_pixels";
        CHECK_CL_ERROR(err, "Program creation failed\n");
      }

    err = FpgaProgram.build(FpgaDevs);
    CHECK_CL_ERROR(err, "Program build failed\n");
    FpgaKernel = cl::Kernel(FpgaProgram, RedPixelKernelName.c_str(), &err);
    CHECK_CL_ERROR(err, "Kernel creation failed\n");

    GpuInputBuffer = cl::Buffer(ClContext, CL_MEM_READ_WRITE, (cl::size_type)(InputBufferSize), nullptr, &err);
    CHECK_CL_ERROR(err, "Input buffer creation failed\n");
    InterBuffer = cl::Buffer(ClContext, CL_MEM_READ_WRITE, (cl::size_type)(InputBufferSize/4), nullptr, &err);
    CHECK_CL_ERROR(err, "Inter buffer creation failed\n");
    FpgaOutputBuffer = cl::Buffer(ClContext, CL_MEM_READ_WRITE, OUTPUT_SIZE, nullptr, &err);
    CHECK_CL_ERROR(err, "Output buffer creation failed\n");

    GpuKernel.setArg(0, GpuInputBuffer);
    GpuKernel.setArg(1, InterBuffer);

    FpgaKernel.setArg(0, InterBuffer);
    FpgaKernel.setArg(1, FpgaOutputBuffer);

    initialized = true;
    return true;
}

bool OpenCL_Context::processCameraFrame(unsigned char* input, unsigned long *output) {
    if (!isAvailable()) {
        LOG_E("Device not available");
        return false;
    }

    std::unique_lock<std::mutex> lock(OclMutex);

#ifdef TIMING
    LOG_D("OpenCL: start processCameraFrame\n");
    auto start_time = std::chrono::steady_clock::now();
#endif

    cl_int err;
    *output = 0;

    std::vector<cl::Event> evts;
    cl::Event ev1, ev2, ev3, ev4;

    err = GpuQueue.enqueueWriteBuffer(GpuInputBuffer, CL_FALSE, 0, InputBufferSize, input, nullptr, &ev1);
    if (err != CL_SUCCESS)
        return false;

    evts.push_back(ev1);
    err = GpuQueue.enqueueNDRangeKernel(GpuKernel, Offset, Global, Local, &evts, &ev2);
    if (err != CL_SUCCESS)
        return false;

    evts.clear();
    evts.push_back(ev2);

    if (!UsingFpgaBuiltinKernel) {
    /* clear the output buffer. only required for some devices */
    err = FpgaQueue.enqueueWriteBuffer(FpgaOutputBuffer, CL_FALSE, 0, sizeof(cl_ulong), output, nullptr, &ev3);
    if (err != CL_SUCCESS)
        return false;
    evts.push_back(ev3);
    }

    err = FpgaQueue.enqueueNDRangeKernel(FpgaKernel, Offset, Global, Local, &evts, &ev4);
    if (err != CL_SUCCESS)
        return false;

    evts.clear();
    evts.push_back(ev4);
    err = FpgaQueue.enqueueReadBuffer(FpgaOutputBuffer, CL_TRUE, 0, sizeof(unsigned long), output, &evts);
    if (err != CL_SUCCESS)
        return false;

#ifdef DUMP_FRAMES
    char filename[1024];
    std::snprintf(filename, 1024, "/tmp/carla_%u_%zu.raw", imgID, *output);
    FILE* outfile = std::fopen(filename, "w");
    std::fwrite(input, 1, InputBufferSize, outfile);
    std::fclose(outfile);
    ++imgID;
#endif

#ifdef TIMING
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<float> diff = end_time - start_time;
    float s = diff.count() * 1000.0f;
    LOG_D("OpenCL: end processCameraFrame: %03.1f ms\n", s);
#endif
    return true;
}


void OpenCL_Context::shutdown() {
    if (GpuQueue())
            GpuQueue.finish();
}
