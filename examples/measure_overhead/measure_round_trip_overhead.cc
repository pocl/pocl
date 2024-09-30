/* Benchmark for measuring the base overhead of ndrangekernel commands

   Copyright (c) 2019 pocl developers

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

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 110

#ifdef CL_HPP_TARGET_OPENCL_VERSION
#undef CL_HPP_TARGET_OPENCL_VERSION
#endif
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include <CL/opencl.hpp>

#include "common.hh"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>

struct {
  int platform_index = -1;
  int device_index = -1;
  int sample_count = 1000;
  int warmup = 10;
} options;

void print_help(const char *name) {
  std::cerr << "Usage: " << name << " [-p platform_index] [-d device_index] "
            << "[-s sample_count]" << std::endl
            << "-p specifies which platform to use. (default:"
            << options.platform_index << ")" << std::endl
            << "-d specifies which device to use. (default:"
            << options.device_index << ")" << std::endl
            << "-s sets the number of samples measured. (default: "
            << options.sample_count << ")" << std::endl
            << "-w sets the number of warmup rounds to do. (default: "
            << options.warmup << ")" << std::endl;
}

bool parse_args(char **argv) {
  const char *name = *argv++;
  while (*argv) {
    const char *arg = *argv;
    if (arg[0] == '-') {
      if (arg[1] == '-') {
        if (!strcmp(arg + 2, "help"))
          goto fail;
        else {
          std::cerr << "Unknown long flag " << arg + 2 << std::endl;
          goto fail;
        }
      } else if (arg[1] == 'p' && arg[2] == 0) {
        argv++;
        if (!*argv) {
          std::cerr << "Missing platform index" << std::endl;
          goto fail;
        }
        options.platform_index = std::stoi(*argv, nullptr);
      } else if (arg[1] == 'd' && arg[2] == 0) {
        argv++;
        if (!*argv) {
          std::cerr << "Missing device index" << std::endl;
          goto fail;
        }
        options.device_index = std::stoi(*argv);
      } else if (arg[1] == 's' && arg[2] == 0) {
        argv++;
        if (!*argv) {
          std::cerr << "Missing sample count" << std::endl;
          goto fail;
        }
        options.sample_count = std::stoi(*argv);
      } else {
        std::cerr << "Unknown flag " << arg + 1 << std::endl;
        goto fail;
      }
    }
    argv++;
  }
  if (options.device_index >= 0 && options.platform_index < 0)
    options.platform_index = 0;
  return true;
fail:
  print_help(name);
  return false;
}

bool measure_round_trip_latency(cl::CommandQueue &cq, cl::Kernel &k) {
  if (options.sample_count <= 0)
    return true;

  using namespace std::chrono;

  struct measurement {
    cl::Event e;
    steady_clock::duration local_duration;
  };
  std::vector<measurement> runs(options.sample_count);

  // Collect duration data.
  for (int i = 0; i < options.sample_count; ++i) {
    auto start = steady_clock::now();

    cl_int err = cq.enqueueNDRangeKernel(k, cl::NullRange, cl::NDRange(1),
                                         cl::NullRange, nullptr, &runs[i].e);
    cq.finish();

    auto end = steady_clock::now();

    if (err) {
      std::cerr << "\t\tKernel enqueue failed!" << std::endl;
      return false;
    }
    runs[i].local_duration = end - start;
  }

  // Print local measurements
  std::vector<double> times(options.sample_count);
  for (int i = 0; i < options.sample_count; ++i)
    times[i] =
        duration_cast<duration<double, std::micro>>(runs[i].local_duration)
            .count();
  print_measurements("host-measured timing:", times, 2);

  // Print measurements from event profiling
  for (int i = 0; i < options.sample_count; ++i) {
    size_t ticks = runs[i].e.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                   runs[i].e.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    times[i] = double(ticks) / 1e3;
  }
  print_measurements("device-measured timing:", times, 2);

  return true;
}

#define TESTMAGIC 31337
bool measure_device(cl::Device &device, int index) {
  try {
    std::cout << "\tDevice " << index << ":" << std::endl
              << "\t\tname: " << device.getInfo<CL_DEVICE_NAME>() << std::endl
              << "\t\tversion: " << device.getInfo<CL_DEVICE_VERSION>()
              << std::endl;
    cl::Context ctx(device);
    cl::CommandQueue cq(ctx, device,
                        cl::QueueProperties::Profiling |
                            cl::QueueProperties::OutOfOrder);

    int zero = 0;
    cl::Buffer output(ctx, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof(int), &zero);

    cl::Program prog(ctx, "__kernel void empty_kernel(int x, __global int "
                          "*arr) { arr[get_global_id(0)] = x; }");
    try {
      prog.build();
    } catch (cl::Error &err) {
      std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
      std::cerr << "\t\tFailed to build kernel: " << log << std::endl;
      return false;
    }
    cl::Kernel kern(prog, "empty_kernel");
    kern.setArg(0, TESTMAGIC);
    kern.setArg(1, output);
    for (int i = 0; i < options.warmup; ++i) {
      cl_int err = cq.enqueueNDRangeKernel(kern, cl::NullRange, cl::NDRange(1),
                                           cl::NullRange, nullptr, nullptr);
      cq.finish();
    }

    measure_round_trip_latency(cq, kern);
    int result = 0;
    cq.enqueueReadBuffer(output, true, 0, sizeof(int), &result);
    if (result != TESTMAGIC) {
      std::cerr << "buffer contained " << result << "instead of " << TESTMAGIC
                << ". you had one job" << std::endl;
      return false;
    }

  } catch (cl::Error &err) {
    std::cerr << err.what() << std::endl;
    return false;
  }
  return true;
}

bool measure_platform(cl::Platform &platform, int index) {
  try {
    std::cout << "Platform " << index << ":" << std::endl
              << "\tname: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl
              << "\tversion: " << platform.getInfo<CL_PLATFORM_VERSION>()
              << std::endl
              << "\tvendor: " << platform.getInfo<CL_PLATFORM_VENDOR>()
              << std::endl;

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    if (options.device_index < 0) {
      bool ret = true;
      for (size_t i = 0; i < devices.size(); ++i)
        ret = measure_device(devices[i], i) && ret;
      return ret;
    } else if ((size_t)options.device_index < devices.size()) {
      return measure_device(devices[options.device_index],
                            options.device_index);
    } else {
      std::cerr << "\t" << devices.size() << " devices found, index "
                << options.device_index << " is out of range." << std::endl;
      return false;
    }
  } catch (cl::Error &err) {
    std::cerr << err.what() << std::endl;
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  (void)argc;
  if (!parse_args(argv))
    return 1;

  std::vector<cl::Platform> platforms;
  if (cl::Platform::get(&platforms) != CL_SUCCESS) {
    std::cerr << "Failed to enumerate OpenCL platforms!" << std::endl;
    return 1;
  }

  if (platforms.size() == 0) {
    std::cerr << "No OpenCL platforms found!" << std::endl;
    return 1;
  }

  if (options.platform_index < 0) {
    bool failed = true;
    for (size_t i = 0; i < platforms.size(); ++i)
      failed = measure_platform(platforms[i], i) && failed;
    if (failed)
      return 1;
  } else if ((size_t)options.platform_index < platforms.size()) {
    if (!measure_platform(platforms[options.platform_index],
                          options.platform_index))
      return 1;
  } else {
    std::cerr << platforms.size() << " platforms found, index "
              << options.platform_index << " is out of range." << std::endl;
    return 1;
  }
  std::cout << "All good" << std::endl;
  return 0;
}
