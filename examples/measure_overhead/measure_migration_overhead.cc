/* Benchmark for measuring overhead of migrating a buffer between devices

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
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#ifdef CL_HPP_TARGET_OPENCL_VERSION
#undef CL_HPP_TARGET_OPENCL_VERSION
#endif
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include "common.hh"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

enum pattern_mode { MODE_ROTATE = 0, MODE_SHUFFLE };

struct {
  int platform_index = -1;
  pattern_mode mode = MODE_ROTATE;
  int sample_count = 1000;
  int warmup_rounds = 10;
  // This is the number of integers in the buffer, not bytes.
  size_t buffer_size = 256;
} options;

void print_help(const char *name) {
  std::cerr << "Usage: " << name << " [-p platform_index] [-m mode] "
            << "[-s sample_count] [-b buffer_size]" << std::endl
            << "-p specifies which platform to use. (default:"
            << options.platform_index << ")" << std::endl
            << "-m specifies which mode is used. Available modes:" << std::endl
            << "\trotate (default, rotates all buffers to the next device)"
            << std::endl
            << "\tshuffle (reassigns buffers randomly)" << std::endl
            << "-s sets the number of samples measured. (default:"
            << options.sample_count << ")" << std::endl
            << "-b sets the size of the test buffer. "
            << "This number is rounded up to the next multiple of "
            << sizeof(uint32_t) << ". (default: " << options.buffer_size << ")"
            << std::endl;
}

void print_progress(
    int p, int total,
    std::chrono::time_point<std::chrono::steady_clock> starttime,
    int width = 80) {
  std::string total_str = std::to_string(total);
  std::string p_str = std::to_string(p);
  auto elapsed = std::chrono::steady_clock::now() - starttime;
  auto remtime =
      (elapsed) * (double(total) / std::max(double(p), 1e-9)) - elapsed;
  std::chrono::hours hours =
      std::chrono::duration_cast<std::chrono::hours>(remtime);
  std::chrono::minutes mins =
      std::chrono::duration_cast<std::chrono::minutes>(remtime - hours);
  std::chrono::seconds secs =
      std::chrono::duration_cast<std::chrono::seconds>(remtime - hours - mins);
  std::stringstream remss;
  remss << std::setw(3) << hours.count() << ":" << std::setw(2)
        << std::setfill('0') << mins.count() << ":" << std::setw(2)
        << std::setfill('0') << secs.count();
  int bar_width = width - total_str.size() * 2 - 4 - 10;

  std::cerr << '\r';
  if (bar_width > 1) {
    int fill_width = bar_width * p / total;
    std::cerr << "[";
    for (int i = 0; i < bar_width; ++i) {
      char c = ' ';
      if (i < fill_width || p == total)
        c = '=';
      else if (i == fill_width)
        c = '>';

      std::cerr << c;
    }
    std::cerr << "] ";
  }
  std::cerr << std::setfill(' ') << std::setw(total_str.size()) << p_str << "/"
            << total_str << " " << remss.str() << std::flush;
  if (p == total)
    std::cerr << std::endl;
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
      } else if (arg[1] == 'm' && arg[2] == 0) {
        argv++;
        if (!*argv) {
          std::cerr << "Missing mode" << std::endl;
          goto fail;
        }

        if (strcmp(*argv, "rotate") == 0)
          options.mode = MODE_ROTATE;
        else if (strcmp(*argv, "shuffle") == 0)
          options.mode = MODE_SHUFFLE;
        else {
          std::cerr << "Unknown mode \"" << *argv << "\"" << std::endl;
          goto fail;
        }
      } else if (arg[1] == 's' && arg[2] == 0) {
        argv++;
        if (!*argv) {
          std::cerr << "Missing sample count" << std::endl;
          goto fail;
        }
        options.sample_count = std::stoi(*argv);
      } else if (arg[1] == 'b' && arg[2] == 0) {
        argv++;
        if (!*argv) {
          std::cerr << "Missing buffer size" << std::endl;
          goto fail;
        }
        options.buffer_size =
            (std::stoull(*argv) + sizeof(int) - 1) / sizeof(int);
      } else {
        std::cerr << "Unknown flag " << arg + 1 << std::endl;
        goto fail;
      }
    }
    argv++;
  }
  return true;
fail:
  print_help(name);
  return false;
}

struct testing_buffer {
  cl::Buffer buffer;
  cl::Event last_write_event;
};

void run_iteration(cl::Kernel &kern,
                   std::vector<cl::CommandQueue> &command_queues,
                   testing_buffer &buf,
                   std::vector<std::vector<double>> &timings_by_device) {
  std::vector<size_t> cq_idx;
  for (size_t i = 0; i < command_queues.size(); ++i)
    cq_idx.push_back(i);

  // Update buffer order
  static std::mt19937 rng;
  switch (options.mode) {
  case MODE_ROTATE:
    // One iteration rotates through all devices in sequence already
    break;
  case MODE_SHUFFLE:
    std::shuffle(cq_idx.begin(), cq_idx.end(), rng);
    break;
  }

  for (size_t i : cq_idx) {
    cl::CommandQueue &cq = command_queues[i];
    std::vector<double> &timings = timings_by_device[i];

    if (buf.last_write_event.get() != nullptr)
      buf.last_write_event.wait();
    cq.finish(); // just in case

    auto start = std::chrono::steady_clock::now();
    cq.enqueueMigrateMemObjects({buf.buffer}, 0);
    cq.finish();
    auto end = std::chrono::steady_clock::now();
    timings.push_back(
        std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(
            end - start)
            .count());

    // uint32_t pre_kernel;
    // cq.enqueueReadBuffer(buf.buffer, true, 0, sizeof(uint32_t), &pre_kernel);

    kern.setArg(0, buf.buffer);
    cq.enqueueNDRangeKernel(kern, cl::NullRange, cl::NDRange(1), cl::NullRange,
                            nullptr, &buf.last_write_event);
    cq.finish();
    // std::cout << i << ": " << timings.back() << std::endl;

    // uint32_t post_kernel;
    // std::vector<cl::Event> kern_event = {buf.last_write_event};
    // cq.enqueueReadBuffer(buf.buffer, true, 0, sizeof(uint32_t), &post_kernel,
    //                      &kern_event);
    // std::cout << "PRE " << pre_kernel << " -> POST " << post_kernel
    //           << std::endl;
  }
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
    if (devices.size() <= 1) {
      std::cout << "\tOnly one device detected, cannot test buffer "
                   "migration with this setup."
                << std::endl;
      return false;
    }
    cl::Context ctx(devices);

    cl::Program prog(ctx, "__kernel void acc_kernel(__global uint* d)"
                          "{ d[get_global_id(0)]++; }");
    if (prog.build(devices) != CL_SUCCESS) {
      for (cl::Device &d : devices) {
        cl_build_status status = prog.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(d);
        if (status != CL_BUILD_ERROR)
          continue;

        std::string dev_name = d.getInfo<CL_DEVICE_NAME>();
        std::string log = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(d);
        std::cerr << "Build error on " << dev_name << ": " << log << std::endl;
      }
      return false;
    }
    cl::Kernel kern(prog, "acc_kernel");

    std::vector<cl::CommandQueue> command_queues(devices.size());
    testing_buffer buffer = {cl::Buffer(ctx, CL_MEM_READ_WRITE,
                                        options.buffer_size * sizeof(uint32_t),
                                        nullptr)};
    std::vector<std::vector<double>> timings_by_device(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
      command_queues[i] = cl::CommandQueue(ctx, devices[i],
                                           cl::QueueProperties::Profiling |
                                               cl::QueueProperties::OutOfOrder);
      timings_by_device[i].reserve(options.sample_count);
    }
    command_queues[command_queues.size() - 1].enqueueMigrateMemObjects(
        {buffer.buffer}, 0);
    uint32_t zero = 0;
    int err = command_queues[command_queues.size() - 1].enqueueFillBuffer(
        buffer.buffer, zero, 0, sizeof(zero));

    // warm up
    auto starttime = std::chrono::steady_clock::now();
    print_progress(0, options.sample_count + options.warmup_rounds, starttime);
    for (int i = 0; i < options.warmup_rounds; ++i) {
      run_iteration(kern, command_queues, buffer, timings_by_device);
      print_progress(i + 1, options.sample_count + options.warmup_rounds,
                     starttime);
    }
    for (int i = 0; i < timings_by_device.size(); ++i)
      timings_by_device[i].clear();

    for (int i = 0; i < options.sample_count; ++i) {
      run_iteration(kern, command_queues, buffer, timings_by_device);
      print_progress(i + 1 + options.warmup_rounds,
                     options.sample_count + options.warmup_rounds, starttime);
    }

    // Print per-device info
    std::vector<double> all_times;
    for (size_t i = 0; i < devices.size(); ++i) {
      std::cout << "\tDevice " << i << ":" << std::endl
                << "\t\tname: " << devices[i].getInfo<CL_DEVICE_NAME>()
                << std::endl
                << "\t\tversion: " << devices[i].getInfo<CL_DEVICE_VERSION>()
                << std::endl;

      std::vector<double> device_times(options.sample_count);
      for (int j = 0; j < options.sample_count; ++j) {
        double t = timings_by_device[i][j];
        device_times[j] = t;
        all_times.push_back(t);
      }

      print_measurements("event timing:", device_times, 2);
    }
    std::cout << "\tOverall:" << std::endl;
    print_measurements("event timing:", all_times, 2);

    for (cl::CommandQueue &cq : command_queues)
      cq.finish();
  } catch (cl::Error &err) {
    std::cerr << err.what() << " = " << err.err() << std::endl;
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
  return 0;
}
