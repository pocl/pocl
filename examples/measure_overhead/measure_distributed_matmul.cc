/* Distributed square matrix multiplication benchmark

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
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include "common.hh"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

#define STRINGIFY(x) #x

void print_help(const char *name) {
  std::cerr << "Usage: " << name << " [-p platform_index] "
            << "[-s sample_count] [-d matrix_size]" << std::endl
            << "-p specifies which platform to use." << std::endl
            << std::endl
            << "\tshuffle (reassigns buffers randomly)" << std::endl
            << "-s sets the number of samples measured." << std::endl
            << "-d N sets the size of the matrix to NxN." << std::endl
            << "-r prints the multiplication result." << std::endl;
}

void print_progress(int p, int total, int width = 80) {
  std::string total_str = std::to_string(total);
  std::string p_str = std::to_string(p);
  int bar_width = width - total_str.size() * 2 - 4;

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
            << total_str << std::flush;
  if (p == total)
    std::cerr << std::endl;
}

struct {
  int platform_index = -1;
  size_t sample_count = 1;
  // This is the square root of the number of integers in the buffer, not bytes.
  size_t matrix_size = 128;
  int show_result = 0;
} options;

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
      } else if (arg[1] == 's' && arg[2] == 0) {
        argv++;
        if (!*argv) {
          std::cerr << "Missing sample count" << std::endl;
          goto fail;
        }
        options.sample_count = std::stoi(*argv);
      } else if (arg[1] == 'd' && arg[2] == 0) {
        argv++;
        if (!*argv) {
          std::cerr << "Missing matrix size" << std::endl;
          goto fail;
        }
        options.matrix_size = std::stoull(*argv);
      } else if (arg[1] == 'r' && arg[2] == 0) {
        options.show_result = 1;
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

struct partition {
  uint64_t y;
  uint64_t nrows;
  uint64_t capacity;
  std::vector<testing_buffer> buf;
};

void run_iteration(cl::Kernel &kern,
                   std::vector<cl::CommandQueue> &command_queues,
                   std::vector<std::vector<testing_buffer>> &buffers,
                   std::vector<partition> &partitions,
                   std::vector<std::vector<cl::Event>> &events_by_device,
                   size_t iteration) {
  for (size_t i = 0; i < partitions.size(); ++i) {
    std::vector<cl::Event> deps{};
    if (partitions[i].buf[iteration].last_write_event.get() != nullptr)
      deps.push_back(std::move(partitions[i].buf[iteration].last_write_event));

    cl::Event ker_ev;
    kern.setArg(0, buffers[0][i].buffer);
    kern.setArg(1, buffers[1][i].buffer);
    kern.setArg(2, partitions[i].buf[iteration].buffer);
    kern.setArg(4, partitions[i].y);
    command_queues[iteration * partitions.size() + i].enqueueNDRangeKernel(
        kern, cl::NDRange(0, partitions[i].y),
        cl::NDRange(options.matrix_size, partitions[i].nrows), cl::NullRange,
        (deps.size() > 0 ? &deps : nullptr), &ker_ev);
    events_by_device[i].push_back(ker_ev);
    partitions[i].buf[iteration].last_write_event = std::move(ker_ev);
  }

  for (size_t step = 1; step < partitions.size(); step *= 2)
    for (size_t i = 0; i + step < partitions.size(); i += step * 2) {
      cl::Event copy_ev;
      std::vector<cl::Event> deps{};
      deps.clear();
      deps.push_back(partitions[i + step].buf[iteration].last_write_event);
      deps.push_back(partitions[i].buf[iteration].last_write_event);
      command_queues[iteration * partitions.size() + i].enqueueCopyBufferRect(
          partitions[i + step].buf[iteration].buffer,
          partitions[i].buf[iteration].buffer, {0, 0, 0},   /* src_origin */
          {0, partitions[i + step].y - partitions[i].y, 0}, /* dst_origin */
          {options.matrix_size * sizeof(float), partitions[i + step].capacity,
           1},                                 /* region */
          options.matrix_size * sizeof(float), /* src_row_pitch */
          0,                                   /* src_slice_pitch */
          options.matrix_size * sizeof(float), /* dst_row_pitch */
          0,                                   /* dst_slice_pitch */
          &deps, &copy_ev);
      partitions[i].buf[iteration].last_write_event = std::move(copy_ev);
    }
}

std::vector<size_t> calc_capacity(const std::vector<size_t> &sizes) {
  std::vector<size_t> capacities(sizes);
  for (size_t step = 1; step < sizes.size(); step *= 2)
    for (size_t i = 0; i + step < sizes.size(); i += step * 2)
      capacities[i] += capacities[i + step];
  return capacities;
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
    if (devices.empty()) {
      std::cerr << "No devices in platform " << index << " !" << std::endl;
      return false;
    }

    cl::Context ctx(devices);
    std::vector<std::vector<testing_buffer>> buffers(2);

    std::vector<cl::CommandQueue> command_queues(devices.size() *
                                                 options.sample_count);
    std::vector<std::vector<cl::Event>> events_by_device(devices.size());
    for (size_t j = 0; j < options.sample_count; ++j) {
      for (size_t i = 0; i < devices.size(); ++i) {
        command_queues[j * devices.size() + i] = cl::CommandQueue(
            ctx, devices[i],
            cl::QueueProperties::Profiling | cl::QueueProperties::OutOfOrder);

        if (!j) {
          float pi = 3.14159;
          buffers[0].push_back({cl::Buffer(ctx, CL_MEM_READ_ONLY,
                                           sizeof(float) * options.matrix_size *
                                               options.matrix_size,
                                           nullptr),
                                cl::Event()});
          buffers[1].push_back({cl::Buffer(ctx, CL_MEM_READ_ONLY,
                                           sizeof(float) * options.matrix_size *
                                               options.matrix_size,
                                           nullptr),
                                cl::Event()});

          events_by_device[i].reserve(options.sample_count);
          command_queues[i].enqueueFillBuffer(
              buffers[0][i].buffer, pi, 0,
              sizeof(float) * options.matrix_size * options.matrix_size,
              nullptr, nullptr);
          command_queues[i].enqueueFillBuffer(
              buffers[1][i].buffer, pi, 0,
              sizeof(float) * options.matrix_size * options.matrix_size,
              nullptr, nullptr);
        }
      }
    }

    cl::Program prog(
        ctx,
        STRINGIFY(
            /*************************************************************************/
            __kernel void partial_matrix_multiply(
                __global const float *const a, __global const float *const b,
                __global float *c, unsigned long dim, unsigned long start) {
              size_t row = get_global_id(1);
              size_t col = get_global_id(0);
              size_t dst_row = row - start;

              float tmp = 0.0f;
              for (size_t i = 0; i < dim; ++i) {
                tmp += a[row * dim + i] * b[i * dim + col];
              }
              c[dst_row * dim + col] = tmp;
            }
            /*************************************************************************/
            ));

    try {
      prog.build(devices);
    } catch (cl::Error &e) {
      if (e.err() == CL_BUILD_PROGRAM_FAILURE) {
        for (cl::Device dev : devices) {
          // Check the build status
          cl_build_status status =
              prog.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
          if (status != CL_BUILD_ERROR)
            continue;

          // Get the build log
          std::string name = dev.getInfo<CL_DEVICE_NAME>();
          std::string buildlog = prog.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
          std::cerr << "Build log for " << name << ":" << std::endl
                    << buildlog << std::endl;
        }
      } else {
        throw e;
      }
    }
    cl::Kernel kern(prog, "partial_matrix_multiply");
    kern.setArg(3, options.matrix_size);

    std::vector<partition> partitions;
    if (options.matrix_size < devices.size())
      std::cout << "Note: " << options.matrix_size
                << "^2 matrix is too small, not all " << devices.size()
                << " devices will be used" << std::endl;

    size_t step = std::max(options.matrix_size / devices.size(), (size_t)1);
    size_t overflow = options.matrix_size > devices.size()
                          ? options.matrix_size % devices.size()
                          : 0;

    std::vector<size_t> offsets;
    std::vector<size_t> sizes;
    for (size_t i = 0, y = 0; y < options.matrix_size && i < devices.size();
         ++i) {
      size_t nrows = step + ((i < overflow) ? 1 : 0);
      offsets.push_back(y);
      sizes.push_back(nrows);
      y += nrows;
    }

    std::vector<size_t> capacities = calc_capacity(sizes);

    for (size_t i = 0; i < sizes.size(); ++i) {
      std::vector<testing_buffer> v;
      for (size_t j = 0; j < options.sample_count; ++j) {
        cl::Buffer buf = cl::Buffer(
            ctx, CL_MEM_READ_WRITE,
            sizeof(float) * options.matrix_size * capacities[i], nullptr);
        v.push_back({buf, cl::Event()});
      }
      partitions.push_back({offsets[i], sizes[i], capacities[i], v});
    }

    for (size_t iteration = 0; iteration < options.sample_count; ++iteration)
      for (size_t step = 1; step < partitions.size(); step *= 2)
        for (size_t i = 0; i + step < partitions.size(); i += step * 2) {
          std::vector<cl::Memory> objs{};
          objs.push_back(partitions[i + step].buf[iteration].buffer);
          command_queues[iteration * partitions.size() + i]
              .enqueueMigrateMemObjects(objs, 0);
          command_queues[iteration * partitions.size() + i + step]
              .enqueueMigrateMemObjects(objs, 0);
        }

    for (size_t i = 0; i < command_queues.size(); ++i) {
      command_queues[i].finish();
    }

    // Print local measurements
    std::vector<std::chrono::steady_clock::time_point> starts(
        options.sample_count);
    std::vector<double> times(options.sample_count);
    print_progress(0, options.sample_count);
    for (size_t i = 0; i < options.sample_count; ++i) {
      using namespace std::chrono;
      starts[i] = steady_clock::now();
      run_iteration(kern, command_queues, buffers, partitions, events_by_device,
                    i);

      for (size_t j = 0; j < partitions.size(); ++j) {
        command_queues[i * partitions.size() + j].finish();
      }
      using namespace std::chrono;
      steady_clock::duration iter_duration = steady_clock::now() - starts[i];
      times[i] =
          duration_cast<duration<double, std::micro>>(iter_duration).count();
      print_progress(i+1, options.sample_count);
    }

    print_measurements("host-measured timing:", times, 2);

    if (options.show_result) {
      std::vector<float> result(options.matrix_size * options.matrix_size,
                                -1.0f);
      command_queues[0].enqueueReadBuffer(
          partitions[0].buf[0].buffer, true, 0,
          options.matrix_size * options.matrix_size * sizeof(float),
          result.data());
      std::cout << "Result: " << std::endl;
      for (size_t i = 0; i < options.matrix_size; ++i) {
        for (size_t j = 0; j < options.matrix_size; ++j) {
          std::cout << result[i * options.matrix_size + j] << ",";
        }
        std::cout << std::endl;
      }
    }

    // Print per-device info
    std::vector<double> all_times;
    for (size_t i = 0; i < devices.size(); ++i) {
      std::cout << "\tDevice " << i << ":" << std::endl
                << "\t\tname: " << devices[i].getInfo<CL_DEVICE_NAME>()
                << std::endl
                << "\t\tversion: " << devices[i].getInfo<CL_DEVICE_VERSION>()
                << std::endl;

      for (cl::CommandQueue &cq : command_queues)
        cq.finish();
      std::vector<double> device_times(options.sample_count);
      for (size_t j = 0; j < events_by_device[i].size(); ++j) {
        cl::Event &e = events_by_device[i][j];
        if (e.get() == nullptr)
          continue;
        e.wait();
        size_t ticks = e.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                       e.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        device_times[j] = double(ticks) / 1e3;
        all_times.push_back(device_times[j]);
      }

      print_measurements("event timing:", device_times, 2);
    }
    std::cout << "\tOverall:" << std::endl;
    print_measurements("event timing:", all_times, 2);

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
