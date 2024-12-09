/* Tests bitshift operators follow OpenCL C standard

   Copyright (c) 2024 Michal Babej / Intel Finland Oy

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
#include <CL/opencl.hpp>

#include <iostream>
#include <random>

#define WORK_ITEMS 128 * 1024

const char *KernelSource = R"RAW(
kernel void test_kernel(const global int* in1,
                        const global int* in2,
                        global int* out){

  size_t gid = get_global_id(0);

  out[gid] = in1[gid] << in2[gid];

}

)RAW";

int main(void) {
  int In1[WORK_ITEMS];
  int In2[WORK_ITEMS];
  int Res[WORK_ITEMS];

  std::random_device RandomDevice;
  std::mt19937 Mersenne{RandomDevice()};
  std::uniform_real_distribution<float> UniDist{100.0f, 200.0f};

  for (int i = 0; i < WORK_ITEMS; ++i) {
    In1[i] = UniDist(Mersenne);
    In2[i] = UniDist(Mersenne);
    Res[i] = 0;
  }

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
    cl::Program::Sources sources({KernelSource});
    cl::Program program(context, sources);

    // Build program
    program.build(devices);

    cl::Buffer aBuffer =
        cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   WORK_ITEMS * sizeof(float), (void *)&In1[0]);

    cl::Buffer bBuffer =
        cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                   WORK_ITEMS * sizeof(float), (void *)&In2[0]);

    cl::Buffer cBuffer =
        cl::Buffer(context, CL_MEM_WRITE_ONLY, WORK_ITEMS * sizeof(int), NULL);

    // Create kernel object
    cl::Kernel kernel(program, "test_kernel");

    // Set kernel args
    kernel.setArg(0, aBuffer);
    kernel.setArg(1, bBuffer);
    kernel.setArg(2, cBuffer);

    // Create command queue
    cl::CommandQueue queue(context, devices[0], 0);

    // Do the work
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(WORK_ITEMS),
                               cl::NullRange);

    queue.enqueueReadBuffer(cBuffer,
                            CL_TRUE, // block
                            0, WORK_ITEMS * sizeof(int), Res);

    ok = true;
    int errs = 0;
    for (int i = 0; i < WORK_ITEMS; ++i) {
      int result = In1[i] << (In2[i] & 31);
      if (result != Res[i] && errs < 16) {
        std::cerr << "Res[" << i << "]: " << Res[i] << " should be: " << result
                  << "; " << In1[i] << " << (" << In2[i] << " & 31)\n";
        ok = false;
        ++errs;
      }
    }

    queue.finish();
  } catch (cl::Error &err) {
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
