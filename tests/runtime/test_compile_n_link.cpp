/* Check the separate compile and link works. Also a regression test
   for cuda device.

   Copyright (c) 2024 Henry Linjamäki / Intel Finland Oy

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

#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

#include <cstdlib>
#include <iostream>

constexpr char SourceA[] = R"ocl(
extern int do_a_thing();
kernel void k(global int *Data, unsigned LID) {
  if (get_local_id(0) == LID)
    *Data = 120 + do_a_thing();
}
)ocl";

constexpr char SourceB[] = R"ocl(
int do_a_thing() { return get_local_id(0); }
)ocl";

int main() try {
  std::vector<cl::Platform> PlatformList;
  cl::Platform::get(&PlatformList);

  if (!PlatformList.size()) {
    std::cerr << "Error: no platforms found!\n";
    return EXIT_FAILURE;
  }
  auto SelectedPlatform = PlatformList[0];

  cl::Context Context(CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU, 0);

  std::vector<cl::Device> Devices = Context.getInfo<CL_CONTEXT_DEVICES>();
  if (!Devices.size()) {
    std::cerr << "Error: no CPU or GPU devices available!\n";
  }
  auto SelectedDevice = Devices[0];

  cl::Program Linked;
  {
    cl::Program ProgA(Context, cl::Program::Sources({SourceA}));
    cl::Program ProgB(Context, cl::Program::Sources({SourceB}));
    ProgA.compile();
    ProgB.compile();
    Linked = cl::linkProgram({ProgA, ProgB});
  }

  cl::Buffer DataBuf =
      cl::Buffer(Context, CL_MEM_WRITE_ONLY, sizeof(cl_int), nullptr);

  cl::Kernel K(Linked, "k");
  K.setArg(0, DataBuf);
  K.setArg<cl_int>(1, 3);

  cl::CommandQueue Queue(Context, SelectedDevice, 0);
  Queue.enqueueNDRangeKernel(K, cl::NullRange, cl::NDRange(10), cl::NullRange);
  cl_int Data;
  Queue.enqueueReadBuffer(DataBuf, /*blocking=*/CL_TRUE, 0, sizeof(cl_int),
                          &Data);

  if (Data != 123) {
    std::cerr << "Expected Data==123. Got Data==" << Data << ".\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
} catch (cl::Error &err) {
  std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
  return EXIT_FAILURE;
}
