/* convol1 - Convolution example based on pytorch/glow kernels

   Copyright (c) 2019 pocl developers

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
#include "pocl_opencl.h"

// Enable OpenCL C++ exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <CL/opencl.hpp>

#include "config.h"

#include <cassert>

#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream> // std::stringstream
#include <streambuf>
#include <string>

#include "sha1.hh"

#define BUF_SIZE 1559680
#define BUF_SIZE_4 389920

typedef std::array<float, BUF_SIZE_4> buffer_t;

typedef char checksum64_t[SHA1_BASE64_SIZE];
typedef char checksum0x_t[SHA1_HEX_SIZE];

/*
static void write_buffer( std::array<float, BUF_SIZE_4> &buffer,
                          const char* fname)
{
  std::ofstream fajl;
  fajl.open (fname, std::ios::out | std::ios::binary);
  fajl.write( (const char*)buffer.data(), BUF_SIZE);
  fajl.close();
}
*/

static void cksum_buffer(buffer_t *b, const char *name) {
  checksum64_t res;
  sha1 sum;
  sum.add(b->data(), BUF_SIZE).finalize().print_base64(res);
  std::cout << name << " : " << res << "\n";
}

#define SEED 902834

int main(void) {
  bool finish_after_every = false;
  size_t iters = 1;
  buffer_t *buffer, *step1, *step2, *step3, *step4;
  buffer = new buffer_t;
  step1 = new buffer_t;
  step2 = new buffer_t;
  step3 = new buffer_t;
  step4 = new buffer_t;

  std::mt19937 gen(SEED);
  auto rnd3 =
      std::bind(std::uniform_real_distribution<float>{0.0f, 1e24f}, gen);

  for (size_t i = 0; i < BUF_SIZE_4; ++i) {
    (*buffer)[i] = rnd3();
  }

  try {
    // Pick platform
    std::vector<cl::Platform> platformList;
    cl::Platform::get(&platformList);
    assert(platformList.size() > 0);

    {
      // Query the set of devices attched to the context
      std::vector<cl::Device> devices;
      platformList[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

      assert(devices.size() > 0);
      cl::Device dev = devices[0];
      if (devices.size() > 1)
        devices.resize(1);
      cl::Context context(devices);

      // Create and program from source
      std::ifstream t(SRCDIR "/examples/convol/kernels.cl");
      std::stringstream shared_cl;
      shared_cl << t.rdbuf();
      t.close();

      std::ifstream t2(SRCDIR "/examples/convol/conv_fwd_mem.cl");
      std::stringstream conv_fwd_mem;
      conv_fwd_mem << t2.rdbuf();
      t2.close();

      cl::Program::Sources sources1({shared_cl.str()});
      cl::Program shared_prog(context, sources1);

      cl::Program::Sources sources2({conv_fwd_mem.str()});
      cl::Program conv_prog(context, sources2);

      // Build programs
      shared_prog.build("-DSIZEOF_HOST_SIZE_T=8 -Ddim_t=uint -Dsdim_t=int");
      conv_prog.build(
          "-Dv_nax=2 -Dv_g=1 -Dv_k_0=5 -Dv_k_1=5 -Dv_p_0=4 -Dv_p_1=4 -Dv_s_0=3 "
          "-Dv_s_1=3 -Dv_d_0=1 -Dv_d_1=1 -Dv_fin=6 -Dv_fout=10 "
          "-Dv_bmul=(float)1 -Dv_imsi_0=41 -Dv_imsi_1=32 -Dv_imso_0=15 "
          "-Dv_imso_1=12 -Dv_pad_A=0 -Dv_pad_B=0 -Dworkgroup_size_0=16 "
          "-Dworkgroup_size_1=4 -DTSK=4 -DTSK_UNROLL=1 -DWPTN=4 -DWPTM=4 "
          "-DVWM=4 -DVWN=4 -Ddim_t=uint -Dsdim_t=int");

      cksum_buffer(buffer, "step0");

      // Create buffer for that uses the host ptr C
      cl::Buffer cBuffer = cl::Buffer(
          context, (cl_mem_flags)(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR),
          BUF_SIZE, (void *)buffer->data());

      // Create command queue
      cl::CommandQueue queue(context, dev, 0);

      // Create kernel objects
      cl::Kernel transposeW(shared_prog, "transposeW");
      cl::Kernel conv_forward_mem(conv_prog, "conv_forward_mem");

      uint32_t temp;
      uint32_t tempA[4];

      for (size_t j = 0; j < iters; ++j) {

        /*********************************************************************************/
        // STEP 1
        // Set kernel args
        transposeW.setArg(0, cBuffer);

        temp = 779840;
        transposeW.setArg(1, sizeof(temp), &temp);

        temp = 629760;
        transposeW.setArg(2, sizeof(temp), &temp);

        tempA[0] = 0x0A;
        tempA[1] = 0x06;
        tempA[2] = 0x05;
        tempA[3] = 0x05;
        transposeW.setArg(3, sizeof(tempA), &tempA);

        tempA[0] = 0x0A;
        tempA[1] = 0x05;
        tempA[2] = 0x05;
        tempA[3] = 0x06;
        transposeW.setArg(4, sizeof(tempA), &tempA);

        tempA[0] = 0x00;
        tempA[1] = 0x03;
        tempA[2] = 0x01;
        tempA[3] = 0x02;
        transposeW.setArg(5, sizeof(tempA), &tempA);

        // offs, glob, loc
        queue.enqueueNDRangeKernel(transposeW, cl::NullRange,
                                   cl::NDRange(10, 5, 1), cl::NDRange(2, 1, 1));

        queue.enqueueReadBuffer(cBuffer,
                                CL_FALSE, // block
                                0, BUF_SIZE, (void *)step1->data());

        if (finish_after_every)
          queue.finish();

        /*********************************************************************************/
        // STEP 2

        // Set kernel args
        transposeW.setArg(0, cBuffer);

        temp = 785856;
        transposeW.setArg(1, sizeof(temp), &temp);

        temp = 0;
        transposeW.setArg(2, sizeof(temp), &temp);

        tempA[0] = 0x14;
        tempA[1] = 0x06;
        tempA[2] = 0x29;
        tempA[3] = 0x20;
        transposeW.setArg(3, sizeof(tempA), &tempA);

        tempA[0] = 0x14;
        tempA[1] = 0x29;
        tempA[2] = 0x20;
        tempA[3] = 0x06;
        transposeW.setArg(4, sizeof(tempA), &tempA);

        tempA[0] = 0x00;
        tempA[1] = 0x03;
        tempA[2] = 0x01;
        tempA[3] = 0x02;
        transposeW.setArg(5, sizeof(tempA), &tempA);

        // offs, glob, loc
        queue.enqueueNDRangeKernel(transposeW, cl::NullRange,
                                   cl::NDRange(20, 41, 1),
                                   cl::NDRange(4, 1, 1));

        queue.enqueueReadBuffer(cBuffer,
                                CL_FALSE, // block
                                0, BUF_SIZE, (void *)step2->data());

        if (finish_after_every)
          queue.finish();

        /*********************************************************************************/
        // STEP 3

        // Set kernel args
        conv_forward_mem.setArg(0, cBuffer);

        temp = 785856;
        conv_forward_mem.setArg(1, sizeof(temp), &temp);

        temp = 779840;
        conv_forward_mem.setArg(2, sizeof(temp), &temp);

        temp = 635776;
        conv_forward_mem.setArg(3, sizeof(temp), &temp);

        temp = 1415616;
        conv_forward_mem.setArg(4, sizeof(temp), &temp);

        // offs, glob, loc
        queue.enqueueNDRangeKernel(conv_forward_mem, cl::NullRange,
                                   cl::NDRange(48, 4, 20),
                                   cl::NDRange(16, 4, 1));

        queue.enqueueReadBuffer(cBuffer,
                                CL_FALSE, // block
                                0, BUF_SIZE, (void *)step3->data());

        if (finish_after_every)
          queue.finish();

        /*********************************************************************************/
        // STEP 4

        // Set kernel args
        transposeW.setArg(0, cBuffer);

        temp = 635840;
        transposeW.setArg(1, sizeof(temp), &temp);

        temp = 1415616;
        transposeW.setArg(2, sizeof(temp), &temp);

        tempA[0] = 0x14;
        tempA[1] = 0x0F;
        tempA[2] = 0x0C;
        tempA[3] = 0x0A;
        transposeW.setArg(3, sizeof(tempA), &tempA);

        tempA[0] = 0x14;
        tempA[1] = 0x0A;
        tempA[2] = 0x0F;
        tempA[3] = 0x0C;
        transposeW.setArg(4, sizeof(tempA), &tempA);

        tempA[0] = 0x00;
        tempA[1] = 0x02;
        tempA[2] = 0x03;
        tempA[3] = 0x01;
        transposeW.setArg(5, sizeof(tempA), &tempA);

        // offs, glob, loc
        queue.enqueueNDRangeKernel(transposeW, cl::NullRange,
                                   cl::NDRange(20, 10, 1),
                                   cl::NDRange(4, 2, 1));

        queue.enqueueReadBuffer(cBuffer,
                                CL_FALSE, // block
                                0, BUF_SIZE, (void *)step4->data());

        if (finish_after_every)
          queue.finish();

        /*********************************************************************************/

        queue.finish();

        cksum_buffer(step1, "step1");
        cksum_buffer(step2, "step2");
        cksum_buffer(step3, "step3");
        cksum_buffer(step4, "step4");
      }
    }

    platformList[0].unloadCompiler();

  } catch (cl::Error &err) {
    std::cerr << "OpenCL ERROR: " << err.what() << "(" << err.err() << ")"
              << std::endl;

    return EXIT_FAILURE;
  }

  delete buffer;
  delete step1;
  delete step2;
  delete step3;
  delete step4;

  return EXIT_SUCCESS;
}
