// Copyright (c) 2025 Henry Linjamäki / Intel Finland Oy

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include "CL/opencl.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

static void test(cl::Device &Dev) {
  auto Ctx = cl::Context(Dev);
  size_t SubBufAlignment = Dev.getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();

  cl::Buffer Buffer(Ctx, 0, SubBufAlignment * 2);

  cl_tensor_layout_blas_exp Layout;
  Layout.leading_dims[0] = 1;
  Layout.leading_dims[1] = 0;

  cl_tensor_desc_exp TensorDesc;
  TensorDesc.rank = 1;
  TensorDesc.dtype = CL_TENSOR_DTYPE_FP32_EXP;
  TensorDesc.properties[0] = 0;
  TensorDesc.shape[0] = SubBufAlignment / sizeof(float);
  TensorDesc.layout = &Layout;
  TensorDesc.layout_type = CL_TENSOR_LAYOUT_BLAS_EXP;

  {
    cl_tensor_view_exp TensorViewConfig{SubBufAlignment, &TensorDesc};
    auto TensorView = Buffer.createSubBuffer(
        0, CL_BUFFER_CREATE_TYPE_TENSOR_VIEW_EXP, &TensorViewConfig);
  }
  std::cout << "OK!" << std::endl;
}

int main() try {
  std::vector<cl::Platform> Platforms;
  std::vector<cl::Device> Devices;
  cl::Platform::get(&Platforms);
  cl::Device Device;

  for (auto P : Platforms) {
    P.getDevices(CL_DEVICE_TYPE_ALL, &Devices);
    if (Devices.size() == 0)
      P.getDevices(CL_DEVICE_TYPE_CUSTOM, &Devices);

    for (cl::Device &D : Devices) {
      std::string Exts = D.getInfo<CL_DEVICE_EXTENSIONS>();
      std::string Name = D.getInfo<CL_DEVICE_NAME>();
      if (Exts.find("cl_exp_tensor") == std::string::npos) {
        std::cerr << "Device " << Name
                  << " does not support cl_exp_tensor: skip\n";
        continue;
      }

      std::cout << "Test on device '" << Name << "'" << std::endl;
      test(D);
    }
  }

  return 0;
} catch (cl::Error &err) {
  std::cerr << "ERROR: " << err.what() << "(" << err.err() << ")" << std::endl;
  return EXIT_FAILURE;
}
