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

#include "CL/opencl.hpp"

#include "dbk_utils.hh"

#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

int main() {
  cl_int Status;
  cl::Platform Platform;
  cl::Device Dev;
  std::string DevName;
  std::tie(Platform, Dev, DevName) = findDeviceWithDBK("add_exp");

  auto Ctx = cl::Context(Dev);
  auto CmdQ = cl::CommandQueue(Ctx, 0, &Status);
  TEST_ASSERT(Status == CL_SUCCESS);

  TensorDesc Src0Desc({3, 4}, CL_TENSOR_DTYPE_FP32_EXP);
  TensorDesc Src1Desc({2, 1, 4}, CL_TENSOR_DTYPE_FP32_EXP);
  TensorDesc DstDesc({2, 3, 4}, CL_TENSOR_DTYPE_FP32_EXP);

  Src0Desc.setLayout(TensorLayoutBLAS({1}));
  Src1Desc.setLayout(TensorLayoutBLAS({2, 1}));
  DstDesc.setLayout(TensorLayoutBLAS({2, 1}));

  cl_dbk_attributes_add_exp AddAttrs{};
  cl_dbk_attributes_mul_exp MulAttrs{};

  memcpy(&AddAttrs.src0, Src0Desc.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&AddAttrs.src1, Src1Desc.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&AddAttrs.dst, DstDesc.get(), sizeof(cl_tensor_desc_exp));

  memcpy(&MulAttrs.src0, Src0Desc.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&MulAttrs.src1, Src1Desc.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&MulAttrs.dst, DstDesc.get(), sizeof(cl_tensor_desc_exp));

  cl::Program Prog;
  std::vector<cl::Kernel> Kernels;
  std::tie(Prog, Kernels) =
      assertCreateDBKs(Ctx, Dev,
                       {
                           {CL_DBK_ADD_EXP, "add", &AddAttrs},
                           {CL_DBK_MUL_EXP, "mul", &MulAttrs},
                       });

  std::vector<float> Src0Data{1, 2,  3,  4, //
                              5, 6,  7,  8, //
                              9, 10, 11, 12};
  std::vector<float> Src1Data{100,  100,  100,  100, //
                              1000, 1000, 1000, 1000};

  auto Src0Tensor = createTensor(Ctx, Src0Desc, Src0Data.data(), &Status);
  TEST_ASSERT(Status == CL_SUCCESS);
  auto Src1Tensor = createTensor(Ctx, Src1Desc, Src1Data.data(), &Status);
  TEST_ASSERT(Status == CL_SUCCESS);
  auto DstTensor = createTensor(Ctx, DstDesc, nullptr, &Status);
  TEST_ASSERT(Status == CL_SUCCESS);

  for (auto &K : Kernels) {
    K.setArg(0, Src0Tensor);
    K.setArg(1, Src1Tensor);
    K.setArg(2, DstTensor);
  }

  // Launch add_exp.
  Status = CmdQ.enqueueNDRangeKernel(Kernels.at(0), cl::NullRange,
                                     cl::NDRange{1, 1}, cl::NullRange);
  TEST_ASSERT(Status == CL_SUCCESS);

  std::vector<float> DstData(DstDesc.numElements());
  Status = CmdQ.enqueueReadBuffer(DstTensor, CL_TRUE, 0,
                                  DstDesc.getStorageSize(), DstData.data());
  TEST_ASSERT(Status == CL_SUCCESS);

  const std::vector<float> AddRef{
      101,  102,  103,  104,  //
      105,  106,  107,  108,  //
      109,  110,  111,  112,  //
      1001, 1002, 1003, 1004, //
      1005, 1006, 1007, 1008, //
      1009, 1010, 1011, 1012, //
  };

  for (unsigned I = 0, E = AddRef.size(); I < E; I++)
    if (DstData.at(I) != AddRef.at(I)) {
      std::cerr << "error: add_exp: mismatch at [" << I << "]: Expected '"
                << AddRef.at(I) << "'. Got '" << DstData.at(I) << "'\n";
      return 1;
    }

  // Launch mul_exp.
  Status = CmdQ.enqueueNDRangeKernel(Kernels.at(1), cl::NullRange,
                                     cl::NDRange{1, 1}, cl::NullRange);
  TEST_ASSERT(Status == CL_SUCCESS);

  Status = CmdQ.enqueueReadBuffer(DstTensor, CL_TRUE, 0,
                                  DstDesc.getStorageSize(), DstData.data());
  TEST_ASSERT(Status == CL_SUCCESS);

  const std::vector<float> MulRef{
      100,  200,   300,   400,   //
      500,  600,   700,   800,   //
      900,  1000,  1100,  1200,  //
      1000, 2000,  3000,  4000,  //
      5000, 6000,  7000,  8000,  //
      9000, 10000, 11000, 12000, //
  };

  for (unsigned I = 0, E = MulRef.size(); I < E; I++)
    if (DstData.at(I) != MulRef.at(I)) {
      std::cerr << "error: mul_exp: mismatch at [" << I << "]: Expected '"
                << MulRef.at(I) << "'. Got '" << DstData.at(I) << "'\n";
      return 1;
    }

  std::cout << "OK\n";
  return 0;
}
