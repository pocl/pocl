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
  std::tie(Platform, Dev, DevName) = findDeviceWithDBK("convert_exp");

  auto Ctx = cl::Context(Dev);
  auto CmdQ = cl::CommandQueue(Ctx, 0, &Status);
  TEST_ASSERT(Status == CL_SUCCESS);

  TensorDesc DescF32({2, 3, 4, 5}, CL_TENSOR_DTYPE_FP32_EXP);
  TensorDesc DescF16({2, 3, 4, 5}, CL_TENSOR_DTYPE_FP16_EXP);
  DescF32.setLayout(TensorLayoutBLAS({3, 2, 1}));
  DescF16.setLayout(TensorLayoutBLAS({3, 2, 1}));

  cl_dbk_attributes_convert_exp CvtAttrsF32ToF16{};
  cl_dbk_attributes_convert_exp CvtAttrsF16ToF32{};
  memcpy(&CvtAttrsF32ToF16.src, DescF32.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&CvtAttrsF32ToF16.dst, DescF16.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&CvtAttrsF16ToF32.src, DescF16.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&CvtAttrsF16ToF32.dst, DescF32.get(), sizeof(cl_tensor_desc_exp));

  cl::Program Prog;
  std::vector<cl::Kernel> ConvertKernels;
  std::tie(Prog, ConvertKernels) = assertCreateDBKs(
      Ctx, Dev,
      {
          {CL_DBK_CONVERT_EXP, "convert_f32_to_f16", &CvtAttrsF32ToF16},
          {CL_DBK_CONVERT_EXP, "convert_f16_to_f32", &CvtAttrsF16ToF32},
      });

  std::vector<float> Input(DescF32.numElements());
  std::iota(Input.begin(), Input.end(), 1.0f);

  auto InputTensor = createTensor(Ctx, DescF32, Input.data(), &Status);
  TEST_ASSERT(Status == CL_SUCCESS);
  auto TmpTensor = createTensor(Ctx, DescF16, nullptr, &Status);
  TEST_ASSERT(Status == CL_SUCCESS);
  auto OutputTensor = createTensor(Ctx, DescF32, nullptr, &Status);
  TEST_ASSERT(Status == CL_SUCCESS);

  ConvertKernels.at(0).setArg(0, InputTensor);
  ConvertKernels.at(0).setArg(1, TmpTensor);

  ConvertKernels.at(1).setArg(0, TmpTensor);
  ConvertKernels.at(1).setArg(1, OutputTensor);

  Status = CmdQ.enqueueNDRangeKernel(ConvertKernels.at(0), cl::NullRange,
                                     cl::NDRange{1, 1}, cl::NullRange);

  TEST_ASSERT(Status == CL_SUCCESS);
  Status = CmdQ.enqueueNDRangeKernel(ConvertKernels.at(1), cl::NullRange,
                                     cl::NDRange{1, 1}, cl::NullRange);

  std::vector<float> Result(DescF32.numElements(), 0.0f);
  Status = CmdQ.enqueueReadBuffer(OutputTensor, CL_FALSE, 0,
                                  DescF32.getStorageSize(), Result.data());
  TEST_ASSERT(Status == CL_SUCCESS);

  Status = CmdQ.finish();
  TEST_ASSERT(Status == CL_SUCCESS);

  float Ref = 1.0f;
  for (unsigned I = 0, E = Result.size(); I < E; I++, Ref += 1.0f) {
    if (Result.at(I) != Ref) {
      std::cerr << "error: mismatch at [" << I << "]: Expected '" << Ref
                << "'. Got '" << Result.at(I) << "'\n";
      return 1;
    }
  }

  return 0;
}
