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
  std::tie(Platform, Dev, DevName) = findDeviceWithDBK("rms_norm_exp");

  auto Ctx = cl::Context(Dev);
  auto CmdQ = cl::CommandQueue(Ctx, 0, &Status);
  TEST_ASSERT(Status == CL_SUCCESS);

  const unsigned NumRows = 2;
  const unsigned NumCols = 8;
  TensorDesc Desc({NumRows, NumCols}, CL_TENSOR_DTYPE_FP32_EXP);
  Desc.setLayout(TensorLayoutBLAS({1}));

  cl_dbk_attributes_rms_norm_exp RmsNormAttrs{};
  memcpy(&RmsNormAttrs.src, Desc.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&RmsNormAttrs.dst, Desc.get(), sizeof(cl_tensor_desc_exp));
  RmsNormAttrs.start_dim = 1;
  RmsNormAttrs.epsilon.ff = 0.1f;

  cl::Program Prog;
  cl::Kernel Kernel;
  std::tie(Prog, Kernel) =
      assertCreateDBK(Ctx, Dev, CL_DBK_RMS_NORM_EXP, "rms_norm", RmsNormAttrs);

  std::vector<float> SrcData(Desc.numElements());
  std::iota(SrcData.begin(), SrcData.end(), 1.0f);

  auto SrcTensor = createTensor(Ctx, Desc, SrcData.data(), &Status);
  TEST_ASSERT(Status == CL_SUCCESS);
  auto DstTensor = createTensor(Ctx, Desc, nullptr, &Status);
  TEST_ASSERT(Status == CL_SUCCESS);

  Kernel.setArg(0, SrcTensor);
  Kernel.setArg(1, DstTensor);
  Status = CmdQ.enqueueNDRangeKernel(Kernel, cl::NullRange, cl::NDRange{1, 1},
                                     cl::NullRange);
  TEST_ASSERT(Status == CL_SUCCESS);

  std::vector<float> Result(Desc.numElements(), 0.0f);
  Status = CmdQ.enqueueReadBuffer(DstTensor, CL_TRUE, 0, Desc.getStorageSize(),
                                  Result.data());
  TEST_ASSERT(Status == CL_SUCCESS);

  std::vector<float> Reference(Desc.numElements(), 0.0f);
  for (unsigned I = 0; I < NumRows; I++) {
    float SquaredSum = 0.0f;
    for (unsigned J = 0; J < NumCols; J++) {
      auto X = SrcData[I * NumCols + J];
      SquaredSum += X * X;
    }
    float Rms = std::sqrt((SquaredSum / NumCols) + RmsNormAttrs.epsilon.ff);

    for (unsigned J = 0; J < NumCols; J++) {
      Reference[I * NumCols + J] = SrcData[I * NumCols + J] / Rms;
    }
  }

  const float ErrorEpsilon = 0.001f;
  for (unsigned I = 0, E = Result.size(); I < E; I++) {
    auto Diff = Result.at(I) - Reference.at(I);
    if (std::abs(Diff) > ErrorEpsilon) {
      std::cerr << "error: mismatch at [" << I << "]: Expected '"
                << Reference.at(I) << "'. Got '" << Result.at(I) << "'\n";
      std::cerr << "Difference: " << Diff << ", Error epsilon: " << ErrorEpsilon
                << "\n";
      return 1;
    }
  }

  std::cout << "OK\n";
  return 0;
}
