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

#include <algorithm>
#include <array>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#define ROW_MAJOR 0
#define COL_MAJOR 1
#define TRANSPOSE_NONE 0
#define TRANSPOSE_A 1
#define TRANSPOSE_B 2

int main() {
  cl_int Status;
  cl::Platform Platform;
  cl::Device Dev;
  std::string DevName;
  std::tie(Platform, Dev, DevName) = findDeviceWithDBK("set_rows_exp");

  auto Ctx = cl::Context(Dev);
  auto CmdQ = cl::CommandQueue(Ctx, 0, &Status);
  TEST_ASSERT(Status == CL_SUCCESS);

  TensorDesc RowsDesc({1, 1, 4, 10}, CL_TENSOR_DTYPE_FP32_EXP);
  TensorDesc IdxsDesc({1, 1, 1, 4}, CL_TENSOR_DTYPE_INT64_EXP);
  TensorDesc DataDesc({1, 1, 8, 10}, CL_TENSOR_DTYPE_FP32_EXP);
  RowsDesc.setLayout(TensorLayoutBLAS({3, 2, 1}));
  IdxsDesc.setLayout(TensorLayoutBLAS({3, 2, 1}));
  DataDesc.setLayout(TensorLayoutBLAS({3, 2, 1}));

  cl_dbk_attributes_set_rows_exp SetRowsAttrs{};
  memcpy(&SetRowsAttrs.data_in, DataDesc.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&SetRowsAttrs.rows, RowsDesc.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&SetRowsAttrs.indices, IdxsDesc.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&SetRowsAttrs.data_out, DataDesc.get(), sizeof(cl_tensor_desc_exp));

  cl::Program Prog;
  cl::Kernel Kernel;
  std::tie(Prog, Kernel) =
      assertCreateDBK(Ctx, Dev, CL_DBK_SET_ROWS_EXP, "set_rows", SetRowsAttrs);

  std::vector<float> RowsData(RowsDesc.numElements());
  std::iota(RowsData.begin(), RowsData.end(), 100.0f);
  std::vector<int64_t> IdxsData{1, 7, 5, 2};
  std::vector<float> DataData(DataDesc.numElements());
  std::iota(DataData.begin(), DataData.end(), 1.0f);

  auto RowsTensor = createTensor(Ctx, RowsDesc, RowsData.data(), &Status);
  TEST_ASSERT(Status == CL_SUCCESS);
  auto IdxsTensor = createTensor(Ctx, IdxsDesc, IdxsData.data(), &Status);
  TEST_ASSERT(Status == CL_SUCCESS);
  auto DataTensor = createTensor(Ctx, DataDesc, DataData.data(), &Status);
  TEST_ASSERT(Status == CL_SUCCESS);

  Kernel.setArg(0, DataTensor);
  Kernel.setArg(1, RowsTensor);
  Kernel.setArg(2, IdxsTensor);
  Kernel.setArg(3, DataTensor);

  Status = CmdQ.enqueueNDRangeKernel(Kernel, cl::NullRange, cl::NDRange{1, 1},
                                     cl::NullRange);

  TEST_ASSERT(Status == CL_SUCCESS);

  std::vector<float> Result(DataDesc.numElements(), 0.0f);
  Status = CmdQ.enqueueReadBuffer(DataTensor, CL_FALSE, 0,
                                  DataDesc.getStorageSize(), Result.data());
  TEST_ASSERT(Status == CL_SUCCESS);

  Status = CmdQ.finish();
  TEST_ASSERT(Status == CL_SUCCESS);

  std::vector<float> Reference = {
      1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  //
      100, 101, 102, 103, 104, 105, 106, 107, 108, 109, //
      130, 131, 132, 133, 134, 135, 136, 137, 138, 139, //
      31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  //
      41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  //
      120, 121, 122, 123, 124, 125, 126, 127, 128, 129, //
      61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  //
      110, 111, 112, 113, 114, 115, 116, 117, 118, 119};

  if (Result != Reference)
    return 1;

  std::cout << "OK\n";
  return 0;
}
