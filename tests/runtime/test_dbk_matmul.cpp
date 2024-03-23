// Copyright (c) 2024 Henry Linjamäki / Intel Finland Oy

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

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#define ROW_MAJOR 0
#define COL_MAJOR 1
#define TRANSPOSE_NONE 0
#define TRANSPOSE_A 1
#define TRANSPOSE_B 2

#define EXPECT(expr)                                                           \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << __FILE__ << ":" << __LINE__ << ": error: expected '"        \
                << #expr << "'\n";                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (false)

bool DevUsesLayoutTypeML;
bool DevSupportsColMajor;
bool DevSupportsRowMajor;
bool DevSupportsStrides;

static std::tuple<cl::Platform, cl::Device, std::string> findDeviceWithMatmulDBK() noexcept {
  std::vector<cl::Platform> Platforms;
  std::vector<cl::Device> Devices;
  cl::Platform::get(&Platforms);
  cl::Device Device;


  for (auto P : Platforms) {
    P.getDevices(CL_DEVICE_TYPE_ALL, &Devices);
    if (Devices.size() == 0) {
      P.getDevices(CL_DEVICE_TYPE_CUSTOM, &Devices);
    }
    for (cl::Device &D : Devices) {
      std::string Exts = D.getInfo<CL_DEVICE_EXTENSIONS>();
      std::string Name = D.getInfo<CL_DEVICE_NAME>();
      if (Exts.find("cl_exp_defined_builtin_kernels") == std::string::npos) {
        std::cerr << "Device " << Name << " does not support cl_exp_defined_builtin_kernels\n";
        continue;
      }

      std::string BiKs = D.getInfo<CL_DEVICE_BUILT_IN_KERNELS>();
      if (BiKs.find("matmul_exp") != std::string::npos) {
        std::cerr << "Selected device: " << D.getInfo<CL_DEVICE_NAME>() << "\n";
        return std::make_tuple(P, D, Name);
      } else {
        std::cerr << "Device " << D.getInfo<CL_DEVICE_NAME>() << " does not support BIKD exp_matmul\n";
      }
    }
  }

  std::cerr << "No suitable device found\n";
  std::exit(77);
}

class TensorLayoutBLAS {
  std::vector<cl_tensor_dim_exp> LeadingDims;
  std::vector<size_t> LeadingStrides;
  cl_tensor_layout_blas_pitched_exp Layout;

public:
  TensorLayoutBLAS() {
    memset(&Layout, 0, sizeof(Layout));
  }
  TensorLayoutBLAS(std::initializer_list<cl_tensor_dim_exp> TheLeadingDims,
                   std::initializer_list<size_t> TheLeadingStrides)
      : LeadingDims(TheLeadingDims), LeadingStrides(TheLeadingStrides) {
    memcpy(Layout.leading_dims, LeadingDims.data(),
           LeadingDims.size() * sizeof(cl_tensor_dim_exp));
    memcpy(Layout.leading_strides, LeadingStrides.data(), LeadingStrides.size()*sizeof(size_t) );
  }

  TensorLayoutBLAS& operator=(const TensorLayoutBLAS& Other) = default;
  TensorLayoutBLAS& operator=(TensorLayoutBLAS&& Other) = delete;
  cl_tensor_layout_blas_pitched_exp *get() noexcept { return &Layout; }

  // In elements.
  size_t getSize() const noexcept { return LeadingStrides.back(); }
  unsigned getNumLeadingDims() const noexcept { return LeadingDims.size(); }
  const std::vector<cl_tensor_dim_exp> &getLeadingDims() const noexcept {
    return LeadingDims;
  }
};

class TensorDesc {
  std::vector<cl_tensor_shape_exp> Shape;
  TensorLayoutBLAS LayoutBLAS;
  cl_tensor_layout_ml_exp LayoutML;
  cl_tensor_desc_exp Desc;
  size_t StorageSize;

public:
  TensorDesc(std::initializer_list<cl_tensor_shape_exp> TheShape,
             cl_tensor_datatype_exp DType)
      : Shape(TheShape), StorageSize(0) {

    Desc.rank = Shape.size();
    assert(Desc.rank <= CL_MEM_MAX_TENSOR_RANK_EXP);
    memset(Desc.shape, 0, sizeof(Desc.shape));
    memcpy(Desc.shape, Shape.data(),
           Shape.size() * sizeof(cl_tensor_shape_exp));

    Desc.dtype = DType;
    Desc.layout = nullptr;
    Desc.layout_type = 0;
    Desc.properties[0] = 0;
  }

  void setLayout(const TensorLayoutBLAS &TheLayout) {
    LayoutBLAS = TheLayout;
    assert(LayoutBLAS.getNumLeadingDims() == 0 ||
           LayoutBLAS.getNumLeadingDims() == Shape.size() - 1);
    Desc.layout_type = CL_TENSOR_LAYOUT_BLAS_PITCHED_EXP;
    Desc.layout = LayoutBLAS.get();
    if (LayoutBLAS.getNumLeadingDims()) {

      // Awkward way to find trailing dimension.
      auto Dims = LayoutBLAS.getLeadingDims();
      std::sort(Dims.begin(), Dims.end());
      unsigned TrailingDim = 0;
      while (TrailingDim < Dims.size() && TrailingDim == Dims[TrailingDim])
        TrailingDim++;

      assert(TrailingDim < rank());
      StorageSize = LayoutBLAS.getSize() * Shape[TrailingDim] * elementSize();
    }
  }

  void setLayout(cl_tensor_layout_ml_type_exp LayoutMLType) {
    LayoutML.ml_type = LayoutMLType;
    Desc.layout_type = CL_TENSOR_LAYOUT_ML_EXP;
    Desc.layout = &LayoutML;

    StorageSize = elementSize();
    for (unsigned i = 0; i < Shape.size(); ++i) StorageSize *= Shape[i];
  }

  const cl_tensor_desc_exp *get() const noexcept { return &Desc; }

  unsigned rank() const noexcept { return Shape.size(); }

  // In bytes.
  unsigned elementSize() const noexcept {
    switch (Desc.dtype) {
    case CL_TENSOR_DTYPE_FP64_EXP:
      return 8;
    case CL_TENSOR_DTYPE_FP32_EXP:
      return 4;
    case CL_TENSOR_DTYPE_FP16_EXP:
      return 2;
    default:
      return 1;
    }
  }

  size_t getStorageSize() const noexcept { return StorageSize; }
};

template <typename T>
static cl::Buffer createTensor(cl::Context &Ctx, const TensorDesc &TDesc,
                               T *HostPtr = nullptr, cl_int *Status = nullptr) {
  cl_mem_properties Props[] = {
      CL_MEM_TENSOR_EXP, reinterpret_cast<cl_mem_properties>(TDesc.get()), 0};

  size_t BufSize = TDesc.getStorageSize();
  return cl::Buffer(clCreateBufferWithProperties(
      Ctx.get(), Props, (HostPtr ? CL_MEM_COPY_HOST_PTR : 0),
      BufSize, // TBC: zero = inherit from the tensor description.
      const_cast<typename std::remove_cv<T>::type *>(HostPtr), Status));
}

cl::Platform Platform;
cl::Device Dev;
std::string DevName;
cl::Context Ctx;
cl::CommandQueue CmdQ;
clCreateProgramWithDefinedBuiltInKernelsEXP_fn createProgramWithDBKs;
cl_dbk_attributes_matmul_exp MatmulAttrs;

void doFloatMatmul(bool ColumnMajor, unsigned Transpose, unsigned M, unsigned N,
                   unsigned K, std::initializer_list<float> AData, unsigned Lda,
                   std::initializer_list<float> BData, unsigned Ldb,
                   std::vector<float> &Result, unsigned Ldc) {

  float MarkerVal = 9999.0f;
  Result = std::vector<float>((ColumnMajor ? N : M) * Ldc, MarkerVal);

  TensorLayoutBLAS ATLayout({ColumnMajor ? 0u : 1u}, {Lda});
  TensorLayoutBLAS BTLayout({ColumnMajor ? 0u : 1u}, {Ldb});
  TensorLayoutBLAS CTLayout({ColumnMajor ? 0u : 1u}, {Ldc});

  TensorDesc ATDesc({M, K}, CL_TENSOR_DTYPE_FP32_EXP);
  TensorDesc BTDesc({K, N}, CL_TENSOR_DTYPE_FP32_EXP);
  TensorDesc CTDesc({M, N}, CL_TENSOR_DTYPE_FP32_EXP);
  cl_tensor_layout_ml_type_exp MLType = ColumnMajor ? CL_TENSOR_LAYOUT_ML_CN_EXP : CL_TENSOR_LAYOUT_ML_NC_EXP;

  if (DevUsesLayoutTypeML) {
    ATDesc.setLayout(MLType);
    BTDesc.setLayout(MLType);
    CTDesc.setLayout(MLType);
  } else {
    ATDesc.setLayout(ATLayout);
    BTDesc.setLayout(BTLayout);
    CTDesc.setLayout(CTLayout);
  }
  memcpy(&MatmulAttrs.a, ATDesc.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&MatmulAttrs.b, BTDesc.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&MatmulAttrs.c, CTDesc.get(), sizeof(cl_tensor_desc_exp));

  MatmulAttrs.trans_a = !!(Transpose & TRANSPOSE_A);
  MatmulAttrs.trans_b = !!(Transpose & TRANSPOSE_B);
  MatmulAttrs.kernel_props[0] = 0;
  memset(MatmulAttrs.kernel_props, 0, sizeof(MatmulAttrs.kernel_props));

  constexpr const char ExpextedKernelName[] = "my_matmul";
  cl_int Status;
  cl_device_id Devices[1] = {Dev()};
  cl_dbk_id_exp IDs[1] = {CL_DBK_MATMUL_EXP};
  const char *Names[1] = {ExpextedKernelName};
  cl_int DeviceStatus[1] = {0};
  cl_dbk_attributes_matmul_exp *Attrs[1] = {&MatmulAttrs};
  cl_program Yolo =
      createProgramWithDBKs(Ctx(), 1, Devices, 1, IDs, Names,
                            (const void **)Attrs, DeviceStatus, &Status);
  EXPECT(Status == CL_SUCCESS);
  cl::Program MatmulProg(Yolo);

  Status = MatmulProg.build();
  EXPECT(Status == CL_SUCCESS);

  auto MatmulKernel = cl::Kernel(MatmulProg, ExpextedKernelName, &Status);
  EXPECT(Status == CL_SUCCESS);

  std::string ActualKernelName =
      MatmulKernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
  EXPECT(ActualKernelName == ExpextedKernelName);

  auto ATensor = createTensor(Ctx, ATDesc, AData.begin(), &Status);
  EXPECT(Status == CL_SUCCESS);
  auto BTensor = createTensor(Ctx, BTDesc, BData.begin(), &Status);
  EXPECT(Status == CL_SUCCESS);
  auto CTensor = createTensor(Ctx, CTDesc, Result.data(), &Status);

  MatmulKernel.setArg(0, ATensor);
  MatmulKernel.setArg(1, BTensor);
  MatmulKernel.setArg(2, CTensor);

  Status = CmdQ.enqueueNDRangeKernel(MatmulKernel, cl::NullRange,
                                     cl::NDRange{1, 1}, cl::NullRange);
  EXPECT(Status == CL_SUCCESS);

  Status = CmdQ.enqueueReadBuffer(CTensor, CL_FALSE, 0, CTDesc.getStorageSize(), Result.data() );
  EXPECT(Status == CL_SUCCESS);

  Status = CmdQ.finish();
  EXPECT(Status == CL_SUCCESS);
}

template <typename T>
void dump2DSlice(unsigned Rows, unsigned Cols, const std::vector<T> &A,
                 unsigned Lda) {
  for (unsigned I = 0; I < Rows; I++) {
    for (unsigned J = 0; J < Cols; J++) {
      auto AVal = A[I * Lda + J];
      std::cerr << (J == 0 ? "" : " ") << AVal;
    }
    std::cerr << "\n";
  }
}

void check2DSlice(unsigned M, unsigned N, const std::vector<float> &A,
                  unsigned Lda, const std::vector<float> &B, unsigned Ldb,
                  float Delta = 0.0f) {
  unsigned ErrorCount = 0;
  for (unsigned I = 0; I < M; I++)
    for (unsigned J = 0; J < N; J++) {
      auto AVal = A[I * Lda + J];
      auto BVal = B[I * Ldb + J];
      if (AVal != BVal) {
        std::cerr << "error: mismatch at [" << I << ", " << J
                  << "]: LHS=" << AVal << ", RHS=" << BVal << "\n";
        if (++ErrorCount > 10)
          goto LoopNestExit;
      }
    }
LoopNestExit:
  if (ErrorCount)
    std::exit(1);
  else
    std::cout << "OK" << std::endl;
}

int main() {
  std::tie(Platform, Dev, DevName) = findDeviceWithMatmulDBK();
  bool isCustom = Dev.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CUSTOM;

  // TODO we should have getInfo queries to query this information
  if (isCustom && DevName.find("AI Boost") != std::string::npos) {
    DevUsesLayoutTypeML = true;
    DevSupportsColMajor = false;
    DevSupportsRowMajor = true;
    DevSupportsStrides = false;
  } else {
    DevUsesLayoutTypeML = false;
    DevSupportsColMajor = true;
    DevSupportsRowMajor = true;
    DevSupportsStrides = true;
  }

  Ctx = cl::Context(Dev);

  cl_int Status = CL_SUCCESS;
  createProgramWithDBKs = (clCreateProgramWithDefinedBuiltInKernelsEXP_fn)
      clGetExtensionFunctionAddressForPlatform(
          Platform(), "clCreateProgramWithDefinedBuiltInKernelsEXP");
  EXPECT(createProgramWithDBKs != nullptr);

  CmdQ = cl::CommandQueue(Ctx, 0, &Status);
  EXPECT(Status == CL_SUCCESS);

  std::vector<float> Result;
  // Basic case. Non-strided inputs and output.
  if (DevSupportsColMajor) {
    std::cout << "--- Matmul 1C ---\n";
    doFloatMatmul(COL_MAJOR, TRANSPOSE_NONE, 4, 4, 4,
                  {1.0f, 2.0f, 3.0f, 4.0f,      //
                   5.0f, 6.0f, 7.0f, 8.0f,      //
                   9.0f, 10.0f, 11.0f, 12.0f,   //
                   13.0f, 14.0f, 15.0f, 16.0f}, //
                  4,
                  {0.0f, 0.0f, 0.0f, 0.0f,  //
                   1.0f, 1.0f, 1.0f, 1.0f,  //
                   0.0f, 0.0f, 0.0f, 0.0f,  //
                   0.0f, 0.0f, 0.0f, 0.0f}, //
                  4, Result, 4);

    check2DSlice(4, 4, Result, 4,
                 {0.0f, 0.0f, 0.0f, 0.0f,     //
                  28.0f, 32.0f, 36.0f, 40.0f, //
                  0.0f, 0.0f, 0.0f, 0.0f,     //
                  0.0f, 0.0f, 0.0f, 0.0f},
                 4);
  }

  if (DevSupportsRowMajor) {
    std::cout << "--- Matmul 1R ---\n";
    doFloatMatmul(ROW_MAJOR, TRANSPOSE_NONE, 4, 4, 4,
                  {1.0f, 5.0f, 9.0f, 13.f,
                   2.0f, 6.0f, 10.0f, 14.0f,
                   3.0f, 7.0f, 11.0f, 15.0f,
                   4.0f, 8.0f, 12.0f, 16.0f},
                  4,
                  {0.0f, 1.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f, 0.0f},
                  4, Result, 4);

    check2DSlice(4, 4, Result, 4,
                 {0.0f, 28.0f, 0.0f, 0.0f,
                  0.0f, 32.0f, 0.0f, 0.0f,
                  0.0f, 36.0f, 0.0f, 0.0f,
                  0.0f, 40.0f, 0.0f, 0.0f},
                 4);
  }

  if (DevSupportsColMajor && DevSupportsStrides) {
    std::cout << "--- Matmul 2 ---" << std::endl;
    // Transpose an input & strided output.
    doFloatMatmul(COL_MAJOR, TRANSPOSE_A, 4, 4, 4,
                  {1.0f, 2.0f, 3.0f, 4.0f,      //
                   5.0f, 6.0f, 7.0f, 8.0f,      //
                   9.0f, 10.0f, 11.0f, 12.0f,   //
                   13.0f, 14.0f, 15.0f, 16.0f}, //
                  4,
                  {0.0f, 0.0f, 0.0f, 0.0f,  //
                   1.0f, 1.0f, 1.0f, 1.0f,  //
                   0.0f, 0.0f, 0.0f, 0.0f,  //
                   0.0f, 0.0f, 0.0f, 0.0f}, //
                  4, Result, 7);
    check2DSlice(4, 4, Result, 7,
                 {0.0f, 0.0f, 0.0f, 0.0f,     //
                  10.0f, 26.0f, 42.0f, 58.0f, //
                  0.0f, 0.0f, 0.0f, 0.0f,     //
                  0.0f, 0.0f, 0.0f, 0.0f},
                 4);
  }

  if (DevSupportsColMajor && DevSupportsStrides) {
  std::cout << "--- Matmul 3 ---" << std::endl;
    // Strided inputs.
    doFloatMatmul(COL_MAJOR, TRANSPOSE_NONE, 2, 2, 2,
                  {1.0f, 2.0f, 999.0f, 999.0f, 999.0f,   //
                   3.0f, -4.0f, 999.0f, 999.0f, 999.0f}, //
                  5,
                  {2.0f, -1.0f, 999.0f, //
                   2.0f, 3.0f, 999.0f}, //
                  3, Result, 2);

    check2DSlice(2, 2, Result, 2, {-1.0f, 8.0f, 11.0f, -8.0f}, 2);
  }

  if (DevSupportsRowMajor && DevSupportsStrides) {
    std::cout << "--- Matmul 4 ---" << std::endl;
    // Same as above but in the row-major order.
    doFloatMatmul(ROW_MAJOR, TRANSPOSE_NONE, 2, 2, 2,
                  {1.0f, 2.0f, 999.0f, 999.0f, 999.0f,   //
                   3.0f, -4.0f, 999.0f, 999.0f, 999.0f}, //
                  5,
                  {2.0f, -1.0f, 999.0f, //
                   2.0f, 3.0f, 999.0f}, //
                  3, Result, 2);

    check2DSlice(2, 2, Result, 2, {6.0f, 5.0f, -2.0f, -15.0f}, 2);
  }

  if (DevSupportsRowMajor && DevSupportsStrides) {
    std::cout << "--- Matmul 5 ---" << std::endl;
    // Row-major, strided inputs and a matrix transpose.
    doFloatMatmul(ROW_MAJOR, TRANSPOSE_A, 2, 2, 2,
                  {1.0f, 2.0f, 999.0f, 999.0f, 999.0f,   //
                   3.0f, -4.0f, 999.0f, 999.0f, 999.0f}, //
                  5,
                  {2.0f, -1.0f, 999.0f, //
                   2.0f, 3.0f, 999.0f}, //
                  3, Result, 2);

    check2DSlice(2, 2, Result, 2, {8.0f, 8.0f, -4.0f, -14.0f}, 2);
  }

  if (DevSupportsRowMajor) {
    std::cout << "--- Matmul 6 ---" << std::endl;
    // Row-major, with rectangle matrix shapes.
    doFloatMatmul(ROW_MAJOR, TRANSPOSE_NONE, 2, 2, 3,
                  {1.0f, 2.0f, -1.0f,  //
                   3.0f, -4.0f, 2.0f}, //
                  3,
                  {2.0f, -1.0f, //
                   1.0f, -1.0f, //
                   2.0f, 3.0f}, //
                  2, Result, 2);

    check2DSlice(2, 2, Result, 2, {2.0f, -6.0f, 6.0f, 7.0f}, 2);
  }

  std::cout << "--- Completed ---" << std::endl;

  return 0;
}
