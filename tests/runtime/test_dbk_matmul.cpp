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
#include <variant>
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

static std::tuple<cl::Platform, cl::Device, std::string>
findDeviceWithMatmulDBK() noexcept {
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
        std::cerr << "Device " << Name
                  << " does not support cl_exp_defined_builtin_kernels\n";
        continue;
      }

      std::string BiKs = D.getInfo<CL_DEVICE_BUILT_IN_KERNELS>();
      if (BiKs.find("matmul_exp") != std::string::npos) {
        std::cerr << "Selected device: " << D.getInfo<CL_DEVICE_NAME>() << "\n";
        return std::make_tuple(P, D, Name);
      } else {
        std::cerr << "Device " << D.getInfo<CL_DEVICE_NAME>()
                  << " does not support BIKD matmul_exp\n";
      }
    }
  }

  std::cerr << "No suitable device found\n";
  std::exit(77);
}

class TensorLayoutBLAS {
protected:
  std::vector<cl_tensor_dim_exp> LeadingDims;
  cl_tensor_layout_blas_exp PackedLayout;

public:
  TensorLayoutBLAS() { memset(&PackedLayout, 0, sizeof(PackedLayout)); }
  TensorLayoutBLAS(const TensorLayoutBLAS &) = default;
  TensorLayoutBLAS(std::initializer_list<cl_tensor_dim_exp> TheLeadingDims)
      : LeadingDims(TheLeadingDims) {
    memcpy(PackedLayout.leading_dims, LeadingDims.data(),
           LeadingDims.size() * sizeof(cl_tensor_dim_exp));
  }

  TensorLayoutBLAS &operator=(const TensorLayoutBLAS &Other) = default;
  TensorLayoutBLAS &operator=(TensorLayoutBLAS &&Other) = delete;
  cl_tensor_layout_blas_exp *get() noexcept { return &PackedLayout; }

  unsigned getNumLeadingDims() const noexcept { return LeadingDims.size(); }
  const std::vector<cl_tensor_dim_exp> &getLeadingDims() const noexcept {
    return LeadingDims;
  }
};

class TensorLayoutBLASPitched : public TensorLayoutBLAS {
  std::vector<size_t> LeadingStrides;
  cl_tensor_layout_blas_pitched_exp PitchedLayout;

public:
  TensorLayoutBLASPitched() : TensorLayoutBLAS() {
    memset(&PitchedLayout, 0, sizeof(PitchedLayout));
  }
  TensorLayoutBLASPitched(
      std::initializer_list<cl_tensor_dim_exp> TheLeadingDims,
      std::initializer_list<size_t> TheLeadingStrides)
      : TensorLayoutBLAS(TheLeadingDims), LeadingStrides(TheLeadingStrides) {
    memcpy(PitchedLayout.leading_strides, LeadingStrides.data(),
           LeadingStrides.size() * sizeof(size_t));
    memcpy(PitchedLayout.leading_dims, LeadingDims.data(),
           LeadingDims.size() * sizeof(cl_tensor_dim_exp));
  }

  TensorLayoutBLASPitched(const TensorLayoutBLASPitched &) = default;

  TensorLayoutBLASPitched &
  operator=(const TensorLayoutBLASPitched &Other) = default;
  TensorLayoutBLASPitched &operator=(TensorLayoutBLASPitched &&Other) = delete;
  cl_tensor_layout_blas_pitched_exp *get() noexcept { return &PitchedLayout; }

  // In elements.
  size_t getSize() const noexcept { return LeadingStrides.back(); }
};

class TensorDesc {
  std::vector<cl_tensor_shape_exp> Shape;
  cl_tensor_desc_exp Desc;
  std::variant<TensorLayoutBLAS, TensorLayoutBLASPitched,
               cl_tensor_layout_ml_exp>
      Layout;
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
    Layout = TheLayout;
    Desc.layout_type = CL_TENSOR_LAYOUT_BLAS_EXP;
    Desc.layout = std::get<TensorLayoutBLAS>(Layout).get();
    StorageSize = numElements() * elementSize();
  }

  void setLayout(const TensorLayoutBLASPitched &TheLayout) {
    Layout = TheLayout;
    assert(TheLayout.getNumLeadingDims() == 0 ||
           TheLayout.getNumLeadingDims() == Shape.size() - 1);
    Desc.layout_type = CL_TENSOR_LAYOUT_BLAS_PITCHED_EXP;
    Desc.layout = std::get<TensorLayoutBLASPitched>(Layout).get();
    if (TheLayout.getNumLeadingDims()) {

      // Awkward way to find trailing dimension.
      auto Dims = TheLayout.getLeadingDims();
      std::sort(Dims.begin(), Dims.end());
      unsigned TrailingDim = 0;
      while (TrailingDim < Dims.size() && TrailingDim == Dims[TrailingDim])
        TrailingDim++;

      assert(TrailingDim < rank());
      StorageSize = TheLayout.getSize() * Shape[TrailingDim] * elementSize();
    }
  }

  void setLayout(cl_tensor_layout_ml_type_exp LayoutMLType) {
    Layout = cl_tensor_layout_ml_exp{LayoutMLType};
    Desc.layout_type = CL_TENSOR_LAYOUT_ML_EXP;
    Desc.layout = &std::get<cl_tensor_layout_ml_exp>(Layout);
    StorageSize = numElements() * elementSize();
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

  size_t numElements() const noexcept {
    size_t Result = 1;
    for (unsigned i = 0; i < Shape.size(); ++i)
      Result *= Shape[i];
    return Result;
  };
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

template <typename DbkAttrT>
std::tuple<cl::Program, cl::Kernel>
assertCreateDBK(cl::Context Ctx, cl::Device Device, cl_dbk_id_exp DbkID,
                const std::string &KernelName, DbkAttrT &Attributes) {

  cl_int Status;
  cl_device_id Devices[1] = {Device()};
  cl_dbk_id_exp IDs[1] = {DbkID};
  const char *Names[1] = {KernelName.c_str()};
  cl_int DeviceStatus[1] = {0};
  DbkAttrT *Attrs[1] = {&Attributes};
  cl_program Yolo =
      createProgramWithDBKs(Ctx(), 1, Devices, 1, IDs, Names,
                            (const void **)Attrs, DeviceStatus, &Status);
  EXPECT(Status == CL_SUCCESS);
  cl::Program Prog(Yolo);

  Status = Prog.build();
  EXPECT(Status == CL_SUCCESS);

  auto MatmulKernel = cl::Kernel(Prog, KernelName, &Status);
  EXPECT(Status == CL_SUCCESS);

  std::string ActualKernelName =
      MatmulKernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
  EXPECT(ActualKernelName == KernelName);

  return std::make_tuple(Prog, MatmulKernel);
}

void doFloatMatmul(bool ColumnMajor, unsigned Transpose, unsigned M, unsigned N,
                   unsigned K, std::initializer_list<float> AData, unsigned Lda,
                   std::initializer_list<float> BData, unsigned Ldb,
                   std::vector<float> &Result, unsigned Ldc) {

  float MarkerVal = 9999.0f;
  Result = std::vector<float>((ColumnMajor ? N : M) * Ldc, MarkerVal);

  TensorLayoutBLASPitched ATLayout({ColumnMajor ? 0u : 1u}, {Lda});
  TensorLayoutBLASPitched BTLayout({ColumnMajor ? 0u : 1u}, {Ldb});
  TensorLayoutBLASPitched CTLayout({ColumnMajor ? 0u : 1u}, {Ldc});

  TensorDesc ATDesc({M, K}, CL_TENSOR_DTYPE_FP32_EXP);
  TensorDesc BTDesc({K, N}, CL_TENSOR_DTYPE_FP32_EXP);
  TensorDesc CTDesc({M, N}, CL_TENSOR_DTYPE_FP32_EXP);
  cl_tensor_layout_ml_type_exp MLType =
      ColumnMajor ? CL_TENSOR_LAYOUT_ML_CN_EXP : CL_TENSOR_LAYOUT_ML_NC_EXP;

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

  cl::Program MatmulProg;
  cl::Kernel MatmulKernel;
  std::tie(MatmulProg, MatmulKernel) =
      assertCreateDBK(Ctx, Dev, CL_DBK_MATMUL_EXP, "my_matmul", MatmulAttrs);

  cl_int Status;
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

  Status = CmdQ.enqueueReadBuffer(CTensor, CL_FALSE, 0, CTDesc.getStorageSize(),
                                  Result.data());
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
                  [[maybe_unused]] float Delta = 0.0f) {
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

// Test matmul(<3x2, f16>, <2x4, f16>) -> <3x4, f32>
static void testMatmulF16F32() {
  std::vector<uint16_t> AData = {0x3C00, 0x0000,  // 1, 0
                                 0x0000, 0x4000,  // 0, 2
                                 0x4200, 0x0000}; // 3, 0
  std::vector<uint16_t> BData = {
      0x4900, 0x0000, 0x63D0, 0x0000,  // 10, 0, 1000, 0
      0x0000, 0x5640, 0x0000, 0x70E2}; // 0, 100, 0, 10000
  std::vector<float> Result = std::vector<float>(12, 0.999f);

  TensorDesc ATDesc({3, 2}, CL_TENSOR_DTYPE_FP16_EXP);
  TensorDesc BTDesc({2, 4}, CL_TENSOR_DTYPE_FP16_EXP);
  TensorDesc CTDesc({3, 4}, CL_TENSOR_DTYPE_FP32_EXP);

  TensorLayoutBLAS DL({1u});
  ATDesc.setLayout(DL);
  BTDesc.setLayout(DL);
  CTDesc.setLayout(DL);

  cl_dbk_attributes_matmul_exp MatmulAttrs;
  memcpy(&MatmulAttrs.a, ATDesc.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&MatmulAttrs.b, BTDesc.get(), sizeof(cl_tensor_desc_exp));
  memcpy(&MatmulAttrs.c, CTDesc.get(), sizeof(cl_tensor_desc_exp));
  MatmulAttrs.trans_a = false;
  MatmulAttrs.trans_b = false;
  memset(MatmulAttrs.kernel_props, 0, sizeof(MatmulAttrs.kernel_props));

  cl::Program MatmulProg;
  cl::Kernel MatmulKernel;
  std::tie(MatmulProg, MatmulKernel) = assertCreateDBK(
      Ctx, Dev, CL_DBK_MATMUL_EXP, "matmul_f16_f32", MatmulAttrs);

  cl_int Status;
  auto ATensor = createTensor(Ctx, ATDesc, AData.data(), &Status);
  EXPECT(Status == CL_SUCCESS);
  auto BTensor = createTensor(Ctx, BTDesc, BData.data(), &Status);
  EXPECT(Status == CL_SUCCESS);
  auto CTensor = createTensor(Ctx, CTDesc, Result.data(), &Status);
  EXPECT(Status == CL_SUCCESS);

  MatmulKernel.setArg(0, ATensor);
  MatmulKernel.setArg(1, BTensor);
  MatmulKernel.setArg(2, CTensor);

  Status = CmdQ.enqueueNDRangeKernel(MatmulKernel, cl::NullRange,
                                     cl::NDRange{1, 1}, cl::NullRange);

  EXPECT(Status == CL_SUCCESS);

  Status = CmdQ.enqueueReadBuffer(CTensor, CL_FALSE, 0, CTDesc.getStorageSize(),
                                  Result.data());
  EXPECT(Status == CL_SUCCESS);

  Status = CmdQ.finish();
  EXPECT(Status == CL_SUCCESS);

  check2DSlice(3, 4, Result, 4,
               {10.f, 0.f, 1000.f, 0.f,   //
                0.f, 200.f, 0.f, 20000.f, //
                30.f, 0.f, 3000.f, 0.f},
               4);
}

int main() {
  std::tie(Platform, Dev, DevName) = findDeviceWithMatmulDBK();
  bool isCustom = Dev.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_CUSTOM;

  // TODO we should have getInfo queries to query this information
  bool DeviceIsIntelNPU = DevName.find("AI Boost") != std::string::npos;
  if (isCustom && DeviceIsIntelNPU) {
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

  // CPU driver fails this test.
  if (DeviceIsIntelNPU) {
    std::cout << "--- Matmul 7  ---" << std::endl;
    testMatmulF16F32();
  }

  std::cout << "--- Completed ---" << std::endl;

  return 0;
}
