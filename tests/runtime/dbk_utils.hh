/**
 * \brief OpenCL runtime library: useful C++ utility functions for
 * handling DBKs.
 *
 * Copyright (c) 2025 Henry Linjamäki / Intel Finland Oy
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * \file
 */

#ifndef DBK_UTILS_HPP
#define DBK_UTILS_HPP

#include "poclu.h"

#include <CL/opencl.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <variant>
#include <vector>

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

  /// Returns tensor size in elements.
  size_t getSize() const noexcept { return LeadingStrides.back(); }
};

/// Wraps cl_tensor_desc_exp and provides utility functions for it.
class TensorDesc {
  std::vector<cl_tensor_shape_exp> Shape;
  cl_tensor_desc_exp Desc;
  std::variant<TensorLayoutBLAS, TensorLayoutBLASPitched,
               cl_tensor_layout_ml_exp>
      Layout;
  size_t StorageSize;

public:
  /// Creates tensor description with opaque data layout.
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

  unsigned dimSize(int Dim) const {
    if (Dim < 0)
      Dim = rank() + Dim;
    assert(Dim < rank());
    return Shape[Dim];
  }

  /// Returns size of the elements in bytes. Sub-byte elements types
  /// will report one byte.
  unsigned elementSize() const noexcept {
    switch (Desc.dtype) {
    case CL_TENSOR_DTYPE_INT64_EXP:
    case CL_TENSOR_DTYPE_FP64_EXP:
      return 8;
    case CL_TENSOR_DTYPE_INT32_EXP:
    case CL_TENSOR_DTYPE_FP32_EXP:
      return 4;
    case CL_TENSOR_DTYPE_INT16_EXP:
    case CL_TENSOR_DTYPE_FP16_EXP:
      return 2;
    default:
      assert(false && "Unknown element type!");
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

inline cl::Buffer createTensor(cl::Context &Ctx, const TensorDesc &TDesc,
                               const void *HostPtr = nullptr,
                               cl_int *Status = nullptr) {
  cl_mem_properties Props[] = {
      CL_MEM_TENSOR_EXP, reinterpret_cast<cl_mem_properties>(TDesc.get()), 0};

  size_t BufSize = TDesc.getStorageSize();
  return cl::Buffer(clCreateBufferWithProperties(
      Ctx.get(), Props, (HostPtr ? CL_MEM_COPY_HOST_PTR : 0),
      // TBC: update spec so that zero means the buffer size is inferred from
      // the tensor description?
      BufSize, const_cast<void *>(HostPtr), Status));
}

/// Utility function for creating a program with single DBK for the
/// given device with assumption that creation will succeed.
template <typename DbkAttrT>
std::tuple<cl::Program, cl::Kernel>
assertCreateDBK(cl::Context Ctx, cl::Device Device, cl_dbk_id_exp DbkID,
                const std::string &KernelName, DbkAttrT &Attributes) {

  auto Platform = Device.getInfo<CL_DEVICE_PLATFORM>();
  auto createProgramWithDBKs =
      reinterpret_cast<clCreateProgramWithDefinedBuiltInKernelsEXP_fn>(
          clGetExtensionFunctionAddressForPlatform(
              Platform(), "clCreateProgramWithDefinedBuiltInKernelsEXP"));
  TEST_ASSERT(createProgramWithDBKs != nullptr);

  cl_int Status;
  cl_device_id Devices[1] = {Device()};
  cl_dbk_id_exp IDs[1] = {DbkID};
  const char *Names[1] = {KernelName.c_str()};
  cl_int DeviceStatus[1] = {0};
  DbkAttrT *Attrs[1] = {&Attributes};
  cl_program ProgHandle =
      createProgramWithDBKs(Ctx(), 1, Devices, 1, IDs, Names,
                            (const void **)Attrs, DeviceStatus, &Status);
  TEST_ASSERT(Status == CL_SUCCESS);
  cl::Program Prog(ProgHandle);

  Status = Prog.build();
  TEST_ASSERT(Status == CL_SUCCESS);

  auto MatmulKernel = cl::Kernel(Prog, KernelName, &Status);
  TEST_ASSERT(Status == CL_SUCCESS);

  std::string ActualKernelName =
      MatmulKernel.getInfo<CL_KERNEL_FUNCTION_NAME>();
  TEST_ASSERT(ActualKernelName == KernelName);

  return std::make_tuple(Prog, MatmulKernel);
}

/// Same as assertCreateDBK() but creates multiple DBKs at once.
std::tuple<cl::Program, std::vector<cl::Kernel>> assertCreateDBKs(
    cl::Context Ctx, cl::Device Device,
    std::initializer_list<std::tuple<cl_dbk_id_exp, std::string, const void *>>
        DBKs) {

  auto Platform = Device.getInfo<CL_DEVICE_PLATFORM>();
  auto createProgramWithDBKs =
      reinterpret_cast<clCreateProgramWithDefinedBuiltInKernelsEXP_fn>(
          clGetExtensionFunctionAddressForPlatform(
              Platform(), "clCreateProgramWithDefinedBuiltInKernelsEXP"));
  TEST_ASSERT(createProgramWithDBKs != nullptr);

  cl_int Status;
  cl_device_id Devices[1] = {Device()};

  std::vector<cl_dbk_id_exp> IDs;
  std::vector<const char *> Names;
  std::vector<const void *> Attrs;
  for (auto &DBK : DBKs) {
    IDs.push_back(std::get<0>(DBK));
    Names.push_back(std::get<1>(DBK).c_str());
    Attrs.push_back(std::get<2>(DBK));
  }

  cl_int DeviceStatus[1] = {0};
  cl_program ProgHandle =
      createProgramWithDBKs(Ctx(), 1, Devices, DBKs.size(), IDs.data(),
                            Names.data(), Attrs.data(), DeviceStatus, &Status);
  TEST_ASSERT(Status == CL_SUCCESS);
  cl::Program Prog(ProgHandle);

  Status = Prog.build();
  TEST_ASSERT(Status == CL_SUCCESS);

  std::vector<cl::Kernel> Kernels;
  for (auto *Name : Names) {
    Kernels.emplace_back(Prog, Name, &Status);
    TEST_ASSERT(Status == CL_SUCCESS);

    std::string ActualKernelName =
        Kernels.back().getInfo<CL_KERNEL_FUNCTION_NAME>();
    TEST_ASSERT(ActualKernelName == Name);
  }

  return std::make_tuple(Prog, Kernels);
}

inline bool deviceHasDBK(cl::Device Dev, const std::string &DBK) {
  std::string DBKs = Dev.getInfo<CL_DEVICE_BUILT_IN_KERNELS>();
  auto Pos = DBKs.find(DBK);
  if (Pos == std::string::npos)
    return false;

  // Check we have a full match.
  if (Pos && DBKs[Pos - 1] != ';')
    return false;

  auto EndPos = Pos + DBK.size();
  if (EndPos < DBKs.size() && DBKs[EndPos] != ';')
    return false;

  return true;
}

inline std::tuple<cl::Platform, cl::Device, std::string>
findDeviceWithDBK(const std::string &DBK) noexcept {
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
      std::string DeviceName = D.getInfo<CL_DEVICE_NAME>();
      if (Exts.find("cl_exp_defined_builtin_kernels") == std::string::npos) {
        std::cerr << "Device " << DeviceName
                  << " does not support cl_exp_defined_builtin_kernels\n";
        continue;
      }

      if (deviceHasDBK(D, DBK)) {
        std::cerr << "Selected device: " << D.getInfo<CL_DEVICE_NAME>() << "\n";
        return std::make_tuple(P, D, DeviceName);
      }

      std::cerr << "Device " << D.getInfo<CL_DEVICE_NAME>()
                << " does not support BKD '" << DBK << "'.\n";
    }
  }

  std::cerr << "No suitable device found\n";
  std::exit(77);
}

#endif // DBK_UTILS_HPP
