// OpenCL runtime library: clCreateBuiltinKernelWithAttributesEXP()

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

#include "pocl_cl.h"

#define HAVE_LIBXSMM // TODO: Have CMake define this.
#ifdef HAVE_LIBXSMM
#include <libxsmm.h>
#endif

#include <cstring>
#include <functional>
#include <string>
#include <vector>

// TODO: Move to utils, refactor clCreateKernel().
extern unsigned long kernel_c;
extern "C" void pocl_program_insert_kernel_thsafe(cl_program program,
                                                  cl_kernel kernel) {
  POCL_LOCK_OBJ(program);
  LL_PREPEND(program->kernels, kernel);
  POCL_RETAIN_OBJECT_UNLOCKED(program);
  POCL_UNLOCK_OBJ(program);
  POCL_ATOMIC_INC(kernel_c);
}

class TensorDescView {
  const cl_tensor_desc *OriginalDesc = nullptr;

public:
  TensorDescView(const cl_tensor_desc *TDesc) {
    if (!TDesc || !TDesc->shape)
      return;

    // TODO: assert the data layout is valid.

    OriginalDesc = TDesc;
  }

  bool valid() const noexcept { return OriginalDesc; }
  operator bool() const noexcept { return valid(); }

  size_t rank() const noexcept {
    assert(valid());
    return OriginalDesc->rank;
  }

  size_t operator[](size_t I) const noexcept {
    assert(valid());
    assert(I < rank());
    return OriginalDesc->shape[I];
  }

  bool shapeEquals(const TensorDescView &Other) {
    assert(valid());
    if (!Other.valid() || rank() != Other.rank())
      return false;

    for (unsigned I = 0, E = rank(); I < E; I++)
      if (operator[](I) != Other[I])
        return false;

    return true;
  }

  cl_tensor_datatype dtype() const noexcept {
    assert(valid());
    return OriginalDesc->dtype;
  }

  size_t numElements() const noexcept {
    assert(valid());
    size_t Result = 1;
    for (unsigned I = 0, E = rank(); I < E; I++)
      Result *= operator[](I);
    assert(Result);
    return Result;
  }

  std::string toString() const {
    std::string Result = "(";
    for (unsigned I = 0, E = rank(); I < E; I++) {
      if (I != 0)
        Result += ", ";
      Result += std::to_string(operator[](I));
    }
    // TODO: dtype
    Result += ")";
    // TODO: layout
    return Result;
  }

  // Get the stride for the 'Dim'th (zero-based) leading dimension,
  // measured in tensor elements. Applicable for 2D+ tensors with BLAS
  // datalayout.
  //
  // NthDim can be rank() - 1 or more in which case the result is
  // getBlasStrideInElts(rank - 2) * shape[trailing_dimension].
  size_t getBlasStrideInElts(unsigned Dim) const {
    assert(valid());
    assert(rank() >= 2);
    assert(OriginalDesc->layout && "Does not have data layout!");
    const auto *BaseDL =
        reinterpret_cast<const cl_tensor_layout_base *>(OriginalDesc->layout);
    assert(
        BaseDL->stype == CL_TENSOR_LAYOUT_BLAS &&
        "The method must not be called for tesnors with non-BLAS data layouts");

    const auto &BlasDL =
        *reinterpret_cast<const cl_tensor_layout_blas *>(BaseDL);

    if (Dim < rank() - 1)
      return BlasDL.leading_strides[Dim];

    return BlasDL.leading_strides[rank() - 1] * operator[](
                                                    getTrailingDim(BlasDL));
  }

  bool isBlasRowMajor() const {
    assert(valid());
    assert(OriginalDesc->layout && "Does not have data layout!");
    const auto *BaseDL =
        reinterpret_cast<const cl_tensor_layout_base *>(OriginalDesc->layout);
    assert(
        BaseDL->stype == CL_TENSOR_LAYOUT_BLAS &&
        "The method must not be called for tesnors with non-BLAS data layouts");
    assert(rank() >= 2 && "Not a (batched) matrix!");

    const auto &BlasDL =
        *reinterpret_cast<const cl_tensor_layout_blas *>(BaseDL);
    return BlasDL.leading_dims[0] == (rank() - 1u);
  }

private:
  unsigned getTrailingDim(const cl_tensor_layout_blas &BlasDL) const {
    assert(valid());
    assert(rank() < sizeof(unsigned) * 8u &&
           "Too many dimensions for the bitset.");

    unsigned DimSet = (1u << rank()) - 1;
    for (unsigned I = 0; I < rank() - 1; I++)
      DimSet &= ~(1u << BlasDL.leading_dims[I]);

    assert(__builtin_popcount(DimSet) == 1 && "Invalid data layout?");
    unsigned TrailingDim = __builtin_ctz(DimSet);
    assert(TrailingDim < rank());
    return TrailingDim;
  }
};

static pocl_kernel_metadata_t *getKernelMetadata(cl_program Program,
                                                 const std::string KernelName) {
  for (size_t I = 0, E = Program->num_kernels; I != E; I++) {
    if (KernelName == Program->kernel_meta[I].name)
      return &Program->kernel_meta[I];
  }
  assert(!"Missing kernel metadata!");
}

static void runDBK(_cl_command_node *Cmd, void *Data) {
  auto *Runner = static_cast<std::function<void(_cl_command_node &)> *>(Data);
  (*Runner)(*Cmd);
}

static void releaseDBK(void *Data) {
  if (Data)
    delete static_cast<std::function<void(cl_kernel)> *>(Data);
}

template <typename PtrT>
static PtrT getBufferDataAs(const _cl_command_node &Cmd, unsigned ArgIdx) {
  const auto &Arg = Cmd.command.run.arguments[ArgIdx];
  if (Arg.is_raw_ptr)
    return static_cast<PtrT>(Arg.value);

  auto Mem = *static_cast<cl_mem *>(Arg.value);
  auto *Ptr =
      static_cast<char *>(Mem->device_ptrs[Cmd.device->global_mem_id].mem_ptr);
  Ptr += Arg.offset;
  return reinterpret_cast<PtrT>(Ptr);
}

static cl_int createGemmKernel(cl_program Program, bool TransposeA,
                               bool TransposeB, TensorDescView A,
                               TensorDescView B, TensorDescView *CIOpt,
                               TensorDescView CO, const void *AlphaAttr,
                               const void *BetaAttr, cl_kernel &Kernel) {
  assert(!Kernel && "Kernel handle is already populated!");

  auto LogError = [](cl_int ErrorCode, const std::string Msg) -> cl_int {
    if (!Msg.empty())
      POCL_MSG_ERR("%s", Msg.c_str());
    return ErrorCode;
  };

  if (!A || !B || !CO)
    return LogError(CL_DBK_INVALID_ATTRIBUTE, "Null attribute.");

  // TBC: 4D+ tensor could be supported by treating the additional
  //      dimensions as batch dimensions - but it might not be
  //      worthwhile due the extra work to support them and processing
  //      overhead they may impose.
  if (A.rank() > 3)
    return LogError(CL_DBK_INVALID_ATTRIBUTE,
                    "Unsupported high-degree tensors.");

  if (A.rank() != B.rank() || B.rank() != CO.rank())
    // TODO: Should we have something like CL_DBK_INVALID_TENSOR_SHAPE?
    return LogError(CL_DBK_INVALID_ATTRIBUTE, "Rank mismatch.");

  if (CIOpt && !CIOpt->shapeEquals(CO))
    return LogError(CL_DBK_INVALID_ATTRIBUTE,
                    "Tensor shape mismatch between c_in and c_out.");

  // FIXME: check tensor shapes are correct respect to the transpose
  //        configurations.

  size_t BatchDims = A.rank() - 2;

  // CO[b][m][n] = sigma_over_m_n_k(A[b][m][k] * B[b][k][n]) + CI[b][m][n].
  auto Am = A[BatchDims + 0];
  auto Ak = A[BatchDims + 1];
  auto Bk = B[BatchDims + 0];
  auto Bn = B[BatchDims + 1];
  auto COm = CO[BatchDims + 0];
  auto COn = CO[BatchDims + 1];

  if (Ak != Bk || Am != COm || Bn != COn)
    // TODO: Should have more descriptive error code? Or would it be better
    //       to have error logging like the cl_program has for building?
    return LogError(CL_DBK_INVALID_ATTRIBUTE, "Matrix shape mismatch.");

  // Check batch dimensions match.
  size_t BatchSize = A.rank() == 3 ? A[0] : 1;
  if (BatchSize > 1 && (BatchSize != B[0] || B[0] != CO[0]))
    return LogError(CL_DBK_INVALID_ATTRIBUTE, "Batch size mismatch.");

  if (BatchSize > 1 && CIOpt && (*CIOpt)[0] != CO[0])
    return LogError(CL_DBK_INVALID_ATTRIBUTE, "Batch size mismatch.");

  if (A.dtype() != B.dtype())
    return LogError(CL_DBK_INVALID_ATTRIBUTE,
                    "dtype mismatch between A and B.");

  if (CIOpt && CIOpt->dtype() != CO.dtype())
    return LogError(CL_DBK_INVALID_ATTRIBUTE,
                    "dtype mismatch between input and output C.");

  // TODO: We probably need to have support for mixed input/output
  // precisions to be able to fit results of large, low precision input
  // matrices. precision inputs. E.g.
  //
  //  * i8 x i8   --> i32
  //  * f16 x f16 --> f32
  if (A.dtype() != CO.dtype())
    return LogError(CL_DBK_INVALID_ATTRIBUTE, "Unsupported I/O dtype");

  // TODO: extend support for other data types.
  if (A.dtype() != CL_TENSOR_FLOAT)
    return LogError(CL_DBK_UNAVAILABLE, "Unimplemented dtype support.");

  // TODO: check validity of data layouts of the tensors. Now assume
  // they are correct and they are using BLAS-like layout.

  float Alpha = 1.0f, Beta = 0.0f;
  if (AlphaAttr)
    std::memcpy(&Alpha, AlphaAttr, sizeof(float));
  if (BetaAttr)
    std::memcpy(&Beta, BetaAttr, sizeof(float));

  // libxsmm does not support arbitrary alpha and beta (for now).
  // [https://github.com/libxsmm/libxsmm/wiki/Development#longer-term-issues].
  if (Alpha != 1.0f || !(Beta == 0.0f || Beta == 1.0f))
    LogError(CL_DBK_UNAVAILABLE,
             "UNIMPLEMENTED: arbitrary alpha and beta attributes");

  // Attributes seems to be correct - proceed to create a matmul/gemm
  // implementation.

  Kernel = (_cl_kernel *)std::calloc(1, sizeof(_cl_kernel));
  if (!Kernel)
    return LogError(CL_OUT_OF_HOST_MEMORY,
                    "Couldn't allocate storage for cl_kernel!");
  POCL_INIT_OBJECT(Kernel);
  Kernel->meta = getKernelMetadata(Program, CIOpt ? "khr_gemm" : "khr_matmul");
  Kernel->data = (void **)calloc(Program->num_devices, sizeof(void *));
  // TODO: Emit unique name for each unique DBK instance as debugging aid.
  // TODO: Does .name claim ownership?
  Kernel->name = "a_pocl_gemm_impl";
  Kernel->context = Program->context;
  Kernel->program = Program;

  assert(Kernel->meta->num_args == (3 + !!CIOpt));
  auto ArgSpace = static_cast<pocl_argument *>(
      calloc(Kernel->meta->num_args, sizeof(pocl_argument)));
  Kernel->dyn_arguments = ArgSpace;

  size_t Lda = A.getBlasStrideInElts(0);
  size_t Ldb = B.getBlasStrideInElts(0);
  size_t Ldc = CO.getBlasStrideInElts(0);
  size_t ABatchStrideInElts = A.getBlasStrideInElts(1);
  size_t BBatchStrideInElts = B.getBlasStrideInElts(1);
  size_t CBatchStrideInElts = CO.getBlasStrideInElts(1);

#ifdef HAVE_LIBXSMM
  // libxsmm expects data in column-major format but we can feed it
  // row-major data by transposing the inputs and and the output.
  bool LibTransposeA = TransposeA ^ A.isBlasRowMajor();
  bool LibTransposeB = TransposeB ^ B.isBlasRowMajor();
  int Flags = (LibTransposeA ? LIBXSMM_GEMM_FLAG_TRANS_A : 0) |
              (LibTransposeB ? LIBXSMM_GEMM_FLAG_TRANS_B : 0);

  std::function<void(float *Dst, const _cl_command_node &Cmd, size_t BatchNum)>
      LoadCBatch;
  std::function<void(float *Dst, const _cl_command_node &Cmd, size_t BatchNum)>
      StoreCBatch;

  if (CIOpt && Beta != 0.0f) {
    if (CIOpt->isBlasRowMajor()) {
      // Need to convert C input to column-major.
      LoadCBatch = [=](float *Batch, const _cl_command_node &Cmd,
                       size_t BatchNum) -> void {
        auto *CIData = getBufferDataAs<float *>(Cmd, 2);
        auto *Src = &CIData[BatchNum * CBatchStrideInElts];
        libxsmm_otrans(Batch, Src, sizeof(float), COm, COn, Ldc, COm);
      };
    } else {
      LoadCBatch = [=](float *Batch, const _cl_command_node &Cmd,
                       size_t BatchNum) -> void {
        auto *CIData = getBufferDataAs<float *>(Cmd, 2);
        auto *Src = &CIData[BatchNum * CBatchStrideInElts];
        libxsmm_matcopy(Batch, Src, sizeof(float), COm, COn, Ldc, COm);
      };
    }
  } else {
    LoadCBatch = [=](float *Batch, const _cl_command_node &Cmd,
                     size_t BatchNum) -> void {
      // Zero-initialize.
      libxsmm_matcopy(Batch, nullptr, sizeof(float), COm, COn, Ldc, COm);
    };
  }

  unsigned COKernelArgIdx = 2 + !!CIOpt;
  if (CO.isBlasRowMajor()) {
    // Results are always in column-major.
    StoreCBatch = [=](float *Batch, const _cl_command_node &Cmd,
                      size_t BatchNum) -> void {
      auto *COData = getBufferDataAs<float *>(Cmd, COKernelArgIdx);
      auto *Dst = &COData[BatchNum * CBatchStrideInElts];
      libxsmm_otrans(Dst, Batch, sizeof(float), COm, COn, COm, Ldc);
    };
  } else {
    StoreCBatch = [=](float *Batch, const _cl_command_node &Cmd,
                      size_t BatchNum) -> void {
      auto *COData = getBufferDataAs<float *>(Cmd, COKernelArgIdx);
      auto *Dst = &COData[BatchNum * CBatchStrideInElts];
      libxsmm_matcopy(Dst, Batch, sizeof(float), COm, COn, COm, Ldc);
    };
  }

  if (auto MatmulInstance = libxsmm_mmfunction<float>(Flags, COm, COn, Ak, Lda,
                                                      Ldb, COm, Alpha, Beta)) {
    auto *RunnerData = new std::function<void(_cl_command_node &)>(
        [=](_cl_command_node &Cmd) -> void {
          auto *AData = getBufferDataAs<float *>(Cmd, 0);
          auto *BData = getBufferDataAs<float *>(Cmd, 1);

          // TODO: Optimization: There is codegen for batched matmul
          //       in libxsmm we could use.
          std::vector<float> CTemp(COm * COn, 0.0f);
          for (size_t Batch = 0; Batch < BatchSize; Batch++) {
            LoadCBatch(CTemp.data(), Cmd, Batch);
            MatmulInstance(&AData[Batch * ABatchStrideInElts],
                           &BData[Batch * BBatchStrideInElts], CTemp.data());
            StoreCBatch(CTemp.data(), Cmd, Batch);
          }
        });

    Kernel->custom_runner = runDBK;
    Kernel->custom_runner_data = static_cast<void *>(RunnerData);
    Kernel->release_custom_runner_data = releaseDBK;
    pocl_program_insert_kernel_thsafe(Program, Kernel);

    return CL_SUCCESS;
  }
#endif // HAVE_LIBXSMM

  free(Kernel);
  free(ArgSpace);
  return LogError(CL_DBK_UNAVAILABLE, "Unsupported matmul/gemm configuration.");
}

static cl_int implementation(cl_kernel &Kernel, cl_program Program,
                             const char *KernelName,
                             const void *KernelAttributes) noexcept try {

  if (!KernelName)
    return CL_INVALID_VALUE;

  if (!IS_CL_OBJECT_VALID(Program))
    return CL_INVALID_PROGRAM;

  assert(Program->num_devices != 0);
  assert(Program->num_builtin_kernels > 0);
  assert(Program->concated_builtin_names);

  std::string BiKNames(Program->concated_builtin_names);
  if (BiKNames.find(KernelName) == std::string::npos)
    return CL_INVALID_KERNEL_NAME;

  if (std::string(KernelName) == "khr_gemm") {
    auto &Attrs =
        *static_cast<const _cl_dbk_attributes_khr_gemm *>(KernelAttributes);

    if (!Attrs.alpha || !Attrs.beta)
      return CL_DBK_INVALID_ATTRIBUTE;

    // TODO: check alpha and beta values are sensible (e.g. not NaNs
    //       or infinities).

    TensorDescView A(Attrs.a);
    TensorDescView B(Attrs.b);
    TensorDescView CI(Attrs.c_in);
    TensorDescView CO(Attrs.c_out);
    return createGemmKernel(Program, Attrs.trans_a, Attrs.trans_b, A, B, &CI,
                            CO, Attrs.alpha, Attrs.beta, Kernel);
  }

  if (std::string(KernelName) == "khr_matmul") {
    auto &Attrs =
        *static_cast<const _cl_dbk_attributes_khr_matmul *>(KernelAttributes);
    TensorDescView A(Attrs.a);
    TensorDescView B(Attrs.b);
    TensorDescView CO(Attrs.c);
    return createGemmKernel(Program, Attrs.trans_a, Attrs.trans_b, A, B,
                            nullptr, CO, nullptr, nullptr, Kernel);
  }

  return CL_DBK_UNAVAILABLE;
} catch (std::bad_alloc &) {
  POCL_MSG_ERR("Caught std::bad_alloc");
  return CL_OUT_OF_HOST_MEMORY;
} catch (...) {
  assert(!"Caught unhandled exception!");
  return CL_OUT_OF_HOST_MEMORY;
}

extern "C" CL_API_ENTRY cl_kernel CL_API_CALL
POname(clCreateBuiltinKernelWithAttributesEXP)(
    cl_program Program, const char *KernelName, const void *KernelAttributes,
    cl_int *ErrCodeRet) CL_API_SUFFIX__VERSION_3_0 {

  cl_kernel Kernel = nullptr;
  cl_int ErrCode =
      implementation(Kernel, Program, KernelName, KernelAttributes);
  if (ErrCodeRet)
    *ErrCodeRet = ErrCode;
  return Kernel;
}

POsym(clCreateBuiltinKernelWithAttributesEXP)
