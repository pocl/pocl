/*******************************************************************************
 * Copyright (c) 2022-2024 Henry Linjamäki, Michal Babej / Intel Finland Oy
 *
 * PoCL-specific proof-of-concept (draft) of Defined Builtin Kernels extension.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
 * KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
 * SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
 *    https://www.khronos.org/registry/
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/


#ifndef OPENCL_EXP_TENSOR_H
#define OPENCL_EXP_TENSOR_H
#include <CL/cl.h>

/* Based on spec v0.2.0
 * https://github.com/KhronosGroup/OpenCL-Docs/pull/1006
 */

/* types for describing dimensions & stride */
typedef cl_ulong cl_tensor_shape_exp;
typedef cl_ulong cl_tensor_stride_exp;
typedef cl_uint cl_tensor_dim_exp;

typedef cl_uint cl_tensor_datatype_exp;
/* cl_tensor_datatype values. List of the datatypes for Tensor data */
/* NOTE: CL_TENSOR_DTYPE_UNKNOWN is not in the spec (v0.2.0). Preserved for
 * internal use. */
#define CL_TENSOR_DTYPE_UNKNOWN 0

#define CL_TENSOR_DTYPE_UINT4_EXP 1
#define CL_TENSOR_DTYPE_UINT8_EXP 2
#define CL_TENSOR_DTYPE_UINT16_EXP 3
#define CL_TENSOR_DTYPE_UINT32_EXP 4
#define CL_TENSOR_DTYPE_UINT64_EXP 5

#define CL_TENSOR_DTYPE_INT4_EXP 6
#define CL_TENSOR_DTYPE_INT8_EXP 7
#define CL_TENSOR_DTYPE_INT16_EXP 8
#define CL_TENSOR_DTYPE_INT32_EXP 9
#define CL_TENSOR_DTYPE_INT64_EXP 10

/* https://discourse.llvm.org/t/rethink-on-approach-to-low-precision-fp-types/
   suggests that there are lot more floating-point types. Perhaps,
   tensor data types should be represented as a bitfield. E.g.
   cl_tensor_datatype_exp fp8e4m3 = CL_FLOAT_KIND | CL_EXPONENT_BITS_4 |
   CL_MANTISSA_BITS_3? */
#define CL_TENSOR_DTYPE_FP8E4M3_EXP 11
#define CL_TENSOR_DTYPE_FP8E5M2_EXP 12
#define CL_TENSOR_DTYPE_FP16_EXP 13
#define CL_TENSOR_DTYPE_FP32_EXP 14
#define CL_TENSOR_DTYPE_FP64_EXP 15

#define CL_TENSOR_DTYPE_BFLOAT16_EXP 16

#define CL_TENSOR_DTYPE_COMPLEX64_EXP 17
#define CL_TENSOR_DTYPE_COMPLEX128_EXP 18

/* NOTE: CL_TENSOR_DTYPELAST is not in the spec (v0.2.0). Preserved for
 * internal use. */
#define CL_TENSOR_DTYPE_LAST 19

/* cl_tensor_layout_type describes the type of layout struct in cl_tensor_desc */
typedef cl_uint cl_tensor_layout_type_exp;

/* CL_TENSOR_LAYOUT_NONE: corresponding layout pointer is expected to be NULL;
 * TODO: does this make sense ? the idea is that the driver
 * could pick the "optimal" storage mode for a Tensor. However,
 * this would likely result in extra memcpy (when the user data is copied
 * to driver storage). */
#define CL_TENSOR_LAYOUT_OPAQUE_EXP 0
/* corresponding layout pointer points to struct cl_tensor_layout_blas_exp */
#define CL_TENSOR_LAYOUT_BLAS_EXP 1
/* corresponding layout pointer points to struct
 * cl_tensor_layout_blas_pitched_exp */
/* NOTE: This layout variant is considered to be removed due to lack of use
 *       cases.  */
#define CL_TENSOR_LAYOUT_BLAS_PITCHED_EXP 2
/* corresponding layout pointer points to struct cl_tensor_layout_ml */
#define CL_TENSOR_LAYOUT_ML_EXP 3

/* cl_tensor_properties for setting extra properties on a cl_mem Tensor.
 * Tensor with a Mutable property means that a clSetKernelArg on DBK with
 * a Tensor cl_mem argument should succeed even if the new Tensor given is
 * different in the respective property (dims,dtype etc) than the original
 * cl_tensor_desc given at clCreateProgramWithDBKs() time.
 * Devices that don't support Mutable attributes should return an error
 * at Program build time. */
typedef cl_properties cl_tensor_properties_exp;

/* NOTE: This is not in the spec (v0.2.0). Preserved for internal use. */
#define CL_TENSOR_PROPERTY_NONE_EXP 0
/* allow the tensor to be mutable with respect to shape */
#define CL_TENSOR_PROPERTY_MUTABLE_SHAPE_EXP 1
/* allow the tensor to be mutable with respect to data types */
#define CL_TENSOR_PROPERTY_MUTABLE_DTYPE_EXP 2
/* allow the tensor to be mutable with respect to layout */
#define CL_TENSOR_PROPERTY_MUTABLE_LAYOUT_EXP 3

/*********************** Additions to cl_mem_object_type ********************/

/* A clCreateBufferWithProperties() property value that implies next
 * property is a pointer to cl_tensor_desc. */
#define CL_MEM_TENSOR_EXP 0x8000

/* TBC: A clCreateSubBuffer() cl_buffer_create_type used for creating a
 * subtensor for the purpose of:
 * coercing an existing buffer (non-tensor) to a tensor (or part of it).
 * splitting large tensors to smaller ones.
 * reshaping existing tensor to another.
 * coercing data type of an existing tensor to other type of same size.
 #define CL_MEM_TENSOR_VIEW_EXP 0x8001 */

/* Maximum tensor rank that can be used in the structs defined below */
/* TODO is there a good reason to make this non-fixed size */
#define CL_MEM_MAX_TENSOR_RANK_EXP 20

/* Maximum number of tensor properties */
#define CL_MAX_TENSOR_PROPERTIES_EXP 16

/* cl_tensor_desc is a struct that must be passed with CL_MEM_TENSOR in
 * properties array of clCreateBufferWithProperties() */
typedef struct _cl_tensor_desc_exp
{
  /* The rank of the tensor. <= CL_MEM_MAX_TENSOR_RANK */
  cl_uint rank;

  /* The element type of the tensor. */
  cl_tensor_datatype_exp dtype;
  /* 0-terminated array of Tensor properties */
  cl_tensor_properties_exp properties[CL_MAX_TENSOR_PROPERTIES_EXP];

  /* The shape of the tensor described by an array. Describes number
   * of elements in the tensor dimensions starting with "outermost"
   * dimension first. E.g. {..., NumOf2DBlocks, NumOf1DBlocks,
   * NumEltsIn1D}.  (This convention is tentatively chosen for
   * matching python, numpy and popular ML frameworks).
   *
   * Conditions:
   * * length of the array must be at least <rank> elements.
   * * TBC: A dimension can be zero meaning the size is unspecified. However,
   *   commands involving tensors must have fully specified shape. */
  cl_tensor_shape_exp shape[CL_MEM_MAX_TENSOR_RANK_EXP];

  /* Optional data layout description. Must be NULL or one of
   * cl_tensor_layout_* structures in the below.
   *
   * TBD: If NULL, cl{Enqueue,Command}{Read,Write}Tensor must be
   * used for transferring data from or to tensor. If a pointer to the
   * tensor data is aquired (somehow), dereferencing that pointer is
   * undefined behavior. */
  const void *layout;
  cl_tensor_layout_type_exp layout_type;

} cl_tensor_desc_exp;

/* Describes Tensor data layout using terms related to BLAS APIs. */
typedef struct _cl_tensor_layout_blas_pitched_exp
{
  /* Leading tensor dimensions. This describes which elements along
   * tensor dimensions are laid out first in the memory. Tensor
   * coodrinates (tensor_coords = {x0, x1, ..., x2}) map to buffer
   * (buffer_offset) as follows:
   *
   *   size_t index = 0;
   *   for (unsigned i = 0; i < tensor_rank; i++) {
   *      index += tensor_coords[leading_dims[i]] * leading_strides[i];
   *   size_t buffer_offset = index;
   *
   * Conditions:
   *
   * * Array length must be at least 'tensor_rank - 1' (last dimension
   *   is implied)
   *
   * * Each tensor dimension 0..<tensor_rank - 1> must appear once in
   *   the array. */
  cl_tensor_dim_exp leading_dims[CL_MEM_MAX_TENSOR_RANK_EXP];

  /* Strides of the leading dimensions. Array length must be at least
   * (tensor_rank - 1) and following assertion must hold:
   *
   *   for (unsigned i = 0; i < tensor_rank - 1; i++) {
   *     size_t tensor_slice_size = 1;
   *     for (unsigned j = 0; j <= i; j++)
   *       tensor_slice_size *= tensor_shape[j];
   *     assert(leading_dims[i] >= tensor_slize_size);
   *   }
   *
   * TBC: Allow leading_strides == 0 (undefined) in which case the
   *      tensor data is non-strided (e.g. for matrices with no gaps
   *      between columns/rows) for convenience? */
  cl_tensor_stride_exp leading_strides[CL_MEM_MAX_TENSOR_RANK_EXP];

  /* TBC: This field specifies an optional alignment guarantee for the
   * first element (an element at coordinate = (0, 0, ..., 0)). The
   * value must be 0 or power-of-two. If zero, the alignment inferred from
   * the dtype. This could also be a layout extension.
   *size_t base_alignment; */

} cl_tensor_layout_blas_pitched_exp;

/* Same as cl_tensor_layout_blas_pitched_exp but does not have member
   for describing pitches/strides. */
typedef struct _cl_tensor_layout_blas_exp
{
  cl_tensor_dim_exp leading_dims[CL_MEM_MAX_TENSOR_RANK_EXP];
} cl_tensor_layout_blas_exp;

/* Describes the Tensor data layout using terms related to ML models.
 * (These can be also expressed with BLAS layout type) */
typedef cl_uint cl_tensor_layout_ml_type_exp;
/* NOTE: ..._ML_UNKNOWN not in the spec (v0.2.0). Preserved for internal
 * use.  */
#define CL_TENSOR_LAYOUT_ML_UNKNOWN 0
#define CL_TENSOR_LAYOUT_ML_C_EXP 1
#define CL_TENSOR_LAYOUT_ML_NC_EXP 2
#define CL_TENSOR_LAYOUT_ML_CN_EXP 3
#define CL_TENSOR_LAYOUT_ML_HW_EXP 4
#define CL_TENSOR_LAYOUT_ML_CHW_EXP 5
#define CL_TENSOR_LAYOUT_ML_NCHW_EXP 6
#define CL_TENSOR_LAYOUT_ML_NHWC_EXP 7
/* NOTE: ..._ML_LAST not in the spec (v0.2.0). Preserved for internal
 * use.  */
#define CL_TENSOR_LAYOUT_ML_LAST 8

typedef struct _cl_tensor_layout_ml_exp
{
  cl_tensor_layout_ml_type_exp ml_type;
} cl_tensor_layout_ml_exp;

#endif /* OPENCL_EXP_TENSOR_H */
