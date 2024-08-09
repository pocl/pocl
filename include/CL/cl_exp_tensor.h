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

/* types for describing dimensions & stride */
typedef cl_ulong cl_tensor_shape;
typedef cl_ulong cl_tensor_stride;
typedef cl_uint cl_tensor_dim;

typedef cl_uint cl_tensor_datatype;
/* cl_tensor_datatype values. List of the datatypes for Tensor data */
#define CL_TENSOR_DTYPE_UNKNOWN 0
#define CL_TENSOR_DTYPE_FP64 1
#define CL_TENSOR_DTYPE_INT64 2
#define CL_TENSOR_DTYPE_UINT64 3
#define CL_TENSOR_DTYPE_FP32 4
#define CL_TENSOR_DTYPE_INT32 5
#define CL_TENSOR_DTYPE_UINT32 6
#define CL_TENSOR_DTYPE_FP16 7
#define CL_TENSOR_DTYPE_INT16 8
#define CL_TENSOR_DTYPE_UINT16 9
#define CL_TENSOR_DTYPE_FP8 10
#define CL_TENSOR_DTYPE_INT8 11
#define CL_TENSOR_DTYPE_UINT8 12
#define CL_TENSOR_DTYPE_INT4 13
#define CL_TENSOR_DTYPE_UINT4 14
#define CL_TENSOR_DTYPE_LAST 15

/* cl_tensor_datatype_value is used to pass POD data to DBKs (like Alpha & Beta
 * parameters to the GEMM); the actual type used is implied. In case of
 * GEMM, the datatype is implied the same as COut Tensor datatype.
 * TODO: this could be done with passing a void* pointer, however
 * this avoids the need for malloc & memcpy */
typedef union
{
  cl_char c;
  cl_short s;
  cl_int i;
  cl_long l;
  cl_half h;
  cl_float f;
  cl_double d;
  char raw[sizeof (cl_double)];
} cl_tensor_datatype_value;

/* cl_tensor_layout_type describes the type of layout struct in cl_tensor_desc */
typedef cl_uint cl_tensor_layout_type;

/* CL_TENSOR_LAYOUT_NONE: corresponding layout pointer is expected to be NULL;
 * TODO: does this make sense ? the idea is that the driver
 * could pick the "optimal" storage mode for a Tensor. However,
 * this would likely result in extra memcpy (when the user data is copied
 * to driver storage). */
#define CL_TENSOR_LAYOUT_NONE 0
/* corresponding layout pointer points to struct cl_tensor_layout_blas */
#define CL_TENSOR_LAYOUT_BLAS 1
/* corresponding layout pointer points to struct cl_tensor_layout_ml */
#define CL_TENSOR_LAYOUT_ML 2


/* cl_tensor_properties for setting extra properties on a cl_mem Tensor.
 * Tensor with a Mutable property means that a clSetKernelArg on DBK with
 * a Tensor cl_mem argument should succeed even if the new Tensor given is
 * different in the respective property (dims,dtype etc) than the original
 * cl_tensor_desc given at clCreateProgramWithDBKs() time.
 * Devices that don't support Mutable attributes should return an error
 * at Program build time. */
typedef cl_properties cl_tensor_properties;

#define CL_TENSOR_PROPERTY_NONE 0
/* allow the tensor to be mutable with respect to dimensions */
#define CL_TENSOR_PROPERTY_MUTABLE_DIMS 1
/* allow the tensor to be mutable with respect to data types */
#define CL_TENSOR_PROPERTY_MUTABLE_DTYPE 2
/* allow the tensor to be mutable with respect to layout */
#define CL_TENSOR_PROPERTY_MUTABLE_LAYOUT 3


/*********************** Additions to cl_mem_object_type ********************/

/* A clCreateBufferWithProperties() property value that implies next
 * property is a pointer to cl_tensor_desc. */
#define CL_MEM_TENSOR 0x8000

/* TBC: A clCreateSubBuffer() cl_buffer_create_type used for creating a
 * subtensor for the purpose of:
 * coercing an existing buffer (non-tensor) to a tensor (or part of it).
 * splitting large tensors to smaller ones.
 * reshaping existing tensor to another.
 * coercing data type of an existing tensor to other type of same size.
 #define CL_MEM_TENSOR_VIEW 0x8001 */

/* Maximum tensor rank that can be used in the structs defined below */
/* TODO is there a good reason to make this non-fixed size */
#define CL_MEM_MAX_TENSOR_RANK 20

/* Maximum number of tensor properties */
#define CL_MAX_TENSOR_PROPERTIES 16

/* cl_tensor_desc is a struct that must be passed with CL_MEM_TENSOR in
 * properties array of clCreateBufferWithProperties() */
typedef struct _cl_tensor_desc
{
  /* The rank of the tensor. <= CL_MEM_MAX_TENSOR_RANK */
  cl_uint rank;

  /* The element type of the tensor. */
  cl_tensor_datatype dtype;
  /* 0-terminated array of Tensor properties */
  cl_tensor_properties properties[CL_MAX_TENSOR_PROPERTIES];

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
  cl_tensor_shape shape[CL_MEM_MAX_TENSOR_RANK];

  /* Optional data layout description. Must be NULL or one of
   * cl_tensor_layout_* structures in the below.
   *
   * TBD: If NULL, cl{Enqueue,Command}{Read,Write}Tensor must be
   * used for transferring data from or to tensor. If a pointer to the
   * tensor data is aquired (somehow), dereferencing that pointer is
   * undefined behavior. */
  const void *layout;
  cl_tensor_layout_type layout_type;

} cl_tensor_desc;

/* Describes Tensor data layout using terms related to BLAS APIs. */
typedef struct _cl_tensor_layout_blas
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
  cl_tensor_dim leading_dims[CL_MEM_MAX_TENSOR_RANK];

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
  cl_tensor_stride leading_strides[CL_MEM_MAX_TENSOR_RANK];

  /* TBC: This field specifies an optional alignment guarantee for the
   * first element (an element at coordinate = (0, 0, ..., 0)). The
   * value must be 0 or power-of-two. If zero, the alignment inferred from
   * the dtype. This could also be a layout extension.
   *size_t base_alignment; */

} cl_tensor_layout_blas;

/* Describes the Tensor data layout using terms related to ML models.
 * (These can be also expressed with BLAS layout type) */
typedef cl_uint cl_tensor_layout_ml_type;
#define CL_TENSOR_LAYOUT_ML_UNKNOWN 0
#define CL_TENSOR_LAYOUT_ML_C 1
#define CL_TENSOR_LAYOUT_ML_NC 2
#define CL_TENSOR_LAYOUT_ML_CN 3
#define CL_TENSOR_LAYOUT_ML_HW 4
#define CL_TENSOR_LAYOUT_ML_CHW 5
#define CL_TENSOR_LAYOUT_ML_NCHW 6
#define CL_TENSOR_LAYOUT_ML_NHWC 7
#define CL_TENSOR_LAYOUT_ML_LAST 8



typedef struct _cl_tensor_layout_ml
{
  cl_tensor_layout_ml_type ml_type;
} cl_tensor_layout_ml;

#endif /* OPENCL_EXP_TENSOR_H */
