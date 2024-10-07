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

#ifndef OPENCL_EXP_DEFINED_BUILTIN_KERNELS
#define OPENCL_EXP_DEFINED_BUILTIN_KERNELS

/* Based on spec v.0.3.1
 * https://github.com/KhronosGroup/OpenCL-Docs/pull/1007
 */

#include "cl_exp_tensor.h"

/* errors returned by the DBK API */
/* TODO numeric values */
#define CL_DBK_INVALID_ID_EXP -2306
#define CL_DBK_INVALID_ATTRIBUTE_EXP -2307
#define CL_DBK_UNSUPPORTED_EXP -2308
#define CL_INVALID_TENSOR_LAYOUT_EXP -2309
#define CL_INVALID_TENSOR_RANK_EXP -2310
#define CL_INVALID_TENSOR_SHAPE_EXP -2311
#define CL_INVALID_TENSOR_DATATYPE_EXP -2312
#define CL_INVALID_TENSOR_PROPERTY_EXP -2313

/* cl_tensor_datatype_value is used to pass POD data to DBKs (like Alpha & Beta
 * parameters to the GEMM); the actual type used is implied. In case of
 * GEMM, the datatype is implied the same as COut Tensor datatype.
 * TODO: this could be done with passing a void* pointer, however
 * this avoids the need for malloc & memcpy */
typedef union
{
  cl_char sc;
  cl_uchar uc;
  cl_short ss;
  cl_ushort us;
  cl_int si;
  cl_uint ui;
  cl_long sl;
  cl_ulong ul;
  cl_half fh;
  cl_float ff;
  cl_double fd;
  void *raw;
} cl_tensor_datatype_value_exp;

/* list of fixed predefined builtin kernel IDs.
 * These should be allocated by the OpenCL extension process
 * TODO convert enum to defines ? */
typedef enum
{
  /* CD = custom device, BI = built-in */
  POCL_CDBI_COPY_I8 = 0,
  POCL_CDBI_ADD_I32 = 1,
  POCL_CDBI_MUL_I32 = 2,
  POCL_CDBI_LEDBLINK = 3,
  POCL_CDBI_COUNTRED = 4,
  POCL_CDBI_DNN_CONV2D_RELU_I8 = 5,
  POCL_CDBI_SGEMM_LOCAL_F32 = 6,
  POCL_CDBI_SGEMM_TENSOR_F16F16F32_SCALE = 7,
  POCL_CDBI_SGEMM_TENSOR_F16F16F32 = 8,
  POCL_CDBI_ABS_F32 = 9,
  POCL_CDBI_DNN_DENSE_RELU_I8 = 10,
  POCL_CDBI_MAXPOOL_I8 = 11,
  POCL_CDBI_ADD_I8 = 12,
  POCL_CDBI_MUL_I8 = 13,
  POCL_CDBI_ADD_I16 = 14,
  POCL_CDBI_MUL_I16 = 15,
  POCL_CDBI_STREAMOUT_I32 = 16,
  POCL_CDBI_STREAMIN_I32 = 17,
  POCL_CDBI_VOTE_U32 = 18,
  POCL_CDBI_VOTE_U8 = 19,
  POCL_CDBI_DNN_CONV2D_NCHW_F32 = 20,
  POCL_CDBI_OPENVX_SCALEIMAGE_NN_U8 = 21,
  POCL_CDBI_OPENVX_SCALEIMAGE_BL_U8 = 22,
  POCL_CDBI_OPENVX_TENSORCONVERTDEPTH_WRAP_U8_F32 = 23,
  POCL_CDBI_OPENVX_MINMAXLOC_R1_U8 = 24,
  POCL_CDBI_SOBEL3X3_U8 = 25,
  POCL_CDBI_PHASE_U8 = 26,
  POCL_CDBI_MAGNITUDE_U16 = 27,
  POCL_CDBI_ORIENTED_NONMAX_U16 = 28,
  POCL_CDBI_CANNY_U8 = 29,
  POCL_CDBI_STREAM_MM2S_P512 = 30,
  POCL_CDBI_STREAM_S2MM_P512 = 31,
  POCL_CDBI_BROADCAST_1TO2_P512 = 32,
  POCL_CDBI_SOBEL3X3_P512 = 33,
  POCL_CDBI_PHASE_P512 = 34,
  POCL_CDBI_MAGNITUDE_P512 = 35,
  POCL_CDBI_ORIENTED_NONMAX_P512 = 36,
  POCL_CDBI_GAUSSIAN3X3_P512 = 37,
  CL_DBK_GEMM_EXP = 38,
  CL_DBK_MATMUL_EXP = 39,
  /* See 'Defined Built-in Kernels:JPEG:Usage' in dbk.rst for details on usage.
   */
  CL_DBK_JPEG_ENCODE_EXP = 40,
  /* See 'Defined Built-in Kernels:JPEG:Usage' in dbk.rst for details on usage.
   */
  CL_DBK_JPEG_DECODE_EXP = 41,
  CL_DBK_ONNX_INFERENCE_EXP = 42,
  POCL_CDBI_LAST,
  POCL_CDBI_JIT_COMPILER = 0xFFFF
} cl_dbk_id_exp; /* NOTE: the spec (v0.3.1) has an error (_exp is missing). */

/* for storing DBK property numbers and actual values */
typedef cl_properties cl_dbk_properties_exp;

/* Maximum relative error in ULPs allowed for the results respect to */
/* infinitely precise result. */
#define CL_DBK_PROPERTY_MAX_RELATIVE_ERROR_EXP 1 /* <float> */

/* Allows the results of the DBK to fluctuate* with the exactly same
 * inputs across kernel launches.
 *
 * *: CL_DBK_PROPERTY_MAX_RELATIVE_ERROR must still be respected if present.
 *
 * Drivers may ignore this property. */
#define CL_DBK_PROPERTY_NON_DETERMINISTIC_EXP 2

/* Allow driver to trade off accuracy for speed by allowing it to flush
 * denormals to zero.
 *
 * Drivers may ignore this property, meaning the behavior is not guaranteed. */
#define CL_DBK_PROPERTY_ALLOW_FTZ_EXP 3

typedef cl_program (*clCreateProgramWithDefinedBuiltInKernelsEXP_fn) (
  cl_context context,
  cl_uint num_devices,
  const cl_device_id *device_list,
  cl_uint num_kernels,
  const cl_dbk_id_exp *kernel_ids,
  const char **kernel_names,
  const void **kernel_attributes,
  cl_int *device_support,
  cl_int *errcode_ret);

/*! \brief Creates a cl_program with Defined Builtin Kernels.
 *
 * The program then must be built before using clCreateKernel.
 *
 * NOTE: spec (v0.3.1) has an error (EXP suffix is missing).
 *
 * @param context [in] Context in which to create the program.
 *
 * @param num_devices [in] The number of associated devices for the program
 * @param device_list [in] The list of associated devices for the program
 *
 * @param num_kernels [in] The number of kernels in the program
 * @param kernel_ids [in] Array of num_kernels integers, each one being a DBK
 *        identifier from the existing values of cl_dbk_id_exp enum
 * @param kernel_names [in] Array of num_kernels C strings, these are
 *        caller-provided names for each DBK that can be later used with OpenCL
 *        API calls to create, query, clone etc the kernels of the program.
 * @param kernel_attributes [in] Array of num_kernels pointers that point to
 *        the respective attribute struct of each DBK. The type of the struct
 *        depends on the DBK
 * @param device_support [out] Optional array of num_devices integers, each
 *        will be filled with an error value, if the respective device does
 *        not support any of the DBK+properties combo
 * @param errcode_ret [out] error
 */
extern CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithDefinedBuiltInKernelsEXP (cl_context context,
                                             cl_uint num_devices,
                                             const cl_device_id *device_list,
                                             cl_uint num_kernels,
                                             const cl_dbk_id_exp *kernel_ids,
                                             const char **kernel_names,
                                             const void **kernel_attributes,
                                             cl_int *device_support,
                                             cl_int *errcode_ret);

/* Maximum number of DBK properties */
/* NOTE: this is CL_MAX_DBK_PROPERTIES in the v0.3.1 spec which is an error. */
#define CL_DBK_MAX_PROPERTIES_EXP 16

/* Name: "gemm_exp"
 * Attributes for General multiply operation for matrices.
 * TODO: adopt Vulkan-like extensible struct ?
 * Note that this DBK can also perform matrix-vector operations if
 * tensor shapes are set accordingly. */
typedef struct _cl_dbk_attributes_gemm_exp
{
  cl_tensor_desc_exp a, b, c_in, c_out;
  cl_bool trans_a, trans_b;
  /* Union, real Type depends on the tensor operands. E.g.
   * CL_TENSOR_FLOAT --> cl_float, CL_TENSOR_DOUBLE --> cl_double. */
  cl_tensor_datatype_value_exp alpha, beta;
  /* 0-terminated array of DBK properties */
  cl_dbk_properties_exp kernel_props[CL_DBK_MAX_PROPERTIES_EXP];
} cl_dbk_attributes_gemm_exp;

/* Name: "matmul_exp"
 * Attributes for Matrix multiplication. Identical to gemm_exp
 * with alpha and beta set to 1 and 0, respectively.
 * TODO: adopt Vulkan-like extensible struct ?
 * Note that this DBK can also perform matrix-vector operations if
 * tensor shapes are set accordingly. */
typedef struct _cl_dbk_attributes_matmul_exp
{
  cl_tensor_desc_exp a, b, c;
  cl_bool trans_a, trans_b;
  /* 0-terminated array */
  cl_dbk_properties_exp kernel_props[CL_DBK_MAX_PROPERTIES_EXP];
} cl_dbk_attributes_matmul_exp;

/**
 * Name: jpeg_encode_exp
 *
 * \param width needs to be within the JPEG specification.
 * \param height needs to be within the JPEG specification.
 * \param quality needs to be within [1 - 100].
 */
typedef struct
{
  cl_int width;
  cl_int height;
  cl_int quality;
} cl_dbk_attributes_jpeg_encode_exp;

/* Name: "onnx_inference_exp"
 * Attributes for constructing an inference session for ONNX format ML models.
 */
typedef struct _cl_dbk_attributes_onnx_inference_exp
{
  size_t model_size;
  const char *model_data;
  size_t num_inputs;
  const char **input_tensor_names;
  const cl_tensor_desc_exp *input_tensor_descs;
  size_t num_outputs;
  const char **output_tensor_names;
  const cl_tensor_desc_exp *output_tensor_descs;
  size_t num_initializers;
  const char **initializer_names;
  const cl_tensor_desc_exp *initializer_tensor_descs;
  const char **initializer_data;
} cl_dbk_attributes_onnx_inference_exp;

#endif /* OPENCL_EXP_DEFINED_BUILTIN_KERNELS */
