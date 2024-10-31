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

#include "cl_exp_tensor.h"

/* errors returned by the DBK API */
/* TODO numeric values */
#define CL_INVALID_DBK_ID          -2306
#define CL_INVALID_DBK_ATTRIBUTE   -2307
#define CL_UNSUPPORTED_DBK         -2308
#define CL_INVALID_TENSOR_LAYOUT   -2309
#define CL_INVALID_TENSOR_RANK     -2310
#define CL_INVALID_TENSOR_SHAPE    -2311
#define CL_INVALID_TENSOR_DATATYPE -2312
#define CL_INVALID_TENSOR_PROPERTY -2313

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
  POCL_CDBI_DBK_EXP_GEMM = 38,
  POCL_CDBI_DBK_EXP_MATMUL = 39,
  /* See 'Defined Built-in Kernels:JPEG:Usage' in dbk.rst for details on usage.
   */
  POCL_CDBI_DBK_EXP_JPEG_ENCODE = 40,
  /* See 'Defined Built-in Kernels:JPEG:Usage' in dbk.rst for details on usage.
   */
  POCL_CDBI_DBK_EXP_JPEG_DECODE = 41,
  POCL_CDBI_DBK_EXP_ONNX_INFERENCE = 42,
  POCL_CDBI_LAST,
  POCL_CDBI_JIT_COMPILER = 0xFFFF
} BuiltinKernelId;

/* for storing DBK property numbers and actual values */
typedef cl_properties cl_dbk_properties;

/* Maximum relative error in ULPs allowed for the results respect to */
/* infinitely precise result. */
#define CL_DBK_PROPERTY_MAX_RELATIVE_ERROR 1 /* <float> */

/* Allows the results of the DBK to fluctuate* with the exactly same
 * inputs across kernel launches.
 *
 * *: CL_DBK_PROPERTY_MAX_RELATIVE_ERROR must still be respected if present.
 *
 * Drivers may ignore this property. */
#define CL_DBK_PROPERTY_NON_DETERMINISTIC 2

/* Allow driver to trade off accuracy for speed by allowing it to flush
 * denormals to zero.
 *
 * Drivers may ignore this property, meaning the behavior is not guaranteed. */
#define CL_DBK_PROPERTY_ALLOW_FTZ 3


typedef cl_program (*clCreateProgramWithDefinedBuiltInKernels_fn) (
    cl_context context, cl_uint num_devices, const cl_device_id *device_list,
    cl_uint num_kernels, const BuiltinKernelId *kernel_ids, const char **kernel_names,
    const void **kernel_attributes, cl_int *device_support, cl_int *errcode_ret);

/*! \brief Creates a cl_program with Defined Builtin Kernels.
 *
 * The program then must be built before using clCreateKernel.
 *
 * @param context [in] Context in which to create the program.
 *
 * @param num_devices [in] The number of associated devices for the program
 * @param device_list [in] The list of associated devices for the program
 *
 * @param num_kernels [in] The number of kernels in the program
 * @param kernel_ids [in] Array of num_kernels integers, each one being a DBK
 *        identifier from the existing values of BuiltinKernelId enum
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
clCreateProgramWithDefinedBuiltInKernels (cl_context context,
                                          cl_uint num_devices,
                                          const cl_device_id *device_list,
                                          cl_uint num_kernels,
                                          const BuiltinKernelId *kernel_ids,
                                          const char **kernel_names,
                                          const void **kernel_attributes,
                                          cl_int *device_support,
                                          cl_int *errcode_ret);

/* Maximum number of DBK properties */
#define CL_MAX_DBK_PROPERTIES 16

/* Name: "exp_gemm"
 * Attributes for General multiply operation for matrices.
 * TODO: adopt Vulkan-like extensible struct ?
 * Note that this DBK can also perform matrix-vector operations if
 * tensor shapes are set accordingly. */
typedef struct _cl_dbk_attributes_exp_gemm
{
  cl_tensor_desc a, b, c_in, c_out;
  cl_bool trans_a, trans_b;
  /* Union, real Type depends on the tensor operands. E.g.
   * CL_TENSOR_FLOAT --> cl_float, CL_TENSOR_DOUBLE --> cl_double. */
  cl_tensor_datatype_value alpha, beta;
  /* 0-terminated array of DBK properties */
  cl_dbk_properties kernel_props[CL_MAX_DBK_PROPERTIES];
} cl_dbk_attributes_exp_gemm;

/* Name: "exp_matmul"
 * Attributes for Matrix multiplication. Identical to exp_gemm
 * with alpha and beta set to 1 and 0, respectively.
 * TODO: adopt Vulkan-like extensible struct ?
 * Note that this DBK can also perform matrix-vector operations if
 * tensor shapes are set accordingly. */
typedef struct _cl_dbk_attributes_exp_matmul
{
  cl_tensor_desc a, b, c;
  cl_bool trans_a, trans_b;
  /* 0-terminated array */
  cl_dbk_properties kernel_props[CL_MAX_DBK_PROPERTIES];
} cl_dbk_attributes_exp_matmul;

/**
 * Name: exp_jpeg_encode
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
} cl_dbk_attributes_exp_jpeg_encode;

/* Name: "exp_onnx_inference"
 * Attributes for constructing an inference session for ONNX format ML models.
 */
typedef struct _cl_dbk_attributes_exp_onnx_inference
{
  size_t model_size;
  const char *model_data;
  size_t num_inputs;
  const char **input_tensor_names;
  const cl_tensor_desc *input_tensor_descs;
  size_t num_outputs;
  const char **output_tensor_names;
  const cl_tensor_desc *output_tensor_descs;
  size_t num_initializers;
  const char **initializer_names;
  const cl_tensor_desc *initializer_tensor_descs;
  const char **initializer_data;
} cl_dbk_attributes_exp_onnx_inference;

#endif /* OPENCL_EXP_DEFINED_BUILTIN_KERNELS */
