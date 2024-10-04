/* pocl_builtin_kernels.h - builtin kernel related function declarations

   Copyright (c) 2022-2024 PoCL developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "pocl_builtin_kernels.h"

#include "pocl_tensor_util.h"

#include "dbk/pocl_dbk_khr_jpeg_shared.h"
#include "dbk/pocl_dbk_khr_onnxrt_shared.h"

#include <string.h>

/*
  steps to add a new builtin kernel:

  1) add it to the end of BuiltinKernelId enum in the
     cl_exp_defined_builtin_kernels.h header, before
     POCL_CDBI_LAST

  2) open builtin_kernels.c and edit pocl_BIDescriptors, add a new struct
     for the new kernel, with argument metadata

  2.5)if adding a Defined Built-in Kernel (DBK) add code specific to that kernel
     to:
       * pocl_validate_dbk_attributes
       * pocl_copy_defined_builtin_attributes
       * pocl_release_defined_builtin_attributes
       * pocl_verify_dbk_kernel_args

  3) make sure that devices where you want to support this builtin kernel,
     report it. Every driver does this a bit differently, but at pocl_XYZ_init
     it must properly fill dev->builtin_kernel_list, dev->num_builtin_kernels
     Note: the kernel name reported to user should use dots as separators
     (example: pocl.add.apples.to.oranges)

  4) add the code for the builtin kernel for each device that will support it.
     Note: if the builtin kernel is in source format, its name in the source
     MUST have the dots replaced with underscore
     (example: pocl_add_apples_to_oranges)

     How to do this, depends on device:
       * CUDA has OpenCL-source builtins in lib/CL/devices/cuda/builtins.cl,
         it also has CUDA-source builtins in lib/CL/devices/cuda/builtins.cu
       * almaif driver with TTASIM backend has opencl-source builtins in
         lib/CL/devices/almaif/tce_builtins.cl
       * almaif driver with other backends has builtin kernels in binary format
  (bitstream)
  4.5) to add DBK kernels to cpu devices, add your specific code to the following
  functions:
       * pocl_basic_create_kernel
       * pocl_basic_free_kernel
       * pocl_cpu_init_common
       * pocl_cpu_supports_dbk
       * pocl_cpu_build_defined_builtin
       * pocl_cpu_execute_dbk

*/

/* initializers for kernel arguments */
#define BI_ARG_FULL(TYPENAME, NAME, TYPE, ADQ, ACQ, TQ, SIZE)                 \
  (const pocl_argument_info)                                                  \
  {                                                                           \
    .type_name = TYPENAME, .name = NAME, .address_qualifier = ADQ,            \
    .access_qualifier = ACQ, .type_qualifier = TQ, .type = TYPE,              \
    .type_size = SIZE                                                         \
  }

#define BI_ARG(TYPENAME, NAME, TYPE)                                          \
  (const pocl_argument_info)                                                  \
  {                                                                           \
    .name = NAME, .type_name = TYPENAME, .type = TYPE,                        \
    .address_qualifier = CL_KERNEL_ARG_ADDRESS_GLOBAL,                        \
    .access_qualifier = CL_KERNEL_ARG_ACCESS_NONE,                            \
    .type_qualifier = CL_KERNEL_ARG_TYPE_NONE, .type_size = 0                 \
  }

/* initializers for builtin kernel */
#define BIKD_FULL(ID, NAME, NARGS, IS_DBK, LOCAL_SIZE, ARGUMENTS...)          \
  {                                                                           \
    .num_args = NARGS, .num_locals = ((LOCAL_SIZE > 0) ? 1 : 0),              \
    .local_sizes = (LOCAL_SIZE > 0) ? (size_t[]){ LOCAL_SIZE } : NULL,        \
    .name = NAME, .attributes = NULL,                                         \
    .arg_info = (pocl_argument_info[NARGS]){ ARGUMENTS },                     \
    .has_arg_metadata                                                         \
      = (POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER                                \
         | POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER                               \
         | POCL_HAS_KERNEL_ARG_TYPE_NAME | POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER \
         | POCL_HAS_KERNEL_ARG_NAME),                                         \
    .reqd_wg_size = { 0, 0, 0 }, .wg_size_hint = { 0, 0, 0 },                 \
    .vectypehint = { 0 }, .total_argument_storage_size = 0,                   \
    .max_subgroups = 0, .compile_subgroups = 0, .max_workgroup_size = NULL,   \
    .preferred_wg_multiple = NULL, .local_mem_size = NULL,                    \
    .private_mem_size = NULL, .spill_mem_size = NULL, .build_hash = NULL,     \
    .builtin_kernel_id = ID, .builtin_max_global_work = { 0, 0, 0 },          \
    .data = NULL                                                              \
  }

// BIKD for non-DBK
#define BIKD(ID, NAME, NARGS, ...)                                            \
  BIKD_FULL (ID, NAME, NARGS, 0, 0, __VA_ARGS__)

// BIKD for DBK
#define BIKD_DBK(ID, NAME, NARGS, ...)                                        \
  BIKD_FULL (ID, NAME, NARGS, 1, 0, __VA_ARGS__)

// BIKD with nonzero local size
#define BIKD_LOCAL(ID, NAME, NARGS, LOCAL_SIZE, ...)                          \
  BIKD_FULL (ID, NAME, NARGS, 0, LOCAL_SIZE, __VA_ARGS__)

// Shortcut handles to make the descriptor list more compact.
#define BI_ARG_READ_BUF(TYPENAME, NAME)                                       \
  BI_ARG_FULL (TYPENAME, NAME, POCL_ARG_TYPE_POINTER,                         \
               CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ACCESS_NONE,       \
               (CL_KERNEL_ARG_TYPE_CONST | CL_KERNEL_ARG_TYPE_RESTRICT), 0)

#define BI_ARG_WRITE_BUF(TYPENAME, NAME)                                      \
  BI_ARG_FULL (TYPENAME, NAME, POCL_ARG_TYPE_POINTER,                         \
               CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ACCESS_NONE,       \
               CL_KERNEL_ARG_TYPE_RESTRICT, 0)

#define BI_ARG_POD_WITH_ATTRS(TYPENAME, NAME, SIZE)                           \
  BI_ARG_FULL (TYPENAME, NAME, POCL_ARG_TYPE_NONE,                            \
               CL_KERNEL_ARG_ADDRESS_PRIVATE, CL_KERNEL_ARG_ACCESS_NONE,      \
               CL_KERNEL_ARG_TYPE_NONE, SIZE)

#define BI_ARG_POD(TYPENAME, NAME, NUM_BITS)                                  \
  BI_ARG_POD_WITH_ATTRS (TYPENAME, NAME, ((NUM_BITS + 7u) / 8u))

#define BI_ARG_POD_32b(TYPENAME, NAME) BI_ARG_POD (TYPENAME, NAME, 32)

#define BI_ARG_POD_MUTABLE(TYPENAME, NAME)                                    \
  BI_ARG_FULL (TYPENAME, NAME, POCL_ARG_TYPE_MUTABLE,                         \
               CL_KERNEL_ARG_ADDRESS_PRIVATE, CL_KERNEL_ARG_ACCESS_NONE,      \
               CL_KERNEL_ARG_TYPE_NONE, UINT32_MAX)

#define BI_ARG_READ_PIPE(TYPENAME, NAME)                                      \
  BI_ARG_FULL (TYPENAME, NAME, POCL_ARG_TYPE_PIPE,                            \
               CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ACCESS_NONE,       \
               CL_KERNEL_ARG_TYPE_NONE, 4)

#define BI_ARG_WRITE_PIPE(TYPENAME, NAME)                                     \
  BI_ARG_FULL (TYPENAME, NAME, POCL_ARG_TYPE_PIPE,                            \
               CL_KERNEL_ARG_ADDRESS_GLOBAL, CL_KERNEL_ARG_ACCESS_NONE,       \
               CL_KERNEL_ARG_TYPE_NONE, 4)

pocl_kernel_metadata_t pocl_BIDescriptors[BIKERNELS];

/* C standards older than C17 refuse to initialize module-scope variables
 * even with const values as they don't consider them "constant".
 * The same restrictions don't apply to initializing stack variables.
 * This function is a workaround to make initialization work with C11/C99 */
void
pocl_init_builtin_kernel_metadata ()
{
  pocl_kernel_metadata_t temporary_BIDescriptors[BIKERNELS] = {
    BIKD (POCL_CDBI_COPY_I8, "pocl.copy.i8", 2,
          BI_ARG_READ_BUF ("char*", "input"),
          BI_ARG_WRITE_BUF ("char*", "output")),
    BIKD (
      POCL_CDBI_ADD_I32, "pocl.add.i32", 3, BI_ARG_READ_BUF ("int*", "input1"),
      BI_ARG_READ_BUF ("int*", "input2"), BI_ARG_WRITE_BUF ("int*", "output")),
    BIKD (
      POCL_CDBI_MUL_I32, "pocl.mul.i32", 3, BI_ARG_READ_BUF ("int*", "input1"),
      BI_ARG_READ_BUF ("int*", "input2"), BI_ARG_WRITE_BUF ("int*", "output")),
    BIKD (POCL_CDBI_LEDBLINK, "pocl.ledblink", 2,
          BI_ARG_READ_BUF ("int*", "input1"),
          BI_ARG_READ_BUF ("int*", "input2")),
    BIKD (POCL_CDBI_COUNTRED, "pocl.countred", 2,
          BI_ARG_READ_BUF ("int*", "input"),
          BI_ARG_WRITE_BUF ("int*", "output")),
    BIKD (
      POCL_CDBI_DNN_CONV2D_RELU_I8, "pocl.dnn.conv2d.relu.i8", 12,
      BI_ARG_READ_BUF ("char*", "input"), BI_ARG_READ_BUF ("char*", "weights"),
      BI_ARG_WRITE_BUF ("char*", "output"), BI_ARG_READ_BUF ("int*", "bias"),
      BI_ARG_READ_BUF ("int*", "scale"), BI_ARG_READ_BUF ("int*", "shift"),
      BI_ARG_READ_BUF ("char*", "zero_point"),
      BI_ARG_POD_32b ("unsigned", "window_size_x"),
      BI_ARG_POD_32b ("unsigned", "window_size_y"),
      BI_ARG_POD_32b ("unsigned", "stride_x"),
      BI_ARG_POD_32b ("unsigned", "stride_y"),
      BI_ARG_POD_32b ("unsigned", "input_depth")),
    BIKD_LOCAL (
      POCL_CDBI_SGEMM_LOCAL_F32, "pocl.sgemm.local.f32", 6,
      (2 * 16 * 16 * 4), // local mem size, 2 float arrays 16x16
      BI_ARG_READ_BUF ("float*", "A"), BI_ARG_READ_BUF ("float*", "B"),
      BI_ARG_WRITE_BUF ("float*", "C"), BI_ARG_POD_32b ("unsigned", "M"),
      BI_ARG_POD_32b ("unsigned", "N"), BI_ARG_POD_32b ("unsigned", "K"), ),
    BIKD (POCL_CDBI_SGEMM_TENSOR_F16F16F32_SCALE,
          "pocl.sgemm.scale.tensor.f16f16f32", 8,
          BI_ARG_READ_BUF ("half*", "A"), BI_ARG_READ_BUF ("half*", "B"),
          BI_ARG_WRITE_BUF ("float*", "C"), BI_ARG_POD_32b ("unsigned", "M"),
          BI_ARG_POD_32b ("unsigned", "N"), BI_ARG_POD_32b ("unsigned", "K"),
          BI_ARG_POD_32b ("float", "alpha"),
          BI_ARG_POD_32b ("float", "beta"), ),
    BIKD (POCL_CDBI_SGEMM_TENSOR_F16F16F32, "pocl.sgemm.tensor.f16f16f32", 6,
          BI_ARG_READ_BUF ("half*", "A"), BI_ARG_READ_BUF ("half*", "B"),
          BI_ARG_WRITE_BUF ("float*", "C"), BI_ARG_POD_32b ("unsigned", "M"),
          BI_ARG_POD_32b ("unsigned", "N"),
          BI_ARG_POD_32b ("unsigned", "K"), ),
    BIKD (POCL_CDBI_ABS_F32, "pocl.abs.f32", 2,

          BI_ARG_READ_BUF ("float*", "input"),
          BI_ARG_WRITE_BUF ("float*", "output"), ),
    BIKD (
      POCL_CDBI_DNN_DENSE_RELU_I8, "pocl.dnn.dense.relu.i8", 9,

      BI_ARG_READ_BUF ("char*", "input"), BI_ARG_READ_BUF ("char*", "weights"),
      BI_ARG_WRITE_BUF ("char*", "output"), BI_ARG_READ_BUF ("int*", "bias"),
      BI_ARG_POD_32b ("unsigned", "scale"),
      BI_ARG_POD_32b ("unsigned", "shift"),
      BI_ARG_POD_32b ("unsigned", "zero_point"),
      BI_ARG_POD_32b ("unsigned", "output_minus"),
      BI_ARG_POD_32b ("unsigned", "input_size"), ),
    BIKD (POCL_CDBI_MAXPOOL_I8, "pocl.maxpool.i8", 6,

          BI_ARG_READ_BUF ("char*", "input"),
          BI_ARG_WRITE_BUF ("char*", "output"),
          BI_ARG_POD_32b ("unsigned", "window_size_x"),
          BI_ARG_POD_32b ("unsigned", "window_size_y"),
          BI_ARG_POD_32b ("unsigned", "stride_x"),
          BI_ARG_POD_32b ("unsigned", "stride_y"), ),
    BIKD (POCL_CDBI_ADD_I8, "pocl.add.i8", 3,

          BI_ARG_READ_BUF ("char*", "input1"),
          BI_ARG_READ_BUF ("char*", "input2"),
          BI_ARG_WRITE_BUF ("char*", "output"), ),
    BIKD (POCL_CDBI_MUL_I8, "pocl.mul.i8", 3,

          BI_ARG_READ_BUF ("char*", "input1"),
          BI_ARG_READ_BUF ("char*", "input2"),
          BI_ARG_WRITE_BUF ("char*", "output"), ),
    BIKD (POCL_CDBI_ADD_I16, "pocl.add.i16", 3,

          BI_ARG_READ_BUF ("short*", "input1"),
          BI_ARG_READ_BUF ("short*", "input2"),
          BI_ARG_WRITE_BUF ("short*", "output"), ),
    BIKD (POCL_CDBI_MUL_I16, "pocl.mul.i16", 3,

          BI_ARG_READ_BUF ("short*", "input1"),
          BI_ARG_READ_BUF ("short*", "input2"),
          BI_ARG_WRITE_BUF ("short*", "output"), ),
    BIKD (POCL_CDBI_STREAMOUT_I32, "pocl.streamout.i32", 1,
          BI_ARG_READ_BUF ("int*", "output")),
    BIKD (POCL_CDBI_STREAMIN_I32, "pocl.streamin.i32", 1,
          BI_ARG_WRITE_BUF ("int*", "output")),
    BIKD (
      POCL_CDBI_VOTE_U32, "pocl.vote.u32", 10,
      BI_ARG_READ_BUF ("int*", "output"),
      BI_ARG_POD_32b ("unsigned", "num_inputs"),
      BI_ARG_READ_BUF ("int*", "input0"), BI_ARG_READ_BUF ("int*", "input1"),
      BI_ARG_READ_BUF ("int*", "input2"), BI_ARG_READ_BUF ("int*", "input3"),
      BI_ARG_READ_BUF ("int*", "input4"), BI_ARG_READ_BUF ("int*", "input5"),
      BI_ARG_READ_BUF ("int*", "input6"),
      BI_ARG_READ_BUF ("int*", "input7"), ),
    BIKD (
      POCL_CDBI_VOTE_U8, "pocl.vote.u8", 10,
      BI_ARG_READ_BUF ("char*", "output"),
      BI_ARG_POD_32b ("unsigned", "num_inputs"),
      BI_ARG_READ_BUF ("char*", "input0"), BI_ARG_READ_BUF ("char*", "input1"),
      BI_ARG_READ_BUF ("char*", "input2"), BI_ARG_READ_BUF ("char*", "input3"),
      BI_ARG_READ_BUF ("char*", "input4"), BI_ARG_READ_BUF ("char*", "input5"),
      BI_ARG_READ_BUF ("char*", "input6"),
      BI_ARG_READ_BUF ("char*", "input7"), ),
    BIKD (
      POCL_CDBI_DNN_CONV2D_NCHW_F32, "pocl.dnn.conv2d.nchw.f32", 20,

      BI_ARG_READ_BUF ("float*", "input"),
      BI_ARG_READ_BUF ("float*", "weights"),
      BI_ARG_WRITE_BUF ("float*", "output"), BI_ARG_POD_32b ("int", "input_n"),
      BI_ARG_POD_32b ("int", "input_c"), BI_ARG_POD_32b ("int", "input_h"),
      BI_ARG_POD_32b ("int", "input_w"), BI_ARG_POD_32b ("int", "filt_k"),
      BI_ARG_POD_32b ("int", "filt_c"), BI_ARG_POD_32b ("int", "filt_h"),
      BI_ARG_POD_32b ("int", "filt_w"), BI_ARG_POD_32b ("int", "stride_h"),
      BI_ARG_POD_32b ("int", "stride_w"), BI_ARG_POD_32b ("int", "dilation_h"),
      BI_ARG_POD_32b ("int", "dilation_w"),
      BI_ARG_POD_32b ("int", "padding_h"), BI_ARG_POD_32b ("int", "padding_w"),
      BI_ARG_POD_32b ("int", "groups"), BI_ARG_POD_32b ("float", "alpha"),
      BI_ARG_POD_32b ("float", "beta"), ),
    BIKD (POCL_CDBI_OPENVX_SCALEIMAGE_NN_U8,
          "org.khronos.openvx.scale_image.nn.u8", 6,

          BI_ARG_READ_BUF ("unsigned char*", "input"),
          BI_ARG_WRITE_BUF ("unsigned char*", "output"),
          BI_ARG_POD_32b ("float", "width_scale"),
          BI_ARG_POD_32b ("float", "height_scale"),
          BI_ARG_POD_32b ("int", "input_width"),
          BI_ARG_POD_32b ("int", "input_height"), ),
    BIKD (POCL_CDBI_OPENVX_SCALEIMAGE_BL_U8,
          "org.khronos.openvx.scale_image.bl.u8", 6,

          BI_ARG_READ_BUF ("unsigned char*", "input"),
          BI_ARG_WRITE_BUF ("unsigned char*", "output"),
          BI_ARG_POD_32b ("float", "width_scale"),
          BI_ARG_POD_32b ("float", "height_scale"),
          BI_ARG_POD_32b ("int", "input_width"),
          BI_ARG_POD_32b ("int", "input_height"), ),
    BIKD (POCL_CDBI_OPENVX_TENSORCONVERTDEPTH_WRAP_U8_F32,
          "org.khronos.openvx.tensor_convert_depth.wrap.u8.f32", 4,

          BI_ARG_READ_BUF ("unsigned char*", "input"),
          BI_ARG_WRITE_BUF ("float*", "output"),
          BI_ARG_POD_32b ("float", "norm"),
          BI_ARG_POD_32b ("float", "offset"), ),
    BIKD (POCL_CDBI_OPENVX_MINMAXLOC_R1_U8,
          "org.khronos.openvx.minmaxloc.r1.u8", 5,

          BI_ARG_READ_BUF ("unsigned char*", "input"),
          BI_ARG_WRITE_BUF ("unsigned char*", "min"),
          BI_ARG_WRITE_BUF ("unsigned char*", "max"),
          BI_ARG_WRITE_BUF ("unsigned int*", "minloc"),
          BI_ARG_WRITE_BUF ("unsigned int*", "maxloc"), ),
    BIKD (POCL_CDBI_SOBEL3X3_U8, "pocl.sobel3x3.u8", 3,

          BI_ARG_READ_BUF ("unsigned char*", "input"),
          BI_ARG_WRITE_BUF ("unsigned short*", "sobel_x"),
          BI_ARG_WRITE_BUF ("unsigned short*", "sobel_y"), ),
    BIKD (POCL_CDBI_PHASE_U8, "pocl.phase.u8", 3,

          BI_ARG_READ_BUF ("unsigned short*", "in_x"),
          BI_ARG_READ_BUF ("unsigned short*", "in_y"),
          BI_ARG_WRITE_BUF ("unsigned char*", "output"), ),
    BIKD (POCL_CDBI_MAGNITUDE_U16, "pocl.magnitude.u16", 3,

          BI_ARG_READ_BUF ("unsigned short*", "in_x"),
          BI_ARG_READ_BUF ("unsigned short*", "in_y"),
          BI_ARG_WRITE_BUF ("unsigned short*", "output"), ),
    BIKD (POCL_CDBI_ORIENTED_NONMAX_U16, "pocl.oriented.nonmaxsuppression.u16",
          5,

          BI_ARG_READ_BUF ("unsigned short*", "magnitude"),
          BI_ARG_READ_BUF ("unsigned char*", "phase"),
          BI_ARG_WRITE_BUF ("unsigned char*", "output"),
          BI_ARG_POD_32b ("unsigned short", "threshold_lower"),
          BI_ARG_POD_32b ("unsigned short", "threshold_upper"), ),
    BIKD (POCL_CDBI_CANNY_U8, "pocl.canny.u8", 4,

          BI_ARG_READ_BUF ("unsigned char*", "input"),
          BI_ARG_WRITE_BUF ("unsigned char*", "output"),
          BI_ARG_POD_32b ("unsigned short", "threshold_lower"),
          BI_ARG_POD_32b ("unsigned short", "threshold_upper"), ),
    BIKD (POCL_CDBI_STREAM_MM2S_P512, "pocl.stream.mm2s.p512", 2,

          BI_ARG_READ_BUF ("char*", "in"),
          BI_ARG_WRITE_PIPE ("uchar64", "out"), ),
    BIKD (POCL_CDBI_STREAM_S2MM_P512, "pocl.stream.s2mm.p512", 2,

          BI_ARG_READ_PIPE ("uchar64", "in"),
          BI_ARG_WRITE_BUF ("char*", "out"), ),
    BIKD (POCL_CDBI_BROADCAST_1TO2_P512, "pocl.broadcast.1to2.p512", 3,

          BI_ARG_READ_PIPE ("uchar64", "in"),
          BI_ARG_WRITE_PIPE ("uchar64", "out0"),
          BI_ARG_WRITE_PIPE ("uchar64", "out1"), ),
    BIKD (POCL_CDBI_SOBEL3X3_P512, "pocl.sobel3x3.p512", 3,

          BI_ARG_READ_PIPE ("uchar64", "in"),
          BI_ARG_WRITE_PIPE ("short32", "sobel_x"),
          BI_ARG_WRITE_PIPE ("short32", "sobel_y"), ),
    BIKD (POCL_CDBI_PHASE_P512, "pocl.phase.p512", 3,

          BI_ARG_READ_PIPE ("short32", "in_x"),
          BI_ARG_READ_PIPE ("short32", "in_y"),
          BI_ARG_WRITE_PIPE ("uchar64", "output"), ),
    BIKD (POCL_CDBI_MAGNITUDE_P512, "pocl.magnitude.p512", 3,

          BI_ARG_READ_PIPE ("short32", "in_x"),
          BI_ARG_READ_PIPE ("short32", "in_y"),
          BI_ARG_WRITE_PIPE ("ushort32", "output"), ),
    BIKD (POCL_CDBI_ORIENTED_NONMAX_P512,
          "pocl.oriented.nonmaxsuppression.p512", 5,

          BI_ARG_READ_PIPE ("ushort32", "magnitude"),
          BI_ARG_READ_PIPE ("uchar64", "phase"),
          BI_ARG_WRITE_PIPE ("uchar64", "output"),
          BI_ARG_POD_32b ("unsigned short", "threshold_lower"),
          BI_ARG_POD_32b ("unsigned short", "threshold_upper"), ),
    BIKD (POCL_CDBI_GAUSSIAN3X3_P512, "pocl.gaussian3x3.p512", 2,
          BI_ARG_READ_PIPE ("uchar64", "in"),
          BI_ARG_WRITE_PIPE ("uchar64", "out"), ),
    BIKD_DBK (POCL_CDBI_DBK_EXP_GEMM, "exp_gemm", 6,
              // The types are placeholders
              BI_ARG_READ_BUF ("unsigned char*", "a"),
              BI_ARG_READ_BUF ("unsigned char*", "b"),
              BI_ARG_READ_BUF ("unsigned char*", "c_in"),
              BI_ARG_WRITE_BUF ("unsigned char*", "c_out"),
              // the type of these in the "compiled" DBK
              // depends on the DBK kernel attributes
              // given to clCreateProgramWithDefinedBuiltinKernels
              BI_ARG_POD_MUTABLE ("mutable", "alpha"),
              BI_ARG_POD_MUTABLE ("mutable", "beta"), ),
    BIKD_DBK (POCL_CDBI_DBK_EXP_MATMUL, "exp_matmul", 3,
              BI_ARG_READ_BUF ("unsigned char*", "a"),
              BI_ARG_READ_BUF ("unsigned char*", "b"),
              BI_ARG_WRITE_BUF ("unsigned char*", "c"), ),
    BIKD_DBK (POCL_CDBI_DBK_EXP_JPEG_ENCODE, "exp_jpeg_encode", 3,
              BI_ARG_READ_BUF ("uint8_t*", "image"),
              BI_ARG_WRITE_BUF ("uint8_t*", "jpeg"),
              BI_ARG_WRITE_BUF ("int64_t*", "jpeg_size"), ),
    BIKD_DBK (POCL_CDBI_DBK_EXP_JPEG_DECODE, "exp_jpeg_decode", 3,
              BI_ARG_READ_BUF ("uint8_t*", "jpeg"),
              BI_ARG_READ_BUF ("int64_t*", "jpeg_size"),
              BI_ARG_WRITE_BUF ("uint8_t*", "image"), ),
    BIKD_DBK (POCL_CDBI_DBK_EXP_ONNX_INFERENCE, "exp_onnx_inference", 4,
              BI_ARG_READ_BUF ("unsigned long*", "input_offsets"),
              BI_ARG_READ_BUF ("unsigned char*", "inputs"),
              BI_ARG_READ_BUF ("unsigned long*", "output_offsets"),
              BI_ARG_WRITE_BUF ("unsigned char*", "outputs"), ),

  };
  memcpy (pocl_BIDescriptors, temporary_BIDescriptors,
          sizeof (pocl_BIDescriptors));
  for (size_t i = 0; i < BIKERNELS; ++i)
    {
      pocl_BIDescriptors[i].arg_info = (pocl_argument_info *)malloc (
        pocl_BIDescriptors[i].num_args * sizeof (pocl_argument_info));
      assert (pocl_BIDescriptors[i].arg_info);
      memcpy (pocl_BIDescriptors[i].arg_info,
              temporary_BIDescriptors[i].arg_info,
              (pocl_BIDescriptors[i].num_args * sizeof (pocl_argument_info)));
    }
}

/* creates a deep copy of pocl_kernel_metadata_t in 'target' */
static void
pocl_clone_builtin_kernel_metadata (cl_device_id dev,
                                    pocl_kernel_metadata_t *source,
                                    pocl_kernel_metadata_t *target,
                                    pocl_argument_info *extra)
{

  memcpy (target, (pocl_kernel_metadata_t *)source,
          sizeof (pocl_kernel_metadata_t));
  target->name = strdup (
      "<internal-error: DBKs are meant to be named by the application>");
  target->arg_info = (struct pocl_argument_info *)calloc (
    source->num_args, sizeof (struct pocl_argument_info));
  memset (target->arg_info, 0,
          sizeof (struct pocl_argument_info) * source->num_args);
  for (unsigned Arg = 0; Arg < source->num_args; ++Arg)
    {
      pocl_argument_info *SrcArg = &source->arg_info[Arg];
      if (SrcArg->type == POCL_ARG_TYPE_MUTABLE)
        {
          assert (extra);
          SrcArg = &extra[Arg];
        }
      memcpy (&target->arg_info[Arg], SrcArg, sizeof (pocl_argument_info));
      target->arg_info[Arg].name = strdup (SrcArg->name);
      target->arg_info[Arg].type_name = strdup (SrcArg->type_name);
      if (target->arg_info[Arg].type == POCL_ARG_TYPE_POINTER
          || target->arg_info[Arg].type == POCL_ARG_TYPE_IMAGE)
        target->arg_info[Arg].type_size = sizeof (cl_mem);
    }
  target->builtin_max_global_work = source->builtin_max_global_work;
  target->has_arg_metadata
    = POCL_HAS_KERNEL_ARG_ADDRESS_QUALIFIER
      | POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER | POCL_HAS_KERNEL_ARG_TYPE_NAME
      | POCL_HAS_KERNEL_ARG_TYPE_QUALIFIER | POCL_HAS_KERNEL_ARG_NAME;
}

static cl_int
pocl_clone_builtin_kernel_metadata_by_name (cl_device_id dev,
                                            const char *kernel_name,
                                            pocl_kernel_metadata_t *target)
{

  pocl_kernel_metadata_t *desc = NULL;
  for (size_t i = 0; i < BIKERNELS; ++i)
    {
      desc = &pocl_BIDescriptors[i];
      if (strcmp (desc->name, kernel_name) == 0)
        {
          pocl_clone_builtin_kernel_metadata (dev, desc, target, NULL);
          return CL_SUCCESS;
        }
    }
  return CL_OUT_OF_RESOURCES;
}

static void
pocl_set_defined_arg_info (pocl_argument_info *DstArg,
                           pocl_argument_info *SrcArg,
                           cl_tensor_datatype dtype)
{
  memcpy (DstArg, SrcArg, sizeof (pocl_argument_info));
  DstArg->type = POCL_ARG_TYPE_NONE;
  switch (dtype)
    {
    case CL_TENSOR_DTYPE_FP16:
      DstArg->type_name = "half";
      DstArg->type_size = 2;
      break;
    case CL_TENSOR_DTYPE_FP32:
      DstArg->type_name = "float";
      DstArg->type_size = 4;
      break;
    case CL_TENSOR_DTYPE_FP64:
      DstArg->type_name = "double";
      DstArg->type_size = 8;
      break;
    // case CL_TENSOR_FP8: DstArg->type_name = "fp8"; DstArg->type_size = 1;
    // break;
    case CL_TENSOR_DTYPE_INT64:
      DstArg->type_name = "long";
      DstArg->type_size = 8;
      break;
    case CL_TENSOR_DTYPE_INT32:
      DstArg->type_name = "int";
      DstArg->type_size = 4;
      break;
    case CL_TENSOR_DTYPE_INT16:
      DstArg->type_name = "short";
      DstArg->type_size = 2;
      break;
    case CL_TENSOR_DTYPE_INT8:
      DstArg->type_name = "char";
      DstArg->type_size = 1;
      break;
    // case CL_TENSOR_INT4: DstArg->type_name = "int4_t"; DstArg->type_size =
    // 1; break;
    case CL_TENSOR_DTYPE_UINT64:
      DstArg->type_name = "ulong";
      DstArg->type_size = 8;
      break;
    case CL_TENSOR_DTYPE_UINT32:
      DstArg->type_name = "uint";
      DstArg->type_size = 4;
      break;
    case CL_TENSOR_DTYPE_UINT16:
      DstArg->type_name = "ushort";
      DstArg->type_size = 2;
      break;
    case CL_TENSOR_DTYPE_UINT8:
      DstArg->type_name = "uchar";
      DstArg->type_size = 1;
      break;
      // case CL_TENSOR_UINT4: DstArg->type_name = "uint4_t"; DstArg->type_size
      // = 1; break;

    default:
      DstArg->type_name = "unknown_t";
      DstArg->type_size = 1;
      break;
    }
}

/* Some DBKs have POD-type arguments; the actual type of these arguments
 * (that should be in the kernel metadata) depends on the builtin kernel
 * attributes. This functions sets up the actual metadata types. */
static void
pocl_generate_dbk_defined_arg_info (BuiltinKernelId kernel_id,
                                    pocl_kernel_metadata_t *desc,
                                    pocl_argument_info *extra,
                                    void *attrs)
{
  switch (kernel_id)
    {
    // currently only GEMM has mutable POD arguments
    case POCL_CDBI_DBK_EXP_GEMM:
      {
        const cl_dbk_attributes_exp_gemm *gemm_attrs
          = (const cl_dbk_attributes_exp_gemm *)attrs;
        // BI_ARG_POD_MUTABLE("mutable", "alpha"),
        // BI_ARG_POD_MUTABLE("mutable", "beta"),
        //  TODO should we support separate datatypes for alpha/beta ?
        pocl_set_defined_arg_info (&extra[4], &desc->arg_info[4],
                                   gemm_attrs->a.dtype);
        pocl_set_defined_arg_info (&extra[5], &desc->arg_info[5],
                                   gemm_attrs->a.dtype);
      }
    default:
      break;
    }
}

/* creates a deep copy of pocl_kernel_metadata_t in 'target' */
static cl_int
pocl_clone_builtin_kernel_metadata_by_id (cl_device_id dev,
                                          BuiltinKernelId id,
                                          pocl_kernel_metadata_t *target,
                                          void *attrs)
{

  pocl_kernel_metadata_t *desc = NULL;
  pocl_argument_info *extra = NULL;
  for (size_t i = 0; i < BIKERNELS; ++i)
    {
      desc = &pocl_BIDescriptors[i];
      if (desc->builtin_kernel_id == id)
        {
          extra = calloc (desc->num_args, sizeof (pocl_argument_info));
          assert (extra);
          pocl_generate_dbk_defined_arg_info (id, desc, extra, attrs);
          pocl_clone_builtin_kernel_metadata (dev, desc, target, extra);
          target->builtin_kernel_id = id;
          target->builtin_kernel_attrs = attrs;
          free (extra);
          return CL_SUCCESS;
        }
    }
  return CL_OUT_OF_RESOURCES;
}

int
pocl_setup_builtin_metadata (cl_device_id device,
                             cl_program program,
                             unsigned program_device_i)
{
  if (program->builtin_kernel_names == NULL)
    return 0;

  program->num_kernels = program->num_builtin_kernels;
  if (program->num_kernels)
    {
      program->kernel_meta = (pocl_kernel_metadata_t *)calloc (
        program->num_kernels, sizeof (pocl_kernel_metadata_t));

      for (size_t i = 0; i < program->num_kernels; ++i)
        {
          if (program->builtin_kernel_attributes)
            {
              pocl_clone_builtin_kernel_metadata_by_id (
                device, program->builtin_kernel_ids[i],
                &program->kernel_meta[i],
                program->builtin_kernel_attributes[i]);
            }
          else
            {
              pocl_clone_builtin_kernel_metadata_by_name (
                device, program->builtin_kernel_names[i],
                &program->kernel_meta[i]);
            }
          program->kernel_meta[i].data
            = (void **)calloc (program->num_devices, sizeof (void *));
        }
    }

  return 1;
}

int
pocl_sanitize_builtin_kernel_name (cl_kernel kernel, char **saved_name)
{
  *saved_name = NULL;
  /* NOTE: this is deliberately limited to builtin kernels.
   * These have names with dots, but the actual implementation is done with
   * C/OpenCL/CUDA sources, where they must have compiler-acceptable names.
   * This function & pocl_restore_builtin_kernel_name do the conversion */
  if (kernel->program->num_builtin_kernels)
    {
      *saved_name = kernel->meta->name;
      char *copied_name = strdup (kernel->name);
      size_t len = strlen (copied_name);
      for (uint i = 0; i < len; ++i)
        if (copied_name[i] == '.')
          copied_name[i] = '_';
      kernel->meta->name = copied_name;
      kernel->name = kernel->meta->name;
    }
  return 0;
}

int
pocl_restore_builtin_kernel_name (cl_kernel kernel, char *saved_name)
{
  if (kernel->program->num_builtin_kernels)
    {
      free ((void *)kernel->name);
      kernel->meta->name = saved_name;
      kernel->name = kernel->meta->name;
    }
  return 0;
}

static cl_bool
pocl_tensor_shape_equals (const cl_tensor_desc *A, const cl_tensor_desc *B)
{
  assert (A);
  assert (B);
  if (A->rank != B->rank)
    return CL_FALSE;

  for (unsigned i = 0; i < A->rank; ++i)
    {
      if (A->shape[i] != B->shape[i])
        return CL_FALSE;
    }
  return CL_TRUE;
}

static int
pocl_validate_khr_gemm (cl_bool TransA,
                        cl_bool TransB,
                        const cl_tensor_desc *TenA,
                        const cl_tensor_desc *TenB,
                        const cl_tensor_desc *TenCIOpt,
                        const cl_tensor_desc *TenCOut,
                        const cl_tensor_datatype_value *Alpha,
                        const cl_tensor_datatype_value *Beta)
{
  POCL_RETURN_ERROR_COND ((TenA == NULL), CL_INVALID_DBK_ATTRIBUTE);
  POCL_RETURN_ERROR_COND ((TenB == NULL), CL_INVALID_DBK_ATTRIBUTE);
  POCL_RETURN_ERROR_COND ((TenCOut == NULL), CL_INVALID_DBK_ATTRIBUTE);

  if (Alpha)
    {
      POCL_RETURN_ERROR_COND (Alpha->l == 0, CL_INVALID_DBK_ATTRIBUTE);
    }
  // beta can be 0
  // POCL_RETURN_ERROR_COND (Beta->l == 0, CL_INVALID_DBK_ATTRIBUTE);

  // TBC: 4D+ tensor could be supported by treating the additional
  //      dimensions as batch dimensions - but it might not be
  //      worthwhile due the extra work to support them and processing
  //      overhead they may impose.
  POCL_RETURN_ERROR_ON ((TenA->rank > 3), CL_INVALID_TENSOR_RANK,
                        "Unsupported high-degree tensors.\n");
  POCL_RETURN_ERROR_ON ((TenA->rank < 2), CL_INVALID_TENSOR_RANK,
                        "Rank of A/B tensors must be in {2,3}.\n");

  POCL_RETURN_ERROR_ON ((TenA->rank != TenB->rank), CL_INVALID_TENSOR_RANK,
                        "Rank mismatch between A and B\n");
  POCL_RETURN_ERROR_ON ((TenB->rank != TenCOut->rank), CL_INVALID_TENSOR_RANK,
                        "Rank mismatch between A/B and COut\n");

  POCL_RETURN_ERROR_ON (
    (TenCIOpt != NULL
     && pocl_tensor_shape_equals (TenCIOpt, TenCOut) == CL_FALSE),
    CL_INVALID_TENSOR_SHAPE, "Tensor shape mismatch between C_in and C_out.");

  size_t BatchDims = TenA->rank - 2;

  size_t Temp;
  // CO[b][m][n] = sigma_over_m_n_k(A[b][m][k] * B[b][k][n]) + CI[b][m][n].
  size_t Am = TenA->shape[BatchDims + 0];
  size_t Ak = TenA->shape[BatchDims + 1];
  if (TransA)
    {
      Temp = Am;
      Am = Ak;
      Ak = Temp;
    }

  size_t Bk = TenB->shape[BatchDims + 0];
  size_t Bn = TenB->shape[BatchDims + 1];
  if (TransB)
    {
      Temp = Bk;
      Bk = Bn;
      Bn = Temp;
    }

  size_t COm = TenCOut->shape[BatchDims + 0];
  size_t COn = TenCOut->shape[BatchDims + 1];

  POCL_RETURN_ERROR_COND ((Ak != Bk), CL_INVALID_DBK_ATTRIBUTE);
  POCL_RETURN_ERROR_COND ((Am != COm), CL_INVALID_DBK_ATTRIBUTE);
  POCL_RETURN_ERROR_COND ((Bn != COn), CL_INVALID_DBK_ATTRIBUTE);

  // Check batch dimensions match.
  if (TenA->rank == 3)
    {
      size_t BatchSize = TenA->shape[0];
      POCL_RETURN_ERROR_ON ((BatchSize > 1
                             && (BatchSize != TenB->shape[0]
                                 || TenB->shape[0] != TenCOut->shape[0])),
                            CL_INVALID_TENSOR_SHAPE, "Batch size mismatch.\n");

      POCL_RETURN_ERROR_ON (
        (BatchSize > 1 && TenCIOpt && TenCIOpt->shape[0] != TenCOut->shape[0]),
        CL_INVALID_TENSOR_SHAPE, "Batch size mismatch.\n");
    }

  // Check datatypes
  POCL_RETURN_ERROR_ON (
    (TenA->dtype >= CL_TENSOR_DTYPE_LAST || TenB->dtype >= CL_TENSOR_DTYPE_LAST
     || TenCOut->dtype >= CL_TENSOR_DTYPE_LAST),
    CL_INVALID_TENSOR_DATATYPE, "Unknown data type in input Tensors");

  POCL_RETURN_ERROR_ON ((TenA->dtype != TenB->dtype), CL_INVALID_TENSOR_DATATYPE,
                        "datatype mismatch between A and B.\n");

  POCL_RETURN_ERROR_ON (TenCIOpt && (TenCIOpt->dtype != TenCOut->dtype),
                        CL_INVALID_TENSOR_DATATYPE,
                        "datatype mismatch between C_ind and C_out\n");

  // TODO: check validity of data layouts of the tensors. Now assumes they're
  // ok

  return CL_SUCCESS;
}

int
pocl_validate_dbk_attributes (BuiltinKernelId kernel_id,
                              const void *kernel_attributes,
                              pocl_validate_khr_gemm_callback_t GemmCB)
{
  if (GemmCB == NULL)
    GemmCB = pocl_validate_khr_gemm;
  switch (kernel_id)
    {
    case POCL_CDBI_DBK_EXP_GEMM:
      {
        const cl_dbk_attributes_exp_gemm *Attrs
          = (const cl_dbk_attributes_exp_gemm *)kernel_attributes;

        return GemmCB (Attrs->trans_a, Attrs->trans_b, &Attrs->a, &Attrs->b,
                       &Attrs->c_in, &Attrs->c_out, &Attrs->alpha,
                       &Attrs->beta);
      }
    case POCL_CDBI_DBK_EXP_MATMUL:
      {
        const cl_dbk_attributes_exp_matmul *Attrs
          = (const cl_dbk_attributes_exp_matmul *)kernel_attributes;

        return GemmCB (Attrs->trans_a, Attrs->trans_b, &Attrs->a, &Attrs->b,
                       NULL, &Attrs->c, NULL, NULL);
      }
    case POCL_CDBI_DBK_EXP_JPEG_DECODE:
    case POCL_CDBI_DBK_EXP_JPEG_ENCODE:
      return pocl_validate_khr_jpeg (kernel_id, kernel_attributes);
#ifdef HAVE_ONNXRT
    case POCL_CDBI_DBK_EXP_ONNX_INFERENCE:
      {
        /* TODO: validate I/O tensor list */
        const cl_dbk_attributes_exp_onnx_inference *attrs
          = (cl_dbk_attributes_exp_onnx_inference *)kernel_attributes;

        if (attrs->num_initializers == 0
            && (attrs->initializer_names != NULL
                || attrs->initializer_data != NULL
                || attrs->initializer_tensor_descs != NULL))
          return CL_INVALID_ARG_VALUE;
        if (attrs->num_initializers == 1
            && (attrs->initializer_names == NULL
                || attrs->initializer_data == NULL
                || attrs->initializer_tensor_descs == NULL))
          return CL_INVALID_ARG_VALUE;

        return CL_SUCCESS;
      }
#endif
  default:
      break;
    }
  POCL_RETURN_ERROR_ON (1, CL_INVALID_DBK_ID, "Unknown builtin kernel ID: %u.\n",
                        kernel_id);
}

void *
pocl_copy_defined_builtin_attributes (BuiltinKernelId kernel_id,
                                      const void *kernel_attributes)
{
  int err = CL_SUCCESS;
  switch (kernel_id)
    {
    case POCL_CDBI_DBK_EXP_GEMM:
      {
        cl_dbk_attributes_exp_gemm *attrs
          = malloc (sizeof (cl_dbk_attributes_exp_gemm));
        if (attrs == NULL)
          return NULL;
        cl_dbk_attributes_exp_gemm *src
          = (cl_dbk_attributes_exp_gemm *)kernel_attributes;
        memcpy (attrs, src, sizeof (cl_dbk_attributes_exp_gemm));
        err = pocl_copy_tensor_desc_layout (&attrs->a, &src->a);
        err = pocl_copy_tensor_desc_layout (&attrs->b, &src->b);
        err = pocl_copy_tensor_desc_layout (&attrs->c_in, &src->c_in);
        err = pocl_copy_tensor_desc_layout (&attrs->c_out, &src->c_out);

        return attrs;
      }
    case POCL_CDBI_DBK_EXP_MATMUL:
      {
        cl_dbk_attributes_exp_matmul *attrs
          = malloc (sizeof (cl_dbk_attributes_exp_matmul));
        if (attrs == NULL)
          return NULL;
        cl_dbk_attributes_exp_matmul *src
          = (cl_dbk_attributes_exp_matmul *)kernel_attributes;
        memcpy (attrs, src, sizeof (cl_dbk_attributes_exp_matmul));

        err = pocl_copy_tensor_desc_layout (&attrs->a, &src->a);
        err = pocl_copy_tensor_desc_layout (&attrs->b, &src->b);
        err = pocl_copy_tensor_desc_layout (&attrs->c, &src->c);

        return attrs;
      }
    case POCL_CDBI_DBK_EXP_JPEG_ENCODE:
    case POCL_CDBI_DBK_EXP_JPEG_DECODE:
      return pocl_copy_dbk_attributes_khr_jpeg (kernel_id, kernel_attributes);
#ifdef HAVE_ONNXRT
    case POCL_CDBI_DBK_EXP_ONNX_INFERENCE:
      return pocl_copy_onnx_inference_dbk_attributes (kernel_attributes);
#endif
  default:
      break;
    }
  POCL_MSG_ERR ("Unknown builtin kernel ID: %u", kernel_id);
  return NULL;
}

int
pocl_release_defined_builtin_attributes (BuiltinKernelId kernel_id,
                                         void *kernel_attributes)
{
  switch (kernel_id)
    {
    case POCL_CDBI_DBK_EXP_GEMM:
      {
        cl_dbk_attributes_exp_gemm *attrs
          = (cl_dbk_attributes_exp_gemm *)kernel_attributes;
        POCL_MEM_FREE (attrs->a.layout);
        POCL_MEM_FREE (attrs->b.layout);
        POCL_MEM_FREE (attrs->c_in.layout);
        POCL_MEM_FREE (attrs->c_out.layout);
        POCL_MEM_FREE (attrs);
        return CL_SUCCESS;
      }
    case POCL_CDBI_DBK_EXP_MATMUL:
      {
        cl_dbk_attributes_exp_matmul *attrs
          = (cl_dbk_attributes_exp_matmul *)kernel_attributes;
        POCL_MEM_FREE (attrs->a.layout);
        POCL_MEM_FREE (attrs->b.layout);
        POCL_MEM_FREE (attrs->c.layout);
        POCL_MEM_FREE (attrs);
        return CL_SUCCESS;
      }
    case POCL_CDBI_DBK_EXP_JPEG_ENCODE:
    case POCL_CDBI_DBK_EXP_JPEG_DECODE:
      return pocl_release_dbk_attributes_khr_jpeg (kernel_id,
                                                   kernel_attributes);
#ifdef HAVE_ONNXRT
    case POCL_CDBI_DBK_EXP_ONNX_INFERENCE:
      {
        pocl_release_onnx_inference_dbk_attributes (kernel_attributes);
        return CL_SUCCESS;
      }
#endif
    default:
      break;
    }
  POCL_RETURN_ERROR_ON (1, CL_INVALID_DBK_ID, "Unknown builtin kernel ID: %u.\n",
                        kernel_id);
}

/** Helper functions for (de)serializing dbk attributes in the remote driver */

/******************************* SERIALIZATION *******************************/
#define SERIALIZE(name)                                                       \
  do                                                                          \
    {                                                                         \
      if (buf)                                                                \
        {                                                                     \
          memcpy (*buf, &(name), sizeof (name));                              \
          *buf += sizeof (name);                                              \
        }                                                                     \
      total += sizeof (name);                                                 \
    }                                                                         \
  while (0)

#define COPY(name, size)                                                      \
  do                                                                          \
    {                                                                         \
      if (buf)                                                                \
        {                                                                     \
          memcpy (*buf, (name), size);                                        \
          *buf += size;                                                       \
        }                                                                     \
      total += size;                                                          \
    }                                                                         \
  while (0)

uint64_t
pocl_serialize_cl_tensor_desc (const cl_tensor_desc *t, char **buf)
{
  uint64_t total = 0;
  uint8_t has_layout = t->layout != NULL;

  SERIALIZE (t->rank);
  SERIALIZE (t->dtype);
  SERIALIZE (t->properties);
  SERIALIZE (t->shape);
  SERIALIZE (t->layout_type);
  SERIALIZE (has_layout);
  if (has_layout)
    {
      switch (t->layout_type)
        {
        case CL_TENSOR_LAYOUT_BLAS:
          {
            cl_tensor_layout_blas *layout = t->layout;
            SERIALIZE (layout->leading_dims);
            SERIALIZE (layout->leading_strides);
            break;
          }
        case CL_TENSOR_LAYOUT_ML:
          {
            cl_tensor_layout_ml *layout = t->layout;
            SERIALIZE (layout->ml_type);
            break;
          }
        default:
          break;
        }
    }

  return total;
}

uint64_t
pocl_serialize_dbk_attribs (BuiltinKernelId id,
                            const void *attributes,
                            char **buf)
{
  /* First item shall be the BuiltinKernelId */
  uint64_t total = 0;
  uint64_t dbk_id = id;

  SERIALIZE (dbk_id);
  switch (id)
    {
    case POCL_CDBI_DBK_EXP_GEMM:
      {
        const cl_dbk_attributes_exp_gemm *attr = attributes;
        total += pocl_serialize_cl_tensor_desc (&attr->a, buf);
        total += pocl_serialize_cl_tensor_desc (&attr->b, buf);
        total += pocl_serialize_cl_tensor_desc (&attr->c_in, buf);
        total += pocl_serialize_cl_tensor_desc (&attr->c_out, buf);
        SERIALIZE (attr->trans_a);
        SERIALIZE (attr->trans_b);
        SERIALIZE (attr->alpha);
        SERIALIZE (attr->beta);
        SERIALIZE (attr->kernel_props);
        break;
      }
    case POCL_CDBI_DBK_EXP_MATMUL:
      {
        const cl_dbk_attributes_exp_matmul *attr = attributes;
        total += pocl_serialize_cl_tensor_desc (&attr->a, buf);
        total += pocl_serialize_cl_tensor_desc (&attr->b, buf);
        total += pocl_serialize_cl_tensor_desc (&attr->c, buf);
        SERIALIZE (attr->trans_a);
        SERIALIZE (attr->trans_b);
        SERIALIZE (attr->kernel_props);
        break;
      }
    case POCL_CDBI_DBK_EXP_JPEG_ENCODE:
      {
        const cl_dbk_attributes_exp_jpeg_encode *attr = attributes;
        SERIALIZE (attr->width);
        SERIALIZE (attr->height);
        SERIALIZE (attr->quality);
        break;
      }
    case POCL_CDBI_DBK_EXP_ONNX_INFERENCE:
      {
        const cl_dbk_attributes_exp_onnx_inference *attr = attributes;
        uint64_t model_size = attr->model_size;
        uint64_t num_inputs = attr->num_inputs;
        uint64_t num_outputs = attr->num_outputs;
        uint64_t num_initializers = attr->num_initializers;
        SERIALIZE (model_size);
        COPY (attr->model_data, model_size);
        SERIALIZE (num_inputs);
        for (size_t i = 0; i < num_inputs; ++i)
          {
            uint64_t name_len = strlen (attr->input_tensor_names[i]);
            SERIALIZE (name_len);
            COPY (attr->input_tensor_names[i], name_len);
            total += pocl_serialize_cl_tensor_desc (
              &(attr->input_tensor_descs[i]), buf);
          }
        SERIALIZE (num_outputs);
        for (size_t i = 0; i < num_outputs; ++i)
          {
            uint64_t name_len = strlen (attr->output_tensor_names[i]);
            SERIALIZE (name_len);
            COPY (attr->output_tensor_names[i], name_len);
            total += pocl_serialize_cl_tensor_desc (
              &(attr->output_tensor_descs[i]), buf);
          }
        SERIALIZE (num_initializers);
        for (size_t i = 0; i < num_initializers; ++i)
          {
            uint64_t name_len = strlen (attr->initializer_names[i]);
            uint64_t data_len
              = pocl_tensor_data_size (&(attr->initializer_tensor_descs[i]));
            SERIALIZE (name_len);
            COPY (attr->initializer_names[i], name_len);
            total += pocl_serialize_cl_tensor_desc (
              &(attr->initializer_tensor_descs[i]), buf);
            SERIALIZE (data_len);
            COPY (attr->initializer_data[i], data_len);
          }
        break;
      }
    default:
      break;
    }

  return total;
}

#undef COPY
#undef SERIALIZE

/****************************** DESERIALIZATION ******************************/

#define DESERIALIZE(name)                                                     \
  do                                                                          \
    {                                                                         \
      memcpy (&(name), *buf, sizeof (name));                                  \
      *buf += sizeof (name);                                                  \
    }                                                                         \
  while (0)

#define COPY(name, size)                                                      \
  do                                                                          \
    {                                                                         \
      if (buf)                                                                \
        {                                                                     \
          memcpy ((name), *buf, size);                                        \
          *buf += size;                                                       \
        }                                                                     \
    }                                                                         \
  while (0)

int
pocl_deserialize_cl_tensor_desc (cl_tensor_desc *t, const char **buf)
{
  uint8_t has_layout = 0;
  DESERIALIZE (t->rank);
  DESERIALIZE (t->dtype);
  DESERIALIZE (t->properties);
  DESERIALIZE (t->shape);
  DESERIALIZE (t->layout_type);
  DESERIALIZE (has_layout);
  if (has_layout)
    {
      switch (t->layout_type)
        {
        case CL_TENSOR_LAYOUT_BLAS:
          {
            cl_tensor_layout_blas *layout
              = malloc (sizeof (cl_tensor_layout_blas));
            DESERIALIZE (layout->leading_dims);
            DESERIALIZE (layout->leading_strides);
            t->layout = layout;
            break;
          }
        case CL_TENSOR_LAYOUT_ML:
          {
            cl_tensor_layout_ml *layout
              = malloc (sizeof (cl_tensor_layout_ml));
            DESERIALIZE (layout->ml_type);
            t->layout = layout;
            break;
          }
        default:
          break;
        }
    }
  else
    t->layout = NULL;

  return 1;
}

int
pocl_deserialize_dbk_attribs (BuiltinKernelId *id,
                              void **attributes,
                              const char **buf)
{
  uint64_t dbk_id;
  DESERIALIZE (dbk_id);
  *id = (BuiltinKernelId)dbk_id;
  switch (dbk_id)
    {
    case POCL_CDBI_DBK_EXP_GEMM:
      {
        cl_dbk_attributes_exp_gemm *attr
          = malloc (sizeof (cl_dbk_attributes_exp_gemm));
        pocl_deserialize_cl_tensor_desc (&attr->a, buf);
        pocl_deserialize_cl_tensor_desc (&attr->b, buf);
        pocl_deserialize_cl_tensor_desc (&attr->c_in, buf);
        pocl_deserialize_cl_tensor_desc (&attr->c_out, buf);
        DESERIALIZE (attr->trans_a);
        DESERIALIZE (attr->trans_b);
        DESERIALIZE (attr->alpha);
        DESERIALIZE (attr->beta);
        DESERIALIZE (attr->kernel_props);
        *attributes = attr;
        break;
      }
    case POCL_CDBI_DBK_EXP_MATMUL:
      {
        cl_dbk_attributes_exp_matmul *attr
          = malloc (sizeof (cl_dbk_attributes_exp_matmul));
        pocl_deserialize_cl_tensor_desc (&attr->a, buf);
        pocl_deserialize_cl_tensor_desc (&attr->b, buf);
        pocl_deserialize_cl_tensor_desc (&attr->c, buf);
        DESERIALIZE (attr->trans_a);
        DESERIALIZE (attr->trans_b);
        DESERIALIZE (attr->kernel_props);
        *attributes = attr;
        break;
      }
    case POCL_CDBI_DBK_EXP_JPEG_ENCODE:
      {
        cl_dbk_attributes_exp_jpeg_encode *attr
          = malloc (sizeof (cl_dbk_attributes_exp_jpeg_encode));
        DESERIALIZE (attr->width);
        DESERIALIZE (attr->height);
        DESERIALIZE (attr->quality);
        *attributes = attr;
        break;
      }
    case POCL_CDBI_DBK_EXP_ONNX_INFERENCE:
      {
        cl_dbk_attributes_exp_onnx_inference *attr
          = calloc (1, sizeof (cl_dbk_attributes_exp_onnx_inference));
        uint64_t model_size;
        uint64_t num_inputs;
        uint64_t num_outputs;
        uint64_t num_initializers;
        DESERIALIZE (model_size);
        attr->model_size = model_size;
        if (model_size > 0)
          {
            attr->model_data = malloc (model_size);
            COPY ((char *)attr->model_data, model_size);
          }
        DESERIALIZE (num_inputs);
        attr->num_inputs = num_inputs;
        if (num_inputs > 0)
          {
            attr->input_tensor_names = malloc (sizeof (char *) * num_inputs);
            attr->input_tensor_descs
              = malloc (sizeof (cl_tensor_desc) * num_inputs);
            for (size_t i = 0; i < num_inputs; ++i)
              {
                uint64_t name_len;

                DESERIALIZE (name_len);
                attr->input_tensor_names[i] = malloc (name_len + 1);
                COPY ((char *)attr->input_tensor_names[i], name_len);
                ((char *)attr->input_tensor_names[i])[name_len] = 0;
                pocl_deserialize_cl_tensor_desc (
                  (cl_tensor_desc *)&(attr->input_tensor_descs[i]), buf);
              }
          }

        DESERIALIZE (num_outputs);
        attr->num_outputs = num_outputs;
        if (num_outputs > 0)
          {
            attr->output_tensor_names = malloc (sizeof (char *) * num_outputs);
            attr->output_tensor_descs
              = malloc (sizeof (cl_tensor_desc) * num_outputs);
            for (size_t i = 0; i < num_outputs; ++i)
              {
                uint64_t name_len;

                DESERIALIZE (name_len);
                attr->output_tensor_names[i] = malloc (name_len + 1);
                COPY ((char *)attr->output_tensor_names[i], name_len);
                ((char *)attr->output_tensor_names[i])[name_len] = 0;
                pocl_deserialize_cl_tensor_desc (
                  (cl_tensor_desc *)&(attr->output_tensor_descs[i]), buf);
              }
          }

        DESERIALIZE (num_initializers);
        attr->num_initializers = num_initializers;
        if (num_initializers > 0)
          {
            attr->initializer_names
              = malloc (sizeof (char *) * num_initializers);
            attr->initializer_tensor_descs
              = malloc (sizeof (cl_tensor_desc) * num_initializers);
            attr->initializer_data
              = malloc (sizeof (char *) * num_initializers);
            for (size_t i = 0; i < num_initializers; ++i)
              {
                uint64_t name_len;
                uint64_t data_len;

                DESERIALIZE (name_len);
                attr->initializer_names[i] = malloc (name_len + 1);
                COPY ((char *)attr->initializer_names[i], name_len);
                ((char *)attr->initializer_names[i])[name_len] = 0;
                pocl_deserialize_cl_tensor_desc (
                  (cl_tensor_desc *)&(attr->initializer_tensor_descs[i]), buf);
                DESERIALIZE (data_len);
                if (data_len > 0)
                  {
                    attr->initializer_data[i] = malloc (data_len);
                    COPY ((char *)attr->initializer_data[i], data_len);
                  }
              }
          }

        *attributes = attr;
        break;
      }
    default:
      break;
    }

  return 1;
}

#undef COPY
#undef DESERIALIZE
