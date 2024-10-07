/* Test basic functionality of the ONNX inference DBK
   (cl_exp_defined_builtin_kernels)

   Copyright (c) 2024 Jan Solanti / Tampere University

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

#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <CL/cl.h>
#include <CL/cl_exp_defined_builtin_kernels.h>

#include "CL/cl_exp_tensor.h"
#include "config.h"
#include "poclu.h"

#define MALLOC_CHECKED(name, bytes)                                           \
  name = malloc (bytes);                                                      \
  if (!name)                                                                  \
    {                                                                         \
      fprintf (stderr, "Failed to allocate %zu bytes at %s:%d\n", bytes,      \
               __FILE__, __LINE__);                                           \
      return EXIT_FAILURE;                                                    \
    }

const unsigned long num_elements = 1024;

int
main (int _argc, char **_argv)
{
  struct
  {
    clCreateProgramWithDefinedBuiltInKernels_fn
      clCreateProgramWithDefinedBuiltInKernels;
  } ext;

  cl_platform_id platform;
  CHECK_CL_ERROR (clGetPlatformIDs (1, &platform, NULL));
  cl_device_id device;
  CHECK_CL_ERROR (
    clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

  ext.clCreateProgramWithDefinedBuiltInKernels
    = clGetExtensionFunctionAddressForPlatform (
      platform, "clCreateProgramWithDefinedBuiltInKernels");
  if (!ext.clCreateProgramWithDefinedBuiltInKernels)
    {
      fprintf (
        stderr,
        "clCreateProgramWithDefinedBuiltInKernels not found in platform\n");
      return EXIT_FAILURE;
    }

  cl_int error;
  cl_context context = clCreateContext (NULL, 1, &device, NULL, NULL, &error);
  CHECK_CL_ERROR (error);

  const char *onnxfile = SRCDIR "/tests/runtime/xor_f32.onnx";
  FILE *f = fopen (onnxfile, "rb");
  if (!f)
    {
      fprintf (stderr, "Failed to open %s\n", onnxfile);
      return EXIT_FAILURE;
    }
  fseek (f, 0, SEEK_END);
  size_t model_size = ftell (f);
  rewind (f);
  char *model_bytes = NULL;
  MALLOC_CHECKED (model_bytes, model_size);
  size_t bytes_read = fread (model_bytes, sizeof (char), model_size, f);
  fclose (f);
  TEST_ASSERT (bytes_read == model_size);

  cl_tensor_layout_ml tensor_layout = { CL_TENSOR_LAYOUT_ML_C };
  cl_tensor_desc tensor_desc = { 1,
                                 CL_TENSOR_DTYPE_FP32,
                                 { CL_TENSOR_PROPERTY_NONE },
                                 { num_elements },
                                 &tensor_layout,
                                 CL_TENSOR_LAYOUT_ML };
  cl_dbk_id_exp dbk_id = POCL_CDBI_DBK_EXP_ONNX_INFERENCE;
  const char *dbk_name = "exp_onnx_inference";
  const char *input_tensor_names[] = { "A", "B" };
  const char *output_tensor_names[] = { "C" };
  /* All tensors have the same format */
  cl_tensor_desc input_tensor_descs[] = { tensor_desc, tensor_desc };
  cl_tensor_desc output_tensor_descs[] = { tensor_desc };

  /* The test model also has IN_MIN but let's not specify that one */
  const char *initializer_names[] = { "IN_MAX" };
  float f1 = 1.0f;
  const float *initializer_data[] = { &f1 };
  cl_tensor_desc initializer_tensor_descs[] = { { 1,
                                                  CL_TENSOR_DTYPE_FP32,
                                                  { CL_TENSOR_PROPERTY_NONE },
                                                  { 1 },
                                                  &tensor_layout,
                                                  CL_TENSOR_LAYOUT_ML } };

  const cl_dbk_attributes_exp_onnx_inference onnx_inference_attributes
    = { model_size, model_bytes, 2, input_tensor_names, input_tensor_descs, 1,
        output_tensor_names, output_tensor_descs,
        /* The below attributes are optional and can be left zeroed out */
        1, initializer_names, initializer_tensor_descs,
        (const char **)initializer_data };
  const void *dbk_attributes[] = { &onnx_inference_attributes };
  cl_int device_support = 0;
  cl_program program = ext.clCreateProgramWithDefinedBuiltInKernels (
    context, 1, &device, 1, &dbk_id, &dbk_name, dbk_attributes,
    &device_support, &error);
  CHECK_CL_ERROR (error);
  free(model_bytes);
  CHECK_CL_ERROR (clBuildProgram (program, 1, &device, NULL, NULL, NULL));
  cl_kernel kernel = clCreateKernel (program, "exp_onnx_inference", &error);
  CHECK_CL_ERROR (error);

  cl_mem_properties mem_props[]
    = { CL_MEM_TENSOR, (cl_mem_properties)&tensor_desc, 0 };

  float *test_data = NULL;
  MALLOC_CHECKED (test_data, sizeof (float) * num_elements * 2);
  float *in_A = test_data;
  float *in_B = test_data + num_elements;

  srand (clock ());
  for (size_t i = 0; i < num_elements; ++i)
    {
      in_A[i] = (float)rand () / (float)RAND_MAX;
      in_B[i] = (float)rand () / (float)RAND_MAX;
    }

  cl_mem all_inputs = clCreateBufferWithProperties (
    context, mem_props, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
    sizeof (float) * num_elements * 2, test_data, &error);
  CHECK_CL_ERROR (error);
  cl_mem all_outputs = clCreateBufferWithProperties (
    context, mem_props, CL_MEM_WRITE_ONLY, sizeof (float) * num_elements, NULL,
    &error);
  CHECK_CL_ERROR (error);
  uint64_t input_offsets[2] = { 0, num_elements * sizeof (float) };
  uint64_t output_offsets[1] = { 0 };
  cl_mem input_offsets_buf
    = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof (uint64_t) * 2, input_offsets, &error);
  CHECK_CL_ERROR (error);
  cl_mem output_offsets_buf
    = clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof (uint64_t) * 1, output_offsets, &error);
  CHECK_CL_ERROR (error);

  CHECK_CL_ERROR (
    clSetKernelArg (kernel, 0, sizeof (cl_mem), &input_offsets_buf));
  CHECK_CL_ERROR (clSetKernelArg (kernel, 1, sizeof (cl_mem), &all_inputs));
  CHECK_CL_ERROR (
    clSetKernelArg (kernel, 2, sizeof (cl_mem), &output_offsets_buf));
  CHECK_CL_ERROR (clSetKernelArg (kernel, 3, sizeof (cl_mem), &all_outputs));

  cl_command_queue command_queue
    = clCreateCommandQueue (context, device, 0, &error);
  CHECK_CL_ERROR (error);

  size_t one = 1;
  CHECK_CL_ERROR (clEnqueueNDRangeKernel (command_queue, kernel, 1, NULL, &one,
                                          NULL, 0, NULL, NULL));

  cl_int err;
  float *buf_map
    = clEnqueueMapBuffer (command_queue, all_outputs, CL_TRUE, CL_MAP_READ, 0,
                          sizeof (float) * num_elements, 0, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN ("clEnqueueMapBuffer");

  for (size_t i = 0; i < num_elements; ++i)
    {
      float ref = (float)(fabs (in_A[i] - in_B[i]) > 0.5);
      //       printf("in_A[%zu]=%f xor in_B[%zu]=%f = buf_map[%zu]=%f ~
      //       ref=%f\n", i, in_A[i], i, in_B[i], i, buf_map[i], ref);
      TEST_ASSERT (fabs (buf_map[i] - ref) < 1e-6);
    }
  CHECK_CL_ERROR (clEnqueueUnmapMemObject (command_queue, all_outputs, buf_map,
                                           0, NULL, NULL));

  CHECK_CL_ERROR (clReleaseCommandQueue (command_queue));

  CHECK_CL_ERROR (clReleaseMemObject (all_inputs));
  CHECK_CL_ERROR (clReleaseMemObject (all_outputs));

  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseContext (context));

  CHECK_CL_ERROR (clUnloadPlatformCompiler (platform));

  printf ("OK\n");
  return EXIT_SUCCESS;
}
