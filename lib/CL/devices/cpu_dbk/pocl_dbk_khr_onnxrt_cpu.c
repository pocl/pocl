/* pocl_dbk_khr_onnxrt_cpu.c - ONNXRuntime Defined Built-in Kernels implementation.

   Copyright (c) 2024 Jan Solanti <jan.solanti@tuni.fi>

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

#include <assert.h>
#include <string.h>

#include "CL/cl.h"
#include "CL/cl_exp_defined_builtin_kernels.h"
#include "CL/cl_exp_tensor.h"
#include "onnxruntime_c_api.h"

#include "pocl_builtin_kernels.h"
#include "pocl_cl.h"
#include "pocl_debug.h"
#include "pocl_dbk_khr_onnxrt_cpu.h"
#include "pocl_tensor_util.h"

/* Update as needed, but make sure to remove uses of deprecated functions when
 * doing so. */
#define TARGET_ORT_API_VERSION 16

static OrtApi *ort_api = NULL;

typedef struct onnxrt_instance
{
  OrtEnv *env;
  OrtMemoryInfo *mem_info;
  OrtSession *session;
  OrtSessionOptions *session_options;
  const cl_dbk_attributes_khr_onnx_inference *attributes;
} onnxrt_instance_t;

static void
pocl_ort_logger (void *_param,
                 OrtLoggingLevel severity,
                 const char *category,
                 const char *_logid,
                 const char *code_location,
                 const char *message)
{
  switch (severity)
    {
    case ORT_LOGGING_LEVEL_INFO:
      POCL_MSG_PRINT_INFO ("[%s] %s: %s\n", category, code_location, message);
      break;
    case ORT_LOGGING_LEVEL_WARNING:
      POCL_MSG_WARN ("[%s] %s: %s\n", category, code_location, message);
      break;
    case ORT_LOGGING_LEVEL_ERROR:
      POCL_MSG_ERR ("[%s] %s: %s\n", category, code_location, message);
      break;
    case ORT_LOGGING_LEVEL_FATAL:
      POCL_MSG_ERR ("[%s] %s: %s\n", category, code_location, message);
      break;
    default:
      POCL_MSG_PRINT_INFO ("[%s] %s: %s\n", category, code_location, message);
    }
}

/** Approximate mapping of ONNXRT errors to CL errors */
static cl_int
ort_to_cl_error_code (OrtStatusPtr status)
{
  switch (ort_api->GetErrorCode (status))
    {
    case ORT_OK:
      return CL_SUCCESS;
    case ORT_FAIL:
    case ORT_ENGINE_ERROR:
    case ORT_RUNTIME_EXCEPTION:
    case ORT_NOT_IMPLEMENTED:
    case ORT_EP_FAIL:
      return CL_DEVICE_NOT_AVAILABLE;
    case ORT_INVALID_ARGUMENT:
    case ORT_NO_SUCHFILE:
    case ORT_NO_MODEL:
    case ORT_INVALID_PROTOBUF:
    case ORT_MODEL_LOADED:
    case ORT_INVALID_GRAPH:
      return CL_INVALID_VALUE;

    default:
      return CL_SUCCESS;
    }
}

static ONNXTensorElementDataType
cl_to_onnxrt_tensor_element_type (cl_tensor_datatype dtype)
{
  switch (dtype)
    {
    case CL_TENSOR_DTYPE_FP64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case CL_TENSOR_DTYPE_INT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case CL_TENSOR_DTYPE_UINT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case CL_TENSOR_DTYPE_FP32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case CL_TENSOR_DTYPE_INT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case CL_TENSOR_DTYPE_UINT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case CL_TENSOR_DTYPE_FP16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case CL_TENSOR_DTYPE_INT16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case CL_TENSOR_DTYPE_UINT16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case CL_TENSOR_DTYPE_FP8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2;
    case CL_TENSOR_DTYPE_INT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case CL_TENSOR_DTYPE_UINT8:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case CL_TENSOR_DTYPE_INT4:
    case CL_TENSOR_DTYPE_UINT4:
    case CL_TENSOR_DTYPE_LAST:
    case CL_TENSOR_DTYPE_UNKNOWN:
    default:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }
}

#define CHECK_ORT_CALL(call)                                                  \
  do                                                                          \
    {                                                                         \
      OrtStatusPtr status = ort_api->call;                                    \
      if (status)                                                             \
        {                                                                     \
          POCL_MSG_ERR ("[onnxruntime] %s\n",                                 \
                        ort_api->GetErrorMessage (status));                   \
          err = ort_to_cl_error_code (status);                                \
          ort_api->ReleaseStatus (status);                                    \
          goto ERROR;                                                         \
        }                                                                     \
    }                                                                         \
  while (0)

cl_int
pocl_create_ort_instance (const cl_dbk_attributes_khr_onnx_inference *attrs,
                          onnxrt_instance_t **onnxrt)
{
  cl_int err = CL_SUCCESS;
  onnxrt_instance_t *ort = NULL;
  OrtValue **initializers = NULL;

  if (ort_api == NULL)
    ort_api = (OrtApi *)OrtGetApiBase ()->GetApi (TARGET_ORT_API_VERSION);

  ort = calloc (1, sizeof (onnxrt_instance_t));
  CHECK_ORT_CALL (
    CreateEnv (ORT_LOGGING_LEVEL_INFO, "ONNXRuntime", &ort->env));
  CHECK_ORT_CALL (CreateSessionOptions (&ort->session_options));
  CHECK_ORT_CALL (
    SetUserLoggingFunction (ort->session_options, &pocl_ort_logger, NULL));
  CHECK_ORT_CALL (
    CreateCpuMemoryInfo (OrtArenaAllocator, OrtMemTypeCPU, &ort->mem_info));

  if (attrs->num_initializers != 0)
    {
      initializers = malloc (sizeof (OrtValue *) * attrs->num_initializers);
      for (size_t i = 0; i < attrs->num_initializers; ++i)
        {
          const cl_tensor_desc *meta = &attrs->initializer_tensor_descs[i];
          void *ptr = (void *)attrs->initializer_data[i];
          size_t len = pocl_tensor_type_size (meta->dtype);
          size_t shape_len = meta->rank;
          ONNXTensorElementDataType type
            = cl_to_onnxrt_tensor_element_type (meta->dtype);
          int64_t shape[CL_MEM_MAX_TENSOR_RANK] = { 0 };
          for (size_t dim = 0; dim < shape_len; ++dim)
            {
              shape[dim] = meta->shape[dim];
              len *= shape[dim];
            }
          CHECK_ORT_CALL (CreateTensorWithDataAsOrtValue (
            ort->mem_info, ptr, len, shape, shape_len, type,
            &initializers[i]));
          CHECK_ORT_CALL (AddInitializer (ort->session_options,
                                          attrs->initializer_names[i],
                                          initializers[i]));
        }
    }
  CHECK_ORT_CALL (
    CreateSessionFromArray (ort->env, attrs->model_data, attrs->model_size,
                            ort->session_options, &ort->session));

  POCL_MEM_FREE (initializers);

  /* Attributes are owned by the program, no need to duplicate them */
  ort->attributes = attrs;

  *onnxrt = ort;
  return err;

ERROR:
  pocl_destroy_ort_instance (&ort);
  return err;
}

cl_int
pocl_destroy_ort_instance (onnxrt_instance_t **onnxrt)
{
  onnxrt_instance_t *ort = *onnxrt;

  if (ort->mem_info)
    {
      ort_api->ReleaseMemoryInfo (ort->mem_info);
      ort->mem_info = NULL;
    }

  if (ort->session)
    {
      ort_api->ReleaseSession (ort->session);
      ort->session = NULL;
    }

  if (ort->session_options)
    ort_api->ReleaseSessionOptions (ort->session_options);


  if (ort->env)
    {
      ort_api->ReleaseEnv (ort->env);
      ort->env = NULL;
    }

  *onnxrt = NULL;
  free (ort);

ERROR:
  return CL_SUCCESS;
}

/** Run the model loaded into the given onnxrt session
 *
 * \param ort An ONNXRuntime inference session initialized with
 * pocl_create_ort_instance
 * \param input_offsets Byte offsets of the input tensors in input_data.
 * Tensors must be in the same order as specified when creating the kernel.
 * \param input_data Pointer to the combined input tensor contents
 * \param output_offsets Byte offsets of the output tensors in output_data.
 * Tensors must be in the same order as specified when creating the kernel.
 * \param output_storage Pointer to the combined outupt tensor storage area
 */
cl_int
pocl_perform_ort_inference (onnxrt_instance_t *ort,
                            const uint64_t *input_offsets,
                            char *input_data,
                            const uint64_t *output_offsets,
                            char *output_storage)
{
  cl_int err = CL_SUCCESS;
  const cl_dbk_attributes_khr_onnx_inference *attrs = ort->attributes;
  OrtValue **outputs = NULL;
  OrtValue **inputs = NULL;
  OrtRunOptions *run_options = NULL;
  size_t num_inputs = ort->attributes->num_inputs;
  size_t num_outputs = ort->attributes->num_outputs;
  size_t num_tensors = num_inputs + num_outputs;
  const char **input_names
    = (const char **)ort->attributes->input_tensor_names;
  const char **output_names
    = (const char **)ort->attributes->output_tensor_names;

  assert (input_offsets != NULL);
  assert (input_data != NULL);
  assert (output_offsets != NULL);
  assert (output_storage != NULL);


  inputs = calloc (num_inputs, sizeof (OrtValue *));
  for (size_t i = 0; i < num_inputs; ++i)
    {
      const cl_tensor_desc *meta = &attrs->input_tensor_descs[i];
      void *ptr = input_data + input_offsets[i];
      size_t len = pocl_tensor_type_size (meta->dtype);
      size_t shape_len = meta->rank;
      ONNXTensorElementDataType type
        = cl_to_onnxrt_tensor_element_type (meta->dtype);
      int64_t shape[CL_MEM_MAX_TENSOR_RANK] = { 0 };
      for (size_t dim = 0; dim < shape_len; ++dim)
        {
          shape[dim] = meta->shape[dim];
          len *= shape[dim];
        }
      CHECK_ORT_CALL (CreateTensorWithDataAsOrtValue (
        ort->mem_info, ptr, len, shape, shape_len, type, &inputs[i]));
    }
  outputs = calloc (num_outputs, sizeof (OrtValue *));
  for (size_t i = 0; i < num_outputs; ++i)
    {
      const cl_tensor_desc *meta = &attrs->output_tensor_descs[i];
      void *ptr = output_storage + output_offsets[i];
      size_t len = pocl_tensor_type_size (meta->dtype);
      size_t shape_len = meta->rank;
      ONNXTensorElementDataType type
        = cl_to_onnxrt_tensor_element_type (meta->dtype);
      int64_t shape[CL_MEM_MAX_TENSOR_RANK] = { 0 };
      for (size_t dim = 0; dim < shape_len; ++dim)
        {
          shape[dim] = meta->shape[dim];
          len *= shape[dim];
        }
      CHECK_ORT_CALL (CreateTensorWithDataAsOrtValue (
        ort->mem_info, ptr, len, shape, shape_len, type, &outputs[i]));
    }

  CHECK_ORT_CALL (Run (ort->session, run_options, input_names,
                       (const OrtValue **)inputs, num_inputs, output_names,
                       num_outputs, outputs));

ERROR:
  if (inputs)
    {
      for (size_t i = 0; i < num_inputs; ++i)
        {
          if (inputs[i])
            ort_api->ReleaseValue (inputs[i]);
        }
      POCL_MEM_FREE (inputs);
    }
  if (outputs)
    {
      for (size_t i = 0; i < num_outputs; ++i)
        {
          if (outputs[i])
            ort_api->ReleaseValue (outputs[i]);
        }
      POCL_MEM_FREE (outputs);
    }

  return err;
}

