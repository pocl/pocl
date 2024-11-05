/* OpenCL runtime library: clSetKernelArg()

   Copyright (c) 2011 Universidad Rey Juan Carlos
                 2013 Pekka Jääskeläinen / Tampere University
                 2023 Pekka Jääskeläinen / Intel Finland Oy

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

#include "CL/cl.h"
#include "config.h"
#include "pocl_cl.h"
#include "pocl_debug.h"
#include "pocl_tensor_util.h"
#include "pocl_util.h"
#include <assert.h>
#include <stdbool.h>
#include <string.h>

static char hex[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8',
                      '9', 'A', 'B', 'C', 'D', 'E', 'F', 'X', 'Y' };

static void
dump_hex (const char *data, size_t size, char *buffer)
{
  size_t j = 0;
  for (size_t i = 0; i < size; ++i)
    {
      if (i % 4 == 0)
        buffer[j++] = ' ';
      char c = data[i];
      buffer[j] = hex[(c & 0xF0) >> 4];
      buffer[j + 1] = hex[c & 0x0F];
      j += 2;
    }
  buffer[j] = 0;
}

static int
pocl_verify_dbk_kernel_arg (cl_mem buf, const cl_tensor_desc *desc)
{
  POCL_RETURN_ERROR_ON ((buf->is_tensor == CL_FALSE), CL_INVALID_ARG_VALUE,
                        "the cl_mem argument must be a tensor\n");

  const cl_tensor_properties *P = desc->properties;
  char tensor_mutable_layout = 0;
  char tensor_mutable_shape = 0;
  char tensor_mutable_dtype = 0;
  while (*P)
    {
      switch (*P)
        {
        case CL_TENSOR_PROPERTY_MUTABLE_SHAPE:
          tensor_mutable_shape = 1;
          break;
        case CL_TENSOR_PROPERTY_MUTABLE_DTYPE:
          tensor_mutable_dtype = 1;
          break;
        case CL_TENSOR_PROPERTY_MUTABLE_LAYOUT:
          tensor_mutable_layout = 1;
          break;
        default:
          break;
        }
      ++P;
    }

  POCL_RETURN_ERROR_ON ((buf->tensor_rank != desc->rank), CL_INVALID_ARG_VALUE,
                        "the cl_mem Tensor argument has incorrect rank\n");

  POCL_RETURN_ERROR_ON (
    (buf->tensor_dtype != desc->dtype && tensor_mutable_dtype == CL_FALSE),
    CL_INVALID_ARG_VALUE,
    "the cl_mem Tensor argument must have identical dtype\n");

  POCL_RETURN_ERROR_ON (
    (buf->tensor_layout_type != desc->layout_type
     && tensor_mutable_layout == CL_FALSE),
    CL_INVALID_ARG_VALUE,
    "the cl_mem Tensor argument has incorrect layout type\n");
  int cmp = 0;
  switch (desc->layout_type)
    {
    case CL_TENSOR_LAYOUT_ML:
      cmp = memcmp (buf->tensor_layout, desc->layout,
                    sizeof (cl_tensor_layout_ml));
      break;
    case CL_TENSOR_LAYOUT_BLAS:
      cmp = memcmp (buf->tensor_layout, desc->layout,
                    sizeof (cl_tensor_layout_blas));
      break;
    default:
      break;
    }
  POCL_RETURN_ERROR_ON (
    (cmp != 0 && tensor_mutable_layout == CL_FALSE), CL_INVALID_ARG_VALUE,
    "the cl_mem Tensor layout is different, and mutable layout == false\n");

  cmp = memcmp (buf->tensor_shape, desc->shape,
                (desc->rank * sizeof (cl_ulong)));
  POCL_RETURN_ERROR_ON (
    (cmp != 0 && tensor_mutable_shape == CL_FALSE), CL_INVALID_ARG_VALUE,
    "the cl_mem Tensor shape is different, and mutable dims == false\n");

  return CL_SUCCESS;
}

static int
pocl_verify_dbk_kernel_args (cl_mem buf,
                             pocl_kernel_metadata_t *meta,
                             size_t arg_index)
{
  switch (meta->builtin_kernel_id)
    {
    case POCL_CDBI_DBK_EXP_GEMM:
      {
        const cl_dbk_attributes_exp_gemm *Attrs
          = (cl_dbk_attributes_exp_gemm *)meta->builtin_kernel_attrs;
        if (arg_index == 0)
          return pocl_verify_dbk_kernel_arg (buf, &Attrs->a);
        if (arg_index == 1)
          return pocl_verify_dbk_kernel_arg (buf, &Attrs->b);
        if (arg_index == 2)
          return pocl_verify_dbk_kernel_arg (buf, &Attrs->c_in);
        if (arg_index == 3)
          return pocl_verify_dbk_kernel_arg (buf, &Attrs->c_out);
        POCL_RETURN_ERROR (CL_INVALID_ARG_INDEX, "invalid arg index to "
                                                 "POCL_CDBI_DBK_EXP_GEMM");
      }
    case POCL_CDBI_DBK_EXP_MATMUL:
      {
        const cl_dbk_attributes_exp_matmul *Attrs
          = (const cl_dbk_attributes_exp_matmul *)meta->builtin_kernel_attrs;
        if (arg_index == 0)
          return pocl_verify_dbk_kernel_arg (buf, &Attrs->a);
        if (arg_index == 1)
          return pocl_verify_dbk_kernel_arg (buf, &Attrs->b);
        if (arg_index == 2)
          return pocl_verify_dbk_kernel_arg (buf, &Attrs->c);
        POCL_RETURN_ERROR (CL_INVALID_ARG_INDEX, "invalid arg index to "
                                                 "POCL_CDBI_DBK_EXP_MATMUL");
      }
    case POCL_CDBI_DBK_EXP_JPEG_ENCODE:
    case POCL_CDBI_DBK_EXP_JPEG_DECODE:
      return CL_SUCCESS;
    case POCL_CDBI_DBK_EXP_ONNX_INFERENCE:
      {
        const cl_dbk_attributes_exp_onnx_inference *attrs
          = (const cl_dbk_attributes_exp_onnx_inference *)
              meta->builtin_kernel_attrs;

        /* Input offsets */
        if (arg_index == 0
            && buf->size < attrs->num_inputs * sizeof (uint64_t))
          return CL_OUT_OF_RESOURCES;

        /* Input tensor data */
        if (arg_index == 1)
          {
            size_t total_input_size = 0;
            for (size_t i = 0; i < attrs->num_inputs; ++i)
              {
                size_t data_len
                  = pocl_tensor_type_size (attrs->input_tensor_descs[i].dtype);
                for (size_t dim = 0; dim < attrs->input_tensor_descs[i].rank;
                     ++dim)
                  {
                    data_len *= attrs->input_tensor_descs[i].shape[dim];
                  }
                total_input_size += data_len;
              }
            if (buf->size < total_input_size)
              return CL_OUT_OF_RESOURCES;
          }

        /* Output offsets */
        if (arg_index == 2
            && buf->size < attrs->num_outputs * sizeof (uint64_t))
          return CL_OUT_OF_RESOURCES;

        /* Output tensor data */
        if (arg_index == 3)
          {
            size_t total_output_size = 0;
            for (size_t i = 0; i < attrs->num_outputs; ++i)
              {
                size_t data_len = pocl_tensor_type_size (
                  attrs->output_tensor_descs[i].dtype);
                for (size_t dim = 0; dim < attrs->output_tensor_descs[i].rank;
                     ++dim)
                  {
                    data_len *= attrs->output_tensor_descs[i].shape[dim];
                  }
                total_output_size += data_len;
              }

            if (buf->size < total_output_size)
              return CL_OUT_OF_RESOURCES;
          }
        return CL_SUCCESS;
      }
  default:
      {
        POCL_MSG_ERR ("pocl_verify_dbk_kernel_args called on "
                      "unknown/unsupported DBK type %d.\n",
                      meta->builtin_kernel_id);
        return CL_INVALID_KERNEL;
      }
    }
}

CL_API_ENTRY cl_int CL_API_CALL
POname(clSetKernelArg)(cl_kernel kernel,
               cl_uint arg_index,
               size_t arg_size,
               const void *arg_value) CL_API_SUFFIX__VERSION_1_0
{
  size_t arg_alignment, arg_alloc_size;
  struct pocl_argument *p;
  struct pocl_argument_info *pi;

  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (kernel)), CL_INVALID_KERNEL);

  POCL_RETURN_ERROR_ON ((arg_index >= kernel->meta->num_args),
                        CL_INVALID_ARG_INDEX,
                        "This kernel has %u args, cannot set arg %u\n",
                        (unsigned)kernel->meta->num_args, (unsigned)arg_index);

  POCL_RETURN_ERROR_ON((kernel->dyn_arguments == NULL), CL_INVALID_KERNEL,
    "This kernel has no arguments that could be set\n");

  pi = &(kernel->meta->arg_info[arg_index]);
  int is_local = ARGP_IS_LOCAL (pi);

  const void *ptr_value = NULL;
  if (((pi->type == POCL_ARG_TYPE_POINTER)
       || (pi->type == POCL_ARG_TYPE_IMAGE))
      && arg_value)
    ptr_value = *(const void **)arg_value;

  if (POCL_DEBUGGING_ON)
    {
      const void *ptr_value = NULL;
      uint32_t uint32_value = 0;
      uint64_t uint64_value = 0;
      if (((pi->type == POCL_ARG_TYPE_POINTER)
           || (pi->type == POCL_ARG_TYPE_IMAGE))
          && arg_value)
        {
          ptr_value = *(const void **)arg_value;
        }
      else
        {
          if (arg_value && (arg_size == 4))
            uint32_value = *(uint32_t *)arg_value;
          if (arg_value && (arg_size == 8))
            uint64_value = *(uint64_t *)arg_value;
        }

      char *hexval = NULL;
      if (arg_value && (arg_size < 1024))
        {
          hexval = (char *)alloca ((arg_size * 2) + (arg_size / 4) + 8);
          dump_hex (arg_value, arg_size, hexval);
        }

      POCL_MSG_PRINT_GENERAL ("Kernel %15s || SetArg idx %3u || %8s || "
                              "Local %1i || Size %6zu || Value %p || "
                              "Pointer %p || *(uint32*)Value: %8u || "
                              "*(uint64*)Value: %8" PRIu64 " ||\nHex Value: %s\n",
                              kernel->name, arg_index, pi->type_name, is_local,
                              arg_size, arg_value, ptr_value, uint32_value,
                              uint64_value, hexval);
    }

  POCL_RETURN_ERROR_ON (
      ((arg_value != NULL) && is_local), CL_INVALID_ARG_VALUE,
      "arg_value != NULL and arg %u is in local address space\n", arg_index);

  /* Trigger CL_INVALID_ARG_VALUE if arg_value specified is NULL
   * for an argument that is not declared with the __local qualifier. */
  POCL_RETURN_ERROR_ON (
      ((arg_value == NULL) && (!is_local)
       && (pi->type != POCL_ARG_TYPE_POINTER)),
      CL_INVALID_ARG_VALUE,
      "arg_value == NULL and arg %u is not in local address space\n",
      arg_index);

  /* Trigger CL_INVALID_ARG_SIZE if arg_size is zero
   * and the argument is declared with the __local qualifier. */
  POCL_RETURN_ERROR_ON (((arg_size == 0) && is_local), CL_INVALID_ARG_SIZE,
                        "arg_size == 0 and arg %u is in local address space\n",
                        arg_index);

  POCL_RETURN_ERROR_ON (
      ((pi->type == POCL_ARG_TYPE_SAMPLER) && (arg_value == NULL)),
      CL_INVALID_SAMPLER, "arg_value == NULL and arg is a cl_sampler\n");

  if (pi->type == POCL_ARG_TYPE_POINTER || pi->type == POCL_ARG_TYPE_IMAGE
      || pi->type == POCL_ARG_TYPE_SAMPLER)
    {
      POCL_RETURN_ERROR_ON (
          ((!is_local) && (arg_size != sizeof (cl_mem))), CL_INVALID_ARG_SIZE,
          "Arg %u is pointer/buffer/image, but arg_size (%zu) is "
          "not sizeof(cl_mem) == %zu\n",
          arg_index, arg_size, sizeof (cl_mem));
      if (ptr_value)
        {
          POCL_RETURN_ERROR_ON (
              !IS_CL_OBJECT_VALID ((const cl_mem)ptr_value),
              CL_INVALID_ARG_VALUE,
              "Arg %u is not a valid CL object\n", arg_index);
        }
    }
  else if (pi->type_size)
    {
      size_t as = arg_size;
      size_t as3 = arg_size;
      /* handle <type>3 vectors, we accept both <type>3 and <type>4 sizes */
      /* pi->type_size for <type>3 vectors is expected = sizeof(<type>4) */
      if (as % 3 == 0)
        as3 = (as / 3) * 4;
      POCL_RETURN_ERROR_ON (
          (pi->type_size != as && pi->type_size != as3), CL_INVALID_ARG_SIZE,
          "Arg %u is type %s, but arg_size (%zu) is not sizeof(type) == %u\n",
          arg_index, pi->type_name, arg_size, pi->type_size);
    }

  p = &(kernel->dyn_arguments[arg_index]);
  if (kernel->dyn_argument_storage == NULL)
    pocl_aligned_free (p->value);
  p->value = NULL;
  p->is_set = 0;
  p->is_readonly = 0;
  p->is_raw_ptr = 0;

  /* Even if the buffer/image is read-write, the kernel might be using it as
   * read-only */
  if ((kernel->meta->has_arg_metadata & POCL_HAS_KERNEL_ARG_ACCESS_QUALIFIER)
      && (pi->address_qualifier == CL_KERNEL_ARG_ADDRESS_GLOBAL))
    {
      if (pi->type == POCL_ARG_TYPE_IMAGE)
        {
          p->is_readonly
              = (pi->access_qualifier & CL_KERNEL_ARG_ACCESS_READ_ONLY ? 1
                                                                       : 0);
        }
      if (pi->type == POCL_ARG_TYPE_POINTER)
        {
          p->is_readonly
              = (pi->type_qualifier & CL_KERNEL_ARG_TYPE_CONST ? 1 : 0);
        }
    }

  if (arg_value != NULL
      && !(pi->type == POCL_ARG_TYPE_POINTER && ptr_value == 0))
    {
      void *value;
      if (kernel->meta->builtin_kernel_attrs && ptr_value)
        {
          cl_mem buf = (const cl_mem)ptr_value;
          int ret = pocl_verify_dbk_kernel_args (buf, kernel->meta, arg_index);
          if (ret)
            return ret;
        }

      if (kernel->dyn_argument_storage != NULL)
        value = kernel->dyn_argument_offsets[arg_index];
      else
        {
          /* FIXME: this is a kludge to determine an acceptable alignment,
           * we should probably extract the argument alignment from the
           * LLVM bytecode during kernel header generation. */
          arg_alignment = pocl_size_ceil2 (arg_size);
          if (arg_alignment >= MAX_EXTENDED_ALIGNMENT)
            arg_alignment = MAX_EXTENDED_ALIGNMENT;

          arg_alloc_size = arg_size;
          if (arg_alloc_size < arg_alignment)
            arg_alloc_size = arg_alignment;

          value = pocl_aligned_malloc (arg_alignment, arg_alloc_size);
          POCL_RETURN_ERROR_COND ((value == NULL), CL_OUT_OF_HOST_MEMORY);
        }

      memcpy (value, arg_value, arg_size);
      p->value = value;
    }

  p->size = arg_size;
  p->is_set = 1;

  return CL_SUCCESS;
}
POsym(clSetKernelArg)
