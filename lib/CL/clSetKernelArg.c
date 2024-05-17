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

#include "config.h"
#include "pocl_cl.h"
#include "pocl_debug.h"
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
      && !(pi->type == POCL_ARG_TYPE_POINTER
           && *(const intptr_t *)arg_value == 0))
    {
      void *value;
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
          if (value == NULL)
            {
              return CL_OUT_OF_HOST_MEMORY;
            }
        }

      if ((pi->type == POCL_ARG_TYPE_POINTER) && (arg_value != NULL))
        {
          cl_mem buf = *(const cl_mem *)arg_value;
          if (buf->parent != NULL)
            {
              p->offset = buf->origin;
              buf = buf->parent;
            }
          else
            {
              p->offset = 0;
            }
          memcpy (value, &buf, arg_size);
        }
      else
        memcpy (value, arg_value, arg_size);
      p->value = value;
    }

  p->size = arg_size;
  p->is_set = 1;

  return CL_SUCCESS;
}
POsym(clSetKernelArg)
