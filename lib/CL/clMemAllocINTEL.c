/* OpenCL runtime library: cl{Host,Device,Shared}MemAllocINTEL()

   Copyright (c) 2023 Michal Babej / Intel Finland Oy
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "devices.h"
#include "pocl_shared.h"
#include "pocl_util.h"
#include "utlist.h"

extern unsigned long usm_buffer_c;

static void *
pocl_usm_alloc (unsigned alloc_type, cl_context context, cl_device_id device,
                const cl_mem_properties_intel *properties, size_t size,
                cl_uint alignment, cl_int *errcode_ret)
{
  unsigned i;
  int p, errcode;
  cl_mem_alloc_flags_intel flags = 0;
  void *ptr = NULL;

  if (properties)
    {
      i = 0;
      while (properties[i])
        {
          if (properties[i] == CL_MEM_ALLOC_FLAGS_INTEL)
            {
              flags = properties[i + 1];
            }
          else
            {
              POCL_GOTO_ERROR_ON (1, CL_INVALID_PROPERTY,
                                  "Unknown property found in "
                                  "cl_mem_properties_intel: %" PRIu64 "\n",
                                  properties[i]);
            }
          i += 2;
        }
    }

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

  POCL_GOTO_ERROR_ON ((!context->usm_allocdev), CL_INVALID_OPERATION,
                      "None of the devices in this context is USM-capable\n");

  if (device)
    {
      POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (device)), CL_INVALID_DEVICE);
      POCL_GOTO_ERROR_ON ((device->ops->usm_alloc == NULL),
                          CL_INVALID_OPERATION,
                          "The device in argument is not USM-capable\n");
    }
  else
    device = context->usm_allocdev;

  switch (alloc_type)
    {
    case CL_MEM_TYPE_HOST_INTEL:
      POCL_GOTO_ERROR_ON ((device->host_usm_capabs == 0), CL_INVALID_OPERATION,
                          "Device does not support Host USM allocations\n");
      break;
    case CL_MEM_TYPE_DEVICE_INTEL:
      POCL_GOTO_ERROR_ON ((device->device_usm_capabs == 0),
                          CL_INVALID_OPERATION,
                          "Device does not support Device USM allocations\n");
      break;
    case CL_MEM_TYPE_SHARED_INTEL:
      POCL_GOTO_ERROR_ON ((device->single_shared_usm_capabs == 0),
                          CL_INVALID_OPERATION,
                          "Device does not support Shared USM allocations\n");
      break;
    default:
      POCL_GOTO_ERROR_ON (1, CL_INVALID_PROPERTY,
                          "Unknown USM AllocType requested\n");
    }

  POCL_GOTO_ERROR_COND ((size == 0), CL_INVALID_BUFFER_SIZE);

  POCL_GOTO_ERROR_ON ((size > context->max_mem_alloc_size),
                      CL_INVALID_BUFFER_SIZE,
                      "size(%zu) > CL_DEVICE_MAX_MEM_ALLOC_SIZE value "
                      "for some device in context\n",
                      size);

  /* these flags are mutually exclusive */
  const cl_mem_alloc_flags_intel placement_flags
      = CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL
        | CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL;
  POCL_GOTO_ERROR_ON (((flags & placement_flags) == placement_flags),
                      CL_INVALID_PROPERTY,
                      "placement flags are mutually exclusive\n");

  POCL_GOTO_ERROR_ON (
      ((flags & placement_flags) && (alloc_type != CL_MEM_TYPE_SHARED_INTEL)),
      CL_INVALID_PROPERTY,
      "placement flags are only valid for 'shared' allocations\n");

  const cl_mem_alloc_flags_intel valid_flags
      = (CL_MEM_ALLOC_INITIAL_PLACEMENT_DEVICE_INTEL
         | CL_MEM_ALLOC_INITIAL_PLACEMENT_HOST_INTEL
         | CL_MEM_ALLOC_WRITE_COMBINED_INTEL);
  POCL_GOTO_ERROR_ON ((flags & (~valid_flags)), CL_INVALID_PROPERTY,
                      "flags argument "
                      "contains invalid bits (unknown flags)\n");

  /* Alignment is the minimum alignment in bytes for the requested device
   * allocation. It must be a power of two and must be equal to or smaller
   * than the size of the largest data type supported by device. If alignment
   * is 0, a default alignment will be used that is equal to the size of
   * largest data type supported by device. */
  if (alignment == 0)
    alignment = device->min_data_type_align_size;

  p = __builtin_popcount (alignment);
  POCL_GOTO_ERROR_ON ((p > 1), CL_INVALID_VALUE,
                      "aligment argument must be a power of 2\n");

  pocl_raw_ptr *item = calloc (1, sizeof (pocl_raw_ptr));
  POCL_GOTO_ERROR_ON ((item == NULL), CL_OUT_OF_HOST_MEMORY,
                      "out of host memory\n");

  ptr
      = device->ops->usm_alloc (device, alloc_type, flags, size, &errcode);
  if (errcode != CL_SUCCESS)
    goto ERROR;
  POCL_GOTO_ERROR_ON ((ptr == NULL), CL_OUT_OF_RESOURCES,
                      "Device failed to allocate USM memory");

  POCL_LOCK_OBJ (context);

  /* Register the pointer as an raw pointer so clCreateBuffer() detects it. */
  item->vm_ptr = ptr;
  item->size = size;
  item->kind = POCL_RAW_PTR_INTEL_USM;
  if (alloc_type == CL_MEM_TYPE_DEVICE_INTEL ||
      alloc_type == CL_MEM_TYPE_SHARED_INTEL)
    item->device = device;
  else
    item->device = NULL;
  item->usm_properties.alloc_type = alloc_type;
  item->usm_properties.flags = flags;
  DL_APPEND (context->raw_ptrs, item);
  POCL_UNLOCK_OBJ (context);

  /* Create a shadow cl_mem object for keeping track of the USM
     allocation and to implement automated migrations, cl_pocl_content_size,
     etc. for USM using the same code paths as with cl_mems. */
  cl_mem clmem_shadow = POname (clCreateBuffer) (
      context,
      CL_MEM_DEVICE_ADDRESS_EXT | CL_MEM_DEVICE_PRIVATE_EXT
      | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
      size, ptr, &errcode);

  if (errcode != CL_SUCCESS)
    {
      POCL_MSG_ERR ("Failed to allocate memory a shadow cl_mem object.\n");
      return NULL;
    }

  item->shadow_cl_mem = clmem_shadow;

  POname (clRetainContext) (context);

  POCL_MSG_PRINT_MEMORY ("Allocated USM: PTR %p, SIZE %zu, FLAGS %" PRIu64
                         " \n",
                         ptr, size, flags);

  POCL_ATOMIC_INC (usm_buffer_c);

ERROR:
  if (errcode_ret)
    {
      *errcode_ret = errcode;
    }
  return ptr;
}

CL_API_ENTRY void *CL_API_CALL
POname (clHostMemAllocINTEL) (cl_context context,
                              const cl_mem_properties_intel *properties,
                              size_t size, cl_uint alignment,
                              cl_int *errcode_ret) CL_API_SUFFIX__VERSION_2_0
{
  return pocl_usm_alloc (CL_MEM_TYPE_HOST_INTEL, context, NULL, properties,
                         size, alignment, errcode_ret);
}
POsym (clHostMemAllocINTEL)

    CL_API_ENTRY void *CL_API_CALL POname (clDeviceMemAllocINTEL) (
        cl_context context, cl_device_id device,
        const cl_mem_properties_intel *properties, size_t size,
        cl_uint alignment, cl_int *errcode_ret) CL_API_SUFFIX__VERSION_2_0
{
  return pocl_usm_alloc (CL_MEM_TYPE_DEVICE_INTEL, context, device, properties,
                         size, alignment, errcode_ret);
}
POsym (clDeviceMemAllocINTEL)

    CL_API_ENTRY void *CL_API_CALL POname (clSharedMemAllocINTEL) (
        cl_context context, cl_device_id device,
        const cl_mem_properties_intel *properties, size_t size,
        cl_uint alignment, cl_int *errcode_ret) CL_API_SUFFIX__VERSION_2_0
{
  return pocl_usm_alloc (CL_MEM_TYPE_SHARED_INTEL, context, device, properties,
                         size, alignment, errcode_ret);
}
POsym (clSharedMemAllocINTEL)
