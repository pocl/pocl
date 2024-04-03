/* OpenCL runtime library: clGetDeviceInfo()

   Copyright (c) 2012 Erik Schnetter
                 2023-2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "pocl_util.h"

CL_API_ENTRY cl_int CL_API_CALL
POname (clGetMemObjectInfo) (
    cl_mem memobj, cl_mem_info param_name, size_t param_value_size,
    void *param_value, size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (memobj)),
                          CL_INVALID_MEM_OBJECT);

  switch (param_name) {
  case CL_MEM_TYPE:
    POCL_RETURN_GETINFO (cl_mem_object_type, memobj->type);
  case CL_MEM_FLAGS:
    POCL_RETURN_GETINFO (cl_mem_flags, memobj->flags);
  case CL_MEM_SIZE:
    POCL_RETURN_GETINFO (size_t, memobj->size);
  case CL_MEM_HOST_PTR:
    if (memobj->flags & CL_MEM_USE_HOST_PTR)
      POCL_RETURN_GETINFO (void *,
                           (void *)(memobj->parent
                                        ? ((char *)memobj->parent->mem_host_ptr
                                           + memobj->origin)
                                        : memobj->mem_host_ptr));
    else
      POCL_RETURN_GETINFO (void *, NULL);
  case CL_MEM_MAP_COUNT:
    POCL_RETURN_GETINFO (cl_uint, memobj->map_count);
  case CL_MEM_REFERENCE_COUNT:
    POCL_RETURN_GETINFO (cl_uint, memobj->pocl_refcount);
  case CL_MEM_CONTEXT:
    POCL_RETURN_GETINFO (cl_context, memobj->context);
  case CL_MEM_ASSOCIATED_MEMOBJECT:
    POCL_RETURN_GETINFO (cl_mem, memobj->parent);
  case CL_MEM_USES_SVM_POINTER:
    {
      pocl_raw_ptr *item = pocl_find_raw_ptr_with_vm_ptr (
          memobj->context, memobj->mem_host_ptr);
      POCL_RETURN_GETINFO (cl_bool, (item != NULL));
    }
  case CL_MEM_OFFSET:
    if (memobj->parent == NULL)
      POCL_RETURN_GETINFO (size_t, 0);
    else
      POCL_RETURN_GETINFO (size_t, memobj->origin);
  case CL_MEM_PROPERTIES:
    POCL_RETURN_GETINFO_ARRAY (cl_mem_properties, memobj->num_properties,
                               memobj->properties);
  case CL_MEM_DEVICE_PTRS_EXT:
    {
      POCL_RETURN_ERROR_COND (!memobj->has_device_address,
                              CL_INVALID_MEM_OBJECT);
      POCL_RETURN_GETINFO_SIZE_CHECK (
          memobj->context->num_devices
          * sizeof (cl_mem_device_address_pair_EXT));
      cl_mem_device_address_pair_EXT *addresses
          = (cl_mem_device_address_pair_EXT *)param_value;
      cl_context context = memobj->context;
      for (size_t i = 0; i < context->num_devices; ++i)
        {
          cl_device_id dev = context->devices[i];
          pocl_mem_identifier *p = &memobj->device_ptrs[dev->global_mem_id];
          addresses[i].device = dev;
          addresses[i].address = p->device_addr;
        }
      return CL_SUCCESS;
    }
  case CL_MEM_DEVICE_PTR_EXT:
    {
      POCL_RETURN_ERROR_COND (!memobj->has_device_address,
                              CL_INVALID_MEM_OBJECT);

      POCL_RETURN_GETINFO_SIZE_CHECK (memobj->context->num_devices
                                      * sizeof (void *));
      void **addr = (void **)param_value;
      *addr = NULL;
      cl_context context = memobj->context;
      for (size_t i = 0; i < context->num_devices; ++i)
        {
          cl_device_id dev = context->devices[i];
          pocl_mem_identifier *p = &memobj->device_ptrs[dev->global_mem_id];
          POCL_MSG_PRINT_MEMORY (
              "Got dev ptr %p for device %d (gmem id %d).\n", p->device_addr,
              i, dev->global_mem_id);
          if (*addr != NULL && p->device_addr != *addr)
            POCL_ABORT ("All devices do not have the same cl_mem address!");
          *addr = p->device_addr;
        }
      return CL_SUCCESS;
    }
  }
  return CL_INVALID_VALUE;
}
POsym(clGetMemObjectInfo)
