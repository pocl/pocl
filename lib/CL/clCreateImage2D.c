/* OpenCL runtime library: clCreateImage2D()

   Copyright (c) 2012 Timo Viitanen
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/
#include "pocl_cl.h"
#include "assert.h"
#include "pocl_image_util.h"

CL_API_ENTRY cl_mem CL_API_CALL
POclCreateImage2D(cl_context              context,
                cl_mem_flags            flags,
                const cl_image_format * image_format,
                size_t                  width,
                size_t                  height,
                size_t                  image_row_pitch, 
                void *                  host_ptr,
                cl_int *                errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
  cl_mem mem;
  cl_device_id device_id;
  void *device_ptr;
  unsigned i, j;
  int size;

  if (context == NULL)
    POCL_ERROR(CL_INVALID_CONTEXT);

  mem = (cl_mem) malloc(sizeof(struct _cl_mem));
  if (mem == NULL)
    POCL_ERROR(CL_OUT_OF_HOST_MEMORY);

  POCL_INIT_OBJECT(mem);
  mem->parent = NULL;
  mem->map_count = 0;
  mem->mappings = NULL;
  mem->type = CL_MEM_OBJECT_IMAGE2D;
  mem->flags = flags;
  mem->is_image = CL_TRUE;
  POCL_INIT_ICD_OBJECT(mem);
  
  cl_channel_order order = image_format->image_channel_order;
  cl_channel_type type = image_format->image_channel_data_type;
  
  int dev_elem_size = sizeof(cl_float); //TODO
  int dev_channels = 4;
  
  if (image_row_pitch == 0)
    image_row_pitch = width;
  
  if (image_row_pitch != width)
    POCL_ABORT_UNIMPLEMENTED();
  
  size = width * height * dev_elem_size * dev_channels;
  
  mem->device_ptrs = (void **) malloc(context->num_devices * sizeof(void *));
  if (mem->device_ptrs == NULL)
    {
      free(mem);
      POCL_ERROR(CL_OUT_OF_HOST_MEMORY);
    }  
  
  int host_channels;
  
  if (order == CL_RGBA)
    host_channels=4;
  else if (order == CL_R)
    host_channels=1;
  else
    POCL_ABORT_UNIMPLEMENTED();
  
  mem->size = size;
  mem->context = context;
  
  mem->image_width = width;
  mem->image_height = height;
  mem->image_row_pitch = image_row_pitch;
  mem->image_channel_data_type = type;
  mem->image_channel_order = order;
    
  for (i = 0; i < context->num_devices; ++i)
    {
      if (i > 0)
        POclRetainMemObject (mem);
      device_id = context->devices[i];
      device_ptr = device_id->malloc(device_id->data, 0, size, NULL);
      
      if (device_ptr == NULL)
        {
          for (j = 0; j < i; ++j)
            {
              device_id = context->devices[j];
              device_id->free(device_id->data, 0, mem->device_ptrs[j]);
            }
          free(mem);
          POCL_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        }
      mem->device_ptrs[i] = device_ptr;
      /* The device allocator allocated from a device-host shared memory. */
      if (flags & CL_MEM_ALLOC_HOST_PTR ||
          flags & CL_MEM_USE_HOST_PTR)
        POCL_ABORT_UNIMPLEMENTED();
      
      if (flags & CL_MEM_COPY_HOST_PTR)  
        {
          size_t origin[3] = { 0, 0, 0 };
          size_t region[3] = { width, height, 1 };
          pocl_write_image( mem,
                            context->devices[i],
                            origin,
                            region,
                            0,
                            1,
                            host_ptr );
        }
    }

  POCL_RETAIN_OBJECT(context);
  
  
  if (errcode_ret != NULL)
    *errcode_ret = CL_SUCCESS;
  
  return mem;
}
POsym(clCreateImage2D) 
