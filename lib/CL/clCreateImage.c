/* OpenCL runtime library: clCreateImage()

   Copyright (c) 2012 Timo Viitanen, 2013 Ville Korhonen
   
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

extern CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateImage)(cl_context              context,
              cl_mem_flags            flags,
              const cl_image_format * image_format,
              const cl_image_desc *   image_desc, 
              void *                  host_ptr,
              cl_int *                errcode_ret) 
CL_API_SUFFIX__VERSION_1_2
{


    cl_mem mem;
    cl_device_id device_id;
    void *device_ptr;
    unsigned i, j;
    cl_uint num_entries = 0;
    cl_image_format *supported_image_formats;
    int size;
    int errcode;
    int row_pitch;
    cl_channel_type ch_type = image_format->image_channel_data_type;
    cl_channel_order ch_order = image_format->image_channel_order;
    int dev_elem_size;
    int dev_channels;
    
    
        
    if (context == NULL) 
      {
      errcode = CL_INVALID_CONTEXT;
        goto ERROR;
      }  
    
    if (image_desc == NULL)
      {
        errcode = CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
        goto ERROR;
      }

    if (image_desc->num_mip_levels != 0 || 
        image_desc->num_samples != 0 )
        POCL_ABORT_UNIMPLEMENTED();
        
        
    errcode = POname(clGetSupportedImageFormats)
      (context, flags, image_desc->image_type, 0, NULL, &num_entries);
    
    if (errcode != CL_SUCCESS || num_entries == 0) 
      {
        goto ERROR;
      } 

    supported_image_formats = malloc( num_entries * sizeof(cl_image_format) );
    if (supported_image_formats == NULL)
      {
        errcode = CL_OUT_OF_HOST_MEMORY;
        goto ERROR;
      }
    
    errcode = POname(clGetSupportedImageFormats)
      (context, flags, image_desc->image_type, num_entries, 
       supported_image_formats, NULL);
    
    if (errcode != CL_SUCCESS )
      {
        goto ERROR_CLEAN_MEM_AND_DEV;
      }

    for( i = 0; i < num_entries; i++)
      {
        if ( supported_image_formats[i].image_channel_order == 
             image_format->image_channel_order &&
             supported_image_formats[i].image_channel_data_type ==
             image_format->image_channel_data_type )
          {
            goto TYPE_SUPPORTED;
          }
      }
    errcode = CL_INVALID_VALUE;
    goto ERROR_CLEAN_MEM_AND_DEV;

TYPE_SUPPORTED:

    if (image_desc->image_type != CL_MEM_OBJECT_IMAGE2D &&
        image_desc->image_type != CL_MEM_OBJECT_IMAGE3D)
        POCL_ABORT_UNIMPLEMENTED();

    mem = (cl_mem) malloc(sizeof(struct _cl_mem));
    if (mem == NULL)
      {
        errcode = CL_OUT_OF_HOST_MEMORY;
        goto ERROR_CLEAN_MEM;
      }
        
    /* element size */
    if (ch_type == CL_SNORM_INT8 || ch_type == CL_UNORM_INT8 ||
        ch_type == CL_SIGNED_INT8 || ch_type == CL_UNSIGNED_INT8 )
      {
        dev_elem_size = 1; /* 1 byte */
      }
    else if (ch_type == CL_UNSIGNED_INT32 || 
             ch_type == CL_FLOAT || ch_type == CL_UNORM_INT_101010 )
      {
        dev_elem_size = 4; /* 32bit -> 4 bytes */
      }
    else if (ch_type == CL_SNORM_INT16 || ch_type == CL_UNORM_INT16 ||
             ch_type == CL_SIGNED_INT16 || ch_type == CL_UNSIGNED_INT16 ||
             ch_type == CL_UNORM_SHORT_555 || ch_type == CL_UNORM_SHORT_565)
      {
        dev_elem_size = 2; /* 16bit -> 2 bytes */
      }

    size = image_desc->image_width * image_desc->image_height * 
      dev_elem_size * dev_channels;
    

    POCL_INIT_OBJECT(mem);
    mem->parent = NULL;
    mem->map_count = 0;
    mem->mappings = NULL;
    mem->type = image_desc->image_type;
    mem->flags = flags;
    mem->is_image = CL_TRUE;

    mem->device_ptrs = (void **) malloc(context->num_devices * sizeof(void *));
    if (mem->device_ptrs == NULL) {
        errcode = CL_OUT_OF_HOST_MEMORY;
        goto ERROR_CLEAN_MEM;
    }  

    if (image_desc->image_row_pitch == 0)
        row_pitch = image_desc->image_width;
    
    /*debuggii*/
    /*printf("image_row_pitch = %d, image_width = %d, size = %d\n", 
           image_desc->image_row_pitch, image_desc->image_width, size);
    */
    /*
    if (image_desc->image_row_pitch != image_desc->image_width)
        POCL_ABORT_UNIMPLEMENTED();
    */

    
    mem->size = size;
    mem->context = context;
  
    mem->image_width = image_desc->image_width;
    mem->image_height = image_desc->image_height;
    mem->image_depth = image_desc->image_depth;
    mem->image_row_pitch = image_desc->image_row_pitch;
    mem->image_slice_pitch = image_desc->image_slice_pitch;
    mem->image_channel_data_type = image_format->image_channel_data_type;
    mem->image_channel_order = image_format->image_channel_order;
        
    for (i = 0; i < context->num_devices; ++i)
    {
        //printf("i arvo = %d\n", i);
        if (i > 0)
            POname(clRetainMemObject) (mem);
        device_id = context->devices[i];
        device_ptr = device_id->malloc(device_id->data, 0, size, NULL);
        
        if (device_ptr == NULL) {
            errcode = CL_MEM_OBJECT_ALLOCATION_FAILURE;
            goto ERROR_CLEAN_MEM_AND_DEV;
        }
        
        mem->device_ptrs[i] = device_ptr;
        /* The device allocator allocated from a device-host shared memory. */
        if (flags & CL_MEM_ALLOC_HOST_PTR ||
            flags & CL_MEM_USE_HOST_PTR)
            POCL_ABORT_UNIMPLEMENTED();
        
        if (flags & CL_MEM_COPY_HOST_PTR)  
        {
            size_t origin[3] = { 0, 0, 0 };
            size_t region[3] = { image_desc->image_width, 
                                 image_desc->image_height, 1 };
            pocl_write_image(mem,
                             context->devices[i],
                             origin,
                             region,
                             image_desc->image_row_pitch,
                             image_desc->image_slice_pitch,
                             host_ptr);
        }
    }
    
    POCL_RETAIN_OBJECT(context);
    
    if (errcode_ret != NULL)
        *errcode_ret = CL_SUCCESS;
    
    return mem;
    
ERROR_CLEAN_MEM_AND_DEV:
    for (j = 0; j < i; ++j) {
        device_id = context->devices[j];
        device_id->free(device_id->data, 0, mem->device_ptrs[j]);
    }
ERROR_CLEAN_MEM:
    free(mem);
ERROR:
    if (errcode_ret) {
        *errcode_ret = errcode;
    }
    return NULL;
    

    
/*
  if (image_desc->num_mip_levels != 0 || 
      image_desc->num_samples != 0 )
    POCL_ABORT_UNIMPLEMENTED();
  
  if (image_desc->image_type != CL_MEM_OBJECT_IMAGE2D)
    POCL_ABORT_UNIMPLEMENTED();
  
  return POname(clCreateImage2D) (context,
                   flags,
                   image_format,
                   image_desc->image_width,
                   image_desc->image_height,
                   image_desc->image_row_pitch,
                   host_ptr,
                   errcode_ret);
*/
}
POsym(clCreateImage)
