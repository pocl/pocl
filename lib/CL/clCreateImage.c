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
#include "pocl_image_util.h"
#include "pocl_util.h"

extern CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateImage) (cl_context              context,
                       cl_mem_flags            flags,
                       const cl_image_format * image_format,
                       const cl_image_desc *   image_desc, 
                       void *                  host_ptr,
                       cl_int *                errcode_ret) 
CL_API_SUFFIX__VERSION_1_2
{
    cl_mem mem = NULL;
    unsigned i, devices_supporting_images = 0;
    cl_uint num_entries = 0;
    cl_image_format *supported_image_formats = NULL;
    size_t size = 0;
    int errcode;
    size_t row_pitch;
    size_t slice_pitch;
    int elem_size;
    int channels;
    size_t elem_bytes;

    POCL_GOTO_ERROR_COND((context == NULL), CL_INVALID_CONTEXT);

    POCL_GOTO_ERROR_COND((image_format == NULL), CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);

    POCL_GOTO_ERROR_COND((image_desc == NULL), CL_INVALID_IMAGE_DESCRIPTOR);

    if (image_desc->num_mip_levels != 0 || image_desc->num_samples != 0) {
      POCL_ABORT_UNIMPLEMENTED("clCreateImage with image_desc->num_mip_levels != 0"
      " || image_desc->num_samples != 0 ");
    }

    errcode = POname(clGetSupportedImageFormats)
      (context, flags, image_desc->image_type, 0, NULL, &num_entries);

    POCL_GOTO_ERROR_ON((errcode != CL_SUCCESS || num_entries == 0),
      CL_INVALID_VALUE, "Couldn't find any supported image formats\n");

    supported_image_formats = (cl_image_format*) malloc (num_entries * sizeof(cl_image_format));
    if (supported_image_formats == NULL)
      {
        errcode = CL_OUT_OF_HOST_MEMORY;
        goto ERROR;
      }

    errcode = POname (clGetSupportedImageFormats) (
        context, flags, image_desc->image_type, num_entries,
        supported_image_formats, NULL);

    if (errcode != CL_SUCCESS){
      POCL_MSG_ERR("Couldn't get the supported image formats\n");
      goto ERROR;
    }

    /* CL_INVALID_IMAGE_SIZE if image dimensions specified in image_desc exceed
     * the minimum maximum image dimensions described in the table of allowed
     * values for param_name for clGetDeviceInfo FOR ALL DEVICES IN CONTEXT.
     */
    for (i = 0; i < context->num_devices; i++)
      {
        cl_device_id dev = context->devices[i];
        if (!dev->image_support)
          continue;
        else
          ++devices_supporting_images;
        if (pocl_check_device_supports_image (dev, image_format, image_desc,
                                              supported_image_formats,
                                              num_entries)
            != CL_SUCCESS)
          goto ERROR;
      }
    POCL_GOTO_ERROR_ON (
        (devices_supporting_images == 0), CL_INVALID_OPERATION,
        "There are no devices in context that support images\n");

    pocl_get_image_information (image_format->image_channel_order,
                                image_format->image_channel_data_type, 
                                &channels, &elem_size);
    elem_bytes = elem_size * channels;

    row_pitch = image_desc->image_row_pitch;
    slice_pitch = image_desc->image_slice_pitch;

    /* This must be 0 if host_ptr is NULL and can be either 0 or ≥
     * image_width * size of element in bytes if host_ptr is not NULL.
     * If host_ptr is not NULL and image_row_pitch = 0, image_row_pitch
     * is calculated as image_width * size of element in bytes. If
     * image_row_pitch is not 0, it must be a multiple of the
     * image element size in bytes.
     */
    if (row_pitch == 0)
      {
        row_pitch = image_desc->image_width * elem_bytes;
      }
    else
      {
        POCL_GOTO_ERROR_COND ((row_pitch % elem_bytes), CL_INVALID_VALUE);
      }

    /* The size in bytes of each 2D slice in the 3D image or the size in bytes
     * of each image in a 1D or 2D image array. This must be 0 if host_ptr is
     * NULL. If host_ptr is not NULL, image_slice_pitch can be either 0 or ≥
     * image_row_pitch * image_height for a 2D image array or 3D image and can
     * be either 0 or ≥ image_row_pitch for a 1D image array. If host_ptr is
     * not NULL and image_slice_pitch = 0, image_slice_pitch is calculated as
     * image_row_pitch * image_height for a 2D image array or 3D image and
     * image_row_pitch for a 1D image array. If image_slice_pitch is not 0,
     * it must be a multiple of the image_row_pitch.
     */

    if (slice_pitch == 0)
      {
        if (image_desc->image_type == CL_MEM_OBJECT_IMAGE3D ||
            image_desc->image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
          {
            slice_pitch = row_pitch * image_desc->image_height;
          }
        if (image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
          {
            slice_pitch = row_pitch;
          }
      }
    else
      {
        POCL_GOTO_ERROR_COND ((slice_pitch % row_pitch), CL_INVALID_VALUE);
      }

    if (image_desc->image_type == CL_MEM_OBJECT_IMAGE3D)
      size = slice_pitch * image_desc->image_depth;

    if (image_desc->image_type == CL_MEM_OBJECT_IMAGE2D)
      size = row_pitch * image_desc->image_height;

    if (image_desc->image_type == CL_MEM_OBJECT_IMAGE1D
        || image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER)
      size = row_pitch;

    if (image_desc->image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY
        || image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
      {
        size = slice_pitch * image_desc->image_array_size;
      }

    /* Create buffer and fill in missing parts */
    if (image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER)
      {
        POCL_GOTO_ERROR_COND ((image_desc->buffer == NULL),
                              CL_INVALID_MEM_OBJECT);
        POCL_GOTO_ERROR_COND ((image_desc->buffer->size < size),
                              CL_INVALID_MEM_OBJECT);

        mem = (cl_mem)malloc (sizeof (struct _cl_mem));
        POCL_GOTO_ERROR_COND ((mem == NULL), CL_OUT_OF_HOST_MEMORY);
        memset (mem, 0, sizeof (struct _cl_mem));
        POCL_INIT_OBJECT (mem);

        cl_mem b = image_desc->buffer;
        mem->buffer = b;

        mem->size = size;
        mem->origin = 0;

        mem->context = context;
        assert (mem->context == b->context);

        pocl_cl_mem_inherit_flags (mem, b, flags);

        /* Retain the buffer we're referencing */
        POname (clRetainMemObject) (b);

        POCL_MSG_PRINT_MEMORY ("CREATED IMAGE: %p REF BUFFER: %p \n\n", mem,
                               b);
      }
    else
      {
        mem = POname (clCreateBuffer) (context, flags, size, host_ptr,
                                       &errcode);
        POCL_GOTO_ERROR_ON ((mem == NULL), CL_OUT_OF_HOST_MEMORY,
                            "clCreateBuffer (for backing the image) failed\n");
        mem->buffer = NULL;
      }

    mem->type = image_desc->image_type;
    mem->is_image = CL_TRUE;
    mem->image_width = image_desc->image_width;
    if (image_desc->image_type == CL_MEM_OBJECT_IMAGE2D
        || image_desc->image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY
        || image_desc->image_type == CL_MEM_OBJECT_IMAGE3D)
      mem->image_height = image_desc->image_height;
    else
      mem->image_height = 0;
    if (image_desc->image_type == CL_MEM_OBJECT_IMAGE3D)
      mem->image_depth = image_desc->image_depth;
    else
      mem->image_depth = 0;
    if (image_desc->image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY
        || image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
      mem->image_array_size = image_desc->image_array_size;
    else
      mem->image_array_size = 0;
    mem->image_row_pitch = row_pitch;
    mem->image_slice_pitch = slice_pitch;
    mem->image_channel_data_type = image_format->image_channel_data_type;
    mem->image_channel_order = image_format->image_channel_order;
    mem->num_mip_levels = image_desc->num_mip_levels;
    mem->num_samples = image_desc->num_samples;
    mem->image_channels = channels;
    mem->image_elem_size = elem_size;

#if 0
    printf("flags = %X\n",mem->flags); 
    printf("mem_image_width %d\n", mem->image_width);
    printf("mem_image_height %d\n", mem->image_height);
    printf("mem_image_depth %d\n", mem->image_depth);
    printf("mem_image_array_size %d\n", mem->image_array_size);
    printf("mem_image_row_pitch %d\n", mem->image_row_pitch);
    printf("mem_image_slice_pitch %d\n", mem->image_slice_pitch);
    printf("mem_host_ptr %u\n", mem->mem_host_ptr);
    printf("mem_image_channel_data_type %x \n",mem->image_channel_data_type);
    printf("device_ptrs[0] %x \n \n", mem->device_ptrs[0]);
#endif

    if (errcode_ret != NULL)
      *errcode_ret = CL_SUCCESS;

    POCL_MEM_FREE (supported_image_formats);
    return mem;
    
 ERROR:
   POCL_MEM_FREE (supported_image_formats);
   if (errcode_ret)
     {
       *errcode_ret = errcode;
     }
   return NULL;
}
POsym(clCreateImage)
