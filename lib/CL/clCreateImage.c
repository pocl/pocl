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
#include "pocl_shared.h"
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
    unsigned i, num_devices_supporting_image = 0;
    size_t size = 0;
    int errcode = CL_SUCCESS;
    int *device_image_support = NULL;
    size_t row_pitch;
    size_t slice_pitch;
    int elem_size;
    int channels;
    size_t elem_bytes;
    cl_int image_type_idx;
    cl_mem_object_type image_type;

    POCL_GOTO_ERROR_COND((context == NULL), CL_INVALID_CONTEXT);

    POCL_GOTO_ERROR_COND((image_format == NULL), CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);

    POCL_GOTO_ERROR_COND((image_desc == NULL), CL_INVALID_IMAGE_DESCRIPTOR);

    if (image_desc->num_mip_levels != 0 || image_desc->num_samples != 0) {
      POCL_ABORT_UNIMPLEMENTED("clCreateImage with image_desc->num_mip_levels != 0"
      " || image_desc->num_samples != 0 ");
    }

    image_type = image_desc->image_type;
    image_type_idx = opencl_image_type_to_index (image_type);
    POCL_GOTO_ERROR_ON ((image_type_idx < 0),
                        CL_INVALID_VALUE, "unknown image type\n");

    /* CL_INVALID_IMAGE_SIZE if image dimensions specified in image_desc exceed
     * the minimum maximum image dimensions described in the table of allowed
     * values for param_name for clGetDeviceInfo FOR ALL DEVICES IN CONTEXT.
     */
    device_image_support = calloc (context->num_devices, sizeof (int));
    for (i = 0; i < context->num_devices; i++)
      {
        cl_device_id dev = context->devices[i];
        if (!dev->image_support)
          continue;
        else
          {
            if (pocl_check_device_supports_image (dev, image_format,
                                                  image_desc, image_type_idx,
                                                  &device_image_support[i])
                == CL_SUCCESS)
              {
                /* can't break here as we need device_image_support[]
                 * for all devices */
                ++num_devices_supporting_image;
              }
          }
      }
    POCL_GOTO_ERROR_ON ((num_devices_supporting_image == 0),
                        CL_INVALID_OPERATION,
                        "There are no devices in context that support this "
                        "image type + size \n");

    pocl_get_image_information (image_format->image_channel_order,
                                image_format->image_channel_data_type,
                                &channels, &elem_size);
    elem_bytes = (size_t)elem_size * (size_t)channels;

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

        POCL_GOTO_ERROR_ON   ((image_desc->buffer->parent != NULL),
                              CL_INVALID_MEM_OBJECT,
                              "pocl does not support Image 1D"
                              " Buffer over SubBuffers\n");

        POCL_GOTO_ERROR_COND ((image_desc->buffer->size < size),
                              CL_INVALID_MEM_OBJECT);

        mem = (cl_mem) calloc (1, sizeof (struct _cl_mem));
        POCL_GOTO_ERROR_COND ((mem == NULL), CL_OUT_OF_HOST_MEMORY);
        POCL_INIT_OBJECT (mem);

        cl_mem b = image_desc->buffer;
        mem->buffer = b;
        mem->device_supports_this_image = device_image_support;

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
        /* create & partially setup a buffer struct, but don't allocate yet */
        mem = pocl_create_memobject (context, flags, size, host_ptr,
                                     errcode_ret);
        if (mem == NULL)
          goto ERROR;

        mem->device_supports_this_image = device_image_support;
        // allocate on every device
        for (i = 0; i < context->num_devices; ++i)
          {
            if (DEVICE_DOESNT_SUPPORT_IMAGE (mem, i))
              continue;

            cl_device_id dev = context->devices[i];
            pocl_mem_identifier *p = &mem->gmem_ptrs[dev->global_mem_id];
            if (p->mem_ptr != NULL)
              continue;

            assert (dev->global_memory->alloc_mem_obj);
            errcode = dev->global_memory->alloc_mem_obj (dev->global_memory,
                                                         mem, p, host_ptr);
            if (errcode) // TODO release mems!!!!
              goto ERROR;
          }

        // if no device has allocated host backing memory for the image,
        // allocate it now.
        if (mem->mem_host_ptr == NULL)
          {
            mem->mem_host_ptr
                = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, size);
            assert (mem->mem_host_ptr);

            /* this needs to be separately for prealloc_memobj, since device
             * may have its own image row/slice pitch, in which case the driver
             * needs to handle the memory copy. */
            if (flags & CL_MEM_COPY_HOST_PTR)
              {
                POCL_MSG_PRINT_MEMORY (
                    "image NOT preallocated and we need to copy it\n");
                memcpy (mem->mem_host_ptr, host_ptr, size);
              }
          }
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

    POCL_RETAIN_OBJECT (context);

    POCL_MSG_PRINT_MEMORY ("Created Image %p, HOST_PTR: %p, SIZE %zu \n", mem,
                           mem->mem_host_ptr, size);

    if (errcode_ret != NULL)
      *errcode_ret = CL_SUCCESS;

    return mem;

 ERROR:
   POCL_MEM_FREE (device_image_support);
   if (errcode_ret)
     {
       *errcode_ret = errcode;
     }
   return NULL;
}
POsym(clCreateImage)
