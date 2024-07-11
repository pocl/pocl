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

extern unsigned long image_c;

cl_mem
pocl_create_image_internal (cl_context context, cl_mem_flags flags,
                            const cl_image_format *image_format,
                            const cl_image_desc *image_desc, void *host_ptr,
                            cl_int *errcode_ret,
                            cl_GLenum gl_target, cl_GLint gl_miplevel,
                            cl_GLuint gl_texture,
                            CLeglDisplayKHR egl_display,
                            CLeglImageKHR egl_image)
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

    POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

    POCL_GOTO_ERROR_COND((image_format == NULL), CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);

    POCL_GOTO_ERROR_COND((image_desc == NULL), CL_INVALID_IMAGE_DESCRIPTOR);

    POCL_GOTO_ERROR_ON (
        (image_desc->num_mip_levels != 0 || image_desc->num_samples != 0),
        CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
        "Unimplemented: clCreateImage with image_desc->num_mip_levels != 0"
        " || image_desc->num_samples != 0 ");

    image_type = image_desc->image_type;
    image_type_idx = pocl_opencl_image_type_to_index (image_type);
    POCL_GOTO_ERROR_ON ((image_type_idx < 0),
                        CL_INVALID_VALUE, "unknown image type\n");

    /* CL_INVALID_IMAGE_SIZE if image dimensions specified in image_desc exceed
     * the minimum maximum image dimensions described in the table of allowed
     * values for param_name for clGetDeviceInfo FOR ALL DEVICES IN CONTEXT.
     */
    device_image_support = (int *)calloc (context->num_devices, sizeof (int));
#ifdef ENABLE_OPENGL_INTEROP
    unsigned is_gl_texture
        = (unsigned)((intptr_t)gl_target | (intptr_t)gl_miplevel
                     | (intptr_t)gl_texture);
#elif defined(ENABLE_EGL_INTEROP)
    unsigned is_gl_texture
        = (unsigned)((intptr_t)egl_display | (intptr_t)egl_image);
#else
    unsigned is_gl_texture = 0;
#endif

    for (i = 0; i < context->num_devices; i++)
      {
        cl_device_id dev = context->devices[i];
        if (!dev->image_support || (*(dev->available) == CL_FALSE))
          continue;
        else
          {
            if (pocl_check_device_supports_image (
                    dev, image_format, image_desc, image_type_idx,
                    is_gl_texture, &device_image_support[i])
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

    /* "For a 1D image buffer created from a buffer object, the
        image_width * size of element in bytes must be <= size of the buffer
        object. The image data in the buffer object is stored as a single
        scanline which is a linear sequence of adjacent elements." */
    if (image_desc->image_type == CL_MEM_OBJECT_IMAGE1D
        || image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER)
      size = image_desc->image_width * elem_bytes;

    if (image_desc->image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY
        || image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
      size = slice_pitch * image_desc->image_array_size;

    /* Create buffer and fill in missing parts */
    if (image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER)
      {
        POCL_GOTO_ERROR_COND ((image_desc->buffer == NULL),
                              CL_INVALID_MEM_OBJECT);

        POCL_GOTO_ERROR_ON   ((image_desc->buffer->parent != NULL),
                              CL_INVALID_MEM_OBJECT,
                              "pocl does not support Image 1D"
                              " Buffer over SubBuffers\n");

        POCL_GOTO_ERROR_ON ((image_desc->buffer->size < size),
                            CL_INVALID_MEM_OBJECT,
                            "Invalid buffer sizes: image_desc->buffer->size"
                            " %zu || Required size: %zu\n",
                            image_desc->buffer->size, size);

        mem = (cl_mem) calloc (1, sizeof (struct _cl_mem));
        POCL_GOTO_ERROR_COND ((mem == NULL), CL_OUT_OF_HOST_MEMORY);
        POCL_INIT_OBJECT (mem);

        cl_mem b = image_desc->buffer;
        mem->is_image = CL_TRUE;
        mem->type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
        mem->device_supports_this_image = device_image_support;
        mem->buffer = b;

        mem->size = size;
        mem->origin = 0;

        mem->context = context;
        assert (mem->context == b->context);

        mem->parent = b;
        pocl_cl_mem_inherit_flags (mem, b, flags);

        /* Retain the buffer we're referencing */
        POname (clRetainMemObject) (b);

        POCL_MSG_PRINT_MEMORY ("Created Image:  %" PRId64
                               " (%p), Refbuffer: %" PRId64 " (%p) \n\n",
                               mem->id, mem, b->id, b);
      }
    else
      {
        POCL_GOTO_ERROR_COND ((image_desc->buffer != NULL),
                              CL_INVALID_OPERATION);
        int host_ptr_is_svm = CL_FALSE;

        if ((flags & CL_MEM_USE_HOST_PTR) && host_ptr != NULL)
          {
            pocl_raw_ptr *item
                = pocl_find_raw_ptr_with_vm_ptr (context, host_ptr);
            if (item)
              {
                POCL_GOTO_ERROR_ON ((item->size < size), CL_INVALID_BUFFER_SIZE,
                                    "The provided host_ptr is SVM pointer, "
                                    "but the allocated SVM size (%zu) is smaller "
                                    "then requested size (%zu)",
                                    item->size, size);
                host_ptr_is_svm = CL_TRUE;
              }
          }

        mem = pocl_create_memobject (context, flags, size,
                                     image_desc->image_type,
                                     device_image_support,
                                     host_ptr, host_ptr_is_svm, &errcode);
        if (mem == NULL)
          goto ERROR;
      }

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

    TP_CREATE_IMAGE (context->id, mem->id);

    mem->is_gl_texture = is_gl_texture;
    if (is_gl_texture)
      {
        mem->is_gl_acquired = 0;
        mem->target = gl_target;
        mem->miplevel = gl_miplevel;
        mem->texture = gl_texture;
        mem->egl_display = egl_display;
        mem->egl_image = egl_image;
      }

    POCL_RETAIN_OBJECT (context);

    POCL_MSG_PRINT_MEMORY (
        "Created Image %" PRId64
        " (%p), HOST_PTR: %p, SIZE %zu RP %zu SP %zu FLAGS %u \n",
        mem->id, mem, mem->mem_host_ptr, size, mem->image_row_pitch,
        mem->image_slice_pitch, (unsigned)flags);

    POCL_ATOMIC_INC (image_c);

 ERROR:
   if (errcode != CL_SUCCESS)
     POCL_MEM_FREE (device_image_support);

   if (errcode_ret)
     {
       *errcode_ret = errcode;
     }

   return mem;
}

CL_API_ENTRY cl_mem CL_API_CALL POname (clCreateImage) (
    cl_context context, cl_mem_flags flags,
    const cl_image_format *image_format, const cl_image_desc *image_desc,
    void *host_ptr, cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
  return pocl_create_image_internal (context, flags, image_format, image_desc,
                                     host_ptr, errcode_ret,
                                     0, 0, 0, NULL, NULL);
}
POsym (clCreateImage)

CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateImageWithProperties)(cl_context                context,
                            const cl_mem_properties * properties,
                            cl_mem_flags              flags,
                            const cl_image_format *   image_format,
                            const cl_image_desc *     image_desc,
                            void *                    host_ptr,
                            cl_int *                  errcode_ret)
CL_API_SUFFIX__VERSION_3_0
{
  int errcode;

  /* pocl doesn't support any extra properties ATM */
  POCL_GOTO_ERROR_ON ((properties && properties[0] != 0), CL_INVALID_PROPERTY,
                      "PoCL doesn't support any properties on images yet\n");

  cl_mem mem_ret = POname (clCreateImage) (context, flags, image_format,
                                           image_desc, host_ptr, errcode_ret);

  if (mem_ret == NULL)
    return NULL;

  if (properties && properties[0] == 0)
    {
      mem_ret->num_properties = 1;
      mem_ret->properties[0] = 0;
    }

  return mem_ret;

ERROR:
  if (errcode_ret)
    {
      *errcode_ret = errcode;
    }
  return NULL;
}
POsym (clCreateImageWithProperties)
