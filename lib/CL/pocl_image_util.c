/* OpenCL runtime library: pocl_image_util image utility functions

   Copyright (c) 2012 Timo Viitanen / Tampere University of Technology
   
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
#include "assert.h"

static unsigned
pocl_get_image_dim (const cl_mem image)
{
  if ((image->type == CL_MEM_OBJECT_IMAGE1D)
      || (image->type == CL_MEM_OBJECT_IMAGE1D_BUFFER))
    return 1;
  if ((image->type == CL_MEM_OBJECT_IMAGE2D)
      || (image->type == CL_MEM_OBJECT_IMAGE1D_ARRAY))
    return 2;
  if ((image->type == CL_MEM_OBJECT_IMAGE3D)
      || (image->type == CL_MEM_OBJECT_IMAGE2D_ARRAY))
    return 3;

  return (unsigned)-1;
}

extern cl_int 
pocl_check_image_origin_region (const cl_mem image, 
                                const size_t *origin, 
                                const size_t *region)
{
  POCL_RETURN_ERROR_COND((image == NULL), CL_INVALID_MEM_OBJECT);

  POCL_RETURN_ERROR_COND((origin == NULL), CL_INVALID_VALUE);
  POCL_RETURN_ERROR_COND((region == NULL), CL_INVALID_VALUE);

  unsigned dim = pocl_get_image_dim (image);

  if (dim < 3)
    {
      /* If image is a 2D image object, origin[2] must be 0.
       * If image is a 1D image or 1D image buffer object,
       * origin[1] and origin[2] must be 0.
       * If image is a 2D image object, region[2] must be 1.
       * If image is a 1D image or 1D image buffer object,
       * region[1] and region[2] must be 1.
       * If image is a 1D image array object, region[2] must be 1.
       */
      unsigned i;
      for (i = dim; i < 3; i++)
        {
          POCL_RETURN_ERROR_ON (
              (origin[i] != 0), CL_INVALID_VALUE,
              "Image origin[x](=%zu) must be 0 for x(=%u) >= image_dim\n",
              origin[i], i);
          POCL_RETURN_ERROR_ON (
              (region[i] != 1), CL_INVALID_VALUE,
              "Image region[x](=%zu) must be 1 for x(=%u) >= image_dim\n",
              region[i], i);
        }
    }

  /* check if origin + region in each dimension is with in image bounds */
  POCL_RETURN_ERROR_ON (
      ((origin[0] + region[0]) > image->image_width), CL_INVALID_VALUE,
      "(origin[0](=%zu) + region[0](=%zu)) > image->image_width(=%zu)",
      origin[0], region[0], image->image_width);
  POCL_RETURN_ERROR_ON (
      (image->image_height > 0
       && ((origin[1] + region[1]) > image->image_height)),
      CL_INVALID_VALUE,
      "(origin[1](=%zu) + region[1](=%zu)) > image->image_height(=%zu)",
      origin[1], region[2], image->image_height);
  POCL_RETURN_ERROR_ON (
      (image->image_depth > 0 && (origin[2] + region[2]) > image->image_depth),
      CL_INVALID_VALUE,
      "(origin[2](=%zu) + region[2](=%zu)) > image->image_depth(=%zu)",
      origin[1], region[2], image->image_depth);
  return CL_SUCCESS;
}

extern cl_int
pocl_check_device_supports_image (cl_device_id device,
                                  const cl_image_format *image_format,
                                  const cl_image_desc *image_desc,
                                  cl_image_format *supported_image_formats,
                                  cl_uint num_entries)
{
  cl_int errcode;
  cl_uint i;
  size_t m;

  POCL_RETURN_ERROR_ON((!device->image_support), CL_INVALID_OPERATION,
          "Device does not support images");

  if (image_desc->image_type == CL_MEM_OBJECT_IMAGE1D
      || image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
    {
      POCL_RETURN_ERROR_ON (
          (image_desc->image_width > device->image2d_max_width),
          CL_INVALID_IMAGE_SIZE, "Image width > device.image2d_max_width\n");
    }

  if (image_desc->image_type == CL_MEM_OBJECT_IMAGE2D
      || image_desc->image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY)
    {
      POCL_RETURN_ERROR_ON (
          (image_desc->image_width > device->image2d_max_width),
          CL_INVALID_IMAGE_SIZE, "Image width > device.image2d_max_width\n");
      POCL_RETURN_ERROR_ON (
          (image_desc->image_height > device->image2d_max_height),
          CL_INVALID_IMAGE_SIZE, "Image height > device.image2d_max_height\n");
    }

  if (image_desc->image_type == CL_MEM_OBJECT_IMAGE3D)
    {
      POCL_RETURN_ERROR_ON (
          (image_desc->image_width > device->image3d_max_width),
          CL_INVALID_IMAGE_SIZE, "Image width > device.image3d_max_width\n");
      POCL_RETURN_ERROR_ON (
          (image_desc->image_height > device->image3d_max_height),
          CL_INVALID_IMAGE_SIZE, "Image height > device.image3d_max_height\n");
      POCL_RETURN_ERROR_ON (
          (image_desc->image_depth > device->image3d_max_depth),
          CL_INVALID_IMAGE_SIZE, "Image depth > device.image3d_max_depth\n");
    }

  if ((image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_ARRAY)
      || (image_desc->image_type == CL_MEM_OBJECT_IMAGE2D_ARRAY))
    {
      POname (clGetDeviceInfo (device, CL_DEVICE_IMAGE_MAX_ARRAY_SIZE,
                               sizeof (m), &m, NULL));
      POCL_RETURN_ERROR_ON ((m < image_desc->image_array_size),
                            CL_INVALID_IMAGE_SIZE,
                            "Image array size > device.max_array_size\n");
    }

  for (i = 0; i < num_entries; i++)
    {
      if (supported_image_formats[i].image_channel_order
              == image_format->image_channel_order
          && supported_image_formats[i].image_channel_data_type
                 == image_format->image_channel_data_type)
        {
          errcode = CL_SUCCESS;
          goto ERROR;
        }
    }

  POCL_GOTO_ERROR_ON (1, CL_INVALID_IMAGE_FORMAT_DESCRIPTOR,
                      "The image format is not supported by the device\n");

ERROR:
  return errcode;
}

extern void
pocl_get_image_information (cl_channel_order ch_order, 
                            cl_channel_type ch_type,
                            int* channels_out,
                            int* elem_size_out)
{
  if (ch_type == CL_SNORM_INT8 || ch_type == CL_UNORM_INT8 ||
      ch_type == CL_SIGNED_INT8 || ch_type == CL_UNSIGNED_INT8)
    {
      *elem_size_out = 1; /* 1 byte */
    }
  else if (ch_type == CL_UNSIGNED_INT32 || ch_type == CL_SIGNED_INT32 ||
           ch_type == CL_FLOAT || ch_type == CL_UNORM_INT_101010)
    {
      *elem_size_out = 4; /* 32bit -> 4 bytes */
    }
  else if (ch_type == CL_SNORM_INT16 || ch_type == CL_UNORM_INT16 ||
           ch_type == CL_SIGNED_INT16 || ch_type == CL_UNSIGNED_INT16 ||
           ch_type == CL_UNORM_SHORT_555 || ch_type == CL_UNORM_SHORT_565 ||
           ch_type == CL_HALF_FLOAT)
    {
      *elem_size_out = 2; /* 16bit -> 2 bytes */
    }
  
  /* channels TODO: verify num of channels*/
  if (ch_order == CL_RGB || ch_order == CL_RGBx || ch_order == CL_R || 
      ch_order == CL_Rx || ch_order == CL_A)
    {
      *channels_out = 1;
    }
  else if (ch_order == CL_RG || ch_order == CL_RGx || ch_order == CL_RA)
    {
      *channels_out = 2;
    }
  else
    {
      *channels_out = 4;
    }
}
