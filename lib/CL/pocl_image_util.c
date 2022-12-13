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

#include "pocl_image_util.h"
#include "assert.h"
#include "pocl_cl.h"
#include "pocl_util.h"

cl_int opencl_image_type_to_index (cl_mem_object_type  image_type)
{
  switch (image_type)
    {
      case CL_MEM_OBJECT_IMAGE2D                      : return 0;
      case CL_MEM_OBJECT_IMAGE3D                      : return 1;
      case CL_MEM_OBJECT_IMAGE2D_ARRAY                : return 2;
      case CL_MEM_OBJECT_IMAGE1D                      : return 3;
      case CL_MEM_OBJECT_IMAGE1D_ARRAY                : return 4;
      case CL_MEM_OBJECT_IMAGE1D_BUFFER               : return 5;
      default: return -1;
    }
}

void
origin_to_bytes (cl_mem mem, const size_t *origin, size_t *byte_offset)
{
  *byte_offset = origin[0] + origin[1] * mem->image_row_pitch
                 + origin[2] * mem->image_slice_pitch;
  *byte_offset *= (mem->image_elem_size * mem->image_channels);
}

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
                                  cl_uint image_type_idx,
                                  cl_bool is_gl_texture, int *device_support)
{
  cl_uint i;
  size_t m;

  *device_support = 0;

  if (device->has_gl_interop || (!is_gl_texture))
    *device_support |= DEVICE_IMAGE_INTEROP_SUPPORT;

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

  if (image_desc->image_type == CL_MEM_OBJECT_IMAGE1D_BUFFER)
    {
      POname (clGetDeviceInfo (device, CL_DEVICE_IMAGE_MAX_BUFFER_SIZE,
                               sizeof (m), &m, NULL));
      POCL_RETURN_ERROR_ON (
          (m < image_desc->image_width), CL_INVALID_IMAGE_SIZE,
          "Image buffer size (width) > device.max_buffer_size\n");
    }

  *device_support |= DEVICE_IMAGE_SIZE_SUPPORT;

  for (i = 0; i < device->num_image_formats[image_type_idx]; i++)
    {
      const cl_image_format *p = device->image_formats[image_type_idx];
      assert (p != NULL);
      if (p[i].image_channel_order
              == image_format->image_channel_order
          && p[i].image_channel_data_type
                 == image_format->image_channel_data_type)
        {
          *device_support |= DEVICE_IMAGE_FORMAT_SUPPORT;

          return CL_SUCCESS;
        }
    }

  POCL_MSG_ERR2 ("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
                 "The image format is not supported by the device\n");
  return CL_INVALID_IMAGE_FORMAT_DESCRIPTOR;
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
  else
    *elem_size_out = 0;

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

/****************************************************/

#define FOR4 unsigned i; for (i = 0; i < 4; i++)
#define FOR2 unsigned i; for (i = 0; i < 2; i++)

cl_char4
convert_char4_sat (cl_float4 x)
{
  cl_char4 r;
  FOR4
    r.s[i] = (cl_char)max (CL_CHAR_MIN, min ((cl_int) (x.s[i]), CL_CHAR_MAX));
  return r;
}

cl_short4
convert_short4_sat (cl_float4 x)
{
  cl_short4 r;
  FOR4
    r.s[i] = (cl_short)max (CL_SHRT_MIN, min ((cl_int) (x.s[i]), CL_SHRT_MAX));
  return r;
}

cl_uchar4
convert_uchar4_sat (cl_float4 x)
{
  cl_uchar4 r;
  FOR4
    r.s[i] = (cl_uchar)max (0, min ((cl_long) (x.s[i]), CL_UCHAR_MAX));
  return r;
}

cl_ushort4
convert_ushort4_sat (cl_float4 x)
{
  cl_ushort4 r;
  FOR4
    r.s[i] = (cl_ushort)max (0, min ((cl_long) (x.s[i]), CL_USHRT_MAX));
  return r;
}

/****************************************************/

cl_char2
convert_char2_sat (cl_float2 x)
{
  cl_char2 r;
  FOR2
    r.s[i] = (cl_char)max (CL_CHAR_MIN, min ((cl_int) (x.s[i]), CL_CHAR_MAX));
  return r;
}

cl_short2
convert_short2_sat (cl_float2 x)
{
  cl_short2 r;
  FOR2
    r.s[i] = (cl_short)max (CL_SHRT_MIN, min ((cl_int) (x.s[i]), CL_SHRT_MAX));
  return r;
}

cl_uchar2
convert_uchar2_sat (cl_float2 x)
{
  cl_uchar2 r;
  FOR2
    r.s[i] = (cl_uchar)max (0, min ((cl_long) (x.s[i]), CL_UCHAR_MAX));
  return r;
}

cl_ushort2
convert_ushort2_sat (cl_float2 x)
{
  cl_ushort2 r;
  FOR2
    r.s[i] = (cl_ushort)max (0, min ((cl_long) (x.s[i]), CL_USHRT_MAX));
  return r;
}

/****************************************************/

cl_char
convert_char_sat (cl_float x)
{
  cl_int y = (cl_int)x;
  return (cl_char)max (CL_CHAR_MIN, min (y, CL_CHAR_MAX));
}

cl_short
convert_short_sat (cl_float x)
{
  cl_int y = (cl_int)x;
  return (cl_short)max (CL_SHRT_MIN, min (y, CL_SHRT_MAX));
}

cl_uchar
convert_uchar_sat (cl_float x)
{
  cl_long y = (cl_long)x;
  return (cl_uchar)max (0, min (y, CL_UCHAR_MAX));
}

cl_ushort
convert_ushort_sat (cl_float x)
{
  cl_long y = (cl_long)x;
  return (cl_ushort)max (0, min (y, CL_USHRT_MAX));
}

/****************************************************/

cl_char4
convert_char4_sat_int (cl_int4 x)
{
  cl_char4 r;
  FOR4
    r.s[i] = (cl_char)max (CL_CHAR_MIN, min ((cl_int) (x.s[i]), CL_CHAR_MAX));
  return r;
}

cl_short4
convert_short4_sat_int (cl_int4 x)
{
  cl_short4 r;
  FOR4
    r.s[i] = (cl_short)max (CL_SHRT_MIN, min ((cl_int) (x.s[i]), CL_SHRT_MAX));
  return r;
}

cl_uchar4
convert_uchar4_sat_int (cl_uint4 x)
{
  cl_uchar4 r;
  FOR4
    r.s[i] = (cl_uchar)min (x.s[i], CL_UCHAR_MAX);
  return r;
}

cl_ushort4
convert_ushort4_sat_int (cl_uint4 x)
{
  cl_ushort4 r;
  FOR4
    r.s[i] = (cl_ushort)min (x.s[i], CL_USHRT_MAX);
  return r;
}

/****************************************************/

cl_char2
convert_char2_sat_int (cl_int2 x)
{
  cl_char2 r;
  FOR2
    r.s[i] = (cl_char)max (CL_CHAR_MIN, min ((cl_int) (x.s[i]), CL_CHAR_MAX));
  return r;
}

cl_short2
convert_short2_sat_int (cl_int2 x)
{
  cl_short2 r;
  FOR2
    r.s[i] = (cl_short)max (CL_SHRT_MIN, min ((cl_int) (x.s[i]), CL_SHRT_MAX));
  return r;
}

cl_uchar2
convert_uchar2_sat_int (cl_uint2 x)
{
  cl_uchar2 r;
  FOR2
    r.s[i] = (cl_uchar)min (x.s[i], CL_UCHAR_MAX);
  return r;
}

cl_ushort2
convert_ushort2_sat_int (cl_uint2 x)
{
  cl_ushort2 r;
  FOR2
    r.s[i] = (cl_ushort)min (x.s[i], CL_USHRT_MAX);
  return r;
}

/****************************************************/

cl_char
convert_char_sat_int (cl_int x)
{
  return (cl_char)max (CL_CHAR_MIN, min (x, CL_CHAR_MAX));
}

cl_short
convert_short_sat_int (cl_int x)
{
  return (cl_short)max (CL_SHRT_MIN, min (x, CL_SHRT_MAX));
}

cl_uchar
convert_uchar_sat_int (cl_uint x)
{
  return (cl_uchar)min (x, CL_UCHAR_MAX);
}

cl_ushort
convert_ushort_sat_int (cl_uint x)
{
  return (cl_ushort)min (x, CL_USHRT_MAX);
}

/****************************************************/

static cl_uint4
map_channels (const cl_uint4 color, int order)
{
  switch (order)
    {
    case CL_ARGB:
      {
        // return color.wxyz;
        cl_uint4 ret;
        ret.s[0] = color.s[3];
        ret.s[1] = color.s[0];
        ret.s[2] = color.s[1];
        ret.s[3] = color.s[2];
        return ret;
      }
    case CL_BGRA:
      {
        // return color.zyxw;
        cl_uint4 ret;
        ret.s[0] = color.s[2];
        ret.s[1] = color.s[1];
        ret.s[2] = color.s[0];
        ret.s[3] = color.s[3];
        return ret;
      }
    case CL_RGBA:
    case CL_RG:
    default:
      return color;
    }
}

/* only for CL_FLOAT, CL_SNORM_INT8, CL_UNORM_INT8,
 * CL_SNORM_INT16, CL_UNORM_INT16 channel types */
static void
write_float4_pixel (cl_float4 color, void *data, int type)
{
  if (type == CL_FLOAT)
    {
      cl_float4 *p = (cl_float4 *)data;
      FOR4
        p->s[i] = color.s[i];
      return;
    }
  if (type == CL_HALF_FLOAT)
    {
      /* TODO: convert to builtins */
      ((uint16_t *)data)[0] = float_to_half (color.s0);
      ((uint16_t *)data)[1] = float_to_half (color.s1);
      ((uint16_t *)data)[2] = float_to_half (color.s2);
      ((uint16_t *)data)[3] = float_to_half (color.s3);
      return;
    }
  const cl_float f127 = ((cl_float) (CL_CHAR_MAX));
  const cl_float f32767 = ((cl_float) (CL_SHRT_MAX));
  const cl_float f255 = ((cl_float) (CL_UCHAR_MAX));
  const cl_float f65535 = ((cl_float) (CL_USHRT_MAX));
  if (type == CL_SNORM_INT8)
    {
      /*  <-1.0, 1.0> to <I*_MIN, I*_MAX> */
      cl_float4 colorf;
      FOR4
        colorf.s[i] = color.s[i] * f127;
      cl_char4 final_color = convert_char4_sat (colorf);
      *((cl_char4 *)data) = final_color;
      return;
    }
  if (type == CL_SNORM_INT16)
    {
      cl_float4 colorf;
      FOR4
        colorf.s[i] = color.s[i] * f32767;
      cl_short4 final_color = convert_short4_sat (colorf);
      *((cl_short4 *)data) = final_color;
      return;
    }
  if (type == CL_UNORM_INT8)
    {
      /* <0, I*_MAX> to <0.0, 1.0> */
      /*  <-1.0, 1.0> to <I*_MIN, I*_MAX> */
      cl_float4 colorf;
      FOR4
        colorf.s[i] = color.s[i] * f255;
      cl_uchar4 final_color = convert_uchar4_sat (colorf);
      *((cl_uchar4 *)data) = final_color;
      return;
    }
  if (type == CL_UNORM_INT16)
    {
      cl_float4 colorf;
      FOR4
        colorf.s[i] = color.s[i] * f65535;
      cl_ushort4 final_color = convert_ushort4_sat (colorf);
      *((cl_ushort4 *)data) = final_color;
      return;
    }

  return;
}

/* only for CL_FLOAT, CL_SNORM_INT8, CL_UNORM_INT8,
 * CL_SNORM_INT16, CL_UNORM_INT16 channel types */
static void
write_float2_pixel (cl_float2 color, void *data, int type)
{
  unsigned i;
  if (type == CL_FLOAT)
    {
      cl_float2 *p = (cl_float2 *)data;
      for (i = 0; i < 2; i++)
        p->s[i] = color.s[i];
      return;
    }
  if (type == CL_HALF_FLOAT)
    {
      /* TODO: convert to builtins */
      ((uint16_t *)data)[0] = float_to_half (color.s0);
      ((uint16_t *)data)[1] = float_to_half (color.s1);
      return;
    }
  const cl_float f127 = ((cl_float) (CL_CHAR_MAX));
  const cl_float f32767 = ((cl_float) (CL_SHRT_MAX));
  const cl_float f255 = ((cl_float) (CL_UCHAR_MAX));
  const cl_float f65535 = ((cl_float) (CL_USHRT_MAX));
  if (type == CL_SNORM_INT8)
    {
      /*  <-1.0, 1.0> to <I*_MIN, I*_MAX> */
      cl_float2 colorf;
      for (i = 0; i < 2; i++)
        colorf.s[i] = color.s[i] * f127;
      cl_char2 final_color = convert_char2_sat (colorf);
      *((cl_char2 *)data) = final_color;
      return;
    }
  if (type == CL_SNORM_INT16)
    {
      cl_float2 colorf;
      for (i = 0; i < 2; i++)
        colorf.s[i] = color.s[i] * f32767;
      cl_short2 final_color = convert_short2_sat (colorf);
      *((cl_short2 *)data) = final_color;
      return;
    }
  if (type == CL_UNORM_INT8)
    {
      /* <0, I*_MAX> to <0.0, 1.0> */
      /*  <-1.0, 1.0> to <I*_MIN, I*_MAX> */
      cl_float2 colorf;
      for (i = 0; i < 2; i++)
        colorf.s[i] = color.s[i] * f255;
      cl_uchar2 final_color = convert_uchar2_sat (colorf);
      *((cl_uchar2 *)data) = final_color;
      return;
    }
  if (type == CL_UNORM_INT16)
    {
      cl_float2 colorf;
      for (i = 0; i < 2; i++)
        colorf.s[i] = color.s[i] * f65535;
      cl_ushort2 final_color = convert_ushort2_sat (colorf);
      *((cl_ushort2 *)data) = final_color;
      return;
    }

  return;
}

/* only for CL_FLOAT, CL_SNORM_INT8, CL_UNORM_INT8,
 * CL_SNORM_INT16, CL_UNORM_INT16 channel types */
static void
write_float_pixel (cl_float color, void *data, int type)
{
  if (type == CL_FLOAT)
    {
      *((float *)data) = color;
      return;
    }
  if (type == CL_HALF_FLOAT)
    {
      /* TODO: convert to builtins */
      *((uint16_t *)data) = float_to_half (color);
      return;
    }
  const cl_float f127 = ((cl_float)CL_CHAR_MAX);
  const cl_float f32767 = ((cl_float)CL_SHRT_MAX);
  const cl_float f255 = ((cl_float)CL_UCHAR_MAX);
  const cl_float f65535 = ((cl_float)CL_USHRT_MAX);
  if (type == CL_SNORM_INT8)
    {
      /*  <-1.0, 1.0> to <I*_MIN, I*_MAX> */
      cl_float colorf = color * f127;
      cl_char final_color = convert_char_sat (colorf);
      *((cl_char *)data) = final_color;
      return;
    }
  if (type == CL_SNORM_INT16)
    {
      cl_float colorf = color * f32767;
      cl_short final_color = convert_short_sat (colorf);
      *((cl_short *)data) = final_color;
      return;
    }
  if (type == CL_UNORM_INT8)
    {
      /* <0, I*_MAX> to <0.0, 1.0> */
      /*  <-1.0, 1.0> to <I*_MIN, I*_MAX> */
      cl_float colorf = color * f255;
      cl_uchar final_color = convert_uchar_sat (colorf);
      *((cl_uchar *)data) = final_color;
      return;
    }
  if (type == CL_UNORM_INT16)
    {
      cl_float colorf = color * f65535;
      cl_ushort final_color = convert_ushort_sat (colorf);
      *((cl_ushort *)data) = final_color;
      return;
    }

  return;
}

/* for use inside filter functions
 * no channel mapping
 * no pointers to img metadata */
static void
pocl_write_pixel_fast_ui (cl_uint4 color, int order, int elem_size, void *data)
{
  if (order == CL_A)
    {
      if (elem_size == 1)
        *((cl_uchar *)data) = convert_uchar_sat_int (color.s[3]);
      else if (elem_size == 2)
        *((cl_ushort *)data) = convert_ushort_sat_int (color.s[3]);
      else if (elem_size == 4)
        *((cl_uint *)data) = color.s[3];
      return;
    }
  if (order == CL_R)
    {
      if (elem_size == 1)
        *((cl_uchar *)data) = convert_uchar_sat_int (color.s[0]);
      else if (elem_size == 2)
        *((cl_ushort *)data) = convert_ushort_sat_int (color.s[0]);
      else if (elem_size == 4)
        *((cl_uint *)data) = color.s[0];
      return;
    }

  if (order == CL_RG)
    {
      cl_uint2 tmp = {color.s[0], color.s[1]};
      if (elem_size == 1)
        *((cl_uchar2 *)data) = convert_uchar2_sat_int (tmp);
      else if (elem_size == 2)
        *((cl_ushort2 *)data) = convert_ushort2_sat_int (tmp);
      else if (elem_size == 4) {
        cl_uint2 tmp = { color.s[0], color.s[0] };
        *((cl_uint2 *)data) = tmp;
      }
      return;
    }

  if (elem_size == 1)
    {
      *((cl_uchar4 *)data) = convert_uchar4_sat_int (color);
    }
  else if (elem_size == 2)
    {
      *((cl_ushort4 *)data) = convert_ushort4_sat_int (color);
    }
  else if (elem_size == 4)
    {
      *((cl_uint4 *)data) = color;
    }

  return;
}

/* for use inside filter functions
 * no channel mapping
 * no pointers to img metadata */
static void
pocl_write_pixel_fast_f (cl_float4 color, int channel_type, int order,
                         void *data)
{
  if (order == CL_A)
    {
      write_float_pixel (color.s[3], data, channel_type);
    }
  else if (order == CL_R)
    {
      write_float_pixel (color.s[0], data, channel_type);
    }
  else if (order == CL_RG)
    {
      cl_float2 tmp = { color.s[0], color.s[0] };
      write_float2_pixel (tmp, data, channel_type);
    }
  else
    {
      write_float4_pixel (color, data, channel_type);
    }

  return;
}

/* for use inside filter functions
 * no channel mapping
 * no pointers to img metadata */
static void
pocl_write_pixel_fast_i (cl_int4 color, int order, int elem_size, void *data)
{
  if (order == CL_A)
    {
      if (elem_size == 1)
        *((cl_char *)data) = convert_char_sat_int (color.s[3]);
      else if (elem_size == 2)
        *((cl_short *)data) = convert_short_sat_int (color.s[3]);
      else if (elem_size == 4)
        *((cl_int *)data) = color.s[3];
      return;
    }

  if (order == CL_R)
    {
      if (elem_size == 1)
        *((cl_char *)data) = convert_char_sat_int (color.s[0]);
      else if (elem_size == 2)
        *((cl_short *)data) = convert_short_sat_int (color.s[0]);
      else if (elem_size == 4)
        *((cl_int *)data) = color.s[0];
      return;
    }

  if (order == CL_RG)
    {
      cl_int2 tmp = {color.s[0], color.s[1]};
      if (elem_size == 1)
        *((cl_char2 *)data) = convert_char2_sat_int (tmp);
      else if (elem_size == 2)
        *((cl_short2 *)data) = convert_short2_sat_int (tmp);
      else if (elem_size == 4) {
        cl_int2 tmp = { color.s[0], color.s[0] };
        *((cl_int2 *)data) = tmp;
      }
      return;
    }

  if (elem_size == 1)
    {
      *((cl_char4 *)data) = convert_char4_sat_int (color);
    }
  else if (elem_size == 2)
    {
      *((cl_short4 *)data) = convert_short4_sat_int (color);
    }
  else if (elem_size == 4)
    {
      *((cl_int4 *)data) = color;
    }
  return;
}

/* full write with channel map conversion etc
 * Writes a four element pixel to an image pixel pointed by integer coords.
 */
void
pocl_write_pixel_zero (void *data, const cl_uint4 input_color, int order,
                       int elem_size, int channel_type)
{
  cl_uint4 in = map_channels (input_color, order);

  typedef union
  {
    cl_uint4 ui;
    cl_int4 i;
    cl_float4 f;
  } u;

  u ucolor;
  ucolor.ui = in;

  if ((channel_type == CL_SIGNED_INT8) || (channel_type == CL_SIGNED_INT16)
      || (channel_type == CL_SIGNED_INT32))
    pocl_write_pixel_fast_i (ucolor.i, order, elem_size, data);
  else if ((channel_type == CL_UNSIGNED_INT8)
           || (channel_type == CL_UNSIGNED_INT16)
           || (channel_type == CL_UNSIGNED_INT32))
    pocl_write_pixel_fast_ui (ucolor.ui, order, elem_size, data);
  else // TODO unsupported channel types
    pocl_write_pixel_fast_f (ucolor.f, channel_type, order, data);
}
