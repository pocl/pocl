/* OpenCL built-in library: write_image()

   Copyright (c) 2013 Ville Korhonen 
   
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

#include "templates.h"
#include "pocl_image_rw_utils.h"

static constant const int ARGB_MAP[] = { 3, 0, 1, 2 };
static constant const int BGRA_MAP[] = { 2, 1, 0, 3 };

static int map_channel(int i, int order) {
  switch(order)
    {
      case CL_ARGB: return ARGB_MAP[i];
      case CL_BGRA: return BGRA_MAP[i];
      case CL_RGBA:
      default: return i;
    }
  return i;
}

/* writes pixel to coord in image */
static void pocl_write_pixel (void* color_, global dev_image_t* dev_image,
                       int4 coord)
{
  uint4 *color = (uint4*)color_;
  int width = dev_image->_width;
  int height = dev_image->_height;
  int num_channels = dev_image->_num_channels;
  int order = dev_image->_order;
  int elem_size = dev_image->_elem_size;
  int const base_index =
    (coord.x + coord.y*width + coord.z*height*width) * num_channels;

  if (order == CL_A)
    {
      if (elem_size == 1)
        ((uchar*) (dev_image->_data))[base_index] = (*color)[3];
      else if (elem_size == 2)
        ((ushort*) (dev_image->_data))[base_index] = (*color)[3];
      else if (elem_size == 4)
        ((uint*) (dev_image->_data))[base_index] = (*color)[3];
      return;
    }

  if (elem_size == 1)
    {
      for (int i=0; i<num_channels; i++)
        {
          ((uchar*) (dev_image->_data))[base_index + i] =
                  (*color)[map_channel(i, order)];
        }
    }
  else if (elem_size == 2)
    {
      for (int i=0; i<num_channels; i++)
        {
          ((ushort*) dev_image->_data)[base_index + i] =
                  (*color)[map_channel(i, order)];
        }
    }
  else if (elem_size == 4)
    {
      for (int i=0; i<num_channels; i++)
        {
          ((uint*) dev_image->_data)[base_index + i] =
                  (*color)[map_channel(i, order)];
        }
    }
}

static constant float4 maxval8 = (float4)UCHAR_MAX;
static constant float4 maxval16 = (float4)USHRT_MAX;
static constant float4 maxval8_2 = (float4)((float)UCHAR_MAX / 2.0f);
static constant float4 maxval16_2 = (float4)((float)USHRT_MAX / 2.0f);
static constant float4 minval8 = ((float4)(SCHAR_MIN));
static constant float4 minval16 = ((float4)(SHRT_MIN));

/* only for CL_SNORM_INT8, CL_UNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT16 */
static uint4 convert_float4_to_uint4(float4 color, int type, int elem_size)
{
  if ((type == CL_SNORM_INT8) ||
      (type == CL_SNORM_INT16))
    {
      // <-1.0, 1.0> to <I*_MIN, I*_MAX>
      // -1.0,1.0 -> 0.0,2.0 -> * Umax/2.0 -> + Smin
      float4 color_p1 = (color + (float4)1.0f);
      if (elem_size == 1)
        return convert_uint4(color_p1 * maxval8_2 + minval8);
      else
        return convert_uint4(color_p1 * maxval16_2 + minval16);
    }
  else
    {
      if (elem_size == 1)
        return convert_uint4(maxval8 * color);
      else
        return convert_uint4(maxval16 * color);
    }
}

/*
write_imagei can only be used with image objects created with
image_channel_data_type set to one of the following values:
CL_SIGNED_INT8, CL_SIGNED_INT16, and CL_SIGNED_INT32.

write_imageui functions can only be used with image objects created with
image_channel_data_type set to one of the following values:
CL_UNSIGNED_INT8, CL_UNSIGNED_INT16, or CL_UNSIGNED_INT32.
*/

#define IMPLEMENT_WRITE_IMAGE_INT_COORD(__IMGTYPE__,__POSTFIX__,        \
                                        __COORD__,__DTYPE__)            \
  void _CL_OVERLOADABLE write_image##__POSTFIX__(__IMGTYPE__ image,     \
                                                  __COORD__ coord,      \
                                                  __DTYPE__ color)      \
  {                                                                     \
    int4 coord4;                                                        \
    INITCOORD##__COORD__(coord4, coord);                                \
    global dev_image_t* i_ptr = __builtin_astype (image, global dev_image_t*); \
    uint4 color2 = as_uint4(color);                                     \
    pocl_write_pixel (&color2, i_ptr, coord4);                          \
  }                                                                     \

/*
 * write_imagef can only be used with image objects created with
 * image_channel_data_type set to one of the pre-defined packed formats,
 * or set to CL_SNORM_INT8, CL_UNORM_INT8, CL_SNORM_INT16,
 * CL_UNORM_INT16, CL_HALF_FLOAT or CL_FLOAT.
*/

#define IMPLEMENT_WRITE_IMAGE_INT_COORD_FLOAT4(__IMGTYPE__,             \
                                               __COORD__)               \
  void _CL_OVERLOADABLE write_imagef             (__IMGTYPE__ image,    \
                                                  __COORD__ coord,      \
                                                 float4 color)          \
  {                                                                     \
    int4 coord4;                                                        \
    INITCOORD##__COORD__(coord4, coord);                                \
    global dev_image_t* i_ptr = __builtin_astype (image, global dev_image_t*); \
    if (i_ptr->_data_type == CL_FLOAT)                                  \
      pocl_write_pixel (&color, i_ptr, coord4);                         \
    else                                                                \
    {                                                                   \
      uint4 color2 = convert_float4_to_uint4(color,                     \
                                             i_ptr->_data_type,         \
                                             i_ptr->_elem_size);        \
      pocl_write_pixel (&color2, i_ptr, coord4);                        \
    }                                                                   \
  }                                                                     \


IMPLEMENT_WRITE_IMAGE_INT_COORD ( IMG_WO_AQ image2d_t, ui, int2, uint4)
IMPLEMENT_WRITE_IMAGE_INT_COORD ( IMG_WO_AQ image2d_t, ui, int4, uint4)

IMPLEMENT_WRITE_IMAGE_INT_COORD ( IMG_WO_AQ image2d_t, i, int2, int4)
IMPLEMENT_WRITE_IMAGE_INT_COORD ( IMG_WO_AQ image2d_t, i, int4, int4)

IMPLEMENT_WRITE_IMAGE_INT_COORD_FLOAT4 ( IMG_WO_AQ image2d_t, int2)
IMPLEMENT_WRITE_IMAGE_INT_COORD_FLOAT4 ( IMG_WO_AQ image3d_t, int4)

#ifdef CLANG_HAS_RW_IMAGES

IMPLEMENT_WRITE_IMAGE_INT_COORD ( IMG_RW_AQ image2d_t, ui, int2, uint4)
IMPLEMENT_WRITE_IMAGE_INT_COORD ( IMG_RW_AQ image2d_t, ui, int4, uint4)

IMPLEMENT_WRITE_IMAGE_INT_COORD ( IMG_RW_AQ image2d_t, i, int2, int4)
IMPLEMENT_WRITE_IMAGE_INT_COORD ( IMG_RW_AQ image2d_t, i, int4, int4)

IMPLEMENT_WRITE_IMAGE_INT_COORD_FLOAT4 ( IMG_RW_AQ image2d_t, int2)
IMPLEMENT_WRITE_IMAGE_INT_COORD_FLOAT4 ( IMG_RW_AQ image3d_t, int4)

#endif
