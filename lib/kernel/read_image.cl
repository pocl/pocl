/* OpenCL built-in library: read_image()

   Copyright (c) 2013 Ville Korhonen
   Copyright (c) 2014 Felix Bytow
   Copyright (c) 2015 Matias Koskela

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

static const int ARGB_MAP[] = { 3, 0, 1, 2 };
static const int BGRA_MAP[] = { 2, 1, 0, 3 };

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


/* checks if integer coord is out of bounds. If out of bounds: Sets coord in
   bounds and returns false OR populates color with border colour and returns
   true. If in bounds, returns false */
static int __pocl_is_out_of_bounds (global dev_image_t* dev_image, int4 *coord,
                             dev_sampler_t* dev_sampler)
{
  if(*dev_sampler & CLK_ADDRESS_CLAMP_TO_EDGE)
    {
      if (coord->x >= dev_image->_width)
        coord->x = dev_image->_width-1;
      if (dev_image->_height != 0 && coord->y >= dev_image->_height)
        coord->y = dev_image->_height-1;
      if (dev_image->_depth != 0 && coord->z >= dev_image->_depth)
        coord->z = dev_image->_depth-1;

      if (coord->x < 0)
        coord->x = 0;
      if (coord->y < 0)
        coord->y = 0;
      if (coord->z < 0)
        coord->z = 0;

      return 0;
    }
  if (*dev_sampler & CLK_ADDRESS_CLAMP)
    {
      if(coord->x >= dev_image->_width || coord->x < 0 ||
         coord->y >= dev_image->_height || coord->y < 0 ||
         (dev_image->_depth != 0 && (coord->z >= dev_image->_depth || coord->z <0)))
        {
          if (dev_image->_order == CL_A || dev_image->_order == CL_INTENSITY ||
              dev_image->_order == CL_RA || dev_image->_order == CL_ARGB ||
              dev_image->_order == CL_BGRA || dev_image->_order == CL_RGBA)
            return 1; // clamp to 0,0,0,0
          else
            return 2; // clamp to 0,0,0,1
        }
    }
  return 0;
}

/* Reads a four element pixel from image pointed by integer coords. */
static void __pocl_read_pixel (void* color, global dev_image_t* dev_image, int4 coord)
{

  uint4* color_ptr = (uint4*)color;
  int width = dev_image->_width;
  int height = dev_image->_height;
  int num_channels = dev_image->_num_channels;
  int i = num_channels;
  int order = dev_image->_order;
  int elem_size = dev_image->_elem_size;
  int const base_index =
    (coord.x + coord.y*width + coord.z*height*width) * num_channels;

  if (order == CL_A)
    {
      /* these can be garbage
      (*color_ptr)[0] = 0;
      (*color_ptr)[1] = 0;
      (*color_ptr)[2] = 0;
      */
      if (elem_size == 1)
        (*color_ptr)[3] = ((uchar*)(dev_image->_data))[base_index];
      else if (elem_size == 2)
        (*color_ptr)[3] = ((ushort*)(dev_image->_data))[base_index];
      else if (elem_size == 4)
        (*color_ptr)[3] = ((uint*)(dev_image->_data))[base_index];
      return;
    }


  if (elem_size == 1)
    {
      for (int i=0; i<num_channels; i++)
        {
          (*color_ptr)[map_channel(i, order)] = ((uchar*)(dev_image->_data))[base_index + i];
        }
    }
  else if (elem_size == 2)
    {
      for (int i=0; i<num_channels; i++)
        {
          (*color_ptr)[map_channel(i, order)] = ((ushort*)(dev_image->_data))[base_index + i];
        }
    }
  else if (elem_size == 4)
    {
      for (int i=0; i<num_channels; i++)
        {
          (*color_ptr)[map_channel(i, order)] = ((uint*)(dev_image->_data))[base_index + i];
        }
    }

}


static constant float4 maxval8_i = ((float4)(1.0f / (float)UCHAR_MAX));
static constant float4 maxval16_i = ((float4)(1.0f / (float)USHRT_MAX));
static constant float4 maxval8_2i = ((float4)(2.0f / (float)UCHAR_MAX));
static constant float4 maxval16_2i = ((float4)(2.0f / (float)USHRT_MAX));
static constant float4 minval8 = ((float4)(SCHAR_MIN));
static constant float4 minval16 = ((float4)(SHRT_MIN));

// only for CL_SNORM_INT8, CL_UNORM_INT8, CL_SNORM_INT16, CL_UNORM_INT16,
static float4 convert_uint4_to_float4(uint4 color, int type, int elem_size)
{
  if ((type == CL_SNORM_INT8) ||
      (type == CL_SNORM_INT16))
    {
      //  <I*_MIN, I*_MAX> to <-1.0, 1.0>
      // Imin,Imax -> 0, Umax -> / UmaxHalf -> -1.0f
      if (elem_size == 1)
        return ((convert_float4(as_int4(color)) - minval8) * maxval8_2i - (float4)1.0f);
      else
        return ((convert_float4(as_int4(color)) - minval16) * maxval16_2i -(float4)1.0f);
    }
  else
    {
      //  <0, I*_MAX> to <0.0, 1.0>
      if (elem_size == 1)
        return convert_float4(color) * maxval8_i;
      else
        return convert_float4(color) * maxval16_i;
    }
}

#define HANDLE_FLOAT_IMGTYPE \
    if (i_ptr->_data_type == CL_FLOAT)                                  \
      {                                                                 \
        float4 color;                                                   \
        __pocl_read_pixel (&color, i_ptr, coord4);                      \
        return color;                                                   \
      }                                                                 \
    else                                                                \
      {                                                                 \
        uint4 color;                                                    \
        __pocl_read_pixel (&color, i_ptr, coord4);                      \
        return convert_uint4_to_float4(color,                           \
                                     i_ptr->_data_type,                 \
                                     i_ptr->_elem_size);                \
      }                                                                 \
    }



/* Implementation for read_image with any image data type and int coordinates
   __IMGTYPE__ = image type (image2d_t, ...)
   __RETVAL__  = return value (int4 or uint4 float4)
   __POSTFIX__ = function name postfix (i, ui, f)
   __COORD__   = coordinate type (int, int2, int4)
*/


#if __clang_major__ > 3
// After Clang 4.0, the sampler_t is passed as an opaque struct (ptr)
// which we convert to int32 with the LLVM pass HandleSamplerInitialization.
#define READ_SAMPLER                                                    \
    dev_sampler_t s = *__builtin_astype(sampler, dev_sampler_t*);
#else
// Before Clang 4.0, the sampler_t was passed as an int32.
#define READ_SAMPLER                                                    \
    dev_sampler_t s = __builtin_astype(sampler, dev_sampler_t);
#endif


#define SAMPLERING(VEC)                                                 \
    READ_SAMPLER                                                        \
    int r = __pocl_is_out_of_bounds (i_ptr, &coord4, &s);               \
    if (r == 1)                                                         \
        return (VEC)0;                                                  \
    if (r == 2)                                                         \
        return (VEC)(0,0,0,1);


#define IMPLEMENT_READ_IMAGE_INT_COORD_INT4(__IMGTYPE__,__RETVAL__,     \
                                            __POSTFIX__,__COORD__)      \
  __RETVAL__ _CL_OVERLOADABLE read_image##__POSTFIX__ (__IMGTYPE__ image, \
                                                       sampler_t sampler, \
                                                       __COORD__ coord) \
  {                                                                     \
    __RETVAL__ color;                                                   \
    int4 coord4;                                                        \
    INITCOORD##__COORD__(coord4, coord);                                \
    global dev_image_t* i_ptr =                                         \
      __builtin_astype (image, global dev_image_t*);                    \
    SAMPLERING(__RETVAL__)                                              \
    __pocl_read_pixel (&color, i_ptr, coord4);                          \
    return color;                                                       \
  }

#define IMPLEMENT_READ_IMAGE_INT_COORD_FLOAT4(__IMGTYPE__,              \
                                              __COORD__)                \
  float4 _CL_OVERLOADABLE read_imagef (__IMGTYPE__ image,               \
                                                       sampler_t sampler, \
                                                       __COORD__ coord) \
  {                                                                     \
    int4 coord4;                                                        \
    INITCOORD##__COORD__(coord4, coord);                                \
    global dev_image_t* i_ptr =                                         \
      __builtin_astype (image, global dev_image_t*);                    \
    SAMPLERING(float4)                                                  \
    HANDLE_FLOAT_IMGTYPE



/* NO Sampler Implementation for read_image with any image data type
   and int coordinates
   __IMGTYPE__ = image type (image2d_t, ...)
   __RETVAL__  = return value (int4 or uint4 float4)
   __POSTFIX__ = function name postfix (i, ui, f)
   __COORD__   = coordinate type (int, int2, int4)
   OCL 1.2 Spec Says
   The samplerless read image functions behave exactly as the
   corresponding read image functions that take integer
   coordinates and a sampler with filter mode set to
   CLK_FILTER_NEAREST, normalized coordinates set to
   CLK_NORMALIZED_COORDS_FALSE and
   addressing mode to CLK_ADDRESS_NONE.
   CLK_ADDRESS_NONE – for this addressing mode the programmer guarantees that
   the image coordinates used to sample elements of the image refer to a
   location inside the image; otherwise the results are undefined.
   Thus we do not need out of bound check for now.
   If we need sampler for other cases it has to be defined as default sampler:
   const sampler_t defualt_sampler = CLK_NORMALIZED_COORDS_FALSE |
                                  CLK_ADDRESS_NONE |
                                  CLK_FILTER_NEAREST;
*/
#define IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_INT4(__IMGTYPE__, __RETVAL__, \
                                                 __POSTFIX__, __COORD__) \
  __RETVAL__ _CL_OVERLOADABLE read_image##__POSTFIX__ (__IMGTYPE__ image, \
                                                       __COORD__ coord) \
  {                                                                     \
    __RETVAL__ color;                                                   \
    int4 coord4;                                                        \
    INITCOORD##__COORD__ (coord4, coord);                               \
    global dev_image_t* i_ptr =                                         \
    __builtin_astype (image, global dev_image_t*);                      \
    __pocl_read_pixel (&color, i_ptr, coord4);                          \
                                                                        \
    return color;                                                       \
  }                                                                     \


#define IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_FLOAT4(__IMGTYPE__,    \
                                                        __COORD__)      \
  float4 _CL_OVERLOADABLE read_imagef (__IMGTYPE__ image,               \
                                                       __COORD__ coord) \
  {                                                                     \
    int4 coord4;                                                        \
    INITCOORD##__COORD__ (coord4, coord);                               \
    global dev_image_t* i_ptr =                                         \
      __builtin_astype (image, global dev_image_t*);                    \
    HANDLE_FLOAT_IMGTYPE





/* read_image 2d function instantions */

IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_FLOAT4 ( IMG_RO_AQ image2d_t, int2)
IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_FLOAT4 ( IMG_RO_AQ image2d_array_t, int4)
IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_INT4 ( IMG_RO_AQ image2d_t, uint4, ui, int2)
IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_INT4 ( IMG_RO_AQ image2d_t, int4, i, int2)
IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_INT4 ( IMG_RO_AQ image2d_array_t, uint4, ui, int2)
IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_INT4 ( IMG_RO_AQ image2d_array_t, int4, i, int2)


IMPLEMENT_READ_IMAGE_INT_COORD_FLOAT4 ( IMG_RO_AQ image2d_t, int2)
IMPLEMENT_READ_IMAGE_INT_COORD_FLOAT4 ( IMG_RO_AQ image2d_array_t, int4)

IMPLEMENT_READ_IMAGE_INT_COORD_INT4 ( IMG_RO_AQ image2d_t, uint4, ui, int2)
IMPLEMENT_READ_IMAGE_INT_COORD_INT4 ( IMG_RO_AQ image2d_t, int4, i, int2)

// read_image 3d function instantions
IMPLEMENT_READ_IMAGE_INT_COORD_INT4 ( IMG_RO_AQ image3d_t, uint4, ui, int4)
IMPLEMENT_READ_IMAGE_INT_COORD_FLOAT4 ( IMG_RO_AQ image3d_t, int4)


#if CLANG_HAS_IMAGE_AS

IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_FLOAT4 ( IMG_RW_AQ image2d_t, int2)
IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_FLOAT4 ( IMG_RW_AQ image2d_array_t, int4)
IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_INT4 ( IMG_RW_AQ image2d_t, uint4, ui, int2)
IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_INT4 ( IMG_RW_AQ image2d_t, int4, i, int2)
IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_INT4 ( IMG_RW_AQ image2d_array_t, uint4, ui, int2)
IMPLEMENT_READ_IMAGE_INT_COORD_NOSAMPLER_INT4 ( IMG_RW_AQ image2d_array_t, int4, i, int2)


IMPLEMENT_READ_IMAGE_INT_COORD_FLOAT4 ( IMG_RW_AQ image2d_t, int2)
IMPLEMENT_READ_IMAGE_INT_COORD_FLOAT4 ( IMG_RW_AQ image2d_array_t, int4)

IMPLEMENT_READ_IMAGE_INT_COORD_INT4 ( IMG_RW_AQ image2d_t, uint4, ui, int2)
IMPLEMENT_READ_IMAGE_INT_COORD_INT4 ( IMG_RW_AQ image2d_t, int4, i, int2)

// read_image 3d function instantions
IMPLEMENT_READ_IMAGE_INT_COORD_INT4 ( IMG_RW_AQ image3d_t, uint4, ui, int4)
IMPLEMENT_READ_IMAGE_INT_COORD_FLOAT4 ( IMG_RW_AQ image3d_t, int4)

#endif
