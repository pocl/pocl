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

#if (__clang_major__ == 3) && (__clang_minor__ >= 5)
// Clang 3.5 crashes in case trying to cast to the private pointer,
// adding the global qualifier fixes it. Clang 3.4 crashes if it's
// there. The issue is in SROA.
#define ADDRESS_SPACE global
#else
#define ADDRESS_SPACE
#endif

/* checks if integer coord is out of bounds. If out of bounds: Sets coord in
   bounds and returns false OR populates color with border colour and returns
   true. If in bounds, returns false */
int __pocl_is_out_of_bounds (ADDRESS_SPACE dev_image_t* dev_image, int4 coord,
                             dev_sampler_t* dev_sampler, void *color_)
{
  uint4 *color = (uint4*)color_;
  if(*dev_sampler & CLK_ADDRESS_CLAMP_TO_EDGE)
    {
      if (coord.x >= dev_image->_width)
        coord.x = dev_image->_width-1;
      if (dev_image->_height != 0 && coord.y >= dev_image->_height)
        coord.y = dev_image->_height-1;
      if (dev_image->_depth != 0 && coord.z >= dev_image->_depth)
        coord.z = dev_image->_depth-1;

      if (coord.x < 0)
        coord.x = 0;
      if (coord.y < 0)
        coord.y = 0;
      if (coord.z < 0)
        coord.z = 0;

      return 0;
    }
  if (*dev_sampler & CLK_ADDRESS_CLAMP)
    {
      if(coord.x >= dev_image->_width || coord.x < 0 ||
         coord.y >= dev_image->_height || coord.y < 0 ||
         (dev_image->_depth != 0 && (coord.z >= dev_image->_depth || coord.z <0)))
        {
          (*color)[0] = 0;
          (*color)[1] = 0;
          (*color)[2] = 0;

          if (dev_image->_order == CL_A || dev_image->_order == CL_INTENSITY ||
              dev_image->_order == CL_RA || dev_image->_order == CL_ARGB ||
              dev_image->_order == CL_BGRA || dev_image->_order == CL_RGBA)
            (*color)[3] = 0;
          else
            (*color)[3] = 1;

          return 1;
        }
    }
  return 0;
}

/* Reads a four element pixel from image pointed by integer coords. */
void __pocl_read_pixel (void* color, ADDRESS_SPACE dev_image_t* dev_image, int4 coord)
{

  uint4* color_ptr = (uint4*)color;
  int width = dev_image->_width;
  int height = dev_image->_height;
  int num_channels = dev_image->_num_channels;
  int i = num_channels;
  int elem_size = dev_image->_elem_size;
  int const base_index =
    (coord.x + coord.y*width + coord.z*height*width) * num_channels;

  if (dev_image->_order == CL_A)
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
      if(dev_image->_order == CL_BGRA)
        {
          (*color_ptr)[0] = ((uchar*)(dev_image->_data))[base_index + 2];
          (*color_ptr)[1] = ((uchar*)(dev_image->_data))[base_index + 1];
          (*color_ptr)[2] = ((uchar*)(dev_image->_data))[base_index + 0];
          (*color_ptr)[3] = ((uchar*)(dev_image->_data))[base_index + 3]; 
        }
      else
        {
          while (i--)
            {
              (*color_ptr)[i] = ((uchar*)(dev_image->_data))[base_index + i];
            }
        }
    }
  else if (elem_size == 2)
    {
      if(dev_image->_order == CL_BGRA)
        {
          (*color_ptr)[0] = ((ushort*)(dev_image->_data))[base_index + 2];
          (*color_ptr)[1] = ((ushort*)(dev_image->_data))[base_index + 1];
          (*color_ptr)[2] = ((ushort*)(dev_image->_data))[base_index + 0];
          (*color_ptr)[3] = ((ushort*)(dev_image->_data))[base_index + 3]; 
        }
      else
        {
          while (i--)
            {
              (*color_ptr)[i] = ((ushort*)(dev_image->_data))[base_index + i];
            }
      }
    }
  else if (elem_size == 4)
    {
      if(dev_image->_order == CL_BGRA)
        {
          (*color_ptr)[0] = ((uint*)(dev_image->_data))[base_index + 2];
          (*color_ptr)[1] = ((uint*)(dev_image->_data))[base_index + 1];
          (*color_ptr)[2] = ((uint*)(dev_image->_data))[base_index + 0];
          (*color_ptr)[3] = ((uint*)(dev_image->_data))[base_index + 3]; 
        }
      else
        {
          while (i--)
            {
              (*color_ptr)[i] = ((uint*)(dev_image->_data))[base_index + i];
            }
        }
    }
}


/* Implementation for read_image with any image data type and int coordinates
   __IMGTYPE__ = image type (image2d_t, ...)
   __RETVAL__  = return value (int4 or uint4 float4)
   __POSTFIX__ = function name postfix (i, ui, f)
   __COORD__   = coordinate type (int, int2, int4)
*/
#define IMPLEMENT_READ_IMAGE_INT_COORD(__IMGTYPE__,__RETVAL__,__POSTFIX__,\
                                            __COORD__)                  \
  __RETVAL__ _CL_OVERLOADABLE read_image##__POSTFIX__ (__IMGTYPE__ image, \
                                                       sampler_t sampler, \
                                                       __COORD__ coord) \
  {                                                                     \
    __RETVAL__ color;                                                   \
    int4 coord4;                                                        \
    INITCOORD##__COORD__(coord4, coord);                                \
    if (__pocl_is_out_of_bounds (*(ADDRESS_SPACE dev_image_t**)&image, coord4, (dev_sampler_t*)&sampler, &color)) \
      {                                                                 \
        return color;                                                   \
      }                                                                 \
    __pocl_read_pixel (&color, (*(ADDRESS_SPACE dev_image_t**)&image), coord4); \
                                                                        \
    return color;                                                       \
  }                                                                     \


/* read_image function instantions */
IMPLEMENT_READ_IMAGE_INT_COORD(image2d_t, uint4, ui, int2)
IMPLEMENT_READ_IMAGE_INT_COORD(image2d_t, int4, i, int2)
IMPLEMENT_READ_IMAGE_INT_COORD(image3d_t, uint4, ui, int4)
IMPLEMENT_READ_IMAGE_INT_COORD(image2d_t, float4, f, int2)
