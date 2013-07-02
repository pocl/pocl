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
#include "image.h"
#include "pocl_image_rw_utils.h"

/* writes pixel to coord in image */
void pocl_write_pixel (void* color_, void* image, int4 coord)
{  
  dev_image_t* dev_image = *((dev_image_t**)image);
  uint4 *color = (uint4*)color_;
  int i, idx;
  int width = dev_image->width;
  int height = dev_image->height;
  int num_channels = dev_image->num_channels;
  int elem_size = dev_image->elem_size;

  for (i = 0; i < num_channels; i++)
    {
      idx = i + (coord.x + coord.y*width + coord.z*height*width)*num_channels;
      if (elem_size == 1)
        {
          ((uchar*)dev_image->data)[idx] = (*color)[i];          
        }
      if (elem_size == 2)
        {
          ((ushort*)dev_image->data)[idx] = (*color)[i];
        }
      if (elem_size == 4)
        {
          ((uint*)dev_image->data)[idx] = (*color)[i];
        }
    }
}

/* Implementation for write_image with any image data type and int coordinates 
   __IMGTYPE__ = image type (image2d_t, ...)
   __DTYPE__  = data type to be read (int4 or uint4 float4)
   __POSTFIX__ = function name postfix (i, ui, f)
   __COORD__   = coordinate type (int, int2, int4)
*/
#define IMPLEMENTATION_WRITE_IMAGE_INT_COORD(__IMGTYPE__,__DTYPE__,__POSTFIX__,__COORD__)\
  void _CL_OVERLOADABLE write_image##__POSTFIX__ (__IMGTYPE__ image,    \
                                                  __COORD__ coord,      \
                                                  __DTYPE__ color)      \
  {                                                                     \
  int4 coord4;                                                          \
  INITCOORD##__COORD__(coord4, coord);                                  \
  pocl_write_pixel (&color, &image, coord4);                             \
  }                                                                     \

IMPLEMENTATION_WRITE_IMAGE_INT_COORD(image2d_t, uint4, ui, int2)

/* Not implemented yet

void _CL_OVERLOADABLE write_imagef (image2d_t image, int2 coord, float4 color) 
{
  ((float4*)image->data)[ coord.x + coord.y*image->row_pitch ] = color;
}

void _CL_OVERLOADABLE write_imagei (image2d_t image, int2 coord, int4 color) 
{
  ((float4*)image->data)[ coord.x + coord.y*image->row_pitch ] = 
    (float4)(color.x,color.y,color.z,color.w);
}

void _CL_OVERLOADABLE write_imagef (image2d_array_t image, int4 coord,
                                    float4 color)
{

}

void _CL_OVERLOADABLE write_imagei (image2d_array_t image, int4 coord,
                                    int4 color)
{

}

void _CL_OVERLOADABLE write_imageui (image2d_array_t image, int4 coord,
                                     uint4 color)
{

}

void _CL_OVERLOADABLE write_imagef (image1d_t image, int2 coord, float4 color){

}

void _CL_OVERLOADABLE write_imagei (image1d_t image, int coord, int4 color)
{

}


void _CL_OVERLOADABLE write_imageui (image1d_t image, int2 coord, uint4 color)
{
  

}


void _CL_OVERLOADABLE write_imagef (image1d_buffer_t image, int2 coord, 
                                    float4 color)
{

}

void _CL_OVERLOADABLE write_imagei (image1d_buffer_t image, int2 coord, 
                                    int4 color){

}

void _CL_OVERLOADABLE write_imageui (image1d_buffer_t image, int2 coord,
                                     uint4 color){

}

void _CL_OVERLOADABLE write_imagef (image1d_array_t image, int2 coord,
                                    float4 color)
{

}

void _CL_OVERLOADABLE write_imagei (image1d_array_t image, int2 coord, 
                                    int4 color)
{

}

void _CL_OVERLOADABLE write_imageui (image1d_array_t image, int2 coord,
                                     uint4 color)
{

}

*/


