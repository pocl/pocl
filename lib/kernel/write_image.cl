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

#ifndef _CL_HAS_IMAGE_ACCESS

#include "templates.h"
#include "image.h"

/* writes pixel to coord in image */
void write_pixel (uint* color, dev_image_t* image, int4 coord)
{
  
  int i, idx;
  int width = image->width;
  int height = image->height;
  int num_channels = image->num_channels;
  int elem_size = image->elem_size;
  
  for (i = 0; i < num_channels; i++)
    {
      idx = i + (coord.x + coord.y*width + coord.z*height*width)*num_channels;
      if (elem_size == 1)
        {
          ((uchar*)image->data)[idx] = color[i];          
        }
      if (elem_size == 2)
        {
          ((ushort*)image->data)[idx] = color[i];
        }
      if (elem_size == 4)
        {
          ((uint*)image->data)[idx] = color[i];
        }
    }
}


void _CL_OVERLOADABLE write_imageui (dev_image_t* image, int2 coord, 
                                     uint4 color)
{
  write_pixel ((uint*)&color, (dev_image_t*)image, (int4)(coord, 0, 0));
}

void _CL_OVERLOADABLE write_imageui (dev_image_t* image, int4 coord, 
                                     uint4 color)
{
  write_pixel((uint*)&color, (dev_image_t*)image, coord);
}

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
                                    float4 color){

}

void _CL_OVERLOADABLE write_imagei (image1d_array_t image, int2 coord, 
                                    int4 color){

}

void _CL_OVERLOADABLE write_imageui (image1d_array_t image, int2 coord,
                                     uint4 color){

}

*/

#endif
