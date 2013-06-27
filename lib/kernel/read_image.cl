/* OpenCL built-in library: read_image()

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

/*#ifndef _CL_HAS_IMAGE_ACCESS*/

#include "templates.h"
#include "image.h"
#include "pocl_image_rw_utils.h"

/* float functions: Not implemented

float4 _CL_OVERLOADABLE read_imagef (image2d_t image, sampler_t sampler,
                                     int2 coord) 
{ 

}

float4 _CL_OVERLOADABLE read_imagef (image2d_t image, sampler_t sampler,
                                     float2 coord) 
{

}

float4 _CL_OVERLOADABLE read_imagef (image2d_t image, int2 coord) {
             
}


float4 _CL_OVERLOADABLE read_imagef (image2d_array_t image, int4 coord) 
{
       
}

float4 _CL_OVERLOADABLE read_imagef (image2d_array_t image, 
                                     sampler_t sampler, int4 coord) 
{
       
}

float4 _CL_OVERLOADABLE read_imagef (image2d_array_t image, 
                                     sampler_t sampler, float4 coord) 
{

}

*/

/* int functions */

uint4 _CL_OVERLOADABLE read_imageui (image2d_t image, sampler_t sampler, 
                                     int2 coord)
{
  uint4 color;
  int4 coord4;
  coord4.x = coord.x;
  coord4.y = coord.y;
  coord4.z = 0;
  coord4.w = 0;

  if(coord.x == 0 && coord.y == 0)
    printf("nolla koords \n");

  if (pocl_out_of_bounds (&image, coord4, sampler, &color))
    {
      return color;
    }  
  pocl_read_pixel ((uint*)&color, &image, coord4);
  
  return color;    
}

uint4 _CL_OVERLOADABLE read_imageui (image2d_t image, sampler_t sampler, 
                                     int4 coord)
{
  uint4 color;
  if (pocl_out_of_bounds(&image, coord, sampler, &color))
    {
      return color;
    }  
  pocl_read_pixel ((uint*)&color, &image, coord);
  return color;    
}

uint4 _CL_OVERLOADABLE read_imageui (image3d_t image, sampler_t sampler, 
                                     int4 coord)
{
  uint4 color;
  if (pocl_out_of_bounds(&image, coord, sampler, &color))
    {
      return color;
    }  
  pocl_read_pixel ((uint*)&color, &image, coord);
  return color;    
}

int4 _CL_OVERLOADABLE read_imagei (image2d_array_t image, sampler_t sampler, 
                                   int2 coord)
{
  int4 color;
  int4 coord4;
  coord4.x = coord.x;
  coord4.y = coord.y;
  coord4.z = 0;
  coord4.w = 0;
  
  if (pocl_out_of_bounds (&image, coord4, sampler, &color))
    {
      return color;
    }  
  pocl_read_pixel ((uint*)&color, &image, coord4);
  return color;
}


/*#endif*/

