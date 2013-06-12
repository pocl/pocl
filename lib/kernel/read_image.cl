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

#ifndef _CL_HAS_IMAGE_ACCESS

#include "templates.h"
#include "image.h"

/* checks if coord is out of bounds. If out of bounds: Sets coord in bounds 
   and returns false OR populates color with border colour and returns true.
   If in bounds, returns false */
inline int _pocl_out_of_bounds( dev_image_t* image, int4 coord, 
                                sampler_t sampler, uint4 *color)
{
  if( sampler & CLK_ADDRESS_CLAMP_TO_EDGE )
    {
      if (coord.x >= image->width)
        coord.x = image->width-1;
      if (coord.y >= image->height)
        coord.y = image->height-1;
      if (image->depth != 0 && coord.z >= image->depth)
        coord.z = image->depth-1;

      if (coord.x < 0)
        coord.x = 0;
      if (coord.y < 0) 
        coord.y = 0;
      if (image->depth != 0 && coord.z < 0)
        coord.z = 0;

      return false;
    }
  if (sampler & CLK_ADDRESS_CLAMP)
    {    
      if(coord.x >= image->width || coord.x < 0 ||
         coord.y >= image->height || coord.y < 0 ||
         ( image->depth != 0 && ( coord.z >= image->depth || coord.z < 0 )))
        {
          (*color)[0] = 0;
          (*color)[1] = 0;
          (*color)[2] = 0;

          if ( image->order == CL_A || image->order == CL_INTENSITY || 
               image->order == CL_RA || image->order == CL_ARGB || 
               image->order == CL_BGRA || image->order == CL_RGBA )
            (*color)[3] = 0;
            
          else
            (*color)[3] = 1; 

          return true;
        }
    }
  return false;
}


void read_pixel( uint* color, dev_image_t* image, int4 coord )
{
  int i, idx;
  int width = image->width;
  int height = image->height;
  int num_channels = image->num_channels;
  int elem_size = image->elem_size;
    
  for (i = 0; i < num_channels; i++)
    { 
      idx = i + (coord.x + coord.y*width + coord.z*height*width) * num_channels;
      if (elem_size == 1)
        {
          color[i] = ((uchar*)(image->data))[idx];
        }
      if (elem_size == 2)
        {
          color[i] = ((ushort*)image->data)[idx];
        }
      if (elem_size == 4)
        {
          color[i] = ((uint*)image->data)[idx];
        }
    }
}

/* float functions: Not implemented

float4 _CL_OVERLOADABLE read_imagef (image2d_t image, sampler_t sampler,
                                     int2 coord) 
{ 

}

float4 _CL_OVERLOADABLE read_imagef (image2d_t image, sampler_t sampler,
                                     float2 coord) 
{

}

float4 _CL_OVERLOADABLE read_imagef (image2d_t image, int2 coord ) {
             
}


float4 _CL_OVERLOADABLE read_imagef (image2d_array_t image, int4 coord ) 
{
       
}

float4 _CL_OVERLOADABLE read_imagef (image2d_array_t image, 
                                     sampler_t sampler, int4 coord ) 
{
       
}

float4 _CL_OVERLOADABLE read_imagef (image2d_array_t image, 
                                     sampler_t sampler, float4 coord ) 
{

}

*/

/* int functions */

uint4 _CL_OVERLOADABLE read_imageui( image2d_t image, sampler_t sampler, 
                                     int2 coord )
{
  uint4 color;
  if (_pocl_out_of_bounds (image, (int4)(coord, 0, 0), sampler, &color))
    {
      return color;
    }  
  read_pixel ((uint*)&color, (dev_image_t*)image, (int4)(coord, 0, 0));
  return color;    
}

uint4 _CL_OVERLOADABLE read_imageui( dev_image_t* image, sampler_t sampler, 
                                     int4 coord )
{
  uint4 color;
  if (_pocl_out_of_bounds( image, coord, sampler, &color ))
    {
      return color;
    }  
  read_pixel ((uint*)&color, (dev_image_t*)image, coord);
  return color;    
}

int4 _CL_OVERLOADABLE read_imagei( image2d_array_t image, sampler_t sampler, 
                                   int2 coord )
{
  int4 color;
  if (_pocl_out_of_bounds (image, (int4)(coord, 0, 0), sampler, 
                           (uint4*)&color ) )
    {
      return color;
    }  
  read_pixel ((uint*)&color, (dev_image_t*)image, (int4)(coord, 0, 0));
  return color;
}


#endif

