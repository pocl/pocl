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
#include "image_utils.h"

inline int out_of_bounds( dev_image_t* image, int4 coord, sampler_t sampler, 
                          uint4 *color)
{
  if( sampler & CLK_ADDRESS_CLAMP )
    {
      if ( coord.x >= image->width )
        coord.x = image->width-1;
      if ( coord.y >= image->height )
        coord.y = image->height-1;
      if ( coord.z >= image->depth )
        coord.z = image->depth-1;

      if ( coord.x < 0 )
        coord.x = 0;
      if ( coord.y < 0 ) 
        coord.y = 0;
      if ( coord.z < 0 )
        coord.z = 0;

      return false;
    }
  if ( sampler & CLK_ADDRESS_CLAMP_TO_EDGE )
    {    
      if(coord.x >= image->width || coord.x < 0 ||
         coord.y >= image->width || coord.y < 0 ||
         coord.z >= image->depth || coord.z < 0)
        {
          if ( image->order == CL_A || image->order == CL_INTENSITY || 
               image->order == CL_RA || image->order == CL_ARGB || 
               image->order == CL_BGRA || image->order == CL_RGBA )
            {
              (*color)[0] = 0;
              (*color)[1] = 0;
              (*color)[2] = 0;
              (*color)[3] = 0;
            }
          else
            {
              (*color)[0] = 0;
              (*color)[1] = 0;
              (*color)[2] = 0;
              (*color)[3] = 1; /*needs 1.0f conversion to int*/
            }
          return true;
        }
    }
  return false;
}


inline int get_image_elem_size(dev_image_t *image)
{
  int ch_type = image->data_type;
  if ( ch_type == CL_SNORM_INT8 || ch_type == CL_UNORM_INT8 ||
       ch_type == CL_SIGNED_INT8 || ch_type == CL_UNSIGNED_INT8 )
    {
      return 1; /* 1 byte */
    }
  else if (ch_type == CL_UNSIGNED_INT32 || ch_type == CL_SIGNED_INT32 ||
           ch_type == CL_FLOAT || ch_type == CL_UNORM_INT_101010 )
    {
      return 4; /* 32bit -> 4 bytes */
    }
  else if (ch_type == CL_SNORM_INT16 || ch_type == CL_UNORM_INT16 ||
           ch_type == CL_SIGNED_INT16 || ch_type == CL_UNSIGNED_INT16 ||
           ch_type == CL_UNORM_SHORT_555 || ch_type == CL_UNORM_SHORT_565 ||
           ch_type == CL_HALF_FLOAT)
    {
      return 2; /* 16bit -> 2 bytes */
    }
  return 0;
}




/* TODO handle possible 1 channel orders */
void read_pixel( uint* color, dev_image_t* image, int4 coord )
{
  int i, idx;
  int width = image->width;
  int height = image->height;
  int num_channels = 4;
  int elem_size = get_image_elem_size( image );
  if(coord.x == 0 && coord.y == 0)
    printf("read_image(): input[0]: %X height=%u, width=%u\n", ((uint*)(image->data))[0], height, width);
  
  for ( i = 0; i < num_channels; i++ )
    { 
      idx = i + (coord.x + coord.y*width + coord.z*height*width) * num_channels;
      if ( elem_size == 1 )
        {
          
          color[i] = ( ((uchar*)(image->data))[idx] << 24 );
          if(coord.x == 0 && coord.y == 0)
            printf("read_image(): color: %X coord.z=%u offset=%u\n", color[i], coord.z,  i + ((coord.x + coord.y*width + coord.z*height*width) * num_channels));

        }
      if ( elem_size == 2 )
        {
          color[i] = ( ((ushort*)image->data)[idx] << 16 );
        }
      if ( elem_size == 4 )
        {
          color[i] = ((uint*)image->data)[idx];
        }
    }
}

/* float functions */

float4 _CL_OVERLOADABLE read_imagef ( image2d_t image,
        sampler_t sampler,
        int2 coord) {
  //TODO: Sampler options
  if( coord.x<0 )
    coord.x=0;
  if( coord.y<0 )
    coord.y=0;
  if( coord.x>=image->width )
    coord.x=image->width-1;
  if( coord.y>=image->height )
    coord.y=image->height-1;

  float4 color = ((float4*)image->data)[ coord.x + coord.y*image->row_pitch ];

  return color;
}

float4 _CL_OVERLOADABLE read_imagef ( image2d_t image,
        sampler_t sampler,
        float2 coord) {
  
  float4 color = ((float4*)image->data)[ (int)coord.x + (int)coord.y*image->row_pitch ];

  return color;
}



/* 
float4 _CL_OVERLOADABLE read_imagef ( image2d_t image, int2 coord ) {
             
}


float4 _CL_OVERLOADABLE read_imagef ( image2d_array_t image, int4 coord ) {
       
}

float4 _CL_OVERLOADABLE read_imagef ( image2d_array_t image, 
       sampler_t sampler, int4 coord ) {
       
}

float4 _CL_OVERLOADABLE read_imagef ( image2d_array_t image, 
       sampler_t sampler, float4 coord ) {

}

************************** */

/* int functions */

uint4 _CL_OVERLOADABLE read_imageui ( image2d_t image, sampler_t sampler, 
                                      int2 coord )
{
  uint4 color;
  if( out_of_bounds ( image, (int4)(coord, 0, 0), sampler, &color ) )
    {
      return color;
    }  
  read_pixel( (uint*)&color, (dev_image_t*)image, (int4)(coord, 0, 0) );
  if(coord.x == 0 && coord.y == 0){
    int *i_ptr = &color;
    printf("read_image(int2): color[0]=%X, color[1]=%X \n", i_ptr[0], i_ptr[1]);
  }
  //uint4 color = ((uint4*)image->data)[ coord.x + coord.y*image->row_pitch ];
  return color;    
}

uint4 _CL_OVERLOADABLE read_imageui ( dev_image_t* image, sampler_t sampler, 
                                      int4 coord )
{
  uint4 color;
  if( out_of_bounds ( image, coord, sampler, &color ) )
    {
      return color;
    }  
  read_pixel( (uint*)&color, (dev_image_t*)image, coord );
  if(coord.x == 0 && coord.y == 0){
    int *i_ptr = &color;
    printf("read_image(int4): color[0]=%X, color[1]=%X \n", i_ptr[0], i_ptr[1]);
  }
  return color;    
}

int4 _CL_OVERLOADABLE read_imagei ( image2d_array_t image, sampler_t sampler, 
                                    int2 coord )
{
  int4 color = ((int4*)image->data)[ coord.x + coord.y*image->row_pitch ];
  return color;
}


#endif

