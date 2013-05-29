


#ifndef _CL_HAS_IMAGE_ACCESS

#include "templates.h"
#include "image.h"

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
  uint4 color = ((uint4*)image->data)[ coord.x + coord.y*image->row_pitch ];
  return color;    
}

uint4 _CL_OVERLOADABLE read_imageui ( image2d_array_t image, sampler_t sampler, 
                                      int4 coord )
{
  uint4 color = ((uint4*)image->data)[ coord.x + coord.y*image->row_pitch ];
  return color; 
}

int4 _CL_OVERLOADABLE read_imagei ( image2d_array_t image, sampler_t sampler, 
                                    int2 coord )
{
  int4 color = ((int4*)image->data)[ coord.x + coord.y*image->row_pitch ];
  return color;
}


#endif

