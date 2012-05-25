#include "templates.h"

#include "image.h"

float4 _cl_overloadable read_imagef ( image2d_t image,
        sampler_t sampler,
        int2 coord) {
  //TODO: Sampler options
  if( coord.x<0 )
    coord.x=0;
  if( coord.y<0 )
    coord.y=0;
  if( coord.x>=image->width )
    coord.x=image->width-1;
  if( coord.x>=image->height )
    coord.x=image->height-1;

  float4 color = ((float4*)image->data)[ coord.x + coord.y*image->rowpitch ];

  return color;
}

float4 _cl_overloadable read_imagef ( image2d_t image,
        sampler_t sampler,
        float2 coord) {
  
  float4 color = ((float4*)image->data)[ (int)coord.x + (int)coord.y*image->rowpitch ];

  return color;
}
