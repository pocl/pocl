#include "templates.h"

#include "image.h"

float4 _cl_overloadable read_imagef ( image2d_t image,
        sampler_t sampler,
        int2 coord) {
  if( coord.x<0 )
    coord.x=0;
  if( coord.y<0 )
    coord.y=0;
  // TODO

  float4 color = ((float4*)image->data)[ coord.x + coord.y*image->rowpitch ];


  //printf( "r %f %f %f %f\n", color.x, color.y, color.z, color.w );
  return color;
}

float4 _cl_overloadable read_imagef ( image2d_t image,
        sampler_t sampler,
        float2 coord) {
  
  uchar4 v = image->data[ (int)(coord.x) + (int)(coord.y)*image->rowpitch ];
  return (float4)(v.x,v.y,v.z,v.w);
}
