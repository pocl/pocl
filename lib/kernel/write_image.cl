#include "templates.h"

#include "image.h"

//typedef struct image2d_t_* image2d_t;

void _cl_overloadable write_imagef (     image2d_t image,
        int2 coord,
        float4 color) {
  //printf( "w %f %f %f %f\n", color.x, color.y, color.z, color.w );

  ((float4*)image->data)[ coord.x + coord.y*image->rowpitch ] = color;
}

void _cl_overloadable write_imagei (     image2d_t image,
        int2 coord,
        int4 color) {
  //printf( "%d\n", image->rowpitch );
  //printf( "%d\n", 10 );
  image->data[ (int)(coord.x) + (int)(coord.y)*image->rowpitch ] = (uchar4)(color.x, color.y, color.z, color.w);
}
