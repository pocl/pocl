#ifndef _CL_HAS_IMAGE_ACCESS

#include "templates.h"
#include "image.h"

void _CL_OVERLOADABLE write_imagef (     image2d_t image,
        int2 coord,
        float4 color) {
  ((float4*)image->data)[ coord.x + coord.y*image->rowpitch ] = color;
}

void _CL_OVERLOADABLE write_imagei (     image2d_t image,
        int2 coord,
        int4 color) {
  ((float4*)image->data)[ coord.x + coord.y*image->rowpitch ] = (float4)(color.x,color.y,color.z,color.w);
}

#endif
