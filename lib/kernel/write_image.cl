#ifndef _CL_HAS_IMAGE_ACCESS

#include "templates.h"
#include "image.h"
#include "image_utils.h"


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



void write_pixel( uint* color, dev_image_t* image, int4 coord )
{
  
  int i, idx;
  int width = image->width;
  int height = image->height;
  int num_channels = 4;
  int elem_size = get_image_elem_size( image );
  
  //printf("write: coord.x = %u, coord.y = %u, coord.z = %u \n", coord.x, coord.y, coord.z);
  //printf("write_imageui: : offset=%u\n", (coord.x + (coord.y * width) + (coord.z*height*width) )*num_channels);
  
  for ( i = 0; i < num_channels; i++ )
    {
      idx = i + (coord.x + coord.y*width + coord.z*height*width) * num_channels;
      if ( elem_size == 1 )
        {
          ((uchar*)image->data)[idx] = (color[i] >> 24);          
        }
      if ( elem_size == 2 )
        {
          ((ushort*)image->data)[idx] = (color[i] >> 16);
        }
      if ( elem_size == 4 )
        {
          ((uint*)image->data)[idx] = color[i];
        }
    }
}

void _CL_OVERLOADABLE write_imagef (     image2d_t image,
        int2 coord,
        float4 color) {
  ((float4*)image->data)[ coord.x + coord.y*image->row_pitch ] = color;
}

void _CL_OVERLOADABLE write_imagei (     image2d_t image,
        int2 coord,
        int4 color) {
  ((float4*)image->data)[ coord.x + coord.y*image->row_pitch ] = (float4)(color.x,color.y,color.z,color.w);
}

void _CL_OVERLOADABLE write_imageui (dev_image_t* image,
                                     int2 coord,      
                                     uint4 color){
  
  if(coord.x == 0 && coord.y == 0){
    int *i_ptr = &color;
    printf("write_image(int2): color[0]=%X, color[1]=%X\n", i_ptr[0], i_ptr[1]);
  }
  write_pixel( (uint*)&color, (dev_image_t*)image, (int4)(coord, 0, 0) );
  
}

void _CL_OVERLOADABLE write_imageui (dev_image_t* image,
                                     int4 coord,      
                                     uint4 color)
{
  if(coord.x == 0 && coord.y == 0){
    int *i_ptr = &color;
    printf("write_image(int4): color[0]=%X, color[1]=%X\n", i_ptr[0], i_ptr[1]);
  }
  write_pixel( (uint*)&color, (dev_image_t*)image, coord );
}
/*

void _CL_OVERLOADABLE write_imagef (
     image2d_array_t image,
     int4 coord,
     float4 color){

}

void _CL_OVERLOADABLE write_imagei (
     image2d_array_t image,
     int4 coord,
     int4 color){

}

void _CL_OVERLOADABLE write_imageui (
     image2d_array_t image,
     int4 coord,
     uint4 color){

}

void _CL_OVERLOADABLE write_imagef (image1d_t image,
     int2 coord,
     float4 color){

}

void _CL_OVERLOADABLE write_imagei (image1d_t image,
     int coord,
     int4 color){

}


void _CL_OVERLOADABLE write_imageui (image1d_t image, 
                                     int2 coord, 
                                     uint4 color){
  

}


void _CL_OVERLOADABLE write_imagef ( 
     image1d_buffer_t image, 
     int2 coord, 
     float4 color){

}

void _CL_OVERLOADABLE write_imagei (
     image1d_buffer_t image,
     int2 coord,
     int4 color){

}

void _CL_OVERLOADABLE write_imageui (
     image1d_buffer_t image,
     int2 coord,
     uint4 color){

}

void _CL_OVERLOADABLE write_imagef (
     image1d_array_t image,
     int2 coord,
     float4 color){

}

void _CL_OVERLOADABLE write_imagei (     
     image1d_array_t image,
     int2 coord,
     int4 color){

}

void _CL_OVERLOADABLE write_imageui (
     image1d_array_t image,
     int2 coord,
     uint4 color){

}

************ */

#endif
