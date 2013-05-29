#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "templates.h"


typedef struct dev_image_t {
  void* data;
  int width;
  int height;
  int depth;
  int image_array_size;
  int row_pitch;
  int slice_pitch;
  int num_mip_levels; /* maybe not needed */
  int num_samples; /* maybe not needed */
  int order;
  int data_type;
} dev_image_t;


/*

typedef struct image2d_t_ {
  uchar4* data;
  int width;
  int height;
  int rowpitch;
  int order;
  int data_type;
} image2d_t_;
*/

#endif
