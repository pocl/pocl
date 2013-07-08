#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "templates.h"

typedef int dev_sampler_t;

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
  int num_channels;
  int elem_size;
} dev_image_t;

#endif
