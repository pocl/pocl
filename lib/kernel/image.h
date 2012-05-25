#ifndef __IMAGE_H__
#define __IMAGE_H__

#include "templates.h"

typedef struct image2d_t_ {
  uchar4* data;
  int width;
  int height;
  int rowpitch;
  int order;
  int data_type;
} image2d_t_;

#endif
