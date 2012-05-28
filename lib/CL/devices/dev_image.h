#ifndef __X86_IMAGE_H__
#define __X86_IMAGE_H__

//Definition of the image datatype used on basic and pthread (and probably tce?)

typedef cl_int dev_sampler_t;

typedef struct dev_image2d_t {
  void* data;
  cl_int width;
  cl_int height;
  cl_int rowpitch;
  cl_int order;
  cl_int data_type;
} dev_image2d_t;

#endif
