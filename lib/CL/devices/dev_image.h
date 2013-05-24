#ifndef __X86_IMAGE_H__
#define __X86_IMAGE_H__

//Definition of the image datatype used on basic and pthread (and probably tce?)

typedef cl_int dev_sampler_t;

typedef struct dev_image_t {
  void* data;
  cl_int width;
  cl_int height;
  cl_int depth;
  cl_int image_array_size;
  cl_int row_pitch;
  cl_int slice_pitch;
  cl_int num_mip_levels; /* maybe not needed */
  cl_int num_samples; /* maybe not needed */
  cl_int order;
  cl_int data_type;
} dev_image_t;

#endif
