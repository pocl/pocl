#include "templates.h"
#include "image.h"

int pocl_get_image_depth(void *image)
{
  return (*(dev_image_t**)image)->depth;
}

#define IMPLEMENT_GET_IMAGE_DEPTH(__IMGTYPE__)              \
  int _CL_OVERLOADABLE get_image_depth(__IMGTYPE__ image){  \
    return (*(dev_image_t**)&image)->depth;                 \
  }                                                         \


IMPLEMENT_GET_IMAGE_DEPTH(image1d_t)
IMPLEMENT_GET_IMAGE_DEPTH(image2d_t)
IMPLEMENT_GET_IMAGE_DEPTH(image3d_t)
