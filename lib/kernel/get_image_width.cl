
#include "templates.h"
#include "image.h"

#define IMPLEMENT_GET_IMAGE_WIDTH(__IMGTYPE__)               \
  int _CL_OVERLOADABLE get_image_width(__IMGTYPE__ image){   \
    return (*(dev_image_t**)&image)->width;                  \
  }                                                          \


IMPLEMENT_GET_IMAGE_WIDTH(image1d_t)
IMPLEMENT_GET_IMAGE_WIDTH(image2d_t)
IMPLEMENT_GET_IMAGE_WIDTH(image3d_t)

