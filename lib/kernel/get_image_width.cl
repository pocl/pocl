
#include "templates.h"
#include "image.h"

int pocl_get_image_width (void* image)
{
  return (*(dev_image_t**)image)->width;
}

int get_image_width (image2d_t image)
{
  return pocl_get_image_width(&image);
}

