
#include "templates.h"
#include "image.h"

int pocl_get_image_height(void *image)
{
  return ((dev_image_t*)image)->height;
}
int get_image_height (image2d_t image)
{
  return pocl_get_image_height(&image);
}


