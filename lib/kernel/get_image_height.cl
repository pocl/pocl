#include "templates.h"
#include "image.h"

#ifdef CLANG_OLDER_THAN_3_3

int get_image_height (image2d_t image)
{
  return image->height;
}

#endif
