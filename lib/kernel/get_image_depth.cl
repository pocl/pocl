#include "templates.h"
#include "image.h"

#define IMPLEMENT_GET_IMAGE_DEPTH(__IMGTYPE__)              \
  int _CL_OVERLOADABLE get_image_depth(__IMGTYPE__ image){  \
    return (*(dev_image_t**)&image)->depth;                 \
  }                                                         \


IMPLEMENT_GET_IMAGE_DEPTH(image1d_t)

#if (__clang_major__ == 3) && (__clang_minor__ >= 3)
// with LLVM 3.2 the image types are not defined by the frontend
// and we reuse the same struct for all dimensions
IMPLEMENT_GET_IMAGE_DEPTH(image2d_t)
IMPLEMENT_GET_IMAGE_DEPTH(image3d_t)
#endif
