/* OpenCL built-in library: get_image_dim()

   Copyright (c) 2013-2014 Ville Korhonen, Pekka Jääskeläinen
                           Tampere University of Technology
   Copyright (c) 2015      Giuseppe Bilotta

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "templates.h"

#define IMPLEMENT_GET_IMAGE_DIM_2D(__IMG_AQ__)                  \
int2 _CL_OVERLOADABLE get_image_dim(__IMG_AQ__ image2d_t image) \
{                                                               \
  global dev_image_t* img =                                     \
    __builtin_astype(image, global dev_image_t*);               \
  return (int2)(img->_width, img->_height);                     \
}

#define IMPLEMENT_GET_IMAGE_DIM_2DA(__IMG_AQ__)                       \
int2 _CL_OVERLOADABLE get_image_dim(__IMG_AQ__ image2d_array_t image) \
{                                                                     \
  global dev_image_t* img =                                           \
    __builtin_astype (image, global dev_image_t*);                    \
  return (int2)(img->_width, img->_height);                           \
}

#define IMPLEMENT_GET_IMAGE_DIM_3D(__IMG_AQ__)                  \
int4 _CL_OVERLOADABLE get_image_dim(__IMG_AQ__ image3d_t image) \
{                                                               \
  global dev_image_t* img =                                     \
    __builtin_astype (image, global dev_image_t*);              \
  return (int4)(img->_width, img->_height, img->_depth, 0);     \
}

IMPLEMENT_GET_IMAGE_DIM_2D(IMG_RO_AQ)
IMPLEMENT_GET_IMAGE_DIM_2DA(IMG_RO_AQ)
IMPLEMENT_GET_IMAGE_DIM_3D(IMG_RO_AQ)

IMPLEMENT_GET_IMAGE_DIM_2D(IMG_WO_AQ)
IMPLEMENT_GET_IMAGE_DIM_2DA(IMG_WO_AQ)
IMPLEMENT_GET_IMAGE_DIM_3D(IMG_WO_AQ)

#ifdef CLANG_HAS_RW_IMAGES
IMPLEMENT_GET_IMAGE_DIM_2D(IMG_RW_AQ)
IMPLEMENT_GET_IMAGE_DIM_2DA(IMG_RW_AQ)
IMPLEMENT_GET_IMAGE_DIM_3D(IMG_RW_AQ)
#endif
