/* OpenCL built-in library: get_image_array_size()

   Copyright (c) 2017 Michal Babej / Tampere University of Technology

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

#ifndef LLVM_OLDER_THAN_3_8

#define IMPLEMENT_GET_IMAGE_ARRAY_SIZE(__IMGTYPE__)             \
size_t _CL_OVERLOADABLE get_image_array_size(__IMGTYPE__ image) \
{                                                               \
  global dev_image_t* img =                                     \
    __builtin_astype (image, global dev_image_t*);              \
  return (size_t)(img->_image_array_size);                      \
}

IMPLEMENT_GET_IMAGE_ARRAY_SIZE (IMG_RO_AQ image1d_array_t)
IMPLEMENT_GET_IMAGE_ARRAY_SIZE (IMG_RO_AQ image2d_array_t)

#ifdef CLANG_HAS_IMAGE_AS
IMPLEMENT_GET_IMAGE_ARRAY_SIZE (IMG_WO_AQ image1d_array_t)
IMPLEMENT_GET_IMAGE_ARRAY_SIZE (IMG_WO_AQ image2d_array_t)
#endif

#ifdef CLANG_HAS_RW_IMAGES
IMPLEMENT_GET_IMAGE_ARRAY_SIZE (IMG_RW_AQ image1d_array_t)
IMPLEMENT_GET_IMAGE_ARRAY_SIZE (IMG_RW_AQ image2d_array_t)
#endif


#endif
