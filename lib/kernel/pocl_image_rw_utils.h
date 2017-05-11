/* OpenCL built-in library: read_image() and write_image() helper macros

   Copyright (c) 2013 Ville Korhonen 
   
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
#ifndef POCL_IMAGE_RW_UTILS_H
#define POCL_IMAGE_RW_UTILS_H

/* coordinate initialization */
#define INITCOORDint(dest, source){             \
  dest.x = source;                              \
  dest.y = 0;                                   \
  dest.z = 0;                                   \
  dest.w = 0;                                   \
  }

#define INITCOORDint2(dest, source){            \
  dest.x = source.x;                            \
  dest.y = source.y;                            \
  dest.z = 0;                                   \
  dest.w = 0;                                   \
  }

#define INITCOORDint4(dest, source){                 \
  dest.x = source.x;                                 \
  dest.y = source.y;                                 \
  dest.z = source.z;                                 \
  dest.w = source.w;                                 \
  }

#define INITCOORDfloat(dest, source)                                          \
  {                                                                           \
    dest.x = source;                                                          \
    dest.y = 0.0f;                                                            \
    dest.z = 0.0f;                                                            \
    dest.w = 0.0f;                                                            \
  }

#define INITCOORDfloat2(dest, source)                                         \
  {                                                                           \
    dest.x = source.x;                                                        \
    dest.y = source.y;                                                        \
    dest.z = 0.0f;                                                            \
    dest.w = 0.0f;                                                            \
  }

#define INITCOORDfloat4(dest, source)                                         \
  {                                                                           \
    dest.x = source.x;                                                        \
    dest.y = source.y;                                                        \
    dest.z = source.z;                                                        \
    dest.w = source.w;                                                        \
  }

#endif
