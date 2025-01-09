/* image_test_utils.h - helper functions useful for testing image related DBKs.

   Copyright (C) 2024 Robin Bijl / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
 */

#ifndef _IMAGE_TEST_UTILS_H_
#define _IMAGE_TEST_UTILS_H_

#include "pocl_export.h"
#include "stdint.h"

/**
 * Calculate the Peak Signal to Noise Ration (psnr) of two images.
 *
 * \param height [in] height of both images.
 * \param width [in] width of both images.
 * \param pixel_stride [in] number of bytes to compare, 3 for RGB and 1 for
 * grayscale.
 * \param image [in] source/ground truth image.
 * \param approx [in] image to be compared.
 * \return PSNR as a double.
 */
double calculate_PSNR (int const height,
                       int const width,
                       int const pixel_stride,
                       uint8_t const *restrict image,
                       uint8_t const *restrict approx);

#endif //_IMAGE_TEST_UTILS_H_
