/* image_test_utils.c - helper functions useful for testing image related DBKs.

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

#include "image_test_utils.h"

#include "math.h"

double
calculate_PSNR (int const height,
                int const width,
                int const pixel_stride,
                uint8_t const *restrict image,
                uint8_t const *restrict approx)
{

  uint64_t sum = 0;
  int points = height * width * pixel_stride;
  int max = 0;
  for (int i = 0; i < points; i++)
    {
      uint8_t image_i = image[i];
      int error = image_i - approx[i];
      sum += error * error;
      if (max < image_i)
        {
          max = image_i;
        }
    }
  double mse = (double)sum / points;

  return 20 * log10 (max) - 10 * log10 (mse);
}