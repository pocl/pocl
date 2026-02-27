/* Correctly-rounded half-revolution arc-sine for binary16 value.

Copyright (c) 2025 Paul Zimmermann

This file is part of the CORE-MATH project
(https://core-math.gitlabpages.inria.fr/).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "common_types.h"

// the following polynomials were generated using Sollya (cf asin.sollya)

/* degree-7 minimax polynomial for asin(x) over [0,0.25], with relative error
   bounded by 2^-26.989, with coefficients of odd degree only, and degree-1
   coefficient forced to 1
*/
static constant float p0[] = {1.0f, 0x1.5555fp-3f, 0x1.32b088p-4f, 0x1.8baf46p-5f};

/* degree-7 minimax polynomial for asin(x) over [0.25,0.5], with relative error
   bounded by 2^-19.202, with coefficients of odd degree only, and degree-1
   coefficient forced to 1. This polynomial was optimized to reduce the
   number of exceptions.
*/
static constant float p1[] = {0x1.000002p+0f, 0x1.55a7ep-3f, 0x1.258e54p-4f, 0x1.08e44p-4f};

#define HALF_PI 0x1.921fb4p+0f
#define HALF 0.5f
#define INV_PI 0x1.45f306p-2f

_CL_OVERLOADABLE half asinpi (half x)
{
  b32u32_u v = {.f = x};
  uint u = v.u;
  uint au = u & 0x7fffffffu;

  if (au >= 0x3f800000u) { // NaN, Inf, or |x| >= 1
    if (au == 0x3f800000u) // |x| = 1
      return (u == 0x3f800000u) ? HALF : -HALF;
    if ((au >> 23) == 0x3ff && ((au & 0x7fffff) != 0)) // qNaN or sNaN
      return x + x;
    return NAN;
  }

  float t = v.f, tt, c1, c3, c5, y;

  /* for |x| < 0.5 we don't reduce t into -t, since the polynomials
     p0 and p1 are valid also on the negative side */

  int reduce = au >= 0x3f000000u; // |x| >= 0.5
  if (reduce) {
    if (au == 0x3f10a000u) // |x|=0x1.214p-1
      return (au == u) ? 0x1.876016p-3f : -0x1.876016p-3f;
    if (au == 0x3f208000u) // |x|=0x1.41p-1
      return (au == u) ? 0x1.b9c002p-3f : -0x1.b9c002p-3f;
    if (au == 0x3f3ca000u) // |x|=0x1.794p-1
      return (au == u) ? 0x1.0dfffap-2f : -0x1.0dfffap-2f;
    if (u >> 31) // x < 0
      t = -t;
    // argument reduction: asin(x) = pi/2 - 2*asin(sqrt((1-x)/2))
    t = sqrt ((1.0f - t) * 0.5f);
    b32u32_u w = {.f = t};
    au = w.u & 0x7fffffffu;
  }

  // now 0 <= t <= 0.5
  tt = t * t;

  if (au < 0x3e800000u) { // 0 <= t < 0.25
    /* For |x| <= 0x1.92p-13, we get underflow, except when x=0x1.92p-13
       with RNDU and x=-0x1.92p-13 with RNDD. */
    if (au <= 0x39490000u) {
      // INV_PI is smaller than 1/pi
      return fma (t, INV_PI, 0x1p-25f * t);
    }
    if (au == 0x3e002000) // |x| = 0x1.004p-3
      return (au == u) ? 0x1.472002p-5f : -0x1.472002p-5f;
    c5 = fma (p0[3], tt, p0[2]); // degree 5
    c3 = fma (c5, tt, p0[1]);    // degree 3
    y = fma (c3, tt * t, t);
  }
  else { // 0.25 <= t <= 0.5
    if (au == 0x3eb7a000u) // |x|=0x1.6f4p-2
      return (au == u) ? 0x1.de400ep-4f : -0x1.de400ep-4f;
    if (au == 0x3ec4c000u) // |x|=0x1.898p-2
      return (au == u) ? 0x1.012006p-3f : -0x1.012006p-3f;
    c5 = fma (p1[3], tt, p1[2]);
    c1 = fma (p1[1], tt, p1[0]);
    y = fma (c5, tt * tt, c1);
    y = t * y;
  }

  if (reduce) // argument reconstruction
  {
    y = HALF_PI - 2.0f * y;
    if (u >> 31) // x < 0
      y = -y;
  }

  /* now y approximates asin(x), we divide by pi */

  return y * INV_PI;
}
