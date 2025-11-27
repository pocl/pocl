/* Correctly-rounded arc-sine for binary16 value.

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

/* degree-5 minimax polynomial for asin(x) over [0,0.25], with relative error
   bounded by 2^-20.817, with coefficients of odd degree only, and degree-1
   coefficient forced to 1
*/
static constant float p0[] = {1.0f, 0x1.552e8ep-3f, 0x1.43696ep-4f};

/* degree-7 minimax polynomial for asin(x) over [0.25,0.5], with relative error
   bounded by 2^-19.202, with coefficients of odd degree only, and degree-1
   coefficient forced to 1
*/
static constant float p1[] = {1.0f, 0x1.55a7ep-3f, 0x1.258e54p-4f, 0x1.08e44p-4f};

_CL_OVERLOADABLE half asin (half x)
{
  b32u32_u v = {.f = x};
  uint u = v.u;
  uint au = u & 0x7fffffffu;

#define HALF_PI 0x1.921fb6p+0f

  if (au >= 0x3f800000u) { // NaN, Inf, or |x| >= 1
    if (au == 0x3f800000u) // |x| = 1
      return (u == 0x3f800000u) ? HALF_PI : -HALF_PI;
    if ((au >> 23) == 0x3ff && ((au & 0x7fffff) != 0)) // qNaN or sNaN
      return x + x;
    return NAN;
  }

  float t = v.f, tt, c1, c5, y;

  /* for |x| < 0.5 we don't reduce t into -t, since the polynomials
     p0 and p1 are valid also on the negative side */

  int reduce = au >= 0x3f000000u; // |x| >= 0.5
  if (reduce) {
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
    /* For |x| <= 0x1.71p-5, asin(x) rounds to x to nearest,
       we deal with that case separately, so that for x subnormal
       and a power of two, we get an underflow. */
    if (au <= 0x3d388000u && !reduce) {
      if (au == 0) return x;
      return (au == u) ? v.f + 0x1p-26f : v.f - 0x1p-26f;
    }
    if (au == 0x3dd30000u) // |x| = 0x1.a6p-4
      return (au == u) ? 0x1.a6c00ap-4f : -0x1.a6c00ap-4f;
    if (au == 0x3d688000u) // |x| = 0x1.d1p-5
      return (au == u) ? 0x1.d14004p-5f : -0x1.d14004p-5f;
    if (au == 0x3dfa0000u) // |x| = 0x1.f4p-4
      return (au == u) ? 0x1.f5400ap-4f : -0x1.f5400ap-4f;
    /* Warning for rounding toward -Inf: let p0(t) = t*q(t). If we first
       compute q(t) and then multiply by t, for tiny t and rounding we will
       get q(t)=1, and then t, whereas the correct result is nextbelow(t). */
    c1 = fma (p0[2], tt, p0[1]);
    y = fma (c1, tt * t, t);
  }
  else { // 0.25 <= t <= 0.5
    if (au == 0x3eb24000u) // |x| = 0x1.648p-2
      return (au == u) ? 0x1.6c2012p-2f : -0x1.6c2012p-2f;
    if (au == 0x3ed96000u) // |x| = 0x1.b2cp-2
      return (au == u) ? 0x1.c0fffap-2f : -0x1.c0fffap-2f;
    if (au == 0x3ef0a000u) // |x| = 0x1.e14p-2
      return (au == u) ? 0x1.f4fffp-2f : -0x1.f4fffp-2f;
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

  return y;
}
