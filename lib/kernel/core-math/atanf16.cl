/* Correctly-rounded arc-tangent for binary16 value.

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

// the following polynomials were generated using Sollya (cf atan.sollya)

/* Degree-7 minimax polynomial for atan(x) over [0,0.25], with relative error
   bounded by 2^-25.419, with coefficients of odd degree only, and degree-1
   coefficient forced to 1. Coefficients were later optimized to reduce the
   number of exceptions.
*/
static constant float p0[] = {0x1.fffffcp-1, -0x1.55546cp-2, 0x1.98d0ep-3,
                           -0x1.0c7c54p-3};

/* degree-4 minimax polynomial for atan(x) over [0.25,0.5], with relative error
   bounded by 2^-22.573 */
static constant float p1[] = {0x1.411612p-14, 0x1.ff076cp-1, 0x1.1ee64cp-6,
                           -0x1.a6fc96p-2, 0x1.81dd3ep-3};

/* degree-4 minimax polynomial for atan(x) over [0.5,0.75], with relative error
   bounded by 2^-21.757 */
static constant float p2[] = {-0x1.95964cp-8, 0x1.0bad76p+0, -0x1.e652e2p-4,
                           -0x1.e70ab4p-3, 0x1.a5eb88p-4};

/* degree-4 minimax polynomial for atan(x) over [0.75,1], with relative error
   bounded by 2^-23.027 */
static constant float p3[] = {-0x1.f75b16p-6, 0x1.2d3d1p+0, -0x1.882376p-2,
                           0x1.d18c96p-13, 0x1.6aa268p-6};

_CL_OVERLOADABLE half atan (half x)
{
  b32u32_u v = {.f = x};
  uint u = v.u;
  uint au = u & 0x7fffffffu;

#define HALF_PI 0x1.921fb6p+0f

  if (au >= 0x7f800000u) { // NaN or Inf
    if (au == 0x7f800000u) // +/-Inf
      return (u == 0x7f800000u) ? HALF_PI : -HALF_PI;
    return NAN;
  }
  
  float t = v.f;

  // for x < 0 we use atan(-x) = -atan(x)
  int neg = u >> 31;
  static constant float Neg[] = {1.0f, -1.0f};
  float s = Neg[neg];
  t = t * s;

  // now t >= 0

  // for x > 1 we use atan(x) = pi/2 - atan(1/x)
  int reduce = au > 0x3f800000u;
  if (reduce) t = 1.0f / t;

  // now 0 <= t <= 1

  constant float *p;
  float y, c1, c2, c3, c5, tt = t * t;
  if (t <= 0.25f) {
    // for |x| < 0x1.d14p-6, atan(x) rounds to x to nearest
    if (!reduce && t <= 0x1.d14p-6f) {
      if (au == 0) return x; // x = 0
      t = s * t;
      return fma (t, -0x1p-23f, t);
    }
    
    // deal with exceptional cases
    if (au == 0x3e56a000u) // |x| = 0x1.ad4p-3
      return (au == u) ? 0x1.a72002p-3f : -0x1.a72002p-3f;
    if (au == 0x4115c000u) // |x| = 0x1.2b8p+3
      return (au == u) ? 0x1.76dffep+0f : -0x1.76dffep+0f;
    if (au == 0x42c32000u) // |x| = 0x1.864p+6
      return (au == u) ? 0x1.8f7ffep+0f : -0x1.8f7ffep+0f;

    p = p0;
    c5 = fma (p[3], tt, p[2]);
    c1 = fma (p[1], tt, p[0]);
    c1 = fma (c5, tt * tt, c1);
    y = t * c1;
  }
  else {
    p = (t <= 0.5f) ? p1 : (t <= 0.75f) ? p2 : p3;
    c3 = fma (p[4], t, p[3]);
    c2 = fma (c3, t, p[2]);
    y = fma (p[1], t, p[0]);
    y = fma (c2, tt, y);
  }

  if (reduce) y = HALF_PI - y; // argument reconstruction

  if (neg) y = -y;

  return y;
}

