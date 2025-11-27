/* Correctly-rounded half-revolution arc-tangent for binary16 value.

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
   coefficient forced to 1.
*/
static constant float p0[] = {0x1.fffffcp-1, -0x1.55546cp-2, 0x1.98d0ep-3,
                           -0x1.0c7c54p-3};

/* degree-4 minimax polynomial for atan(x) over [0.25,0.5], with relative error
   bounded by 2^-22.573, and coefficients later optimized */
static constant float p1[] = {0x1.411612p-14, 0x1.ff076ep-1, 0x1.1ee64cp-6,
                           -0x1.a6fc96p-2, 0x1.81dd3ep-3};

/* degree-4 minimax polynomial for atan(x) over [0.5,0.75], with relative error
   bounded by 2^-21.757, and coefficients later optimized */
static constant float p2[] = {-0x1.95964cp-8, 0x1.0bad74p+0, -0x1.e652e2p-4,
                           -0x1.e70ab4p-3, 0x1.a5eb88p-4};

/* degree-4 minimax polynomial for atan(x) over [0.75,1], with relative error
   bounded by 2^-23.027, and coefficients later optimized */
static constant float p3[] = {-0x1.f75b26p-6, 0x1.2d3d1cp+0, -0x1.882376p-2,
                           0x1.d18c84p-13, 0x1.6aa26p-6};

_CL_OVERLOADABLE half atanpi (half x)
{
  b32u32_u v = {.f = x};
  uint u = v.u;
  uint au = u & 0x7fffffffu;

#define HALF_PI 0x1.921fb6p+0f // approximation of pi/2
#define INV_PI  0x1.45f306p-2f // approximation of 1/pi

  if (au >= 0x7f800000u) { // NaN or Inf
    if (au == 0x7f800000u) // +/-Inf
      return (u == 0x7f800000u) ? 0.5f : -0.5f;
    return NAN;
  }
  
  float t = v.f;

  // we first approximate atan(x), then divide by pi

  // for x < 0 we use atan(-x) = -atan(x)
  int neg = u >> 31;
  static constant float Neg[] = {1.0f, -1.0f};
  float s = Neg[neg];
  t = t * s;

  // now t >= 0

  // for x > 1 we use atan(x) = pi/2 - atan(1/x)
  int reduce = au > 0x3f800000u;
  if (reduce) {
    // deal with exceptional cases
    if (au == 0x3fd24000u) return s * 0x1.4dbffep-2f; // |x| = 0x1.a48p+0
    if (au == 0x42ba4000u) return s * 0x1.fc8002p-2f; // |x| = 0x1.748p+6
    if (au == 0x412ca000u) return s * 0x1.e1dffep-2f;  // |x| = 0x1.594p+3
    if (au == 0x41fb4000u) return s * 0x1.f5a002p-2f; // |x| = 0x1.f68p+4
    t = 1.0f / t;
    v.f = t;
    au = v.u;
  }

  // now 0 <= t <= 1

  constant float *p;
  float y, c1, c2, c3, c5, tt = t * t;
  if (au < 0x3e800000u) { // t < 0.25
    if (!reduce && au < 0x3a800000u) { // t < 0x1p-10
      // use direct degree-1 polynomial for atanpi (cf atanpi.sollya)
      // deal with exceptional cases
      if (au == 0x3a052000u) return s * 0x1.52ffep-13f;  // |x| = 0x1.0a4p-11
      if (au == 0x3a318000u) return s * 0x1.c3fffep-13f; // |x| = 0x1.63p-11
      return s * t * fma (-0x1.cb4bcep-14f, t, 0x1.45f30ap-2f);
    }
    // deal with exceptional cases 
    if (au == 0x3af4c000u) return s * 0x1.37a002p-11f; // |x| = 0x1.e98p-10
    if (au == 0x3be78000u) return s * 0x1.26c004p-9f;  // |x| = 0x1.cfp-8
    if (au == 0x3b92a000u) return s * 0x1.756002p-10f; // |x| = 0x1.254p-8
    if (au == 0x3e1ea000u) return s * 0x1.90c002p-5f;  // |x| = 0x1.3d4p-3
    if (au == 0x3e4b2000u) return s * 0x1.fea002p-5f;  // |x| = 0x1.964p-3
    p = p0;
    c5 = fma (p[3], tt, p[2]);
    c1 = fma (p[1], tt, p[0]);
    c1 = fma (c5, tt * tt, c1);
    y = t * c1;
  }
  else {
    // use p1 for t <= 0.5, p2 for 0.5 < t <= 0.75, p3 for 0.75 < t <= 1
    if (au <= 0x3f000000u)
      p = p1;
    else if (au <= 0x3f400000u)
      p = p2;
    else
    {
      p = p3;
      // deal with exceptional cases
      if (au == 0x3f800000u) return s * 0x1p-2f;        // |x| = 1
      if (au == 0x3f712000u) return s * 0x1.ec7feap-3f; // |x| = 0x1.e24p-1
    }
    c3 = fma (p[4], t, p[3]);
    c2 = fma (c3, t, p[2]);
    y = fma (p[1], t, p[0]);
    y = fma (c2, tt, y);
  }

  if (reduce) y = HALF_PI - y; // argument reconstruction

  if (neg) y = -y;

  // divide by pi
  return y * INV_PI;
}
