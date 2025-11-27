/* Correctly-rounded half-revolution arc-cosine for binary16 value.

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

// the following polynomials were generated using Sollya (cf acos.sollya)

/* degree-4 minimax polynomial for acos(x) over [0,0.25], with relative error
   bounded by 2^-22.943, manually optimized to reduce the number of exceptions
*/
static constant float p0[] = {0x1.921fb4p0f, -0x1.fffb44p-1f, -0x1.25e6b8p-10f,
                           -0x1.3cc114p-3f, -0x1.a85b22p-5f};

/* degree-4 minimax polynomial for acos(x) over [0.25,0.5], with relative error
   bounded by 2^-20.789, manually optimized to reduce the number of exceptions
*/
static constant float p1[] = {0x1.91b67p0f, -0x1.f515cap-1f, -0x1.bd043ap-4f,
                           0x1.7e2d5ap-4f, -0x1.190806p-2f};

/* degree-4 minimax polynomial for acos(x)/sqrt(1-x) over [0.5,1],
   with relative error bounded by 2^-23.583 */
static constant float p2[] = {0x1.91fa1cp0f, -0x1.ae5c5ep-3f, 0x1.31640cp-4f,
                           -0x1.98038p-6f, 0x1.251b5p-8f};

_CL_OVERLOADABLE half acospi (half x)
{
  b32u32_u v = {.f = x};
  uint u = v.u;
  uint au = u & 0x7fffffffu;

  if (au >= 0x3f800000u) { // NaN, Inf, or |x| >= 1
    if (au == 0x3f800000u)
      return (u == 0x3f800000u) ? 0.0f16 : 1.0f;
    if ((au >> 23) == 0x3ff && ((au & 0x7fffff) != 0)) // qNaN or sNaN
      return x;
    return NAN;
  }

  float t = v.f, tt = t * t, c1, c3, y;

  if (u >> 31) // x < 0
    t = -t;

#define INV_PI 0x1.45f306p-2f

  if (au < 0x3e800000u) { // |x| < 0.25
    if (au == 0) return 0.5f; // would be inexact with the code below
    if (au == 0x3c55a000u) // |x| = 0x1.ab4p-7
      return (au == u) ? 0x1.fbbffcp-2f : 0x1.022002p-1f;
    if (au == 0x3d8a2000u) // |x| = 0x1.144p-4
      return (au == u) ? 0x1.ea0002p-2f : 0x1.0afffep-1f;
    if (u == 0x3d21c000u) // x = 0x1.438p-5
      return 0x1.f31ffep-2f;
    if (u == 0x3db3a000u) // x = 0x1.674p-4
      return 0x1.e36002p-2f;
    c1 = fma (p0[2], t, p0[1]);
    c3 = fma (p0[4], t, p0[3]);
    y = fma (c3, tt, c1);
    y = fma (y, t, p0[0]);
    y = y * INV_PI;
    if (u >> 31) // x < 0
      y = 1.0f - y;
  }
  else if (au < 0x3f000000u) { // 0.25 <= |x| < 0.5
    c1 = fma (p1[2], t, p1[1]);
    c3 = fma (p1[4], t, p1[3]);
    y = fma (c3, tt, c1);
    y = fma (y, t, p1[0]);
    y = y * INV_PI;
    if (u >> 31) // x < 0
      y = 1.0f - y;
  }
  else { // 0.5 <= |x| <= 1
    if (au == 0x3f3ca000u) // |x| = 0x1.794p-1
      return (au == u) ? 0x1.e4000ap-3f : 0x1.86fffep-1f;
    c1 = fma (p2[2], t, p2[1]);
    c3 = fma (p2[4], t, p2[3]);
    y = fma (c3, tt, c1);
    y = fma (y, t, p2[0]);
    y = sqrt (1.0f - t) * y;
    y = y * INV_PI;
    if (u >> 31) // x < 0
      y = 1.0f - y;
  }

  return y;
}
