/* Correctly-rounded hyperbolic arc-sine for binary16 value.

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

/* The following polynomials were generated using Sollya (cf asinh.sollya).
   P[0] is a degree-7 polynomial with odd-degree coefficients only
   approximating asinh(x) over [0,2^-2], and for 1 <= i < 19,
   P[i] is a degree-6 polynomial approximating asinh(x) over [2^(i-3),2^(i-2)).
   They were afterwards optimized to decrease the number of exceptions. */

static constant float P[19][7] = {
  {0x1.000002p+0, -0x1.5554d6p-3, 0x1.32c4a4p-4, -0x1.5336f2p-5}, // [0,2^-2]
  {0x1.707d7ap-17, 0x1.ffe5ap-1, 0x1.86a284p-10, -0x1.608874p-3, 0x1.1d36dp-7, 0x1.435f08p-4, -0x1.2861d6p-5}, // [2^-2,2^-1)
  {0x1.2c296cp-10, 0x1.f9d57p-1, 0x1.b8aab8p-5, -0x1.30b6a4p-2, 0x1.73af8ap-3, -0x1.a1564ep-5, 0x1.6908p-8}, // [2^-1,2^0)
  {-0x1.171628p-6, 0x1.1382e6p+0, -0x1.f8ac18p-4, -0x1.c6795cp-4, 0x1.2d2dfep-4, -0x1.38f626p-6, 0x1.f9d716p-10}, // [2^0,2^1)
  {-0x1.29739cp-4, 0x1.43decap+0, -0x1.91761cp-2, 0x1.82261ap-4, -0x1.eee226p-7, 0x1.746546p-10, -0x1.f02e6ap-15}, // [2^1,2^2)
  {0x1.6722aep-3, 0x1.d6b2e4p-1, -0x1.7dbe34p-3, 0x1.c10cbep-6, -0x1.502cc2p-9, 0x1.1e2eb2p-13, -0x1.a56434p-19}, // [2^2,2^3)
  {0x1.6f7ecp-1, 0x1.05fe1p-1, -0x1.c290eap-5, 0x1.12d0bep-8, -0x1.a5b282p-13, 0x1.6d57cp-18, -0x1.108528p-24}, // [2^3,2^4)
  {0x1.5ed2bep+0, 0x1.0d3bp-2, -0x1.d55ab8p-7, 0x1.20902cp-11, -0x1.bd0e6ap-17, 0x1.82e9d2p-23, -0x1.214f5cp-30}, // [2^4,2^5)
  {0x1.06cee4p+1, 0x1.0f17c2p-3, -0x1.da366ep-9, 0x1.24229ap-14, -0x1.c327d6p-21, 0x1.889372p-28, -0x1.25be4cp-36}, // [2^5,2^6)
  {0x1.5f39aep+1, 0x1.0f8076p-4, -0x1.db39e6p-11, 0x1.24d726p-17, -0x1.c44a96p-25, 0x1.899114p-33, -0x1.26784ap-42}, // [2^6,2^7)
  {0x1.b7e21p+1, 0x1.0f9492p-5, -0x1.db65a6p-13, 0x1.24f0f8p-20, -0x1.c46c0ep-29, 0x1.89a692p-38, -0x1.2681d4p-48}, // [2^7,2^8)
  {0x1.084d12p+2, 0x1.0f923p-6, -0x1.db5616p-15, 0x1.24df02p-23, -0x1.c4422p-33, 0x1.89751cp-43, -0x1.2653cp-54}, // [2^8,2^9)
  {0x1.34afcep+2, 0x1.0f76f6p-7, -0x1.daf3c8p-17, 0x1.24818cp-26, -0x1.c37d7ep-37, 0x1.889b7ep-48, -0x1.258ca6p-60}, // [2^9,2^10)
  {0x1.610b6ep+2, 0x1.0f7ad8p-8, -0x1.db022p-19, 0x1.248fa8p-29, -0x1.c39c7p-41, 0x1.88bf6p-53, -0x1.25aeeep-66}, // [2^10,2^11)
  {0x1.8d6a98p+2, 0x1.0f6f2ep-9, -0x1.dad874p-21, 0x1.24689p-32, -0x1.c34ab6p-45, 0x1.886532p-58, -0x1.255d5ep-72}, // [2^11,2^12)
  {0x1.b9c986p+2, 0x1.0f651cp-10, -0x1.dab52cp-23, 0x1.2447e4p-35, -0x1.c3079ep-49, 0x1.881ca6p-63, -0x1.251c46p-78}, // [2^12,2^13)
  {0x1.e6269ap+2, 0x1.0f62aep-11, -0x1.daac6ep-25, 0x1.243f9ep-38, -0x1.c2f63cp-53, 0x1.88096cp-68, -0x1.250ac6p-84}, // [2^13,2^14)
  {0x1.09425p+3, 0x1.0f5c6ap-12, -0x1.da96ep-27, 0x1.242c12p-41, -0x1.c2ceccp-57, 0x1.87df72p-73, -0x1.24e5eep-90}, // [2^14,2^15)
  {0x1.1f707ep+3, 0x1.0f5d12p-13, -0x1.da990ap-29, 0x1.242df2p-44, -0x1.c2d272p-61, 0x1.87e32ep-78, -0x1.24e916p-96}, // [2^15,2^16)
};


_CL_OVERLOADABLE half asinh (half x)
{
  b32u32_u v = {.f = x};
  uint u = v.u;
  uint au = u & 0x7fffffffu;

  if (au >= 0x7f800000u) { // NaN or Inf
    // asinh(+Inf) = +Inf, otherwise we get qNaN
    if (u == 0x7f800000u || (au & 0x7fffff)) // +Inf or NaN
      return x + x;
    return x; // -Inf
  }

  int i = (au >> 23) - 124; // 2^(i-3) <= x < 2^(i-2)
  float t = v.f, tt = t * t, y;
  constant float *p;
  if (i <= 0) { // |x| < 2^-2
    /* For |x| <= 0x1.714p-5, asinh(x) rounds to x to nearest,
       we deal with that case separately, so that for x subnormal
       and a power of two, we get an underflow. */
    if (au <= 0x3d38a000u) {
      if (au == 0) return x;
      return (au == u) ? v.f - 0x1p-26f : v.f + 0x1p-26f;
    }
    p = P[0];
    float c5 = fma (p[3], tt, p[2]);
    float c1 = fma (p[1], tt, p[0]);
    c1 = fma (c5, tt * tt, c1);
    y = t * c1;
  }
  else { // |x| >= 2^-2
    static constant float s[2] = {1.0, -1.0};
    /* we make t positive since the polynomials for |x| >= 2^-2 are not odd,
       thus only work for x > 0 */
    t = t * s[u>>31];
    p = P[i];
    float c4 = fma (p[5], t, p[4]);
    float c2 = fma (p[3], t, p[2]);
    float c0 = fma (p[1], t, p[0]);
    c4 = fma (p[6], tt, c4);
    c0 = fma (c2, tt, c0);
    y = fma (c4, tt * tt, c0);

    // deal with exceptions
    switch (i) { // i is the exponent of x, plus 3
    case 7:
      if (au == 0x41936000u) y = 0x1.cdbffep+1f; // |x|=0x1.26cp+4
      break;
    case 10:
      if (au == 0x436cc000u) y = 0x1.8a4002p+2f; // |x|=0x1.d98p+7
      break;
    case 11:
      if (au == 0x43f10000u) y = 0x1.b7bffap+2f; // |x|=0x1.e2p+8
      break;
    case 12:
      if (au == 0x44588000u) y = 0x1.dd4004p+2f; // |x|=0x1.b1p+9
      break;
    }

    y = y * s[u>>31]; // restore sign
  }
  
  return y;
}
