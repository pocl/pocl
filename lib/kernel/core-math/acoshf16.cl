/* Correctly-rounded hyperbolic arc-cosine for binary16 value.

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

/* The following polynomials were generated using Sollya (cf acosh.sollya).
   For 0 <= i < 16, P[i] is a degree-5 polynomial approximating
   acosh(x)/sqrt(x-1) for 0 <= i < 5, and acosh(x) for 5 <= i < 16.
   They were afterwards optimized to decrease the number of exceptions. */

static constant float P[16][6] = {
  {0x1.91940ap+0, -0x1.a1ee52p-3, 0x1.0dc64ap-4, -0x1.453108p-6, 0x1.040a46p-8, -0x1.803198p-12}, /* [2^0,2^1) */
  {0x1.8eb5fap+0, -0x1.6c52a2p-3, 0x1.4dddcp-5, -0x1.e9e95cp-8, 0x1.b697b2p-11, -0x1.5b2df4p-15}, /* [2^1,2^2) */
  {0x1.854f5ap+0, -0x1.11e128p-3, 0x1.362606p-6, -0x1.fb23bp-10, 0x1.e2d03p-14, -0x1.8d21eap-19}, /* [2^2,2^3) */
  {0x1.717322p+0, -0x1.608d74p-4, 0x1.c3ceb2p-8, -0x1.8859d4p-12, 0x1.828e08p-17, -0x1.44eab2p-23}, /* [2^3,2^4) */
  {0x1.52b1b4p+0, -0x1.8e60acp-5, 0x1.12e502p-9, -0x1.ee8866p-15, 0x1.f10d92p-21, -0x1.a71f3ep-28}, /* [2^4,2^5) */
  {0x1.1b4176p+1, 0x1.c58fdp-4, -0x1.3d491cp-9, 0x1.244e58p-15, -0x1.2b59e6p-22, 0x1.029682p-30}, /* [2^5,2^6) */
  {0x1.744536p+1, 0x1.c4d402p-5, -0x1.3c787ap-11, 0x1.235b58p-18, -0x1.2a375cp-26, 0x1.017ceap-35}, /* [2^6,2^7) */ // 4 exceptions
  {0x1.cd16eap+1, 0x1.c490cp-6, -0x1.3c27dep-13, 0x1.22f70cp-21, -0x1.29b90ep-30, 0x1.00fd26p-40}, /* [2^7,2^8) */ // 1 exception
  {0x1.12ec74p+2, 0x1.c475e4p-7, -0x1.3c05dap-15, 0x1.22cb2cp-24, -0x1.29804ap-34, 0x1.00c29ap-45}, /* [2^8,2^9) */ // 6 exceptions
  {0x1.3f4c7ep+2, 0x1.c45ddap-8, -0x1.3be5p-17, 0x1.229e72p-27, -0x1.2943fp-38, 0x1.008224p-50}, /* [2^9,2^10) */ // // 1 exception
  {0x1.6baa5p+2, 0x1.c45576p-9, -0x1.3bd988p-19, 0x1.228f6ep-30, -0x1.29308ep-42, 0x1.006ddp-55}, /* [2^10,2^11) */ // no exception
  {0x1.980882p+2, 0x1.c449a8p-10, -0x1.3bc944p-21, 0x1.2278ep-33, -0x1.2911bap-46, 0x1.004d3p-60}, /* [2^11,2^12) */ // no exception
  {0x1.c46664p+2, 0x1.c44006p-11, -0x1.3bbbdp-23, 0x1.22665ep-36, -0x1.28f8a4p-50, 0x1.00326p-65}, /* [2^12,2^13) */ // 1 exception
  {0x1.f0c4d6p+2, 0x1.c43246p-12, -0x1.3ba86ap-25, 0x1.224b62p-39, -0x1.28d39ep-54, 0x1.000a46p-70}, /* [2^13,2^14) */ // 1 exception
  {0x1.0e912cp+3, 0x1.c42b4ep-13, -0x1.3b9eb4p-27, 0x1.223e1p-42, -0x1.28c1ap-58, 0x1.ffee3cp-76}, /* [2^14,2^15) */ // no exception
  {0x1.24bfccp+3, 0x1.c4263p-14, -0x1.3b97b8p-29, 0x1.22347ap-45, -0x1.28b4c4p-62, 0x1.ffd2ecp-81}, /* [2^15,2^16) */ // no exception
};


_CL_OVERLOADABLE half acosh (half x)
{
  b32u32_u v = {.f = x};
  uint u = v.u;
  uint au = u & 0x7fffffffu;

  if (au >= 0x7f800000u) { // NaN or Inf
    // acosh(+Inf) = +Inf, otherwise we get qNaN
    if (u == 0x7f800000u || (au & 0x7fffff)) // +Inf or NaN
      return x + x;
    return NAN;
  }

  if (u >= 0x80000000 || au <= 0x3f800000) { // x <= 1
    if (u == 0x3f800000) // x = 1
      return +0.0f; // for x=1 the code below yields -0
    // for x=1 the code below yields -0
    return NAN;
  }

  int i = (au >> 23) - 127; // 2^i <= x < 2^(i+1)
  float t = v.f, tt = t * t;
  constant float *p = P[i];
  float c4 = fma (p[5], t, p[4]);
  float c2 = fma (p[3], t, p[2]);
  c2 = fma (c4, tt, c2);
  float c0 = fma (p[1], t, p[0]);
  float y = fma (c2, tt, c0);
  if (i < 5)
    y = sqrt (t - 1.0f) * y;
  
  // deal with exceptions
  switch (i) {
  case 1:
    if (u == 0x40422000) y = 0x1.c63ffcp+0f; // x = 0x1.844p+1
    break;
  case 3:
    if (u == 0x411c0000) y = 0x1.7be006p+1f; // x = 0x1.38p+3
    if (u == 0x415f6000) y = 0x1.aa0002p+1f; // x = 0x1.becp+3
    break;
  case 4 :
    if (u == 0x41bfe000) y = 0x1.ef5fecp+1f; // x = 0x1.7fcp+4
    if (u == 0x41c04000) y = 0x1.ef9ff6p+1f; // x = 0x1.808p+4
    if (u == 0x41dbc000) y = 0x1.006016p+2f; // x = 0x1.b78p+4
    if (u == 0x41d3c000) y = 0x1.fc002ap+1f; // x = 0x1.a78p+4
    break;
  case 5:
    if (u == 0x42374000) y = 0x1.21201cp+2f; // x = 0x1.6e8p+5
    if (u == 0x4245c000) y = 0x1.260012p+2f; // x = 0x1.8b8p+5
    break;
  case 6:
    if (u == 0x42890000) y = 0x1.3ae018p+2f; // x = 0x1.12p+6
    if (u == 0x429d6000) y = 0x1.43bff4p+2f; // x = 0x1.3acp+6
    break;
  case 7:
    if (u == 0x433d2000) y = 0x1.7be006p+2f; // x = 0x1.7a4p+7
    break;
  case 8:
    if (u == 0x43884000) y = 0x1.934004p+2f; // x = 0x1.108p+8
    if (u == 0x43f9a000) y = 0x1.ba000ep+2f; // x = 0x1.f34p+8
    break;
  case 9:
    if (u == 0x443b0000) y = 0x1.d3e00cp+2f; // x = 0x1.76p+9
    break;
  case 12:
    if (u == 0x45b78000) y = 0x1.2be008p+3f; // x = 0x1.6fp+12
    break;
  case 13:
    if (u == 0x4673a000) y = 0x1.4b2008p+3f; // x = 0x1.e74p+13
  }

  return y;
}
