/* OpenCL built-in library: lgamma_r() for float, for the Metal (AIR) target

   Copyright (c) 2026 PoCL developers

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

/* lgamma_r(x, *signp): lgamma(x), and the sign of tgamma(x) in *signp.
   Gamma is positive for x > 0; for negative non-integers it is negative
   when floor(x) is odd. (The host library takes this from SLEEF.) */

static int
_pocl_gamma_sign (float x)
{
  if (x > 0.0f || isnan (x))
    return 1;
  if (x == 0.0f)
    return signbit (x) ? -1 : 1;
  float f = floor (x);
  if (f == x)
    return 1; /* pole */
  return (((int)f) & 1) ? -1 : 1;
}

#define IMPL_LGAMMA_R_AS(AS)                                                  \
  float _CL_OVERLOADABLE lgamma_r (float x, AS int *signp)                    \
  {                                                                           \
    *signp = _pocl_gamma_sign (x);                                            \
    return lgamma (x);                                                        \
  }                                                                           \
  IMPL_LGAMMA_R_V (2, AS)                                                     \
  IMPL_LGAMMA_R_V (3, AS)                                                     \
  IMPL_LGAMMA_R_V (4, AS)                                                     \
  IMPL_LGAMMA_R_V (8, AS)                                                     \
  IMPL_LGAMMA_R_V (16, AS)

#define IMPL_LGAMMA_R_V(N, AS)                                                \
  float##N _CL_OVERLOADABLE lgamma_r (float##N x, AS int##N *signp)           \
  {                                                                           \
    float##N r;                                                               \
    int##N s;                                                                 \
    for (int i = 0; i < N; ++i)                                               \
      {                                                                       \
        int si;                                                               \
        r[i] = lgamma_r (x[i], &si);                                          \
        s[i] = si;                                                            \
      }                                                                       \
    *signp = s;                                                               \
    return r;                                                                 \
  }

IMPL_LGAMMA_R_AS (__private)
IMPL_LGAMMA_R_AS (__global)
IMPL_LGAMMA_R_AS (__local)

#ifdef cl_khr_fp16
/* core-math/promotedf16.cl has the other half overloads */
half _CL_OVERLOADABLE
lgamma_r (half x, __private int *signp)
{
  int s;
  float r = lgamma_r ((float)x, &s);
  *signp = s;
  return (half)r;
}
#endif
