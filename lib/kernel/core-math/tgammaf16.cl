/* Correctly-rounded true gamma function for binary16 value.

Copyright (c) 2023-2025 Alexei Sibidanov and Paul Zimmermann

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

/* This code is based on the binary32 code tgammaf.c:
   at input we convert the inputs to half (exactly),
   we then use the same code than for tgammaf,
   and at output we round to half.
   The changes with respect to tgammaf.c are marked with a comment
   "specific tgammaf16". */

#include "common_types.h"

/* __builtin_roundeven was introduced in gcc 10:
   https://gcc.gnu.org/gcc-10/changes.html,
   and in clang 17 */
#if ((defined(__GNUC__) && __GNUC__ >= 10) || (defined(__clang__) && __clang_major__ >= 17)) && (defined(__aarch64__) || defined(__x86_64__) || defined(__i386__))
# define roundeven_finite(x) __builtin_roundeven (x)
#else
/* round x to nearest integer, breaking ties to even */
static double
roundeven_finite (double x)
{
  double ix;
# if (defined(__GNUC__) || defined(__clang__)) && (defined(__AVX__) || defined(__SSE4_1__) || (__ARM_ARCH >= 8))
#  if defined __AVX__
   __asm__("vroundsd $0x8,%1,%1,%0":"=x"(ix):"x"(x));
#  elif __ARM_ARCH >= 8
   __asm__ ("frintn %d0, %d1":"=w"(ix):"w"(x));
#  else /* __SSE4_1__ */
   __asm__("roundsd $0x8,%1,%0":"=x"(ix):"x"(x));
#  endif
# else
  ix = __builtin_round (x); /* nearest, away from 0 */
  if (fabs (ix - x) == 0.5)
  {
    /* if ix is odd, we should return ix-1 if x>0, and ix+1 if x<0 */
    union { double f; ulong n; } u, v;
    u.f = ix;
    v.f = ix - copysign (1.0, x);
    if (__builtin_ctz (v.n) > __builtin_ctz (u.n))
      ix = v.f;
  }
# endif
  return ix;
}
#endif


// specific tgammaf16: input renamed to xf16
_CL_OVERLOADABLE half tgamma(half xf16){
  float x = xf16;
  b32u32_u t = {.f = x};
  uint ax = t.u<<1;
  if(__builtin_expect(ax>=(0xffu<<24), 0)){ /* x=NaN or +/-Inf */
    if(ax==(0xffu<<24)){ /* x=+/-Inf */
      if(t.u>>31){ /* x=-Inf */
        return NAN;
      }
      return x; /* x=+Inf */
    }
    return NAN;
  }
  double z = x;
  // specific tgammaf16: changed threshold to deal with all overflow cases here
  if(__builtin_expect(ax<=0x6f000000u, 0)){ /* |x| <= 0x1p-16 */
    volatile double d = (0x1.fa658c23b1578p-1 - 0x1.d0a118f324b63p-1*z)*z - 0x1.2788cfc6fb619p-1;
    double f = 1.0/z + d;
    half r = f;
#ifdef CORE_MATH_SUPPORT_ERRNO
    /* tgamma(x) overflows for:
     * 0 <= x < 0x1p-16 whatever the rounding mode
     * x = 0x1p-16 and rounding to nearest or away from zero
       (in which case the result is +Inf)
     * -0x1p-16 <= x <= 0 whatever the rounding mode
     */
    if (x != 0x1p-16f || r > 0x1.fffffep+127f)
      errno = ERANGE; // overflow
#endif
    return r;
  }
  float fx = __builtin_floorf(x);
  if(__builtin_expect(x >= 0x1.274p+3f, 0)){
#ifdef CORE_MATH_SUPPORT_ERRNO
    /* The C standard says that if the function overflows,
       errno is set to ERANGE. */
    errno = ERANGE;
#endif
    return 0x1p127f * 0x1p127f;
  }
  int k = fx; // specific tgammaf16: no overflow possible
  if(__builtin_expect(fx==x, 0)){ /* x is integer */
    if(x == 0.0f){
#ifdef CORE_MATH_SUPPORT_ERRNO
      errno = ERANGE;
#endif
      return 1.0f/x;
    }
    if(x < 0.0f) {
#ifdef CORE_MATH_SUPPORT_ERRNO
      errno = EDOM;
#endif
      return NAN; /* should raise the "Invalid operation" exception */
    }
    double t0 = 1, x0 = 1;
    for(int i=1; i<k; i++, x0 += 1.0) t0 *= x0;
    return t0;
  }
  // specific tgammaf16: changed threshold of -42 to -16
  if(__builtin_expect(x<-16.0f, 0)){ /* negative non-integer */
    /* For x < -16, x non-integer, |gamma(x)| < 2^-25.  */
    static constant float sgn[2] = {0x1p-127f, -0x1p-127f};
#ifdef CORE_MATH_SUPPORT_ERRNO
    /* The C standard says that if the function underflows,
       errno is set to ERANGE. */
    errno = ERANGE; // underflow
#endif
    return 0x1p-127f * sgn[k&1];
  }
  static constant double c[] =
    {0x1.c9a76be577123p+0, 0x1.8f2754ddcf90dp+0, 0x1.0d1191949419bp+0, 0x1.e1f42cf0ae4a1p-2,
     0x1.82b358a3ab638p-3, 0x1.e1f2b30cd907bp-5, 0x1.240f6d4071bd8p-6, 0x1.1522c9f3cd012p-8,
     0x1.1fd0051a0525bp-10, 0x1.9808a8b96c37ep-13, 0x1.b3f78e01152b5p-15, 0x1.49c85a7e1fd04p-18,
     0x1.471ca49184475p-19, -0x1.368f0b7ed9e36p-23, 0x1.882222f9049efp-23, -0x1.a69ed2042842cp-25};

  double m = z - 0x1.7p+1, i = roundeven_finite(m), step = copysign(1.0,i);
  double d = m - i, d2 = d*d, d4 = d2*d2, d8 = d4*d4;
  double f = (c[0] + d*c[1]) + d2*(c[2] + d*c[3]) + d4*((c[4] + d*c[5]) + d2*(c[6] + d*c[7]))
    + d8*((c[8] + d*c[9]) + d2*(c[10] + d*c[11]) + d4*((c[12] + d*c[13]) + d2*(c[14] + d*c[15])));
  int jm = fabs(i);
  double w = 1;
  if(jm){
    z -= 0.5 + step*0.5;
    w = z;
    for(int j=jm-1; j; j--) {z -= step; w *= z;}
  }
  if(i<=-0.5) w = 1/w;
  f *= w;
  half r = f;
#ifdef CORE_MATH_SUPPORT_ERRNO
  if (fabs (f) < 0x1p-14f)
    errno = ERANGE; // underflow
#endif
  return r;
}
