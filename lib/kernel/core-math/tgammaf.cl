/* Correctly-rounded true gamma function for binary32 value.

Copyright (c) 2023-2025 Alexei Sibidanov.

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

/* OpenCL C port of CORE-MATH's src/binary32/tgamma/tgammaf.c (master
   284b3b0), in the form of tgammaf16.cl: the same code with C's types,
   includes and FENV pragma replaced by OpenCL C's, and its tables in the
   constant address space. */

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
    if (v.n == 0 || __builtin_ctzl (v.n) > __builtin_ctzl (u.n))
      ix = v.f;
  }
# endif
  return ix;
}
#endif

_CL_OVERLOADABLE float tgamma(float x){
  /* List of exceptional cases.  */
  static constant struct {b32u32_u x; float f, df;} tb[] = {
    {{.u = 0x27de86a9u}, 0x1.268266p+47f, 0x1p22f},
    {{.u = 0x27e05475u}, 0x1.242422p+47f, 0x1p22f},
    {{.u = 0xb63befb3u}, -0x1.5cb6e4p+18f, 0x1p-7f},
    {{.u = 0x3c7bb570u}, 0x1.021d9p+6f, 0x1p-19f},
    {{.u = 0x41e886d1u}, 0x1.33136ap+98f, 0x1p73f},
    {{.u = 0xc067d177u}, 0x1.f6850cp-3f, 0x1p-28f},
    {{.f = -0x1.33b462p-4}, -0x1.befe66p+3, -0x1p-22f},
    {{.f = -0x1.a988b4p-1}, -0x1.a6b4ecp+2, +0x1p-23f},
    {{.f = 0x1.dceffcp+4}, 0x1.d3631cp+101, -0x1p-76f},
    {{.f = 0x1.0874c8p+0}, 0x1.f6c638p-1, 0x1p-26f},
  };

  b32u32_u t = {.f = x};
  uint ax = t.u<<1;
  if(__builtin_expect(ax>=(0xffu<<24), 0)){ /* x=NaN or +/-Inf */
    if(ax==(0xffu<<24)){ /* x=+/-Inf */
      if(t.u>>31){ /* x=-Inf */
#ifdef CORE_MATH_SUPPORT_ERRNO
	errno = EDOM;
#endif
	return x / x; /* will raise the "Invalid operation" exception */
      }
      return x; /* x=+Inf */
    }
    return x + x; /* x=NaN, where x+x ensures the "Invalid operation"
                     exception is set if x is sNaN, and it yields a qNaN */
  }
  double z = x;
  if(__builtin_expect(ax<0x6d000000u, 0)){ /* |x| < 0x1p-18 */
    volatile double d = (0x1.fa658c23b1578p-1 - 0x1.d0a118f324b63p-1*z)*z - 0x1.2788cfc6fb619p-1;
    double f = 1.0/z + d;
    float r = f;
#ifdef CORE_MATH_SUPPORT_ERRNO
    /* tgamma(x) overflows for:
     * 0 <= x < 0x1p-128 whatever the rounding mode
     * x = 0x1p-128 and rounding to nearest or away from zero
       (in which case the result is +Inf)
     * -0x1p-128 <= x <= 0 whatever the rounding mode
     */
    if (__builtin_fabsf (x) < 0x1p-128f ||
        (x == 0x1p-128f && r > 0x1.fffffep+127f) || x == -0x1p-128f)
      errno = ERANGE; // overflow
#endif
    b64u64_u rt = {.f = f};
    if(((rt.u+2)&0xfffffff) < 4){
      for(uint i=0;i<sizeof(tb)/sizeof(tb[0]);i++)
	if(t.u==tb[i].x.u) return tb[i].f + tb[i].df;
    }
    return r;
  }
  float fx = __builtin_floorf(x);
  if(__builtin_expect(x >= 0x1.18522p+5f, 0)){
#ifdef CORE_MATH_SUPPORT_ERRNO
    /* The C standard says that if the function overflows,
       errno is set to ERANGE. */
    errno = ERANGE;
#endif
    return 0x1p127f * 0x1p127f;
  }
  /* compute k only after the overflow check, otherwise the cast to integer
     might overflow */
  int k;
  if(__builtin_expect(x <= -0x1p+31f, 0)) {
    k = INT_MIN;
  }
  else {
    k = fx;
  }
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
      return 0.0f / 0.0f; /* should raise the "Invalid operation" exception */
    }
    double t0 = 1, x0 = 1;
    for(int i=1; i<k; i++, x0 += 1.0) t0 *= x0;
    return t0;
  }
  if(__builtin_expect(x<-42.0f, 0)){ /* negative non-integer */
    /* For x < -42, x non-integer, |gamma(x)| < 2^-151.  */
    static constant float sgn[2] = {0x1p-127f, -0x1p-127f};
#ifdef CORE_MATH_SUPPORT_ERRNO
    /* The C standard says that if the function underflows,
       errno is set to ERANGE. */
    errno = ERANGE;
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
  b64u64_u rt = {.f = f};
  float r = f;
#ifdef CORE_MATH_SUPPORT_ERRNO
  if (__builtin_fabsf (r) < 0x1p-126f)
    errno = ERANGE; // underflow
#endif
  /* Deal with exceptional cases.  */
  if(__builtin_expect(((rt.u+2)&0xfffffff) < 8, 0)){
    for(uint j=0;j<sizeof(tb)/sizeof(tb[0]);j++) {
      if(t.u==tb[j].x.u) return tb[j].f + tb[j].df;
    }
  }
  return r;
}

DEFINE_FP32_EXPR_V_V(tgamma)
