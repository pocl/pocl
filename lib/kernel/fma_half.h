/* fma_half.h: a correctly rounded half fma, for x86 with Clang before 22.

   x86 has no native half arithmetic, and Clang before 22 lowers @llvm.fma.f16
   through a float fma: the result is rounded twice, to float and then to
   half, and is off by half an ulp for about one random input in 24,000
   (OpenCL-CTS math_fma, fp16). This computes in double instead, where the fma
   of half inputs is exact whenever the half result is finite, and narrows to
   float with rounding to odd: truncate, and set the last bit when inexact. A
   float has more than two bits beyond a half's precision, so the final
   rounding to half is then the correct one. (Clang 22 itself lowers the fma
   through double.)

   Used by fma.cl (vectorized builtins) and core-math/additionalf16.cl (the
   rest, conformance builds among them), which define the half overloads of
   fma from it where POCL_FMA_HALF_THROUGH_DOUBLE is defined.

   Copyright (c) 2026 PoCL developers. MIT license, see COPYING. */

#ifndef POCL_FMA_HALF_H
#define POCL_FMA_HALF_H

#if defined(cl_khr_fp16) && defined(cl_khr_fp64)                             \
    && (defined(__x86_64__) || defined(__i386__)) && __clang_major__ < 22

#define POCL_FMA_HALF_THROUGH_DOUBLE

static half
_cl_fma_half (half a, half b, half c)
{
  double r = __builtin_fma ((double)a, (double)b, (double)c);
  float f = (float)r;
  if (r == r && (double)f != r)
    {
      uint u = as_uint (f);
      if (__builtin_fabs ((double)f) > __builtin_fabs (r))
        u -= 1;
      f = as_float (u | 1u);
    }
  return (half)f;
}

#endif

#endif
