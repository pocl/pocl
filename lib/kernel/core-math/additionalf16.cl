#include "common_types.h"

/* FP16 overloads for math builtins with a Clang/LLVM FP16 lowering: the
   __builtin_elementwise_* intrinsics (scalar + @llvm.<op>.vNf16, so they
   vectorize without SLEEF) plus fmod and frexp, which map to single LLVM
   intrinsics.
 *
 * These supplement the kernel library only in the non-vectorized configuration
 * (ENABLE_HOST_CPU_VECTORIZE_BUILTINS=OFF). When vectorization is ON the generic
 * <fn>.cl files provide these same FP16 overloads via __builtin_<fn>f16, so this
 * file is excluded then (see lib/kernel/host/CMakeLists.txt) to avoid multiple
 * definitions.
 *
 * Functions without a native FP16 builtin (logb, ilogb, ldexp, rootn, pown,
 * remainder, nextafter, powr, modf, remquo) are in promotedf16.cl. That file is
 * compiled in both configurations because the vectorized generic path does not
 * cover those overloads either. (sincos and lgamma_r half are provided by
 * core-math/sincosf16.cl and lgammaf16.cl.) */

#undef isfinite
#undef isnormal
#undef isinf
#undef isnan

DEFINE_FP16_BUILTIN_FPCLASS (isfinite, 504)
DEFINE_FP16_BUILTIN_FPCLASS (isnormal, 264)
DEFINE_FP16_BUILTIN_FPCLASS (isinf, 516)
DEFINE_FP16_BUILTIN_FPCLASS (isnan, 3)

DEFINE_FP16_BUILTIN_V_V (sqrt, __builtin_elementwise_sqrt)
DEFINE_FP16_BUILTIN_V_V (ceil, __builtin_elementwise_ceil)
DEFINE_FP16_BUILTIN_V_V (floor, __builtin_elementwise_floor)
DEFINE_FP16_BUILTIN_V_V (trunc, __builtin_elementwise_trunc)
DEFINE_FP16_BUILTIN_V_V (rint, __builtin_elementwise_rint)
DEFINE_FP16_BUILTIN_V_V (round, __builtin_elementwise_round)
DEFINE_FP16_BUILTIN_V_V (fabs, __builtin_elementwise_abs)

/* __builtin_elementwise_max/min lower to @llvm.maxnum/minnum, matching the
   OpenCL fmax/fmin NaN-quieting semantics. */
DEFINE_FP16_BUILTIN_V_VV (fmax, __builtin_elementwise_max)
DEFINE_FP16_BUILTIN_V_VV (fmin, __builtin_elementwise_min)
DEFINE_FP16_BUILTIN_V_VVV (fma, __builtin_elementwise_fma)

/* fdim(x, y) = (x > y) ? x - y : +0, returning NaN if either input is NaN.
   No single builtin exists; build it from fmax (defined above), keeping the
   isnan guards because fmax quiets NaNs. This file undefines isnan above so it
   can define _cl_isnan, so call the prefixed overload directly. (maxmag/minmag
   then resolve.) */
#define IMPLEMENT_FP16_FDIM(TYPE)                                             \
  TYPE _CL_OVERLOADABLE fdim (TYPE a, TYPE b)                                 \
  {                                                                           \
    return _cl_isnan (a) ? a : (_cl_isnan (b) ? b : fmax (a - b, (TYPE)0));   \
  }

IMPLEMENT_FP16_FDIM (half)
IMPLEMENT_FP16_FDIM (half2)
IMPLEMENT_FP16_FDIM (half3)
IMPLEMENT_FP16_FDIM (half4)
IMPLEMENT_FP16_FDIM (half8)
IMPLEMENT_FP16_FDIM (half16)

/* fmod -> @llvm.frem and frexp -> @llvm.frexp are single LLVM intrinsics (no
   libm libcall), so they belong with the Clang-builtin-backed overloads. */
half _CL_OVERLOADABLE fmod (half a, half b) { return (half)__builtin_fmodf ((float)a, (float)b); }
DEFINE_FP16_EXPR_V_VV (fmod)

half _CL_OVERLOADABLE
frexp (half x, private int *e) { return (half)__builtin_frexpf ((float)x, e); }
#define IMPLEMENT_FP16_FREXP_AS(AS)                                          \
  half _CL_OVERLOADABLE frexp (half x, AS int *e)                            \
  { int t; half r = frexp (x, &t); *e = t; return r; }
IMPLEMENT_FP16_FREXP_AS (local)
IMPLEMENT_FP16_FREXP_AS (global)
#ifdef __opencl_c_generic_address_space
IMPLEMENT_FP16_FREXP_AS (generic)
#endif
