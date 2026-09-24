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
 * cover those overloads either. pow is there too: a vector library can deny
 * pow (SLEEF < 3.8, double), which keeps it on the libclc source and out of
 * the generic swap, so its half overloads must come from a file that is
 * compiled in both configurations. (sincos and lgamma_r half are provided by
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

/*
  llvm.minimumnum and llvm.maximumnum: Return the other argument if one is NaN.
  llvm.minnum and llvm.maxnum: For quiet NaNs behaves like
  minimumnum/maximumnum. For signaling NaNs, non-deterministically returns NaN
  or the other operand.

  OpenCL CTS test requires that fmax/fmin returns the other argument for both
  Quiet and Signalling NaNs. Therefore the non-deterministic behaviour of
  minnum/maxnum is not always suitable; it is failing on ARM64
*/
#if __has_builtin(__builtin_elementwise_maximumnum)                           \
  && __has_builtin(__builtin_elementwise_minimumnum)                          \
  && (defined(__arm__) || defined(__aarch64__))
DEFINE_FP16_BUILTIN_V_VV (fmax, __builtin_elementwise_maximumnum)
DEFINE_FP16_BUILTIN_V_VV (fmin, __builtin_elementwise_minimumnum)
#else
DEFINE_FP16_BUILTIN_V_VV (fmax, __builtin_elementwise_max)
DEFINE_FP16_BUILTIN_V_VV (fmin, __builtin_elementwise_min)
#endif
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

/* fmod -> @llvm.frem is a single LLVM intrinsic (no libm libcall), so it
   belongs with the Clang-builtin-backed overloads. frexp used to be here for
   the same reason; it moved to promotedf16.cl when frexp left the builtin
   swap, because that file is compiled in both configurations. */
half _CL_OVERLOADABLE fmod (half a, half b) { return (half)__builtin_fmodf ((float)a, (float)b); }
DEFINE_FP16_EXPR_V_VV (fmod)

