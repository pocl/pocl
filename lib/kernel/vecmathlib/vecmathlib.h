// -*-C++-*-

#ifndef VECMATHLIB_H
#define VECMATHLIB_H

#if defined VML_DEBUG || defined VML_NODEBUG
#if defined VML_DEBUG && defined VML_NODEBUG
#error "Only one of VML_DEBUG or VML_NODEBUG may be defined"
#endif
#else
// default
#define VML_DEBUG
#endif

// FP settings

// Possible effects of not having VML_HAVE_FP_CONTRACT:
// - can re-associate
// - can replace division by reciprocal
// - (can break ties differently when rounding) no, this seems too invasive
// - can evaluate functions with reduced precision (80% of significant digits)

// default settings
#undef VML_HAVE_DENORMALS // TODO
#define VML_HAVE_FP_CONTRACT
#define VML_HAVE_INF
#define VML_HAVE_NAN
#define VML_HAVE_SIGNED_ZERO

// optimized settings
#ifdef __FAST_MATH__
#undef VML_HAVE_DENORMALS
#undef VML_HAVE_FP_CONTRACT
#undef VML_HAVE_INF
#undef VML_HAVE_NAN
#endif

#ifdef VML_DEBUG
#define VML_CONFIG_DEBUG " debug"
#else
#define VML_CONFIG_DEBUG " no-debug"
#endif
#ifdef VML_DENORMALS
#define VML_CONFIG_DENORMALS " denormals"
#else
#define VML_CONFIG_DENORMALS " no-denormals"
#endif
#ifdef VML_FP_CONTRACT
#define VML_CONFIG_FP_CONTRACT " fp-contract"
#else
#define VML_CONFIG_FP_CONTRACT " no-fp-contract"
#endif
#ifdef VML_INF
#define VML_CONFIG_INF " inf"
#else
#define VML_CONFIG_INF " no-inf"
#endif
#ifdef VML_NAN
#define VML_CONFIG_NAN " nan"
#else
#define VML_CONFIG_NAN " no-nan"
#endif

// TODO: introduce mad, as fast version of fma (check FP_FAST_FMA)
// TODO: introduce ieee_isnan and friends
// TODO: switch between isnan and ieee_isnan at an outside level

// This workaround is needed for older libstdc++ versions such as the
// one in Debian 6.0 when compiled with clang++
// <http://lists.cs.uiuc.edu/pipermail/cfe-dev/2011-February/013207.html>.
// The version time stamp used below is the one in Debian 6.0.
#include <cstring> // pull in __GLIBCXX__
#if defined __GLIBCXX__ && __GLIBCXX__ <= 20101114
namespace std {
class type_info;
}
#endif

#include <cassert>

#ifdef VML_DEBUG
#define VML_ASSERT(x) assert(x)
#else
#define VML_ASSERT(x) ((void)0)
#endif

// Scalarise all vector operations, and use libm's functions (mostly
// useful as fallback)
#include "vec_pseudo.h"

#ifdef __clang__
// Use compiler-provided vector types
#include "vec_builtin.h"
#endif

// Scalarise all vector operations; don't use libm, use only
// Vecmathlib's functions (mostly useful for testing Vecmathlib)
#include "vec_test.h"

#if defined __ARM_NEON__ // ARM NEON
#include "vec_neon_float2.h"
#include "vec_neon_float4.h"
#define VML_CONFIG_NEON " NEON"
#else
#define VML_CONFIG_NEON
#endif

#if defined __SSE2__ // Intel SSE 2
#include "vec_sse_float1.h"
#include "vec_sse_float4.h"
#include "vec_sse_double1.h"
#include "vec_sse_double2.h"
#if defined __SSE3__
#define VML_CONFIG_SSE3 " SSE3"
#else
#define VML_CONFIG_SSE3
#endif
#if defined __SSSE3__
#define VML_CONFIG_SSSE3 " SSSE3"
#else
#define VML_CONFIG_SSSE3
#endif
#if defined __SSE4_1__
#define VML_CONFIG_SSE4_1 " SSE4.1"
#else
#define VML_CONFIG_SSE4_1
#endif
#if defined __SSE4a__
#define VML_CONFIG_SSE4a " SSE4a"
#else
#define VML_CONFIG_SSE4a
#endif
#define VML_CONFIG_SSE2                                                        \
  " SSE2" VML_CONFIG_SSE3 VML_CONFIG_SSSE3 VML_CONFIG_SSE4_1 VML_CONFIG_SSE4a
#else
#define VML_CONFIG_SSE2
#endif

#if defined __AVX__ // Intel AVX
#include "vec_avx_fp8_32.h"
#include "vec_avx_fp16_16.h"
#include "vec_avx_float8.h"
#include "vec_avx_double4.h"
#define VML_CONFIG_AVX " AVX"
#else
#define VML_CONFIG_AVX
#endif

#if defined __MIC__ // Intel MIC
// TODO: single precision?
#include "vec_mic_double8.h"
#define VML_CONFIG_MIC " MIC"
#else
#define VML_CONFIG_MIC
#endif

#if defined __ALTIVEC__ // IBM Altivec
#include "vec_altivec_float4.h"
#define VML_CONFIG_ALTIVEC " Altivec"
#else
#define VML_CONFIG_ALTIVEC
#endif
#if defined __ALTIVEC__ && defined _ARCH_PWR7 // IBM VSX
#include "vec_vsx_double2.h"
#define VML_CONFIG_VSX " VSX"
#else
#define VML_CONFIG_VSX
#endif

// TODO: IBM Blue Gene/P DoubleHummer

#if defined __bgq__ && defined __VECTOR4DOUBLE__ // IBM Blue Gene/Q QPX
// TODO: vec_qpx_float4
#include "vec_qpx_double4.h"
#define VML_CONFIG_QPX " QPX"
#else
#define VML_CONFIG_QPX
#endif

#define VECMATHLIB_CONFIGURATION                                               \
  "VecmathlibConfiguration" VML_CONFIG_DEBUG VML_CONFIG_DENORMALS              \
      VML_CONFIG_FP_CONTRACT VML_CONFIG_INF VML_CONFIG_NAN VML_CONFIG_NEON     \
          VML_CONFIG_SSE2 VML_CONFIG_AVX VML_CONFIG_MIC VML_CONFIG_ALTIVEC     \
              VML_CONFIG_VSX VML_CONFIG_QPX

// Define "best" vector types
namespace vecmathlib {

#if defined VECMATHLIB_HAVE_VEC_FLOAT_16
#define VECMATHLIB_MAX_FLOAT_VECSIZE 16
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_8
#define VECMATHLIB_MAX_FLOAT_VECSIZE 8
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
#define VECMATHLIB_MAX_FLOAT_VECSIZE 4
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_2
#define VECMATHLIB_MAX_FLOAT_VECSIZE 2
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_1
#define VECMATHLIB_MAX_FLOAT_VECSIZE 1
#endif

#if defined VECMATHLIB_HAVE_VEC_DOUBLE_8
#define VECMATHLIB_MAX_DOUBLE_VECSIZE 8
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_4
#define VECMATHLIB_MAX_DOUBLE_VECSIZE 4
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_2
#define VECMATHLIB_MAX_DOUBLE_VECSIZE 2
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_1
#define VECMATHLIB_MAX_DOUBLE_VECSIZE 1
#endif

#ifdef VECMATHLIB_MAX_FLOAT_VECSIZE
typedef realvec<float, VECMATHLIB_MAX_FLOAT_VECSIZE> float32_vec;
typedef intvec<float, VECMATHLIB_MAX_FLOAT_VECSIZE> int32_vec;
typedef boolvec<float, VECMATHLIB_MAX_FLOAT_VECSIZE> bool32_vec;
#else
typedef realpseudovec<float, 1> float32_vec;
typedef intpseudovec<float, 1> int32_vec;
typedef boolpseudovec<float, 1> bool32_vec;
#endif

#ifdef VECMATHLIB_MAX_DOUBLE_VECSIZE
typedef realvec<double, VECMATHLIB_MAX_DOUBLE_VECSIZE> float64_vec;
typedef intvec<double, VECMATHLIB_MAX_DOUBLE_VECSIZE> int64_vec;
typedef boolvec<double, VECMATHLIB_MAX_DOUBLE_VECSIZE> bool64_vec;
#else
typedef realpseudovec<double, 1> float64_vec;
typedef intpseudovec<double, 1> int64_vec;
typedef boolpseudovec<double, 1> bool64_vec;
#endif
}

#endif // #ifndef VECMATHLIB_H
