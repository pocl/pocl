//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#ifdef __SLEEF_CL_H__
#error You must include sleef_cl.h AFTER sleef.h
#endif

#ifndef __SLEEF_H__
#define __SLEEF_H__

#include <stddef.h>
#include <stdint.h>

#if (defined(__GNUC__) || defined(__CLANG__)) && !defined(__INTEL_COMPILER)
#define CONST const
#else
#define CONST
#endif

#if (defined(__GNUC__) || defined(__CLANG__))                                 \
    && (defined(__i386__) || defined(__x86_64__))
#include <x86intrin.h>
#endif

#if (defined(_MSC_VER))
#include <intrin.h>
#endif

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* Function/type attributes supported by Clang/SPIR */
#if __has_attribute(__always_inline__)
#define _CL_ALWAYSINLINE __attribute__ ((__always_inline__))
#else
#define _CL_ALWAYSINLINE
#endif
#if __has_attribute(__noinline__)
#define _CL_NOINLINE __attribute__ ((__noinline__))
#else
#define _CL_NOINLINE
#endif
#if __has_attribute(__overloadable__)
#define _CL_OVERLOADABLE __attribute__ ((__overloadable__))
#else
#define _CL_OVERLOADABLE
#endif
#if __has_attribute(__const__)
#define _CL_READNONE __attribute__ ((__const__))
#else
#define _CL_READNONE
#endif
#if __has_attribute(__pure__)
#define _CL_READONLY __attribute__ ((__pure__))
#else
#define _CL_READONLY
#endif
#if __has_attribute(__unavailable__)
#define _CL_UNAVAILABLE __attribute__ ((__unavailable__))
#else
#define _CL_UNAVAILABLE
#endif

#ifndef Sleef_double2_DEFINED
#define Sleef_double2_DEFINED
typedef struct
{
  double x, y;
} Sleef_double2;
#endif

#ifndef Sleef_float2_DEFINED
#define Sleef_float2_DEFINED
typedef struct
{
  float x, y;
} Sleef_float2;
#endif

double Sleef_sin_u35 (double);
double Sleef_cos_u35 (double);
Sleef_double2 Sleef_sincos_u35 (double);
double Sleef_tan_u35 (double);
double Sleef_asin_u35 (double);
double Sleef_acos_u35 (double);
double Sleef_atan_u35 (double);
double Sleef_atan2_u35 (double, double);
double Sleef_log_u35 (double);
double Sleef_cbrt_u35 (double);
double Sleef_sin_u10 (double);
double Sleef_cos_u10 (double);
Sleef_double2 Sleef_sincos_u10 (double);
double Sleef_tan_u10 (double);
double Sleef_asin_u10 (double);
double Sleef_acos_u10 (double);
double Sleef_atan_u10 (double);
double Sleef_atan2_u10 (double, double);
double Sleef_log_u10 (double);
double Sleef_cbrt_u10 (double);
double Sleef_exp_u10 (double);
double Sleef_pow_u10 (double, double);
double Sleef_sinh_u10 (double);
double Sleef_cosh_u10 (double);
double Sleef_tanh_u10 (double);
double Sleef_asinh_u10 (double);
double Sleef_acosh_u10 (double);
double Sleef_atanh_u10 (double);
double Sleef_exp2_u10 (double);
double Sleef_exp10_u10 (double);
double Sleef_expm1_u10 (double);
double Sleef_log10_u10 (double);
double Sleef_log1p_u10 (double);
Sleef_double2 Sleef_sincospi_u05 (double);
Sleef_double2 Sleef_sincospi_u35 (double);
double Sleef_sinpi_u05 (double);
double Sleef_cospi_u05 (double);
double Sleef_ldexp (double, int);
int Sleef_ilogb (double);
double Sleef_fma (double, double, double);
double Sleef_sqrt_u05 (double);
double Sleef_hypot_u05 (double, double);
double Sleef_hypot_u35 (double, double);
double Sleef_fabs (double);
double Sleef_copysign (double, double);
double Sleef_fmax (double, double);
double Sleef_fmin (double, double);
double Sleef_fdim (double, double);
double Sleef_trunc (double);
double Sleef_floor (double);
double Sleef_ceil (double);
double Sleef_round (double);
double Sleef_rint (double);
double Sleef_nextafter (double, double);
double Sleef_frfrexp (double);
int Sleef_expfrexp (double);
double Sleef_fmod (double, double);
Sleef_double2 Sleef_modf (double);
double Sleef_lgamma_u10 (double);
double Sleef_tgamma_u10 (double);
double Sleef_erf_u10 (double);
double Sleef_erfc_u15 (double);

float Sleef_sinf_u35 (float);
float Sleef_cosf_u35 (float);
Sleef_float2 Sleef_sincosf_u35 (float);
float Sleef_tanf_u35 (float);
float Sleef_asinf_u35 (float);
float Sleef_acosf_u35 (float);
float Sleef_atanf_u35 (float);
float Sleef_atan2f_u35 (float, float);
float Sleef_logf_u35 (float);
float Sleef_cbrtf_u35 (float);
float Sleef_sinf_u10 (float);
float Sleef_cosf_u10 (float);
Sleef_float2 Sleef_sincosf_u10 (float);
float Sleef_tanf_u10 (float);
float Sleef_asinf_u10 (float);
float Sleef_acosf_u10 (float);
float Sleef_atanf_u10 (float);
float Sleef_atan2f_u10 (float, float);
float Sleef_logf_u10 (float);
float Sleef_cbrtf_u10 (float);
float Sleef_expf_u10 (float);
float Sleef_powf_u10 (float, float);
float Sleef_sinhf_u10 (float);
float Sleef_coshf_u10 (float);
float Sleef_tanhf_u10 (float);
float Sleef_asinhf_u10 (float);
float Sleef_acoshf_u10 (float);
float Sleef_atanhf_u10 (float);
float Sleef_exp2f_u10 (float);
float Sleef_exp10f_u10 (float);
float Sleef_expm1f_u10 (float);
float Sleef_log10f_u10 (float);
float Sleef_log1pf_u10 (float);
Sleef_float2 Sleef_sincospif_u05 (float);
Sleef_float2 Sleef_sincospif_u35 (float);
float Sleef_sinpif_u05 (float d);
float Sleef_cospif_u05 (float d);
float Sleef_ldexpf (float, int);
int Sleef_ilogbf (float);
float Sleef_fmaf (float, float, float);
float Sleef_sqrtf_u05 (float);
float Sleef_sqrtf_u35 (float);
float Sleef_hypotf_u05 (float, float);
float Sleef_hypotf_u35 (float, float);
float Sleef_fabsf (float);
float Sleef_copysignf (float, float);
float Sleef_fmaxf (float, float);
float Sleef_fminf (float, float);
float Sleef_fdimf (float, float);
float Sleef_truncf (float);
float Sleef_floorf (float);
float Sleef_ceilf (float);
float Sleef_roundf (float);
float Sleef_rintf (float);
float Sleef_nextafterf (float, float);
float Sleef_frfrexpf (float);
int Sleef_expfrexpf (float);
float Sleef_fmodf (float, float);
Sleef_float2 Sleef_modff (float);
float Sleef_lgammaf_u10 (float);
float Sleef_tgammaf_u10 (float);
float Sleef_erff_u10 (float);
float Sleef_erfcf_u15 (float);

#ifdef __AVX512F__

#define SLEEF_VEC_512_AVAILABLE

typedef __m512 reg512f;
typedef __m512d reg512d;
typedef __m512i reg512i;

#ifndef Sleef___m512d_2_DEFINED
typedef struct
{
  __m512d x, y;
} Sleef___m512d_2;
#define Sleef___m512d_2_DEFINED
#endif
typedef Sleef___m512d_2 Sleef_reg512d_2;

__m512d Sleef_sind8_u35_intrin (__m512d);
__m512d Sleef_cosd8_u35_intrin (__m512d);
Sleef___m512d_2 Sleef_sincosd8_u35_intrin (__m512d);
__m512d Sleef_tand8_u35_intrin (__m512d);
__m512d Sleef_asind8_u35_intrin (__m512d);
__m512d Sleef_acosd8_u35_intrin (__m512d);
__m512d Sleef_atand8_u35_intrin (__m512d);
__m512d Sleef_atan2d8_u35_intrin (__m512d, __m512d);
__m512d Sleef_logd8_u35_intrin (__m512d);
__m512d Sleef_cbrtd8_u35_intrin (__m512d);
__m512d Sleef_sind8_u10_intrin (__m512d);
__m512d Sleef_cosd8_u10_intrin (__m512d);
Sleef___m512d_2 Sleef_sincosd8_u10_intrin (__m512d);
__m512d Sleef_tand8_u10_intrin (__m512d);
__m512d Sleef_asind8_u10_intrin (__m512d);
__m512d Sleef_acosd8_u10_intrin (__m512d);
__m512d Sleef_atand8_u10_intrin (__m512d);
__m512d Sleef_atan2d8_u10_intrin (__m512d, __m512d);
__m512d Sleef_logd8_u10_intrin (__m512d);
__m512d Sleef_cbrtd8_u10_intrin (__m512d);
__m512d Sleef_expd8_u10_intrin (__m512d);
__m512d Sleef_powd8_u10_intrin (__m512d, __m512d);
__m512d Sleef_sinhd8_u10_intrin (__m512d);
__m512d Sleef_coshd8_u10_intrin (__m512d);
__m512d Sleef_tanhd8_u10_intrin (__m512d);
__m512d Sleef_asinhd8_u10_intrin (__m512d);
__m512d Sleef_acoshd8_u10_intrin (__m512d);
__m512d Sleef_atanhd8_u10_intrin (__m512d);
__m512d Sleef_exp2d8_u10_intrin (__m512d);
__m512d Sleef_exp10d8_u10_intrin (__m512d);
__m512d Sleef_expm1d8_u10_intrin (__m512d);
__m512d Sleef_log10d8_u10_intrin (__m512d);
__m512d Sleef_log1pd8_u10_intrin (__m512d);
Sleef___m512d_2 Sleef_sincospid8_u05_intrin (__m512d);
Sleef___m512d_2 Sleef_sincospid8_u35_intrin (__m512d);
__m512d Sleef_sinpid8_u05_intrin (__m512d);
__m512d Sleef_cospid8_u05_intrin (__m512d);
__m512d Sleef_ldexpd8_intrin (__m512d, __m256i);
__m256i Sleef_ilogbd8_intrin (__m512d);
__m512d Sleef_fmad8_intrin (__m512d, __m512d, __m512d);
__m512d Sleef_sqrtd8_u05_intrin (__m512d);
__m512d Sleef_sqrtd8_u35_intrin (__m512d);
__m512d Sleef_hypotd8_u05_intrin (__m512d, __m512d);
__m512d Sleef_hypotd8_u35_intrin (__m512d, __m512d);
__m512d Sleef_fabsd8_intrin (__m512d);
__m512d Sleef_copysignd8_intrin (__m512d, __m512d);
__m512d Sleef_fmaxd8_intrin (__m512d, __m512d);
__m512d Sleef_fmind8_intrin (__m512d, __m512d);
__m512d Sleef_fdimd8_intrin (__m512d, __m512d);
__m512d Sleef_truncd8_intrin (__m512d);
__m512d Sleef_floord8_intrin (__m512d);
__m512d Sleef_ceild8_intrin (__m512d);
__m512d Sleef_roundd8_intrin (__m512d);
__m512d Sleef_rintd8_intrin (__m512d);
__m512d Sleef_nextafterd8_intrin (__m512d, __m512d);
__m512d Sleef_frfrexpd8_intrin (__m512d);
__m512i Sleef_expfrexpd8_intrin (__m512d);
__m512d Sleef_fmodd8_intrin (__m512d, __m512d);
Sleef___m512d_2 Sleef_modfd8_intrin (__m512d);
__m512d Sleef_lgammad8_u10_intrin (__m512d);
__m512d Sleef_tgammad8_u10_intrin (__m512d);
__m512d Sleef_erfd8_u10_intrin (__m512d);
__m512d Sleef_erfcd8_u15_intrin (__m512d);

#ifndef Sleef___m512_2_DEFINED
typedef struct
{
  __m512 x, y;
} Sleef___m512_2;
#define Sleef___m512_2_DEFINED
#endif
typedef Sleef___m512_2 Sleef_reg512f_2;

__m512 Sleef_sinf16_u35_intrin (__m512);
__m512 Sleef_cosf16_u35_intrin (__m512);
Sleef___m512_2 Sleef_sincosf16_u35_intrin (__m512);
__m512 Sleef_tanf16_u35_intrin (__m512);
__m512 Sleef_asinf16_u35_intrin (__m512);
__m512 Sleef_acosf16_u35_intrin (__m512);
__m512 Sleef_atanf16_u35_intrin (__m512);
__m512 Sleef_atan2f16_u35_intrin (__m512, __m512);
__m512 Sleef_logf16_u35_intrin (__m512);
__m512 Sleef_cbrtf16_u35_intrin (__m512);
__m512 Sleef_sinf16_u10_intrin (__m512);
__m512 Sleef_cosf16_u10_intrin (__m512);
Sleef___m512_2 Sleef_sincosf16_u10_intrin (__m512);
__m512 Sleef_tanf16_u10_intrin (__m512);
__m512 Sleef_asinf16_u10_intrin (__m512);
__m512 Sleef_acosf16_u10_intrin (__m512);
__m512 Sleef_atanf16_u10_intrin (__m512);
__m512 Sleef_atan2f16_u10_intrin (__m512, __m512);
__m512 Sleef_logf16_u10_intrin (__m512);
__m512 Sleef_cbrtf16_u10_intrin (__m512);
__m512 Sleef_expf16_u10_intrin (__m512);
__m512 Sleef_powf16_u10_intrin (__m512, __m512);
__m512 Sleef_sinhf16_u10_intrin (__m512);
__m512 Sleef_coshf16_u10_intrin (__m512);
__m512 Sleef_tanhf16_u10_intrin (__m512);
__m512 Sleef_asinhf16_u10_intrin (__m512);
__m512 Sleef_acoshf16_u10_intrin (__m512);
__m512 Sleef_atanhf16_u10_intrin (__m512);
__m512 Sleef_exp2f16_u10_intrin (__m512);
__m512 Sleef_exp10f16_u10_intrin (__m512);
__m512 Sleef_expm1f16_u10_intrin (__m512);
__m512 Sleef_log10f16_u10_intrin (__m512);
__m512 Sleef_log1pf16_u10_intrin (__m512);
Sleef___m512_2 Sleef_sincospif16_u05_intrin (__m512);
Sleef___m512_2 Sleef_sincospif16_u35_intrin (__m512);
__m512 Sleef_sinpif16_u05_intrin (__m512);
__m512 Sleef_cospif16_u05_intrin (__m512);
__m512 Sleef_ldexpf16_intrin (__m512, __m512i);
__m512i Sleef_ilogbf16_intrin (__m512);
__m512 Sleef_fmaf16_intrin (__m512, __m512, __m512);
__m512 Sleef_sqrtf16_u05_intrin (__m512);
__m512 Sleef_sqrtf16_u35_intrin (__m512);
__m512 Sleef_hypotf16_u05_intrin (__m512, __m512);
__m512 Sleef_hypotf16_u35_intrin (__m512, __m512);
__m512 Sleef_fabsf16_intrin (__m512);
__m512 Sleef_copysignf16_intrin (__m512, __m512);
__m512 Sleef_fmaxf16_intrin (__m512, __m512);
__m512 Sleef_fminf16_intrin (__m512, __m512);
__m512 Sleef_fdimf16_intrin (__m512, __m512);
__m512 Sleef_truncf16_intrin (__m512);
__m512 Sleef_floorf16_intrin (__m512);
__m512 Sleef_ceilf16_intrin (__m512);
__m512 Sleef_roundf16_intrin (__m512);
__m512 Sleef_rintf16_intrin (__m512);
__m512 Sleef_nextafterf16_intrin (__m512, __m512);
__m512 Sleef_frfrexpf16_intrin (__m512);
__m512i Sleef_expfrexpf16_intrin (__m512);
__m512 Sleef_fmodf16_intrin (__m512, __m512);
Sleef___m512_2 Sleef_modff16_intrin (__m512);
__m512 Sleef_lgammaf16_u10_intrin (__m512);
__m512 Sleef_tgammaf16_u10_intrin (__m512);
__m512 Sleef_erff16_u10_intrin (__m512);
__m512 Sleef_erfcf16_u15_intrin (__m512);
#endif

#if defined(__AVX2__) || defined(__AVX__)

#define SLEEF_VEC_256_AVAILABLE

/*
#ifndef __AVX2__

typedef struct
{
  __m128i x, y;
} __m256i;

#endif
*/

typedef __m256 reg256f;
typedef __m256d reg256d;
typedef __m256i reg256i;

#ifndef Sleef___m256d_2_DEFINED
typedef struct
{
  __m256d x, y;
} Sleef___m256d_2;
#define Sleef___m256d_2_DEFINED
#endif
typedef Sleef___m256d_2 Sleef_reg256d_2;

__m256d Sleef_sind4_u35_intrin (__m256d);
__m256d Sleef_cosd4_u35_intrin (__m256d);
Sleef___m256d_2 Sleef_sincosd4_u35_intrin (__m256d);
__m256d Sleef_tand4_u35_intrin (__m256d);
__m256d Sleef_asind4_u35_intrin (__m256d);
__m256d Sleef_acosd4_u35_intrin (__m256d);
__m256d Sleef_atand4_u35_intrin (__m256d);
__m256d Sleef_atan2d4_u35_intrin (__m256d, __m256d);
__m256d Sleef_logd4_u35_intrin (__m256d);
__m256d Sleef_cbrtd4_u35_intrin (__m256d);
__m256d Sleef_sind4_u10_intrin (__m256d);
__m256d Sleef_cosd4_u10_intrin (__m256d);
Sleef___m256d_2 Sleef_sincosd4_u10_intrin (__m256d);
__m256d Sleef_tand4_u10_intrin (__m256d);
__m256d Sleef_asind4_u10_intrin (__m256d);
__m256d Sleef_acosd4_u10_intrin (__m256d);
__m256d Sleef_atand4_u10_intrin (__m256d);
__m256d Sleef_atan2d4_u10_intrin (__m256d, __m256d);
__m256d Sleef_logd4_u10_intrin (__m256d);
__m256d Sleef_cbrtd4_u10_intrin (__m256d);
__m256d Sleef_expd4_u10_intrin (__m256d);
__m256d Sleef_powd4_u10_intrin (__m256d, __m256d);
__m256d Sleef_sinhd4_u10_intrin (__m256d);
__m256d Sleef_coshd4_u10_intrin (__m256d);
__m256d Sleef_tanhd4_u10_intrin (__m256d);
__m256d Sleef_asinhd4_u10_intrin (__m256d);
__m256d Sleef_acoshd4_u10_intrin (__m256d);
__m256d Sleef_atanhd4_u10_intrin (__m256d);
__m256d Sleef_exp2d4_u10_intrin (__m256d);
__m256d Sleef_exp10d4_u10_intrin (__m256d);
__m256d Sleef_expm1d4_u10_intrin (__m256d);
__m256d Sleef_log10d4_u10_intrin (__m256d);
__m256d Sleef_log1pd4_u10_intrin (__m256d);
Sleef___m256d_2 Sleef_sincospid4_u05_intrin (__m256d);
Sleef___m256d_2 Sleef_sincospid4_u35_intrin (__m256d);
__m256d Sleef_sinpid4_u05_intrin (__m256d);
__m256d Sleef_cospid4_u05_intrin (__m256d);
__m256d Sleef_ldexpd4_intrin (__m256d, __m128i);
__m128i Sleef_ilogbd4_intrin (__m256d);
__m256d Sleef_fmad4_intrin (__m256d, __m256d, __m256d);
__m256d Sleef_sqrtd4_u05_intrin (__m256d);
__m256d Sleef_sqrtd4_u35_intrin (__m256d);
__m256d Sleef_hypotd4_u05_intrin (__m256d, __m256d);
__m256d Sleef_hypotd4_u35_intrin (__m256d, __m256d);
__m256d Sleef_fabsd4_intrin (__m256d);
__m256d Sleef_copysignd4_intrin (__m256d, __m256d);
__m256d Sleef_fmaxd4_intrin (__m256d, __m256d);
__m256d Sleef_fmind4_intrin (__m256d, __m256d);
__m256d Sleef_fdimd4_intrin (__m256d, __m256d);
__m256d Sleef_truncd4_intrin (__m256d);
__m256d Sleef_floord4_intrin (__m256d);
__m256d Sleef_ceild4_intrin (__m256d);
__m256d Sleef_roundd4_intrin (__m256d);
__m256d Sleef_rintd4_intrin (__m256d);
__m256d Sleef_nextafterd4_intrin (__m256d, __m256d);
__m256d Sleef_frfrexpd4_intrin (__m256d);
__m256i Sleef_expfrexpd4_intrin (__m256d);
__m256d Sleef_fmodd4_intrin (__m256d, __m256d);
Sleef___m256d_2 Sleef_modfd4_intrin (__m256d);
__m256d Sleef_lgammad4_u10_intrin (__m256d);
__m256d Sleef_tgammad4_u10_intrin (__m256d);
__m256d Sleef_erfd4_u10_intrin (__m256d);
__m256d Sleef_erfcd4_u15_intrin (__m256d);

#ifndef Sleef___m256_2_DEFINED
typedef struct
{
  __m256 x, y;
} Sleef___m256_2;
#define Sleef___m256_2_DEFINED
#endif
typedef Sleef___m256_2 Sleef_reg256f_2;

__m256 Sleef_sinf8_u35_intrin (__m256);
__m256 Sleef_cosf8_u35_intrin (__m256);
Sleef___m256_2 Sleef_sincosf8_u35_intrin (__m256);
__m256 Sleef_tanf8_u35_intrin (__m256);
__m256 Sleef_asinf8_u35_intrin (__m256);
__m256 Sleef_acosf8_u35_intrin (__m256);
__m256 Sleef_atanf8_u35_intrin (__m256);
__m256 Sleef_atan2f8_u35_intrin (__m256, __m256);
__m256 Sleef_logf8_u35_intrin (__m256);
__m256 Sleef_cbrtf8_u35_intrin (__m256);
__m256 Sleef_sinf8_u10_intrin (__m256);
__m256 Sleef_cosf8_u10_intrin (__m256);
Sleef___m256_2 Sleef_sincosf8_u10_intrin (__m256);
__m256 Sleef_tanf8_u10_intrin (__m256);
__m256 Sleef_asinf8_u10_intrin (__m256);
__m256 Sleef_acosf8_u10_intrin (__m256);
__m256 Sleef_atanf8_u10_intrin (__m256);
__m256 Sleef_atan2f8_u10_intrin (__m256, __m256);
__m256 Sleef_logf8_u10_intrin (__m256);
__m256 Sleef_cbrtf8_u10_intrin (__m256);
__m256 Sleef_expf8_u10_intrin (__m256);
__m256 Sleef_powf8_u10_intrin (__m256, __m256);
__m256 Sleef_sinhf8_u10_intrin (__m256);
__m256 Sleef_coshf8_u10_intrin (__m256);
__m256 Sleef_tanhf8_u10_intrin (__m256);
__m256 Sleef_asinhf8_u10_intrin (__m256);
__m256 Sleef_acoshf8_u10_intrin (__m256);
__m256 Sleef_atanhf8_u10_intrin (__m256);
__m256 Sleef_exp2f8_u10_intrin (__m256);
__m256 Sleef_exp10f8_u10_intrin (__m256);
__m256 Sleef_expm1f8_u10_intrin (__m256);
__m256 Sleef_log10f8_u10_intrin (__m256);
__m256 Sleef_log1pf8_u10_intrin (__m256);
Sleef___m256_2 Sleef_sincospif8_u05_intrin (__m256);
Sleef___m256_2 Sleef_sincospif8_u35_intrin (__m256);
__m256 Sleef_sinpif8_u05_intrin (__m256);
__m256 Sleef_cospif8_u05_intrin (__m256);
__m256 Sleef_ldexpf8_intrin (__m256, __m256i);
__m256i Sleef_ilogbf8_intrin (__m256);
__m256 Sleef_fmaf8_intrin (__m256, __m256, __m256);
__m256 Sleef_sqrtf8_u05_intrin (__m256);
__m256 Sleef_sqrtf8_u35_intrin (__m256);
__m256 Sleef_hypotf8_u05_intrin (__m256, __m256);
__m256 Sleef_hypotf8_u35_intrin (__m256, __m256);
__m256 Sleef_fabsf8_intrin (__m256);
__m256 Sleef_copysignf8_intrin (__m256, __m256);
__m256 Sleef_fmaxf8_intrin (__m256, __m256);
__m256 Sleef_fminf8_intrin (__m256, __m256);
__m256 Sleef_fdimf8_intrin (__m256, __m256);
__m256 Sleef_truncf8_intrin (__m256);
__m256 Sleef_floorf8_intrin (__m256);
__m256 Sleef_ceilf8_intrin (__m256);
__m256 Sleef_roundf8_intrin (__m256);
__m256 Sleef_rintf8_intrin (__m256);
__m256 Sleef_nextafterf8_intrin (__m256, __m256);
__m256 Sleef_frfrexpf8_intrin (__m256);
__m256i Sleef_expfrexpf8_intrin (__m256);
__m256 Sleef_fmodf8_intrin (__m256, __m256);
Sleef___m256_2 Sleef_modff8_intrin (__m256);
__m256 Sleef_lgammaf8_u10_intrin (__m256);
__m256 Sleef_tgammaf8_u10_intrin (__m256);
__m256 Sleef_erff8_u10_intrin (__m256);
__m256 Sleef_erfcf8_u15_intrin (__m256);
#endif

#if defined(__SSE2__)

#define SLEEF_VEC_128_AVAILABLE

typedef __m128 reg128f;
typedef __m128d reg128d;
typedef __m128i reg128i;

#ifndef Sleef___m128d_2_DEFINED
typedef struct
{
  __m128d x, y;
} Sleef___m128d_2;
#define Sleef___m128d_2_DEFINED
#endif
typedef Sleef___m128d_2 Sleef_reg128d_2;

__m128d Sleef_sind2_u35_intrin (__m128d);
__m128d Sleef_cosd2_u35_intrin (__m128d);
Sleef___m128d_2 Sleef_sincosd2_u35_intrin (__m128d);
__m128d Sleef_tand2_u35_intrin (__m128d);
__m128d Sleef_asind2_u35_intrin (__m128d);
__m128d Sleef_acosd2_u35_intrin (__m128d);
__m128d Sleef_atand2_u35_intrin (__m128d);
__m128d Sleef_atan2d2_u35_intrin (__m128d, __m128d);
__m128d Sleef_logd2_u35_intrin (__m128d);
__m128d Sleef_cbrtd2_u35_intrin (__m128d);
__m128d Sleef_sind2_u10_intrin (__m128d);
__m128d Sleef_cosd2_u10_intrin (__m128d);
Sleef___m128d_2 Sleef_sincosd2_u10_intrin (__m128d);
__m128d Sleef_tand2_u10_intrin (__m128d);
__m128d Sleef_asind2_u10_intrin (__m128d);
__m128d Sleef_acosd2_u10_intrin (__m128d);
__m128d Sleef_atand2_u10_intrin (__m128d);
__m128d Sleef_atan2d2_u10_intrin (__m128d, __m128d);
__m128d Sleef_logd2_u10_intrin (__m128d);
__m128d Sleef_cbrtd2_u10_intrin (__m128d);
__m128d Sleef_expd2_u10_intrin (__m128d);
__m128d Sleef_powd2_u10_intrin (__m128d, __m128d);
__m128d Sleef_sinhd2_u10_intrin (__m128d);
__m128d Sleef_coshd2_u10_intrin (__m128d);
__m128d Sleef_tanhd2_u10_intrin (__m128d);
__m128d Sleef_asinhd2_u10_intrin (__m128d);
__m128d Sleef_acoshd2_u10_intrin (__m128d);
__m128d Sleef_atanhd2_u10_intrin (__m128d);
__m128d Sleef_exp2d2_u10_intrin (__m128d);
__m128d Sleef_exp10d2_u10_intrin (__m128d);
__m128d Sleef_expm1d2_u10_intrin (__m128d);
__m128d Sleef_log10d2_u10_intrin (__m128d);
__m128d Sleef_log1pd2_u10_intrin (__m128d);
Sleef___m128d_2 Sleef_sincospid2_u05_intrin (__m128d);
Sleef___m128d_2 Sleef_sincospid2_u35_intrin (__m128d);
__m128d Sleef_sinpid2_u05_intrin (__m128d);
__m128d Sleef_cospid2_u05_intrin (__m128d);
__m128d Sleef_ldexpd2_intrin (__m128d, __m128i);
__m128i Sleef_ilogbd2_intrin (__m128d);
__m128d Sleef_fmad2_intrin (__m128d, __m128d, __m128d);
__m128d Sleef_sqrtd2_u05_intrin (__m128d);
__m128d Sleef_sqrtd2_u35_intrin (__m128d);
__m128d Sleef_hypotd2_u05_intrin (__m128d, __m128d);
__m128d Sleef_hypotd2_u35_intrin (__m128d, __m128d);
__m128d Sleef_fabsd2_intrin (__m128d);
__m128d Sleef_copysignd2_intrin (__m128d, __m128d);
__m128d Sleef_fmaxd2_intrin (__m128d, __m128d);
__m128d Sleef_fmind2_intrin (__m128d, __m128d);
__m128d Sleef_fdimd2_intrin (__m128d, __m128d);
__m128d Sleef_truncd2_intrin (__m128d);
__m128d Sleef_floord2_intrin (__m128d);
__m128d Sleef_ceild2_intrin (__m128d);
__m128d Sleef_roundd2_intrin (__m128d);
__m128d Sleef_rintd2_intrin (__m128d);
__m128d Sleef_nextafterd2_intrin (__m128d, __m128d);
__m128d Sleef_frfrexpd2_intrin (__m128d);
__m128i Sleef_expfrexpd2_intrin (__m128d);
__m128d Sleef_fmodd2_intrin (__m128d, __m128d);
Sleef___m128d_2 Sleef_modfd2_intrin (__m128d);
__m128d Sleef_lgammad2_u10_intrin (__m128d);
__m128d Sleef_tgammad2_u10_intrin (__m128d);
__m128d Sleef_erfd2_u10_intrin (__m128d);
__m128d Sleef_erfcd2_u15_intrin (__m128d);

#ifndef Sleef___m128_2_DEFINED
typedef struct
{
  __m128 x, y;
} Sleef___m128_2;
#define Sleef___m128_2_DEFINED
#endif
typedef Sleef___m128_2 Sleef_reg128f_2;

__m128 Sleef_sinf4_u35_intrin (__m128);
__m128 Sleef_cosf4_u35_intrin (__m128);
Sleef___m128_2 Sleef_sincosf4_u35_intrin (__m128);
__m128 Sleef_tanf4_u35_intrin (__m128);
__m128 Sleef_asinf4_u35_intrin (__m128);
__m128 Sleef_acosf4_u35_intrin (__m128);
__m128 Sleef_atanf4_u35_intrin (__m128);
__m128 Sleef_atan2f4_u35_intrin (__m128, __m128);
__m128 Sleef_logf4_u35_intrin (__m128);
__m128 Sleef_cbrtf4_u35_intrin (__m128);
__m128 Sleef_sinf4_u10_intrin (__m128);
__m128 Sleef_cosf4_u10_intrin (__m128);
Sleef___m128_2 Sleef_sincosf4_u10_intrin (__m128);
__m128 Sleef_tanf4_u10_intrin (__m128);
__m128 Sleef_asinf4_u10_intrin (__m128);
__m128 Sleef_acosf4_u10_intrin (__m128);
__m128 Sleef_atanf4_u10_intrin (__m128);
__m128 Sleef_atan2f4_u10_intrin (__m128, __m128);
__m128 Sleef_logf4_u10_intrin (__m128);
__m128 Sleef_cbrtf4_u10_intrin (__m128);
__m128 Sleef_expf4_u10_intrin (__m128);
__m128 Sleef_powf4_u10_intrin (__m128, __m128);
__m128 Sleef_sinhf4_u10_intrin (__m128);
__m128 Sleef_coshf4_u10_intrin (__m128);
__m128 Sleef_tanhf4_u10_intrin (__m128);
__m128 Sleef_asinhf4_u10_intrin (__m128);
__m128 Sleef_acoshf4_u10_intrin (__m128);
__m128 Sleef_atanhf4_u10_intrin (__m128);
__m128 Sleef_exp2f4_u10_intrin (__m128);
__m128 Sleef_exp10f4_u10_intrin (__m128);
__m128 Sleef_expm1f4_u10_intrin (__m128);
__m128 Sleef_log10f4_u10_intrin (__m128);
__m128 Sleef_log1pf4_u10_intrin (__m128);
Sleef___m128_2 Sleef_sincospif4_u05_intrin (__m128);
Sleef___m128_2 Sleef_sincospif4_u35_intrin (__m128);
__m128 Sleef_sinpif4_u05_intrin (__m128);
__m128 Sleef_cospif4_u05_intrin (__m128);
__m128 Sleef_ldexpf4_intrin (__m128, __m128i);
__m128i Sleef_ilogbf4_intrin (__m128);
__m128 Sleef_fmaf4_intrin (__m128, __m128, __m128);
__m128 Sleef_sqrtf4_u05_intrin (__m128);
__m128 Sleef_sqrtf4_u35_intrin (__m128);
__m128 Sleef_hypotf4_u05_intrin (__m128, __m128);
__m128 Sleef_hypotf4_u35_intrin (__m128, __m128);
__m128 Sleef_fabsf4_intrin (__m128);
__m128 Sleef_copysignf4_intrin (__m128, __m128);
__m128 Sleef_fmaxf4_intrin (__m128, __m128);
__m128 Sleef_fminf4_intrin (__m128, __m128);
__m128 Sleef_fdimf4_intrin (__m128, __m128);
__m128 Sleef_truncf4_intrin (__m128);
__m128 Sleef_floorf4_intrin (__m128);
__m128 Sleef_ceilf4_intrin (__m128);
__m128 Sleef_roundf4_intrin (__m128);
__m128 Sleef_rintf4_intrin (__m128);
__m128 Sleef_nextafterf4_intrin (__m128, __m128);
__m128 Sleef_frfrexpf4_intrin (__m128);
__m128i Sleef_expfrexpf4_intrin (__m128);
__m128 Sleef_fmodf4_intrin (__m128, __m128);
Sleef___m128_2 Sleef_modff4_intrin (__m128);
__m128 Sleef_lgammaf4_u10_intrin (__m128);
__m128 Sleef_tgammaf4_u10_intrin (__m128);
__m128 Sleef_erff4_u10_intrin (__m128);
__m128 Sleef_erfcf4_u15_intrin (__m128);
#endif

#ifdef __ARM_NEON

#define SLEEF_VEC_128_AVAILABLE

typedef float32x4_t reg128f;
typedef int32x4_t reg128i;

#ifndef Sleef_float32x4_t_2_DEFINED
typedef struct
{
  float32x4_t x, y;
} Sleef_float32x4_t_2;
#define Sleef_float32x4_t_2_DEFINED
#endif
typedef Sleef_float32x4_t_2 Sleef_reg128f_2;

float32x4_t Sleef_sinf4_u35_intrin (float32x4_t);
float32x4_t Sleef_cosf4_u35_intrin (float32x4_t);
Sleef_float32x4_t_2 Sleef_sincosf4_u35_intrin (float32x4_t);
float32x4_t Sleef_tanf4_u35_intrin (float32x4_t);
float32x4_t Sleef_asinf4_u35_intrin (float32x4_t);
float32x4_t Sleef_acosf4_u35_intrin (float32x4_t);
float32x4_t Sleef_atanf4_u35_intrin (float32x4_t);
float32x4_t Sleef_atan2f4_u35_intrin (float32x4_t, float32x4_t);
float32x4_t Sleef_logf4_u35_intrin (float32x4_t);
float32x4_t Sleef_cbrtf4_u35_intrin (float32x4_t);
float32x4_t Sleef_sinf4_u10_intrin (float32x4_t);
float32x4_t Sleef_cosf4_u10_intrin (float32x4_t);
Sleef_float32x4_t_2 Sleef_sincosf4_u10_intrin (float32x4_t);
float32x4_t Sleef_tanf4_u10_intrin (float32x4_t);
float32x4_t Sleef_asinf4_u10_intrin (float32x4_t);
float32x4_t Sleef_acosf4_u10_intrin (float32x4_t);
float32x4_t Sleef_atanf4_u10_intrin (float32x4_t);
float32x4_t Sleef_atan2f4_u10_intrin (float32x4_t, float32x4_t);
float32x4_t Sleef_logf4_u10_intrin (float32x4_t);
float32x4_t Sleef_cbrtf4_u10_intrin (float32x4_t);
float32x4_t Sleef_expf4_u10_intrin (float32x4_t);
float32x4_t Sleef_powf4_u10_intrin (float32x4_t, float32x4_t);
float32x4_t Sleef_sinhf4_u10_intrin (float32x4_t);
float32x4_t Sleef_coshf4_u10_intrin (float32x4_t);
float32x4_t Sleef_tanhf4_u10_intrin (float32x4_t);
float32x4_t Sleef_asinhf4_u10_intrin (float32x4_t);
float32x4_t Sleef_acoshf4_u10_intrin (float32x4_t);
float32x4_t Sleef_atanhf4_u10_intrin (float32x4_t);
float32x4_t Sleef_exp2f4_u10_intrin (float32x4_t);
float32x4_t Sleef_exp10f4_u10_intrin (float32x4_t);
float32x4_t Sleef_expm1f4_u10_intrin (float32x4_t);
float32x4_t Sleef_log10f4_u10_intrin (float32x4_t);
float32x4_t Sleef_log1pf4_u10_intrin (float32x4_t);
Sleef_float32x4_t_2 Sleef_sincospif4_u05_intrin (float32x4_t);
Sleef_float32x4_t_2 Sleef_sincospif4_u35_intrin (float32x4_t);
float32x4_t Sleef_sinpif4_u05_intrin (float32x4_t);
float32x4_t Sleef_cospif4_u05_intrin (float32x4_t);
float32x4_t Sleef_ldexpf4_intrin (float32x4_t, int32x4_t);
int32x4_t Sleef_ilogbf4_intrin (float32x4_t);

float32x4_t Sleef_fmaf4_intrin (float32x4_t, float32x4_t, float32x4_t);
float32x4_t Sleef_sqrtf4_u05_intrin (float32x4_t);
float32x4_t Sleef_sqrtf4_u35_intrin (float32x4_t);
float32x4_t Sleef_hypotf4_u05_intrin (float32x4_t, float32x4_t);
float32x4_t Sleef_hypotf4_u35_intrin (float32x4_t, float32x4_t);
float32x4_t Sleef_fabsf4_intrin (float32x4_t);
float32x4_t Sleef_copysignf4_intrin (float32x4_t, float32x4_t);
float32x4_t Sleef_fmaxf4_intrin (float32x4_t, float32x4_t);
float32x4_t Sleef_fminf4_intrin (float32x4_t, float32x4_t);
float32x4_t Sleef_fdimf4_intrin (float32x4_t, float32x4_t);
float32x4_t Sleef_truncf4_intrin (float32x4_t);
float32x4_t Sleef_floorf4_intrin (float32x4_t);
float32x4_t Sleef_ceilf4_intrin (float32x4_t);
float32x4_t Sleef_roundf4_intrin (float32x4_t);
float32x4_t Sleef_rintf4_intrin (float32x4_t);
float32x4_t Sleef_nextafterf4_intrin (float32x4_t, float32x4_t);
float32x4_t Sleef_frfrexpf4_intrin (float32x4_t);
int32x4_t Sleef_expfrexpf4_intrin (float32x4_t);

float32x4_t Sleef_fmodf4_intrin (float32x4_t, float32x4_t);
Sleef_float32x4_t_2 Sleef_modff4_intrin (float32x4_t);
float32x4_t Sleef_lgammaf4_u10_intrin (float32x4_t);
float32x4_t Sleef_tgammaf4_u10_intrin (float32x4_t);
float32x4_t Sleef_erff4_u10_intrin (float32x4_t);
float32x4_t Sleef_erfcf4_u15_intrin (float32x4_t);

#ifdef SLEEF_DOUBLE_VEC_AVAILABLE
typedef float64x2_t reg128d;

#ifndef Sleef_float64x2_t_2_DEFINED
typedef struct
{
  float64x2_t x, y;
} Sleef_float64x2_t_2;
#define Sleef_float64x2_t_2_DEFINED
#endif
typedef Sleef_float64x2_t_2 Sleef_reg128d_2;

float64x2_t Sleef_sind2_u35_intrin (float64x2_t);
float64x2_t Sleef_cosd2_u35_intrin (float64x2_t);
Sleef_float64x2_t_2 Sleef_sincosd2_u35_intrin (float64x2_t);
float64x2_t Sleef_tand2_u35_intrin (float64x2_t);
float64x2_t Sleef_asind2_u35_intrin (float64x2_t);
float64x2_t Sleef_acosd2_u35_intrin (float64x2_t);
float64x2_t Sleef_atand2_u35_intrin (float64x2_t);
float64x2_t Sleef_atan2d2_u35_intrin (float64x2_t, float64x2_t);
float64x2_t Sleef_logd2_u35_intrin (float64x2_t);
float64x2_t Sleef_cbrtd2_u35_intrin (float64x2_t);
float64x2_t Sleef_sind2_u10_intrin (float64x2_t);
float64x2_t Sleef_cosd2_u10_intrin (float64x2_t);
Sleef_float64x2_t_2 Sleef_sincosd2_u10_intrin (float64x2_t);
float64x2_t Sleef_tand2_u10_intrin (float64x2_t);
float64x2_t Sleef_asind2_u10_intrin (float64x2_t);
float64x2_t Sleef_acosd2_u10_intrin (float64x2_t);
float64x2_t Sleef_atand2_u10_intrin (float64x2_t);
float64x2_t Sleef_atan2d2_u10_intrin (float64x2_t, float64x2_t);
float64x2_t Sleef_logd2_u10_intrin (float64x2_t);
float64x2_t Sleef_cbrtd2_u10_intrin (float64x2_t);
float64x2_t Sleef_expd2_u10_intrin (float64x2_t);
float64x2_t Sleef_powd2_u10_intrin (float64x2_t, float64x2_t);
float64x2_t Sleef_sinhd2_u10_intrin (float64x2_t);
float64x2_t Sleef_coshd2_u10_intrin (float64x2_t);
float64x2_t Sleef_tanhd2_u10_intrin (float64x2_t);
float64x2_t Sleef_asinhd2_u10_intrin (float64x2_t);
float64x2_t Sleef_acoshd2_u10_intrin (float64x2_t);
float64x2_t Sleef_atanhd2_u10_intrin (float64x2_t);
float64x2_t Sleef_exp2d2_u10_intrin (float64x2_t);
float64x2_t Sleef_exp10d2_u10_intrin (float64x2_t);
float64x2_t Sleef_expm1d2_u10_intrin (float64x2_t);
float64x2_t Sleef_log10d2_u10_intrin (float64x2_t);
float64x2_t Sleef_log1pd2_u10_intrin (float64x2_t);
Sleef_float64x2_t_2 Sleef_sincospid2_u05_intrin (float64x2_t);
Sleef_float64x2_t_2 Sleef_sincospid2_u35_intrin (float64x2_t);
float64x2_t Sleef_sinpid2_u05_intrin (float64x2_t);
float64x2_t Sleef_cospid2_u05_intrin (float64x2_t);
float64x2_t Sleef_ldexpd2_intrin (float64x2_t, int32x4_t);
int32x4_t Sleef_ilogbd2_intrin (float64x2_t);

float64x2_t Sleef_fmad2_intrin (float64x2_t, float64x2_t, float64x2_t);
float64x2_t Sleef_sqrtd2_u05_intrin (float64x2_t);
float64x2_t Sleef_sqrtd2_u35_intrin (float64x2_t);
float64x2_t Sleef_hypotd2_u05_intrin (float64x2_t, float64x2_t);
float64x2_t Sleef_hypotd2_u35_intrin (float64x2_t, float64x2_t);
float64x2_t Sleef_fabsd2_intrin (float64x2_t);
float64x2_t Sleef_copysignd2_intrin (float64x2_t, float64x2_t);
float64x2_t Sleef_fmaxd2_intrin (float64x2_t, float64x2_t);
float64x2_t Sleef_fmind2_intrin (float64x2_t, float64x2_t);
float64x2_t Sleef_fdimd2_intrin (float64x2_t, float64x2_t);
float64x2_t Sleef_truncd2_intrin (float64x2_t);
float64x2_t Sleef_floord2_intrin (float64x2_t);
float64x2_t Sleef_ceild2_intrin (float64x2_t);
float64x2_t Sleef_roundd2_intrin (float64x2_t);
float64x2_t Sleef_rintd2_intrin (float64x2_t);
float64x2_t Sleef_nextafterd2_intrin (float64x2_t, float64x2_t);
float64x2_t Sleef_frfrexpd2_intrin (float64x2_t);
int32x4_t Sleef_expfrexpd2_intrin (float64x2_t);

float64x2_t Sleef_fmodd2_intrin (float64x2_t, float64x2_t);
Sleef_float64x2_t_2 Sleef_modfd2_intrin (float64x2_t);
float64x2_t Sleef_lgammad2_u10_intrin (float64x2_t);
float64x2_t Sleef_tgammad2_u10_intrin (float64x2_t);
float64x2_t Sleef_erfd2_u10_intrin (float64x2_t);
float64x2_t Sleef_erfcd2_u15_intrin (float64x2_t);

#endif

#endif


#endif // __SLEEF_H__
