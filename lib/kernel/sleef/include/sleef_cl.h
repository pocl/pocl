/* OpenCL built-in library: SLEEF OpenCL prototypes

   Copyright (c) 2017 Michal Babej / Tampere University of Technology

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#ifndef __SLEEF_CL_H__
#define __SLEEF_CL_H__

#ifndef __OPENCL_VERSION__

typedef int32_t int2 __attribute__ ((__ext_vector_type__ (2)));
typedef int32_t int3 __attribute__ ((__ext_vector_type__ (3)));
typedef int32_t int4 __attribute__ ((__ext_vector_type__ (4)));
typedef int32_t int8 __attribute__ ((__ext_vector_type__ (8)));
typedef int32_t int16 __attribute__ ((__ext_vector_type__ (16)));

typedef uint32_t uint;
typedef uint32_t uint2 __attribute__ ((__ext_vector_type__ (2)));
typedef uint32_t uint3 __attribute__ ((__ext_vector_type__ (3)));
typedef uint32_t uint4 __attribute__ ((__ext_vector_type__ (4)));
typedef uint32_t uint8 __attribute__ ((__ext_vector_type__ (8)));
typedef uint32_t uint16 __attribute__ ((__ext_vector_type__ (16)));

typedef int64_t long2 __attribute__ ((__ext_vector_type__ (2)));
typedef int64_t long3 __attribute__ ((__ext_vector_type__ (3)));
typedef int64_t long4 __attribute__ ((__ext_vector_type__ (4)));
typedef int64_t long8 __attribute__ ((__ext_vector_type__ (8)));
typedef int64_t long16 __attribute__ ((__ext_vector_type__ (16)));

typedef uint64_t ulong;

typedef uint64_t ulong2 __attribute__ ((__ext_vector_type__ (2)));
typedef uint64_t ulong3 __attribute__ ((__ext_vector_type__ (3)));
typedef uint64_t ulong4 __attribute__ ((__ext_vector_type__ (4)));
typedef uint64_t ulong8 __attribute__ ((__ext_vector_type__ (8)));
typedef uint64_t ulong16 __attribute__ ((__ext_vector_type__ (16)));

typedef float float2 __attribute__ ((__ext_vector_type__ (2)));
typedef float float3 __attribute__ ((__ext_vector_type__ (3)));
typedef float float4 __attribute__ ((__ext_vector_type__ (4)));
typedef float float8 __attribute__ ((__ext_vector_type__ (8)));
typedef float float16 __attribute__ ((__ext_vector_type__ (16)));

#ifdef SLEEF_DOUBLE_AVAILABLE
typedef double double2 __attribute__ ((__ext_vector_type__ (2)));
typedef double double3 __attribute__ ((__ext_vector_type__ (3)));
typedef double double4 __attribute__ ((__ext_vector_type__ (4)));
typedef double double8 __attribute__ ((__ext_vector_type__ (8)));
typedef double double16 __attribute__ ((__ext_vector_type__ (16)));

#define cl_khr_fp64

#endif

#endif

#ifdef cl_khr_fp64

#ifndef Sleef_double2_DEFINED
#define Sleef_double2_DEFINED
typedef struct
{
  double x, y;
} Sleef_double2;
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
double Sleef_log2_u10 (double);
double Sleef_log1p_u10 (double);
Sleef_double2 Sleef_sincospi_u05 (double);
Sleef_double2 Sleef_sincospi_u35 (double);
double Sleef_sinpi_u05 (double);
double Sleef_cospi_u05 (double);
double Sleef_ldexp (double, int);
int Sleef_ilogb (double);
double Sleef_fma (double, double, double);
double Sleef_sqrt (double);
double Sleef_sqrt_u05 (double);
double Sleef_sqrt_u35 (double);
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
Sleef_double2 Sleef_lgamma_r_u10 (double);
double Sleef_tgamma_u10 (double);
double Sleef_erf_u10 (double);
double Sleef_erfc_u15 (double);

double Sleef_pown_u10 (double, int);
double Sleef_powr_u10 (double, double);

#endif

#ifndef Sleef_float2_DEFINED
#define Sleef_float2_DEFINED
typedef struct
{
  float x, y;
} Sleef_float2;
#endif

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
float Sleef_log2f_u10 (float);
float Sleef_log1pf_u10 (float);
Sleef_float2 Sleef_sincospif_u05 (float);
Sleef_float2 Sleef_sincospif_u35 (float);
float Sleef_sinpif_u05 (float d);
float Sleef_cospif_u05 (float d);
float Sleef_ldexpf (float, int);
int Sleef_ilogbf (float);
float Sleef_fmaf (float, float, float);
float Sleef_sqrtf (float);
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
Sleef_float2 Sleef_lgamma_rf_u10 (float);
float Sleef_tgammaf_u10 (float);
float Sleef_erff_u10 (float);
float Sleef_erfcf_u15 (float);

float Sleef_pownf_u10 (float, int);
float Sleef_powrf_u10 (float, float);


// #####################

#ifdef SLEEF_VEC_512_AVAILABLE

#ifdef cl_khr_fp64

#ifndef Sleef_double8_2_DEFINED
typedef struct
{
  double8 x, y;
} Sleef_double8_2;
#define Sleef_double8_2_DEFINED
#endif

double8 Sleef_sind8_u35 (double8);
double8 Sleef_cosd8_u35 (double8);
Sleef_double8_2 Sleef_sincosd8_u35 (double8);
double8 Sleef_tand8_u35 (double8);
double8 Sleef_asind8_u35 (double8);
double8 Sleef_acosd8_u35 (double8);
double8 Sleef_atand8_u35 (double8);
double8 Sleef_atan2d8_u35 (double8, double8);
double8 Sleef_logd8_u35 (double8);
double8 Sleef_cbrtd8_u35 (double8);
double8 Sleef_sind8_u10 (double8);
double8 Sleef_cosd8_u10 (double8);
Sleef_double8_2 Sleef_sincosd8_u10 (double8);
double8 Sleef_tand8_u10 (double8);
double8 Sleef_asind8_u10 (double8);
double8 Sleef_acosd8_u10 (double8);
double8 Sleef_atand8_u10 (double8);
double8 Sleef_atan2d8_u10 (double8, double8);
double8 Sleef_logd8_u10 (double8);
double8 Sleef_cbrtd8_u10 (double8);
double8 Sleef_expd8_u10 (double8);
double8 Sleef_powd8_u10 (double8, double8);
double8 Sleef_sinhd8_u10 (double8);
double8 Sleef_coshd8_u10 (double8);
double8 Sleef_tanhd8_u10 (double8);
double8 Sleef_asinhd8_u10 (double8);
double8 Sleef_acoshd8_u10 (double8);
double8 Sleef_atanhd8_u10 (double8);
double8 Sleef_exp2d8_u10 (double8);
double8 Sleef_exp10d8_u10 (double8);
double8 Sleef_expm1d8_u10 (double8);
double8 Sleef_log10d8_u10 (double8);
double8 Sleef_log2d8_u10 (double8);
double8 Sleef_log1pd8_u10 (double8);
Sleef_double8_2 Sleef_sincospid8_u05 (double8);
Sleef_double8_2 Sleef_sincospid8_u35 (double8);
double8 Sleef_sinpid8_u05 (double8);
double8 Sleef_cospid8_u05 (double8);
double8 Sleef_ldexpd8 (double8, int8);
int8 Sleef_ilogbd8 (double8);
double8 Sleef_fmad8 (double8, double8, double8);
double8 Sleef_sqrtd8 (double8);
double8 Sleef_sqrtd8_u05 (double8);
double8 Sleef_sqrtd8_u35 (double8);
double8 Sleef_hypotd8_u05 (double8, double8);
double8 Sleef_hypotd8_u35 (double8, double8);
double8 Sleef_fabsd8 (double8);
double8 Sleef_copysignd8 (double8, double8);
double8 Sleef_fmaxd8 (double8, double8);
double8 Sleef_fmind8 (double8, double8);
double8 Sleef_fdimd8 (double8, double8);
double8 Sleef_truncd8 (double8);
double8 Sleef_floord8 (double8);
double8 Sleef_ceild8 (double8);
double8 Sleef_roundd8 (double8);
double8 Sleef_rintd8 (double8);
double8 Sleef_nextafterd8 (double8, double8);
double8 Sleef_frfrexpd8 (double8);
int8 Sleef_expfrexpd8 (double8);
double8 Sleef_fmodd8 (double8, double8);
Sleef_double8_2 Sleef_modfd8 (double8);
double8 Sleef_lgammad8_u10 (double8);
Sleef_double8_2 Sleef_lgamma_rd8_u10 (double8);
double8 Sleef_tgammad8_u10 (double8);
double8 Sleef_erfd8_u10 (double8);
double8 Sleef_erfcd8_u15 (double8);

#endif

#ifndef Sleef_float16_2_DEFINED
typedef struct
{
  float16 x, y;
} Sleef_float16_2;
#define Sleef_float16_2_DEFINED
#endif

float16 Sleef_sinf16_u35 (float16);
float16 Sleef_cosf16_u35 (float16);
Sleef_float16_2 Sleef_sincosf16_u35 (float16);
float16 Sleef_tanf16_u35 (float16);
float16 Sleef_asinf16_u35 (float16);
float16 Sleef_acosf16_u35 (float16);
float16 Sleef_atanf16_u35 (float16);
float16 Sleef_atan2f16_u35 (float16, float16);
float16 Sleef_logf16_u35 (float16);
float16 Sleef_cbrtf16_u35 (float16);
float16 Sleef_sinf16_u10 (float16);
float16 Sleef_cosf16_u10 (float16);
Sleef_float16_2 Sleef_sincosf16_u10 (float16);
float16 Sleef_tanf16_u10 (float16);
float16 Sleef_asinf16_u10 (float16);
float16 Sleef_acosf16_u10 (float16);
float16 Sleef_atanf16_u10 (float16);
float16 Sleef_atan2f16_u10 (float16, float16);
float16 Sleef_logf16_u10 (float16);
float16 Sleef_cbrtf16_u10 (float16);
float16 Sleef_expf16_u10 (float16);
float16 Sleef_powf16_u10 (float16, float16);
float16 Sleef_sinhf16_u10 (float16);
float16 Sleef_coshf16_u10 (float16);
float16 Sleef_tanhf16_u10 (float16);
float16 Sleef_asinhf16_u10 (float16);
float16 Sleef_acoshf16_u10 (float16);
float16 Sleef_atanhf16_u10 (float16);
float16 Sleef_exp2f16_u10 (float16);
float16 Sleef_exp10f16_u10 (float16);
float16 Sleef_expm1f16_u10 (float16);
float16 Sleef_log10f16_u10 (float16);
float16 Sleef_log2f16_u10 (float16);
float16 Sleef_log1pf16_u10 (float16);
Sleef_float16_2 Sleef_sincospif16_u05 (float16);
Sleef_float16_2 Sleef_sincospif16_u35 (float16);
float16 Sleef_sinpif16_u05 (float16);
float16 Sleef_cospif16_u05 (float16);
float16 Sleef_ldexpf16 (float16, int16);
int16 Sleef_ilogbf16 (float16);
float16 Sleef_fmaf16 (float16, float16, float16);
float16 Sleef_sqrtf16 (float16);
float16 Sleef_sqrtf16_u05 (float16);
float16 Sleef_sqrtf16_u35 (float16);
float16 Sleef_hypotf16_u05 (float16, float16);
float16 Sleef_hypotf16_u35 (float16, float16);
float16 Sleef_fabsf16 (float16);
float16 Sleef_copysignf16 (float16, float16);
float16 Sleef_fmaxf16 (float16, float16);
float16 Sleef_fminf16 (float16, float16);
float16 Sleef_fdimf16 (float16, float16);
float16 Sleef_truncf16 (float16);
float16 Sleef_floorf16 (float16);
float16 Sleef_ceilf16 (float16);
float16 Sleef_roundf16 (float16);
float16 Sleef_rintf16 (float16);
float16 Sleef_nextafterf16 (float16, float16);
float16 Sleef_frfrexpf16 (float16);
int16 Sleef_expfrexpf16 (float16);
float16 Sleef_fmodf16 (float16, float16);
Sleef_float16_2 Sleef_modff16 (float16);
float16 Sleef_lgammaf16_u10 (float16);
Sleef_float16_2 Sleef_lgamma_rf16_u10 (float16);
float16 Sleef_tgammaf16_u10 (float16);
float16 Sleef_erff16_u10 (float16);
float16 Sleef_erfcf16_u15 (float16);


double8 Sleef_pownd8_u10 (double8, int8);
float16 Sleef_pownf16_u10 (float16, int16);
double8 Sleef_powrd8_u10 (double8, double8);
float16 Sleef_powrf16_u10 (float16, float16);

#endif

// #####################

#ifdef SLEEF_VEC_256_AVAILABLE

#ifdef cl_khr_fp64

#ifndef Sleef_double4_2_DEFINED
typedef struct
{
  double4 x, y;
} Sleef_double4_2;
#define Sleef_double4_2_DEFINED
#endif

double4 Sleef_sind4_u35 (double4);
double4 Sleef_cosd4_u35 (double4);
Sleef_double4_2 Sleef_sincosd4_u35 (double4);
double4 Sleef_tand4_u35 (double4);
double4 Sleef_asind4_u35 (double4);
double4 Sleef_acosd4_u35 (double4);
double4 Sleef_atand4_u35 (double4);
double4 Sleef_atan2d4_u35 (double4, double4);
double4 Sleef_logd4_u35 (double4);
double4 Sleef_cbrtd4_u35 (double4);
double4 Sleef_sind4_u10 (double4);
double4 Sleef_cosd4_u10 (double4);
Sleef_double4_2 Sleef_sincosd4_u10 (double4);
double4 Sleef_tand4_u10 (double4);
double4 Sleef_asind4_u10 (double4);
double4 Sleef_acosd4_u10 (double4);
double4 Sleef_atand4_u10 (double4);
double4 Sleef_atan2d4_u10 (double4, double4);
double4 Sleef_logd4_u10 (double4);
double4 Sleef_cbrtd4_u10 (double4);
double4 Sleef_expd4_u10 (double4);
double4 Sleef_powd4_u10 (double4, double4);
double4 Sleef_sinhd4_u10 (double4);
double4 Sleef_coshd4_u10 (double4);
double4 Sleef_tanhd4_u10 (double4);
double4 Sleef_asinhd4_u10 (double4);
double4 Sleef_acoshd4_u10 (double4);
double4 Sleef_atanhd4_u10 (double4);
double4 Sleef_exp2d4_u10 (double4);
double4 Sleef_exp10d4_u10 (double4);
double4 Sleef_expm1d4_u10 (double4);
double4 Sleef_log10d4_u10 (double4);
double4 Sleef_log2d4_u10 (double4);
double4 Sleef_log1pd4_u10 (double4);
Sleef_double4_2 Sleef_sincospid4_u05 (double4);
Sleef_double4_2 Sleef_sincospid4_u35 (double4);
double4 Sleef_sinpid4_u05 (double4);
double4 Sleef_cospid4_u05 (double4);
double4 Sleef_ldexpd4 (double4, int4);
int4 Sleef_ilogbd4 (double4);
double4 Sleef_fmad4 (double4, double4, double4);
double4 Sleef_sqrtd4 (double4);
double4 Sleef_sqrtd4_u05 (double4);
double4 Sleef_sqrtd4_u35 (double4);
double4 Sleef_hypotd4_u05 (double4, double4);
double4 Sleef_hypotd4_u35 (double4, double4);
double4 Sleef_fabsd4 (double4);
double4 Sleef_copysignd4 (double4, double4);
double4 Sleef_fmaxd4 (double4, double4);
double4 Sleef_fmind4 (double4, double4);
double4 Sleef_fdimd4 (double4, double4);
double4 Sleef_truncd4 (double4);
double4 Sleef_floord4 (double4);
double4 Sleef_ceild4 (double4);
double4 Sleef_roundd4 (double4);
double4 Sleef_rintd4 (double4);
double4 Sleef_nextafterd4 (double4, double4);
double4 Sleef_frfrexpd4 (double4);
int4 Sleef_expfrexpd4 (double4);
double4 Sleef_fmodd4 (double4, double4);
Sleef_double4_2 Sleef_modfd4 (double4);
double4 Sleef_lgammad4_u10 (double4);
Sleef_double4_2 Sleef_lgamma_rd4_u10 (double4);
double4 Sleef_tgammad4_u10 (double4);
double4 Sleef_erfd4_u10 (double4);
double4 Sleef_erfcd4_u15 (double4);

#endif

#ifndef Sleef_float8_2_DEFINED
typedef struct
{
  float8 x, y;
} Sleef_float8_2;
#define Sleef_float8_2_DEFINED
#endif

float8 Sleef_sinf8_u35 (float8);
float8 Sleef_cosf8_u35 (float8);
Sleef_float8_2 Sleef_sincosf8_u35 (float8);
float8 Sleef_tanf8_u35 (float8);
float8 Sleef_asinf8_u35 (float8);
float8 Sleef_acosf8_u35 (float8);
float8 Sleef_atanf8_u35 (float8);
float8 Sleef_atan2f8_u35 (float8, float8);
float8 Sleef_logf8_u35 (float8);
float8 Sleef_cbrtf8_u35 (float8);
float8 Sleef_sinf8_u10 (float8);
float8 Sleef_cosf8_u10 (float8);
Sleef_float8_2 Sleef_sincosf8_u10 (float8);
float8 Sleef_tanf8_u10 (float8);
float8 Sleef_asinf8_u10 (float8);
float8 Sleef_acosf8_u10 (float8);
float8 Sleef_atanf8_u10 (float8);
float8 Sleef_atan2f8_u10 (float8, float8);
float8 Sleef_logf8_u10 (float8);
float8 Sleef_cbrtf8_u10 (float8);
float8 Sleef_expf8_u10 (float8);
float8 Sleef_powf8_u10 (float8, float8);
float8 Sleef_sinhf8_u10 (float8);
float8 Sleef_coshf8_u10 (float8);
float8 Sleef_tanhf8_u10 (float8);
float8 Sleef_asinhf8_u10 (float8);
float8 Sleef_acoshf8_u10 (float8);
float8 Sleef_atanhf8_u10 (float8);
float8 Sleef_exp2f8_u10 (float8);
float8 Sleef_exp10f8_u10 (float8);
float8 Sleef_expm1f8_u10 (float8);
float8 Sleef_log10f8_u10 (float8);
float8 Sleef_log2f8_u10 (float8);
float8 Sleef_log1pf8_u10 (float8);
Sleef_float8_2 Sleef_sincospif8_u05 (float8);
Sleef_float8_2 Sleef_sincospif8_u35 (float8);
float8 Sleef_sinpif8_u05 (float8);
float8 Sleef_cospif8_u05 (float8);
float8 Sleef_ldexpf8 (float8, int8);
int8 Sleef_ilogbf8 (float8);
float8 Sleef_fmaf8 (float8, float8, float8);
float8 Sleef_sqrtf8 (float8);
float8 Sleef_sqrtf8_u05 (float8);
float8 Sleef_sqrtf8_u35 (float8);
float8 Sleef_hypotf8_u05 (float8, float8);
float8 Sleef_hypotf8_u35 (float8, float8);
float8 Sleef_fabsf8 (float8);
float8 Sleef_copysignf8 (float8, float8);
float8 Sleef_fmaxf8 (float8, float8);
float8 Sleef_fminf8 (float8, float8);
float8 Sleef_fdimf8 (float8, float8);
float8 Sleef_truncf8 (float8);
float8 Sleef_floorf8 (float8);
float8 Sleef_ceilf8 (float8);
float8 Sleef_roundf8 (float8);
float8 Sleef_rintf8 (float8);
float8 Sleef_nextafterf8 (float8, float8);
float8 Sleef_frfrexpf8 (float8);
int8 Sleef_expfrexpf8 (float8);
float8 Sleef_fmodf8 (float8, float8);
Sleef_float8_2 Sleef_modff8 (float8);
float8 Sleef_lgammaf8_u10 (float8);
Sleef_float8_2 Sleef_lgamma_rf8_u10 (float8);
float8 Sleef_tgammaf8_u10 (float8);
float8 Sleef_erff8_u10 (float8);
float8 Sleef_erfcf8_u15 (float8);

double4 Sleef_pownd4_u10 (double4, int4);
float8 Sleef_pownf8_u10 (float8, int8);
double4 Sleef_powrd4_u10 (double4, double4);
float8 Sleef_powrf8_u10 (float8, float8);

#endif

#ifdef SLEEF_VEC_128_AVAILABLE

#ifdef cl_khr_fp64

#ifndef Sleef_double2_2_DEFINED
typedef struct
{
  double2 x, y;
} Sleef_double2_2;
#define Sleef_double2_2_DEFINED
#endif

double2 Sleef_sind2_u35 (double2);
double2 Sleef_cosd2_u35 (double2);
Sleef_double2_2 Sleef_sincosd2_u35 (double2);
double2 Sleef_tand2_u35 (double2);
double2 Sleef_asind2_u35 (double2);
double2 Sleef_acosd2_u35 (double2);
double2 Sleef_atand2_u35 (double2);
double2 Sleef_atan2d2_u35 (double2, double2);
double2 Sleef_logd2_u35 (double2);
double2 Sleef_cbrtd2_u35 (double2);
double2 Sleef_sind2_u10 (double2);
double2 Sleef_cosd2_u10 (double2);
Sleef_double2_2 Sleef_sincosd2_u10 (double2);
double2 Sleef_tand2_u10 (double2);
double2 Sleef_asind2_u10 (double2);
double2 Sleef_acosd2_u10 (double2);
double2 Sleef_atand2_u10 (double2);
double2 Sleef_atan2d2_u10 (double2, double2);
double2 Sleef_logd2_u10 (double2);
double2 Sleef_cbrtd2_u10 (double2);
double2 Sleef_expd2_u10 (double2);
double2 Sleef_powd2_u10 (double2, double2);
double2 Sleef_sinhd2_u10 (double2);
double2 Sleef_coshd2_u10 (double2);
double2 Sleef_tanhd2_u10 (double2);
double2 Sleef_asinhd2_u10 (double2);
double2 Sleef_acoshd2_u10 (double2);
double2 Sleef_atanhd2_u10 (double2);
double2 Sleef_exp2d2_u10 (double2);
double2 Sleef_exp10d2_u10 (double2);
double2 Sleef_expm1d2_u10 (double2);
double2 Sleef_log10d2_u10 (double2);
double2 Sleef_log2d2_u10 (double2);
double2 Sleef_log1pd2_u10 (double2);
Sleef_double2_2 Sleef_sincospid2_u05 (double2);
Sleef_double2_2 Sleef_sincospid2_u35 (double2);
double2 Sleef_sinpid2_u05 (double2);
double2 Sleef_cospid2_u05 (double2);
double2 Sleef_ldexpd2 (double2, int2);
int2 Sleef_ilogbd2 (double2);
double2 Sleef_fmad2 (double2, double2, double2);
double2 Sleef_sqrtd2 (double2);
double2 Sleef_sqrtd2_u05 (double2);
double2 Sleef_sqrtd2_u35 (double2);
double2 Sleef_hypotd2_u05 (double2, double2);
double2 Sleef_hypotd2_u35 (double2, double2);
double2 Sleef_fabsd2 (double2);
double2 Sleef_copysignd2 (double2, double2);
double2 Sleef_fmaxd2 (double2, double2);
double2 Sleef_fmind2 (double2, double2);
double2 Sleef_fdimd2 (double2, double2);
double2 Sleef_truncd2 (double2);
double2 Sleef_floord2 (double2);
double2 Sleef_ceild2 (double2);
double2 Sleef_roundd2 (double2);
double2 Sleef_rintd2 (double2);
double2 Sleef_nextafterd2 (double2, double2);
double2 Sleef_frfrexpd2 (double2);
int2 Sleef_expfrexpd2 (double2);
double2 Sleef_fmodd2 (double2, double2);
Sleef_double2_2 Sleef_modfd2 (double2);
double2 Sleef_lgammad2_u10 (double2);
Sleef_double2_2 Sleef_lgamma_rd2_u10 (double2);
double2 Sleef_tgammad2_u10 (double2);
double2 Sleef_erfd2_u10 (double2);
double2 Sleef_erfcd2_u15 (double2);

double2 Sleef_pownd2_u10 (double2, int2);
double2 Sleef_powrd2_u10 (double2, double2);

#endif

#ifndef Sleef_float4_2_DEFINED
typedef struct
{
  float4 x, y;
} Sleef_float4_2;
#define Sleef_float4_2_DEFINED
#endif

float4 Sleef_sinf4_u35 (float4);
float4 Sleef_cosf4_u35 (float4);
Sleef_float4_2 Sleef_sincosf4_u35 (float4);
float4 Sleef_tanf4_u35 (float4);
float4 Sleef_asinf4_u35 (float4);
float4 Sleef_acosf4_u35 (float4);
float4 Sleef_atanf4_u35 (float4);
float4 Sleef_atan2f4_u35 (float4, float4);
float4 Sleef_logf4_u35 (float4);
float4 Sleef_cbrtf4_u35 (float4);
float4 Sleef_sinf4_u10 (float4);
float4 Sleef_cosf4_u10 (float4);
Sleef_float4_2 Sleef_sincosf4_u10 (float4);
float4 Sleef_tanf4_u10 (float4);
float4 Sleef_asinf4_u10 (float4);
float4 Sleef_acosf4_u10 (float4);
float4 Sleef_atanf4_u10 (float4);
float4 Sleef_atan2f4_u10 (float4, float4);
float4 Sleef_logf4_u10 (float4);
float4 Sleef_cbrtf4_u10 (float4);
float4 Sleef_expf4_u10 (float4);
float4 Sleef_powf4_u10 (float4, float4);
float4 Sleef_sinhf4_u10 (float4);
float4 Sleef_coshf4_u10 (float4);
float4 Sleef_tanhf4_u10 (float4);
float4 Sleef_asinhf4_u10 (float4);
float4 Sleef_acoshf4_u10 (float4);
float4 Sleef_atanhf4_u10 (float4);
float4 Sleef_exp2f4_u10 (float4);
float4 Sleef_exp10f4_u10 (float4);
float4 Sleef_expm1f4_u10 (float4);
float4 Sleef_log10f4_u10 (float4);
float4 Sleef_log2f4_u10 (float4);
float4 Sleef_log1pf4_u10 (float4);
Sleef_float4_2 Sleef_sincospif4_u05 (float4);
Sleef_float4_2 Sleef_sincospif4_u35 (float4);
float4 Sleef_sinpif4_u05 (float4);
float4 Sleef_cospif4_u05 (float4);
float4 Sleef_ldexpf4 (float4, int4);
int4 Sleef_ilogbf4 (float4);
float4 Sleef_fmaf4 (float4, float4, float4);
float4 Sleef_sqrtf4 (float4);
float4 Sleef_sqrtf4_u05 (float4);
float4 Sleef_sqrtf4_u35 (float4);
float4 Sleef_hypotf4_u05 (float4, float4);
float4 Sleef_hypotf4_u35 (float4, float4);
float4 Sleef_fabsf4 (float4);
float4 Sleef_copysignf4 (float4, float4);
float4 Sleef_fmaxf4 (float4, float4);
float4 Sleef_fminf4 (float4, float4);
float4 Sleef_fdimf4 (float4, float4);
float4 Sleef_truncf4 (float4);
float4 Sleef_floorf4 (float4);
float4 Sleef_ceilf4 (float4);
float4 Sleef_roundf4 (float4);
float4 Sleef_rintf4 (float4);
float4 Sleef_nextafterf4 (float4, float4);
float4 Sleef_frfrexpf4 (float4);
int4 Sleef_expfrexpf4 (float4);
float4 Sleef_fmodf4 (float4, float4);
Sleef_float4_2 Sleef_modff4 (float4);
float4 Sleef_lgammaf4_u10 (float4);
Sleef_float4_2 Sleef_lgamma_rf4_u10 (float4);
float4 Sleef_tgammaf4_u10 (float4);
float4 Sleef_erff4_u10 (float4);
float4 Sleef_erfcf4_u15 (float4);

float4 Sleef_pownf4_u10 (float4, int4);
float4 Sleef_powrf4_u10 (float4, float4);

#endif

#endif
