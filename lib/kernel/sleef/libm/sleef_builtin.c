/* OpenCL built-in library: SLEEF C fallback using libm / compiler builtins

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

#define _GNU_SOURCE
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

#include "sleef.h"
#include "sleef_cl.h"
#include "rename.h"

//##################################################################

static int64_t
doubleToRawLongBits (double d)
{
  union
  {
    double f;
    int64_t i;
  } tmp;
  tmp.f = d;
  return tmp.i;
}

static double
longBitsToDouble (int64_t i)
{
  union
  {
    double f;
    int64_t i;
  } tmp;
  tmp.i = i;
  return tmp.f;
}

static double
fabsk (double x)
{
  return longBitsToDouble (0x7fffffffffffffffLL & doubleToRawLongBits (x));
}

static double
mulsign (double x, double y)
{
  return longBitsToDouble (doubleToRawLongBits (x)
                           ^ (doubleToRawLongBits (y) & (1LL << 63)));
}

//##################################################################

#define INFINITYf ((float)INFINITY)
#define NANf ((float)NAN)

static int32_t
floatToRawIntBits (float d)
{
  union
  {
    float f;
    int32_t i;
  } tmp;
  tmp.f = d;
  return tmp.i;
}

static float
intBitsToFloat (int32_t i)
{
  union
  {
    float f;
    int32_t i;
  } tmp;
  tmp.i = i;
  return tmp.f;
}

static float
fabsfk (float x)
{
  return intBitsToFloat (0x7fffffffL & floatToRawIntBits (x));
}

static float
mulsignf (float x, float y)
{
  return intBitsToFloat (floatToRawIntBits (x)
                         ^ (floatToRawIntBits (y) & (1 << 31)));
}

//##################################################################


double
xsin (double x)
{
  return __builtin_sin (x);
}
double
xcos (double x)
{
  return __builtin_cos (x);
}
double
xtan (double x)
{
  return __builtin_tan (x);
}
double
xasin (double x)
{
  return __builtin_asin (x);
}
double
xacos (double x)
{
  return __builtin_acos (x);
}
double
xatan (double x)
{
  return __builtin_atan (x);
}
double
xatan2 (double x, double y)
{
  return __builtin_atan2 (x, y);
}
double
xlog (double x)
{
  return __builtin_log (x);
}
double
xcbrt (double x)
{
  return __builtin_cbrt (x);
}

double
xsin_u1 (double x)
{
  return __builtin_sin (x);
}
double
xcos_u1 (double x)
{
  return __builtin_cos (x);
}
double
xtan_u1 (double x)
{
  return __builtin_tan (x);
}
double
xasin_u1 (double x)
{
  return __builtin_asin (x);
}
double
xacos_u1 (double x)
{
  return __builtin_acos (x);
}
double
xatan_u1 (double x)
{
  return __builtin_atan (x);
}
double
xatan2_u1 (double x, double y)
{
  return __builtin_atan2 (x, y);
}
double
xlog_u1 (double x)
{
  return __builtin_log (x);
}
double
xcbrt_u1 (double x)
{
  return __builtin_cbrt (x);
}

double
xexp (double x)
{
  return __builtin_exp (x);
}

double
xpow (double x, double y)
{
  return __builtin_pow (x, y);
}

double
xpown (double x, int y)
{
  return __builtin_pow (x, (double)y);
}

double
xpowr (double x, double y)
{
  if (x < 0.0)
    return NAN;
  if (isnan(y))
    return y;
  double res = __builtin_pow (x, y);
  return res;
}

double
xsinh (double x)
{
  return __builtin_sinh (x);
}

double
xcosh (double x)
{
  return __builtin_cosh (x);
}
double
xtanh (double x)
{
  return __builtin_tanh (x);
}

double
xasinh (double x)
{
  return __builtin_asinh (x);
}
double
xacosh (double x)
{
  return __builtin_acosh (x);
}
double
xatanh (double x)
{
  return __builtin_atanh (x);
}

double
xexp2 (double x)
{
  return __builtin_exp2 (x);
}
double
xexp10 (double x)
{
  return exp10 (x);
}

double
xexpm1 (double x)
{
  return __builtin_expm1 (x);
}

double
xlog10 (double x)
{
  return __builtin_log10 (x);
}
double
xlog1p (double x)
{
  return __builtin_log1p (x);
}

double
xsinpi_u05 (double x)
{
  return __builtin_sin (x * (double)M_PI);
}
double
xcospi_u05 (double x)
{
  return __builtin_cos (x * (double)M_PI);
}

Sleef_double2
xsincos (double x)
{
  Sleef_double2 tmp;
  sincos (x, &tmp.x, &tmp.y);
  return tmp;
}

Sleef_double2
xsincos_u1 (double x)
{
  return xsincos (x);
}

double
xldexp (double x, int k)
{
  return __builtin_ldexp (x, k);
}

int
xilogb (double x)
{
  return __builtin_ilogb (x);
}

double
xfma (double x, double y, double z)
{
  return __builtin_fma (x, y, z);
}

double
xsqrt_u05 (double x)
{
  return __builtin_sqrt (x);
}

double
xsqrt_u35 (double x)
{
  return __builtin_sqrt (x);
}

double
xhypot_u05 (double x, double y)
{
  return __builtin_hypot (x, y);
}

double
xhypot_u35 (double x, double y)
{
  return __builtin_hypot (x, y);
}

double
xfabs (double x)
{
  return __builtin_fabs (x);
}

double
xcopysign (double x, double y)
{
  return __builtin_copysign (x, y);
}
double
xfmax (double x, double y)
{
  return __builtin_fmax (x, y);
}
double
xfmin (double x, double y)
{
  return __builtin_fmin (x, y);
}

double
xfdim (double x, double y)
{
  return __builtin_fdim (x, y);
}
double
xtrunc (double x)
{
  return __builtin_trunc (x);
}
double
xfloor (double x)
{
  return __builtin_floor (x);
}

double
xceil (double x)
{
  return __builtin_ceil (x);
}
double
xround (double x)
{
  return __builtin_round (x);
}
double
xrint (double x)
{
  return __builtin_rint (x);
}

double
xnextafter (double x, double y)
{
  return __builtin_nextafter (x, y);
}

double
xfrfrexp (double x)
{
  union
  {
    double f;
    uint64_t u;
  } cx;

  if (__builtin_isnan (x))
    return x;

  if (fabsk (x) < DBL_MIN)
    x *= (1ULL << 63);

  cx.f = x;
  cx.u &= ~0x7ff0000000000000ULL;
  cx.u |= 0x3fe0000000000000ULL;

  if (__builtin_isinf (x))
    cx.f = mulsign (INFINITY, x);
  if (x == 0)
    cx.f = x;

  return cx.f;
}

int
xexpfrexp (double x)
{
  union
  {
    double f;
    uint64_t u;
  } cx;

  int ret = 0;

  if (fabsk (x) < DBL_MIN)
    {
      x *= (1ULL << 63);
      ret = -63;
    }

  cx.f = x;
  ret += (int32_t) (((cx.u >> 52) & 0x7ff)) - 0x3fe;

  if (x == 0 || __builtin_isnan (x) || __builtin_isinf (x))
    ret = 0;

  return ret;
}

double
xfmod (double x, double y)
{
  return __builtin_fmod (x, y);
}

Sleef_double2
xmodf (double x)
{
  Sleef_double2 res;
  double tmp;
  res.x = __builtin_modf (x, &tmp);
  res.y = tmp;
  return res;
}

double
xlgamma_u1 (double x)
{
  return __builtin_lgamma (x);
}
double
xtgamma_u1 (double x)
{
  return __builtin_tgamma (x);
}
double
xerf_u1 (double x)
{
  return __builtin_erf (x);
}
double
xerfc_u15 (double x)
{
  return __builtin_erfc (x);
}

// *********************************************************************
// *********************************************************************
// *********************************************************************
// *********************************************************************

float
xsinf (float x)
{
  return __builtin_sinf (x);
}
float
xcosf (float x)
{
  return __builtin_cosf (x);
}
float
xtanf (float x)
{
  return __builtin_tanf (x);
}
float
xasinf (float x)
{
  return __builtin_asinf (x);
}
float
xacosf (float x)
{
  return __builtin_acosf (x);
}
float
xatanf (float x)
{
  return __builtin_atanf (x);
}
float
xatan2f (float x, float y)
{
  return __builtin_atan2f (x, y);
}
float
xlogf (float x)
{
  return __builtin_logf (x);
}
float
xcbrtf (float x)
{
  return __builtin_cbrtf (x);
}

float
xsinf_u1 (float x)
{
  return __builtin_sinf (x);
}
float
xcosf_u1 (float x)
{
  return __builtin_cosf (x);
}
float
xtanf_u1 (float x)
{
  return __builtin_tanf (x);
}
float
xasinf_u1 (float x)
{
  return __builtin_asinf (x);
}
float
xacosf_u1 (float x)
{
  return __builtin_acosf (x);
}
float
xatanf_u1 (float x)
{
  return __builtin_atanf (x);
}
float
xatan2f_u1 (float x, float y)
{
  return __builtin_atan2f (x, y);
}
float
xlogf_u1 (float x)
{
  return __builtin_logf (x);
}
float
xcbrtf_u1 (float x)
{
  return __builtin_cbrtf (x);
}

float
xexpf (float x)
{
  return __builtin_expf (x);
}

float
xpowf (float x, float y)
{

  return (float) __builtin_pow ((double)x, (double)y);
}

float
xpownf (float x, int y)
{
  return (float) __builtin_pow ((double)x, (double)y);
}

float
xpowrf (float x, float y)
{
  if (x < 0.0f)
    return NAN;
  float res = (float) __builtin_pow ((double)x, (double)y);
  return res;
}

float
xsinhf (float x)
{
  return __builtin_sinhf (x);
}
float
xcoshf (float x)
{
  return __builtin_coshf (x);
}
float
xtanhf (float x)
{
  return __builtin_tanhf (x);
}

float
xasinhf (float x)
{
  return __builtin_asinhf (x);
}
float
xacoshf (float x)
{
  return __builtin_acoshf (x);
}
float
xatanhf (float x)
{
  return __builtin_atanhf (x);
}

float
xexp2f (float x)
{
  return __builtin_exp2f (x);
}

float
xexp10f (float x)
{
  return exp10f (x);
}

float
xexpm1f (float x)
{
  return __builtin_expm1f (x);
}

float
xlog10f (float x)
{
  return __builtin_log10f (x);
}
float
xlog1pf (float x)
{
  return __builtin_log1pf (x);
}

float
xsinpif_u05 (float x)
{
  return __builtin_sinf (x * (float)M_PI);
}
float
xcospif_u05 (float x)
{
  return __builtin_cosf (x * (float)M_PI);
}

Sleef_float2
xsincosf (float x)
{
  Sleef_float2 tmp;
  sincosf (x, &tmp.x, &tmp.y);
  return tmp;
}

Sleef_float2
xsincosf_u1 (float x)
{
  return xsincosf (x);
}

float
xsqrtf_u05 (float x)
{
  return __builtin_sqrtf (x);
}

float
xsqrtf_u35 (float x)
{
  return __builtin_sqrtf (x);
}

float
xhypotf_u05 (float x, float y)
{
  return __builtin_hypotf (x, y);
}

float
xhypotf_u35 (float x, float y)
{
  return __builtin_hypotf (x, y);
}

float
xldexpf (float x, int k)
{
  return __builtin_ldexpf (x, k);
}

int
xilogbf (float x)
{
  return __builtin_ilogbf (x);
}

float
xfmaf (float x, float y, float z)
{
  return __builtin_fmaf (x, y, z);
}

float
xfabsf (float x)
{
  return __builtin_fabsf (x);
}

float
xcopysignf (float x, float y)
{
  return __builtin_copysignf (x, y);
}
float
xfmaxf (float x, float y)
{
  return __builtin_fmaxf (x, y);
}
float
xfminf (float x, float y)
{
  return __builtin_fminf (x, y);
}

float
xfdimf (float x, float y)
{
  return __builtin_fdimf (x, y);
}
float
xtruncf (float x)
{
  return __builtin_truncf (x);
}
float
xfloorf (float x)
{
  return __builtin_floorf (x);
}

float
xceilf (float x)
{
  return __builtin_ceilf (x);
}
float
xroundf (float x)
{
  return __builtin_roundf (x);
}
float
xrintf (float x)
{
  return __builtin_rintf (x);
}

float
xnextafterf (float x, float y)
{
  return __builtin_nextafterf (x, y);
}


float
xfrfrexpf (float x)
{
  union
  {
    float f;
    int32_t u;
  } cx;

  if (__builtin_isnan (x))
    return x;

  if (fabsfk (x) < FLT_MIN)
    x *= (1 << 30);

  cx.f = x;
  cx.u &= ~0x7f800000U;
  cx.u |= 0x3f000000U;

  if (__builtin_isinf (x))
    cx.f = mulsignf (INFINITYf, x);
  if (x == 0)
    cx.f = x;

  return cx.f;
}

int
xexpfrexpf (float x)
{
  union
  {
    float f;
    uint32_t u;
  } cx;

  int ret = 0;

  if (fabsfk (x) < FLT_MIN)
    {
      x *= (1 << 30);
      ret = -30;
    }

  cx.f = x;
  ret += (int32_t) (((cx.u >> 23) & 0xff)) - 0x7e;

  if (x == 0 || __builtin_isnan (x) || __builtin_isinf (x))
    ret = 0;

  return ret;
}

float
xfmodf (float x, float y)
{
  return __builtin_fmodf (x, y);
}

Sleef_float2
xmodff (float x)
{
  Sleef_float2 res;
  float tmp;
  res.x = __builtin_modff (x, &tmp);
  res.y = tmp;
  return res;
}

float
xlgammaf_u1 (float x)
{
  return __builtin_lgammaf (x);
}
float
xtgammaf_u1 (float x)
{
  return __builtin_tgammaf (x);
}
float
xerff_u1 (float x)
{
  return __builtin_erff (x);
}
float
xerfcf_u15 (float x)
{
  return __builtin_erfcf (x);
}
