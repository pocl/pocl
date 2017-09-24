#include "sleef_cl.h"

_CL_OVERLOADABLE
float
_cl_sincos (float x, global float *cosval)
{
  Sleef_float2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosf_u10 (x);
#else
  temp = Sleef_sincosf_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
}

_CL_OVERLOADABLE
float3
_cl_sincos (float3 x, global float3 *cosval)
{
  float4 temp;
  float4 x_3to4;
  x_3to4.xyz = x;
  float4 r = _cl_sincos (x_3to4, &temp);
  *cosval = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
float2
_cl_sincos (float2 x, global float2 *cosval)
{
  float plo, phi;
  float lo = _cl_sincos (x.lo, &plo);
  float hi = _cl_sincos (x.hi, &phi);

  *cosval = (float2) (plo, phi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4
_cl_sincos (float4 x, global float4 *cosval)
{
#if defined(SLEEF_VEC_128_AVAILABLE)
  Sleef_float4_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosf4_u10 (x);
#else
  temp = Sleef_sincosf4_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  float2 plo, phi;
  float2 lo = _cl_sincos (x.lo, &plo);
  float2 hi = _cl_sincos (x.hi, &phi);

  *cosval = (float4) (plo, phi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_sincos (float8 x, global float8 *cosval)
{
#if defined(SLEEF_VEC_256_AVAILABLE)
  Sleef_float8_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosf8_u10 (x);
#else
  temp = Sleef_sincosf8_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  float4 plo, phi;
  float4 lo = _cl_sincos (x.lo, &plo);
  float4 hi = _cl_sincos (x.hi, &phi);

  *cosval = (float8) (plo, phi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_sincos (float16 x, global float16 *cosval)
{
#if defined(SLEEF_VEC_512_AVAILABLE)
  Sleef_float16_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosf16_u10 (x);
#else
  temp = Sleef_sincosf16_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  float8 plo, phi;
  float8 lo = _cl_sincos (x.lo, &plo);
  float8 hi = _cl_sincos (x.hi, &phi);

  *cosval = (float16) (plo, phi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_sincos (double x, global double *cosval)
{
  Sleef_double2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincos_u10 (x);
#else
  temp = Sleef_sincos_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
}

_CL_OVERLOADABLE
double3
_cl_sincos (double3 x, global double3 *cosval)
{
  double4 temp;
  double4 x_3to4;
  x_3to4.xyz = x;
  double4 r = _cl_sincos (x_3to4, &temp);
  *cosval = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
double16
_cl_sincos (double16 x, global double16 *cosval)
{
  double8 plo, phi;
  double8 lo = _cl_sincos (x.lo, &plo);
  double8 hi = _cl_sincos (x.hi, &phi);

  *cosval = (double16) (plo, phi);
  return (double16) (lo, hi);
}

_CL_OVERLOADABLE
double2
_cl_sincos (double2 x, global double2 *cosval)
{
#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double2_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosd2_u10 (x);
#else
  temp = Sleef_sincosd2_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  double plo, phi;
  double lo = _cl_sincos (x.lo, &plo);
  double hi = _cl_sincos (x.hi, &phi);

  *cosval = (double2) (plo, phi);
  return (double2) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double4
_cl_sincos (double4 x, global double4 *cosval)
{
#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double4_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosd4_u10 (x);
#else
  temp = Sleef_sincosd4_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  double2 plo, phi;
  double2 lo = _cl_sincos (x.lo, &plo);
  double2 hi = _cl_sincos (x.hi, &phi);

  *cosval = (double4) (plo, phi);
  return (double4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double8
_cl_sincos (double8 x, global double8 *cosval)
{
#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double8_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosd8_u10 (x);
#else
  temp = Sleef_sincosd8_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  double4 plo, phi;
  double4 lo = _cl_sincos (x.lo, &plo);
  double4 hi = _cl_sincos (x.hi, &phi);

  *cosval = (double8) (plo, phi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

_CL_OVERLOADABLE
float
_cl_sincos (float x, local float *cosval)
{
  Sleef_float2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosf_u10 (x);
#else
  temp = Sleef_sincosf_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
}

_CL_OVERLOADABLE
float3
_cl_sincos (float3 x, local float3 *cosval)
{
  float4 temp;
  float4 x_3to4;
  x_3to4.xyz = x;
  float4 r = _cl_sincos (x_3to4, &temp);
  *cosval = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
float2
_cl_sincos (float2 x, local float2 *cosval)
{
  float plo, phi;
  float lo = _cl_sincos (x.lo, &plo);
  float hi = _cl_sincos (x.hi, &phi);

  *cosval = (float2) (plo, phi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4
_cl_sincos (float4 x, local float4 *cosval)
{
#if defined(SLEEF_VEC_128_AVAILABLE)
  Sleef_float4_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosf4_u10 (x);
#else
  temp = Sleef_sincosf4_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  float2 plo, phi;
  float2 lo = _cl_sincos (x.lo, &plo);
  float2 hi = _cl_sincos (x.hi, &phi);

  *cosval = (float4) (plo, phi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_sincos (float8 x, local float8 *cosval)
{
#if defined(SLEEF_VEC_256_AVAILABLE)
  Sleef_float8_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosf8_u10 (x);
#else
  temp = Sleef_sincosf8_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  float4 plo, phi;
  float4 lo = _cl_sincos (x.lo, &plo);
  float4 hi = _cl_sincos (x.hi, &phi);

  *cosval = (float8) (plo, phi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_sincos (float16 x, local float16 *cosval)
{
#if defined(SLEEF_VEC_512_AVAILABLE)
  Sleef_float16_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosf16_u10 (x);
#else
  temp = Sleef_sincosf16_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  float8 plo, phi;
  float8 lo = _cl_sincos (x.lo, &plo);
  float8 hi = _cl_sincos (x.hi, &phi);

  *cosval = (float16) (plo, phi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_sincos (double x, local double *cosval)
{
  Sleef_double2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincos_u10 (x);
#else
  temp = Sleef_sincos_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
}

_CL_OVERLOADABLE
double3
_cl_sincos (double3 x, local double3 *cosval)
{
  double4 temp;
  double4 x_3to4;
  x_3to4.xyz = x;
  double4 r = _cl_sincos (x_3to4, &temp);
  *cosval = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
double16
_cl_sincos (double16 x, local double16 *cosval)
{
  double8 plo, phi;
  double8 lo = _cl_sincos (x.lo, &plo);
  double8 hi = _cl_sincos (x.hi, &phi);

  *cosval = (double16) (plo, phi);
  return (double16) (lo, hi);
}

_CL_OVERLOADABLE
double2
_cl_sincos (double2 x, local double2 *cosval)
{
#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double2_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosd2_u10 (x);
#else
  temp = Sleef_sincosd2_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  double plo, phi;
  double lo = _cl_sincos (x.lo, &plo);
  double hi = _cl_sincos (x.hi, &phi);

  *cosval = (double2) (plo, phi);
  return (double2) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double4
_cl_sincos (double4 x, local double4 *cosval)
{
#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double4_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosd4_u10 (x);
#else
  temp = Sleef_sincosd4_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  double2 plo, phi;
  double2 lo = _cl_sincos (x.lo, &plo);
  double2 hi = _cl_sincos (x.hi, &phi);

  *cosval = (double4) (plo, phi);
  return (double4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double8
_cl_sincos (double8 x, local double8 *cosval)
{
#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double8_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosd8_u10 (x);
#else
  temp = Sleef_sincosd8_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  double4 plo, phi;
  double4 lo = _cl_sincos (x.lo, &plo);
  double4 hi = _cl_sincos (x.hi, &phi);

  *cosval = (double8) (plo, phi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

_CL_OVERLOADABLE
float
_cl_sincos (float x, private float *cosval)
{
  Sleef_float2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosf_u10 (x);
#else
  temp = Sleef_sincosf_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
}

_CL_OVERLOADABLE
float3
_cl_sincos (float3 x, private float3 *cosval)
{
  float4 temp;
  float4 x_3to4;
  x_3to4.xyz = x;
  float4 r = _cl_sincos (x_3to4, &temp);
  *cosval = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
float2
_cl_sincos (float2 x, private float2 *cosval)
{
  float plo, phi;
  float lo = _cl_sincos (x.lo, &plo);
  float hi = _cl_sincos (x.hi, &phi);

  *cosval = (float2) (plo, phi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4
_cl_sincos (float4 x, private float4 *cosval)
{
#if defined(SLEEF_VEC_128_AVAILABLE)
  Sleef_float4_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosf4_u10 (x);
#else
  temp = Sleef_sincosf4_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  float2 plo, phi;
  float2 lo = _cl_sincos (x.lo, &plo);
  float2 hi = _cl_sincos (x.hi, &phi);

  *cosval = (float4) (plo, phi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_sincos (float8 x, private float8 *cosval)
{
#if defined(SLEEF_VEC_256_AVAILABLE)
  Sleef_float8_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosf8_u10 (x);
#else
  temp = Sleef_sincosf8_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  float4 plo, phi;
  float4 lo = _cl_sincos (x.lo, &plo);
  float4 hi = _cl_sincos (x.hi, &phi);

  *cosval = (float8) (plo, phi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_sincos (float16 x, private float16 *cosval)
{
#if defined(SLEEF_VEC_512_AVAILABLE)
  Sleef_float16_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosf16_u10 (x);
#else
  temp = Sleef_sincosf16_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  float8 plo, phi;
  float8 lo = _cl_sincos (x.lo, &plo);
  float8 hi = _cl_sincos (x.hi, &phi);

  *cosval = (float16) (plo, phi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_sincos (double x, private double *cosval)
{
  Sleef_double2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincos_u10 (x);
#else
  temp = Sleef_sincos_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
}

_CL_OVERLOADABLE
double3
_cl_sincos (double3 x, private double3 *cosval)
{
  double4 temp;
  double4 x_3to4;
  x_3to4.xyz = x;
  double4 r = _cl_sincos (x_3to4, &temp);
  *cosval = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
double16
_cl_sincos (double16 x, private double16 *cosval)
{
  double8 plo, phi;
  double8 lo = _cl_sincos (x.lo, &plo);
  double8 hi = _cl_sincos (x.hi, &phi);

  *cosval = (double16) (plo, phi);
  return (double16) (lo, hi);
}

_CL_OVERLOADABLE
double2
_cl_sincos (double2 x, private double2 *cosval)
{
#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double2_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosd2_u10 (x);
#else
  temp = Sleef_sincosd2_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  double plo, phi;
  double lo = _cl_sincos (x.lo, &plo);
  double hi = _cl_sincos (x.hi, &phi);

  *cosval = (double2) (plo, phi);
  return (double2) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double4
_cl_sincos (double4 x, private double4 *cosval)
{
#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double4_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosd4_u10 (x);
#else
  temp = Sleef_sincosd4_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  double2 plo, phi;
  double2 lo = _cl_sincos (x.lo, &plo);
  double2 hi = _cl_sincos (x.hi, &phi);

  *cosval = (double4) (plo, phi);
  return (double4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double8
_cl_sincos (double8 x, private double8 *cosval)
{
#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double8_2 temp;

#ifdef MAX_PRECISION
  temp = Sleef_sincosd8_u10 (x);
#else
  temp = Sleef_sincosd8_u35 (x);
#endif

  *cosval = temp.y;
  return temp.x;
#else

  double4 plo, phi;
  double4 lo = _cl_sincos (x.lo, &plo);
  double4 hi = _cl_sincos (x.hi, &phi);

  *cosval = (double8) (plo, phi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */
