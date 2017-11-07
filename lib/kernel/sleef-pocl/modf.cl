#include "sleef_cl.h"

_CL_OVERLOADABLE
float
_cl_modf (float x, global float *iptr)
{
  Sleef_float2 temp;
  temp = Sleef_modff (x);
  *iptr = temp.y;
  return temp.x;
}

_CL_OVERLOADABLE
float3
_cl_modf (float3 x, global float3 *iptr)
{
  float4 temp;
  float4 x_3to4;
  x_3to4.xyz = x;
  float4 r = _cl_modf (x_3to4, &temp);
  *iptr = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
float2
_cl_modf (float2 x, global float2 *iptr)
{
  float plo, phi;
  float lo = _cl_modf (x.lo, &plo);
  float hi = _cl_modf (x.hi, &phi);

  *iptr = (float2) (plo, phi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4
_cl_modf (float4 x, global float4 *iptr)
{
#if defined(SLEEF_VEC_128_AVAILABLE)
  Sleef_float4_2 temp;
  temp = Sleef_modff4 (x);
  *iptr = temp.y;
  return temp.x;
#else

  float2 plo, phi;
  float2 lo = _cl_modf (x.lo, &plo);
  float2 hi = _cl_modf (x.hi, &phi);

  *iptr = (float4) (plo, phi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_modf (float8 x, global float8 *iptr)
{
#if defined(SLEEF_VEC_256_AVAILABLE)
  Sleef_float8_2 temp;
  temp = Sleef_modff8 (x);
  *iptr = temp.y;
  return temp.x;
#else

  float4 plo, phi;
  float4 lo = _cl_modf (x.lo, &plo);
  float4 hi = _cl_modf (x.hi, &phi);

  *iptr = (float8) (plo, phi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_modf (float16 x, global float16 *iptr)
{
#if defined(SLEEF_VEC_512_AVAILABLE)
  Sleef_float16_2 temp;
  temp = Sleef_modff16 (x);
  *iptr = temp.y;
  return temp.x;
#else

  float8 plo, phi;
  float8 lo = _cl_modf (x.lo, &plo);
  float8 hi = _cl_modf (x.hi, &phi);

  *iptr = (float16) (plo, phi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_modf (double x, global double *iptr)
{
  Sleef_double2 temp;
  temp = Sleef_modf (x);
  *iptr = temp.y;
  return temp.x;
}

_CL_OVERLOADABLE
double3
_cl_modf (double3 x, global double3 *iptr)
{
  double4 temp;
  double4 x_3to4;
  x_3to4.xyz = x;
  double4 r = _cl_modf (x_3to4, &temp);
  *iptr = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
double16
_cl_modf (double16 x, global double16 *iptr)
{
  double8 plo, phi;
  double8 lo = _cl_modf (x.lo, &plo);
  double8 hi = _cl_modf (x.hi, &phi);

  *iptr = (double16) (plo, phi);
  return (double16) (lo, hi);
}

_CL_OVERLOADABLE
double2
_cl_modf (double2 x, global double2 *iptr)
{
#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double2_2 temp;
  temp = Sleef_modfd2 (x);
  *iptr = temp.y;
  return temp.x;
#else

  double plo, phi;
  double lo = _cl_modf (x.lo, &plo);
  double hi = _cl_modf (x.hi, &phi);

  *iptr = (double2) (plo, phi);
  return (double2) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double4
_cl_modf (double4 x, global double4 *iptr)
{
#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double4_2 temp;
  temp = Sleef_modfd4 (x);
  *iptr = temp.y;
  return temp.x;
#else

  double2 plo, phi;
  double2 lo = _cl_modf (x.lo, &plo);
  double2 hi = _cl_modf (x.hi, &phi);

  *iptr = (double4) (plo, phi);
  return (double4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double8
_cl_modf (double8 x, global double8 *iptr)
{
#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double8_2 temp;
  temp = Sleef_modfd8 (x);
  *iptr = temp.y;
  return temp.x;
#else

  double4 plo, phi;
  double4 lo = _cl_modf (x.lo, &plo);
  double4 hi = _cl_modf (x.hi, &phi);

  *iptr = (double8) (plo, phi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

_CL_OVERLOADABLE
float
_cl_modf (float x, local float *iptr)
{
  Sleef_float2 temp;
  temp = Sleef_modff (x);
  *iptr = temp.y;
  return temp.x;
}

_CL_OVERLOADABLE
float3
_cl_modf (float3 x, local float3 *iptr)
{
  float4 temp;
  float4 x_3to4;
  x_3to4.xyz = x;
  float4 r = _cl_modf (x_3to4, &temp);
  *iptr = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
float2
_cl_modf (float2 x, local float2 *iptr)
{
  float plo, phi;
  float lo = _cl_modf (x.lo, &plo);
  float hi = _cl_modf (x.hi, &phi);

  *iptr = (float2) (plo, phi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4
_cl_modf (float4 x, local float4 *iptr)
{
#if defined(SLEEF_VEC_128_AVAILABLE)
  Sleef_float4_2 temp;
  temp = Sleef_modff4 (x);
  *iptr = temp.y;
  return temp.x;
#else

  float2 plo, phi;
  float2 lo = _cl_modf (x.lo, &plo);
  float2 hi = _cl_modf (x.hi, &phi);

  *iptr = (float4) (plo, phi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_modf (float8 x, local float8 *iptr)
{
#if defined(SLEEF_VEC_256_AVAILABLE)
  Sleef_float8_2 temp;
  temp = Sleef_modff8 (x);
  *iptr = temp.y;
  return temp.x;
#else

  float4 plo, phi;
  float4 lo = _cl_modf (x.lo, &plo);
  float4 hi = _cl_modf (x.hi, &phi);

  *iptr = (float8) (plo, phi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_modf (float16 x, local float16 *iptr)
{
#if defined(SLEEF_VEC_512_AVAILABLE)
  Sleef_float16_2 temp;
  temp = Sleef_modff16 (x);
  *iptr = temp.y;
  return temp.x;
#else

  float8 plo, phi;
  float8 lo = _cl_modf (x.lo, &plo);
  float8 hi = _cl_modf (x.hi, &phi);

  *iptr = (float16) (plo, phi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_modf (double x, local double *iptr)
{
  Sleef_double2 temp;
  temp = Sleef_modf (x);
  *iptr = temp.y;
  return temp.x;
}

_CL_OVERLOADABLE
double3
_cl_modf (double3 x, local double3 *iptr)
{
  double4 temp;
  double4 x_3to4;
  x_3to4.xyz = x;
  double4 r = _cl_modf (x_3to4, &temp);
  *iptr = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
double16
_cl_modf (double16 x, local double16 *iptr)
{
  double8 plo, phi;
  double8 lo = _cl_modf (x.lo, &plo);
  double8 hi = _cl_modf (x.hi, &phi);

  *iptr = (double16) (plo, phi);
  return (double16) (lo, hi);
}

_CL_OVERLOADABLE
double2
_cl_modf (double2 x, local double2 *iptr)
{
#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double2_2 temp;
  temp = Sleef_modfd2 (x);
  *iptr = temp.y;
  return temp.x;
#else

  double plo, phi;
  double lo = _cl_modf (x.lo, &plo);
  double hi = _cl_modf (x.hi, &phi);

  *iptr = (double2) (plo, phi);
  return (double2) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double4
_cl_modf (double4 x, local double4 *iptr)
{
#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double4_2 temp;
  temp = Sleef_modfd4 (x);
  *iptr = temp.y;
  return temp.x;
#else

  double2 plo, phi;
  double2 lo = _cl_modf (x.lo, &plo);
  double2 hi = _cl_modf (x.hi, &phi);

  *iptr = (double4) (plo, phi);
  return (double4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double8
_cl_modf (double8 x, local double8 *iptr)
{
#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double8_2 temp;
  temp = Sleef_modfd8 (x);
  *iptr = temp.y;
  return temp.x;
#else

  double4 plo, phi;
  double4 lo = _cl_modf (x.lo, &plo);
  double4 hi = _cl_modf (x.hi, &phi);

  *iptr = (double8) (plo, phi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

_CL_OVERLOADABLE
float
_cl_modf (float x, private float *iptr)
{
  Sleef_float2 temp;
  temp = Sleef_modff (x);
  *iptr = temp.y;
  return temp.x;
}

_CL_OVERLOADABLE
float3
_cl_modf (float3 x, private float3 *iptr)
{
  float4 temp;
  float4 x_3to4;
  x_3to4.xyz = x;
  float4 r = _cl_modf (x_3to4, &temp);
  *iptr = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
float2
_cl_modf (float2 x, private float2 *iptr)
{
  float plo, phi;
  float lo = _cl_modf (x.lo, &plo);
  float hi = _cl_modf (x.hi, &phi);

  *iptr = (float2) (plo, phi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4
_cl_modf (float4 x, private float4 *iptr)
{
#if defined(SLEEF_VEC_128_AVAILABLE)
  Sleef_float4_2 temp;
  temp = Sleef_modff4 (x);
  *iptr = temp.y;
  return temp.x;
#else

  float2 plo, phi;
  float2 lo = _cl_modf (x.lo, &plo);
  float2 hi = _cl_modf (x.hi, &phi);

  *iptr = (float4) (plo, phi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_modf (float8 x, private float8 *iptr)
{
#if defined(SLEEF_VEC_256_AVAILABLE)
  Sleef_float8_2 temp;
  temp = Sleef_modff8 (x);
  *iptr = temp.y;
  return temp.x;
#else

  float4 plo, phi;
  float4 lo = _cl_modf (x.lo, &plo);
  float4 hi = _cl_modf (x.hi, &phi);

  *iptr = (float8) (plo, phi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_modf (float16 x, private float16 *iptr)
{
#if defined(SLEEF_VEC_512_AVAILABLE)
  Sleef_float16_2 temp;
  temp = Sleef_modff16 (x);
  *iptr = temp.y;
  return temp.x;
#else

  float8 plo, phi;
  float8 lo = _cl_modf (x.lo, &plo);
  float8 hi = _cl_modf (x.hi, &phi);

  *iptr = (float16) (plo, phi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_modf (double x, private double *iptr)
{
  Sleef_double2 temp;
  temp = Sleef_modf (x);
  *iptr = temp.y;
  return temp.x;
}

_CL_OVERLOADABLE
double3
_cl_modf (double3 x, private double3 *iptr)
{
  double4 temp;
  double4 x_3to4;
  x_3to4.xyz = x;
  double4 r = _cl_modf (x_3to4, &temp);
  *iptr = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
double16
_cl_modf (double16 x, private double16 *iptr)
{
  double8 plo, phi;
  double8 lo = _cl_modf (x.lo, &plo);
  double8 hi = _cl_modf (x.hi, &phi);

  *iptr = (double16) (plo, phi);
  return (double16) (lo, hi);
}

_CL_OVERLOADABLE
double2
_cl_modf (double2 x, private double2 *iptr)
{
#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double2_2 temp;
  temp = Sleef_modfd2 (x);
  *iptr = temp.y;
  return temp.x;
#else

  double plo, phi;
  double lo = _cl_modf (x.lo, &plo);
  double hi = _cl_modf (x.hi, &phi);

  *iptr = (double2) (plo, phi);
  return (double2) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double4
_cl_modf (double4 x, private double4 *iptr)
{
#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double4_2 temp;
  temp = Sleef_modfd4 (x);
  *iptr = temp.y;
  return temp.x;
#else

  double2 plo, phi;
  double2 lo = _cl_modf (x.lo, &plo);
  double2 hi = _cl_modf (x.hi, &phi);

  *iptr = (double4) (plo, phi);
  return (double4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double8
_cl_modf (double8 x, private double8 *iptr)
{
#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double8_2 temp;
  temp = Sleef_modfd8 (x);
  *iptr = temp.y;
  return temp.x;
#else

  double4 plo, phi;
  double4 lo = _cl_modf (x.lo, &plo);
  double4 hi = _cl_modf (x.hi, &phi);

  *iptr = (double8) (plo, phi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */
