#include "sleef_cl.h"

_CL_OVERLOADABLE
float
_cl_lgamma_r (float x, global int *iptr)
{
  Sleef_float2 temp;
  temp = Sleef_lgamma_rf_u10(x);
  *iptr = convert_int(temp.y);
  return temp.x;
}

_CL_OVERLOADABLE
float3
_cl_lgamma_r (float3 x, global int3 *iptr)
{
  int4 temp;
  float4 x_3to4;
  x_3to4.xyz = x;
  float4 r = _cl_lgamma_r (x_3to4, &temp);
  *iptr = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
float2
_cl_lgamma_r (float2 x, global int2 *iptr)
{
  int plo, phi;
  float lo = _cl_lgamma_r (x.lo, &plo);
  float hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int2)(plo, phi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4
_cl_lgamma_r (float4 x, global int4 *iptr)
{
#if defined(SLEEF_VEC_128_AVAILABLE)
  Sleef_float4_2 temp;
  temp = Sleef_lgamma_rf4_u10(x);
  *iptr = convert_int4(temp.y);
  return temp.x;
#else

  int2 plo, phi;
  float2 lo = _cl_lgamma_r (x.lo, &plo);
  float2 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int4) (plo, phi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_lgamma_r (float8 x, global int8 *iptr)
{
#if defined(SLEEF_VEC_256_AVAILABLE)
  Sleef_float8_2 temp;
  temp = Sleef_lgamma_rf8_u10(x);
  *iptr = convert_int8(temp.y);
  return temp.x;
#else

  int4 plo, phi;
  float4 lo = _cl_lgamma_r (x.lo, &plo);
  float4 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int8) (plo, phi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_lgamma_r (float16 x, global int16 *iptr)
{
#if defined(SLEEF_VEC_512_AVAILABLE)
  Sleef_float16_2 temp;
  temp = Sleef_lgamma_rf16_u10(x);
  *iptr = convert_int16(temp.y);
  return temp.x;
#else

  int8 plo, phi;
  float8 lo = _cl_lgamma_r (x.lo, &plo);
  float8 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int16) (plo, phi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_lgamma_r (double x, global int *iptr)
{
  Sleef_double2 temp;
  temp = Sleef_lgamma_r_u10(x);
  *iptr = convert_int(temp.y);
  return temp.x;
}

_CL_OVERLOADABLE
double3
_cl_lgamma_r (double3 x, global int3 *iptr)
{
  int4 temp;
  double4 x_3to4;
  x_3to4.xyz = x;
  double4 r = _cl_lgamma_r (x_3to4, &temp);
  *iptr = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
double16
_cl_lgamma_r (double16 x, global int16 *iptr)
{
  int8 plo, phi;
  double8 lo = _cl_lgamma_r (x.lo, &plo);
  double8 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int16) (plo, phi);
  return (double16) (lo, hi);
}

_CL_OVERLOADABLE
double2
_cl_lgamma_r (double2 x, global int2 *iptr)
{
#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double2_2 temp;
  temp = Sleef_lgamma_rd2_u10(x);
  *iptr = convert_int2(temp.y);
  return temp.x;
#else

  int plo, phi;
  double lo = _cl_lgamma_r (x.lo, &plo);
  double hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int2) (plo, phi);
  return (double2) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double4
_cl_lgamma_r (double4 x, global int4 *iptr)
{
#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double4_2 temp;
  temp = Sleef_lgamma_rd4_u10(x);
  *iptr = convert_int4(temp.y);
  return temp.x;
#else

  int2 plo, phi;
  double2 lo = _cl_lgamma_r (x.lo, &plo);
  double2 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int4) (plo, phi);
  return (double4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double8
_cl_lgamma_r (double8 x, global int8 *iptr)
{
#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double8_2 temp;
  temp = Sleef_lgamma_rd8_u10(x);
  *iptr = convert_int8(temp.y);
  return temp.x;
#else

  int4 plo, phi;
  double4 lo = _cl_lgamma_r (x.lo, &plo);
  double4 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int8) (plo, phi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

/****************************************************************/
/****************************************************************/
/****************************************************************/
/****************************************************************/


_CL_OVERLOADABLE
float
_cl_lgamma_r (float x, local int *iptr)
{
  Sleef_float2 temp;
  temp = Sleef_lgamma_rf_u10(x);
  *iptr = convert_int(temp.y);
  return temp.x;
}

_CL_OVERLOADABLE
float3
_cl_lgamma_r (float3 x, local int3 *iptr)
{
  int4 temp;
  float4 x_3to4;
  x_3to4.xyz = x;
  float4 r = _cl_lgamma_r (x_3to4, &temp);
  *iptr = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
float2
_cl_lgamma_r (float2 x, local int2 *iptr)
{
  int plo, phi;
  float lo = _cl_lgamma_r (x.lo, &plo);
  float hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int2)(plo, phi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4
_cl_lgamma_r (float4 x, local int4 *iptr)
{
#if defined(SLEEF_VEC_128_AVAILABLE)
  Sleef_float4_2 temp;
  temp = Sleef_lgamma_rf4_u10(x);
  *iptr = convert_int4(temp.y);
  return temp.x;
#else

  int2 plo, phi;
  float2 lo = _cl_lgamma_r (x.lo, &plo);
  float2 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int4) (plo, phi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_lgamma_r (float8 x, local int8 *iptr)
{
#if defined(SLEEF_VEC_256_AVAILABLE)
  Sleef_float8_2 temp;
  temp = Sleef_lgamma_rf8_u10(x);
  *iptr = convert_int8(temp.y);
  return temp.x;
#else

  int4 plo, phi;
  float4 lo = _cl_lgamma_r (x.lo, &plo);
  float4 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int8) (plo, phi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_lgamma_r (float16 x, local int16 *iptr)
{
#if defined(SLEEF_VEC_512_AVAILABLE)
  Sleef_float16_2 temp;
  temp = Sleef_lgamma_rf16_u10(x);
  *iptr = convert_int16(temp.y);
  return temp.x;
#else

  int8 plo, phi;
  float8 lo = _cl_lgamma_r (x.lo, &plo);
  float8 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int16) (plo, phi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_lgamma_r (double x, local int *iptr)
{
  Sleef_double2 temp;
  temp = Sleef_lgamma_r_u10(x);
  *iptr = convert_int(temp.y);
  return temp.x;
}

_CL_OVERLOADABLE
double3
_cl_lgamma_r (double3 x, local int3 *iptr)
{
  int4 temp;
  double4 x_3to4;
  x_3to4.xyz = x;
  double4 r = _cl_lgamma_r (x_3to4, &temp);
  *iptr = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
double16
_cl_lgamma_r (double16 x, local int16 *iptr)
{
  int8 plo, phi;
  double8 lo = _cl_lgamma_r (x.lo, &plo);
  double8 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int16) (plo, phi);
  return (double16) (lo, hi);
}

_CL_OVERLOADABLE
double2
_cl_lgamma_r (double2 x, local int2 *iptr)
{
#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double2_2 temp;
  temp = Sleef_lgamma_rd2_u10(x);
  *iptr = convert_int2(temp.y);
  return temp.x;
#else

  int plo, phi;
  double lo = _cl_lgamma_r (x.lo, &plo);
  double hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int2) (plo, phi);
  return (double2) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double4
_cl_lgamma_r (double4 x, local int4 *iptr)
{
#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double4_2 temp;
  temp = Sleef_lgamma_rd4_u10(x);
  *iptr = convert_int4(temp.y);
  return temp.x;
#else

  int2 plo, phi;
  double2 lo = _cl_lgamma_r (x.lo, &plo);
  double2 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int4) (plo, phi);
  return (double4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double8
_cl_lgamma_r (double8 x, local int8 *iptr)
{
#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double8_2 temp;
  temp = Sleef_lgamma_rd8_u10(x);
  *iptr = convert_int8(temp.y);
  return temp.x;
#else

  int4 plo, phi;
  double4 lo = _cl_lgamma_r (x.lo, &plo);
  double4 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int8) (plo, phi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */


/****************************************************************/
/****************************************************************/
/****************************************************************/
/****************************************************************/


_CL_OVERLOADABLE
float
_cl_lgamma_r (float x, private int *iptr)
{
  Sleef_float2 temp;
  temp = Sleef_lgamma_rf_u10(x);
  *iptr = convert_int(temp.y);
  return temp.x;
}

_CL_OVERLOADABLE
float3
_cl_lgamma_r (float3 x, private int3 *iptr)
{
  int4 temp;
  float4 x_3to4;
  x_3to4.xyz = x;
  float4 r = _cl_lgamma_r (x_3to4, &temp);
  *iptr = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
float2
_cl_lgamma_r (float2 x, private int2 *iptr)
{
  int plo, phi;
  float lo = _cl_lgamma_r (x.lo, &plo);
  float hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int2)(plo, phi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4
_cl_lgamma_r (float4 x, private int4 *iptr)
{
#if defined(SLEEF_VEC_128_AVAILABLE)
  Sleef_float4_2 temp;
  temp = Sleef_lgamma_rf4_u10(x);
  *iptr = convert_int4(temp.y);
  return temp.x;
#else

  int2 plo, phi;
  float2 lo = _cl_lgamma_r (x.lo, &plo);
  float2 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int4) (plo, phi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_lgamma_r (float8 x, private int8 *iptr)
{
#if defined(SLEEF_VEC_256_AVAILABLE)
  Sleef_float8_2 temp;
  temp = Sleef_lgamma_rf8_u10(x);
  *iptr = convert_int8(temp.y);
  return temp.x;
#else

  int4 plo, phi;
  float4 lo = _cl_lgamma_r (x.lo, &plo);
  float4 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int8) (plo, phi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_lgamma_r (float16 x, private int16 *iptr)
{
#if defined(SLEEF_VEC_512_AVAILABLE)
  Sleef_float16_2 temp;
  temp = Sleef_lgamma_rf16_u10(x);
  *iptr = convert_int16(temp.y);
  return temp.x;
#else

  int8 plo, phi;
  float8 lo = _cl_lgamma_r (x.lo, &plo);
  float8 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int16) (plo, phi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_lgamma_r (double x, private int *iptr)
{
  Sleef_double2 temp;
  temp = Sleef_lgamma_r_u10(x);
  *iptr = convert_int(temp.y);
  return temp.x;
}

_CL_OVERLOADABLE
double3
_cl_lgamma_r (double3 x, private int3 *iptr)
{
  int4 temp;
  double4 x_3to4;
  x_3to4.xyz = x;
  double4 r = _cl_lgamma_r (x_3to4, &temp);
  *iptr = temp.xyz;
  return r.xyz;
}

_CL_OVERLOADABLE
double16
_cl_lgamma_r (double16 x, private int16 *iptr)
{
  int8 plo, phi;
  double8 lo = _cl_lgamma_r (x.lo, &plo);
  double8 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int16) (plo, phi);
  return (double16) (lo, hi);
}

_CL_OVERLOADABLE
double2
_cl_lgamma_r (double2 x, private int2 *iptr)
{
#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double2_2 temp;
  temp = Sleef_lgamma_rd2_u10(x);
  *iptr = convert_int2(temp.y);
  return temp.x;
#else

  int plo, phi;
  double lo = _cl_lgamma_r (x.lo, &plo);
  double hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int2) (plo, phi);
  return (double2) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double4
_cl_lgamma_r (double4 x, private int4 *iptr)
{
#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double4_2 temp;
  temp = Sleef_lgamma_rd4_u10(x);
  *iptr = convert_int4(temp.y);
  return temp.x;
#else

  int2 plo, phi;
  double2 lo = _cl_lgamma_r (x.lo, &plo);
  double2 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int4) (plo, phi);
  return (double4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double8
_cl_lgamma_r (double8 x, private int8 *iptr)
{
#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  Sleef_double8_2 temp;
  temp = Sleef_lgamma_rd8_u10(x);
  *iptr = convert_int8(temp.y);
  return temp.x;
#else

  int4 plo, phi;
  double4 lo = _cl_lgamma_r (x.lo, &plo);
  double4 hi = _cl_lgamma_r (x.hi, &phi);

  *iptr = (int8) (plo, phi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */
