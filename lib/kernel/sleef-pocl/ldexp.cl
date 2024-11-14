#include "sleef_cl.h"

#ifndef ENABLE_CONFORMANCE
#error Must define ENABLE_CONFORMANCE to 0 or 1
#endif

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_ldexp (double x, int k)
{
  return Sleef_ldexp (x, k);
}

_CL_OVERLOADABLE
double2
_cl_ldexp (double2 x, int2 k)
{

#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_ldexpd2 (x, k);
#else

  double lo = _cl_ldexp (x.lo, k.lo);
  double hi = _cl_ldexp (x.hi, k.hi);
  return (double2) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double4 _cl_ldexp (double4, int4);

_CL_OVERLOADABLE
double3
_cl_ldexp (double3 x, int3 k)
{

  double4 x_3to4 = (double4) (x, (double)0);
  int4 k_3to4 = (int4) (k, (double)0);

  double4 r = _cl_ldexp (x_3to4, k_3to4);
  return r.xyz;
}

_CL_OVERLOADABLE
double4
_cl_ldexp (double4 x, int4 k)
{

#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_ldexpd4 (x, k);
#else

  double2 lo = _cl_ldexp (x.lo, k.lo);
  double2 hi = _cl_ldexp (x.hi, k.hi);
  return (double4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double8
_cl_ldexp (double8 x, int8 k)
{

#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_ldexpd8 (x, k);
#else

  double4 lo = _cl_ldexp (x.lo, k.lo);
  double4 hi = _cl_ldexp (x.hi, k.hi);
  return (double8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
double16
_cl_ldexp (double16 x, int16 k)
{

  double8 lo = _cl_ldexp (x.lo, k.lo);
  double8 hi = _cl_ldexp (x.hi, k.hi);
  return (double16) (lo, hi);
}

#endif /* cl_khr_fp64 */


#if ENABLE_CONFORMANCE == 1

#ifndef cl_khr_fp64
#error ldexp_fp32 requires cl_khr_fp64
#endif

/* calculate the fp32 ldexp values using the fp64 ldexp. This is necessary to
   pass the Full CTS math test, because some values (involving denormals) are
   not correctly calculated by the fp32 version of ldexp from SLEEF.  */

_CL_OVERLOADABLE
float
_cl_ldexp (float x, int k)
{
  double xd = convert_double (x);
  double ret = _cl_ldexp (xd, k);
  return convert_float (ret);
}

_CL_OVERLOADABLE
float2
_cl_ldexp (float2 x, int2 k)
{
  double2 xd = convert_double2 (x);
  double2 ret = _cl_ldexp (xd, k);
  return convert_float2 (ret);
}

_CL_OVERLOADABLE
float3
_cl_ldexp (float3 x, int3 k)
{
  double3 xd = convert_double3 (x);
  double3 ret = _cl_ldexp (xd, k);
  return convert_float3 (ret);
}

_CL_OVERLOADABLE
float4
_cl_ldexp (float4 x, int4 k)
{
  double4 xd = convert_double4 (x);
  double4 ret = _cl_ldexp (xd, k);
  return convert_float4 (ret);
}

_CL_OVERLOADABLE
float8
_cl_ldexp (float8 x, int8 k)
{
  double8 xd = convert_double8 (x);
  double8 ret = _cl_ldexp (xd, k);
  return convert_float8 (ret);
}

_CL_OVERLOADABLE
float16
_cl_ldexp (float16 x, int16 k)
{
  double16 xd = convert_double16 (x);
  double16 ret = _cl_ldexp (xd, k);
  return convert_float16 (ret);
}

#else /* ENABLE_CONFORMANCE */

_CL_OVERLOADABLE
float
_cl_ldexp (float x, int k)
{
  return Sleef_ldexpf (x, k);
}

_CL_OVERLOADABLE
float2
_cl_ldexp (float2 x, int2 k)
{

  float lo = _cl_ldexp (x.lo, k.lo);
  float hi = _cl_ldexp (x.hi, k.hi);
  return (float2)(lo, hi);
}

_CL_OVERLOADABLE
float4 _cl_ldexp (float4, int4);

_CL_OVERLOADABLE
float3
_cl_ldexp (float3 x, int3 k)
{

  float4 x_3to4 = (float4)(x, (float)0);
  int4 k_3to4 = (int4)(k, (float)0);

  float4 r = _cl_ldexp (x_3to4, k_3to4);
  return r.xyz;
}

_CL_OVERLOADABLE
float4
_cl_ldexp (float4 x, int4 k)
{

#if defined(SLEEF_VEC_128_AVAILABLE)
  return Sleef_ldexpf4 (x, k);
#else

  float2 lo = _cl_ldexp (x.lo, k.lo);
  float2 hi = _cl_ldexp (x.hi, k.hi);
  return (float4)(lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_ldexp (float8 x, int8 k)
{

#if defined(SLEEF_VEC_256_AVAILABLE)
  return Sleef_ldexpf8 (x, k);
#else

  float4 lo = _cl_ldexp (x.lo, k.lo);
  float4 hi = _cl_ldexp (x.hi, k.hi);
  return (float8)(lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_ldexp (float16 x, int16 k)
{

#if defined(SLEEF_VEC_512_AVAILABLE)
  return Sleef_ldexpf16 (x, k);
#else

  float8 lo = _cl_ldexp (x.lo, k.lo);
  float8 hi = _cl_ldexp (x.hi, k.hi);
  return (float16)(lo, hi);

#endif
}

#endif /* ENABLE_CONFORMANCE */
