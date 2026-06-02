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

/* calculate the fp32 ldexp values using the non-SIMD SLEEF code. This is necessary
   to pass the Full CTS math test, because some values (involving denormals) are
   not correctly calculated by the fp32 SIMD version of ldexp from SLEEF.  */

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
  float s0 = _cl_ldexp (x.s0, k.s0);
  float s1 = _cl_ldexp (x.s1, k.s1);
  float s2 = _cl_ldexp (x.s2, k.s2);
  return (float3)(s0, s1, s2);
}

_CL_OVERLOADABLE
float4
_cl_ldexp (float4 x, int4 k)
{
  float s0 = _cl_ldexp (x.s0, k.s0);
  float s1 = _cl_ldexp (x.s1, k.s1);
  float s2 = _cl_ldexp (x.s2, k.s2);
  float s3 = _cl_ldexp (x.s3, k.s3);
  return (float4)(s0, s1, s2, s3);
}

_CL_OVERLOADABLE
float8
_cl_ldexp (float8 x, int8 k)
{
  float s0 = _cl_ldexp (x.s0, k.s0);
  float s1 = _cl_ldexp (x.s1, k.s1);
  float s2 = _cl_ldexp (x.s2, k.s2);
  float s3 = _cl_ldexp (x.s3, k.s3);
  float s4 = _cl_ldexp (x.s4, k.s4);
  float s5 = _cl_ldexp (x.s5, k.s5);
  float s6 = _cl_ldexp (x.s6, k.s6);
  float s7 = _cl_ldexp (x.s7, k.s7);
  return (float8)(s0, s1, s2, s3, s4, s5, s6, s7);
}

_CL_OVERLOADABLE
float16
_cl_ldexp (float16 x, int16 k)
{
  float8 lo = _cl_ldexp (x.lo, k.lo);
  float8 hi = _cl_ldexp (x.hi, k.hi);
  return (float16)(lo, hi);
}
