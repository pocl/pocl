#include "sleef_cl.h"

_CL_OVERLOADABLE
float
_cl_native_tan (float x)
{
  return Sleef_tanf_u35 (x);
}

_CL_OVERLOADABLE
float2
_cl_native_tan (float2 x)
{

  float lo = _cl_native_tan (x.lo);
  float hi = _cl_native_tan (x.hi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4 _cl_native_tan (float4);

_CL_OVERLOADABLE
float3
_cl_native_tan (float3 x)
{

  float4 x_3to4 = (float4) (x, (float)0);

  float4 r = _cl_native_tan (x_3to4);
  return r.xyz;
}

_CL_OVERLOADABLE
float4
_cl_native_tan (float4 x)
{

#if defined(SLEEF_VEC_128_AVAILABLE)
  return Sleef_tanf4_u35 (x);
#else

  float2 lo = _cl_native_tan (x.lo);
  float2 hi = _cl_native_tan (x.hi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_native_tan (float8 x)
{

#if defined(SLEEF_VEC_256_AVAILABLE)
  return Sleef_tanf8_u35 (x);
#else

  float4 lo = _cl_native_tan (x.lo);
  float4 hi = _cl_native_tan (x.hi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_native_tan (float16 x)
{

#if defined(SLEEF_VEC_512_AVAILABLE)
  return Sleef_tanf16_u35 (x);
#else

  float8 lo = _cl_native_tan (x.lo);
  float8 hi = _cl_native_tan (x.hi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_native_tan (double x)
{
  return Sleef_tan_u35 (x);
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double2
_cl_native_tan (double2 x)
{

#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_tand2_u35 (x);
#else

  double lo = _cl_native_tan (x.lo);
  double hi = _cl_native_tan (x.hi);
  return (double2) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4 _cl_native_tan (double4);

_CL_OVERLOADABLE
double3
_cl_native_tan (double3 x)
{

  double4 x_3to4 = (double4) (x, (double)0);

  double4 r = _cl_native_tan (x_3to4);
  return r.xyz;
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4
_cl_native_tan (double4 x)
{

#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_tand4_u35 (x);
#else

  double2 lo = _cl_native_tan (x.lo);
  double2 hi = _cl_native_tan (x.hi);
  return (double4) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double8
_cl_native_tan (double8 x)
{

#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_tand8_u35 (x);
#else

  double4 lo = _cl_native_tan (x.lo);
  double4 hi = _cl_native_tan (x.hi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double16
_cl_native_tan (double16 x)
{

  double8 lo = _cl_native_tan (x.lo);
  double8 hi = _cl_native_tan (x.hi);
  return (double16) (lo, hi);
}

#endif /* cl_khr_fp64 */
