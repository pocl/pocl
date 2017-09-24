#include "sleef_cl.h"

_CL_OVERLOADABLE
int
_cl_ilogb (float x)
{
  return Sleef_ilogbf (x);
}

_CL_OVERLOADABLE
int2
_cl_ilogb (float2 x)
{

  int lo = _cl_ilogb (x.lo);
  int hi = _cl_ilogb (x.hi);
  return (int2) (lo, hi);
}

_CL_OVERLOADABLE
int4 _cl_ilogb (float4);

_CL_OVERLOADABLE
int3
_cl_ilogb (float3 x)
{

  float4 x_3to4 = (float4) (x, (float)0);

  int4 r = _cl_ilogb (x_3to4);
  return r.xyz;
}

_CL_OVERLOADABLE
int4
_cl_ilogb (float4 x)
{

#if defined(SLEEF_VEC_128_AVAILABLE)
  return Sleef_ilogbf4 (x);
#else

  int2 lo = _cl_ilogb (x.lo);
  int2 hi = _cl_ilogb (x.hi);
  return (int4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
int8
_cl_ilogb (float8 x)
{

#if defined(SLEEF_VEC_256_AVAILABLE)
  return Sleef_ilogbf8 (x);
#else

  int4 lo = _cl_ilogb (x.lo);
  int4 hi = _cl_ilogb (x.hi);
  return (int8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
int16
_cl_ilogb (float16 x)
{

#if defined(SLEEF_VEC_512_AVAILABLE)
  return Sleef_ilogbf16 (x);
#else

  int8 lo = _cl_ilogb (x.lo);
  int8 hi = _cl_ilogb (x.hi);
  return (int16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
int
_cl_ilogb (double x)
{
  return Sleef_ilogb (x);
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
int2
_cl_ilogb (double2 x)
{

#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_ilogbd2 (x);
#else

  int lo = _cl_ilogb (x.lo);
  int hi = _cl_ilogb (x.hi);
  return (int2) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
int4 _cl_ilogb (double4);

_CL_OVERLOADABLE
int3
_cl_ilogb (double3 x)
{

  double4 x_3to4 = (double4) (x, (double)0);

  int4 r = _cl_ilogb (x_3to4);
  return r.xyz;
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
int4
_cl_ilogb (double4 x)
{

#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_ilogbd4 (x);
#else

  int2 lo = _cl_ilogb (x.lo);
  int2 hi = _cl_ilogb (x.hi);
  return (int4) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
int8
_cl_ilogb (double8 x)
{

#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_ilogbd8 (x);
#else

  int4 lo = _cl_ilogb (x.lo);
  int4 hi = _cl_ilogb (x.hi);
  return (int8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
int16
_cl_ilogb (double16 x)
{

  int8 lo = _cl_ilogb (x.lo);
  int8 hi = _cl_ilogb (x.hi);
  return (int16) (lo, hi);
}

#endif /* cl_khr_fp64 */
