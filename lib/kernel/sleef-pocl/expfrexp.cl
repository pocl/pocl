#include "sleef_cl.h"

_CL_OVERLOADABLE
int
_cl_expfrexp (float x)
{
  return Sleef_expfrexpf (x);
}

_CL_OVERLOADABLE
int2
_cl_expfrexp (float2 x)
{

  int lo = _cl_expfrexp (x.lo);
  int hi = _cl_expfrexp (x.hi);
  return (int2) (lo, hi);
}

_CL_OVERLOADABLE
int4 _cl_expfrexp (float4);

_CL_OVERLOADABLE
int3
_cl_expfrexp (float3 x)
{

  float4 x_3to4 = (float4) (x, (float)0);

  int4 r = _cl_expfrexp (x_3to4);
  return r.xyz;
}

_CL_OVERLOADABLE
int4
_cl_expfrexp (float4 x)
{

#if defined(SLEEF_VEC_128_AVAILABLE)
  return Sleef_expfrexpf4 (x);
#else

  int2 lo = _cl_expfrexp (x.lo);
  int2 hi = _cl_expfrexp (x.hi);
  return (int4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
int8
_cl_expfrexp (float8 x)
{

#if defined(SLEEF_VEC_256_AVAILABLE)
  return Sleef_expfrexpf8 (x);
#else

  int4 lo = _cl_expfrexp (x.lo);
  int4 hi = _cl_expfrexp (x.hi);
  return (int8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
int16
_cl_expfrexp (float16 x)
{

#if defined(SLEEF_VEC_512_AVAILABLE)
  return Sleef_expfrexpf16 (x);
#else

  int8 lo = _cl_expfrexp (x.lo);
  int8 hi = _cl_expfrexp (x.hi);
  return (int16) (lo, hi);

#endif
}

/******************************************************************/
/******************************************************************/
/******************************************************************/
/******************************************************************/

#ifdef cl_khr_fp64

_CL_ALWAYSINLINE long2 Sleef_expfrexpd2_long (double2 x);
_CL_ALWAYSINLINE long4 Sleef_expfrexpd4_long (double4 x);
_CL_ALWAYSINLINE long8 Sleef_expfrexpd8_long (double8 x);

_CL_OVERLOADABLE
long
_cl_expfrexp (double x)
{
  return convert_long (Sleef_expfrexp (x));
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
long2
_cl_expfrexp (double2 x)
{

#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_expfrexpd2_long (x);
#else

  long lo = _cl_expfrexp (x.lo);
  long hi = _cl_expfrexp (x.hi);
  return (long2) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
long4 _cl_expfrexp (double4);

_CL_OVERLOADABLE
long3
_cl_expfrexp (double3 x)
{

  double4 x_3to4 = (double4) (x, (double)0);

  long4 r = _cl_expfrexp (x_3to4);
  return r.xyz;
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
long4
_cl_expfrexp (double4 x)
{

#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_expfrexpd4_long (x);
#else

  long2 lo = _cl_expfrexp (x.lo);
  long2 hi = _cl_expfrexp (x.hi);
  return (long4) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
long8
_cl_expfrexp (double8 x)
{

#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_expfrexpd8_long (x);
#else

  long4 lo = _cl_expfrexp (x.lo);
  long4 hi = _cl_expfrexp (x.hi);
  return (long8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
long16
_cl_expfrexp (double16 x)
{

  long8 lo = _cl_expfrexp (x.lo);
  long8 hi = _cl_expfrexp (x.hi);
  return (long16) (lo, hi);
}

#endif /* cl_khr_fp64 */
