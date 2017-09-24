#include "sleef_cl.h"

_CL_OVERLOADABLE
float
_cl_cos (float x)
{

#ifdef MAX_PRECISION
  return Sleef_cosf_u10 (x);
#else
  return Sleef_cosf_u35 (x);
#endif
}

_CL_OVERLOADABLE
float2
_cl_cos (float2 x)
{

  float lo = _cl_cos (x.lo);
  float hi = _cl_cos (x.hi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4 _cl_cos (float4);

_CL_OVERLOADABLE
float3
_cl_cos (float3 x)
{

  float4 x_3to4 = (float4) (x, (float)0);

  float4 r = _cl_cos (x_3to4);
  return r.xyz;
}

_CL_OVERLOADABLE
float4
_cl_cos (float4 x)
{

#if defined(SLEEF_VEC_128_AVAILABLE)

#ifdef MAX_PRECISION
  return Sleef_cosf4_u10 (x);
#else
  return Sleef_cosf4_u35 (x);
#endif

#else

  float2 lo = _cl_cos (x.lo);
  float2 hi = _cl_cos (x.hi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_cos (float8 x)
{

#if defined(SLEEF_VEC_256_AVAILABLE)

#ifdef MAX_PRECISION
  return Sleef_cosf8_u10 (x);
#else
  return Sleef_cosf8_u35 (x);
#endif

#else

  float4 lo = _cl_cos (x.lo);
  float4 hi = _cl_cos (x.hi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_cos (float16 x)
{

#if defined(SLEEF_VEC_512_AVAILABLE)

#ifdef MAX_PRECISION
  return Sleef_cosf16_u10 (x);
#else
  return Sleef_cosf16_u35 (x);
#endif

#else

  float8 lo = _cl_cos (x.lo);
  float8 hi = _cl_cos (x.hi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_cos (double x)
{

#ifdef MAX_PRECISION
  return Sleef_cos_u10 (x);
#else
  return Sleef_cos_u35 (x);
#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double2
_cl_cos (double2 x)
{

#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)

#ifdef MAX_PRECISION
  return Sleef_cosd2_u10 (x);
#else
  return Sleef_cosd2_u35 (x);
#endif

#else

  double lo = _cl_cos (x.lo);
  double hi = _cl_cos (x.hi);
  return (double2) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4 _cl_cos (double4);

_CL_OVERLOADABLE
double3
_cl_cos (double3 x)
{

  double4 x_3to4 = (double4) (x, (double)0);

  double4 r = _cl_cos (x_3to4);
  return r.xyz;
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4
_cl_cos (double4 x)
{

#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)

#ifdef MAX_PRECISION
  return Sleef_cosd4_u10 (x);
#else
  return Sleef_cosd4_u35 (x);
#endif

#else

  double2 lo = _cl_cos (x.lo);
  double2 hi = _cl_cos (x.hi);
  return (double4) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double8
_cl_cos (double8 x)
{

#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)

#ifdef MAX_PRECISION
  return Sleef_cosd8_u10 (x);
#else
  return Sleef_cosd8_u35 (x);
#endif

#else

  double4 lo = _cl_cos (x.lo);
  double4 hi = _cl_cos (x.hi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double16
_cl_cos (double16 x)
{

  double8 lo = _cl_cos (x.lo);
  double8 hi = _cl_cos (x.hi);
  return (double16) (lo, hi);
}

#endif /* cl_khr_fp64 */
