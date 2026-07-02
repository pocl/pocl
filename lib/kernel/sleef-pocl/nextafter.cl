#include "sleef_cl.h"

_CL_OVERLOADABLE
float
_cl_nextafter (float x, float y)
{
  return Sleef_nextafterf (x, y);
}

_CL_OVERLOADABLE
float2
_cl_nextafter (float2 x, float2 y)
{

  float lo = _cl_nextafter (x.lo, y.lo);
  float hi = _cl_nextafter (x.hi, y.hi);
  return (float2) (lo, hi);
}

_CL_OVERLOADABLE
float4 _cl_nextafter (float4, float4);

_CL_OVERLOADABLE
float3
_cl_nextafter (float3 x, float3 y)
{

  float4 x_3to4 = (float4) (x, (float)0);
  float4 y_3to4 = (float4) (y, (float)0);

  float4 r = _cl_nextafter (x_3to4, y_3to4);
  return r.xyz;
}

_CL_OVERLOADABLE
float4
_cl_nextafter (float4 x, float4 y)
{

#if defined(SLEEF_VEC_128_AVAILABLE)
  return Sleef_nextafterf4 (x, y);
#else

  float2 lo = _cl_nextafter (x.lo, y.lo);
  float2 hi = _cl_nextafter (x.hi, y.hi);
  return (float4) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float8
_cl_nextafter (float8 x, float8 y)
{

#if defined(SLEEF_VEC_256_AVAILABLE)
  return Sleef_nextafterf8 (x, y);
#else

  float4 lo = _cl_nextafter (x.lo, y.lo);
  float4 hi = _cl_nextafter (x.hi, y.hi);
  return (float8) (lo, hi);

#endif
}

_CL_OVERLOADABLE
float16
_cl_nextafter (float16 x, float16 y)
{

#if defined(SLEEF_VEC_512_AVAILABLE)
  return Sleef_nextafterf16 (x, y);
#else

  float8 lo = _cl_nextafter (x.lo, y.lo);
  float8 hi = _cl_nextafter (x.hi, y.hi);
  return (float16) (lo, hi);

#endif
}

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double
_cl_nextafter (double x, double y)
{
  return Sleef_nextafter (x, y);
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double2
_cl_nextafter (double2 x, double2 y)
{

#if defined(SLEEF_VEC_128_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_nextafterd2 (x, y);
#else

  double lo = _cl_nextafter (x.lo, y.lo);
  double hi = _cl_nextafter (x.hi, y.hi);
  return (double2) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4 _cl_nextafter (double4, double4);

_CL_OVERLOADABLE
double3
_cl_nextafter (double3 x, double3 y)
{

  double4 x_3to4 = (double4) (x, (double)0);
  double4 y_3to4 = (double4) (y, (double)0);

  double4 r = _cl_nextafter (x_3to4, y_3to4);
  return r.xyz;
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double4
_cl_nextafter (double4 x, double4 y)
{

#if defined(SLEEF_VEC_256_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_nextafterd4 (x, y);
#else

  double2 lo = _cl_nextafter (x.lo, y.lo);
  double2 hi = _cl_nextafter (x.hi, y.hi);
  return (double4) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double8
_cl_nextafter (double8 x, double8 y)
{

#if defined(SLEEF_VEC_512_AVAILABLE) && defined(SLEEF_DOUBLE_VEC_AVAILABLE)
  return Sleef_nextafterd8 (x, y);
#else

  double4 lo = _cl_nextafter (x.lo, y.lo);
  double4 hi = _cl_nextafter (x.hi, y.hi);
  return (double8) (lo, hi);

#endif
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp64

_CL_OVERLOADABLE
double16
_cl_nextafter (double16 x, double16 y)
{

  double8 lo = _cl_nextafter (x.lo, y.lo);
  double8 hi = _cl_nextafter (x.hi, y.hi);
  return (double16) (lo, hi);
}

#endif /* cl_khr_fp64 */

#ifdef cl_khr_fp16

_CL_OVERLOADABLE
half
_cl_nextafter (half x, half y)
{
  const short sign_bit = as_short((ushort)0x8000);
  const short sign_bit_mask = 0x7fff;

  short ix = as_short(x);
  short ax = ix & sign_bit_mask;
  short mx = sign_bit - ix;
  mx = ix < (short)0 ? mx : ix;

  short iy = as_short(y);
  short ay = iy & sign_bit_mask;
  short my = sign_bit - iy;
  my = iy < (short)0 ? my : iy;

  short t = mx + (mx < my ? 1 : -1);
  short r = sign_bit - t;
  r = t < (short)0 ? r : t;

  if (t == 0 && ix < 0) {
    r = sign_bit;
  }

  r = isnan(x) ? ix : r;
  r = isnan(y) ? iy : r;
  r = (((ax | ay) == (short)0) | (ix == iy)) ? iy : r;
  return as_half(r);
}

_CL_OVERLOADABLE
half2
_cl_nextafter (half2 x, half2 y)
{
  half lo = _cl_nextafter (x.lo, y.lo);
  half hi = _cl_nextafter (x.hi, y.hi);
  return (half2) (lo, hi);
}

_CL_OVERLOADABLE
half4 _cl_nextafter (half4, half4);

_CL_OVERLOADABLE
half3
_cl_nextafter (half3 x, half3 y)
{
  half4 x_3to4 = (half4) (x, (half)0);
  half4 y_3to4 = (half4) (y, (half)0);

  half4 r = _cl_nextafter (x_3to4, y_3to4);
  return r.xyz;
}

_CL_OVERLOADABLE
half4
_cl_nextafter (half4 x, half4 y)
{
  half2 lo = _cl_nextafter (x.lo, y.lo);
  half2 hi = _cl_nextafter (x.hi, y.hi);
  return (half4) (lo, hi);
}

_CL_OVERLOADABLE
half8
_cl_nextafter (half8 x, half8 y)
{
  half4 lo = _cl_nextafter (x.lo, y.lo);
  half4 hi = _cl_nextafter (x.hi, y.hi);
  return (half8) (lo, hi);
}

_CL_OVERLOADABLE
half16
_cl_nextafter (half16 x, half16 y)
{
  half8 lo = _cl_nextafter (x.lo, y.lo);
  half8 hi = _cl_nextafter (x.hi, y.hi);
  return (half16) (lo, hi);
}

#endif /* cl_khr_fp16 */

