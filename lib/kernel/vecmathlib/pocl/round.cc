// Note: This file has been automatically generated. Do not modify.

#include "pocl-compat.h"

// round: ['VF'] -> VF

// round: VF=float
#if defined VECMATHLIB_HAVE_VEC_FLOAT_1
// Implement round by calling vecmathlib
float _cl_round(float x0)
{
  vecmathlib::realvec<float,1> y0 = bitcast<float,vecmathlib::realvec<float,1> >(x0);
  vecmathlib::realvec<float,1> r = vecmathlib::round(y0);
  return bitcast<vecmathlib::realvec<float,1>,float>((r));
}
#else
// Implement round by calling libm
float _cl_round(float x0)
{
  vecmathlib::realpseudovec<float,1> y0 = x0;
  vecmathlib::realpseudovec<float,1> r = round(y0);
  return (r)[0];
}
#endif

// round: VF=float2
#if defined VECMATHLIB_HAVE_VEC_FLOAT_2
// Implement round by calling vecmathlib
float2 _cl_round(float2 x0)
{
  vecmathlib::realvec<float,2> y0 = bitcast<float2,vecmathlib::realvec<float,2> >(x0);
  vecmathlib::realvec<float,2> r = vecmathlib::round(y0);
  return bitcast<vecmathlib::realvec<float,2>,float2>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement round by using a larger vector size
float4 _cl_round(float4);
float2 _cl_round(float2 x0)
{
  float4 y0 = bitcast<float2,float4>(x0);
  float4 r = _cl_round(y0);
  return bitcast<float4,float2>(r);
}
#else
// Implement round by splitting into a smaller vector size
float _cl_round(float);
float2 _cl_round(float2 x0)
{
  pair_float y0 = bitcast<float2,pair_float>(x0);
  pair_float r;
  r.lo = _cl_round(y0.lo);
  r.hi = _cl_round(y0.hi);
  pocl_static_assert(sizeof(pair_float) == sizeof(float2));
  return bitcast<pair_float,float2>(r);
}
#endif

// round: VF=float3
#if defined VECMATHLIB_HAVE_VEC_FLOAT_3
// Implement round by calling vecmathlib
float3 _cl_round(float3 x0)
{
  vecmathlib::realvec<float,3> y0 = bitcast<float3,vecmathlib::realvec<float,3> >(x0);
  vecmathlib::realvec<float,3> r = vecmathlib::round(y0);
  return bitcast<vecmathlib::realvec<float,3>,float3>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement round by using a larger vector size
float4 _cl_round(float4);
float3 _cl_round(float3 x0)
{
  float4 y0 = bitcast<float3,float4>(x0);
  float4 r = _cl_round(y0);
  return bitcast<float4,float3>(r);
}
#else
// Implement round by splitting into a smaller vector size
float2 _cl_round(float2);
float3 _cl_round(float3 x0)
{
  pair_float2 y0 = bitcast<float3,pair_float2>(x0);
  pair_float2 r;
  r.lo = _cl_round(y0.lo);
  r.hi = _cl_round(y0.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float3));
  return bitcast<pair_float2,float3>(r);
}
#endif

// round: VF=float4
#if defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement round by calling vecmathlib
float4 _cl_round(float4 x0)
{
  vecmathlib::realvec<float,4> y0 = bitcast<float4,vecmathlib::realvec<float,4> >(x0);
  vecmathlib::realvec<float,4> r = vecmathlib::round(y0);
  return bitcast<vecmathlib::realvec<float,4>,float4>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_8
// Implement round by using a larger vector size
float8 _cl_round(float8);
float4 _cl_round(float4 x0)
{
  float8 y0 = bitcast<float4,float8>(x0);
  float8 r = _cl_round(y0);
  return bitcast<float8,float4>(r);
}
#else
// Implement round by splitting into a smaller vector size
float2 _cl_round(float2);
float4 _cl_round(float4 x0)
{
  pair_float2 y0 = bitcast<float4,pair_float2>(x0);
  pair_float2 r;
  r.lo = _cl_round(y0.lo);
  r.hi = _cl_round(y0.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float4));
  return bitcast<pair_float2,float4>(r);
}
#endif

// round: VF=float8
#if defined VECMATHLIB_HAVE_VEC_FLOAT_8
// Implement round by calling vecmathlib
float8 _cl_round(float8 x0)
{
  vecmathlib::realvec<float,8> y0 = bitcast<float8,vecmathlib::realvec<float,8> >(x0);
  vecmathlib::realvec<float,8> r = vecmathlib::round(y0);
  return bitcast<vecmathlib::realvec<float,8>,float8>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_16
// Implement round by using a larger vector size
float16 _cl_round(float16);
float8 _cl_round(float8 x0)
{
  float16 y0 = bitcast<float8,float16>(x0);
  float16 r = _cl_round(y0);
  return bitcast<float16,float8>(r);
}
#else
// Implement round by splitting into a smaller vector size
float4 _cl_round(float4);
float8 _cl_round(float8 x0)
{
  pair_float4 y0 = bitcast<float8,pair_float4>(x0);
  pair_float4 r;
  r.lo = _cl_round(y0.lo);
  r.hi = _cl_round(y0.hi);
  pocl_static_assert(sizeof(pair_float4) == sizeof(float8));
  return bitcast<pair_float4,float8>(r);
}
#endif

// round: VF=float16
#if defined VECMATHLIB_HAVE_VEC_FLOAT_16
// Implement round by calling vecmathlib
float16 _cl_round(float16 x0)
{
  vecmathlib::realvec<float,16> y0 = bitcast<float16,vecmathlib::realvec<float,16> >(x0);
  vecmathlib::realvec<float,16> r = vecmathlib::round(y0);
  return bitcast<vecmathlib::realvec<float,16>,float16>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_32
// Implement round by using a larger vector size
float32 _cl_round(float32);
float16 _cl_round(float16 x0)
{
  float32 y0 = bitcast<float16,float32>(x0);
  float32 r = _cl_round(y0);
  return bitcast<float32,float16>(r);
}
#else
// Implement round by splitting into a smaller vector size
float8 _cl_round(float8);
float16 _cl_round(float16 x0)
{
  pair_float8 y0 = bitcast<float16,pair_float8>(x0);
  pair_float8 r;
  r.lo = _cl_round(y0.lo);
  r.hi = _cl_round(y0.hi);
  pocl_static_assert(sizeof(pair_float8) == sizeof(float16));
  return bitcast<pair_float8,float16>(r);
}
#endif

#ifdef cl_khr_fp64

// round: VF=double
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_1
// Implement round by calling vecmathlib
double _cl_round(double x0)
{
  vecmathlib::realvec<double,1> y0 = bitcast<double,vecmathlib::realvec<double,1> >(x0);
  vecmathlib::realvec<double,1> r = vecmathlib::round(y0);
  return bitcast<vecmathlib::realvec<double,1>,double>((r));
}
#else
// Implement round by calling libm
double _cl_round(double x0)
{
  vecmathlib::realpseudovec<double,1> y0 = x0;
  vecmathlib::realpseudovec<double,1> r = round(y0);
  return (r)[0];
}
#endif

// round: VF=double2
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_2
// Implement round by calling vecmathlib
double2 _cl_round(double2 x0)
{
  vecmathlib::realvec<double,2> y0 = bitcast<double2,vecmathlib::realvec<double,2> >(x0);
  vecmathlib::realvec<double,2> r = vecmathlib::round(y0);
  return bitcast<vecmathlib::realvec<double,2>,double2>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement round by using a larger vector size
double4 _cl_round(double4);
double2 _cl_round(double2 x0)
{
  double4 y0 = bitcast<double2,double4>(x0);
  double4 r = _cl_round(y0);
  return bitcast<double4,double2>(r);
}
#else
// Implement round by splitting into a smaller vector size
double _cl_round(double);
double2 _cl_round(double2 x0)
{
  pair_double y0 = bitcast<double2,pair_double>(x0);
  pair_double r;
  r.lo = _cl_round(y0.lo);
  r.hi = _cl_round(y0.hi);
  pocl_static_assert(sizeof(pair_double) == sizeof(double2));
  return bitcast<pair_double,double2>(r);
}
#endif

// round: VF=double3
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_3
// Implement round by calling vecmathlib
double3 _cl_round(double3 x0)
{
  vecmathlib::realvec<double,3> y0 = bitcast<double3,vecmathlib::realvec<double,3> >(x0);
  vecmathlib::realvec<double,3> r = vecmathlib::round(y0);
  return bitcast<vecmathlib::realvec<double,3>,double3>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement round by using a larger vector size
double4 _cl_round(double4);
double3 _cl_round(double3 x0)
{
  double4 y0 = bitcast<double3,double4>(x0);
  double4 r = _cl_round(y0);
  return bitcast<double4,double3>(r);
}
#else
// Implement round by splitting into a smaller vector size
double2 _cl_round(double2);
double3 _cl_round(double3 x0)
{
  pair_double2 y0 = bitcast<double3,pair_double2>(x0);
  pair_double2 r;
  r.lo = _cl_round(y0.lo);
  r.hi = _cl_round(y0.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double3));
  return bitcast<pair_double2,double3>(r);
}
#endif

// round: VF=double4
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement round by calling vecmathlib
double4 _cl_round(double4 x0)
{
  vecmathlib::realvec<double,4> y0 = bitcast<double4,vecmathlib::realvec<double,4> >(x0);
  vecmathlib::realvec<double,4> r = vecmathlib::round(y0);
  return bitcast<vecmathlib::realvec<double,4>,double4>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_8
// Implement round by using a larger vector size
double8 _cl_round(double8);
double4 _cl_round(double4 x0)
{
  double8 y0 = bitcast<double4,double8>(x0);
  double8 r = _cl_round(y0);
  return bitcast<double8,double4>(r);
}
#else
// Implement round by splitting into a smaller vector size
double2 _cl_round(double2);
double4 _cl_round(double4 x0)
{
  pair_double2 y0 = bitcast<double4,pair_double2>(x0);
  pair_double2 r;
  r.lo = _cl_round(y0.lo);
  r.hi = _cl_round(y0.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double4));
  return bitcast<pair_double2,double4>(r);
}
#endif

// round: VF=double8
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_8
// Implement round by calling vecmathlib
double8 _cl_round(double8 x0)
{
  vecmathlib::realvec<double,8> y0 = bitcast<double8,vecmathlib::realvec<double,8> >(x0);
  vecmathlib::realvec<double,8> r = vecmathlib::round(y0);
  return bitcast<vecmathlib::realvec<double,8>,double8>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_16
// Implement round by using a larger vector size
double16 _cl_round(double16);
double8 _cl_round(double8 x0)
{
  double16 y0 = bitcast<double8,double16>(x0);
  double16 r = _cl_round(y0);
  return bitcast<double16,double8>(r);
}
#else
// Implement round by splitting into a smaller vector size
double4 _cl_round(double4);
double8 _cl_round(double8 x0)
{
  pair_double4 y0 = bitcast<double8,pair_double4>(x0);
  pair_double4 r;
  r.lo = _cl_round(y0.lo);
  r.hi = _cl_round(y0.hi);
  pocl_static_assert(sizeof(pair_double4) == sizeof(double8));
  return bitcast<pair_double4,double8>(r);
}
#endif

// round: VF=double16
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_16
// Implement round by calling vecmathlib
double16 _cl_round(double16 x0)
{
  vecmathlib::realvec<double,16> y0 = bitcast<double16,vecmathlib::realvec<double,16> >(x0);
  vecmathlib::realvec<double,16> r = vecmathlib::round(y0);
  return bitcast<vecmathlib::realvec<double,16>,double16>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_32
// Implement round by using a larger vector size
double32 _cl_round(double32);
double16 _cl_round(double16 x0)
{
  double32 y0 = bitcast<double16,double32>(x0);
  double32 r = _cl_round(y0);
  return bitcast<double32,double16>(r);
}
#else
// Implement round by splitting into a smaller vector size
double8 _cl_round(double8);
double16 _cl_round(double16 x0)
{
  pair_double8 y0 = bitcast<double16,pair_double8>(x0);
  pair_double8 r;
  r.lo = _cl_round(y0.lo);
  r.hi = _cl_round(y0.hi);
  pocl_static_assert(sizeof(pair_double8) == sizeof(double16));
  return bitcast<pair_double8,double16>(r);
}
#endif

#endif // #ifdef cl_khr_fp64
