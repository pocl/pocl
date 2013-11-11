// Note: This file has been automatically generated. Do not modify.

#include "pocl-compat.h"

// fmin: ['VF', 'VF'] -> VF

// fmin: VF=float
#if defined VECMATHLIB_HAVE_VEC_FLOAT_1 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling vecmathlib
float _cl_fmin(float x0, float x1)
{
  vecmathlib::realvec<float,1> y0 = bitcast<float,vecmathlib::realvec<float,1> >(x0);
  vecmathlib::realvec<float,1> y1 = bitcast<float,vecmathlib::realvec<float,1> >(x1);
  vecmathlib::realvec<float,1> r = vecmathlib::fmin(y0, y1);
  return bitcast<vecmathlib::realvec<float,1>,float>((r));
}
#elif ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling libm
float _cl_fmin(float x0, float x1)
{
  vecmathlib::realpseudovec<float,1> y0 = x0;
  vecmathlib::realpseudovec<float,1> y1 = x1;
  vecmathlib::realpseudovec<float,1> r = fmin(y0, y1);
  return (r)[0];
}
#else
// Implement fmin by calling builtin
float _cl_fmin(float x0, float x1)
{
  vecmathlib::realbuiltinvec<float,1> y0 = x0;
  vecmathlib::realbuiltinvec<float,1> y1 = x1;
  vecmathlib::realbuiltinvec<float,1> r = fmin(y0, y1);
  return (r)[0];
}
#endif

// fmin: VF=float2
#if defined VECMATHLIB_HAVE_VEC_FLOAT_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling vecmathlib
float2 _cl_fmin(float2 x0, float2 x1)
{
  vecmathlib::realvec<float,2> y0 = bitcast<float2,vecmathlib::realvec<float,2> >(x0);
  vecmathlib::realvec<float,2> y1 = bitcast<float2,vecmathlib::realvec<float,2> >(x1);
  vecmathlib::realvec<float,2> r = vecmathlib::fmin(y0, y1);
  return bitcast<vecmathlib::realvec<float,2>,float2>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_4 || defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement fmin by using a larger vector size
float4 _cl_fmin(float4, float4);
float2 _cl_fmin(float2 x0, float2 x1)
{
  float4 y0 = bitcast<float2,float4>(x0);
  float4 y1 = bitcast<float2,float4>(x1);
  float4 r = _cl_fmin(y0, y1);
  return bitcast<float4,float2>(r);
}
#else
// Implement fmin by splitting into a smaller vector size
float _cl_fmin(float, float);
float2 _cl_fmin(float2 x0, float2 x1)
{
  pair_float y0 = bitcast<float2,pair_float>(x0);
  pair_float y1 = bitcast<float2,pair_float>(x1);
  pair_float r;
  r.lo = _cl_fmin(y0.lo, y1.lo);
  r.hi = _cl_fmin(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float) == sizeof(float2));
  return bitcast<pair_float,float2>(r);
}
#endif

// fmin: VF=float3
#if defined VECMATHLIB_HAVE_VEC_FLOAT_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling vecmathlib
float3 _cl_fmin(float3 x0, float3 x1)
{
  vecmathlib::realvec<float,3> y0 = bitcast<float3,vecmathlib::realvec<float,3> >(x0);
  vecmathlib::realvec<float,3> y1 = bitcast<float3,vecmathlib::realvec<float,3> >(x1);
  vecmathlib::realvec<float,3> r = vecmathlib::fmin(y0, y1);
  return bitcast<vecmathlib::realvec<float,3>,float3>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_4 || defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement fmin by using a larger vector size
float4 _cl_fmin(float4, float4);
float3 _cl_fmin(float3 x0, float3 x1)
{
  float4 y0 = bitcast<float3,float4>(x0);
  float4 y1 = bitcast<float3,float4>(x1);
  float4 r = _cl_fmin(y0, y1);
  return bitcast<float4,float3>(r);
}
#else
// Implement fmin by splitting into a smaller vector size
float2 _cl_fmin(float2, float2);
float3 _cl_fmin(float3 x0, float3 x1)
{
  pair_float2 y0 = bitcast<float3,pair_float2>(x0);
  pair_float2 y1 = bitcast<float3,pair_float2>(x1);
  pair_float2 r;
  r.lo = _cl_fmin(y0.lo, y1.lo);
  r.hi = _cl_fmin(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float3));
  return bitcast<pair_float2,float3>(r);
}
#endif

// fmin: VF=float4
#if defined VECMATHLIB_HAVE_VEC_FLOAT_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling vecmathlib
float4 _cl_fmin(float4 x0, float4 x1)
{
  vecmathlib::realvec<float,4> y0 = bitcast<float4,vecmathlib::realvec<float,4> >(x0);
  vecmathlib::realvec<float,4> y1 = bitcast<float4,vecmathlib::realvec<float,4> >(x1);
  vecmathlib::realvec<float,4> r = vecmathlib::fmin(y0, y1);
  return bitcast<vecmathlib::realvec<float,4>,float4>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement fmin by using a larger vector size
float8 _cl_fmin(float8, float8);
float4 _cl_fmin(float4 x0, float4 x1)
{
  float8 y0 = bitcast<float4,float8>(x0);
  float8 y1 = bitcast<float4,float8>(x1);
  float8 r = _cl_fmin(y0, y1);
  return bitcast<float8,float4>(r);
}
#else
// Implement fmin by splitting into a smaller vector size
float2 _cl_fmin(float2, float2);
float4 _cl_fmin(float4 x0, float4 x1)
{
  pair_float2 y0 = bitcast<float4,pair_float2>(x0);
  pair_float2 y1 = bitcast<float4,pair_float2>(x1);
  pair_float2 r;
  r.lo = _cl_fmin(y0.lo, y1.lo);
  r.hi = _cl_fmin(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float4));
  return bitcast<pair_float2,float4>(r);
}
#endif

// fmin: VF=float8
#if defined VECMATHLIB_HAVE_VEC_FLOAT_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling vecmathlib
float8 _cl_fmin(float8 x0, float8 x1)
{
  vecmathlib::realvec<float,8> y0 = bitcast<float8,vecmathlib::realvec<float,8> >(x0);
  vecmathlib::realvec<float,8> y1 = bitcast<float8,vecmathlib::realvec<float,8> >(x1);
  vecmathlib::realvec<float,8> r = vecmathlib::fmin(y0, y1);
  return bitcast<vecmathlib::realvec<float,8>,float8>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement fmin by using a larger vector size
float16 _cl_fmin(float16, float16);
float8 _cl_fmin(float8 x0, float8 x1)
{
  float16 y0 = bitcast<float8,float16>(x0);
  float16 y1 = bitcast<float8,float16>(x1);
  float16 r = _cl_fmin(y0, y1);
  return bitcast<float16,float8>(r);
}
#else
// Implement fmin by splitting into a smaller vector size
float4 _cl_fmin(float4, float4);
float8 _cl_fmin(float8 x0, float8 x1)
{
  pair_float4 y0 = bitcast<float8,pair_float4>(x0);
  pair_float4 y1 = bitcast<float8,pair_float4>(x1);
  pair_float4 r;
  r.lo = _cl_fmin(y0.lo, y1.lo);
  r.hi = _cl_fmin(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float4) == sizeof(float8));
  return bitcast<pair_float4,float8>(r);
}
#endif

// fmin: VF=float16
#if defined VECMATHLIB_HAVE_VEC_FLOAT_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling vecmathlib
float16 _cl_fmin(float16 x0, float16 x1)
{
  vecmathlib::realvec<float,16> y0 = bitcast<float16,vecmathlib::realvec<float,16> >(x0);
  vecmathlib::realvec<float,16> y1 = bitcast<float16,vecmathlib::realvec<float,16> >(x1);
  vecmathlib::realvec<float,16> r = vecmathlib::fmin(y0, y1);
  return bitcast<vecmathlib::realvec<float,16>,float16>((r));
}
#else
// Implement fmin by splitting into a smaller vector size
float8 _cl_fmin(float8, float8);
float16 _cl_fmin(float16 x0, float16 x1)
{
  pair_float8 y0 = bitcast<float16,pair_float8>(x0);
  pair_float8 y1 = bitcast<float16,pair_float8>(x1);
  pair_float8 r;
  r.lo = _cl_fmin(y0.lo, y1.lo);
  r.hi = _cl_fmin(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float8) == sizeof(float16));
  return bitcast<pair_float8,float16>(r);
}
#endif

#ifdef cl_khr_fp64

// fmin: VF=double
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_1 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling vecmathlib
double _cl_fmin(double x0, double x1)
{
  vecmathlib::realvec<double,1> y0 = bitcast<double,vecmathlib::realvec<double,1> >(x0);
  vecmathlib::realvec<double,1> y1 = bitcast<double,vecmathlib::realvec<double,1> >(x1);
  vecmathlib::realvec<double,1> r = vecmathlib::fmin(y0, y1);
  return bitcast<vecmathlib::realvec<double,1>,double>((r));
}
#elif ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling libm
double _cl_fmin(double x0, double x1)
{
  vecmathlib::realpseudovec<double,1> y0 = x0;
  vecmathlib::realpseudovec<double,1> y1 = x1;
  vecmathlib::realpseudovec<double,1> r = fmin(y0, y1);
  return (r)[0];
}
#else
// Implement fmin by calling builtin
double _cl_fmin(double x0, double x1)
{
  vecmathlib::realbuiltinvec<double,1> y0 = x0;
  vecmathlib::realbuiltinvec<double,1> y1 = x1;
  vecmathlib::realbuiltinvec<double,1> r = fmin(y0, y1);
  return (r)[0];
}
#endif

// fmin: VF=double2
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling vecmathlib
double2 _cl_fmin(double2 x0, double2 x1)
{
  vecmathlib::realvec<double,2> y0 = bitcast<double2,vecmathlib::realvec<double,2> >(x0);
  vecmathlib::realvec<double,2> y1 = bitcast<double2,vecmathlib::realvec<double,2> >(x1);
  vecmathlib::realvec<double,2> r = vecmathlib::fmin(y0, y1);
  return bitcast<vecmathlib::realvec<double,2>,double2>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_4 || defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement fmin by using a larger vector size
double4 _cl_fmin(double4, double4);
double2 _cl_fmin(double2 x0, double2 x1)
{
  double4 y0 = bitcast<double2,double4>(x0);
  double4 y1 = bitcast<double2,double4>(x1);
  double4 r = _cl_fmin(y0, y1);
  return bitcast<double4,double2>(r);
}
#else
// Implement fmin by splitting into a smaller vector size
double _cl_fmin(double, double);
double2 _cl_fmin(double2 x0, double2 x1)
{
  pair_double y0 = bitcast<double2,pair_double>(x0);
  pair_double y1 = bitcast<double2,pair_double>(x1);
  pair_double r;
  r.lo = _cl_fmin(y0.lo, y1.lo);
  r.hi = _cl_fmin(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double) == sizeof(double2));
  return bitcast<pair_double,double2>(r);
}
#endif

// fmin: VF=double3
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling vecmathlib
double3 _cl_fmin(double3 x0, double3 x1)
{
  vecmathlib::realvec<double,3> y0 = bitcast<double3,vecmathlib::realvec<double,3> >(x0);
  vecmathlib::realvec<double,3> y1 = bitcast<double3,vecmathlib::realvec<double,3> >(x1);
  vecmathlib::realvec<double,3> r = vecmathlib::fmin(y0, y1);
  return bitcast<vecmathlib::realvec<double,3>,double3>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_4 || defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement fmin by using a larger vector size
double4 _cl_fmin(double4, double4);
double3 _cl_fmin(double3 x0, double3 x1)
{
  double4 y0 = bitcast<double3,double4>(x0);
  double4 y1 = bitcast<double3,double4>(x1);
  double4 r = _cl_fmin(y0, y1);
  return bitcast<double4,double3>(r);
}
#else
// Implement fmin by splitting into a smaller vector size
double2 _cl_fmin(double2, double2);
double3 _cl_fmin(double3 x0, double3 x1)
{
  pair_double2 y0 = bitcast<double3,pair_double2>(x0);
  pair_double2 y1 = bitcast<double3,pair_double2>(x1);
  pair_double2 r;
  r.lo = _cl_fmin(y0.lo, y1.lo);
  r.hi = _cl_fmin(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double3));
  return bitcast<pair_double2,double3>(r);
}
#endif

// fmin: VF=double4
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling vecmathlib
double4 _cl_fmin(double4 x0, double4 x1)
{
  vecmathlib::realvec<double,4> y0 = bitcast<double4,vecmathlib::realvec<double,4> >(x0);
  vecmathlib::realvec<double,4> y1 = bitcast<double4,vecmathlib::realvec<double,4> >(x1);
  vecmathlib::realvec<double,4> r = vecmathlib::fmin(y0, y1);
  return bitcast<vecmathlib::realvec<double,4>,double4>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement fmin by using a larger vector size
double8 _cl_fmin(double8, double8);
double4 _cl_fmin(double4 x0, double4 x1)
{
  double8 y0 = bitcast<double4,double8>(x0);
  double8 y1 = bitcast<double4,double8>(x1);
  double8 r = _cl_fmin(y0, y1);
  return bitcast<double8,double4>(r);
}
#else
// Implement fmin by splitting into a smaller vector size
double2 _cl_fmin(double2, double2);
double4 _cl_fmin(double4 x0, double4 x1)
{
  pair_double2 y0 = bitcast<double4,pair_double2>(x0);
  pair_double2 y1 = bitcast<double4,pair_double2>(x1);
  pair_double2 r;
  r.lo = _cl_fmin(y0.lo, y1.lo);
  r.hi = _cl_fmin(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double4));
  return bitcast<pair_double2,double4>(r);
}
#endif

// fmin: VF=double8
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling vecmathlib
double8 _cl_fmin(double8 x0, double8 x1)
{
  vecmathlib::realvec<double,8> y0 = bitcast<double8,vecmathlib::realvec<double,8> >(x0);
  vecmathlib::realvec<double,8> y1 = bitcast<double8,vecmathlib::realvec<double,8> >(x1);
  vecmathlib::realvec<double,8> r = vecmathlib::fmin(y0, y1);
  return bitcast<vecmathlib::realvec<double,8>,double8>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement fmin by using a larger vector size
double16 _cl_fmin(double16, double16);
double8 _cl_fmin(double8 x0, double8 x1)
{
  double16 y0 = bitcast<double8,double16>(x0);
  double16 y1 = bitcast<double8,double16>(x1);
  double16 r = _cl_fmin(y0, y1);
  return bitcast<double16,double8>(r);
}
#else
// Implement fmin by splitting into a smaller vector size
double4 _cl_fmin(double4, double4);
double8 _cl_fmin(double8 x0, double8 x1)
{
  pair_double4 y0 = bitcast<double8,pair_double4>(x0);
  pair_double4 y1 = bitcast<double8,pair_double4>(x1);
  pair_double4 r;
  r.lo = _cl_fmin(y0.lo, y1.lo);
  r.hi = _cl_fmin(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double4) == sizeof(double8));
  return bitcast<pair_double4,double8>(r);
}
#endif

// fmin: VF=double16
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement fmin by calling vecmathlib
double16 _cl_fmin(double16 x0, double16 x1)
{
  vecmathlib::realvec<double,16> y0 = bitcast<double16,vecmathlib::realvec<double,16> >(x0);
  vecmathlib::realvec<double,16> y1 = bitcast<double16,vecmathlib::realvec<double,16> >(x1);
  vecmathlib::realvec<double,16> r = vecmathlib::fmin(y0, y1);
  return bitcast<vecmathlib::realvec<double,16>,double16>((r));
}
#else
// Implement fmin by splitting into a smaller vector size
double8 _cl_fmin(double8, double8);
double16 _cl_fmin(double16 x0, double16 x1)
{
  pair_double8 y0 = bitcast<double16,pair_double8>(x0);
  pair_double8 y1 = bitcast<double16,pair_double8>(x1);
  pair_double8 r;
  r.lo = _cl_fmin(y0.lo, y1.lo);
  r.hi = _cl_fmin(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double8) == sizeof(double16));
  return bitcast<pair_double8,double16>(r);
}
#endif

#endif // #ifdef cl_khr_fp64
