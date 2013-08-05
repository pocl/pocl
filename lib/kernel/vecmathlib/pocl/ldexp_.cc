// Note: This file has been automatically generated. Do not modify.

#include "pocl-compat.h"

// ldexp_: ['VF', 'VJ'] -> VF

// ldexp_: VF=float
#if defined VECMATHLIB_HAVE_VEC_FLOAT_1
// Implement ldexp_ by calling vecmathlib
float _cl_ldexp_(float x0, int x1)
{
  vecmathlib::realvec<float,1> y0 = bitcast<float,vecmathlib::realvec<float,1> >(x0);
  vecmathlib::realvec<float,1>::intvec_t y1 = bitcast<int,vecmathlib::realvec<float,1>::intvec_t >(x1);
  vecmathlib::realvec<float,1> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,1>,float>((r));
}
#else
// Implement ldexp_ by calling libm
float _cl_ldexp_(float x0, int x1)
{
  vecmathlib::realpseudovec<float,1> y0 = x0;
  vecmathlib::realpseudovec<float,1>::intvec_t y1 = x1;
  vecmathlib::realpseudovec<float,1> r = ldexp(y0, y1);
  return (r)[0];
}
#endif

// ldexp_: VF=float2
#if defined VECMATHLIB_HAVE_VEC_FLOAT_2
// Implement ldexp_ by calling vecmathlib
float2 _cl_ldexp_(float2 x0, int2 x1)
{
  vecmathlib::realvec<float,2> y0 = bitcast<float2,vecmathlib::realvec<float,2> >(x0);
  vecmathlib::realvec<float,2>::intvec_t y1 = bitcast<int2,vecmathlib::realvec<float,2>::intvec_t >(x1);
  vecmathlib::realvec<float,2> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,2>,float2>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement ldexp_ by using a larger vector size
float4 _cl_ldexp_(float4, int4);
float2 _cl_ldexp_(float2 x0, int2 x1)
{
  float4 y0 = bitcast<float2,float4>(x0);
  int4 y1 = bitcast<int2,int4>(x1);
  float4 r = _cl_ldexp_(y0, y1);
  return bitcast<float4,float2>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
float _cl_ldexp_(float, int);
float2 _cl_ldexp_(float2 x0, int2 x1)
{
  pair_float y0 = bitcast<float2,pair_float>(x0);
  pair_int y1 = bitcast<int2,pair_int>(x1);
  pair_float r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float) == sizeof(float2));
  return bitcast<pair_float,float2>(r);
}
#endif

// ldexp_: VF=float3
#if defined VECMATHLIB_HAVE_VEC_FLOAT_3
// Implement ldexp_ by calling vecmathlib
float3 _cl_ldexp_(float3 x0, int3 x1)
{
  vecmathlib::realvec<float,3> y0 = bitcast<float3,vecmathlib::realvec<float,3> >(x0);
  vecmathlib::realvec<float,3>::intvec_t y1 = bitcast<int3,vecmathlib::realvec<float,3>::intvec_t >(x1);
  vecmathlib::realvec<float,3> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,3>,float3>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement ldexp_ by using a larger vector size
float4 _cl_ldexp_(float4, int4);
float3 _cl_ldexp_(float3 x0, int3 x1)
{
  float4 y0 = bitcast<float3,float4>(x0);
  int4 y1 = bitcast<int3,int4>(x1);
  float4 r = _cl_ldexp_(y0, y1);
  return bitcast<float4,float3>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
float2 _cl_ldexp_(float2, int2);
float3 _cl_ldexp_(float3 x0, int3 x1)
{
  pair_float2 y0 = bitcast<float3,pair_float2>(x0);
  pair_int2 y1 = bitcast<int3,pair_int2>(x1);
  pair_float2 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float3));
  return bitcast<pair_float2,float3>(r);
}
#endif

// ldexp_: VF=float4
#if defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement ldexp_ by calling vecmathlib
float4 _cl_ldexp_(float4 x0, int4 x1)
{
  vecmathlib::realvec<float,4> y0 = bitcast<float4,vecmathlib::realvec<float,4> >(x0);
  vecmathlib::realvec<float,4>::intvec_t y1 = bitcast<int4,vecmathlib::realvec<float,4>::intvec_t >(x1);
  vecmathlib::realvec<float,4> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,4>,float4>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_8
// Implement ldexp_ by using a larger vector size
float8 _cl_ldexp_(float8, int8);
float4 _cl_ldexp_(float4 x0, int4 x1)
{
  float8 y0 = bitcast<float4,float8>(x0);
  int8 y1 = bitcast<int4,int8>(x1);
  float8 r = _cl_ldexp_(y0, y1);
  return bitcast<float8,float4>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
float2 _cl_ldexp_(float2, int2);
float4 _cl_ldexp_(float4 x0, int4 x1)
{
  pair_float2 y0 = bitcast<float4,pair_float2>(x0);
  pair_int2 y1 = bitcast<int4,pair_int2>(x1);
  pair_float2 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float4));
  return bitcast<pair_float2,float4>(r);
}
#endif

// ldexp_: VF=float8
#if defined VECMATHLIB_HAVE_VEC_FLOAT_8
// Implement ldexp_ by calling vecmathlib
float8 _cl_ldexp_(float8 x0, int8 x1)
{
  vecmathlib::realvec<float,8> y0 = bitcast<float8,vecmathlib::realvec<float,8> >(x0);
  vecmathlib::realvec<float,8>::intvec_t y1 = bitcast<int8,vecmathlib::realvec<float,8>::intvec_t >(x1);
  vecmathlib::realvec<float,8> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,8>,float8>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_16
// Implement ldexp_ by using a larger vector size
float16 _cl_ldexp_(float16, int16);
float8 _cl_ldexp_(float8 x0, int8 x1)
{
  float16 y0 = bitcast<float8,float16>(x0);
  int16 y1 = bitcast<int8,int16>(x1);
  float16 r = _cl_ldexp_(y0, y1);
  return bitcast<float16,float8>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
float4 _cl_ldexp_(float4, int4);
float8 _cl_ldexp_(float8 x0, int8 x1)
{
  pair_float4 y0 = bitcast<float8,pair_float4>(x0);
  pair_int4 y1 = bitcast<int8,pair_int4>(x1);
  pair_float4 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float4) == sizeof(float8));
  return bitcast<pair_float4,float8>(r);
}
#endif

// ldexp_: VF=float16
#if defined VECMATHLIB_HAVE_VEC_FLOAT_16
// Implement ldexp_ by calling vecmathlib
float16 _cl_ldexp_(float16 x0, int16 x1)
{
  vecmathlib::realvec<float,16> y0 = bitcast<float16,vecmathlib::realvec<float,16> >(x0);
  vecmathlib::realvec<float,16>::intvec_t y1 = bitcast<int16,vecmathlib::realvec<float,16>::intvec_t >(x1);
  vecmathlib::realvec<float,16> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,16>,float16>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_32
// Implement ldexp_ by using a larger vector size
float32 _cl_ldexp_(float32, int32);
float16 _cl_ldexp_(float16 x0, int16 x1)
{
  float32 y0 = bitcast<float16,float32>(x0);
  int32 y1 = bitcast<int16,int32>(x1);
  float32 r = _cl_ldexp_(y0, y1);
  return bitcast<float32,float16>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
float8 _cl_ldexp_(float8, int8);
float16 _cl_ldexp_(float16 x0, int16 x1)
{
  pair_float8 y0 = bitcast<float16,pair_float8>(x0);
  pair_int8 y1 = bitcast<int16,pair_int8>(x1);
  pair_float8 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float8) == sizeof(float16));
  return bitcast<pair_float8,float16>(r);
}
#endif

#ifdef cl_khr_fp64

// ldexp_: VF=double
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_1
// Implement ldexp_ by calling vecmathlib
double _cl_ldexp_(double x0, int x1)
{
  vecmathlib::realvec<double,1> y0 = bitcast<double,vecmathlib::realvec<double,1> >(x0);
  vecmathlib::realvec<double,1>::intvec_t y1 = bitcast<int,vecmathlib::realvec<double,1>::intvec_t >(x1);
  vecmathlib::realvec<double,1> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,1>,double>((r));
}
#else
// Implement ldexp_ by calling libm
double _cl_ldexp_(double x0, int x1)
{
  vecmathlib::realpseudovec<double,1> y0 = x0;
  vecmathlib::realpseudovec<double,1>::intvec_t y1 = x1;
  vecmathlib::realpseudovec<double,1> r = ldexp(y0, y1);
  return (r)[0];
}
#endif

// ldexp_: VF=double2
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_2
// Implement ldexp_ by calling vecmathlib
double2 _cl_ldexp_(double2 x0, long2 x1)
{
  vecmathlib::realvec<double,2> y0 = bitcast<double2,vecmathlib::realvec<double,2> >(x0);
  vecmathlib::realvec<double,2>::intvec_t y1 = bitcast<long2,vecmathlib::realvec<double,2>::intvec_t >(x1);
  vecmathlib::realvec<double,2> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,2>,double2>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement ldexp_ by using a larger vector size
double4 _cl_ldexp_(double4, long4);
double2 _cl_ldexp_(double2 x0, long2 x1)
{
  double4 y0 = bitcast<double2,double4>(x0);
  long4 y1 = bitcast<long2,long4>(x1);
  double4 r = _cl_ldexp_(y0, y1);
  return bitcast<double4,double2>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
double _cl_ldexp_(double, int);
double2 _cl_ldexp_(double2 x0, long2 x1)
{
  pair_double y0 = bitcast<double2,pair_double>(x0);
  pair_int y1 = bitcast<long2,pair_int>(x1);
  pair_double r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double) == sizeof(double2));
  return bitcast<pair_double,double2>(r);
}
#endif

// ldexp_: VF=double3
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_3
// Implement ldexp_ by calling vecmathlib
double3 _cl_ldexp_(double3 x0, long3 x1)
{
  vecmathlib::realvec<double,3> y0 = bitcast<double3,vecmathlib::realvec<double,3> >(x0);
  vecmathlib::realvec<double,3>::intvec_t y1 = bitcast<long3,vecmathlib::realvec<double,3>::intvec_t >(x1);
  vecmathlib::realvec<double,3> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,3>,double3>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement ldexp_ by using a larger vector size
double4 _cl_ldexp_(double4, long4);
double3 _cl_ldexp_(double3 x0, long3 x1)
{
  double4 y0 = bitcast<double3,double4>(x0);
  long4 y1 = bitcast<long3,long4>(x1);
  double4 r = _cl_ldexp_(y0, y1);
  return bitcast<double4,double3>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
double2 _cl_ldexp_(double2, long2);
double3 _cl_ldexp_(double3 x0, long3 x1)
{
  pair_double2 y0 = bitcast<double3,pair_double2>(x0);
  pair_long2 y1 = bitcast<long3,pair_long2>(x1);
  pair_double2 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double3));
  return bitcast<pair_double2,double3>(r);
}
#endif

// ldexp_: VF=double4
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement ldexp_ by calling vecmathlib
double4 _cl_ldexp_(double4 x0, long4 x1)
{
  vecmathlib::realvec<double,4> y0 = bitcast<double4,vecmathlib::realvec<double,4> >(x0);
  vecmathlib::realvec<double,4>::intvec_t y1 = bitcast<long4,vecmathlib::realvec<double,4>::intvec_t >(x1);
  vecmathlib::realvec<double,4> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,4>,double4>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_8
// Implement ldexp_ by using a larger vector size
double8 _cl_ldexp_(double8, long8);
double4 _cl_ldexp_(double4 x0, long4 x1)
{
  double8 y0 = bitcast<double4,double8>(x0);
  long8 y1 = bitcast<long4,long8>(x1);
  double8 r = _cl_ldexp_(y0, y1);
  return bitcast<double8,double4>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
double2 _cl_ldexp_(double2, long2);
double4 _cl_ldexp_(double4 x0, long4 x1)
{
  pair_double2 y0 = bitcast<double4,pair_double2>(x0);
  pair_long2 y1 = bitcast<long4,pair_long2>(x1);
  pair_double2 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double4));
  return bitcast<pair_double2,double4>(r);
}
#endif

// ldexp_: VF=double8
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_8
// Implement ldexp_ by calling vecmathlib
double8 _cl_ldexp_(double8 x0, long8 x1)
{
  vecmathlib::realvec<double,8> y0 = bitcast<double8,vecmathlib::realvec<double,8> >(x0);
  vecmathlib::realvec<double,8>::intvec_t y1 = bitcast<long8,vecmathlib::realvec<double,8>::intvec_t >(x1);
  vecmathlib::realvec<double,8> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,8>,double8>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_16
// Implement ldexp_ by using a larger vector size
double16 _cl_ldexp_(double16, long16);
double8 _cl_ldexp_(double8 x0, long8 x1)
{
  double16 y0 = bitcast<double8,double16>(x0);
  long16 y1 = bitcast<long8,long16>(x1);
  double16 r = _cl_ldexp_(y0, y1);
  return bitcast<double16,double8>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
double4 _cl_ldexp_(double4, long4);
double8 _cl_ldexp_(double8 x0, long8 x1)
{
  pair_double4 y0 = bitcast<double8,pair_double4>(x0);
  pair_long4 y1 = bitcast<long8,pair_long4>(x1);
  pair_double4 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double4) == sizeof(double8));
  return bitcast<pair_double4,double8>(r);
}
#endif

// ldexp_: VF=double16
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_16
// Implement ldexp_ by calling vecmathlib
double16 _cl_ldexp_(double16 x0, long16 x1)
{
  vecmathlib::realvec<double,16> y0 = bitcast<double16,vecmathlib::realvec<double,16> >(x0);
  vecmathlib::realvec<double,16>::intvec_t y1 = bitcast<long16,vecmathlib::realvec<double,16>::intvec_t >(x1);
  vecmathlib::realvec<double,16> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,16>,double16>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_32
// Implement ldexp_ by using a larger vector size
double32 _cl_ldexp_(double32, long32);
double16 _cl_ldexp_(double16 x0, long16 x1)
{
  double32 y0 = bitcast<double16,double32>(x0);
  long32 y1 = bitcast<long16,long32>(x1);
  double32 r = _cl_ldexp_(y0, y1);
  return bitcast<double32,double16>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
double8 _cl_ldexp_(double8, long8);
double16 _cl_ldexp_(double16 x0, long16 x1)
{
  pair_double8 y0 = bitcast<double16,pair_double8>(x0);
  pair_long8 y1 = bitcast<long16,pair_long8>(x1);
  pair_double8 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double8) == sizeof(double16));
  return bitcast<pair_double8,double16>(r);
}
#endif

#endif // #ifdef cl_khr_fp64



// ldexp_: ['VF', 'SK'] -> VF

// ldexp_: VF=float2
#if defined VECMATHLIB_HAVE_VEC_FLOAT_2
// Implement ldexp_ by calling vecmathlib
float2 _cl_ldexp_(float2 x0, int x1)
{
  vecmathlib::realvec<float,2> y0 = bitcast<float2,vecmathlib::realvec<float,2> >(x0);
  vecmathlib::realvec<float,2>::int_t y1 = bitcast<int,vecmathlib::realvec<float,2>::int_t >(x1);
  vecmathlib::realvec<float,2> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,2>,float2>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement ldexp_ by using a larger vector size
float4 _cl_ldexp_(float4, int);
float2 _cl_ldexp_(float2 x0, int x1)
{
  float4 y0 = bitcast<float2,float4>(x0);
  int y1 = bitcast<int,int>(x1);
  float4 r = _cl_ldexp_(y0, y1);
  return bitcast<float4,float2>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
float _cl_ldexp_(float, int);
float2 _cl_ldexp_(float2 x0, int x1)
{
  pair_float y0 = bitcast<float2,pair_float>(x0);
  pair_int y1 = bitcast<int,pair_int>(x1);
  pair_float r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float) == sizeof(float2));
  return bitcast<pair_float,float2>(r);
}
#endif

// ldexp_: VF=float3
#if defined VECMATHLIB_HAVE_VEC_FLOAT_3
// Implement ldexp_ by calling vecmathlib
float3 _cl_ldexp_(float3 x0, int x1)
{
  vecmathlib::realvec<float,3> y0 = bitcast<float3,vecmathlib::realvec<float,3> >(x0);
  vecmathlib::realvec<float,3>::int_t y1 = bitcast<int,vecmathlib::realvec<float,3>::int_t >(x1);
  vecmathlib::realvec<float,3> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,3>,float3>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement ldexp_ by using a larger vector size
float4 _cl_ldexp_(float4, int);
float3 _cl_ldexp_(float3 x0, int x1)
{
  float4 y0 = bitcast<float3,float4>(x0);
  int y1 = bitcast<int,int>(x1);
  float4 r = _cl_ldexp_(y0, y1);
  return bitcast<float4,float3>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
float2 _cl_ldexp_(float2, int);
float3 _cl_ldexp_(float3 x0, int x1)
{
  pair_float2 y0 = bitcast<float3,pair_float2>(x0);
  pair_int y1 = bitcast<int,pair_int>(x1);
  pair_float2 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float3));
  return bitcast<pair_float2,float3>(r);
}
#endif

// ldexp_: VF=float4
#if defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement ldexp_ by calling vecmathlib
float4 _cl_ldexp_(float4 x0, int x1)
{
  vecmathlib::realvec<float,4> y0 = bitcast<float4,vecmathlib::realvec<float,4> >(x0);
  vecmathlib::realvec<float,4>::int_t y1 = bitcast<int,vecmathlib::realvec<float,4>::int_t >(x1);
  vecmathlib::realvec<float,4> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,4>,float4>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_8
// Implement ldexp_ by using a larger vector size
float8 _cl_ldexp_(float8, int);
float4 _cl_ldexp_(float4 x0, int x1)
{
  float8 y0 = bitcast<float4,float8>(x0);
  int y1 = bitcast<int,int>(x1);
  float8 r = _cl_ldexp_(y0, y1);
  return bitcast<float8,float4>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
float2 _cl_ldexp_(float2, int);
float4 _cl_ldexp_(float4 x0, int x1)
{
  pair_float2 y0 = bitcast<float4,pair_float2>(x0);
  pair_int y1 = bitcast<int,pair_int>(x1);
  pair_float2 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float4));
  return bitcast<pair_float2,float4>(r);
}
#endif

// ldexp_: VF=float8
#if defined VECMATHLIB_HAVE_VEC_FLOAT_8
// Implement ldexp_ by calling vecmathlib
float8 _cl_ldexp_(float8 x0, int x1)
{
  vecmathlib::realvec<float,8> y0 = bitcast<float8,vecmathlib::realvec<float,8> >(x0);
  vecmathlib::realvec<float,8>::int_t y1 = bitcast<int,vecmathlib::realvec<float,8>::int_t >(x1);
  vecmathlib::realvec<float,8> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,8>,float8>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_16
// Implement ldexp_ by using a larger vector size
float16 _cl_ldexp_(float16, int);
float8 _cl_ldexp_(float8 x0, int x1)
{
  float16 y0 = bitcast<float8,float16>(x0);
  int y1 = bitcast<int,int>(x1);
  float16 r = _cl_ldexp_(y0, y1);
  return bitcast<float16,float8>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
float4 _cl_ldexp_(float4, int);
float8 _cl_ldexp_(float8 x0, int x1)
{
  pair_float4 y0 = bitcast<float8,pair_float4>(x0);
  pair_int y1 = bitcast<int,pair_int>(x1);
  pair_float4 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float4) == sizeof(float8));
  return bitcast<pair_float4,float8>(r);
}
#endif

// ldexp_: VF=float16
#if defined VECMATHLIB_HAVE_VEC_FLOAT_16
// Implement ldexp_ by calling vecmathlib
float16 _cl_ldexp_(float16 x0, int x1)
{
  vecmathlib::realvec<float,16> y0 = bitcast<float16,vecmathlib::realvec<float,16> >(x0);
  vecmathlib::realvec<float,16>::int_t y1 = bitcast<int,vecmathlib::realvec<float,16>::int_t >(x1);
  vecmathlib::realvec<float,16> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,16>,float16>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_32
// Implement ldexp_ by using a larger vector size
float32 _cl_ldexp_(float32, int);
float16 _cl_ldexp_(float16 x0, int x1)
{
  float32 y0 = bitcast<float16,float32>(x0);
  int y1 = bitcast<int,int>(x1);
  float32 r = _cl_ldexp_(y0, y1);
  return bitcast<float32,float16>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
float8 _cl_ldexp_(float8, int);
float16 _cl_ldexp_(float16 x0, int x1)
{
  pair_float8 y0 = bitcast<float16,pair_float8>(x0);
  pair_int y1 = bitcast<int,pair_int>(x1);
  pair_float8 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float8) == sizeof(float16));
  return bitcast<pair_float8,float16>(r);
}
#endif

#ifdef cl_khr_fp64

// ldexp_: VF=double2
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_2
// Implement ldexp_ by calling vecmathlib
double2 _cl_ldexp_(double2 x0, long x1)
{
  vecmathlib::realvec<double,2> y0 = bitcast<double2,vecmathlib::realvec<double,2> >(x0);
  vecmathlib::realvec<double,2>::int_t y1 = bitcast<long,vecmathlib::realvec<double,2>::int_t >(x1);
  vecmathlib::realvec<double,2> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,2>,double2>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement ldexp_ by using a larger vector size
double4 _cl_ldexp_(double4, long);
double2 _cl_ldexp_(double2 x0, long x1)
{
  double4 y0 = bitcast<double2,double4>(x0);
  long y1 = bitcast<long,long>(x1);
  double4 r = _cl_ldexp_(y0, y1);
  return bitcast<double4,double2>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
double _cl_ldexp_(double, int);
double2 _cl_ldexp_(double2 x0, long x1)
{
  pair_double y0 = bitcast<double2,pair_double>(x0);
  pair_int y1 = bitcast<long,pair_int>(x1);
  pair_double r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double) == sizeof(double2));
  return bitcast<pair_double,double2>(r);
}
#endif

// ldexp_: VF=double3
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_3
// Implement ldexp_ by calling vecmathlib
double3 _cl_ldexp_(double3 x0, long x1)
{
  vecmathlib::realvec<double,3> y0 = bitcast<double3,vecmathlib::realvec<double,3> >(x0);
  vecmathlib::realvec<double,3>::int_t y1 = bitcast<long,vecmathlib::realvec<double,3>::int_t >(x1);
  vecmathlib::realvec<double,3> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,3>,double3>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement ldexp_ by using a larger vector size
double4 _cl_ldexp_(double4, long);
double3 _cl_ldexp_(double3 x0, long x1)
{
  double4 y0 = bitcast<double3,double4>(x0);
  long y1 = bitcast<long,long>(x1);
  double4 r = _cl_ldexp_(y0, y1);
  return bitcast<double4,double3>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
double2 _cl_ldexp_(double2, long);
double3 _cl_ldexp_(double3 x0, long x1)
{
  pair_double2 y0 = bitcast<double3,pair_double2>(x0);
  pair_long y1 = bitcast<long,pair_long>(x1);
  pair_double2 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double3));
  return bitcast<pair_double2,double3>(r);
}
#endif

// ldexp_: VF=double4
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement ldexp_ by calling vecmathlib
double4 _cl_ldexp_(double4 x0, long x1)
{
  vecmathlib::realvec<double,4> y0 = bitcast<double4,vecmathlib::realvec<double,4> >(x0);
  vecmathlib::realvec<double,4>::int_t y1 = bitcast<long,vecmathlib::realvec<double,4>::int_t >(x1);
  vecmathlib::realvec<double,4> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,4>,double4>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_8
// Implement ldexp_ by using a larger vector size
double8 _cl_ldexp_(double8, long);
double4 _cl_ldexp_(double4 x0, long x1)
{
  double8 y0 = bitcast<double4,double8>(x0);
  long y1 = bitcast<long,long>(x1);
  double8 r = _cl_ldexp_(y0, y1);
  return bitcast<double8,double4>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
double2 _cl_ldexp_(double2, long);
double4 _cl_ldexp_(double4 x0, long x1)
{
  pair_double2 y0 = bitcast<double4,pair_double2>(x0);
  pair_long y1 = bitcast<long,pair_long>(x1);
  pair_double2 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double4));
  return bitcast<pair_double2,double4>(r);
}
#endif

// ldexp_: VF=double8
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_8
// Implement ldexp_ by calling vecmathlib
double8 _cl_ldexp_(double8 x0, long x1)
{
  vecmathlib::realvec<double,8> y0 = bitcast<double8,vecmathlib::realvec<double,8> >(x0);
  vecmathlib::realvec<double,8>::int_t y1 = bitcast<long,vecmathlib::realvec<double,8>::int_t >(x1);
  vecmathlib::realvec<double,8> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,8>,double8>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_16
// Implement ldexp_ by using a larger vector size
double16 _cl_ldexp_(double16, long);
double8 _cl_ldexp_(double8 x0, long x1)
{
  double16 y0 = bitcast<double8,double16>(x0);
  long y1 = bitcast<long,long>(x1);
  double16 r = _cl_ldexp_(y0, y1);
  return bitcast<double16,double8>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
double4 _cl_ldexp_(double4, long);
double8 _cl_ldexp_(double8 x0, long x1)
{
  pair_double4 y0 = bitcast<double8,pair_double4>(x0);
  pair_long y1 = bitcast<long,pair_long>(x1);
  pair_double4 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double4) == sizeof(double8));
  return bitcast<pair_double4,double8>(r);
}
#endif

// ldexp_: VF=double16
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_16
// Implement ldexp_ by calling vecmathlib
double16 _cl_ldexp_(double16 x0, long x1)
{
  vecmathlib::realvec<double,16> y0 = bitcast<double16,vecmathlib::realvec<double,16> >(x0);
  vecmathlib::realvec<double,16>::int_t y1 = bitcast<long,vecmathlib::realvec<double,16>::int_t >(x1);
  vecmathlib::realvec<double,16> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,16>,double16>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_32
// Implement ldexp_ by using a larger vector size
double32 _cl_ldexp_(double32, long);
double16 _cl_ldexp_(double16 x0, long x1)
{
  double32 y0 = bitcast<double16,double32>(x0);
  long y1 = bitcast<long,long>(x1);
  double32 r = _cl_ldexp_(y0, y1);
  return bitcast<double32,double16>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
double8 _cl_ldexp_(double8, long);
double16 _cl_ldexp_(double16 x0, long x1)
{
  pair_double8 y0 = bitcast<double16,pair_double8>(x0);
  pair_long y1 = bitcast<long,pair_long>(x1);
  pair_double8 r;
  r.lo = _cl_ldexp_(y0.lo, y1.lo);
  r.hi = _cl_ldexp_(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double8) == sizeof(double16));
  return bitcast<pair_double8,double16>(r);
}
#endif

#endif // #ifdef cl_khr_fp64
