// Note: This file has been automatically generated. Do not modify.

#include "pocl-compat.h"

// isnan: ['VF'] -> VJ

// isnan: VF=float
#if defined VECMATHLIB_HAVE_VEC_FLOAT_1
// Implement isnan by calling vecmathlib
int _cl_isnan(float x0)
{
  vecmathlib::realvec<float,1> y0 = bitcast<float,vecmathlib::realvec<float,1> >(x0);
  vecmathlib::realvec<float,1>::boolvec_t r = vecmathlib::isnan(y0);
  return bitcast<vecmathlib::realvec<float,1>::intvec_t,int>(vecmathlib::convert_int(r));
}
#else
// Implement isnan by calling libm
int _cl_isnan(float x0)
{
  vecmathlib::realpseudovec<float,1> y0 = x0;
  vecmathlib::realpseudovec<float,1>::boolvec_t r = isnan(y0);
  return vecmathlib::convert_int(r)[0];
}
#endif

// isnan: VF=float2
#if defined VECMATHLIB_HAVE_VEC_FLOAT_2
// Implement isnan by calling vecmathlib
int2 _cl_isnan(float2 x0)
{
  vecmathlib::realvec<float,2> y0 = bitcast<float2,vecmathlib::realvec<float,2> >(x0);
  vecmathlib::realvec<float,2>::boolvec_t r = vecmathlib::isnan(y0);
  return bitcast<vecmathlib::realvec<float,2>::intvec_t,int2>(-vecmathlib::convert_int(r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement isnan by using a larger vector size
int4 _cl_isnan(float4);
int2 _cl_isnan(float2 x0)
{
  float4 y0 = bitcast<float2,float4>(x0);
  int4 r = _cl_isnan(y0);
  return bitcast<int4,int2>(r);
}
#else
// Implement isnan by splitting into a smaller vector size
int _cl_isnan(float);
int2 _cl_isnan(float2 x0)
{
  pair_float y0 = bitcast<float2,pair_float>(x0);
  pair_int r;
  r.lo = -_cl_isnan(y0.lo);
  r.hi = -_cl_isnan(y0.hi);
  pocl_static_assert(sizeof(pair_int) == sizeof(int2));
  return bitcast<pair_int,int2>(r);
}
#endif

// isnan: VF=float3
#if defined VECMATHLIB_HAVE_VEC_FLOAT_3
// Implement isnan by calling vecmathlib
int3 _cl_isnan(float3 x0)
{
  vecmathlib::realvec<float,3> y0 = bitcast<float3,vecmathlib::realvec<float,3> >(x0);
  vecmathlib::realvec<float,3>::boolvec_t r = vecmathlib::isnan(y0);
  return bitcast<vecmathlib::realvec<float,3>::intvec_t,int3>(-vecmathlib::convert_int(r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement isnan by using a larger vector size
int4 _cl_isnan(float4);
int3 _cl_isnan(float3 x0)
{
  float4 y0 = bitcast<float3,float4>(x0);
  int4 r = _cl_isnan(y0);
  return bitcast<int4,int3>(r);
}
#else
// Implement isnan by splitting into a smaller vector size
int2 _cl_isnan(float2);
int3 _cl_isnan(float3 x0)
{
  pair_float2 y0 = bitcast<float3,pair_float2>(x0);
  pair_int2 r;
  r.lo = _cl_isnan(y0.lo);
  r.hi = _cl_isnan(y0.hi);
  pocl_static_assert(sizeof(pair_int2) == sizeof(int3));
  return bitcast<pair_int2,int3>(r);
}
#endif

// isnan: VF=float4
#if defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement isnan by calling vecmathlib
int4 _cl_isnan(float4 x0)
{
  vecmathlib::realvec<float,4> y0 = bitcast<float4,vecmathlib::realvec<float,4> >(x0);
  vecmathlib::realvec<float,4>::boolvec_t r = vecmathlib::isnan(y0);
  return bitcast<vecmathlib::realvec<float,4>::intvec_t,int4>(-vecmathlib::convert_int(r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_8
// Implement isnan by using a larger vector size
int8 _cl_isnan(float8);
int4 _cl_isnan(float4 x0)
{
  float8 y0 = bitcast<float4,float8>(x0);
  int8 r = _cl_isnan(y0);
  return bitcast<int8,int4>(r);
}
#else
// Implement isnan by splitting into a smaller vector size
int2 _cl_isnan(float2);
int4 _cl_isnan(float4 x0)
{
  pair_float2 y0 = bitcast<float4,pair_float2>(x0);
  pair_int2 r;
  r.lo = _cl_isnan(y0.lo);
  r.hi = _cl_isnan(y0.hi);
  pocl_static_assert(sizeof(pair_int2) == sizeof(int4));
  return bitcast<pair_int2,int4>(r);
}
#endif

// isnan: VF=float8
#if defined VECMATHLIB_HAVE_VEC_FLOAT_8
// Implement isnan by calling vecmathlib
int8 _cl_isnan(float8 x0)
{
  vecmathlib::realvec<float,8> y0 = bitcast<float8,vecmathlib::realvec<float,8> >(x0);
  vecmathlib::realvec<float,8>::boolvec_t r = vecmathlib::isnan(y0);
  return bitcast<vecmathlib::realvec<float,8>::intvec_t,int8>(-vecmathlib::convert_int(r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_16
// Implement isnan by using a larger vector size
int16 _cl_isnan(float16);
int8 _cl_isnan(float8 x0)
{
  float16 y0 = bitcast<float8,float16>(x0);
  int16 r = _cl_isnan(y0);
  return bitcast<int16,int8>(r);
}
#else
// Implement isnan by splitting into a smaller vector size
int4 _cl_isnan(float4);
int8 _cl_isnan(float8 x0)
{
  pair_float4 y0 = bitcast<float8,pair_float4>(x0);
  pair_int4 r;
  r.lo = _cl_isnan(y0.lo);
  r.hi = _cl_isnan(y0.hi);
  pocl_static_assert(sizeof(pair_int4) == sizeof(int8));
  return bitcast<pair_int4,int8>(r);
}
#endif

// isnan: VF=float16
#if defined VECMATHLIB_HAVE_VEC_FLOAT_16
// Implement isnan by calling vecmathlib
int16 _cl_isnan(float16 x0)
{
  vecmathlib::realvec<float,16> y0 = bitcast<float16,vecmathlib::realvec<float,16> >(x0);
  vecmathlib::realvec<float,16>::boolvec_t r = vecmathlib::isnan(y0);
  return bitcast<vecmathlib::realvec<float,16>::intvec_t,int16>(-vecmathlib::convert_int(r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_32
// Implement isnan by using a larger vector size
int32 _cl_isnan(float32);
int16 _cl_isnan(float16 x0)
{
  float32 y0 = bitcast<float16,float32>(x0);
  int32 r = _cl_isnan(y0);
  return bitcast<int32,int16>(r);
}
#else
// Implement isnan by splitting into a smaller vector size
int8 _cl_isnan(float8);
int16 _cl_isnan(float16 x0)
{
  pair_float8 y0 = bitcast<float16,pair_float8>(x0);
  pair_int8 r;
  r.lo = _cl_isnan(y0.lo);
  r.hi = _cl_isnan(y0.hi);
  pocl_static_assert(sizeof(pair_int8) == sizeof(int16));
  return bitcast<pair_int8,int16>(r);
}
#endif

#ifdef cl_khr_fp64

// isnan: VF=double
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_1
// Implement isnan by calling vecmathlib
int _cl_isnan(double x0)
{
  vecmathlib::realvec<double,1> y0 = bitcast<double,vecmathlib::realvec<double,1> >(x0);
  vecmathlib::realvec<double,1>::boolvec_t r = vecmathlib::isnan(y0);
  return bitcast<vecmathlib::realvec<double,1>::intvec_t,long>(vecmathlib::convert_int(r));
}
#else
// Implement isnan by calling libm
int _cl_isnan(double x0)
{
  vecmathlib::realpseudovec<double,1> y0 = x0;
  vecmathlib::realpseudovec<double,1>::boolvec_t r = isnan(y0);
  return vecmathlib::convert_int(r)[0];
}
#endif

// isnan: VF=double2
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_2
// Implement isnan by calling vecmathlib
long2 _cl_isnan(double2 x0)
{
  vecmathlib::realvec<double,2> y0 = bitcast<double2,vecmathlib::realvec<double,2> >(x0);
  vecmathlib::realvec<double,2>::boolvec_t r = vecmathlib::isnan(y0);
  return bitcast<vecmathlib::realvec<double,2>::intvec_t,long2>(-vecmathlib::convert_int(r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement isnan by using a larger vector size
long4 _cl_isnan(double4);
long2 _cl_isnan(double2 x0)
{
  double4 y0 = bitcast<double2,double4>(x0);
  long4 r = _cl_isnan(y0);
  return bitcast<long4,long2>(r);
}
#else
// Implement isnan by splitting into a smaller vector size
int _cl_isnan(double);
long2 _cl_isnan(double2 x0)
{
  pair_double y0 = bitcast<double2,pair_double>(x0);
  pair_long r;
  r.lo = -_cl_isnan(y0.lo);
  r.hi = -_cl_isnan(y0.hi);
  pocl_static_assert(sizeof(pair_long) == sizeof(long2));
  return bitcast<pair_long,long2>(r);
}
#endif

// isnan: VF=double3
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_3
// Implement isnan by calling vecmathlib
long3 _cl_isnan(double3 x0)
{
  vecmathlib::realvec<double,3> y0 = bitcast<double3,vecmathlib::realvec<double,3> >(x0);
  vecmathlib::realvec<double,3>::boolvec_t r = vecmathlib::isnan(y0);
  return bitcast<vecmathlib::realvec<double,3>::intvec_t,long3>(-vecmathlib::convert_int(r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement isnan by using a larger vector size
long4 _cl_isnan(double4);
long3 _cl_isnan(double3 x0)
{
  double4 y0 = bitcast<double3,double4>(x0);
  long4 r = _cl_isnan(y0);
  return bitcast<long4,long3>(r);
}
#else
// Implement isnan by splitting into a smaller vector size
long2 _cl_isnan(double2);
long3 _cl_isnan(double3 x0)
{
  pair_double2 y0 = bitcast<double3,pair_double2>(x0);
  pair_long2 r;
  r.lo = _cl_isnan(y0.lo);
  r.hi = _cl_isnan(y0.hi);
  pocl_static_assert(sizeof(pair_long2) == sizeof(long3));
  return bitcast<pair_long2,long3>(r);
}
#endif

// isnan: VF=double4
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement isnan by calling vecmathlib
long4 _cl_isnan(double4 x0)
{
  vecmathlib::realvec<double,4> y0 = bitcast<double4,vecmathlib::realvec<double,4> >(x0);
  vecmathlib::realvec<double,4>::boolvec_t r = vecmathlib::isnan(y0);
  return bitcast<vecmathlib::realvec<double,4>::intvec_t,long4>(-vecmathlib::convert_int(r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_8
// Implement isnan by using a larger vector size
long8 _cl_isnan(double8);
long4 _cl_isnan(double4 x0)
{
  double8 y0 = bitcast<double4,double8>(x0);
  long8 r = _cl_isnan(y0);
  return bitcast<long8,long4>(r);
}
#else
// Implement isnan by splitting into a smaller vector size
long2 _cl_isnan(double2);
long4 _cl_isnan(double4 x0)
{
  pair_double2 y0 = bitcast<double4,pair_double2>(x0);
  pair_long2 r;
  r.lo = _cl_isnan(y0.lo);
  r.hi = _cl_isnan(y0.hi);
  pocl_static_assert(sizeof(pair_long2) == sizeof(long4));
  return bitcast<pair_long2,long4>(r);
}
#endif

// isnan: VF=double8
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_8
// Implement isnan by calling vecmathlib
long8 _cl_isnan(double8 x0)
{
  vecmathlib::realvec<double,8> y0 = bitcast<double8,vecmathlib::realvec<double,8> >(x0);
  vecmathlib::realvec<double,8>::boolvec_t r = vecmathlib::isnan(y0);
  return bitcast<vecmathlib::realvec<double,8>::intvec_t,long8>(-vecmathlib::convert_int(r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_16
// Implement isnan by using a larger vector size
long16 _cl_isnan(double16);
long8 _cl_isnan(double8 x0)
{
  double16 y0 = bitcast<double8,double16>(x0);
  long16 r = _cl_isnan(y0);
  return bitcast<long16,long8>(r);
}
#else
// Implement isnan by splitting into a smaller vector size
long4 _cl_isnan(double4);
long8 _cl_isnan(double8 x0)
{
  pair_double4 y0 = bitcast<double8,pair_double4>(x0);
  pair_long4 r;
  r.lo = _cl_isnan(y0.lo);
  r.hi = _cl_isnan(y0.hi);
  pocl_static_assert(sizeof(pair_long4) == sizeof(long8));
  return bitcast<pair_long4,long8>(r);
}
#endif

// isnan: VF=double16
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_16
// Implement isnan by calling vecmathlib
long16 _cl_isnan(double16 x0)
{
  vecmathlib::realvec<double,16> y0 = bitcast<double16,vecmathlib::realvec<double,16> >(x0);
  vecmathlib::realvec<double,16>::boolvec_t r = vecmathlib::isnan(y0);
  return bitcast<vecmathlib::realvec<double,16>::intvec_t,long16>(-vecmathlib::convert_int(r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_32
// Implement isnan by using a larger vector size
long32 _cl_isnan(double32);
long16 _cl_isnan(double16 x0)
{
  double32 y0 = bitcast<double16,double32>(x0);
  long32 r = _cl_isnan(y0);
  return bitcast<long32,long16>(r);
}
#else
// Implement isnan by splitting into a smaller vector size
long8 _cl_isnan(double8);
long16 _cl_isnan(double16 x0)
{
  pair_double8 y0 = bitcast<double16,pair_double8>(x0);
  pair_long8 r;
  r.lo = _cl_isnan(y0.lo);
  r.hi = _cl_isnan(y0.hi);
  pocl_static_assert(sizeof(pair_long8) == sizeof(long16));
  return bitcast<pair_long8,long16>(r);
}
#endif

#endif // #ifdef cl_khr_fp64
