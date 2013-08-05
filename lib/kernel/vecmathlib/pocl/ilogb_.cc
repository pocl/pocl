// Note: This file has been automatically generated. Do not modify.

#include "pocl-compat.h"

// ilogb_: ['VF'] -> VJ

// ilogb_: VF=float
#if defined VECMATHLIB_HAVE_VEC_FLOAT_1
// Implement ilogb_ by calling vecmathlib
int _cl_ilogb_(float x0)
{
  vecmathlib::realvec<float,1> y0 = bitcast<float,vecmathlib::realvec<float,1> >(x0);
  vecmathlib::realvec<float,1>::intvec_t r = vecmathlib::ilogb(y0);
  return bitcast<vecmathlib::realvec<float,1>::intvec_t,int>((r));
}
#else
// Implement ilogb_ by calling libm
int _cl_ilogb_(float x0)
{
  vecmathlib::realpseudovec<float,1> y0 = x0;
  vecmathlib::realpseudovec<float,1>::intvec_t r = ilogb(y0);
  return (r)[0];
}
#endif

// ilogb_: VF=float2
#if defined VECMATHLIB_HAVE_VEC_FLOAT_2
// Implement ilogb_ by calling vecmathlib
int2 _cl_ilogb_(float2 x0)
{
  vecmathlib::realvec<float,2> y0 = bitcast<float2,vecmathlib::realvec<float,2> >(x0);
  vecmathlib::realvec<float,2>::intvec_t r = vecmathlib::ilogb(y0);
  return bitcast<vecmathlib::realvec<float,2>::intvec_t,int2>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement ilogb_ by using a larger vector size
int4 _cl_ilogb_(float4);
int2 _cl_ilogb_(float2 x0)
{
  float4 y0 = bitcast<float2,float4>(x0);
  int4 r = _cl_ilogb_(y0);
  return bitcast<int4,int2>(r);
}
#else
// Implement ilogb_ by splitting into a smaller vector size
int _cl_ilogb_(float);
int2 _cl_ilogb_(float2 x0)
{
  pair_float y0 = bitcast<float2,pair_float>(x0);
  pair_int r;
  r.lo = _cl_ilogb_(y0.lo);
  r.hi = _cl_ilogb_(y0.hi);
  pocl_static_assert(sizeof(pair_int) == sizeof(int2));
  return bitcast<pair_int,int2>(r);
}
#endif

// ilogb_: VF=float3
#if defined VECMATHLIB_HAVE_VEC_FLOAT_3
// Implement ilogb_ by calling vecmathlib
int3 _cl_ilogb_(float3 x0)
{
  vecmathlib::realvec<float,3> y0 = bitcast<float3,vecmathlib::realvec<float,3> >(x0);
  vecmathlib::realvec<float,3>::intvec_t r = vecmathlib::ilogb(y0);
  return bitcast<vecmathlib::realvec<float,3>::intvec_t,int3>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement ilogb_ by using a larger vector size
int4 _cl_ilogb_(float4);
int3 _cl_ilogb_(float3 x0)
{
  float4 y0 = bitcast<float3,float4>(x0);
  int4 r = _cl_ilogb_(y0);
  return bitcast<int4,int3>(r);
}
#else
// Implement ilogb_ by splitting into a smaller vector size
int2 _cl_ilogb_(float2);
int3 _cl_ilogb_(float3 x0)
{
  pair_float2 y0 = bitcast<float3,pair_float2>(x0);
  pair_int2 r;
  r.lo = _cl_ilogb_(y0.lo);
  r.hi = _cl_ilogb_(y0.hi);
  pocl_static_assert(sizeof(pair_int2) == sizeof(int3));
  return bitcast<pair_int2,int3>(r);
}
#endif

// ilogb_: VF=float4
#if defined VECMATHLIB_HAVE_VEC_FLOAT_4
// Implement ilogb_ by calling vecmathlib
int4 _cl_ilogb_(float4 x0)
{
  vecmathlib::realvec<float,4> y0 = bitcast<float4,vecmathlib::realvec<float,4> >(x0);
  vecmathlib::realvec<float,4>::intvec_t r = vecmathlib::ilogb(y0);
  return bitcast<vecmathlib::realvec<float,4>::intvec_t,int4>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_8
// Implement ilogb_ by using a larger vector size
int8 _cl_ilogb_(float8);
int4 _cl_ilogb_(float4 x0)
{
  float8 y0 = bitcast<float4,float8>(x0);
  int8 r = _cl_ilogb_(y0);
  return bitcast<int8,int4>(r);
}
#else
// Implement ilogb_ by splitting into a smaller vector size
int2 _cl_ilogb_(float2);
int4 _cl_ilogb_(float4 x0)
{
  pair_float2 y0 = bitcast<float4,pair_float2>(x0);
  pair_int2 r;
  r.lo = _cl_ilogb_(y0.lo);
  r.hi = _cl_ilogb_(y0.hi);
  pocl_static_assert(sizeof(pair_int2) == sizeof(int4));
  return bitcast<pair_int2,int4>(r);
}
#endif

// ilogb_: VF=float8
#if defined VECMATHLIB_HAVE_VEC_FLOAT_8
// Implement ilogb_ by calling vecmathlib
int8 _cl_ilogb_(float8 x0)
{
  vecmathlib::realvec<float,8> y0 = bitcast<float8,vecmathlib::realvec<float,8> >(x0);
  vecmathlib::realvec<float,8>::intvec_t r = vecmathlib::ilogb(y0);
  return bitcast<vecmathlib::realvec<float,8>::intvec_t,int8>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_16
// Implement ilogb_ by using a larger vector size
int16 _cl_ilogb_(float16);
int8 _cl_ilogb_(float8 x0)
{
  float16 y0 = bitcast<float8,float16>(x0);
  int16 r = _cl_ilogb_(y0);
  return bitcast<int16,int8>(r);
}
#else
// Implement ilogb_ by splitting into a smaller vector size
int4 _cl_ilogb_(float4);
int8 _cl_ilogb_(float8 x0)
{
  pair_float4 y0 = bitcast<float8,pair_float4>(x0);
  pair_int4 r;
  r.lo = _cl_ilogb_(y0.lo);
  r.hi = _cl_ilogb_(y0.hi);
  pocl_static_assert(sizeof(pair_int4) == sizeof(int8));
  return bitcast<pair_int4,int8>(r);
}
#endif

// ilogb_: VF=float16
#if defined VECMATHLIB_HAVE_VEC_FLOAT_16
// Implement ilogb_ by calling vecmathlib
int16 _cl_ilogb_(float16 x0)
{
  vecmathlib::realvec<float,16> y0 = bitcast<float16,vecmathlib::realvec<float,16> >(x0);
  vecmathlib::realvec<float,16>::intvec_t r = vecmathlib::ilogb(y0);
  return bitcast<vecmathlib::realvec<float,16>::intvec_t,int16>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_FLOAT_32
// Implement ilogb_ by using a larger vector size
int32 _cl_ilogb_(float32);
int16 _cl_ilogb_(float16 x0)
{
  float32 y0 = bitcast<float16,float32>(x0);
  int32 r = _cl_ilogb_(y0);
  return bitcast<int32,int16>(r);
}
#else
// Implement ilogb_ by splitting into a smaller vector size
int8 _cl_ilogb_(float8);
int16 _cl_ilogb_(float16 x0)
{
  pair_float8 y0 = bitcast<float16,pair_float8>(x0);
  pair_int8 r;
  r.lo = _cl_ilogb_(y0.lo);
  r.hi = _cl_ilogb_(y0.hi);
  pocl_static_assert(sizeof(pair_int8) == sizeof(int16));
  return bitcast<pair_int8,int16>(r);
}
#endif

#ifdef cl_khr_fp64

// ilogb_: VF=double
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_1
// Implement ilogb_ by calling vecmathlib
int _cl_ilogb_(double x0)
{
  vecmathlib::realvec<double,1> y0 = bitcast<double,vecmathlib::realvec<double,1> >(x0);
  vecmathlib::realvec<double,1>::intvec_t r = vecmathlib::ilogb(y0);
  return bitcast<vecmathlib::realvec<double,1>::intvec_t,long>((r));
}
#else
// Implement ilogb_ by calling libm
int _cl_ilogb_(double x0)
{
  vecmathlib::realpseudovec<double,1> y0 = x0;
  vecmathlib::realpseudovec<double,1>::intvec_t r = ilogb(y0);
  return (r)[0];
}
#endif

// ilogb_: VF=double2
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_2
// Implement ilogb_ by calling vecmathlib
long2 _cl_ilogb_(double2 x0)
{
  vecmathlib::realvec<double,2> y0 = bitcast<double2,vecmathlib::realvec<double,2> >(x0);
  vecmathlib::realvec<double,2>::intvec_t r = vecmathlib::ilogb(y0);
  return bitcast<vecmathlib::realvec<double,2>::intvec_t,long2>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement ilogb_ by using a larger vector size
long4 _cl_ilogb_(double4);
long2 _cl_ilogb_(double2 x0)
{
  double4 y0 = bitcast<double2,double4>(x0);
  long4 r = _cl_ilogb_(y0);
  return bitcast<long4,long2>(r);
}
#else
// Implement ilogb_ by splitting into a smaller vector size
int _cl_ilogb_(double);
long2 _cl_ilogb_(double2 x0)
{
  pair_double y0 = bitcast<double2,pair_double>(x0);
  pair_long r;
  r.lo = _cl_ilogb_(y0.lo);
  r.hi = _cl_ilogb_(y0.hi);
  pocl_static_assert(sizeof(pair_long) == sizeof(long2));
  return bitcast<pair_long,long2>(r);
}
#endif

// ilogb_: VF=double3
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_3
// Implement ilogb_ by calling vecmathlib
long3 _cl_ilogb_(double3 x0)
{
  vecmathlib::realvec<double,3> y0 = bitcast<double3,vecmathlib::realvec<double,3> >(x0);
  vecmathlib::realvec<double,3>::intvec_t r = vecmathlib::ilogb(y0);
  return bitcast<vecmathlib::realvec<double,3>::intvec_t,long3>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement ilogb_ by using a larger vector size
long4 _cl_ilogb_(double4);
long3 _cl_ilogb_(double3 x0)
{
  double4 y0 = bitcast<double3,double4>(x0);
  long4 r = _cl_ilogb_(y0);
  return bitcast<long4,long3>(r);
}
#else
// Implement ilogb_ by splitting into a smaller vector size
long2 _cl_ilogb_(double2);
long3 _cl_ilogb_(double3 x0)
{
  pair_double2 y0 = bitcast<double3,pair_double2>(x0);
  pair_long2 r;
  r.lo = _cl_ilogb_(y0.lo);
  r.hi = _cl_ilogb_(y0.hi);
  pocl_static_assert(sizeof(pair_long2) == sizeof(long3));
  return bitcast<pair_long2,long3>(r);
}
#endif

// ilogb_: VF=double4
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_4
// Implement ilogb_ by calling vecmathlib
long4 _cl_ilogb_(double4 x0)
{
  vecmathlib::realvec<double,4> y0 = bitcast<double4,vecmathlib::realvec<double,4> >(x0);
  vecmathlib::realvec<double,4>::intvec_t r = vecmathlib::ilogb(y0);
  return bitcast<vecmathlib::realvec<double,4>::intvec_t,long4>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_8
// Implement ilogb_ by using a larger vector size
long8 _cl_ilogb_(double8);
long4 _cl_ilogb_(double4 x0)
{
  double8 y0 = bitcast<double4,double8>(x0);
  long8 r = _cl_ilogb_(y0);
  return bitcast<long8,long4>(r);
}
#else
// Implement ilogb_ by splitting into a smaller vector size
long2 _cl_ilogb_(double2);
long4 _cl_ilogb_(double4 x0)
{
  pair_double2 y0 = bitcast<double4,pair_double2>(x0);
  pair_long2 r;
  r.lo = _cl_ilogb_(y0.lo);
  r.hi = _cl_ilogb_(y0.hi);
  pocl_static_assert(sizeof(pair_long2) == sizeof(long4));
  return bitcast<pair_long2,long4>(r);
}
#endif

// ilogb_: VF=double8
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_8
// Implement ilogb_ by calling vecmathlib
long8 _cl_ilogb_(double8 x0)
{
  vecmathlib::realvec<double,8> y0 = bitcast<double8,vecmathlib::realvec<double,8> >(x0);
  vecmathlib::realvec<double,8>::intvec_t r = vecmathlib::ilogb(y0);
  return bitcast<vecmathlib::realvec<double,8>::intvec_t,long8>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_16
// Implement ilogb_ by using a larger vector size
long16 _cl_ilogb_(double16);
long8 _cl_ilogb_(double8 x0)
{
  double16 y0 = bitcast<double8,double16>(x0);
  long16 r = _cl_ilogb_(y0);
  return bitcast<long16,long8>(r);
}
#else
// Implement ilogb_ by splitting into a smaller vector size
long4 _cl_ilogb_(double4);
long8 _cl_ilogb_(double8 x0)
{
  pair_double4 y0 = bitcast<double8,pair_double4>(x0);
  pair_long4 r;
  r.lo = _cl_ilogb_(y0.lo);
  r.hi = _cl_ilogb_(y0.hi);
  pocl_static_assert(sizeof(pair_long4) == sizeof(long8));
  return bitcast<pair_long4,long8>(r);
}
#endif

// ilogb_: VF=double16
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_16
// Implement ilogb_ by calling vecmathlib
long16 _cl_ilogb_(double16 x0)
{
  vecmathlib::realvec<double,16> y0 = bitcast<double16,vecmathlib::realvec<double,16> >(x0);
  vecmathlib::realvec<double,16>::intvec_t r = vecmathlib::ilogb(y0);
  return bitcast<vecmathlib::realvec<double,16>::intvec_t,long16>((r));
}
#elif defined VECMATHLIB_HAVE_VEC_DOUBLE_32
// Implement ilogb_ by using a larger vector size
long32 _cl_ilogb_(double32);
long16 _cl_ilogb_(double16 x0)
{
  double32 y0 = bitcast<double16,double32>(x0);
  long32 r = _cl_ilogb_(y0);
  return bitcast<long32,long16>(r);
}
#else
// Implement ilogb_ by splitting into a smaller vector size
long8 _cl_ilogb_(double8);
long16 _cl_ilogb_(double16 x0)
{
  pair_double8 y0 = bitcast<double16,pair_double8>(x0);
  pair_long8 r;
  r.lo = _cl_ilogb_(y0.lo);
  r.hi = _cl_ilogb_(y0.hi);
  pocl_static_assert(sizeof(pair_long8) == sizeof(long16));
  return bitcast<pair_long8,long16>(r);
}
#endif

#endif // #ifdef cl_khr_fp64
