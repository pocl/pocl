// Note: This file has been automatically generated. Do not modify.

#include "pocl-compat.h"

// isfinite: ['VF'] -> VJ

#ifdef cl_khr_fp16

// isfinite: VF=half
#if defined VECMATHLIB_HAVE_VEC_HALF_1 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
int _cl_isfinite(half x0)
{
  vecmathlib::realvec<half,1> y0 = bitcast<half,vecmathlib::realvec<half,1> >(x0);
  vecmathlib::realvec<half,1>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<half,1>::intvec_t,short>(vecmathlib::convert_int(r));
}
#elif ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling libm
int _cl_isfinite(half x0)
{
  vecmathlib::realpseudovec<half,1> y0 = x0;
  vecmathlib::realpseudovec<half,1>::boolvec_t r = isfinite(y0);
  return vecmathlib::convert_int(r)[0];
}
#else
// Implement isfinite by calling builtin
int _cl_isfinite(half x0)
{
  vecmathlib::realbuiltinvec<half,1> y0 = x0;
  vecmathlib::realbuiltinvec<half,1>::boolvec_t r = isfinite(y0);
  return vecmathlib::convert_int(r)[0];
}
#endif

// isfinite: VF=half2
#if defined VECMATHLIB_HAVE_VEC_HALF_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
short2 _cl_isfinite(half2 x0)
{
  vecmathlib::realvec<half,2> y0 = bitcast<half2,vecmathlib::realvec<half,2> >(x0);
  vecmathlib::realvec<half,2>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<half,2>::intvec_t,short2>(-vecmathlib::convert_int(r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_4 || defined VECMATHLIB_HAVE_VEC_HALF_8 || defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement isfinite by using a larger vector size
short4 _cl_isfinite(half4);
short2 _cl_isfinite(half2 x0)
{
  half4 y0 = bitcast<half2,half4>(x0);
  short4 r = _cl_isfinite(y0);
  return bitcast<short4,short2>(r);
}
#else
// Implement isfinite by splitting into a smaller vector size
int _cl_isfinite(half);
short2 _cl_isfinite(half2 x0)
{
  pair_half y0 = bitcast<half2,pair_half>(x0);
  pair_short r;
  r.lo = -_cl_isfinite(y0.lo);
  r.hi = -_cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_short) == sizeof(short2));
  return bitcast<pair_short,short2>(r);
}
#endif

// isfinite: VF=half3
#if defined VECMATHLIB_HAVE_VEC_HALF_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
short3 _cl_isfinite(half3 x0)
{
  vecmathlib::realvec<half,3> y0 = bitcast<half3,vecmathlib::realvec<half,3> >(x0);
  vecmathlib::realvec<half,3>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<half,3>::intvec_t,short3>(-vecmathlib::convert_int(r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_4 || defined VECMATHLIB_HAVE_VEC_HALF_8 || defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement isfinite by using a larger vector size
short4 _cl_isfinite(half4);
short3 _cl_isfinite(half3 x0)
{
  half4 y0 = bitcast<half3,half4>(x0);
  short4 r = _cl_isfinite(y0);
  return bitcast<short4,short3>(r);
}
#else
// Implement isfinite by splitting into a smaller vector size
short2 _cl_isfinite(half2);
short3 _cl_isfinite(half3 x0)
{
  pair_half2 y0 = bitcast<half3,pair_half2>(x0);
  pair_short2 r;
  r.lo = _cl_isfinite(y0.lo);
  r.hi = _cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_short2) == sizeof(short3));
  return bitcast<pair_short2,short3>(r);
}
#endif

// isfinite: VF=half4
#if defined VECMATHLIB_HAVE_VEC_HALF_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
short4 _cl_isfinite(half4 x0)
{
  vecmathlib::realvec<half,4> y0 = bitcast<half4,vecmathlib::realvec<half,4> >(x0);
  vecmathlib::realvec<half,4>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<half,4>::intvec_t,short4>(-vecmathlib::convert_int(r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_8 || defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement isfinite by using a larger vector size
short8 _cl_isfinite(half8);
short4 _cl_isfinite(half4 x0)
{
  half8 y0 = bitcast<half4,half8>(x0);
  short8 r = _cl_isfinite(y0);
  return bitcast<short8,short4>(r);
}
#else
// Implement isfinite by splitting into a smaller vector size
short2 _cl_isfinite(half2);
short4 _cl_isfinite(half4 x0)
{
  pair_half2 y0 = bitcast<half4,pair_half2>(x0);
  pair_short2 r;
  r.lo = _cl_isfinite(y0.lo);
  r.hi = _cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_short2) == sizeof(short4));
  return bitcast<pair_short2,short4>(r);
}
#endif

// isfinite: VF=half8
#if defined VECMATHLIB_HAVE_VEC_HALF_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
short8 _cl_isfinite(half8 x0)
{
  vecmathlib::realvec<half,8> y0 = bitcast<half8,vecmathlib::realvec<half,8> >(x0);
  vecmathlib::realvec<half,8>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<half,8>::intvec_t,short8>(-vecmathlib::convert_int(r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement isfinite by using a larger vector size
short16 _cl_isfinite(half16);
short8 _cl_isfinite(half8 x0)
{
  half16 y0 = bitcast<half8,half16>(x0);
  short16 r = _cl_isfinite(y0);
  return bitcast<short16,short8>(r);
}
#else
// Implement isfinite by splitting into a smaller vector size
short4 _cl_isfinite(half4);
short8 _cl_isfinite(half8 x0)
{
  pair_half4 y0 = bitcast<half8,pair_half4>(x0);
  pair_short4 r;
  r.lo = _cl_isfinite(y0.lo);
  r.hi = _cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_short4) == sizeof(short8));
  return bitcast<pair_short4,short8>(r);
}
#endif

// isfinite: VF=half16
#if defined VECMATHLIB_HAVE_VEC_HALF_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
short16 _cl_isfinite(half16 x0)
{
  vecmathlib::realvec<half,16> y0 = bitcast<half16,vecmathlib::realvec<half,16> >(x0);
  vecmathlib::realvec<half,16>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<half,16>::intvec_t,short16>(-vecmathlib::convert_int(r));
}
#else
// Implement isfinite by splitting into a smaller vector size
short8 _cl_isfinite(half8);
short16 _cl_isfinite(half16 x0)
{
  pair_half8 y0 = bitcast<half16,pair_half8>(x0);
  pair_short8 r;
  r.lo = _cl_isfinite(y0.lo);
  r.hi = _cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_short8) == sizeof(short16));
  return bitcast<pair_short8,short16>(r);
}
#endif

#endif // #ifdef cl_khr_fp16

// isfinite: VF=float
#if defined VECMATHLIB_HAVE_VEC_FLOAT_1 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
int _cl_isfinite(float x0)
{
  vecmathlib::realvec<float,1> y0 = bitcast<float,vecmathlib::realvec<float,1> >(x0);
  vecmathlib::realvec<float,1>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<float,1>::intvec_t,int>(vecmathlib::convert_int(r));
}
#elif ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling libm
int _cl_isfinite(float x0)
{
  vecmathlib::realpseudovec<float,1> y0 = x0;
  vecmathlib::realpseudovec<float,1>::boolvec_t r = isfinite(y0);
  return vecmathlib::convert_int(r)[0];
}
#else
// Implement isfinite by calling builtin
int _cl_isfinite(float x0)
{
  vecmathlib::realbuiltinvec<float,1> y0 = x0;
  vecmathlib::realbuiltinvec<float,1>::boolvec_t r = isfinite(y0);
  return vecmathlib::convert_int(r)[0];
}
#endif

// isfinite: VF=float2
#if defined VECMATHLIB_HAVE_VEC_FLOAT_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
int2 _cl_isfinite(float2 x0)
{
  vecmathlib::realvec<float,2> y0 = bitcast<float2,vecmathlib::realvec<float,2> >(x0);
  vecmathlib::realvec<float,2>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<float,2>::intvec_t,int2>(-vecmathlib::convert_int(r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_4 || defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement isfinite by using a larger vector size
int4 _cl_isfinite(float4);
int2 _cl_isfinite(float2 x0)
{
  float4 y0 = bitcast<float2,float4>(x0);
  int4 r = _cl_isfinite(y0);
  return bitcast<int4,int2>(r);
}
#else
// Implement isfinite by splitting into a smaller vector size
int _cl_isfinite(float);
int2 _cl_isfinite(float2 x0)
{
  pair_float y0 = bitcast<float2,pair_float>(x0);
  pair_int r;
  r.lo = -_cl_isfinite(y0.lo);
  r.hi = -_cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_int) == sizeof(int2));
  return bitcast<pair_int,int2>(r);
}
#endif

// isfinite: VF=float3
#if defined VECMATHLIB_HAVE_VEC_FLOAT_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
int3 _cl_isfinite(float3 x0)
{
  vecmathlib::realvec<float,3> y0 = bitcast<float3,vecmathlib::realvec<float,3> >(x0);
  vecmathlib::realvec<float,3>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<float,3>::intvec_t,int3>(-vecmathlib::convert_int(r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_4 || defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement isfinite by using a larger vector size
int4 _cl_isfinite(float4);
int3 _cl_isfinite(float3 x0)
{
  float4 y0 = bitcast<float3,float4>(x0);
  int4 r = _cl_isfinite(y0);
  return bitcast<int4,int3>(r);
}
#else
// Implement isfinite by splitting into a smaller vector size
int2 _cl_isfinite(float2);
int3 _cl_isfinite(float3 x0)
{
  pair_float2 y0 = bitcast<float3,pair_float2>(x0);
  pair_int2 r;
  r.lo = _cl_isfinite(y0.lo);
  r.hi = _cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_int2) == sizeof(int3));
  return bitcast<pair_int2,int3>(r);
}
#endif

// isfinite: VF=float4
#if defined VECMATHLIB_HAVE_VEC_FLOAT_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
int4 _cl_isfinite(float4 x0)
{
  vecmathlib::realvec<float,4> y0 = bitcast<float4,vecmathlib::realvec<float,4> >(x0);
  vecmathlib::realvec<float,4>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<float,4>::intvec_t,int4>(-vecmathlib::convert_int(r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement isfinite by using a larger vector size
int8 _cl_isfinite(float8);
int4 _cl_isfinite(float4 x0)
{
  float8 y0 = bitcast<float4,float8>(x0);
  int8 r = _cl_isfinite(y0);
  return bitcast<int8,int4>(r);
}
#else
// Implement isfinite by splitting into a smaller vector size
int2 _cl_isfinite(float2);
int4 _cl_isfinite(float4 x0)
{
  pair_float2 y0 = bitcast<float4,pair_float2>(x0);
  pair_int2 r;
  r.lo = _cl_isfinite(y0.lo);
  r.hi = _cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_int2) == sizeof(int4));
  return bitcast<pair_int2,int4>(r);
}
#endif

// isfinite: VF=float8
#if defined VECMATHLIB_HAVE_VEC_FLOAT_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
int8 _cl_isfinite(float8 x0)
{
  vecmathlib::realvec<float,8> y0 = bitcast<float8,vecmathlib::realvec<float,8> >(x0);
  vecmathlib::realvec<float,8>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<float,8>::intvec_t,int8>(-vecmathlib::convert_int(r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement isfinite by using a larger vector size
int16 _cl_isfinite(float16);
int8 _cl_isfinite(float8 x0)
{
  float16 y0 = bitcast<float8,float16>(x0);
  int16 r = _cl_isfinite(y0);
  return bitcast<int16,int8>(r);
}
#else
// Implement isfinite by splitting into a smaller vector size
int4 _cl_isfinite(float4);
int8 _cl_isfinite(float8 x0)
{
  pair_float4 y0 = bitcast<float8,pair_float4>(x0);
  pair_int4 r;
  r.lo = _cl_isfinite(y0.lo);
  r.hi = _cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_int4) == sizeof(int8));
  return bitcast<pair_int4,int8>(r);
}
#endif

// isfinite: VF=float16
#if defined VECMATHLIB_HAVE_VEC_FLOAT_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
int16 _cl_isfinite(float16 x0)
{
  vecmathlib::realvec<float,16> y0 = bitcast<float16,vecmathlib::realvec<float,16> >(x0);
  vecmathlib::realvec<float,16>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<float,16>::intvec_t,int16>(-vecmathlib::convert_int(r));
}
#else
// Implement isfinite by splitting into a smaller vector size
int8 _cl_isfinite(float8);
int16 _cl_isfinite(float16 x0)
{
  pair_float8 y0 = bitcast<float16,pair_float8>(x0);
  pair_int8 r;
  r.lo = _cl_isfinite(y0.lo);
  r.hi = _cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_int8) == sizeof(int16));
  return bitcast<pair_int8,int16>(r);
}
#endif

#ifdef cl_khr_fp64

// isfinite: VF=double
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_1 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
int _cl_isfinite(double x0)
{
  vecmathlib::realvec<double,1> y0 = bitcast<double,vecmathlib::realvec<double,1> >(x0);
  vecmathlib::realvec<double,1>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<double,1>::intvec_t,long>(vecmathlib::convert_int(r));
}
#elif ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling libm
int _cl_isfinite(double x0)
{
  vecmathlib::realpseudovec<double,1> y0 = x0;
  vecmathlib::realpseudovec<double,1>::boolvec_t r = isfinite(y0);
  return vecmathlib::convert_int(r)[0];
}
#else
// Implement isfinite by calling builtin
int _cl_isfinite(double x0)
{
  vecmathlib::realbuiltinvec<double,1> y0 = x0;
  vecmathlib::realbuiltinvec<double,1>::boolvec_t r = isfinite(y0);
  return vecmathlib::convert_int(r)[0];
}
#endif

// isfinite: VF=double2
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
long2 _cl_isfinite(double2 x0)
{
  vecmathlib::realvec<double,2> y0 = bitcast<double2,vecmathlib::realvec<double,2> >(x0);
  vecmathlib::realvec<double,2>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<double,2>::intvec_t,long2>(-vecmathlib::convert_int(r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_4 || defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement isfinite by using a larger vector size
long4 _cl_isfinite(double4);
long2 _cl_isfinite(double2 x0)
{
  double4 y0 = bitcast<double2,double4>(x0);
  long4 r = _cl_isfinite(y0);
  return bitcast<long4,long2>(r);
}
#else
// Implement isfinite by splitting into a smaller vector size
int _cl_isfinite(double);
long2 _cl_isfinite(double2 x0)
{
  pair_double y0 = bitcast<double2,pair_double>(x0);
  pair_long r;
  r.lo = -_cl_isfinite(y0.lo);
  r.hi = -_cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_long) == sizeof(long2));
  return bitcast<pair_long,long2>(r);
}
#endif

// isfinite: VF=double3
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
long3 _cl_isfinite(double3 x0)
{
  vecmathlib::realvec<double,3> y0 = bitcast<double3,vecmathlib::realvec<double,3> >(x0);
  vecmathlib::realvec<double,3>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<double,3>::intvec_t,long3>(-vecmathlib::convert_int(r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_4 || defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement isfinite by using a larger vector size
long4 _cl_isfinite(double4);
long3 _cl_isfinite(double3 x0)
{
  double4 y0 = bitcast<double3,double4>(x0);
  long4 r = _cl_isfinite(y0);
  return bitcast<long4,long3>(r);
}
#else
// Implement isfinite by splitting into a smaller vector size
long2 _cl_isfinite(double2);
long3 _cl_isfinite(double3 x0)
{
  pair_double2 y0 = bitcast<double3,pair_double2>(x0);
  pair_long2 r;
  r.lo = _cl_isfinite(y0.lo);
  r.hi = _cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_long2) == sizeof(long3));
  return bitcast<pair_long2,long3>(r);
}
#endif

// isfinite: VF=double4
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
long4 _cl_isfinite(double4 x0)
{
  vecmathlib::realvec<double,4> y0 = bitcast<double4,vecmathlib::realvec<double,4> >(x0);
  vecmathlib::realvec<double,4>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<double,4>::intvec_t,long4>(-vecmathlib::convert_int(r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement isfinite by using a larger vector size
long8 _cl_isfinite(double8);
long4 _cl_isfinite(double4 x0)
{
  double8 y0 = bitcast<double4,double8>(x0);
  long8 r = _cl_isfinite(y0);
  return bitcast<long8,long4>(r);
}
#else
// Implement isfinite by splitting into a smaller vector size
long2 _cl_isfinite(double2);
long4 _cl_isfinite(double4 x0)
{
  pair_double2 y0 = bitcast<double4,pair_double2>(x0);
  pair_long2 r;
  r.lo = _cl_isfinite(y0.lo);
  r.hi = _cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_long2) == sizeof(long4));
  return bitcast<pair_long2,long4>(r);
}
#endif

// isfinite: VF=double8
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
long8 _cl_isfinite(double8 x0)
{
  vecmathlib::realvec<double,8> y0 = bitcast<double8,vecmathlib::realvec<double,8> >(x0);
  vecmathlib::realvec<double,8>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<double,8>::intvec_t,long8>(-vecmathlib::convert_int(r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement isfinite by using a larger vector size
long16 _cl_isfinite(double16);
long8 _cl_isfinite(double8 x0)
{
  double16 y0 = bitcast<double8,double16>(x0);
  long16 r = _cl_isfinite(y0);
  return bitcast<long16,long8>(r);
}
#else
// Implement isfinite by splitting into a smaller vector size
long4 _cl_isfinite(double4);
long8 _cl_isfinite(double8 x0)
{
  pair_double4 y0 = bitcast<double8,pair_double4>(x0);
  pair_long4 r;
  r.lo = _cl_isfinite(y0.lo);
  r.hi = _cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_long4) == sizeof(long8));
  return bitcast<pair_long4,long8>(r);
}
#endif

// isfinite: VF=double16
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement isfinite by calling vecmathlib
long16 _cl_isfinite(double16 x0)
{
  vecmathlib::realvec<double,16> y0 = bitcast<double16,vecmathlib::realvec<double,16> >(x0);
  vecmathlib::realvec<double,16>::boolvec_t r = vecmathlib::isfinite(y0);
  return bitcast<vecmathlib::realvec<double,16>::intvec_t,long16>(-vecmathlib::convert_int(r));
}
#else
// Implement isfinite by splitting into a smaller vector size
long8 _cl_isfinite(double8);
long16 _cl_isfinite(double16 x0)
{
  pair_double8 y0 = bitcast<double16,pair_double8>(x0);
  pair_long8 r;
  r.lo = _cl_isfinite(y0.lo);
  r.hi = _cl_isfinite(y0.hi);
  pocl_static_assert(sizeof(pair_long8) == sizeof(long16));
  return bitcast<pair_long8,long16>(r);
}
#endif

#endif // #ifdef cl_khr_fp64
