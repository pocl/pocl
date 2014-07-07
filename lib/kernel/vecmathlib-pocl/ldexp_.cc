// Note: This file has been automatically generated. Do not modify.

#include "pocl-compat.h"

// ldexp_: ['VF', 'VI'] -> VF

#ifdef cl_khr_fp16

// ldexp_: VF=half
#if defined VECMATHLIB_HAVE_VEC_HALF_1 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" half _cl_ldexp_half_short(half x0, short x1)
{
  vecmathlib::realvec<half,1> y0 = bitcast<half,vecmathlib::realvec<half,1> >(x0);
  vecmathlib::realvec<half,1>::intvec_t y1 = bitcast<short,vecmathlib::realvec<half,1>::intvec_t >(x1);
  vecmathlib::realvec<half,1> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<half,1>,half>((r));
}
#elif ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling libm
extern "C" half _cl_ldexp_half_short(half x0, short x1)
{
  vecmathlib::realpseudovec<half,1> y0 = x0;
  vecmathlib::realpseudovec<half,1>::intvec_t y1 = x1;
  vecmathlib::realpseudovec<half,1> r = ldexp(y0, y1);
  return (r)[0];
}
#else
// Implement ldexp_ by calling builtin
extern "C" half _cl_ldexp_half_short(half x0, short x1)
{
  vecmathlib::realbuiltinvec<half,1> y0 = x0;
  vecmathlib::realbuiltinvec<half,1>::intvec_t y1 = x1;
  vecmathlib::realbuiltinvec<half,1> r = ldexp(y0, y1);
  return (r)[0];
}
#endif

// ldexp_: VF=half2
#if defined VECMATHLIB_HAVE_VEC_HALF_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" half2 _cl_ldexp_half2_short2(half2 x0, short2 x1)
{
  vecmathlib::realvec<half,2> y0 = bitcast<half2,vecmathlib::realvec<half,2> >(x0);
  vecmathlib::realvec<half,2>::intvec_t y1 = bitcast<short2,vecmathlib::realvec<half,2>::intvec_t >(x1);
  vecmathlib::realvec<half,2> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<half,2>,half2>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_4 || defined VECMATHLIB_HAVE_VEC_HALF_8 || defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" half4 _cl_ldexp_half4_short4(half4, short4);
extern "C" half2 _cl_ldexp_half2_short2(half2 x0, short2 x1)
{
  half4 y0 = bitcast<half2,half4>(x0);
  short4 y1 = bitcast<short2,short4>(x1);
  half4 r = _cl_ldexp_half4_short4(y0, y1);
  return bitcast<half4,half2>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" half _cl_ldexp_half_short(half, short);
extern "C" half2 _cl_ldexp_half2_short2(half2 x0, short2 x1)
{
  pair_half y0 = bitcast<half2,pair_half>(x0);
  pair_short y1 = bitcast<short2,pair_short>(x1);
  pair_half r;
  r.lo = _cl_ldexp_half_short(y0.lo, y1.lo);
  r.hi = _cl_ldexp_half_short(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_half) == sizeof(half2));
  return bitcast<pair_half,half2>(r);
}
#endif

// ldexp_: VF=half3
#if defined VECMATHLIB_HAVE_VEC_HALF_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" half3 _cl_ldexp_half3_short3(half3 x0, short3 x1)
{
  vecmathlib::realvec<half,3> y0 = bitcast<half3,vecmathlib::realvec<half,3> >(x0);
  vecmathlib::realvec<half,3>::intvec_t y1 = bitcast<short3,vecmathlib::realvec<half,3>::intvec_t >(x1);
  vecmathlib::realvec<half,3> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<half,3>,half3>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_4 || defined VECMATHLIB_HAVE_VEC_HALF_8 || defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" half4 _cl_ldexp_half4_short4(half4, short4);
extern "C" half3 _cl_ldexp_half3_short3(half3 x0, short3 x1)
{
  half4 y0 = bitcast<half3,half4>(x0);
  short4 y1 = bitcast<short3,short4>(x1);
  half4 r = _cl_ldexp_half4_short4(y0, y1);
  return bitcast<half4,half3>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" half2 _cl_ldexp_half2_short2(half2, short2);
extern "C" half3 _cl_ldexp_half3_short3(half3 x0, short3 x1)
{
  pair_half2 y0 = bitcast<half3,pair_half2>(x0);
  pair_short2 y1 = bitcast<short3,pair_short2>(x1);
  pair_half2 r;
  r.lo = _cl_ldexp_half2_short2(y0.lo, y1.lo);
  r.hi = _cl_ldexp_half2_short2(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_half2) == sizeof(half3));
  return bitcast<pair_half2,half3>(r);
}
#endif

// ldexp_: VF=half4
#if defined VECMATHLIB_HAVE_VEC_HALF_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" half4 _cl_ldexp_half4_short4(half4 x0, short4 x1)
{
  vecmathlib::realvec<half,4> y0 = bitcast<half4,vecmathlib::realvec<half,4> >(x0);
  vecmathlib::realvec<half,4>::intvec_t y1 = bitcast<short4,vecmathlib::realvec<half,4>::intvec_t >(x1);
  vecmathlib::realvec<half,4> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<half,4>,half4>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_8 || defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" half8 _cl_ldexp_half8_short8(half8, short8);
extern "C" half4 _cl_ldexp_half4_short4(half4 x0, short4 x1)
{
  half8 y0 = bitcast<half4,half8>(x0);
  short8 y1 = bitcast<short4,short8>(x1);
  half8 r = _cl_ldexp_half8_short8(y0, y1);
  return bitcast<half8,half4>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" half2 _cl_ldexp_half2_short2(half2, short2);
extern "C" half4 _cl_ldexp_half4_short4(half4 x0, short4 x1)
{
  pair_half2 y0 = bitcast<half4,pair_half2>(x0);
  pair_short2 y1 = bitcast<short4,pair_short2>(x1);
  pair_half2 r;
  r.lo = _cl_ldexp_half2_short2(y0.lo, y1.lo);
  r.hi = _cl_ldexp_half2_short2(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_half2) == sizeof(half4));
  return bitcast<pair_half2,half4>(r);
}
#endif

// ldexp_: VF=half8
#if defined VECMATHLIB_HAVE_VEC_HALF_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" half8 _cl_ldexp_half8_short8(half8 x0, short8 x1)
{
  vecmathlib::realvec<half,8> y0 = bitcast<half8,vecmathlib::realvec<half,8> >(x0);
  vecmathlib::realvec<half,8>::intvec_t y1 = bitcast<short8,vecmathlib::realvec<half,8>::intvec_t >(x1);
  vecmathlib::realvec<half,8> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<half,8>,half8>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" half16 _cl_ldexp_half16_short16(half16, short16);
extern "C" half8 _cl_ldexp_half8_short8(half8 x0, short8 x1)
{
  half16 y0 = bitcast<half8,half16>(x0);
  short16 y1 = bitcast<short8,short16>(x1);
  half16 r = _cl_ldexp_half16_short16(y0, y1);
  return bitcast<half16,half8>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" half4 _cl_ldexp_half4_short4(half4, short4);
extern "C" half8 _cl_ldexp_half8_short8(half8 x0, short8 x1)
{
  pair_half4 y0 = bitcast<half8,pair_half4>(x0);
  pair_short4 y1 = bitcast<short8,pair_short4>(x1);
  pair_half4 r;
  r.lo = _cl_ldexp_half4_short4(y0.lo, y1.lo);
  r.hi = _cl_ldexp_half4_short4(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_half4) == sizeof(half8));
  return bitcast<pair_half4,half8>(r);
}
#endif

// ldexp_: VF=half16
#if defined VECMATHLIB_HAVE_VEC_HALF_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" half16 _cl_ldexp_half16_short16(half16 x0, short16 x1)
{
  vecmathlib::realvec<half,16> y0 = bitcast<half16,vecmathlib::realvec<half,16> >(x0);
  vecmathlib::realvec<half,16>::intvec_t y1 = bitcast<short16,vecmathlib::realvec<half,16>::intvec_t >(x1);
  vecmathlib::realvec<half,16> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<half,16>,half16>((r));
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" half8 _cl_ldexp_half8_short8(half8, short8);
extern "C" half16 _cl_ldexp_half16_short16(half16 x0, short16 x1)
{
  pair_half8 y0 = bitcast<half16,pair_half8>(x0);
  pair_short8 y1 = bitcast<short16,pair_short8>(x1);
  pair_half8 r;
  r.lo = _cl_ldexp_half8_short8(y0.lo, y1.lo);
  r.hi = _cl_ldexp_half8_short8(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_half8) == sizeof(half16));
  return bitcast<pair_half8,half16>(r);
}
#endif

#endif // #ifdef cl_khr_fp16

// ldexp_: VF=float
#if defined VECMATHLIB_HAVE_VEC_FLOAT_1 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" float _cl_ldexp_float_int(float x0, int x1)
{
  vecmathlib::realvec<float,1> y0 = bitcast<float,vecmathlib::realvec<float,1> >(x0);
  vecmathlib::realvec<float,1>::intvec_t y1 = bitcast<int,vecmathlib::realvec<float,1>::intvec_t >(x1);
  vecmathlib::realvec<float,1> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,1>,float>((r));
}
#elif ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling libm
extern "C" float _cl_ldexp_float_int(float x0, int x1)
{
  vecmathlib::realpseudovec<float,1> y0 = x0;
  vecmathlib::realpseudovec<float,1>::intvec_t y1 = x1;
  vecmathlib::realpseudovec<float,1> r = ldexp(y0, y1);
  return (r)[0];
}
#else
// Implement ldexp_ by calling builtin
extern "C" float _cl_ldexp_float_int(float x0, int x1)
{
  vecmathlib::realbuiltinvec<float,1> y0 = x0;
  vecmathlib::realbuiltinvec<float,1>::intvec_t y1 = x1;
  vecmathlib::realbuiltinvec<float,1> r = ldexp(y0, y1);
  return (r)[0];
}
#endif

// ldexp_: VF=float2
#if defined VECMATHLIB_HAVE_VEC_FLOAT_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" float2 _cl_ldexp_float2_int2(float2 x0, int2 x1)
{
  vecmathlib::realvec<float,2> y0 = bitcast<float2,vecmathlib::realvec<float,2> >(x0);
  vecmathlib::realvec<float,2>::intvec_t y1 = bitcast<int2,vecmathlib::realvec<float,2>::intvec_t >(x1);
  vecmathlib::realvec<float,2> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,2>,float2>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_4 || defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" float4 _cl_ldexp_float4_int4(float4, int4);
extern "C" float2 _cl_ldexp_float2_int2(float2 x0, int2 x1)
{
  float4 y0 = bitcast<float2,float4>(x0);
  int4 y1 = bitcast<int2,int4>(x1);
  float4 r = _cl_ldexp_float4_int4(y0, y1);
  return bitcast<float4,float2>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" float _cl_ldexp_float_int(float, int);
extern "C" float2 _cl_ldexp_float2_int2(float2 x0, int2 x1)
{
  pair_float y0 = bitcast<float2,pair_float>(x0);
  pair_int y1 = bitcast<int2,pair_int>(x1);
  pair_float r;
  r.lo = _cl_ldexp_float_int(y0.lo, y1.lo);
  r.hi = _cl_ldexp_float_int(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float) == sizeof(float2));
  return bitcast<pair_float,float2>(r);
}
#endif

// ldexp_: VF=float3
#if defined VECMATHLIB_HAVE_VEC_FLOAT_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" float3 _cl_ldexp_float3_int3(float3 x0, int3 x1)
{
  vecmathlib::realvec<float,3> y0 = bitcast<float3,vecmathlib::realvec<float,3> >(x0);
  vecmathlib::realvec<float,3>::intvec_t y1 = bitcast<int3,vecmathlib::realvec<float,3>::intvec_t >(x1);
  vecmathlib::realvec<float,3> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,3>,float3>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_4 || defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" float4 _cl_ldexp_float4_int4(float4, int4);
extern "C" float3 _cl_ldexp_float3_int3(float3 x0, int3 x1)
{
  float4 y0 = bitcast<float3,float4>(x0);
  int4 y1 = bitcast<int3,int4>(x1);
  float4 r = _cl_ldexp_float4_int4(y0, y1);
  return bitcast<float4,float3>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" float2 _cl_ldexp_float2_int2(float2, int2);
extern "C" float3 _cl_ldexp_float3_int3(float3 x0, int3 x1)
{
  pair_float2 y0 = bitcast<float3,pair_float2>(x0);
  pair_int2 y1 = bitcast<int3,pair_int2>(x1);
  pair_float2 r;
  r.lo = _cl_ldexp_float2_int2(y0.lo, y1.lo);
  r.hi = _cl_ldexp_float2_int2(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float3));
  return bitcast<pair_float2,float3>(r);
}
#endif

// ldexp_: VF=float4
#if defined VECMATHLIB_HAVE_VEC_FLOAT_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" float4 _cl_ldexp_float4_int4(float4 x0, int4 x1)
{
  vecmathlib::realvec<float,4> y0 = bitcast<float4,vecmathlib::realvec<float,4> >(x0);
  vecmathlib::realvec<float,4>::intvec_t y1 = bitcast<int4,vecmathlib::realvec<float,4>::intvec_t >(x1);
  vecmathlib::realvec<float,4> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,4>,float4>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" float8 _cl_ldexp_float8_int8(float8, int8);
extern "C" float4 _cl_ldexp_float4_int4(float4 x0, int4 x1)
{
  float8 y0 = bitcast<float4,float8>(x0);
  int8 y1 = bitcast<int4,int8>(x1);
  float8 r = _cl_ldexp_float8_int8(y0, y1);
  return bitcast<float8,float4>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" float2 _cl_ldexp_float2_int2(float2, int2);
extern "C" float4 _cl_ldexp_float4_int4(float4 x0, int4 x1)
{
  pair_float2 y0 = bitcast<float4,pair_float2>(x0);
  pair_int2 y1 = bitcast<int4,pair_int2>(x1);
  pair_float2 r;
  r.lo = _cl_ldexp_float2_int2(y0.lo, y1.lo);
  r.hi = _cl_ldexp_float2_int2(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float4));
  return bitcast<pair_float2,float4>(r);
}
#endif

// ldexp_: VF=float8
#if defined VECMATHLIB_HAVE_VEC_FLOAT_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" float8 _cl_ldexp_float8_int8(float8 x0, int8 x1)
{
  vecmathlib::realvec<float,8> y0 = bitcast<float8,vecmathlib::realvec<float,8> >(x0);
  vecmathlib::realvec<float,8>::intvec_t y1 = bitcast<int8,vecmathlib::realvec<float,8>::intvec_t >(x1);
  vecmathlib::realvec<float,8> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,8>,float8>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" float16 _cl_ldexp_float16_int16(float16, int16);
extern "C" float8 _cl_ldexp_float8_int8(float8 x0, int8 x1)
{
  float16 y0 = bitcast<float8,float16>(x0);
  int16 y1 = bitcast<int8,int16>(x1);
  float16 r = _cl_ldexp_float16_int16(y0, y1);
  return bitcast<float16,float8>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" float4 _cl_ldexp_float4_int4(float4, int4);
extern "C" float8 _cl_ldexp_float8_int8(float8 x0, int8 x1)
{
  pair_float4 y0 = bitcast<float8,pair_float4>(x0);
  pair_int4 y1 = bitcast<int8,pair_int4>(x1);
  pair_float4 r;
  r.lo = _cl_ldexp_float4_int4(y0.lo, y1.lo);
  r.hi = _cl_ldexp_float4_int4(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float4) == sizeof(float8));
  return bitcast<pair_float4,float8>(r);
}
#endif

// ldexp_: VF=float16
#if defined VECMATHLIB_HAVE_VEC_FLOAT_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" float16 _cl_ldexp_float16_int16(float16 x0, int16 x1)
{
  vecmathlib::realvec<float,16> y0 = bitcast<float16,vecmathlib::realvec<float,16> >(x0);
  vecmathlib::realvec<float,16>::intvec_t y1 = bitcast<int16,vecmathlib::realvec<float,16>::intvec_t >(x1);
  vecmathlib::realvec<float,16> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,16>,float16>((r));
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" float8 _cl_ldexp_float8_int8(float8, int8);
extern "C" float16 _cl_ldexp_float16_int16(float16 x0, int16 x1)
{
  pair_float8 y0 = bitcast<float16,pair_float8>(x0);
  pair_int8 y1 = bitcast<int16,pair_int8>(x1);
  pair_float8 r;
  r.lo = _cl_ldexp_float8_int8(y0.lo, y1.lo);
  r.hi = _cl_ldexp_float8_int8(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float8) == sizeof(float16));
  return bitcast<pair_float8,float16>(r);
}
#endif

#ifdef cl_khr_fp64

// ldexp_: VF=double
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_1 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" double _cl_ldexp_double_long(double x0, long x1)
{
  vecmathlib::realvec<double,1> y0 = bitcast<double,vecmathlib::realvec<double,1> >(x0);
  vecmathlib::realvec<double,1>::intvec_t y1 = bitcast<long,vecmathlib::realvec<double,1>::intvec_t >(x1);
  vecmathlib::realvec<double,1> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,1>,double>((r));
}
#elif ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling libm
extern "C" double _cl_ldexp_double_long(double x0, long x1)
{
  vecmathlib::realpseudovec<double,1> y0 = x0;
  vecmathlib::realpseudovec<double,1>::intvec_t y1 = x1;
  vecmathlib::realpseudovec<double,1> r = ldexp(y0, y1);
  return (r)[0];
}
#else
// Implement ldexp_ by calling builtin
extern "C" double _cl_ldexp_double_long(double x0, long x1)
{
  vecmathlib::realbuiltinvec<double,1> y0 = x0;
  vecmathlib::realbuiltinvec<double,1>::intvec_t y1 = x1;
  vecmathlib::realbuiltinvec<double,1> r = ldexp(y0, y1);
  return (r)[0];
}
#endif

// ldexp_: VF=double2
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" double2 _cl_ldexp_double2_long2(double2 x0, long2 x1)
{
  vecmathlib::realvec<double,2> y0 = bitcast<double2,vecmathlib::realvec<double,2> >(x0);
  vecmathlib::realvec<double,2>::intvec_t y1 = bitcast<long2,vecmathlib::realvec<double,2>::intvec_t >(x1);
  vecmathlib::realvec<double,2> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,2>,double2>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_4 || defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" double4 _cl_ldexp_double4_long4(double4, long4);
extern "C" double2 _cl_ldexp_double2_long2(double2 x0, long2 x1)
{
  double4 y0 = bitcast<double2,double4>(x0);
  long4 y1 = bitcast<long2,long4>(x1);
  double4 r = _cl_ldexp_double4_long4(y0, y1);
  return bitcast<double4,double2>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" double _cl_ldexp_double_long(double, long);
extern "C" double2 _cl_ldexp_double2_long2(double2 x0, long2 x1)
{
  pair_double y0 = bitcast<double2,pair_double>(x0);
  pair_long y1 = bitcast<long2,pair_long>(x1);
  pair_double r;
  r.lo = _cl_ldexp_double_long(y0.lo, y1.lo);
  r.hi = _cl_ldexp_double_long(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double) == sizeof(double2));
  return bitcast<pair_double,double2>(r);
}
#endif

// ldexp_: VF=double3
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" double3 _cl_ldexp_double3_long3(double3 x0, long3 x1)
{
  vecmathlib::realvec<double,3> y0 = bitcast<double3,vecmathlib::realvec<double,3> >(x0);
  vecmathlib::realvec<double,3>::intvec_t y1 = bitcast<long3,vecmathlib::realvec<double,3>::intvec_t >(x1);
  vecmathlib::realvec<double,3> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,3>,double3>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_4 || defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" double4 _cl_ldexp_double4_long4(double4, long4);
extern "C" double3 _cl_ldexp_double3_long3(double3 x0, long3 x1)
{
  double4 y0 = bitcast<double3,double4>(x0);
  long4 y1 = bitcast<long3,long4>(x1);
  double4 r = _cl_ldexp_double4_long4(y0, y1);
  return bitcast<double4,double3>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" double2 _cl_ldexp_double2_long2(double2, long2);
extern "C" double3 _cl_ldexp_double3_long3(double3 x0, long3 x1)
{
  pair_double2 y0 = bitcast<double3,pair_double2>(x0);
  pair_long2 y1 = bitcast<long3,pair_long2>(x1);
  pair_double2 r;
  r.lo = _cl_ldexp_double2_long2(y0.lo, y1.lo);
  r.hi = _cl_ldexp_double2_long2(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double3));
  return bitcast<pair_double2,double3>(r);
}
#endif

// ldexp_: VF=double4
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" double4 _cl_ldexp_double4_long4(double4 x0, long4 x1)
{
  vecmathlib::realvec<double,4> y0 = bitcast<double4,vecmathlib::realvec<double,4> >(x0);
  vecmathlib::realvec<double,4>::intvec_t y1 = bitcast<long4,vecmathlib::realvec<double,4>::intvec_t >(x1);
  vecmathlib::realvec<double,4> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,4>,double4>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" double8 _cl_ldexp_double8_long8(double8, long8);
extern "C" double4 _cl_ldexp_double4_long4(double4 x0, long4 x1)
{
  double8 y0 = bitcast<double4,double8>(x0);
  long8 y1 = bitcast<long4,long8>(x1);
  double8 r = _cl_ldexp_double8_long8(y0, y1);
  return bitcast<double8,double4>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" double2 _cl_ldexp_double2_long2(double2, long2);
extern "C" double4 _cl_ldexp_double4_long4(double4 x0, long4 x1)
{
  pair_double2 y0 = bitcast<double4,pair_double2>(x0);
  pair_long2 y1 = bitcast<long4,pair_long2>(x1);
  pair_double2 r;
  r.lo = _cl_ldexp_double2_long2(y0.lo, y1.lo);
  r.hi = _cl_ldexp_double2_long2(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double4));
  return bitcast<pair_double2,double4>(r);
}
#endif

// ldexp_: VF=double8
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" double8 _cl_ldexp_double8_long8(double8 x0, long8 x1)
{
  vecmathlib::realvec<double,8> y0 = bitcast<double8,vecmathlib::realvec<double,8> >(x0);
  vecmathlib::realvec<double,8>::intvec_t y1 = bitcast<long8,vecmathlib::realvec<double,8>::intvec_t >(x1);
  vecmathlib::realvec<double,8> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,8>,double8>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" double16 _cl_ldexp_double16_long16(double16, long16);
extern "C" double8 _cl_ldexp_double8_long8(double8 x0, long8 x1)
{
  double16 y0 = bitcast<double8,double16>(x0);
  long16 y1 = bitcast<long8,long16>(x1);
  double16 r = _cl_ldexp_double16_long16(y0, y1);
  return bitcast<double16,double8>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" double4 _cl_ldexp_double4_long4(double4, long4);
extern "C" double8 _cl_ldexp_double8_long8(double8 x0, long8 x1)
{
  pair_double4 y0 = bitcast<double8,pair_double4>(x0);
  pair_long4 y1 = bitcast<long8,pair_long4>(x1);
  pair_double4 r;
  r.lo = _cl_ldexp_double4_long4(y0.lo, y1.lo);
  r.hi = _cl_ldexp_double4_long4(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double4) == sizeof(double8));
  return bitcast<pair_double4,double8>(r);
}
#endif

// ldexp_: VF=double16
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" double16 _cl_ldexp_double16_long16(double16 x0, long16 x1)
{
  vecmathlib::realvec<double,16> y0 = bitcast<double16,vecmathlib::realvec<double,16> >(x0);
  vecmathlib::realvec<double,16>::intvec_t y1 = bitcast<long16,vecmathlib::realvec<double,16>::intvec_t >(x1);
  vecmathlib::realvec<double,16> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,16>,double16>((r));
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" double8 _cl_ldexp_double8_long8(double8, long8);
extern "C" double16 _cl_ldexp_double16_long16(double16 x0, long16 x1)
{
  pair_double8 y0 = bitcast<double16,pair_double8>(x0);
  pair_long8 y1 = bitcast<long16,pair_long8>(x1);
  pair_double8 r;
  r.lo = _cl_ldexp_double8_long8(y0.lo, y1.lo);
  r.hi = _cl_ldexp_double8_long8(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double8) == sizeof(double16));
  return bitcast<pair_double8,double16>(r);
}
#endif

#endif // #ifdef cl_khr_fp64



// ldexp_: ['VF', 'SI'] -> VF

#ifdef cl_khr_fp16

// ldexp_: VF=half2
#if defined VECMATHLIB_HAVE_VEC_HALF_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" half2 _cl_ldexp_half2_short(half2 x0, short x1)
{
  vecmathlib::realvec<half,2> y0 = bitcast<half2,vecmathlib::realvec<half,2> >(x0);
  vecmathlib::realvec<half,2>::int_t y1 = bitcast<short,vecmathlib::realvec<half,2>::int_t >(x1);
  vecmathlib::realvec<half,2> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<half,2>,half2>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_4 || defined VECMATHLIB_HAVE_VEC_HALF_8 || defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" half4 _cl_ldexp_half4_short(half4, short);
extern "C" half2 _cl_ldexp_half2_short(half2 x0, short x1)
{
  half4 y0 = bitcast<half2,half4>(x0);
  short y1 = bitcast<short,short>(x1);
  half4 r = _cl_ldexp_half4_short(y0, y1);
  return bitcast<half4,half2>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" half _cl_ldexp_half_short(half, short);
extern "C" half2 _cl_ldexp_half2_short(half2 x0, short x1)
{
  pair_half y0 = bitcast<half2,pair_half>(x0);
  pair_short y1 = bitcast<short,pair_short>(x1);
  pair_half r;
  r.lo = _cl_ldexp_half_short(y0.lo, y1.lo);
  r.hi = _cl_ldexp_half_short(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_half) == sizeof(half2));
  return bitcast<pair_half,half2>(r);
}
#endif

// ldexp_: VF=half3
#if defined VECMATHLIB_HAVE_VEC_HALF_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" half3 _cl_ldexp_half3_short(half3 x0, short x1)
{
  vecmathlib::realvec<half,3> y0 = bitcast<half3,vecmathlib::realvec<half,3> >(x0);
  vecmathlib::realvec<half,3>::int_t y1 = bitcast<short,vecmathlib::realvec<half,3>::int_t >(x1);
  vecmathlib::realvec<half,3> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<half,3>,half3>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_4 || defined VECMATHLIB_HAVE_VEC_HALF_8 || defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" half4 _cl_ldexp_half4_short(half4, short);
extern "C" half3 _cl_ldexp_half3_short(half3 x0, short x1)
{
  half4 y0 = bitcast<half3,half4>(x0);
  short y1 = bitcast<short,short>(x1);
  half4 r = _cl_ldexp_half4_short(y0, y1);
  return bitcast<half4,half3>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" half2 _cl_ldexp_half2_short(half2, short);
extern "C" half3 _cl_ldexp_half3_short(half3 x0, short x1)
{
  pair_half2 y0 = bitcast<half3,pair_half2>(x0);
  pair_short y1 = bitcast<short,pair_short>(x1);
  pair_half2 r;
  r.lo = _cl_ldexp_half2_short(y0.lo, y1.lo);
  r.hi = _cl_ldexp_half2_short(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_half2) == sizeof(half3));
  return bitcast<pair_half2,half3>(r);
}
#endif

// ldexp_: VF=half4
#if defined VECMATHLIB_HAVE_VEC_HALF_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" half4 _cl_ldexp_half4_short(half4 x0, short x1)
{
  vecmathlib::realvec<half,4> y0 = bitcast<half4,vecmathlib::realvec<half,4> >(x0);
  vecmathlib::realvec<half,4>::int_t y1 = bitcast<short,vecmathlib::realvec<half,4>::int_t >(x1);
  vecmathlib::realvec<half,4> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<half,4>,half4>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_8 || defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" half8 _cl_ldexp_half8_short(half8, short);
extern "C" half4 _cl_ldexp_half4_short(half4 x0, short x1)
{
  half8 y0 = bitcast<half4,half8>(x0);
  short y1 = bitcast<short,short>(x1);
  half8 r = _cl_ldexp_half8_short(y0, y1);
  return bitcast<half8,half4>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" half2 _cl_ldexp_half2_short(half2, short);
extern "C" half4 _cl_ldexp_half4_short(half4 x0, short x1)
{
  pair_half2 y0 = bitcast<half4,pair_half2>(x0);
  pair_short y1 = bitcast<short,pair_short>(x1);
  pair_half2 r;
  r.lo = _cl_ldexp_half2_short(y0.lo, y1.lo);
  r.hi = _cl_ldexp_half2_short(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_half2) == sizeof(half4));
  return bitcast<pair_half2,half4>(r);
}
#endif

// ldexp_: VF=half8
#if defined VECMATHLIB_HAVE_VEC_HALF_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" half8 _cl_ldexp_half8_short(half8 x0, short x1)
{
  vecmathlib::realvec<half,8> y0 = bitcast<half8,vecmathlib::realvec<half,8> >(x0);
  vecmathlib::realvec<half,8>::int_t y1 = bitcast<short,vecmathlib::realvec<half,8>::int_t >(x1);
  vecmathlib::realvec<half,8> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<half,8>,half8>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_HALF_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" half16 _cl_ldexp_half16_short(half16, short);
extern "C" half8 _cl_ldexp_half8_short(half8 x0, short x1)
{
  half16 y0 = bitcast<half8,half16>(x0);
  short y1 = bitcast<short,short>(x1);
  half16 r = _cl_ldexp_half16_short(y0, y1);
  return bitcast<half16,half8>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" half4 _cl_ldexp_half4_short(half4, short);
extern "C" half8 _cl_ldexp_half8_short(half8 x0, short x1)
{
  pair_half4 y0 = bitcast<half8,pair_half4>(x0);
  pair_short y1 = bitcast<short,pair_short>(x1);
  pair_half4 r;
  r.lo = _cl_ldexp_half4_short(y0.lo, y1.lo);
  r.hi = _cl_ldexp_half4_short(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_half4) == sizeof(half8));
  return bitcast<pair_half4,half8>(r);
}
#endif

// ldexp_: VF=half16
#if defined VECMATHLIB_HAVE_VEC_HALF_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" half16 _cl_ldexp_half16_short(half16 x0, short x1)
{
  vecmathlib::realvec<half,16> y0 = bitcast<half16,vecmathlib::realvec<half,16> >(x0);
  vecmathlib::realvec<half,16>::int_t y1 = bitcast<short,vecmathlib::realvec<half,16>::int_t >(x1);
  vecmathlib::realvec<half,16> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<half,16>,half16>((r));
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" half8 _cl_ldexp_half8_short(half8, short);
extern "C" half16 _cl_ldexp_half16_short(half16 x0, short x1)
{
  pair_half8 y0 = bitcast<half16,pair_half8>(x0);
  pair_short y1 = bitcast<short,pair_short>(x1);
  pair_half8 r;
  r.lo = _cl_ldexp_half8_short(y0.lo, y1.lo);
  r.hi = _cl_ldexp_half8_short(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_half8) == sizeof(half16));
  return bitcast<pair_half8,half16>(r);
}
#endif

#endif // #ifdef cl_khr_fp16

// ldexp_: VF=float2
#if defined VECMATHLIB_HAVE_VEC_FLOAT_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" float2 _cl_ldexp_float2_int(float2 x0, int x1)
{
  vecmathlib::realvec<float,2> y0 = bitcast<float2,vecmathlib::realvec<float,2> >(x0);
  vecmathlib::realvec<float,2>::int_t y1 = bitcast<int,vecmathlib::realvec<float,2>::int_t >(x1);
  vecmathlib::realvec<float,2> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,2>,float2>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_4 || defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" float4 _cl_ldexp_float4_int(float4, int);
extern "C" float2 _cl_ldexp_float2_int(float2 x0, int x1)
{
  float4 y0 = bitcast<float2,float4>(x0);
  int y1 = bitcast<int,int>(x1);
  float4 r = _cl_ldexp_float4_int(y0, y1);
  return bitcast<float4,float2>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" float _cl_ldexp_float_int(float, int);
extern "C" float2 _cl_ldexp_float2_int(float2 x0, int x1)
{
  pair_float y0 = bitcast<float2,pair_float>(x0);
  pair_int y1 = bitcast<int,pair_int>(x1);
  pair_float r;
  r.lo = _cl_ldexp_float_int(y0.lo, y1.lo);
  r.hi = _cl_ldexp_float_int(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float) == sizeof(float2));
  return bitcast<pair_float,float2>(r);
}
#endif

// ldexp_: VF=float3
#if defined VECMATHLIB_HAVE_VEC_FLOAT_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" float3 _cl_ldexp_float3_int(float3 x0, int x1)
{
  vecmathlib::realvec<float,3> y0 = bitcast<float3,vecmathlib::realvec<float,3> >(x0);
  vecmathlib::realvec<float,3>::int_t y1 = bitcast<int,vecmathlib::realvec<float,3>::int_t >(x1);
  vecmathlib::realvec<float,3> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,3>,float3>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_4 || defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" float4 _cl_ldexp_float4_int(float4, int);
extern "C" float3 _cl_ldexp_float3_int(float3 x0, int x1)
{
  float4 y0 = bitcast<float3,float4>(x0);
  int y1 = bitcast<int,int>(x1);
  float4 r = _cl_ldexp_float4_int(y0, y1);
  return bitcast<float4,float3>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" float2 _cl_ldexp_float2_int(float2, int);
extern "C" float3 _cl_ldexp_float3_int(float3 x0, int x1)
{
  pair_float2 y0 = bitcast<float3,pair_float2>(x0);
  pair_int y1 = bitcast<int,pair_int>(x1);
  pair_float2 r;
  r.lo = _cl_ldexp_float2_int(y0.lo, y1.lo);
  r.hi = _cl_ldexp_float2_int(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float3));
  return bitcast<pair_float2,float3>(r);
}
#endif

// ldexp_: VF=float4
#if defined VECMATHLIB_HAVE_VEC_FLOAT_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" float4 _cl_ldexp_float4_int(float4 x0, int x1)
{
  vecmathlib::realvec<float,4> y0 = bitcast<float4,vecmathlib::realvec<float,4> >(x0);
  vecmathlib::realvec<float,4>::int_t y1 = bitcast<int,vecmathlib::realvec<float,4>::int_t >(x1);
  vecmathlib::realvec<float,4> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,4>,float4>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_8 || defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" float8 _cl_ldexp_float8_int(float8, int);
extern "C" float4 _cl_ldexp_float4_int(float4 x0, int x1)
{
  float8 y0 = bitcast<float4,float8>(x0);
  int y1 = bitcast<int,int>(x1);
  float8 r = _cl_ldexp_float8_int(y0, y1);
  return bitcast<float8,float4>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" float2 _cl_ldexp_float2_int(float2, int);
extern "C" float4 _cl_ldexp_float4_int(float4 x0, int x1)
{
  pair_float2 y0 = bitcast<float4,pair_float2>(x0);
  pair_int y1 = bitcast<int,pair_int>(x1);
  pair_float2 r;
  r.lo = _cl_ldexp_float2_int(y0.lo, y1.lo);
  r.hi = _cl_ldexp_float2_int(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float2) == sizeof(float4));
  return bitcast<pair_float2,float4>(r);
}
#endif

// ldexp_: VF=float8
#if defined VECMATHLIB_HAVE_VEC_FLOAT_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" float8 _cl_ldexp_float8_int(float8 x0, int x1)
{
  vecmathlib::realvec<float,8> y0 = bitcast<float8,vecmathlib::realvec<float,8> >(x0);
  vecmathlib::realvec<float,8>::int_t y1 = bitcast<int,vecmathlib::realvec<float,8>::int_t >(x1);
  vecmathlib::realvec<float,8> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,8>,float8>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_FLOAT_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" float16 _cl_ldexp_float16_int(float16, int);
extern "C" float8 _cl_ldexp_float8_int(float8 x0, int x1)
{
  float16 y0 = bitcast<float8,float16>(x0);
  int y1 = bitcast<int,int>(x1);
  float16 r = _cl_ldexp_float16_int(y0, y1);
  return bitcast<float16,float8>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" float4 _cl_ldexp_float4_int(float4, int);
extern "C" float8 _cl_ldexp_float8_int(float8 x0, int x1)
{
  pair_float4 y0 = bitcast<float8,pair_float4>(x0);
  pair_int y1 = bitcast<int,pair_int>(x1);
  pair_float4 r;
  r.lo = _cl_ldexp_float4_int(y0.lo, y1.lo);
  r.hi = _cl_ldexp_float4_int(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float4) == sizeof(float8));
  return bitcast<pair_float4,float8>(r);
}
#endif

// ldexp_: VF=float16
#if defined VECMATHLIB_HAVE_VEC_FLOAT_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" float16 _cl_ldexp_float16_int(float16 x0, int x1)
{
  vecmathlib::realvec<float,16> y0 = bitcast<float16,vecmathlib::realvec<float,16> >(x0);
  vecmathlib::realvec<float,16>::int_t y1 = bitcast<int,vecmathlib::realvec<float,16>::int_t >(x1);
  vecmathlib::realvec<float,16> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<float,16>,float16>((r));
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" float8 _cl_ldexp_float8_int(float8, int);
extern "C" float16 _cl_ldexp_float16_int(float16 x0, int x1)
{
  pair_float8 y0 = bitcast<float16,pair_float8>(x0);
  pair_int y1 = bitcast<int,pair_int>(x1);
  pair_float8 r;
  r.lo = _cl_ldexp_float8_int(y0.lo, y1.lo);
  r.hi = _cl_ldexp_float8_int(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_float8) == sizeof(float16));
  return bitcast<pair_float8,float16>(r);
}
#endif

#ifdef cl_khr_fp64

// ldexp_: VF=double2
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_2 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" double2 _cl_ldexp_double2_long(double2 x0, long x1)
{
  vecmathlib::realvec<double,2> y0 = bitcast<double2,vecmathlib::realvec<double,2> >(x0);
  vecmathlib::realvec<double,2>::int_t y1 = bitcast<long,vecmathlib::realvec<double,2>::int_t >(x1);
  vecmathlib::realvec<double,2> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,2>,double2>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_4 || defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" double4 _cl_ldexp_double4_long(double4, long);
extern "C" double2 _cl_ldexp_double2_long(double2 x0, long x1)
{
  double4 y0 = bitcast<double2,double4>(x0);
  long y1 = bitcast<long,long>(x1);
  double4 r = _cl_ldexp_double4_long(y0, y1);
  return bitcast<double4,double2>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" double _cl_ldexp_double_long(double, long);
extern "C" double2 _cl_ldexp_double2_long(double2 x0, long x1)
{
  pair_double y0 = bitcast<double2,pair_double>(x0);
  pair_long y1 = bitcast<long,pair_long>(x1);
  pair_double r;
  r.lo = _cl_ldexp_double_long(y0.lo, y1.lo);
  r.hi = _cl_ldexp_double_long(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double) == sizeof(double2));
  return bitcast<pair_double,double2>(r);
}
#endif

// ldexp_: VF=double3
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_3 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" double3 _cl_ldexp_double3_long(double3 x0, long x1)
{
  vecmathlib::realvec<double,3> y0 = bitcast<double3,vecmathlib::realvec<double,3> >(x0);
  vecmathlib::realvec<double,3>::int_t y1 = bitcast<long,vecmathlib::realvec<double,3>::int_t >(x1);
  vecmathlib::realvec<double,3> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,3>,double3>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_4 || defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" double4 _cl_ldexp_double4_long(double4, long);
extern "C" double3 _cl_ldexp_double3_long(double3 x0, long x1)
{
  double4 y0 = bitcast<double3,double4>(x0);
  long y1 = bitcast<long,long>(x1);
  double4 r = _cl_ldexp_double4_long(y0, y1);
  return bitcast<double4,double3>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" double2 _cl_ldexp_double2_long(double2, long);
extern "C" double3 _cl_ldexp_double3_long(double3 x0, long x1)
{
  pair_double2 y0 = bitcast<double3,pair_double2>(x0);
  pair_long y1 = bitcast<long,pair_long>(x1);
  pair_double2 r;
  r.lo = _cl_ldexp_double2_long(y0.lo, y1.lo);
  r.hi = _cl_ldexp_double2_long(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double3));
  return bitcast<pair_double2,double3>(r);
}
#endif

// ldexp_: VF=double4
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_4 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" double4 _cl_ldexp_double4_long(double4 x0, long x1)
{
  vecmathlib::realvec<double,4> y0 = bitcast<double4,vecmathlib::realvec<double,4> >(x0);
  vecmathlib::realvec<double,4>::int_t y1 = bitcast<long,vecmathlib::realvec<double,4>::int_t >(x1);
  vecmathlib::realvec<double,4> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,4>,double4>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_8 || defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" double8 _cl_ldexp_double8_long(double8, long);
extern "C" double4 _cl_ldexp_double4_long(double4 x0, long x1)
{
  double8 y0 = bitcast<double4,double8>(x0);
  long y1 = bitcast<long,long>(x1);
  double8 r = _cl_ldexp_double8_long(y0, y1);
  return bitcast<double8,double4>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" double2 _cl_ldexp_double2_long(double2, long);
extern "C" double4 _cl_ldexp_double4_long(double4 x0, long x1)
{
  pair_double2 y0 = bitcast<double4,pair_double2>(x0);
  pair_long y1 = bitcast<long,pair_long>(x1);
  pair_double2 r;
  r.lo = _cl_ldexp_double2_long(y0.lo, y1.lo);
  r.hi = _cl_ldexp_double2_long(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double2) == sizeof(double4));
  return bitcast<pair_double2,double4>(r);
}
#endif

// ldexp_: VF=double8
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_8 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" double8 _cl_ldexp_double8_long(double8 x0, long x1)
{
  vecmathlib::realvec<double,8> y0 = bitcast<double8,vecmathlib::realvec<double,8> >(x0);
  vecmathlib::realvec<double,8>::int_t y1 = bitcast<long,vecmathlib::realvec<double,8>::int_t >(x1);
  vecmathlib::realvec<double,8> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,8>,double8>((r));
}
#elif (defined VECMATHLIB_HAVE_VEC_DOUBLE_16) && ! defined POCL_VECMATHLIB_BUILTIN 
// Implement ldexp_ by using a larger vector size
extern "C" double16 _cl_ldexp_double16_long(double16, long);
extern "C" double8 _cl_ldexp_double8_long(double8 x0, long x1)
{
  double16 y0 = bitcast<double8,double16>(x0);
  long y1 = bitcast<long,long>(x1);
  double16 r = _cl_ldexp_double16_long(y0, y1);
  return bitcast<double16,double8>(r);
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" double4 _cl_ldexp_double4_long(double4, long);
extern "C" double8 _cl_ldexp_double8_long(double8 x0, long x1)
{
  pair_double4 y0 = bitcast<double8,pair_double4>(x0);
  pair_long y1 = bitcast<long,pair_long>(x1);
  pair_double4 r;
  r.lo = _cl_ldexp_double4_long(y0.lo, y1.lo);
  r.hi = _cl_ldexp_double4_long(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double4) == sizeof(double8));
  return bitcast<pair_double4,double8>(r);
}
#endif

// ldexp_: VF=double16
#if defined VECMATHLIB_HAVE_VEC_DOUBLE_16 && ! defined POCL_VECMATHLIB_BUILTIN
// Implement ldexp_ by calling vecmathlib
extern "C" double16 _cl_ldexp_double16_long(double16 x0, long x1)
{
  vecmathlib::realvec<double,16> y0 = bitcast<double16,vecmathlib::realvec<double,16> >(x0);
  vecmathlib::realvec<double,16>::int_t y1 = bitcast<long,vecmathlib::realvec<double,16>::int_t >(x1);
  vecmathlib::realvec<double,16> r = vecmathlib::ldexp(y0, y1);
  return bitcast<vecmathlib::realvec<double,16>,double16>((r));
}
#else
// Implement ldexp_ by splitting into a smaller vector size
extern "C" double8 _cl_ldexp_double8_long(double8, long);
extern "C" double16 _cl_ldexp_double16_long(double16 x0, long x1)
{
  pair_double8 y0 = bitcast<double16,pair_double8>(x0);
  pair_long y1 = bitcast<long,pair_long>(x1);
  pair_double8 r;
  r.lo = _cl_ldexp_double8_long(y0.lo, y1.lo);
  r.hi = _cl_ldexp_double8_long(y0.hi, y1.hi);
  pocl_static_assert(sizeof(pair_double8) == sizeof(double16));
  return bitcast<pair_double8,double16>(r);
}
#endif

#endif // #ifdef cl_khr_fp64
