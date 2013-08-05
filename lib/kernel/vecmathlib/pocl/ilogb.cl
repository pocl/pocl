// Note: This file has been automatically generated. Do not modify.

// Needed for fract()
#define POCL_FRACT_MIN   0x1.fffffffffffffp-1
#define POCL_FRACT_MIN_F 0x1.fffffep-1f

// If double precision is not supported, then define
// single-precision (dummy) values to avoid compiler warnings
// for double precision values
#ifndef khr_fp64
#  undef M_PI
#  define M_PI M_PI_F
#  undef M_PI_2
#  define M_PI_2 M_PI_2_F
#  undef LONG_MAX
#  define LONG_MAX INT_MAX
#  undef LONG_MIN
#  define LONG_MIN INT_MIN
#  undef POCL_FRACT_MIN
#  define POCL_FRACT_MIN POCL_FRACT_MIN_F
#endif

// ilogb: ['VF'] -> VK

// ilogb: VF=float
// Implement ilogb directly
__attribute__((__overloadable__))
int _cl_ilogb(float x0)
{
  typedef int kscalar_t;
  typedef float scalar_t;
  typedef int ivector_t;
  typedef int jvector_t;
  typedef int kvector_t;
  typedef float vector_t;
#define convert_ivector_t convert_int
#define convert_jvector_t convert_int
#define convert_kvector_t convert_int
#define convert_vector_t convert_float
  return convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// ilogb: VF=float2
// Implement ilogb directly
__attribute__((__overloadable__))
int2 _cl_ilogb(float2 x0)
{
  typedef int kscalar_t;
  typedef float scalar_t;
  typedef int2 ivector_t;
  typedef int2 jvector_t;
  typedef int2 kvector_t;
  typedef float2 vector_t;
#define convert_ivector_t convert_int2
#define convert_jvector_t convert_int2
#define convert_kvector_t convert_int2
#define convert_vector_t convert_float2
  return convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// ilogb: VF=float3
// Implement ilogb directly
__attribute__((__overloadable__))
int3 _cl_ilogb(float3 x0)
{
  typedef int kscalar_t;
  typedef float scalar_t;
  typedef int3 ivector_t;
  typedef int3 jvector_t;
  typedef int3 kvector_t;
  typedef float3 vector_t;
#define convert_ivector_t convert_int3
#define convert_jvector_t convert_int3
#define convert_kvector_t convert_int3
#define convert_vector_t convert_float3
  return convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// ilogb: VF=float4
// Implement ilogb directly
__attribute__((__overloadable__))
int4 _cl_ilogb(float4 x0)
{
  typedef int kscalar_t;
  typedef float scalar_t;
  typedef int4 ivector_t;
  typedef int4 jvector_t;
  typedef int4 kvector_t;
  typedef float4 vector_t;
#define convert_ivector_t convert_int4
#define convert_jvector_t convert_int4
#define convert_kvector_t convert_int4
#define convert_vector_t convert_float4
  return convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// ilogb: VF=float8
// Implement ilogb directly
__attribute__((__overloadable__))
int8 _cl_ilogb(float8 x0)
{
  typedef int kscalar_t;
  typedef float scalar_t;
  typedef int8 ivector_t;
  typedef int8 jvector_t;
  typedef int8 kvector_t;
  typedef float8 vector_t;
#define convert_ivector_t convert_int8
#define convert_jvector_t convert_int8
#define convert_kvector_t convert_int8
#define convert_vector_t convert_float8
  return convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// ilogb: VF=float16
// Implement ilogb directly
__attribute__((__overloadable__))
int16 _cl_ilogb(float16 x0)
{
  typedef int kscalar_t;
  typedef float scalar_t;
  typedef int16 ivector_t;
  typedef int16 jvector_t;
  typedef int16 kvector_t;
  typedef float16 vector_t;
#define convert_ivector_t convert_int16
#define convert_jvector_t convert_int16
#define convert_kvector_t convert_int16
#define convert_vector_t convert_float16
  return convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

#ifdef cl_khr_fp64

// ilogb: VF=double
// Implement ilogb directly
__attribute__((__overloadable__))
int _cl_ilogb(double x0)
{
  typedef int kscalar_t;
  typedef double scalar_t;
  typedef long ivector_t;
  typedef int jvector_t;
  typedef int kvector_t;
  typedef double vector_t;
#define convert_ivector_t convert_long
#define convert_jvector_t convert_int
#define convert_kvector_t convert_int
#define convert_vector_t convert_double
  return convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// ilogb: VF=double2
// Implement ilogb directly
__attribute__((__overloadable__))
int2 _cl_ilogb(double2 x0)
{
  typedef long kscalar_t;
  typedef double scalar_t;
  typedef long2 ivector_t;
  typedef long2 jvector_t;
  typedef int2 kvector_t;
  typedef double2 vector_t;
#define convert_ivector_t convert_long2
#define convert_jvector_t convert_long2
#define convert_kvector_t convert_int2
#define convert_vector_t convert_double2
  return convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// ilogb: VF=double3
// Implement ilogb directly
__attribute__((__overloadable__))
int3 _cl_ilogb(double3 x0)
{
  typedef long kscalar_t;
  typedef double scalar_t;
  typedef long3 ivector_t;
  typedef long3 jvector_t;
  typedef int3 kvector_t;
  typedef double3 vector_t;
#define convert_ivector_t convert_long3
#define convert_jvector_t convert_long3
#define convert_kvector_t convert_int3
#define convert_vector_t convert_double3
  return convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// ilogb: VF=double4
// Implement ilogb directly
__attribute__((__overloadable__))
int4 _cl_ilogb(double4 x0)
{
  typedef long kscalar_t;
  typedef double scalar_t;
  typedef long4 ivector_t;
  typedef long4 jvector_t;
  typedef int4 kvector_t;
  typedef double4 vector_t;
#define convert_ivector_t convert_long4
#define convert_jvector_t convert_long4
#define convert_kvector_t convert_int4
#define convert_vector_t convert_double4
  return convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// ilogb: VF=double8
// Implement ilogb directly
__attribute__((__overloadable__))
int8 _cl_ilogb(double8 x0)
{
  typedef long kscalar_t;
  typedef double scalar_t;
  typedef long8 ivector_t;
  typedef long8 jvector_t;
  typedef int8 kvector_t;
  typedef double8 vector_t;
#define convert_ivector_t convert_long8
#define convert_jvector_t convert_long8
#define convert_kvector_t convert_int8
#define convert_vector_t convert_double8
  return convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// ilogb: VF=double16
// Implement ilogb directly
__attribute__((__overloadable__))
int16 _cl_ilogb(double16 x0)
{
  typedef long kscalar_t;
  typedef double scalar_t;
  typedef long16 ivector_t;
  typedef long16 jvector_t;
  typedef int16 kvector_t;
  typedef double16 vector_t;
#define convert_ivector_t convert_long16
#define convert_jvector_t convert_long16
#define convert_kvector_t convert_int16
#define convert_vector_t convert_double16
  return convert_kvector_t(({ __attribute__((__overloadable__)) jvector_t ilogb_(vector_t); jvector_t jmin=sizeof(jvector_t)==sizeof(int)?INT_MIN:LONG_MIN; jvector_t r=ilogb_(x0); select(r, (jvector_t)FP_ILOGB0, r==jmin); }));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

#endif // #ifdef cl_khr_fp64
