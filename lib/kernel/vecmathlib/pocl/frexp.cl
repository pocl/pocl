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

// frexp: ['VF', 'PVK'] -> VF

// frexp: VF=globalfloat
// Implement frexp directly
__attribute__((__overloadable__))
float _cl_frexp(float x0, global int* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=localfloat
// Implement frexp directly
__attribute__((__overloadable__))
float _cl_frexp(float x0, local int* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=privatefloat
// Implement frexp directly
__attribute__((__overloadable__))
float _cl_frexp(float x0, private int* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=globalfloat2
// Implement frexp directly
__attribute__((__overloadable__))
float2 _cl_frexp(float2 x0, global int2* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=localfloat2
// Implement frexp directly
__attribute__((__overloadable__))
float2 _cl_frexp(float2 x0, local int2* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=privatefloat2
// Implement frexp directly
__attribute__((__overloadable__))
float2 _cl_frexp(float2 x0, private int2* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=globalfloat3
// Implement frexp directly
__attribute__((__overloadable__))
float3 _cl_frexp(float3 x0, global int3* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=localfloat3
// Implement frexp directly
__attribute__((__overloadable__))
float3 _cl_frexp(float3 x0, local int3* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=privatefloat3
// Implement frexp directly
__attribute__((__overloadable__))
float3 _cl_frexp(float3 x0, private int3* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=globalfloat4
// Implement frexp directly
__attribute__((__overloadable__))
float4 _cl_frexp(float4 x0, global int4* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=localfloat4
// Implement frexp directly
__attribute__((__overloadable__))
float4 _cl_frexp(float4 x0, local int4* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=privatefloat4
// Implement frexp directly
__attribute__((__overloadable__))
float4 _cl_frexp(float4 x0, private int4* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=globalfloat8
// Implement frexp directly
__attribute__((__overloadable__))
float8 _cl_frexp(float8 x0, global int8* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=localfloat8
// Implement frexp directly
__attribute__((__overloadable__))
float8 _cl_frexp(float8 x0, local int8* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=privatefloat8
// Implement frexp directly
__attribute__((__overloadable__))
float8 _cl_frexp(float8 x0, private int8* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=globalfloat16
// Implement frexp directly
__attribute__((__overloadable__))
float16 _cl_frexp(float16 x0, global int16* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=localfloat16
// Implement frexp directly
__attribute__((__overloadable__))
float16 _cl_frexp(float16 x0, local int16* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=privatefloat16
// Implement frexp directly
__attribute__((__overloadable__))
float16 _cl_frexp(float16 x0, private int16* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

#ifdef cl_khr_fp64

// frexp: VF=globaldouble
// Implement frexp directly
__attribute__((__overloadable__))
double _cl_frexp(double x0, global int* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=localdouble
// Implement frexp directly
__attribute__((__overloadable__))
double _cl_frexp(double x0, local int* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=privatedouble
// Implement frexp directly
__attribute__((__overloadable__))
double _cl_frexp(double x0, private int* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=globaldouble2
// Implement frexp directly
__attribute__((__overloadable__))
double2 _cl_frexp(double2 x0, global int2* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=localdouble2
// Implement frexp directly
__attribute__((__overloadable__))
double2 _cl_frexp(double2 x0, local int2* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=privatedouble2
// Implement frexp directly
__attribute__((__overloadable__))
double2 _cl_frexp(double2 x0, private int2* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=globaldouble3
// Implement frexp directly
__attribute__((__overloadable__))
double3 _cl_frexp(double3 x0, global int3* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=localdouble3
// Implement frexp directly
__attribute__((__overloadable__))
double3 _cl_frexp(double3 x0, local int3* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=privatedouble3
// Implement frexp directly
__attribute__((__overloadable__))
double3 _cl_frexp(double3 x0, private int3* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=globaldouble4
// Implement frexp directly
__attribute__((__overloadable__))
double4 _cl_frexp(double4 x0, global int4* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=localdouble4
// Implement frexp directly
__attribute__((__overloadable__))
double4 _cl_frexp(double4 x0, local int4* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=privatedouble4
// Implement frexp directly
__attribute__((__overloadable__))
double4 _cl_frexp(double4 x0, private int4* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=globaldouble8
// Implement frexp directly
__attribute__((__overloadable__))
double8 _cl_frexp(double8 x0, global int8* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=localdouble8
// Implement frexp directly
__attribute__((__overloadable__))
double8 _cl_frexp(double8 x0, local int8* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=privatedouble8
// Implement frexp directly
__attribute__((__overloadable__))
double8 _cl_frexp(double8 x0, private int8* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=globaldouble16
// Implement frexp directly
__attribute__((__overloadable__))
double16 _cl_frexp(double16 x0, global int16* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=localdouble16
// Implement frexp directly
__attribute__((__overloadable__))
double16 _cl_frexp(double16 x0, local int16* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// frexp: VF=privatedouble16
// Implement frexp directly
__attribute__((__overloadable__))
double16 _cl_frexp(double16 x0, private int16* x1)
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
  return *x1=ilogb(x0), ldexp(x0,-ilogb(x0));
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

#endif // #ifdef cl_khr_fp64
