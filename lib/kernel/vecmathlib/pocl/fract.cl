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

// fract: ['VF', 'PVF'] -> VF

// fract: VF=globalfloat
// Implement fract directly
__attribute__((__overloadable__))
float _cl_fract(float x0, global float* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=localfloat
// Implement fract directly
__attribute__((__overloadable__))
float _cl_fract(float x0, local float* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=privatefloat
// Implement fract directly
__attribute__((__overloadable__))
float _cl_fract(float x0, private float* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=globalfloat2
// Implement fract directly
__attribute__((__overloadable__))
float2 _cl_fract(float2 x0, global float2* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=localfloat2
// Implement fract directly
__attribute__((__overloadable__))
float2 _cl_fract(float2 x0, local float2* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=privatefloat2
// Implement fract directly
__attribute__((__overloadable__))
float2 _cl_fract(float2 x0, private float2* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=globalfloat3
// Implement fract directly
__attribute__((__overloadable__))
float3 _cl_fract(float3 x0, global float3* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=localfloat3
// Implement fract directly
__attribute__((__overloadable__))
float3 _cl_fract(float3 x0, local float3* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=privatefloat3
// Implement fract directly
__attribute__((__overloadable__))
float3 _cl_fract(float3 x0, private float3* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=globalfloat4
// Implement fract directly
__attribute__((__overloadable__))
float4 _cl_fract(float4 x0, global float4* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=localfloat4
// Implement fract directly
__attribute__((__overloadable__))
float4 _cl_fract(float4 x0, local float4* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=privatefloat4
// Implement fract directly
__attribute__((__overloadable__))
float4 _cl_fract(float4 x0, private float4* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=globalfloat8
// Implement fract directly
__attribute__((__overloadable__))
float8 _cl_fract(float8 x0, global float8* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=localfloat8
// Implement fract directly
__attribute__((__overloadable__))
float8 _cl_fract(float8 x0, local float8* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=privatefloat8
// Implement fract directly
__attribute__((__overloadable__))
float8 _cl_fract(float8 x0, private float8* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=globalfloat16
// Implement fract directly
__attribute__((__overloadable__))
float16 _cl_fract(float16 x0, global float16* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=localfloat16
// Implement fract directly
__attribute__((__overloadable__))
float16 _cl_fract(float16 x0, local float16* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=privatefloat16
// Implement fract directly
__attribute__((__overloadable__))
float16 _cl_fract(float16 x0, private float16* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

#ifdef cl_khr_fp64

// fract: VF=globaldouble
// Implement fract directly
__attribute__((__overloadable__))
double _cl_fract(double x0, global double* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=localdouble
// Implement fract directly
__attribute__((__overloadable__))
double _cl_fract(double x0, local double* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=privatedouble
// Implement fract directly
__attribute__((__overloadable__))
double _cl_fract(double x0, private double* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=globaldouble2
// Implement fract directly
__attribute__((__overloadable__))
double2 _cl_fract(double2 x0, global double2* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=localdouble2
// Implement fract directly
__attribute__((__overloadable__))
double2 _cl_fract(double2 x0, local double2* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=privatedouble2
// Implement fract directly
__attribute__((__overloadable__))
double2 _cl_fract(double2 x0, private double2* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=globaldouble3
// Implement fract directly
__attribute__((__overloadable__))
double3 _cl_fract(double3 x0, global double3* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=localdouble3
// Implement fract directly
__attribute__((__overloadable__))
double3 _cl_fract(double3 x0, local double3* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=privatedouble3
// Implement fract directly
__attribute__((__overloadable__))
double3 _cl_fract(double3 x0, private double3* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=globaldouble4
// Implement fract directly
__attribute__((__overloadable__))
double4 _cl_fract(double4 x0, global double4* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=localdouble4
// Implement fract directly
__attribute__((__overloadable__))
double4 _cl_fract(double4 x0, local double4* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=privatedouble4
// Implement fract directly
__attribute__((__overloadable__))
double4 _cl_fract(double4 x0, private double4* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=globaldouble8
// Implement fract directly
__attribute__((__overloadable__))
double8 _cl_fract(double8 x0, global double8* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=localdouble8
// Implement fract directly
__attribute__((__overloadable__))
double8 _cl_fract(double8 x0, local double8* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=privatedouble8
// Implement fract directly
__attribute__((__overloadable__))
double8 _cl_fract(double8 x0, private double8* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=globaldouble16
// Implement fract directly
__attribute__((__overloadable__))
double16 _cl_fract(double16 x0, global double16* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=localdouble16
// Implement fract directly
__attribute__((__overloadable__))
double16 _cl_fract(double16 x0, local double16* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// fract: VF=privatedouble16
// Implement fract directly
__attribute__((__overloadable__))
double16 _cl_fract(double16 x0, private double16* x1)
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
  return *x1=floor(x0), fmin(x0-floor(x0), sizeof(scalar_t)==sizeof(float) ? (scalar_t)POCL_FRACT_MIN_F : (scalar_t)POCL_FRACT_MIN);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

#endif // #ifdef cl_khr_fp64
