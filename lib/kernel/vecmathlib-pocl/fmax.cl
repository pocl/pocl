// Note: This file has been automatically generated. Do not modify.

// Needed for fract()
#define POCL_FRACT_MIN_H 0x1.ffcp-1h
#define POCL_FRACT_MIN   0x1.fffffffffffffp-1
#define POCL_FRACT_MIN_F 0x1.fffffep-1f

// Choose a constant with a particular precision
#ifdef cl_khr_fp16
#  define IF_HALF(TYPE, VAL, OTHER) \
          (sizeof(TYPE)==sizeof(half) ? (TYPE)(VAL) : (TYPE)(OTHER))
#else
#  define IF_HALF(TYPE, VAL, OTHER) (OTHER)
#endif

#ifdef cl_khr_fp64
#  define IF_DOUBLE(TYPE, VAL, OTHER) \
          (sizeof(TYPE)==sizeof(double) ? (TYPE)(VAL) : (TYPE)(OTHER))
#else
#  define IF_DOUBLE(TYPE, VAL, OTHER) (OTHER)
#endif

#define TYPED_CONST(TYPE, HALF_VAL, SINGLE_VAL, DOUBLE_VAL) \
        IF_HALF(TYPE, HALF_VAL, IF_DOUBLE(TYPE, DOUBLE_VAL, SINGLE_VAL))



// fmax: ['VF', 'SF'] -> VF

#ifdef cl_khr_fp16

// fmax: VF=half2
// Implement fmax directly
__attribute__((__overloadable__))
half2 _cl_fmax(half2 x0, half x1)
{
  typedef short iscalar_t;
  typedef short jscalar_t;
  typedef int kscalar_t;
  typedef half scalar_t;
  typedef short2 ivector_t;
  typedef short2 jvector_t;
  typedef int2 kvector_t;
  typedef half2 vector_t;
#define convert_ivector_t convert_short2
#define convert_jvector_t convert_short2
#define convert_kvector_t convert_int2
#define convert_vector_t convert_half2
#define ilogb_ _cl_ilogb_half2
#define ldexp_scalar_ _cl_ldexp_half2_short
#define ldexp_vector_ _cl_ldexp_half2_short2
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// fmax: VF=half3
// Implement fmax directly
__attribute__((__overloadable__))
half3 _cl_fmax(half3 x0, half x1)
{
  typedef short iscalar_t;
  typedef short jscalar_t;
  typedef int kscalar_t;
  typedef half scalar_t;
  typedef short3 ivector_t;
  typedef short3 jvector_t;
  typedef int3 kvector_t;
  typedef half3 vector_t;
#define convert_ivector_t convert_short3
#define convert_jvector_t convert_short3
#define convert_kvector_t convert_int3
#define convert_vector_t convert_half3
#define ilogb_ _cl_ilogb_half3
#define ldexp_scalar_ _cl_ldexp_half3_short
#define ldexp_vector_ _cl_ldexp_half3_short3
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// fmax: VF=half4
// Implement fmax directly
__attribute__((__overloadable__))
half4 _cl_fmax(half4 x0, half x1)
{
  typedef short iscalar_t;
  typedef short jscalar_t;
  typedef int kscalar_t;
  typedef half scalar_t;
  typedef short4 ivector_t;
  typedef short4 jvector_t;
  typedef int4 kvector_t;
  typedef half4 vector_t;
#define convert_ivector_t convert_short4
#define convert_jvector_t convert_short4
#define convert_kvector_t convert_int4
#define convert_vector_t convert_half4
#define ilogb_ _cl_ilogb_half4
#define ldexp_scalar_ _cl_ldexp_half4_short
#define ldexp_vector_ _cl_ldexp_half4_short4
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// fmax: VF=half8
// Implement fmax directly
__attribute__((__overloadable__))
half8 _cl_fmax(half8 x0, half x1)
{
  typedef short iscalar_t;
  typedef short jscalar_t;
  typedef int kscalar_t;
  typedef half scalar_t;
  typedef short8 ivector_t;
  typedef short8 jvector_t;
  typedef int8 kvector_t;
  typedef half8 vector_t;
#define convert_ivector_t convert_short8
#define convert_jvector_t convert_short8
#define convert_kvector_t convert_int8
#define convert_vector_t convert_half8
#define ilogb_ _cl_ilogb_half8
#define ldexp_scalar_ _cl_ldexp_half8_short
#define ldexp_vector_ _cl_ldexp_half8_short8
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// fmax: VF=half16
// Implement fmax directly
__attribute__((__overloadable__))
half16 _cl_fmax(half16 x0, half x1)
{
  typedef short iscalar_t;
  typedef short jscalar_t;
  typedef int kscalar_t;
  typedef half scalar_t;
  typedef short16 ivector_t;
  typedef short16 jvector_t;
  typedef int16 kvector_t;
  typedef half16 vector_t;
#define convert_ivector_t convert_short16
#define convert_jvector_t convert_short16
#define convert_kvector_t convert_int16
#define convert_vector_t convert_half16
#define ilogb_ _cl_ilogb_half16
#define ldexp_scalar_ _cl_ldexp_half16_short
#define ldexp_vector_ _cl_ldexp_half16_short16
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

#endif // #ifdef cl_khr_fp16

// fmax: VF=float2
// Implement fmax directly
__attribute__((__overloadable__))
float2 _cl_fmax(float2 x0, float x1)
{
  typedef int iscalar_t;
  typedef int jscalar_t;
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
#define ilogb_ _cl_ilogb_float2
#define ldexp_scalar_ _cl_ldexp_float2_int
#define ldexp_vector_ _cl_ldexp_float2_int2
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// fmax: VF=float3
// Implement fmax directly
__attribute__((__overloadable__))
float3 _cl_fmax(float3 x0, float x1)
{
  typedef int iscalar_t;
  typedef int jscalar_t;
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
#define ilogb_ _cl_ilogb_float3
#define ldexp_scalar_ _cl_ldexp_float3_int
#define ldexp_vector_ _cl_ldexp_float3_int3
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// fmax: VF=float4
// Implement fmax directly
__attribute__((__overloadable__))
float4 _cl_fmax(float4 x0, float x1)
{
  typedef int iscalar_t;
  typedef int jscalar_t;
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
#define ilogb_ _cl_ilogb_float4
#define ldexp_scalar_ _cl_ldexp_float4_int
#define ldexp_vector_ _cl_ldexp_float4_int4
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// fmax: VF=float8
// Implement fmax directly
__attribute__((__overloadable__))
float8 _cl_fmax(float8 x0, float x1)
{
  typedef int iscalar_t;
  typedef int jscalar_t;
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
#define ilogb_ _cl_ilogb_float8
#define ldexp_scalar_ _cl_ldexp_float8_int
#define ldexp_vector_ _cl_ldexp_float8_int8
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// fmax: VF=float16
// Implement fmax directly
__attribute__((__overloadable__))
float16 _cl_fmax(float16 x0, float x1)
{
  typedef int iscalar_t;
  typedef int jscalar_t;
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
#define ilogb_ _cl_ilogb_float16
#define ldexp_scalar_ _cl_ldexp_float16_int
#define ldexp_vector_ _cl_ldexp_float16_int16
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

#ifdef cl_khr_fp64

// fmax: VF=double2
// Implement fmax directly
__attribute__((__overloadable__))
double2 _cl_fmax(double2 x0, double x1)
{
  typedef long iscalar_t;
  typedef long jscalar_t;
  typedef int kscalar_t;
  typedef double scalar_t;
  typedef long2 ivector_t;
  typedef long2 jvector_t;
  typedef int2 kvector_t;
  typedef double2 vector_t;
#define convert_ivector_t convert_long2
#define convert_jvector_t convert_long2
#define convert_kvector_t convert_int2
#define convert_vector_t convert_double2
#define ilogb_ _cl_ilogb_double2
#define ldexp_scalar_ _cl_ldexp_double2_long
#define ldexp_vector_ _cl_ldexp_double2_long2
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// fmax: VF=double3
// Implement fmax directly
__attribute__((__overloadable__))
double3 _cl_fmax(double3 x0, double x1)
{
  typedef long iscalar_t;
  typedef long jscalar_t;
  typedef int kscalar_t;
  typedef double scalar_t;
  typedef long3 ivector_t;
  typedef long3 jvector_t;
  typedef int3 kvector_t;
  typedef double3 vector_t;
#define convert_ivector_t convert_long3
#define convert_jvector_t convert_long3
#define convert_kvector_t convert_int3
#define convert_vector_t convert_double3
#define ilogb_ _cl_ilogb_double3
#define ldexp_scalar_ _cl_ldexp_double3_long
#define ldexp_vector_ _cl_ldexp_double3_long3
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// fmax: VF=double4
// Implement fmax directly
__attribute__((__overloadable__))
double4 _cl_fmax(double4 x0, double x1)
{
  typedef long iscalar_t;
  typedef long jscalar_t;
  typedef int kscalar_t;
  typedef double scalar_t;
  typedef long4 ivector_t;
  typedef long4 jvector_t;
  typedef int4 kvector_t;
  typedef double4 vector_t;
#define convert_ivector_t convert_long4
#define convert_jvector_t convert_long4
#define convert_kvector_t convert_int4
#define convert_vector_t convert_double4
#define ilogb_ _cl_ilogb_double4
#define ldexp_scalar_ _cl_ldexp_double4_long
#define ldexp_vector_ _cl_ldexp_double4_long4
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// fmax: VF=double8
// Implement fmax directly
__attribute__((__overloadable__))
double8 _cl_fmax(double8 x0, double x1)
{
  typedef long iscalar_t;
  typedef long jscalar_t;
  typedef int kscalar_t;
  typedef double scalar_t;
  typedef long8 ivector_t;
  typedef long8 jvector_t;
  typedef int8 kvector_t;
  typedef double8 vector_t;
#define convert_ivector_t convert_long8
#define convert_jvector_t convert_long8
#define convert_kvector_t convert_int8
#define convert_vector_t convert_double8
#define ilogb_ _cl_ilogb_double8
#define ldexp_scalar_ _cl_ldexp_double8_long
#define ldexp_vector_ _cl_ldexp_double8_long8
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// fmax: VF=double16
// Implement fmax directly
__attribute__((__overloadable__))
double16 _cl_fmax(double16 x0, double x1)
{
  typedef long iscalar_t;
  typedef long jscalar_t;
  typedef int kscalar_t;
  typedef double scalar_t;
  typedef long16 ivector_t;
  typedef long16 jvector_t;
  typedef int16 kvector_t;
  typedef double16 vector_t;
#define convert_ivector_t convert_long16
#define convert_jvector_t convert_long16
#define convert_kvector_t convert_int16
#define convert_vector_t convert_double16
#define ilogb_ _cl_ilogb_double16
#define ldexp_scalar_ _cl_ldexp_double16_long
#define ldexp_vector_ _cl_ldexp_double16_long16
  return fmax(x0,(vector_t)x1);
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

#endif // #ifdef cl_khr_fp64
