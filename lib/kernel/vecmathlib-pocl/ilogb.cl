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



// ilogb: ['VF'] -> VK

#ifdef cl_khr_fp16

// ilogb: VF=half
// Implement ilogb directly
__attribute__((__overloadable__))
int _cl_ilogb(half x0)
{
  typedef short iscalar_t;
  typedef int jscalar_t;
  typedef int kscalar_t;
  typedef half scalar_t;
  typedef short ivector_t;
  typedef int jvector_t;
  typedef int kvector_t;
  typedef half vector_t;
#define convert_ivector_t convert_short
#define convert_jvector_t convert_int
#define convert_kvector_t convert_int
#define convert_vector_t convert_half
#define ilogb_ _cl_ilogb_half
#define ldexp_scalar_ _cl_ldexp_half_short
#define ldexp_vector_ _cl_ldexp_half_short
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=half2
// Implement ilogb directly
__attribute__((__overloadable__))
int2 _cl_ilogb(half2 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=half3
// Implement ilogb directly
__attribute__((__overloadable__))
int3 _cl_ilogb(half3 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=half4
// Implement ilogb directly
__attribute__((__overloadable__))
int4 _cl_ilogb(half4 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=half8
// Implement ilogb directly
__attribute__((__overloadable__))
int8 _cl_ilogb(half8 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=half16
// Implement ilogb directly
__attribute__((__overloadable__))
int16 _cl_ilogb(half16 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

#endif // #ifdef cl_khr_fp16

// ilogb: VF=float
// Implement ilogb directly
__attribute__((__overloadable__))
int _cl_ilogb(float x0)
{
  typedef int iscalar_t;
  typedef int jscalar_t;
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
#define ilogb_ _cl_ilogb_float
#define ldexp_scalar_ _cl_ldexp_float_int
#define ldexp_vector_ _cl_ldexp_float_int
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=float2
// Implement ilogb directly
__attribute__((__overloadable__))
int2 _cl_ilogb(float2 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=float3
// Implement ilogb directly
__attribute__((__overloadable__))
int3 _cl_ilogb(float3 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=float4
// Implement ilogb directly
__attribute__((__overloadable__))
int4 _cl_ilogb(float4 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=float8
// Implement ilogb directly
__attribute__((__overloadable__))
int8 _cl_ilogb(float8 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=float16
// Implement ilogb directly
__attribute__((__overloadable__))
int16 _cl_ilogb(float16 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

#ifdef cl_khr_fp64

// ilogb: VF=double
// Implement ilogb directly
__attribute__((__overloadable__))
int _cl_ilogb(double x0)
{
  typedef long iscalar_t;
  typedef int jscalar_t;
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
#define ilogb_ _cl_ilogb_double
#define ldexp_scalar_ _cl_ldexp_double_long
#define ldexp_vector_ _cl_ldexp_double_long
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=double2
// Implement ilogb directly
__attribute__((__overloadable__))
int2 _cl_ilogb(double2 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=double3
// Implement ilogb directly
__attribute__((__overloadable__))
int3 _cl_ilogb(double3 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=double4
// Implement ilogb directly
__attribute__((__overloadable__))
int4 _cl_ilogb(double4 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=double8
// Implement ilogb directly
__attribute__((__overloadable__))
int8 _cl_ilogb(double8 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

// ilogb: VF=double16
// Implement ilogb directly
__attribute__((__overloadable__))
int16 _cl_ilogb(double16 x0)
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
  return 
    ({
      ivector_t ilogb_(vector_t);
      ivector_t r = ilogb_(x0);
      iscalar_t jmin = TYPED_CONST(iscalar_t, SHORT_MIN, INT_MIN, LONG_MIN);
      iscalar_t jmax = TYPED_CONST(iscalar_t, SHORT_MAX, INT_MAX, LONG_MAX);
      r = r==jmin ? (ivector_t)INT_MIN : r;
      r = r==jmax ? (ivector_t)INT_MAX : r;
      convert_kvector_t(r);
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
#undef ilogb_
#undef ldexp_scalar_
#undef ldexp_vector_
}

#endif // #ifdef cl_khr_fp64
