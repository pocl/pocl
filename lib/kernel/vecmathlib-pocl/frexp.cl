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



// frexp: ['VF', 'PVK'] -> VF

#ifdef cl_khr_fp16

// frexp: VF=globalhalf
// Implement frexp directly
__attribute__((__overloadable__))
half _cl_frexp(half x0, global int* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localhalf
// Implement frexp directly
__attribute__((__overloadable__))
half _cl_frexp(half x0, local int* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatehalf
// Implement frexp directly
__attribute__((__overloadable__))
half _cl_frexp(half x0, private int* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globalhalf2
// Implement frexp directly
__attribute__((__overloadable__))
half2 _cl_frexp(half2 x0, global int2* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localhalf2
// Implement frexp directly
__attribute__((__overloadable__))
half2 _cl_frexp(half2 x0, local int2* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatehalf2
// Implement frexp directly
__attribute__((__overloadable__))
half2 _cl_frexp(half2 x0, private int2* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globalhalf3
// Implement frexp directly
__attribute__((__overloadable__))
half3 _cl_frexp(half3 x0, global int3* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localhalf3
// Implement frexp directly
__attribute__((__overloadable__))
half3 _cl_frexp(half3 x0, local int3* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatehalf3
// Implement frexp directly
__attribute__((__overloadable__))
half3 _cl_frexp(half3 x0, private int3* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globalhalf4
// Implement frexp directly
__attribute__((__overloadable__))
half4 _cl_frexp(half4 x0, global int4* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localhalf4
// Implement frexp directly
__attribute__((__overloadable__))
half4 _cl_frexp(half4 x0, local int4* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatehalf4
// Implement frexp directly
__attribute__((__overloadable__))
half4 _cl_frexp(half4 x0, private int4* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globalhalf8
// Implement frexp directly
__attribute__((__overloadable__))
half8 _cl_frexp(half8 x0, global int8* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localhalf8
// Implement frexp directly
__attribute__((__overloadable__))
half8 _cl_frexp(half8 x0, local int8* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatehalf8
// Implement frexp directly
__attribute__((__overloadable__))
half8 _cl_frexp(half8 x0, private int8* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globalhalf16
// Implement frexp directly
__attribute__((__overloadable__))
half16 _cl_frexp(half16 x0, global int16* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localhalf16
// Implement frexp directly
__attribute__((__overloadable__))
half16 _cl_frexp(half16 x0, local int16* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatehalf16
// Implement frexp directly
__attribute__((__overloadable__))
half16 _cl_frexp(half16 x0, private int16* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globalfloat
// Implement frexp directly
__attribute__((__overloadable__))
float _cl_frexp(float x0, global int* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localfloat
// Implement frexp directly
__attribute__((__overloadable__))
float _cl_frexp(float x0, local int* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatefloat
// Implement frexp directly
__attribute__((__overloadable__))
float _cl_frexp(float x0, private int* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globalfloat2
// Implement frexp directly
__attribute__((__overloadable__))
float2 _cl_frexp(float2 x0, global int2* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localfloat2
// Implement frexp directly
__attribute__((__overloadable__))
float2 _cl_frexp(float2 x0, local int2* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatefloat2
// Implement frexp directly
__attribute__((__overloadable__))
float2 _cl_frexp(float2 x0, private int2* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globalfloat3
// Implement frexp directly
__attribute__((__overloadable__))
float3 _cl_frexp(float3 x0, global int3* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localfloat3
// Implement frexp directly
__attribute__((__overloadable__))
float3 _cl_frexp(float3 x0, local int3* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatefloat3
// Implement frexp directly
__attribute__((__overloadable__))
float3 _cl_frexp(float3 x0, private int3* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globalfloat4
// Implement frexp directly
__attribute__((__overloadable__))
float4 _cl_frexp(float4 x0, global int4* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localfloat4
// Implement frexp directly
__attribute__((__overloadable__))
float4 _cl_frexp(float4 x0, local int4* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatefloat4
// Implement frexp directly
__attribute__((__overloadable__))
float4 _cl_frexp(float4 x0, private int4* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globalfloat8
// Implement frexp directly
__attribute__((__overloadable__))
float8 _cl_frexp(float8 x0, global int8* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localfloat8
// Implement frexp directly
__attribute__((__overloadable__))
float8 _cl_frexp(float8 x0, local int8* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatefloat8
// Implement frexp directly
__attribute__((__overloadable__))
float8 _cl_frexp(float8 x0, private int8* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globalfloat16
// Implement frexp directly
__attribute__((__overloadable__))
float16 _cl_frexp(float16 x0, global int16* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localfloat16
// Implement frexp directly
__attribute__((__overloadable__))
float16 _cl_frexp(float16 x0, local int16* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatefloat16
// Implement frexp directly
__attribute__((__overloadable__))
float16 _cl_frexp(float16 x0, private int16* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globaldouble
// Implement frexp directly
__attribute__((__overloadable__))
double _cl_frexp(double x0, global int* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localdouble
// Implement frexp directly
__attribute__((__overloadable__))
double _cl_frexp(double x0, local int* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatedouble
// Implement frexp directly
__attribute__((__overloadable__))
double _cl_frexp(double x0, private int* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globaldouble2
// Implement frexp directly
__attribute__((__overloadable__))
double2 _cl_frexp(double2 x0, global int2* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localdouble2
// Implement frexp directly
__attribute__((__overloadable__))
double2 _cl_frexp(double2 x0, local int2* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatedouble2
// Implement frexp directly
__attribute__((__overloadable__))
double2 _cl_frexp(double2 x0, private int2* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globaldouble3
// Implement frexp directly
__attribute__((__overloadable__))
double3 _cl_frexp(double3 x0, global int3* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localdouble3
// Implement frexp directly
__attribute__((__overloadable__))
double3 _cl_frexp(double3 x0, local int3* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatedouble3
// Implement frexp directly
__attribute__((__overloadable__))
double3 _cl_frexp(double3 x0, private int3* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globaldouble4
// Implement frexp directly
__attribute__((__overloadable__))
double4 _cl_frexp(double4 x0, global int4* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localdouble4
// Implement frexp directly
__attribute__((__overloadable__))
double4 _cl_frexp(double4 x0, local int4* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatedouble4
// Implement frexp directly
__attribute__((__overloadable__))
double4 _cl_frexp(double4 x0, private int4* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globaldouble8
// Implement frexp directly
__attribute__((__overloadable__))
double8 _cl_frexp(double8 x0, global int8* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localdouble8
// Implement frexp directly
__attribute__((__overloadable__))
double8 _cl_frexp(double8 x0, local int8* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatedouble8
// Implement frexp directly
__attribute__((__overloadable__))
double8 _cl_frexp(double8 x0, private int8* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=globaldouble16
// Implement frexp directly
__attribute__((__overloadable__))
double16 _cl_frexp(double16 x0, global int16* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=localdouble16
// Implement frexp directly
__attribute__((__overloadable__))
double16 _cl_frexp(double16 x0, local int16* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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

// frexp: VF=privatedouble16
// Implement frexp directly
__attribute__((__overloadable__))
double16 _cl_frexp(double16 x0, private int16* x1)
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
      kvector_t e0 = ilogb(x0);
      kvector_t e = e0==INT_MIN || e0==INT_MAX ? (kvector_t)0 : e0+1;
      *x1 = e;
      convert_ivector_t(e0==INT_MIN || e0==INT_MAX) ? x0 : ldexp(x0, -e);
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
