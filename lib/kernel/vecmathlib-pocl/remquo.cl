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



// remquo: ['VF', 'VF', 'PVK'] -> VF

#ifdef cl_khr_fp16

// remquo: VF=globalhalf
// Implement remquo directly
__attribute__((__overloadable__))
half _cl_remquo(half x0, half x1, global int* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localhalf
// Implement remquo directly
__attribute__((__overloadable__))
half _cl_remquo(half x0, half x1, local int* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatehalf
// Implement remquo directly
__attribute__((__overloadable__))
half _cl_remquo(half x0, half x1, private int* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globalhalf2
// Implement remquo directly
__attribute__((__overloadable__))
half2 _cl_remquo(half2 x0, half2 x1, global int2* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localhalf2
// Implement remquo directly
__attribute__((__overloadable__))
half2 _cl_remquo(half2 x0, half2 x1, local int2* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatehalf2
// Implement remquo directly
__attribute__((__overloadable__))
half2 _cl_remquo(half2 x0, half2 x1, private int2* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globalhalf3
// Implement remquo directly
__attribute__((__overloadable__))
half3 _cl_remquo(half3 x0, half3 x1, global int3* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localhalf3
// Implement remquo directly
__attribute__((__overloadable__))
half3 _cl_remquo(half3 x0, half3 x1, local int3* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatehalf3
// Implement remquo directly
__attribute__((__overloadable__))
half3 _cl_remquo(half3 x0, half3 x1, private int3* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globalhalf4
// Implement remquo directly
__attribute__((__overloadable__))
half4 _cl_remquo(half4 x0, half4 x1, global int4* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localhalf4
// Implement remquo directly
__attribute__((__overloadable__))
half4 _cl_remquo(half4 x0, half4 x1, local int4* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatehalf4
// Implement remquo directly
__attribute__((__overloadable__))
half4 _cl_remquo(half4 x0, half4 x1, private int4* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globalhalf8
// Implement remquo directly
__attribute__((__overloadable__))
half8 _cl_remquo(half8 x0, half8 x1, global int8* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localhalf8
// Implement remquo directly
__attribute__((__overloadable__))
half8 _cl_remquo(half8 x0, half8 x1, local int8* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatehalf8
// Implement remquo directly
__attribute__((__overloadable__))
half8 _cl_remquo(half8 x0, half8 x1, private int8* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globalhalf16
// Implement remquo directly
__attribute__((__overloadable__))
half16 _cl_remquo(half16 x0, half16 x1, global int16* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localhalf16
// Implement remquo directly
__attribute__((__overloadable__))
half16 _cl_remquo(half16 x0, half16 x1, local int16* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatehalf16
// Implement remquo directly
__attribute__((__overloadable__))
half16 _cl_remquo(half16 x0, half16 x1, private int16* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

#endif // #ifdef cl_khr_fp16

// remquo: VF=globalfloat
// Implement remquo directly
__attribute__((__overloadable__))
float _cl_remquo(float x0, float x1, global int* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localfloat
// Implement remquo directly
__attribute__((__overloadable__))
float _cl_remquo(float x0, float x1, local int* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatefloat
// Implement remquo directly
__attribute__((__overloadable__))
float _cl_remquo(float x0, float x1, private int* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globalfloat2
// Implement remquo directly
__attribute__((__overloadable__))
float2 _cl_remquo(float2 x0, float2 x1, global int2* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localfloat2
// Implement remquo directly
__attribute__((__overloadable__))
float2 _cl_remquo(float2 x0, float2 x1, local int2* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatefloat2
// Implement remquo directly
__attribute__((__overloadable__))
float2 _cl_remquo(float2 x0, float2 x1, private int2* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globalfloat3
// Implement remquo directly
__attribute__((__overloadable__))
float3 _cl_remquo(float3 x0, float3 x1, global int3* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localfloat3
// Implement remquo directly
__attribute__((__overloadable__))
float3 _cl_remquo(float3 x0, float3 x1, local int3* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatefloat3
// Implement remquo directly
__attribute__((__overloadable__))
float3 _cl_remquo(float3 x0, float3 x1, private int3* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globalfloat4
// Implement remquo directly
__attribute__((__overloadable__))
float4 _cl_remquo(float4 x0, float4 x1, global int4* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localfloat4
// Implement remquo directly
__attribute__((__overloadable__))
float4 _cl_remquo(float4 x0, float4 x1, local int4* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatefloat4
// Implement remquo directly
__attribute__((__overloadable__))
float4 _cl_remquo(float4 x0, float4 x1, private int4* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globalfloat8
// Implement remquo directly
__attribute__((__overloadable__))
float8 _cl_remquo(float8 x0, float8 x1, global int8* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localfloat8
// Implement remquo directly
__attribute__((__overloadable__))
float8 _cl_remquo(float8 x0, float8 x1, local int8* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatefloat8
// Implement remquo directly
__attribute__((__overloadable__))
float8 _cl_remquo(float8 x0, float8 x1, private int8* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globalfloat16
// Implement remquo directly
__attribute__((__overloadable__))
float16 _cl_remquo(float16 x0, float16 x1, global int16* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localfloat16
// Implement remquo directly
__attribute__((__overloadable__))
float16 _cl_remquo(float16 x0, float16 x1, local int16* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatefloat16
// Implement remquo directly
__attribute__((__overloadable__))
float16 _cl_remquo(float16 x0, float16 x1, private int16* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

#ifdef cl_khr_fp64

// remquo: VF=globaldouble
// Implement remquo directly
__attribute__((__overloadable__))
double _cl_remquo(double x0, double x1, global int* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localdouble
// Implement remquo directly
__attribute__((__overloadable__))
double _cl_remquo(double x0, double x1, local int* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatedouble
// Implement remquo directly
__attribute__((__overloadable__))
double _cl_remquo(double x0, double x1, private int* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globaldouble2
// Implement remquo directly
__attribute__((__overloadable__))
double2 _cl_remquo(double2 x0, double2 x1, global int2* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localdouble2
// Implement remquo directly
__attribute__((__overloadable__))
double2 _cl_remquo(double2 x0, double2 x1, local int2* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatedouble2
// Implement remquo directly
__attribute__((__overloadable__))
double2 _cl_remquo(double2 x0, double2 x1, private int2* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globaldouble3
// Implement remquo directly
__attribute__((__overloadable__))
double3 _cl_remquo(double3 x0, double3 x1, global int3* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localdouble3
// Implement remquo directly
__attribute__((__overloadable__))
double3 _cl_remquo(double3 x0, double3 x1, local int3* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatedouble3
// Implement remquo directly
__attribute__((__overloadable__))
double3 _cl_remquo(double3 x0, double3 x1, private int3* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globaldouble4
// Implement remquo directly
__attribute__((__overloadable__))
double4 _cl_remquo(double4 x0, double4 x1, global int4* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localdouble4
// Implement remquo directly
__attribute__((__overloadable__))
double4 _cl_remquo(double4 x0, double4 x1, local int4* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatedouble4
// Implement remquo directly
__attribute__((__overloadable__))
double4 _cl_remquo(double4 x0, double4 x1, private int4* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globaldouble8
// Implement remquo directly
__attribute__((__overloadable__))
double8 _cl_remquo(double8 x0, double8 x1, global int8* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localdouble8
// Implement remquo directly
__attribute__((__overloadable__))
double8 _cl_remquo(double8 x0, double8 x1, local int8* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatedouble8
// Implement remquo directly
__attribute__((__overloadable__))
double8 _cl_remquo(double8 x0, double8 x1, private int8* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=globaldouble16
// Implement remquo directly
__attribute__((__overloadable__))
double16 _cl_remquo(double16 x0, double16 x1, global int16* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=localdouble16
// Implement remquo directly
__attribute__((__overloadable__))
double16 _cl_remquo(double16 x0, double16 x1, local int16* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

// remquo: VF=privatedouble16
// Implement remquo directly
__attribute__((__overloadable__))
double16 _cl_remquo(double16 x0, double16 x1, private int16* x2)
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
  return 
    ({
      vector_t k = rint(x0/x1);
      *x2 = (convert_kvector_t(k) & 0x7f) * (1-2*convert_kvector_t(signbit(k)));
      x0-k*x1;
    })
;
#undef convert_ivector_t
#undef convert_jvector_t
#undef convert_kvector_t
#undef convert_vector_t
}

#endif // #ifdef cl_khr_fp64
