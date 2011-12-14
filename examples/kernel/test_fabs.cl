#define IMPLEMENT_BODY_V(NAME, BODY, VTYPE, STYPE, JTYPE, SJTYPE)       \
  void NAME##_##VTYPE()                                                 \
  {                                                                     \
    typedef VTYPE vtype;                                                \
    typedef STYPE stype;                                                \
    typedef JTYPE jtype;                                                \
    typedef SJTYPE sjtype;                                              \
    char const *const typename = #VTYPE;                                \
    BODY;                                                               \
  }
#define DEFINE_BODY_V(NAME, EXPR)                               \
  IMPLEMENT_BODY_V(NAME, EXPR, float   , float , int   , int )  \
  IMPLEMENT_BODY_V(NAME, EXPR, float2  , float , int2  , int )  \
  IMPLEMENT_BODY_V(NAME, EXPR, float3  , float , int3  , int )  \
  IMPLEMENT_BODY_V(NAME, EXPR, float4  , float , int4  , int )  \
  IMPLEMENT_BODY_V(NAME, EXPR, float8  , float , int8  , int )  \
  IMPLEMENT_BODY_V(NAME, EXPR, float16 , float , int16 , int )  \
  __IF_FP64(                                                    \
  IMPLEMENT_BODY_V(NAME, EXPR, double  , double, long  , long)  \
  IMPLEMENT_BODY_V(NAME, EXPR, double2 , double, long2 , long)  \
  IMPLEMENT_BODY_V(NAME, EXPR, double3 , double, long3 , long)  \
  IMPLEMENT_BODY_V(NAME, EXPR, double4 , double, long4 , long)  \
  IMPLEMENT_BODY_V(NAME, EXPR, double8 , double, long8 , long)  \
  IMPLEMENT_BODY_V(NAME, EXPR, double16, double, long16, long))

#define CALL_FUNC_V(NAME)                       \
  NAME##_float   ();                            \
  NAME##_float2  ();                            \
  NAME##_float3  ();                            \
  NAME##_float4  ();                            \
  NAME##_float8  ();                            \
  NAME##_float16 ();                            \
  __IF_FP64(                                    \
  NAME##_double  ();                            \
  NAME##_double2 ();                            \
  NAME##_double3 ();                            \
  NAME##_double4 ();                            \
  NAME##_double8 ();                            \
  NAME##_double16();)



#define is_signed(T)   ((T)-1 < (T)+1)
#define is_floating(T) ((T)0.1 > (T)0.0)
#define count_bits(T)  (CHAR_BIT * sizeof(T))

DEFINE_BODY_V
(test_fabs,
 ({
   _cl_static_assert(stype, is_floating(stype));
   float const values[] = {
     0.0f,
     0.1f,
     0.9f,
     1.0f,
     1.1f,
     10.0f,
     1000000.0f,
     1000000000000.0f,
     MAXFLOAT,
     HUGE_VALF,
     INFINITY,
     /* NAN,   a nan has no specific sign */
     FLT_MAX,
     FLT_MIN,
     FLT_EPSILON,
   };
   int const nvalues = sizeof(values) / sizeof(*values);
   int ninputs = 1;
#ifdef cl_khr_fp64
   double const dvalues[] = {
     0.0,
     0.1,
     0.9,
     1.0,
     1.1,
     10.0,
     1000000.0,
     1000000000000.0,
     1000000000000000000000000.0,
     HUGE_VAL,
     INFINITY,
     /* NAN,   a nan has no specific sign */
     DBL_MAX,
     DBL_MIN,
     DBL_EPSILON,
   };
   int const ndvalues = sizeof(dvalues) / sizeof(*dvalues);
   ++ninputs;
#endif
   
   for (int input=0; input<ninputs; ++input) {
     for (int iter=0; iter<nvalues; ++iter) {
       for (int sign=-1; sign<=+1; sign+=2) {
         typedef union {
           vtype v;
           stype s[16];
         } Tvec;
         Tvec val, good, val2;
         int vecsize = vec_step(vtype);
         for (int n=0; n<vecsize; ++n) {
           if (input==0) {
             val.s[n]  = sign * values[(iter+n) % nvalues];
             good.s[n] =        values[(iter+n) % nvalues];
             val2.s[n] =        values[(iter+n+1) % nvalues];
           } else {
#ifdef cl_khr_fp64
             val.s[n]  = sign * dvalues[(iter+n) % ndvalues];
             good.s[n] =        dvalues[(iter+n) % ndvalues];
             val2.s[n] =        dvalues[(iter+n+1) % ndvalues];
#endif
           }
         }
         Tvec res;
         bool equal;
         typedef union {
           stype  s;
           sjtype sj;
         } S;
         typedef union {
           jtype  v;
           sjtype s[16];
         } Jvec;
         /* fabs */
         res.v = fabs(val.v);
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           S r, g;
           r.s = res.s[n];
           g.s = good.s[n];
           equal = equal && r.sj == g.sj;
         }
         if (!equal) {
           printf("FAIL: fabs type=%s val=%.17g res=%.17g\n",
                  typename, val.s[0], res.s[0]);
         }
         /* signbit */
         Jvec ires;
         ires.v = signbit(val.v);
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           equal = equal && ires.s[n] == (sign>0 ? 0 : vecsize==1 ? +1 : -1);
         }
         if (!equal) {
           printf("FAIL: signbit type=%s val=%.17g res=%d\n",
                  typename, val.s[0], (int)ires.s[0]);
         }
         /* copysign */
         for (int sign2=-1; sign2<=+1; sign2+=2) {
           res.v = copysign(val.v, (stype)sign2*val2.v);
           equal = true;
           for (int n=0; n<vecsize; ++n) {
             S r, g;
             r.s = res.s[n];
             g.s = sign2*good.s[n];
             equal = equal && r.sj == g.sj;
           }
           if (!equal) {
             for (int n=0; n<vecsize; ++n) {
               printf("FAIL: copysign type=%s val=%.17g sign=%.17g res=%.17g\n",
                      typename, val.s[n], sign2*val2.s[n], res.s[n]);
             }
           }
         }
       }
     }
   }
 })
 )

void test_fabs()
{
  CALL_FUNC_V(test_fabs)
}
