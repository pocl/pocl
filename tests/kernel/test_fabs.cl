// TESTING: copysign
// TESTING: fabs
// TESTING: isfinite
// TESTING: isinf
// TESTING: isnan
// TESTING: isnormal
// TESTING: signbit
// TESTING: ilogb
// TESTING: ldexp

typedef constant char* string;

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define IMPLEMENT_BODY_V(NAME, BODY, SIZE,                              \
                         VTYPE, STYPE, ITYPE, SITYPE, JTYPE, SJTYPE)    \
  void NAME##_##VTYPE()                                                 \
  {                                                                     \
    typedef VTYPE vtype;                                                \
    typedef STYPE stype;                                                \
    typedef ITYPE itype;                                                \
    typedef SITYPE sitype;                                              \
    typedef JTYPE jtype;                                                \
    typedef SJTYPE sjtype;                                              \
    string const typename = #VTYPE;                                     \
    const int vecsize = SIZE;                                           \
    BODY;                                                               \
  }
#define DEFINE_BODY_V(NAME, EXPR)                                       \
  IMPLEMENT_BODY_V(NAME, EXPR,  1, float   , float , int  , int, int   , int ) \
  IMPLEMENT_BODY_V(NAME, EXPR,  2, float2  , float , int2 , int, int2  , int ) \
  IMPLEMENT_BODY_V(NAME, EXPR,  3, float3  , float , int3 , int, int3  , int ) \
  IMPLEMENT_BODY_V(NAME, EXPR,  4, float4  , float , int4 , int, int4  , int ) \
  IMPLEMENT_BODY_V(NAME, EXPR,  8, float8  , float , int8 , int, int8  , int ) \
  IMPLEMENT_BODY_V(NAME, EXPR, 16, float16 , float , int16, int, int16 , int ) \
  __IF_FP64(                                                            \
  IMPLEMENT_BODY_V(NAME, EXPR,  1, double  , double, int  , int, long  , long) \
  IMPLEMENT_BODY_V(NAME, EXPR,  2, double2 , double, int2 , int, long2 , long) \
  IMPLEMENT_BODY_V(NAME, EXPR,  3, double3 , double, int3 , int, long3 , long) \
  IMPLEMENT_BODY_V(NAME, EXPR,  4, double4 , double, int4 , int, long4 , long) \
  IMPLEMENT_BODY_V(NAME, EXPR,  8, double8 , double, int8 , int, long8 , long) \
  IMPLEMENT_BODY_V(NAME, EXPR, 16, double16, double, int16, int, long16, long))

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

#if __has_extension(c_generic_selections) && (__clang_major__ < 6)
 #ifdef cl_khr_fp64
 # define is_floating(T) _Generic((T)0, float: 1, double: 1, default: 0)
 #else
 # define is_floating(T) _Generic((T)0, float: 1, default: 0)
 #endif 
#else
# define is_floating(T) 1
#endif
#define is_signed(T)   ((T)-1 < (T)+1)
#define count_bits(T)  (CHAR_BIT * sizeof(T))

#define ISNAN(x) (isnan(x) || as_int((float)(x)) == as_int((float)NAN))
#define ISEQ(x, y) (ISNAN(x) == ISNAN(y) && (ISNAN(x) || (x)==(y)))

DEFINE_BODY_V
(test_fabs,
 ({
   _CL_STATIC_ASSERT(stype, is_floating(stype));
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
     NAN,
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
     NAN,
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
           volatile stype s[16];
           /* s is declared as volatile to force all accesses through memory,
            * avoiding excess precision from x87 fpu registers
            * see https://github.com/pocl/pocl/issues/621, https://gcc.gnu.org/wiki/x87note
            */
         } Tvec;
         Tvec val, good, val2;
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
           itype  v;
           sitype s[16];
         } Ivec;
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
           equal = equal && (ISNAN(val.s[n]) || r.sj == g.sj);
         }
         if (!equal) {
           for (int n=0; n<vecsize; ++n) {
             printf("FAIL: fabs type=%s val=%.17g res=%.17g good=%.17g\n",
                    typename, val.s[n], res.s[n], good.s[n]);
           }
           return;
         }
         /* signbit */
         Jvec jres;
         jres.v = signbit(val.v);
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           equal = equal &&
             (ISNAN(val.s[n]) ||
              jres.s[n] == (sign>0 ? 0 : vecsize==1 ? +1 : -1));
         }
         if (!equal) {
           for (int n=0; n<vecsize; ++n) {
             printf("FAIL: signbit type=%s val=%.17g res=%d good=%d\n",
                    typename, val.s[n], (int)jres.s[n],
                    (sign>0 ? 0 : vecsize==1 ? +1 : -1));
           }
           return;
         }
         /* isfinite */
         jres.v = isfinite(val.v);
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           equal = equal &&
             jres.s[n] == (isfinite(val.s[n]) ? (vecsize==1 ? +1 : -1) : 0);
         }
         if (!equal) {
           for (int n=0; n<vecsize; ++n) {
             printf("FAIL: isfinite type=%s val=%.17g res=%d good=%d\n",
                    typename, val.s[n], (int)jres.s[n],
                    (isfinite(val.s[n]) ? (vecsize==1 ? +1 : -1) : 0));
           }
           return;
         }
         /* isinf */
         jres.v = isinf(val.v);
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           equal = equal &&
             jres.s[n] == (isinf(val.s[n]) ? (vecsize==1 ? +1 : -1) : 0);
         }
         if (!equal) {
           for (int n=0; n<vecsize; ++n) {
             printf("FAIL: isinf type=%s val=%.17g res=%d good=%d\n",
                    typename, val.s[n], (int)jres.s[n],
                    (isinf(val.s[n]) ? (vecsize==1 ? +1 : -1) : 0));
           }
           return;
         }
         /* isnan */
         jres.v = isnan(val.v);
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           equal = equal &&
             jres.s[n] == (isnan(val.s[n]) ? (vecsize==1 ? +1 : -1) : 0);
         }
         if (!equal) {
           for (int n=0; n<vecsize; ++n) {
             printf("FAIL: isnan type=%s val=%.17g res=%d good=%d\n",
                    typename, val.s[n], (int)jres.s[n],
                    (isnan(val.s[n]) ? (vecsize==1 ? +1 : -1) : 0));
           }
           return;
         }
         /* isnormal */
         jres.v = isnormal(val.v);
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           equal = equal &&
             jres.s[n] == (isnormal(val.s[n]) ? (vecsize==1 ? +1 : -1) : 0);
         }
         if (!equal) {
           for (int n=0; n<vecsize; ++n) {
             printf("FAIL: isnormal type=%s val=%.17g res=%d good=%d\n",
                    typename, val.s[n], (int)jres.s[n],
                    (isnormal(val.s[n]) ? (vecsize==1 ? +1 : -1) : 0));
           }
           return;
         }
         /* copysign */
         for (int sign2=-1; sign2<=+1; sign2+=2) {
           res.v = copysign(val.v, (stype)sign2*val2.v);
           equal = true;
           for (int n=0; n<vecsize; ++n) {
             S r, g;
             r.s = res.s[n];
             g.s = sign2*good.s[n];
             equal = equal &&
               (ISNAN(val.s[n]) || ISNAN(val2.s[n]) || r.sj == g.sj);
           }
           if (!equal) {
             for (int n=0; n<vecsize; ++n) {
               printf("FAIL: copysign type=%s val=%.17g sign=%.17g res=%.17g good=%.17g\n",
                      typename, val.s[n], sign2*val2.s[n], res.s[n], good.s[n]);
             }
             return;
           }
         }
         /* ilogb */
         Ivec ires;
         ires.v = ilogb(val.v);
         Ivec igoodres;
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           if (ISNAN(val.s[n])) {
             igoodres.s[n] = FP_ILOGBNAN;
           } else if (val.s[n] == (stype)0) {
             igoodres.s[n] = FP_ILOGB0;
           } else if (isinf(val.s[n])) {
             igoodres.s[n] = INT_MAX;
           } else {
             // We round down to "correct" for inaccuracies in log2
             // We divide by 2 since log2 is wrong for large inputs
             igoodres.s[n] = 1+rint(floor(0.999999f*log2(0.5*fabs(val.s[n]))));
           }
           equal = equal && ires.s[n] == igoodres.s[n];
         }
         if (!equal) {
           for (int n=0; n<vecsize; ++n) {
             printf("FAIL: ilogb type=%s val=%.17g res=%d good=%d\n",
                    typename, val.s[n], (int)ires.s[n], (int)igoodres.s[n]);
           }
           return;
         }
         /* ldexp */
         Ivec ival2;
         ival2.v = (itype)4;
         res.v = ldexp(val.v, ival2.v);
         Tvec goodres;
         equal = true;
         for (int n=0; n<vecsize; ++n) {
           if (ISNAN(val.s[n])) {
             goodres.s[n] = NAN;
           } else if (val.s[n] == (stype)0) {
             goodres.s[n] = val.s[n];
           } else if (isinf(val.s[n])) {
             goodres.s[n] = val.s[n];
           } else {
             goodres.s[n] = val.s[n] * (stype)(16.0);
           }
           equal = equal && ISEQ(res.s[n], goodres.s[n]);
         }
         if (!equal) {
           for (int n=0; n<vecsize; ++n) {
             printf("FAIL: ldexp type=%s val=%.17g val2=%d res=%.17g good=%.17g\n",
                    typename, val.s[n], (int)ival2.s[n], res.s[n], goodres.s[n]);
           }
           return;
         }
       }
     }
   }
 })
 )

kernel void test_fabs()
{
  CALL_FUNC_V(test_fabs)
}
