// -*-C-*-



#define CCTK_ATTRIBUTE_UNUSED    __attribute__((__unused__))
#define CCTK_BUILTIN_EXPECT(a,b) __builtin_expect(a,b)
#define CCTK_UNROLL              _Pragma("unroll")



// doubleV           vector of double
// convert_doubleV   convert to doubleV
// longV             vector of long (same size as double)
// indicesV          longV containing (0,1,2,...)
// vloadV            load unaligned vector
// vstoreV           store unaligned vector

#if VECTOR_SIZE_I == 1
#  define doubleV            double
#  define convert_doubleV    convert_double
#  define longV              long
#  define indicesV           ((longV)(0))
#  define vloadV(i,p)        ((p)[i])
#  define vstoreV(x,i,p)     ((p)[i]=(x))
#elif VECTOR_SIZE_I == 2
#  define doubleV            double2
#  define convert_doubleV    convert_double2
#  define longV              long2
#  define indicesV           ((longV)(0,1))
#  define vloadV             vload2
#  define vstoreV            vstore2
#elif VECTOR_SIZE_I == 4
#  define doubleV            double4
#  define convert_doubleV    convert_double4
#  define longV              long4
#  define indicesV           ((longV)(0,1,2,3))
#  define vloadV             vload4
#  define vstoreV            vstore4
#elif VECTOR_SIZE_I == 8
#  define doubleV            double8
#  define convert_doubleV    convert_double8
#  define longV              long8
#  define indicesV           ((longV)(0,1,2,3,4,5,6,7))
#  define vloadV             vload8
#  define vstoreV            vstore8
#elif VECTOR_SIZE_I == 16
#  define doubleV            double16
#  define convert_doubleV    convert_double16
#  define longV              long16
#  define indicesV           ((longV)(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15))
#  define vloadV             vload16
#  define vstoreV            vstore16
#else
#  error
#endif

#if VECTOR_SIZE_J!=1 || VECTOR_SIZE_K!=1
#  error
#endif



#define CCTK_REAL double
#define CCTK_INT  int
#define CCTK_LONG long

#define CCTK_REAL_VEC_SIZE VECTOR_SIZE_I
#define CCTK_REAL_VEC      doubleV
#define CCTK_INT_VEC       longV
#define convert_real_vec   convert_doubleV
#define vec_index          convert_real_vec(indicesV)



// vec_loada               load aligned vector
// vec_loadu               load unaligned vector
// vec_load                load regular vector
// vec_loadu_maybe3        load unaligned vector
// vec_storea              store aligned vector
// vec_storeu              store unaligned vector
// vec_store_nta           store regular vector
// vec_store_nta_partial   store regular vector partially

// VECTORISE_ALIGNED_ARRAYS assumes that all grid points [0,j,k] are
// aligned, and arrays are padded as necessary

#define vec_loada(p) (* (CCTK_REAL_VEC const global *) & (p))
#define vec_loadu(p) vloadV(0, & (p))

#if VECTORISE_ALIGNED_ARRAYS
#  define vec_load(p) vec_loada(p)
#  define vec_loadu_maybe3(off1,off2,off3, p)                           \
  ((off1) % CCTK_REAL_VEC_SIZE == 0 ? vec_loada(p) : vec_loadu(p))
#else
#  define vec_load(p) vec_loadu(p)
#  define vec_loadu_maybe3(off1,off2,off3, p) vec_loadu(p)
#endif

#define vec_storea(p, x) (* (CCTK_REAL_VEC global *) & (p) = (x))
#define vec_storeu(p, x) vstoreV(x, 0, & (p))

#if VECTORISE_ALIGNED_ARRAYS
#  define vec_store_nta(p, x) vec_storea(p, x)
#else
#  define vec_store_nta(p, x) vec_storeu(p, x)
#endif

#define vec_store_partial_prepare(i, imin, imax)

#if CCTK_REAL_VEC_SIZE == 1

#  define vec_store_nta_partial(p, x)                                   \
  do {                                                                  \
    if (CCTK_BUILTIN_EXPECT(lc_vec_any_I && lc_vec_any_J && lc_vec_any_K, \
                            true))                                      \
    {                                                                   \
      vec_store_nta(p, x);                                              \
    }                                                                   \
  } while(0)

#elif CCTK_REAL_VEC_SIZE == 2

#  define vec_store_nta_partial(p, x)                                   \
  do {                                                                  \
    if (CCTK_BUILTIN_EXPECT(lc_vec_any_I && lc_vec_any_J && lc_vec_any_K, \
                            true))                                      \
    {                                                                   \
      if (CCTK_BUILTIN_EXPECT(lc_vec_all_I, true)) {                    \
        vec_store_nta(p, x);                                            \
      } else {                                                          \
        if (lc_vec_lo_I) {                                              \
          (&(p))[0] = (x).s0;                                           \
        } else {                                                        \
          (&(p))[1] = (x).s1;                                           \
        }                                                               \
      }                                                                 \
    }                                                                   \
  } while (0)

#else

#  define vec_store_nta_partial(p, x)                                   \
  do {                                                                  \
    if (CCTK_BUILTIN_EXPECT(lc_vec_any_I && lc_vec_any_J && lc_vec_any_K, \
                            true))                                      \
    {                                                                   \
      if (CCTK_BUILTIN_EXPECT(lc_vec_all_I, true)) {                    \
        vec_store_nta(p, x);                                            \
      } else {                                                          \
        /* select(a,b,c) = MSB(c) ? b : a */                            \
        vec_store_nta(p, select(vec_load(p), x, lc_vec_mask_I));        \
      }                                                                 \
    }                                                                   \
  } while(0)

#endif



#define kneg(x) (-(x))

#define kadd(x,y) ((x)+(y))
#define ksub(x,y) ((x)-(y))
#define kmul(x,y) ((x)*(y))
#define kdiv(x,y) ((x)/(y))

#define kmadd(x,y,z)  mad(x,y,z)   // faster than fma(x,y,z)
#define kmsub(x,y,z)  mad(x,y,-(z))
#define knmadd(x,y,z) (-mad(x,y,z))
#define knmsub(x,y,z) (-mad(x,y,-(z)))

#define kcopysign(x,y) copysign(x,y)
#define kfabs(x)       fabs(x)
#define kfmax(x,y)     fmax(x,y)
#define kfmin(x,y)     fmin(x,y)
#define kfnabs(x)      (-fabs(x))
#define ksqrt(x)       sqrt(x)

#define kacos(x)    acos(x)
#define kacosh(x)   acosh(x)
#define kasin(x)    asin(x)
#define kasinh(x)   asinh(x)
#define katan(x)    atan(x)
#define katan2(x,y) atan2(x,y)
#define katanh(x)   atanh(x)
#define kcos(x)     cos(x)
#define kcosh(x)    cosh(x)
#define kexp(x)     exp(x)
#define klog(x)     log(x)
#define kpow(x,a)   pow(x,(CCTK_REAL)(a))
#define ksin(x)     sin(x)
#define ksinh(x)    sinh(x)
#define ktan(x)     tan(x)
#define ktanh(x)    tanh(x)

// Choice   [sign(x)>0 ? y : z]
#define kifpos(x,y,z) select(y,z,x)
#define kifneg(x,y,z) select(z,y,x)

// Choice   [x ? y : z]
#define kifthen(x,y,z)                                                  \
  select((CCTK_REAL_VEC)(z), (CCTK_REAL_VEC)(y), (CCTK_INT_VEC)(x))



#if 0 && defined(__APPLE__)

// Apple's pow implementation is much better than their pown
#  undef pown
#  define pown pow

inline CCTK_REAL myfabs(CCTK_REAL x);
inline CCTK_REAL myfabs(CCTK_REAL x)
{
  return x>=0 ? x : -x;
}

inline CCTK_REAL mycos1(CCTK_REAL x);
inline CCTK_REAL mycos1(CCTK_REAL x)
{
  // 0<=x<=pi/2
  CCTK_REAL const c1 = +1.0;
  CCTK_REAL const c2 = -1.0/2.0;
  CCTK_REAL const c3 = +1.0/24.0;
  CCTK_REAL const c4 = -1.0/720.0;
  CCTK_REAL const c5 = +1.0/40320.0;
  CCTK_REAL const c6 = -1.0/3628800.0;
  CCTK_REAL const c7 = +1.0/479001600.0;
  CCTK_REAL const c8 = -1.0/87178291200.0;
  CCTK_REAL const x2 = pown(x,2);
  return (c1 + x2 *
          (c2 + x2 *
           (c3 + x2 *
            (c4 + x2 *
             (c5 + x2 *
              (c6 + x2 *
               (c7 + x2 * c8)))))));
}

inline CCTK_REAL mycos(CCTK_REAL x);
inline CCTK_REAL mycos(CCTK_REAL x)
{
  x = myfabs(x);
  x = fmod(x,2*M_PI);
  if (x>M_PI) x=M_PI-x;
  bool const isneg = x>M_PI/2;
  if (isneg) x=M_PI/2-x;
  CCTK_REAL y = mycos1(x);
  if (isneg) y=-y;
  return y;
}

#  undef cos
#  define cos mycos



#  undef kcos
// Apple's OpenCL compiler segfaults when calling cos on a vector, so
// we serialise this operation explicitly
inline CCTK_REAL_VEC kcos(CCTK_REAL_VEC const x);
#  if CCTK_REAL_VEC_SIZE==1
inline CCTK_REAL_VEC kcos(CCTK_REAL_VEC const x)
{
  return cos(x);
}
#  elif CCTK_REAL_VEC_SIZE==2
inline CCTK_REAL_VEC kcos(CCTK_REAL_VEC const x)
{
  return (CCTK_REAL_VEC)(cos(x.s0), cos(x.s1));
}
#  else
#    error
#  endif

#endif



////////////////////////////////////////////////////////////////////////////////



#define dim 3



typedef struct {
  // Doubles first, then ints, to ensure proper alignment
  // Coordinates:
  double cctk_origin_space[dim];
  double cctk_delta_space[dim];
  double cctk_time;
  double cctk_delta_time;
  // Grid structure properties:
  int cctk_gsh[dim];
  int cctk_lbnd[dim];
  int cctk_lsh[dim];
  int cctk_ash[dim];
  // Loop settings:
  int lmin[dim];                 // loop region
  int lmax[dim];
  int imin[dim];                 // active region
  int imax[dim];
} cGH;



// Cactus compatibility definitions

#define DECLARE_CCTK_ARGUMENTS                                          \
  ptrdiff_t const cctk_lbnd[] =                                         \
    {cctkGH->cctk_lbnd[0], cctkGH->cctk_lbnd[1], cctkGH->cctk_lbnd[2]}; \
  ptrdiff_t const cctk_lsh[] =                                          \
    {cctkGH->cctk_lsh[0], cctkGH->cctk_lsh[1], cctkGH->cctk_lsh[2]};    \
  ptrdiff_t const imin[] =                                              \
    {cctkGH->imin[0], cctkGH->imin[1], cctkGH->imin[2]};                \
  ptrdiff_t const imax[] =                                              \
    {cctkGH->imax[0], cctkGH->imax[1], cctkGH->imax[2]};                \
  CCTK_REAL const cctk_time = cctkGH->cctk_time;                        \
  CCTK_REAL const cctk_delta_time = cctkGH->cctk_delta_time;            \
  CCTK_REAL constant const *restrict const cctk_origin_space =          \
    cctkGH->cctk_origin_space;                                          \
  CCTK_REAL constant const *restrict const cctk_delta_space =           \
    cctkGH->cctk_delta_space;                                           \
  bool const stress_energy_state1 = 0;

#define CCTK_GFINDEX3D(cctkGH,i,j,k)                    \
  ((i) + cctk_lsh[0] * ((j) + cctk_lsh[1] * (k)))
 
#define CCTK_ORIGIN_SPACE(d) (cctkGH->cctk_origin_space[d])
#define CCTK_DELTA_SPACE(d)  (cctkGH->cctk_delta_space[d])
#define CCTK_DELTA_TIME      (cctkGH->cctk_delta_time)



// Kranc compatibility definitions

#define Pi            M_PI
#define IfThen(c,x,y) ((c)?(x):(y))
#define ToReal(x)     ((CCTK_REAL_VEC)(CCTK_REAL)(x))

CCTK_REAL ScalarINV(CCTK_REAL const x);
CCTK_REAL ScalarINV(CCTK_REAL const x)
{
  return ((CCTK_REAL)1)/x;
}
CCTK_REAL_VEC INV(CCTK_REAL_VEC const x);
CCTK_REAL_VEC INV(CCTK_REAL_VEC const x)
{
  return ToReal(1)/x;
}
/* CCTK_REAL_VEC Sign(CCTK_REAL_VEC const x); */
/* CCTK_REAL_VEC Sign(CCTK_REAL_VEC const x) */
/* { */
/*   return x==ToReal(0) ? ToReal(0) : copysign(ToReal(1), x); */
/* } */
// CCTK_REAL_VEC SQR(CCTK_REAL_VEC const x)
// {
//   return pown(x,2);
// }
CCTK_REAL_VEC SQR(CCTK_REAL_VEC const x);
CCTK_REAL_VEC SQR(CCTK_REAL_VEC const x)
{
  return x*x;
}
CCTK_REAL_VEC ksgn(CCTK_REAL_VEC x);
CCTK_REAL_VEC ksgn(CCTK_REAL_VEC x)
{
  return kifthen(x==ToReal(0.0), ToReal(0.0), kcopysign(ToReal(1.0), x));
}
CCTK_INT_VEC kisgn(CCTK_REAL_VEC x);
CCTK_INT_VEC kisgn(CCTK_REAL_VEC x)
{
  return select(select((CCTK_INT_VEC)+1,
                       (CCTK_INT_VEC)-1, (CCTK_INT_VEC)(x<ToReal(0.0))),
                (CCTK_INT_VEC)0, (CCTK_INT_VEC)(x==ToReal(0.0)));
}

#define KRANC_GFOFFSET3D(u,i,j,k)                       \
  vec_loadu_maybe3(i,j,k,(u)[di*(i)+dj*(j)+dk*(k)])

#define eTtt ((CCTK_REAL global const *)0)
#define eTtx ((CCTK_REAL global const *)0)
#define eTty ((CCTK_REAL global const *)0)
#define eTtz ((CCTK_REAL global const *)0)
#define eTxx ((CCTK_REAL global const *)0)
#define eTxy ((CCTK_REAL global const *)0)
#define eTxz ((CCTK_REAL global const *)0)
#define eTyy ((CCTK_REAL global const *)0)
#define eTyz ((CCTK_REAL global const *)0)
#define eTzz ((CCTK_REAL global const *)0)
#define jacobian_derivative_group ""
#define jacobian_group            ""
#define jacobian_identity_map     0
#define stress_energy_state       (&stress_energy_state1)
#define CCTK_IsFunctionAliased(x)            0
#define CCTK_WARN(lev,msg)                   ((void)0)
#define GenericFD_GroupDataPointers(a,b,c,d) ((void)0)
#define MultiPatch_GetMap(x)                 0
#define strlen(x)                            0



////////////////////////////////////////////////////////////////////////////////



#define LC_SET_GROUP_VARS(D)                                            \
  ptrdiff_t const ind##D CCTK_ATTRIBUTE_UNUSED =                        \
    (lc_off##D + VECTOR_SIZE_##D * UNROLL_SIZE_##D *                    \
     (lc_grp##D + GROUP_SIZE_##D *                                      \
      (lc_til##D + TILE_SIZE_##D * lc_grd##D)));                        \
  bool const lc_grp_done_##D CCTK_ATTRIBUTE_UNUSED =                    \
    ind##D >= lc_##D##max;

#define vecVI indicesV
#define vecVJ ((CCTK_INT_VEC)0)
#define vecVK ((CCTK_INT_VEC)0)

#define LC_SET_VECTOR_VARS(IND,D)                                       \
  ptrdiff_t const IND CCTK_ATTRIBUTE_UNUSED =                           \
    (lc_off##D + VECTOR_SIZE_##D *                                      \
     (lc_unr##D + UNROLL_SIZE_##D *                                     \
      (lc_grp##D + GROUP_SIZE_##D *                                     \
       (lc_til##D + TILE_SIZE_##D * lc_grd##D))));                      \
  bool const lc_vec_trivial_##D CCTK_ATTRIBUTE_UNUSED =                 \
    VECTOR_SIZE_##D * UNROLL_SIZE_##D == 1;                             \
  bool const lc_vec_any_##D CCTK_ATTRIBUTE_UNUSED =                     \
    /*TODO because unroll size is 1*/                                   \
    1 /*TODO lc_vec_trivial_##D ||                                      \
        (IND+VECTOR_SIZE_##D-1 >= lc_##D##min && IND < lc_##D##max)*/;  \
  bool const lc_vec_lo_##D CCTK_ATTRIBUTE_UNUSED =                      \
    lc_vec_trivial_##D ||                                               \
    IND >= lc_##D##min;                                                 \
  bool const lc_vec_hi_##D CCTK_ATTRIBUTE_UNUSED =                      \
    lc_vec_trivial_##D ||                                               \
    IND+VECTOR_SIZE_##D-1 < lc_##D##max;                                \
  bool const lc_vec_all_##D CCTK_ATTRIBUTE_UNUSED =                     \
    lc_vec_trivial_##D ||                                               \
    (lc_vec_lo_##D && lc_vec_hi_##D);                                   \
  CCTK_INT_VEC const lc_vec_mask_##D CCTK_ATTRIBUTE_UNUSED =            \
    lc_vec_trivial_##D ?                                                \
    (CCTK_INT_VEC)true :                                                \
    ((CCTK_INT_VEC)IND+vecV##D >= (CCTK_INT_VEC)lc_##D##min) &          \
    ((CCTK_INT_VEC)IND+vecV##D <  (CCTK_INT_VEC)lc_##D##max);

#define LC_LOOP3VEC(name,                                               \
                    i,j,k,                                              \
                    imin,jmin,kmin,                                     \
                    imax,jmax,kmax,                                     \
                    ilsh,jlsh,klsh,                                     \
                    vecsize)                                            \
  do {                                                                  \
    typedef int lc_loop3_##name;                                        \
                                                                        \
    ptrdiff_t const lc_Imin = (imin);                                   \
    ptrdiff_t const lc_Jmin = (jmin);                                   \
    ptrdiff_t const lc_Kmin = (kmin);                                   \
    ptrdiff_t const lc_Imax = (imax);                                   \
    ptrdiff_t const lc_Jmax = (jmax);                                   \
    ptrdiff_t const lc_Kmax = (kmax);                                   \
    ptrdiff_t const lc_offI = cctkGH->lmin[0]; /* offset */             \
    ptrdiff_t const lc_offJ = cctkGH->lmin[1];                          \
    ptrdiff_t const lc_offK = cctkGH->lmin[2];                          \
    ptrdiff_t const lc_grpI = get_local_id(0); /* index in group */     \
    ptrdiff_t const lc_grpJ = get_local_id(1);                          \
    ptrdiff_t const lc_grpK = get_local_id(2);                          \
    ptrdiff_t const lc_grdI = get_group_id(0); /* index in grid */      \
    ptrdiff_t const lc_grdJ = get_group_id(1);                          \
    ptrdiff_t const lc_grdK = get_group_id(2);                          \
                                                                        \
    ptrdiff_t const lc_imin = lc_Imin;                                  \
    ptrdiff_t const lc_imax = lc_Imax;                                  \
                                                                        \
    for (ptrdiff_t lc_tilK = 0; lc_tilK < TILE_SIZE_K; ++lc_tilK) {     \
    LC_SET_GROUP_VARS(K);                                               \
    if (CCTK_BUILTIN_EXPECT(lc_grp_done_K, 0)) break;                   \
    for (ptrdiff_t lc_tilJ = 0; lc_tilJ < TILE_SIZE_J; ++lc_tilJ) {     \
    LC_SET_GROUP_VARS(J);                                               \
    if (CCTK_BUILTIN_EXPECT(lc_grp_done_J, 0)) break;                   \
    for (ptrdiff_t lc_tilI = 0; lc_tilI < TILE_SIZE_I; ++lc_tilI) {     \
    LC_SET_GROUP_VARS(I);                                               \
    if (CCTK_BUILTIN_EXPECT(lc_grp_done_I, 0)) break;                   \
                                                                        \
      ptrdiff_t const lc_unrK = 0;                                      \
      /*TODO CCTK_UNROLL                                                \
        for (ptrdiff_t lc_unrK = 0; lc_unrK < UNROLL_SIZE_K; ++lc_unrK)*/ { \
      LC_SET_VECTOR_VARS(k,K);                                          \
      ptrdiff_t const lc_unrJ = 0;                                      \
      /*TODO CCTK_UNROLL                                                \
        for (ptrdiff_t lc_unrJ = 0; lc_unrJ < UNROLL_SIZE_J; ++lc_unrJ)*/ { \
      LC_SET_VECTOR_VARS(j,J);                                          \
      ptrdiff_t const lc_unrI = 0;                                      \
      /*TODO CCTK_UNROLL                                                \
        for (ptrdiff_t lc_unrI = 0; lc_unrI < UNROLL_SIZE_I; ++lc_unrI)*/ { \
      LC_SET_VECTOR_VARS(i,I);                                          \
                                                                        \
        {
#define LC_ENDLOOP3VEC(name)                            \
        }                                               \
      }                                                 \
      }                                                 \
      }                                                 \
    }                                                   \
    }                                                   \
    }                                                   \
    typedef lc_loop3_##name lc_ensure_proper_nesting;   \
  } while(0)

#define LC_LOOP3(name,                          \
                 i,j,k,                         \
                 imin,jmin,kmin,                \
                 imax,jmax,kmax,                \
                 ilsh,jlsh,klsh)                \
  LC_LOOP3VEC(name,                             \
              i,j,k,                            \
              imin,jmin,kmin,                   \
              imax,jmax,kmax,                   \
              ilsh,jlsh,klsh,                   \
              1)
#define LC_ENDLOOP3(name)                       \
  LC_ENDLOOP3VEC(name)

// Cactus parameters:
typedef struct {
  CCTK_REAL A_bound_limit;
  CCTK_REAL A_bound_scalar;
  CCTK_REAL A_bound_speed;
  CCTK_REAL alpha_bound_limit;
  CCTK_REAL alpha_bound_scalar;
  CCTK_REAL alpha_bound_speed;
  CCTK_REAL AlphaDriver;
  CCTK_REAL At11_bound_limit;
  CCTK_REAL At11_bound_scalar;
  CCTK_REAL At11_bound_speed;
  CCTK_REAL At12_bound_limit;
  CCTK_REAL At12_bound_scalar;
  CCTK_REAL At12_bound_speed;
  CCTK_REAL At13_bound_limit;
  CCTK_REAL At13_bound_scalar;
  CCTK_REAL At13_bound_speed;
  CCTK_REAL At22_bound_limit;
  CCTK_REAL At22_bound_scalar;
  CCTK_REAL At22_bound_speed;
  CCTK_REAL At23_bound_limit;
  CCTK_REAL At23_bound_scalar;
  CCTK_REAL At23_bound_speed;
  CCTK_REAL At33_bound_limit;
  CCTK_REAL At33_bound_scalar;
  CCTK_REAL At33_bound_speed;
  CCTK_REAL B1_bound_limit;
  CCTK_REAL B1_bound_scalar;
  CCTK_REAL B1_bound_speed;
  CCTK_REAL B2_bound_limit;
  CCTK_REAL B2_bound_scalar;
  CCTK_REAL B2_bound_speed;
  CCTK_REAL B3_bound_limit;
  CCTK_REAL B3_bound_scalar;
  CCTK_REAL B3_bound_speed;
  CCTK_REAL beta1_bound_limit;
  CCTK_REAL beta1_bound_scalar;
  CCTK_REAL beta1_bound_speed;
  CCTK_REAL beta2_bound_limit;
  CCTK_REAL beta2_bound_scalar;
  CCTK_REAL beta2_bound_speed;
  CCTK_REAL beta3_bound_limit;
  CCTK_REAL beta3_bound_scalar;
  CCTK_REAL beta3_bound_speed;
  CCTK_REAL BetaDriver;
  CCTK_REAL EpsDiss;
  CCTK_REAL gt11_bound_limit;
  CCTK_REAL gt11_bound_scalar;
  CCTK_REAL gt11_bound_speed;
  CCTK_REAL gt12_bound_limit;
  CCTK_REAL gt12_bound_scalar;
  CCTK_REAL gt12_bound_speed;
  CCTK_REAL gt13_bound_limit;
  CCTK_REAL gt13_bound_scalar;
  CCTK_REAL gt13_bound_speed;
  CCTK_REAL gt22_bound_limit;
  CCTK_REAL gt22_bound_scalar;
  CCTK_REAL gt22_bound_speed;
  CCTK_REAL gt23_bound_limit;
  CCTK_REAL gt23_bound_scalar;
  CCTK_REAL gt23_bound_speed;
  CCTK_REAL gt33_bound_limit;
  CCTK_REAL gt33_bound_scalar;
  CCTK_REAL gt33_bound_speed;
  CCTK_REAL harmonicF;
  CCTK_REAL LapseACoeff;
  CCTK_REAL LapseAdvectionCoeff;
  CCTK_REAL MinimumLapse;
  CCTK_REAL ML_curv_bound_limit;
  CCTK_REAL ML_curv_bound_scalar;
  CCTK_REAL ML_curv_bound_speed;
  CCTK_REAL ML_dtlapse_bound_limit;
  CCTK_REAL ML_dtlapse_bound_scalar;
  CCTK_REAL ML_dtlapse_bound_speed;
  CCTK_REAL ML_dtshift_bound_limit;
  CCTK_REAL ML_dtshift_bound_scalar;
  CCTK_REAL ML_dtshift_bound_speed;
  CCTK_REAL ML_Gamma_bound_limit;
  CCTK_REAL ML_Gamma_bound_scalar;
  CCTK_REAL ML_Gamma_bound_speed;
  CCTK_REAL ML_lapse_bound_limit;
  CCTK_REAL ML_lapse_bound_scalar;
  CCTK_REAL ML_lapse_bound_speed;
  CCTK_REAL ML_log_confac_bound_limit;
  CCTK_REAL ML_log_confac_bound_scalar;
  CCTK_REAL ML_log_confac_bound_speed;
  CCTK_REAL ML_metric_bound_limit;
  CCTK_REAL ML_metric_bound_scalar;
  CCTK_REAL ML_metric_bound_speed;
  CCTK_REAL ML_shift_bound_limit;
  CCTK_REAL ML_shift_bound_scalar;
  CCTK_REAL ML_shift_bound_speed;
  CCTK_REAL ML_trace_curv_bound_limit;
  CCTK_REAL ML_trace_curv_bound_scalar;
  CCTK_REAL ML_trace_curv_bound_speed;
  CCTK_REAL phi_bound_limit;
  CCTK_REAL phi_bound_scalar;
  CCTK_REAL phi_bound_speed;
  CCTK_REAL ShiftAdvectionCoeff;
  CCTK_REAL ShiftBCoeff;
  CCTK_REAL ShiftGammaCoeff;
  CCTK_REAL SpatialBetaDriverRadius;
  CCTK_REAL SpatialShiftGammaCoeffRadius;
  CCTK_REAL trK_bound_limit;
  CCTK_REAL trK_bound_scalar;
  CCTK_REAL trK_bound_speed;
  CCTK_REAL Xt1_bound_limit;
  CCTK_REAL Xt1_bound_scalar;
  CCTK_REAL Xt1_bound_speed;
  CCTK_REAL Xt2_bound_limit;
  CCTK_REAL Xt2_bound_scalar;
  CCTK_REAL Xt2_bound_speed;
  CCTK_REAL Xt3_bound_limit;
  CCTK_REAL Xt3_bound_scalar;
  CCTK_REAL Xt3_bound_speed;
  CCTK_INT conformalMethod;
  CCTK_INT fdOrder;
  CCTK_INT harmonicN;
  CCTK_INT harmonicShift;
  CCTK_INT ML_BSSN_CL_Advect_calc_every;
  CCTK_INT ML_BSSN_CL_Advect_calc_offset;
  CCTK_INT ML_BSSN_CL_boundary_calc_every;
  CCTK_INT ML_BSSN_CL_boundary_calc_offset;
  CCTK_INT ML_BSSN_CL_constraints1_calc_every;
  CCTK_INT ML_BSSN_CL_constraints1_calc_offset;
  CCTK_INT ML_BSSN_CL_constraints2_calc_every;
  CCTK_INT ML_BSSN_CL_constraints2_calc_offset;
  CCTK_INT ML_BSSN_CL_convertFromADMBase_calc_every;
  CCTK_INT ML_BSSN_CL_convertFromADMBase_calc_offset;
  CCTK_INT ML_BSSN_CL_convertFromADMBaseGamma_calc_every;
  CCTK_INT ML_BSSN_CL_convertFromADMBaseGamma_calc_offset;
  CCTK_INT ML_BSSN_CL_convertToADMBase_calc_every;
  CCTK_INT ML_BSSN_CL_convertToADMBase_calc_offset;
  CCTK_INT ML_BSSN_CL_convertToADMBaseDtLapseShift_calc_every;
  CCTK_INT ML_BSSN_CL_convertToADMBaseDtLapseShift_calc_offset;
  CCTK_INT ML_BSSN_CL_convertToADMBaseDtLapseShiftBoundary_calc_every;
  CCTK_INT ML_BSSN_CL_convertToADMBaseDtLapseShiftBoundary_calc_offset;
  CCTK_INT ML_BSSN_CL_convertToADMBaseFakeDtLapseShift_calc_every;
  CCTK_INT ML_BSSN_CL_convertToADMBaseFakeDtLapseShift_calc_offset;
  CCTK_INT ML_BSSN_CL_Dissipation_calc_every;
  CCTK_INT ML_BSSN_CL_Dissipation_calc_offset;
  CCTK_INT ML_BSSN_CL_enforce_calc_every;
  CCTK_INT ML_BSSN_CL_enforce_calc_offset;
  CCTK_INT ML_BSSN_CL_InitGamma_calc_every;
  CCTK_INT ML_BSSN_CL_InitGamma_calc_offset;
  CCTK_INT ML_BSSN_CL_InitRHS_calc_every;
  CCTK_INT ML_BSSN_CL_InitRHS_calc_offset;
  CCTK_INT ML_BSSN_CL_MaxNumArrayEvolvedVars;
  CCTK_INT ML_BSSN_CL_MaxNumEvolvedVars;
  CCTK_INT ML_BSSN_CL_Minkowski_calc_every;
  CCTK_INT ML_BSSN_CL_Minkowski_calc_offset;
  CCTK_INT ML_BSSN_CL_RHS1_calc_every;
  CCTK_INT ML_BSSN_CL_RHS1_calc_offset;
  CCTK_INT ML_BSSN_CL_RHS2_calc_every;
  CCTK_INT ML_BSSN_CL_RHS2_calc_offset;
  CCTK_INT ML_BSSN_CL_RHSStaticBoundary_calc_every;
  CCTK_INT ML_BSSN_CL_RHSStaticBoundary_calc_offset;
  CCTK_INT other_timelevels;
  CCTK_INT rhs_timelevels;
  CCTK_INT ShiftAlphaPower;
  CCTK_INT timelevels;
  CCTK_INT verbose;
} cctk_parameters_t;
#define DECLARE_CCTK_PARAMETERS \
  CCTK_REAL const A_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const A_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const A_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const alpha_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const alpha_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const alpha_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const AlphaDriver CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const At11_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const At11_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const At11_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const At12_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const At12_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const At12_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const At13_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const At13_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const At13_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const At22_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const At22_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const At22_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const At23_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const At23_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const At23_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const At33_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const At33_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const At33_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const B1_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const B1_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const B1_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const B2_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const B2_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const B2_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const B3_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const B3_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const B3_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const beta1_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const beta1_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const beta1_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const beta2_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const beta2_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const beta2_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const beta3_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const beta3_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const beta3_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const BetaDriver CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const EpsDiss CCTK_ATTRIBUTE_UNUSED = 0.20000000000000001; \
  CCTK_REAL const gt11_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const gt11_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const gt11_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const gt12_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const gt12_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const gt12_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const gt13_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const gt13_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const gt13_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const gt22_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const gt22_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const gt22_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const gt23_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const gt23_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const gt23_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const gt33_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const gt33_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const gt33_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const harmonicF CCTK_ATTRIBUTE_UNUSED = 2; \
  CCTK_REAL const LapseACoeff CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const LapseAdvectionCoeff CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const MinimumLapse CCTK_ATTRIBUTE_UNUSED = 1e-08; \
  CCTK_REAL const ML_curv_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_curv_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_curv_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const ML_dtlapse_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_dtlapse_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_dtlapse_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const ML_dtshift_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_dtshift_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_dtshift_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const ML_Gamma_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_Gamma_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_Gamma_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const ML_lapse_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_lapse_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_lapse_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const ML_log_confac_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_log_confac_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_log_confac_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const ML_metric_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_metric_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_metric_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const ML_shift_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_shift_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_shift_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const ML_trace_curv_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_trace_curv_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const ML_trace_curv_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const phi_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const phi_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const phi_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const ShiftAdvectionCoeff CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const ShiftBCoeff CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const ShiftGammaCoeff CCTK_ATTRIBUTE_UNUSED = 0.75; \
  CCTK_REAL const SpatialBetaDriverRadius CCTK_ATTRIBUTE_UNUSED = 1000000000000; \
  CCTK_REAL const SpatialShiftGammaCoeffRadius CCTK_ATTRIBUTE_UNUSED = 1000000000000; \
  CCTK_REAL const trK_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const trK_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const trK_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const Xt1_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const Xt1_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const Xt1_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const Xt2_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const Xt2_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const Xt2_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_REAL const Xt3_bound_limit CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const Xt3_bound_scalar CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_REAL const Xt3_bound_speed CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const conformalMethod CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const fdOrder CCTK_ATTRIBUTE_UNUSED = 4; \
  CCTK_INT const harmonicN CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const harmonicShift CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_Advect_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_Advect_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_boundary_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_boundary_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_constraints1_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_constraints1_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_constraints2_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_constraints2_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_convertFromADMBase_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_convertFromADMBase_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_convertFromADMBaseGamma_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_convertFromADMBaseGamma_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_convertToADMBase_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_convertToADMBase_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_convertToADMBaseDtLapseShift_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_convertToADMBaseDtLapseShift_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_convertToADMBaseDtLapseShiftBoundary_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_convertToADMBaseDtLapseShiftBoundary_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_convertToADMBaseFakeDtLapseShift_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_convertToADMBaseFakeDtLapseShift_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_Dissipation_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_Dissipation_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_enforce_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_enforce_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_InitGamma_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_InitGamma_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_InitRHS_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_InitRHS_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_MaxNumArrayEvolvedVars CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_MaxNumEvolvedVars CCTK_ATTRIBUTE_UNUSED = 25; \
  CCTK_INT const ML_BSSN_CL_Minkowski_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_Minkowski_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_RHS1_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_RHS1_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_RHS2_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_RHS2_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const ML_BSSN_CL_RHSStaticBoundary_calc_every CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ML_BSSN_CL_RHSStaticBoundary_calc_offset CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const other_timelevels CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const rhs_timelevels CCTK_ATTRIBUTE_UNUSED = 1; \
  CCTK_INT const ShiftAlphaPower CCTK_ATTRIBUTE_UNUSED = 0; \
  CCTK_INT const timelevels CCTK_ATTRIBUTE_UNUSED = 3; \
  CCTK_INT const verbose CCTK_ATTRIBUTE_UNUSED = 0;

// Kranc's FD operators:
#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder21(u) (kmul(p1o2dx,ksub(KRANC_GFOFFSET3D(u,1,0,0),KRANC_GFOFFSET3D(u,-1,0,0))))
#else
#  define PDstandardNthfdOrder21(u) (PDstandardNthfdOrder21_impl(u,p1o2dx,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder21_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o2dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder21_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o2dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o2dx,ksub(KRANC_GFOFFSET3D(u,1,0,0),KRANC_GFOFFSET3D(u,-1,0,0)));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder22(u) (kmul(p1o2dy,ksub(KRANC_GFOFFSET3D(u,0,1,0),KRANC_GFOFFSET3D(u,0,-1,0))))
#else
#  define PDstandardNthfdOrder22(u) (PDstandardNthfdOrder22_impl(u,p1o2dy,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder22_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o2dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder22_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o2dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o2dy,ksub(KRANC_GFOFFSET3D(u,0,1,0),KRANC_GFOFFSET3D(u,0,-1,0)));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder23(u) (kmul(p1o2dz,ksub(KRANC_GFOFFSET3D(u,0,0,1),KRANC_GFOFFSET3D(u,0,0,-1))))
#else
#  define PDstandardNthfdOrder23(u) (PDstandardNthfdOrder23_impl(u,p1o2dz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder23_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o2dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder23_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o2dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o2dz,ksub(KRANC_GFOFFSET3D(u,0,0,1),KRANC_GFOFFSET3D(u,0,0,-1)));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder41(u) (kmul(p1o12dx,kadd(KRANC_GFOFFSET3D(u,-2,0,0),kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-8),kmsub(KRANC_GFOFFSET3D(u,1,0,0),ToReal(8),KRANC_GFOFFSET3D(u,2,0,0))))))
#else
#  define PDstandardNthfdOrder41(u) (PDstandardNthfdOrder41_impl(u,p1o12dx,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder41_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o12dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder41_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o12dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o12dx,kadd(KRANC_GFOFFSET3D(u,-2,0,0),kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-8),kmsub(KRANC_GFOFFSET3D(u,1,0,0),ToReal(8),KRANC_GFOFFSET3D(u,2,0,0)))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder42(u) (kmul(p1o12dy,kadd(KRANC_GFOFFSET3D(u,0,-2,0),kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-8),kmsub(KRANC_GFOFFSET3D(u,0,1,0),ToReal(8),KRANC_GFOFFSET3D(u,0,2,0))))))
#else
#  define PDstandardNthfdOrder42(u) (PDstandardNthfdOrder42_impl(u,p1o12dy,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder42_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o12dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder42_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o12dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o12dy,kadd(KRANC_GFOFFSET3D(u,0,-2,0),kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-8),kmsub(KRANC_GFOFFSET3D(u,0,1,0),ToReal(8),KRANC_GFOFFSET3D(u,0,2,0)))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder43(u) (kmul(p1o12dz,kadd(KRANC_GFOFFSET3D(u,0,0,-2),kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-8),kmsub(KRANC_GFOFFSET3D(u,0,0,1),ToReal(8),KRANC_GFOFFSET3D(u,0,0,2))))))
#else
#  define PDstandardNthfdOrder43(u) (PDstandardNthfdOrder43_impl(u,p1o12dz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder43_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o12dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder43_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o12dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o12dz,kadd(KRANC_GFOFFSET3D(u,0,0,-2),kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-8),kmsub(KRANC_GFOFFSET3D(u,0,0,1),ToReal(8),KRANC_GFOFFSET3D(u,0,0,2)))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder61(u) (kmul(p1o60dx,kadd(KRANC_GFOFFSET3D(u,3,0,0),kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-45),kmadd(KRANC_GFOFFSET3D(u,2,0,0),ToReal(-9),ksub(kmadd(KRANC_GFOFFSET3D(u,-2,0,0),ToReal(9),kmul(KRANC_GFOFFSET3D(u,1,0,0),ToReal(45))),KRANC_GFOFFSET3D(u,-3,0,0)))))))
#else
#  define PDstandardNthfdOrder61(u) (PDstandardNthfdOrder61_impl(u,p1o60dx,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder61_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o60dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder61_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o60dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o60dx,kadd(KRANC_GFOFFSET3D(u,3,0,0),kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-45),kmadd(KRANC_GFOFFSET3D(u,2,0,0),ToReal(-9),ksub(kmadd(KRANC_GFOFFSET3D(u,-2,0,0),ToReal(9),kmul(KRANC_GFOFFSET3D(u,1,0,0),ToReal(45))),KRANC_GFOFFSET3D(u,-3,0,0))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder62(u) (kmul(p1o60dy,kadd(KRANC_GFOFFSET3D(u,0,3,0),kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-45),kmadd(KRANC_GFOFFSET3D(u,0,2,0),ToReal(-9),ksub(kmadd(KRANC_GFOFFSET3D(u,0,-2,0),ToReal(9),kmul(KRANC_GFOFFSET3D(u,0,1,0),ToReal(45))),KRANC_GFOFFSET3D(u,0,-3,0)))))))
#else
#  define PDstandardNthfdOrder62(u) (PDstandardNthfdOrder62_impl(u,p1o60dy,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder62_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o60dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder62_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o60dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o60dy,kadd(KRANC_GFOFFSET3D(u,0,3,0),kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-45),kmadd(KRANC_GFOFFSET3D(u,0,2,0),ToReal(-9),ksub(kmadd(KRANC_GFOFFSET3D(u,0,-2,0),ToReal(9),kmul(KRANC_GFOFFSET3D(u,0,1,0),ToReal(45))),KRANC_GFOFFSET3D(u,0,-3,0))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder63(u) (kmul(p1o60dz,kadd(KRANC_GFOFFSET3D(u,0,0,3),kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-45),kmadd(KRANC_GFOFFSET3D(u,0,0,2),ToReal(-9),ksub(kmadd(KRANC_GFOFFSET3D(u,0,0,-2),ToReal(9),kmul(KRANC_GFOFFSET3D(u,0,0,1),ToReal(45))),KRANC_GFOFFSET3D(u,0,0,-3)))))))
#else
#  define PDstandardNthfdOrder63(u) (PDstandardNthfdOrder63_impl(u,p1o60dz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder63_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o60dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder63_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o60dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o60dz,kadd(KRANC_GFOFFSET3D(u,0,0,3),kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-45),kmadd(KRANC_GFOFFSET3D(u,0,0,2),ToReal(-9),ksub(kmadd(KRANC_GFOFFSET3D(u,0,0,-2),ToReal(9),kmul(KRANC_GFOFFSET3D(u,0,0,1),ToReal(45))),KRANC_GFOFFSET3D(u,0,0,-3))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder81(u) (kmul(p1o840dx,kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-672),kmadd(KRANC_GFOFFSET3D(u,2,0,0),ToReal(-168),kmadd(KRANC_GFOFFSET3D(u,-3,0,0),ToReal(-32),kmadd(KRANC_GFOFFSET3D(u,4,0,0),ToReal(-3),kmadd(KRANC_GFOFFSET3D(u,-4,0,0),ToReal(3),kmadd(KRANC_GFOFFSET3D(u,3,0,0),ToReal(32),kmadd(KRANC_GFOFFSET3D(u,-2,0,0),ToReal(168),kmul(KRANC_GFOFFSET3D(u,1,0,0),ToReal(672)))))))))))
#else
#  define PDstandardNthfdOrder81(u) (PDstandardNthfdOrder81_impl(u,p1o840dx,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder81_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o840dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder81_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o840dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o840dx,kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-672),kmadd(KRANC_GFOFFSET3D(u,2,0,0),ToReal(-168),kmadd(KRANC_GFOFFSET3D(u,-3,0,0),ToReal(-32),kmadd(KRANC_GFOFFSET3D(u,4,0,0),ToReal(-3),kmadd(KRANC_GFOFFSET3D(u,-4,0,0),ToReal(3),kmadd(KRANC_GFOFFSET3D(u,3,0,0),ToReal(32),kmadd(KRANC_GFOFFSET3D(u,-2,0,0),ToReal(168),kmul(KRANC_GFOFFSET3D(u,1,0,0),ToReal(672))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder82(u) (kmul(p1o840dy,kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-672),kmadd(KRANC_GFOFFSET3D(u,0,2,0),ToReal(-168),kmadd(KRANC_GFOFFSET3D(u,0,-3,0),ToReal(-32),kmadd(KRANC_GFOFFSET3D(u,0,4,0),ToReal(-3),kmadd(KRANC_GFOFFSET3D(u,0,-4,0),ToReal(3),kmadd(KRANC_GFOFFSET3D(u,0,3,0),ToReal(32),kmadd(KRANC_GFOFFSET3D(u,0,-2,0),ToReal(168),kmul(KRANC_GFOFFSET3D(u,0,1,0),ToReal(672)))))))))))
#else
#  define PDstandardNthfdOrder82(u) (PDstandardNthfdOrder82_impl(u,p1o840dy,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder82_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o840dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder82_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o840dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o840dy,kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-672),kmadd(KRANC_GFOFFSET3D(u,0,2,0),ToReal(-168),kmadd(KRANC_GFOFFSET3D(u,0,-3,0),ToReal(-32),kmadd(KRANC_GFOFFSET3D(u,0,4,0),ToReal(-3),kmadd(KRANC_GFOFFSET3D(u,0,-4,0),ToReal(3),kmadd(KRANC_GFOFFSET3D(u,0,3,0),ToReal(32),kmadd(KRANC_GFOFFSET3D(u,0,-2,0),ToReal(168),kmul(KRANC_GFOFFSET3D(u,0,1,0),ToReal(672))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder83(u) (kmul(p1o840dz,kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-672),kmadd(KRANC_GFOFFSET3D(u,0,0,2),ToReal(-168),kmadd(KRANC_GFOFFSET3D(u,0,0,-3),ToReal(-32),kmadd(KRANC_GFOFFSET3D(u,0,0,4),ToReal(-3),kmadd(KRANC_GFOFFSET3D(u,0,0,-4),ToReal(3),kmadd(KRANC_GFOFFSET3D(u,0,0,3),ToReal(32),kmadd(KRANC_GFOFFSET3D(u,0,0,-2),ToReal(168),kmul(KRANC_GFOFFSET3D(u,0,0,1),ToReal(672)))))))))))
#else
#  define PDstandardNthfdOrder83(u) (PDstandardNthfdOrder83_impl(u,p1o840dz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder83_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o840dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder83_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o840dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o840dz,kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-672),kmadd(KRANC_GFOFFSET3D(u,0,0,2),ToReal(-168),kmadd(KRANC_GFOFFSET3D(u,0,0,-3),ToReal(-32),kmadd(KRANC_GFOFFSET3D(u,0,0,4),ToReal(-3),kmadd(KRANC_GFOFFSET3D(u,0,0,-4),ToReal(3),kmadd(KRANC_GFOFFSET3D(u,0,0,3),ToReal(32),kmadd(KRANC_GFOFFSET3D(u,0,0,-2),ToReal(168),kmul(KRANC_GFOFFSET3D(u,0,0,1),ToReal(672))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder211(u) (kmul(p1odx2,kadd(KRANC_GFOFFSET3D(u,-1,0,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-2),KRANC_GFOFFSET3D(u,1,0,0)))))
#else
#  define PDstandardNthfdOrder211(u) (PDstandardNthfdOrder211_impl(u,p1odx2,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder211_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1odx2, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder211_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1odx2, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1odx2,kadd(KRANC_GFOFFSET3D(u,-1,0,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-2),KRANC_GFOFFSET3D(u,1,0,0))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder222(u) (kmul(p1ody2,kadd(KRANC_GFOFFSET3D(u,0,-1,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-2),KRANC_GFOFFSET3D(u,0,1,0)))))
#else
#  define PDstandardNthfdOrder222(u) (PDstandardNthfdOrder222_impl(u,p1ody2,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder222_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1ody2, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder222_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1ody2, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1ody2,kadd(KRANC_GFOFFSET3D(u,0,-1,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-2),KRANC_GFOFFSET3D(u,0,1,0))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder233(u) (kmul(p1odz2,kadd(KRANC_GFOFFSET3D(u,0,0,-1),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-2),KRANC_GFOFFSET3D(u,0,0,1)))))
#else
#  define PDstandardNthfdOrder233(u) (PDstandardNthfdOrder233_impl(u,p1odz2,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder233_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1odz2, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder233_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1odz2, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1odz2,kadd(KRANC_GFOFFSET3D(u,0,0,-1),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-2),KRANC_GFOFFSET3D(u,0,0,1))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder411(u) (kmul(pm1o12dx2,kadd(KRANC_GFOFFSET3D(u,-2,0,0),kadd(KRANC_GFOFFSET3D(u,2,0,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(-16),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(30)))))))
#else
#  define PDstandardNthfdOrder411(u) (PDstandardNthfdOrder411_impl(u,pm1o12dx2,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder411_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o12dx2, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder411_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o12dx2, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(pm1o12dx2,kadd(KRANC_GFOFFSET3D(u,-2,0,0),kadd(KRANC_GFOFFSET3D(u,2,0,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(-16),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(30))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder422(u) (kmul(pm1o12dy2,kadd(KRANC_GFOFFSET3D(u,0,-2,0),kadd(KRANC_GFOFFSET3D(u,0,2,0),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(-16),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(30)))))))
#else
#  define PDstandardNthfdOrder422(u) (PDstandardNthfdOrder422_impl(u,pm1o12dy2,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder422_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o12dy2, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder422_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o12dy2, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(pm1o12dy2,kadd(KRANC_GFOFFSET3D(u,0,-2,0),kadd(KRANC_GFOFFSET3D(u,0,2,0),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(-16),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(30))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder433(u) (kmul(pm1o12dz2,kadd(KRANC_GFOFFSET3D(u,0,0,-2),kadd(KRANC_GFOFFSET3D(u,0,0,2),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(-16),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(30)))))))
#else
#  define PDstandardNthfdOrder433(u) (PDstandardNthfdOrder433_impl(u,pm1o12dz2,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder433_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o12dz2, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder433_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o12dz2, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(pm1o12dz2,kadd(KRANC_GFOFFSET3D(u,0,0,-2),kadd(KRANC_GFOFFSET3D(u,0,0,2),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(-16),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(30))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder611(u) (kmul(p1o180dx2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-490),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(-27),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,0),KRANC_GFOFFSET3D(u,3,0,0)),ToReal(2),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(270)))))))
#else
#  define PDstandardNthfdOrder611(u) (PDstandardNthfdOrder611_impl(u,p1o180dx2,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder611_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o180dx2, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder611_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o180dx2, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o180dx2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-490),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(-27),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,0),KRANC_GFOFFSET3D(u,3,0,0)),ToReal(2),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(270))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder622(u) (kmul(p1o180dy2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-490),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(-27),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,0),KRANC_GFOFFSET3D(u,0,3,0)),ToReal(2),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(270)))))))
#else
#  define PDstandardNthfdOrder622(u) (PDstandardNthfdOrder622_impl(u,p1o180dy2,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder622_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o180dy2, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder622_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o180dy2, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o180dy2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-490),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(-27),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,0),KRANC_GFOFFSET3D(u,0,3,0)),ToReal(2),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(270))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder633(u) (kmul(p1o180dz2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-490),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(-27),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-3),KRANC_GFOFFSET3D(u,0,0,3)),ToReal(2),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(270)))))))
#else
#  define PDstandardNthfdOrder633(u) (PDstandardNthfdOrder633_impl(u,p1o180dz2,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder633_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o180dz2, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder633_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o180dz2, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o180dz2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-490),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(-27),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-3),KRANC_GFOFFSET3D(u,0,0,3)),ToReal(2),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(270))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder811(u) (kmul(p1o5040dx2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-14350),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(-1008),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,0),KRANC_GFOFFSET3D(u,4,0,0)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,0),KRANC_GFOFFSET3D(u,3,0,0)),ToReal(128),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(8064))))))))
#else
#  define PDstandardNthfdOrder811(u) (PDstandardNthfdOrder811_impl(u,p1o5040dx2,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder811_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o5040dx2, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder811_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o5040dx2, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o5040dx2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-14350),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(-1008),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,0),KRANC_GFOFFSET3D(u,4,0,0)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,0),KRANC_GFOFFSET3D(u,3,0,0)),ToReal(128),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(8064)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder822(u) (kmul(p1o5040dy2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-14350),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(-1008),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,0),KRANC_GFOFFSET3D(u,0,4,0)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,0),KRANC_GFOFFSET3D(u,0,3,0)),ToReal(128),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(8064))))))))
#else
#  define PDstandardNthfdOrder822(u) (PDstandardNthfdOrder822_impl(u,p1o5040dy2,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder822_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o5040dy2, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder822_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o5040dy2, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o5040dy2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-14350),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(-1008),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,0),KRANC_GFOFFSET3D(u,0,4,0)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,0),KRANC_GFOFFSET3D(u,0,3,0)),ToReal(128),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(8064)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder833(u) (kmul(p1o5040dz2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-14350),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(-1008),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-4),KRANC_GFOFFSET3D(u,0,0,4)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-3),KRANC_GFOFFSET3D(u,0,0,3)),ToReal(128),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(8064))))))))
#else
#  define PDstandardNthfdOrder833(u) (PDstandardNthfdOrder833_impl(u,p1o5040dz2,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder833_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o5040dz2, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder833_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o5040dz2, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o5040dz2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-14350),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(-1008),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-4),KRANC_GFOFFSET3D(u,0,0,4)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-3),KRANC_GFOFFSET3D(u,0,0,3)),ToReal(128),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(8064)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder212(u) (kmul(p1o4dxdy,kadd(KRANC_GFOFFSET3D(u,-1,-1,0),ksub(KRANC_GFOFFSET3D(u,1,1,0),kadd(KRANC_GFOFFSET3D(u,1,-1,0),KRANC_GFOFFSET3D(u,-1,1,0))))))
#else
#  define PDstandardNthfdOrder212(u) (PDstandardNthfdOrder212_impl(u,p1o4dxdy,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder212_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder212_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o4dxdy,kadd(KRANC_GFOFFSET3D(u,-1,-1,0),ksub(KRANC_GFOFFSET3D(u,1,1,0),kadd(KRANC_GFOFFSET3D(u,1,-1,0),KRANC_GFOFFSET3D(u,-1,1,0)))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder213(u) (kmul(p1o4dxdz,kadd(KRANC_GFOFFSET3D(u,-1,0,-1),ksub(KRANC_GFOFFSET3D(u,1,0,1),kadd(KRANC_GFOFFSET3D(u,1,0,-1),KRANC_GFOFFSET3D(u,-1,0,1))))))
#else
#  define PDstandardNthfdOrder213(u) (PDstandardNthfdOrder213_impl(u,p1o4dxdz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder213_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder213_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o4dxdz,kadd(KRANC_GFOFFSET3D(u,-1,0,-1),ksub(KRANC_GFOFFSET3D(u,1,0,1),kadd(KRANC_GFOFFSET3D(u,1,0,-1),KRANC_GFOFFSET3D(u,-1,0,1)))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder221(u) (kmul(p1o4dxdy,kadd(KRANC_GFOFFSET3D(u,-1,-1,0),ksub(KRANC_GFOFFSET3D(u,1,1,0),kadd(KRANC_GFOFFSET3D(u,1,-1,0),KRANC_GFOFFSET3D(u,-1,1,0))))))
#else
#  define PDstandardNthfdOrder221(u) (PDstandardNthfdOrder221_impl(u,p1o4dxdy,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder221_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder221_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o4dxdy,kadd(KRANC_GFOFFSET3D(u,-1,-1,0),ksub(KRANC_GFOFFSET3D(u,1,1,0),kadd(KRANC_GFOFFSET3D(u,1,-1,0),KRANC_GFOFFSET3D(u,-1,1,0)))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder223(u) (kmul(p1o4dydz,kadd(KRANC_GFOFFSET3D(u,0,-1,-1),ksub(KRANC_GFOFFSET3D(u,0,1,1),kadd(KRANC_GFOFFSET3D(u,0,1,-1),KRANC_GFOFFSET3D(u,0,-1,1))))))
#else
#  define PDstandardNthfdOrder223(u) (PDstandardNthfdOrder223_impl(u,p1o4dydz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder223_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dydz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder223_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dydz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o4dydz,kadd(KRANC_GFOFFSET3D(u,0,-1,-1),ksub(KRANC_GFOFFSET3D(u,0,1,1),kadd(KRANC_GFOFFSET3D(u,0,1,-1),KRANC_GFOFFSET3D(u,0,-1,1)))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder231(u) (kmul(p1o4dxdz,kadd(KRANC_GFOFFSET3D(u,-1,0,-1),ksub(KRANC_GFOFFSET3D(u,1,0,1),kadd(KRANC_GFOFFSET3D(u,1,0,-1),KRANC_GFOFFSET3D(u,-1,0,1))))))
#else
#  define PDstandardNthfdOrder231(u) (PDstandardNthfdOrder231_impl(u,p1o4dxdz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder231_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder231_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o4dxdz,kadd(KRANC_GFOFFSET3D(u,-1,0,-1),ksub(KRANC_GFOFFSET3D(u,1,0,1),kadd(KRANC_GFOFFSET3D(u,1,0,-1),KRANC_GFOFFSET3D(u,-1,0,1)))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder232(u) (kmul(p1o4dydz,kadd(KRANC_GFOFFSET3D(u,0,-1,-1),ksub(KRANC_GFOFFSET3D(u,0,1,1),kadd(KRANC_GFOFFSET3D(u,0,1,-1),KRANC_GFOFFSET3D(u,0,-1,1))))))
#else
#  define PDstandardNthfdOrder232(u) (PDstandardNthfdOrder232_impl(u,p1o4dydz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder232_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dydz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder232_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dydz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o4dydz,kadd(KRANC_GFOFFSET3D(u,0,-1,-1),ksub(KRANC_GFOFFSET3D(u,0,1,1),kadd(KRANC_GFOFFSET3D(u,0,1,-1),KRANC_GFOFFSET3D(u,0,-1,1)))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder412(u) (kmul(p1o144dxdy,kadd(KRANC_GFOFFSET3D(u,-2,-2,0),kadd(KRANC_GFOFFSET3D(u,2,2,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,1,0),KRANC_GFOFFSET3D(u,1,-1,0)),ToReal(-64),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-2,0),kadd(KRANC_GFOFFSET3D(u,1,2,0),kadd(KRANC_GFOFFSET3D(u,-2,-1,0),KRANC_GFOFFSET3D(u,2,1,0)))),ToReal(-8),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-1,2,0),kadd(KRANC_GFOFFSET3D(u,1,-2,0),kadd(KRANC_GFOFFSET3D(u,-2,1,0),KRANC_GFOFFSET3D(u,2,-1,0)))),ToReal(8),kmul(kadd(KRANC_GFOFFSET3D(u,-1,-1,0),KRANC_GFOFFSET3D(u,1,1,0)),ToReal(64))),KRANC_GFOFFSET3D(u,2,-2,0)),KRANC_GFOFFSET3D(u,-2,2,0))))))))
#else
#  define PDstandardNthfdOrder412(u) (PDstandardNthfdOrder412_impl(u,p1o144dxdy,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder412_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o144dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder412_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o144dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o144dxdy,kadd(KRANC_GFOFFSET3D(u,-2,-2,0),kadd(KRANC_GFOFFSET3D(u,2,2,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,1,0),KRANC_GFOFFSET3D(u,1,-1,0)),ToReal(-64),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-2,0),kadd(KRANC_GFOFFSET3D(u,1,2,0),kadd(KRANC_GFOFFSET3D(u,-2,-1,0),KRANC_GFOFFSET3D(u,2,1,0)))),ToReal(-8),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-1,2,0),kadd(KRANC_GFOFFSET3D(u,1,-2,0),kadd(KRANC_GFOFFSET3D(u,-2,1,0),KRANC_GFOFFSET3D(u,2,-1,0)))),ToReal(8),kmul(kadd(KRANC_GFOFFSET3D(u,-1,-1,0),KRANC_GFOFFSET3D(u,1,1,0)),ToReal(64))),KRANC_GFOFFSET3D(u,2,-2,0)),KRANC_GFOFFSET3D(u,-2,2,0)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder413(u) (kmul(p1o144dxdz,kadd(KRANC_GFOFFSET3D(u,-2,0,-2),kadd(KRANC_GFOFFSET3D(u,2,0,2),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,1),KRANC_GFOFFSET3D(u,1,0,-1)),ToReal(-64),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-2),kadd(KRANC_GFOFFSET3D(u,1,0,2),kadd(KRANC_GFOFFSET3D(u,-2,0,-1),KRANC_GFOFFSET3D(u,2,0,1)))),ToReal(-8),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,2),kadd(KRANC_GFOFFSET3D(u,1,0,-2),kadd(KRANC_GFOFFSET3D(u,-2,0,1),KRANC_GFOFFSET3D(u,2,0,-1)))),ToReal(8),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,-1),KRANC_GFOFFSET3D(u,1,0,1)),ToReal(64))),KRANC_GFOFFSET3D(u,2,0,-2)),KRANC_GFOFFSET3D(u,-2,0,2))))))))
#else
#  define PDstandardNthfdOrder413(u) (PDstandardNthfdOrder413_impl(u,p1o144dxdz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder413_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o144dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder413_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o144dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o144dxdz,kadd(KRANC_GFOFFSET3D(u,-2,0,-2),kadd(KRANC_GFOFFSET3D(u,2,0,2),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,1),KRANC_GFOFFSET3D(u,1,0,-1)),ToReal(-64),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-2),kadd(KRANC_GFOFFSET3D(u,1,0,2),kadd(KRANC_GFOFFSET3D(u,-2,0,-1),KRANC_GFOFFSET3D(u,2,0,1)))),ToReal(-8),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,2),kadd(KRANC_GFOFFSET3D(u,1,0,-2),kadd(KRANC_GFOFFSET3D(u,-2,0,1),KRANC_GFOFFSET3D(u,2,0,-1)))),ToReal(8),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,-1),KRANC_GFOFFSET3D(u,1,0,1)),ToReal(64))),KRANC_GFOFFSET3D(u,2,0,-2)),KRANC_GFOFFSET3D(u,-2,0,2)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder421(u) (kmul(p1o144dxdy,kadd(KRANC_GFOFFSET3D(u,-2,-2,0),kadd(KRANC_GFOFFSET3D(u,2,2,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,1,0),KRANC_GFOFFSET3D(u,1,-1,0)),ToReal(-64),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-2,0),kadd(KRANC_GFOFFSET3D(u,1,2,0),kadd(KRANC_GFOFFSET3D(u,-2,-1,0),KRANC_GFOFFSET3D(u,2,1,0)))),ToReal(-8),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-1,2,0),kadd(KRANC_GFOFFSET3D(u,1,-2,0),kadd(KRANC_GFOFFSET3D(u,-2,1,0),KRANC_GFOFFSET3D(u,2,-1,0)))),ToReal(8),kmul(kadd(KRANC_GFOFFSET3D(u,-1,-1,0),KRANC_GFOFFSET3D(u,1,1,0)),ToReal(64))),KRANC_GFOFFSET3D(u,2,-2,0)),KRANC_GFOFFSET3D(u,-2,2,0))))))))
#else
#  define PDstandardNthfdOrder421(u) (PDstandardNthfdOrder421_impl(u,p1o144dxdy,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder421_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o144dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder421_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o144dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o144dxdy,kadd(KRANC_GFOFFSET3D(u,-2,-2,0),kadd(KRANC_GFOFFSET3D(u,2,2,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,1,0),KRANC_GFOFFSET3D(u,1,-1,0)),ToReal(-64),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-2,0),kadd(KRANC_GFOFFSET3D(u,1,2,0),kadd(KRANC_GFOFFSET3D(u,-2,-1,0),KRANC_GFOFFSET3D(u,2,1,0)))),ToReal(-8),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-1,2,0),kadd(KRANC_GFOFFSET3D(u,1,-2,0),kadd(KRANC_GFOFFSET3D(u,-2,1,0),KRANC_GFOFFSET3D(u,2,-1,0)))),ToReal(8),kmul(kadd(KRANC_GFOFFSET3D(u,-1,-1,0),KRANC_GFOFFSET3D(u,1,1,0)),ToReal(64))),KRANC_GFOFFSET3D(u,2,-2,0)),KRANC_GFOFFSET3D(u,-2,2,0)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder423(u) (kmul(p1o144dydz,kadd(KRANC_GFOFFSET3D(u,0,-2,-2),kadd(KRANC_GFOFFSET3D(u,0,2,2),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,1),KRANC_GFOFFSET3D(u,0,1,-1)),ToReal(-64),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-2),kadd(KRANC_GFOFFSET3D(u,0,1,2),kadd(KRANC_GFOFFSET3D(u,0,-2,-1),KRANC_GFOFFSET3D(u,0,2,1)))),ToReal(-8),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,2),kadd(KRANC_GFOFFSET3D(u,0,1,-2),kadd(KRANC_GFOFFSET3D(u,0,-2,1),KRANC_GFOFFSET3D(u,0,2,-1)))),ToReal(8),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,-1),KRANC_GFOFFSET3D(u,0,1,1)),ToReal(64))),KRANC_GFOFFSET3D(u,0,2,-2)),KRANC_GFOFFSET3D(u,0,-2,2))))))))
#else
#  define PDstandardNthfdOrder423(u) (PDstandardNthfdOrder423_impl(u,p1o144dydz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder423_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o144dydz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder423_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o144dydz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o144dydz,kadd(KRANC_GFOFFSET3D(u,0,-2,-2),kadd(KRANC_GFOFFSET3D(u,0,2,2),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,1),KRANC_GFOFFSET3D(u,0,1,-1)),ToReal(-64),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-2),kadd(KRANC_GFOFFSET3D(u,0,1,2),kadd(KRANC_GFOFFSET3D(u,0,-2,-1),KRANC_GFOFFSET3D(u,0,2,1)))),ToReal(-8),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,2),kadd(KRANC_GFOFFSET3D(u,0,1,-2),kadd(KRANC_GFOFFSET3D(u,0,-2,1),KRANC_GFOFFSET3D(u,0,2,-1)))),ToReal(8),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,-1),KRANC_GFOFFSET3D(u,0,1,1)),ToReal(64))),KRANC_GFOFFSET3D(u,0,2,-2)),KRANC_GFOFFSET3D(u,0,-2,2)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder431(u) (kmul(p1o144dxdz,kadd(KRANC_GFOFFSET3D(u,-2,0,-2),kadd(KRANC_GFOFFSET3D(u,2,0,2),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,1),KRANC_GFOFFSET3D(u,1,0,-1)),ToReal(-64),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-2),kadd(KRANC_GFOFFSET3D(u,1,0,2),kadd(KRANC_GFOFFSET3D(u,-2,0,-1),KRANC_GFOFFSET3D(u,2,0,1)))),ToReal(-8),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,2),kadd(KRANC_GFOFFSET3D(u,1,0,-2),kadd(KRANC_GFOFFSET3D(u,-2,0,1),KRANC_GFOFFSET3D(u,2,0,-1)))),ToReal(8),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,-1),KRANC_GFOFFSET3D(u,1,0,1)),ToReal(64))),KRANC_GFOFFSET3D(u,2,0,-2)),KRANC_GFOFFSET3D(u,-2,0,2))))))))
#else
#  define PDstandardNthfdOrder431(u) (PDstandardNthfdOrder431_impl(u,p1o144dxdz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder431_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o144dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder431_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o144dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o144dxdz,kadd(KRANC_GFOFFSET3D(u,-2,0,-2),kadd(KRANC_GFOFFSET3D(u,2,0,2),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,1),KRANC_GFOFFSET3D(u,1,0,-1)),ToReal(-64),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-2),kadd(KRANC_GFOFFSET3D(u,1,0,2),kadd(KRANC_GFOFFSET3D(u,-2,0,-1),KRANC_GFOFFSET3D(u,2,0,1)))),ToReal(-8),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,2),kadd(KRANC_GFOFFSET3D(u,1,0,-2),kadd(KRANC_GFOFFSET3D(u,-2,0,1),KRANC_GFOFFSET3D(u,2,0,-1)))),ToReal(8),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,-1),KRANC_GFOFFSET3D(u,1,0,1)),ToReal(64))),KRANC_GFOFFSET3D(u,2,0,-2)),KRANC_GFOFFSET3D(u,-2,0,2)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder432(u) (kmul(p1o144dydz,kadd(KRANC_GFOFFSET3D(u,0,-2,-2),kadd(KRANC_GFOFFSET3D(u,0,2,2),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,1),KRANC_GFOFFSET3D(u,0,1,-1)),ToReal(-64),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-2),kadd(KRANC_GFOFFSET3D(u,0,1,2),kadd(KRANC_GFOFFSET3D(u,0,-2,-1),KRANC_GFOFFSET3D(u,0,2,1)))),ToReal(-8),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,2),kadd(KRANC_GFOFFSET3D(u,0,1,-2),kadd(KRANC_GFOFFSET3D(u,0,-2,1),KRANC_GFOFFSET3D(u,0,2,-1)))),ToReal(8),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,-1),KRANC_GFOFFSET3D(u,0,1,1)),ToReal(64))),KRANC_GFOFFSET3D(u,0,2,-2)),KRANC_GFOFFSET3D(u,0,-2,2))))))))
#else
#  define PDstandardNthfdOrder432(u) (PDstandardNthfdOrder432_impl(u,p1o144dydz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder432_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o144dydz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder432_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o144dydz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o144dydz,kadd(KRANC_GFOFFSET3D(u,0,-2,-2),kadd(KRANC_GFOFFSET3D(u,0,2,2),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,1),KRANC_GFOFFSET3D(u,0,1,-1)),ToReal(-64),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-2),kadd(KRANC_GFOFFSET3D(u,0,1,2),kadd(KRANC_GFOFFSET3D(u,0,-2,-1),KRANC_GFOFFSET3D(u,0,2,1)))),ToReal(-8),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,2),kadd(KRANC_GFOFFSET3D(u,0,1,-2),kadd(KRANC_GFOFFSET3D(u,0,-2,1),KRANC_GFOFFSET3D(u,0,2,-1)))),ToReal(8),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,-1),KRANC_GFOFFSET3D(u,0,1,1)),ToReal(64))),KRANC_GFOFFSET3D(u,0,2,-2)),KRANC_GFOFFSET3D(u,0,-2,2)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder612(u) (kmul(p1o3600dxdy,kadd(KRANC_GFOFFSET3D(u,-3,-3,0),kadd(KRANC_GFOFFSET3D(u,3,3,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,1,0),KRANC_GFOFFSET3D(u,1,-1,0)),ToReal(-2025),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-2,0),kadd(KRANC_GFOFFSET3D(u,1,2,0),kadd(KRANC_GFOFFSET3D(u,-2,-1,0),KRANC_GFOFFSET3D(u,2,1,0)))),ToReal(-405),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,2,0),KRANC_GFOFFSET3D(u,2,-2,0)),ToReal(-81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,3,0),kadd(KRANC_GFOFFSET3D(u,1,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,1,0),KRANC_GFOFFSET3D(u,3,-1,0)))),ToReal(-45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-3,0),kadd(KRANC_GFOFFSET3D(u,2,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-2,0),KRANC_GFOFFSET3D(u,3,2,0)))),ToReal(-9),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-2,3,0),kadd(KRANC_GFOFFSET3D(u,2,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,2,0),KRANC_GFOFFSET3D(u,3,-2,0)))),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-3,0),kadd(KRANC_GFOFFSET3D(u,1,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-1,0),KRANC_GFOFFSET3D(u,3,1,0)))),ToReal(45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-2,0),KRANC_GFOFFSET3D(u,2,2,0)),ToReal(81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,2,0),kadd(KRANC_GFOFFSET3D(u,1,-2,0),kadd(KRANC_GFOFFSET3D(u,-2,1,0),KRANC_GFOFFSET3D(u,2,-1,0)))),ToReal(405),kmul(kadd(KRANC_GFOFFSET3D(u,-1,-1,0),KRANC_GFOFFSET3D(u,1,1,0)),ToReal(2025)))))),KRANC_GFOFFSET3D(u,3,-3,0)),KRANC_GFOFFSET3D(u,-3,3,0)))))))))))
#else
#  define PDstandardNthfdOrder612(u) (PDstandardNthfdOrder612_impl(u,p1o3600dxdy,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder612_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o3600dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder612_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o3600dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o3600dxdy,kadd(KRANC_GFOFFSET3D(u,-3,-3,0),kadd(KRANC_GFOFFSET3D(u,3,3,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,1,0),KRANC_GFOFFSET3D(u,1,-1,0)),ToReal(-2025),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-2,0),kadd(KRANC_GFOFFSET3D(u,1,2,0),kadd(KRANC_GFOFFSET3D(u,-2,-1,0),KRANC_GFOFFSET3D(u,2,1,0)))),ToReal(-405),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,2,0),KRANC_GFOFFSET3D(u,2,-2,0)),ToReal(-81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,3,0),kadd(KRANC_GFOFFSET3D(u,1,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,1,0),KRANC_GFOFFSET3D(u,3,-1,0)))),ToReal(-45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-3,0),kadd(KRANC_GFOFFSET3D(u,2,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-2,0),KRANC_GFOFFSET3D(u,3,2,0)))),ToReal(-9),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-2,3,0),kadd(KRANC_GFOFFSET3D(u,2,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,2,0),KRANC_GFOFFSET3D(u,3,-2,0)))),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-3,0),kadd(KRANC_GFOFFSET3D(u,1,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-1,0),KRANC_GFOFFSET3D(u,3,1,0)))),ToReal(45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-2,0),KRANC_GFOFFSET3D(u,2,2,0)),ToReal(81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,2,0),kadd(KRANC_GFOFFSET3D(u,1,-2,0),kadd(KRANC_GFOFFSET3D(u,-2,1,0),KRANC_GFOFFSET3D(u,2,-1,0)))),ToReal(405),kmul(kadd(KRANC_GFOFFSET3D(u,-1,-1,0),KRANC_GFOFFSET3D(u,1,1,0)),ToReal(2025)))))),KRANC_GFOFFSET3D(u,3,-3,0)),KRANC_GFOFFSET3D(u,-3,3,0))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder613(u) (kmul(p1o3600dxdz,kadd(KRANC_GFOFFSET3D(u,-3,0,-3),kadd(KRANC_GFOFFSET3D(u,3,0,3),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,1),KRANC_GFOFFSET3D(u,1,0,-1)),ToReal(-2025),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-2),kadd(KRANC_GFOFFSET3D(u,1,0,2),kadd(KRANC_GFOFFSET3D(u,-2,0,-1),KRANC_GFOFFSET3D(u,2,0,1)))),ToReal(-405),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,2),KRANC_GFOFFSET3D(u,2,0,-2)),ToReal(-81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,3),kadd(KRANC_GFOFFSET3D(u,1,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,1),KRANC_GFOFFSET3D(u,3,0,-1)))),ToReal(-45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-3),kadd(KRANC_GFOFFSET3D(u,2,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-2),KRANC_GFOFFSET3D(u,3,0,2)))),ToReal(-9),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,3),kadd(KRANC_GFOFFSET3D(u,2,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,2),KRANC_GFOFFSET3D(u,3,0,-2)))),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-3),kadd(KRANC_GFOFFSET3D(u,1,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-1),KRANC_GFOFFSET3D(u,3,0,1)))),ToReal(45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-2),KRANC_GFOFFSET3D(u,2,0,2)),ToReal(81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,2),kadd(KRANC_GFOFFSET3D(u,1,0,-2),kadd(KRANC_GFOFFSET3D(u,-2,0,1),KRANC_GFOFFSET3D(u,2,0,-1)))),ToReal(405),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,-1),KRANC_GFOFFSET3D(u,1,0,1)),ToReal(2025)))))),KRANC_GFOFFSET3D(u,3,0,-3)),KRANC_GFOFFSET3D(u,-3,0,3)))))))))))
#else
#  define PDstandardNthfdOrder613(u) (PDstandardNthfdOrder613_impl(u,p1o3600dxdz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder613_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o3600dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder613_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o3600dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o3600dxdz,kadd(KRANC_GFOFFSET3D(u,-3,0,-3),kadd(KRANC_GFOFFSET3D(u,3,0,3),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,1),KRANC_GFOFFSET3D(u,1,0,-1)),ToReal(-2025),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-2),kadd(KRANC_GFOFFSET3D(u,1,0,2),kadd(KRANC_GFOFFSET3D(u,-2,0,-1),KRANC_GFOFFSET3D(u,2,0,1)))),ToReal(-405),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,2),KRANC_GFOFFSET3D(u,2,0,-2)),ToReal(-81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,3),kadd(KRANC_GFOFFSET3D(u,1,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,1),KRANC_GFOFFSET3D(u,3,0,-1)))),ToReal(-45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-3),kadd(KRANC_GFOFFSET3D(u,2,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-2),KRANC_GFOFFSET3D(u,3,0,2)))),ToReal(-9),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,3),kadd(KRANC_GFOFFSET3D(u,2,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,2),KRANC_GFOFFSET3D(u,3,0,-2)))),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-3),kadd(KRANC_GFOFFSET3D(u,1,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-1),KRANC_GFOFFSET3D(u,3,0,1)))),ToReal(45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-2),KRANC_GFOFFSET3D(u,2,0,2)),ToReal(81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,2),kadd(KRANC_GFOFFSET3D(u,1,0,-2),kadd(KRANC_GFOFFSET3D(u,-2,0,1),KRANC_GFOFFSET3D(u,2,0,-1)))),ToReal(405),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,-1),KRANC_GFOFFSET3D(u,1,0,1)),ToReal(2025)))))),KRANC_GFOFFSET3D(u,3,0,-3)),KRANC_GFOFFSET3D(u,-3,0,3))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder621(u) (kmul(p1o3600dxdy,kadd(KRANC_GFOFFSET3D(u,-3,-3,0),kadd(KRANC_GFOFFSET3D(u,3,3,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,1,0),KRANC_GFOFFSET3D(u,1,-1,0)),ToReal(-2025),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-2,0),kadd(KRANC_GFOFFSET3D(u,1,2,0),kadd(KRANC_GFOFFSET3D(u,-2,-1,0),KRANC_GFOFFSET3D(u,2,1,0)))),ToReal(-405),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,2,0),KRANC_GFOFFSET3D(u,2,-2,0)),ToReal(-81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,3,0),kadd(KRANC_GFOFFSET3D(u,1,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,1,0),KRANC_GFOFFSET3D(u,3,-1,0)))),ToReal(-45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-3,0),kadd(KRANC_GFOFFSET3D(u,2,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-2,0),KRANC_GFOFFSET3D(u,3,2,0)))),ToReal(-9),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-2,3,0),kadd(KRANC_GFOFFSET3D(u,2,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,2,0),KRANC_GFOFFSET3D(u,3,-2,0)))),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-3,0),kadd(KRANC_GFOFFSET3D(u,1,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-1,0),KRANC_GFOFFSET3D(u,3,1,0)))),ToReal(45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-2,0),KRANC_GFOFFSET3D(u,2,2,0)),ToReal(81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,2,0),kadd(KRANC_GFOFFSET3D(u,1,-2,0),kadd(KRANC_GFOFFSET3D(u,-2,1,0),KRANC_GFOFFSET3D(u,2,-1,0)))),ToReal(405),kmul(kadd(KRANC_GFOFFSET3D(u,-1,-1,0),KRANC_GFOFFSET3D(u,1,1,0)),ToReal(2025)))))),KRANC_GFOFFSET3D(u,3,-3,0)),KRANC_GFOFFSET3D(u,-3,3,0)))))))))))
#else
#  define PDstandardNthfdOrder621(u) (PDstandardNthfdOrder621_impl(u,p1o3600dxdy,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder621_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o3600dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder621_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o3600dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o3600dxdy,kadd(KRANC_GFOFFSET3D(u,-3,-3,0),kadd(KRANC_GFOFFSET3D(u,3,3,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,1,0),KRANC_GFOFFSET3D(u,1,-1,0)),ToReal(-2025),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-2,0),kadd(KRANC_GFOFFSET3D(u,1,2,0),kadd(KRANC_GFOFFSET3D(u,-2,-1,0),KRANC_GFOFFSET3D(u,2,1,0)))),ToReal(-405),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,2,0),KRANC_GFOFFSET3D(u,2,-2,0)),ToReal(-81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,3,0),kadd(KRANC_GFOFFSET3D(u,1,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,1,0),KRANC_GFOFFSET3D(u,3,-1,0)))),ToReal(-45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-3,0),kadd(KRANC_GFOFFSET3D(u,2,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-2,0),KRANC_GFOFFSET3D(u,3,2,0)))),ToReal(-9),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-2,3,0),kadd(KRANC_GFOFFSET3D(u,2,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,2,0),KRANC_GFOFFSET3D(u,3,-2,0)))),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-3,0),kadd(KRANC_GFOFFSET3D(u,1,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-1,0),KRANC_GFOFFSET3D(u,3,1,0)))),ToReal(45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-2,0),KRANC_GFOFFSET3D(u,2,2,0)),ToReal(81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,2,0),kadd(KRANC_GFOFFSET3D(u,1,-2,0),kadd(KRANC_GFOFFSET3D(u,-2,1,0),KRANC_GFOFFSET3D(u,2,-1,0)))),ToReal(405),kmul(kadd(KRANC_GFOFFSET3D(u,-1,-1,0),KRANC_GFOFFSET3D(u,1,1,0)),ToReal(2025)))))),KRANC_GFOFFSET3D(u,3,-3,0)),KRANC_GFOFFSET3D(u,-3,3,0))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder623(u) (kmul(p1o3600dydz,kadd(KRANC_GFOFFSET3D(u,0,-3,-3),kadd(KRANC_GFOFFSET3D(u,0,3,3),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,1),KRANC_GFOFFSET3D(u,0,1,-1)),ToReal(-2025),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-2),kadd(KRANC_GFOFFSET3D(u,0,1,2),kadd(KRANC_GFOFFSET3D(u,0,-2,-1),KRANC_GFOFFSET3D(u,0,2,1)))),ToReal(-405),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,2),KRANC_GFOFFSET3D(u,0,2,-2)),ToReal(-81),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,3),kadd(KRANC_GFOFFSET3D(u,0,1,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,1),KRANC_GFOFFSET3D(u,0,3,-1)))),ToReal(-45),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-3),kadd(KRANC_GFOFFSET3D(u,0,2,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-2),KRANC_GFOFFSET3D(u,0,3,2)))),ToReal(-9),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,3),kadd(KRANC_GFOFFSET3D(u,0,2,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,2),KRANC_GFOFFSET3D(u,0,3,-2)))),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-3),kadd(KRANC_GFOFFSET3D(u,0,1,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-1),KRANC_GFOFFSET3D(u,0,3,1)))),ToReal(45),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-2),KRANC_GFOFFSET3D(u,0,2,2)),ToReal(81),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,2),kadd(KRANC_GFOFFSET3D(u,0,1,-2),kadd(KRANC_GFOFFSET3D(u,0,-2,1),KRANC_GFOFFSET3D(u,0,2,-1)))),ToReal(405),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,-1),KRANC_GFOFFSET3D(u,0,1,1)),ToReal(2025)))))),KRANC_GFOFFSET3D(u,0,3,-3)),KRANC_GFOFFSET3D(u,0,-3,3)))))))))))
#else
#  define PDstandardNthfdOrder623(u) (PDstandardNthfdOrder623_impl(u,p1o3600dydz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder623_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o3600dydz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder623_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o3600dydz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o3600dydz,kadd(KRANC_GFOFFSET3D(u,0,-3,-3),kadd(KRANC_GFOFFSET3D(u,0,3,3),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,1),KRANC_GFOFFSET3D(u,0,1,-1)),ToReal(-2025),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-2),kadd(KRANC_GFOFFSET3D(u,0,1,2),kadd(KRANC_GFOFFSET3D(u,0,-2,-1),KRANC_GFOFFSET3D(u,0,2,1)))),ToReal(-405),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,2),KRANC_GFOFFSET3D(u,0,2,-2)),ToReal(-81),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,3),kadd(KRANC_GFOFFSET3D(u,0,1,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,1),KRANC_GFOFFSET3D(u,0,3,-1)))),ToReal(-45),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-3),kadd(KRANC_GFOFFSET3D(u,0,2,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-2),KRANC_GFOFFSET3D(u,0,3,2)))),ToReal(-9),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,3),kadd(KRANC_GFOFFSET3D(u,0,2,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,2),KRANC_GFOFFSET3D(u,0,3,-2)))),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-3),kadd(KRANC_GFOFFSET3D(u,0,1,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-1),KRANC_GFOFFSET3D(u,0,3,1)))),ToReal(45),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-2),KRANC_GFOFFSET3D(u,0,2,2)),ToReal(81),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,2),kadd(KRANC_GFOFFSET3D(u,0,1,-2),kadd(KRANC_GFOFFSET3D(u,0,-2,1),KRANC_GFOFFSET3D(u,0,2,-1)))),ToReal(405),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,-1),KRANC_GFOFFSET3D(u,0,1,1)),ToReal(2025)))))),KRANC_GFOFFSET3D(u,0,3,-3)),KRANC_GFOFFSET3D(u,0,-3,3))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder631(u) (kmul(p1o3600dxdz,kadd(KRANC_GFOFFSET3D(u,-3,0,-3),kadd(KRANC_GFOFFSET3D(u,3,0,3),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,1),KRANC_GFOFFSET3D(u,1,0,-1)),ToReal(-2025),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-2),kadd(KRANC_GFOFFSET3D(u,1,0,2),kadd(KRANC_GFOFFSET3D(u,-2,0,-1),KRANC_GFOFFSET3D(u,2,0,1)))),ToReal(-405),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,2),KRANC_GFOFFSET3D(u,2,0,-2)),ToReal(-81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,3),kadd(KRANC_GFOFFSET3D(u,1,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,1),KRANC_GFOFFSET3D(u,3,0,-1)))),ToReal(-45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-3),kadd(KRANC_GFOFFSET3D(u,2,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-2),KRANC_GFOFFSET3D(u,3,0,2)))),ToReal(-9),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,3),kadd(KRANC_GFOFFSET3D(u,2,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,2),KRANC_GFOFFSET3D(u,3,0,-2)))),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-3),kadd(KRANC_GFOFFSET3D(u,1,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-1),KRANC_GFOFFSET3D(u,3,0,1)))),ToReal(45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-2),KRANC_GFOFFSET3D(u,2,0,2)),ToReal(81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,2),kadd(KRANC_GFOFFSET3D(u,1,0,-2),kadd(KRANC_GFOFFSET3D(u,-2,0,1),KRANC_GFOFFSET3D(u,2,0,-1)))),ToReal(405),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,-1),KRANC_GFOFFSET3D(u,1,0,1)),ToReal(2025)))))),KRANC_GFOFFSET3D(u,3,0,-3)),KRANC_GFOFFSET3D(u,-3,0,3)))))))))))
#else
#  define PDstandardNthfdOrder631(u) (PDstandardNthfdOrder631_impl(u,p1o3600dxdz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder631_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o3600dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder631_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o3600dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o3600dxdz,kadd(KRANC_GFOFFSET3D(u,-3,0,-3),kadd(KRANC_GFOFFSET3D(u,3,0,3),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,1),KRANC_GFOFFSET3D(u,1,0,-1)),ToReal(-2025),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-2),kadd(KRANC_GFOFFSET3D(u,1,0,2),kadd(KRANC_GFOFFSET3D(u,-2,0,-1),KRANC_GFOFFSET3D(u,2,0,1)))),ToReal(-405),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,2),KRANC_GFOFFSET3D(u,2,0,-2)),ToReal(-81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,3),kadd(KRANC_GFOFFSET3D(u,1,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,1),KRANC_GFOFFSET3D(u,3,0,-1)))),ToReal(-45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-3),kadd(KRANC_GFOFFSET3D(u,2,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-2),KRANC_GFOFFSET3D(u,3,0,2)))),ToReal(-9),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,3),kadd(KRANC_GFOFFSET3D(u,2,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,2),KRANC_GFOFFSET3D(u,3,0,-2)))),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-3),kadd(KRANC_GFOFFSET3D(u,1,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-1),KRANC_GFOFFSET3D(u,3,0,1)))),ToReal(45),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-2),KRANC_GFOFFSET3D(u,2,0,2)),ToReal(81),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,2),kadd(KRANC_GFOFFSET3D(u,1,0,-2),kadd(KRANC_GFOFFSET3D(u,-2,0,1),KRANC_GFOFFSET3D(u,2,0,-1)))),ToReal(405),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,-1),KRANC_GFOFFSET3D(u,1,0,1)),ToReal(2025)))))),KRANC_GFOFFSET3D(u,3,0,-3)),KRANC_GFOFFSET3D(u,-3,0,3))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder632(u) (kmul(p1o3600dydz,kadd(KRANC_GFOFFSET3D(u,0,-3,-3),kadd(KRANC_GFOFFSET3D(u,0,3,3),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,1),KRANC_GFOFFSET3D(u,0,1,-1)),ToReal(-2025),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-2),kadd(KRANC_GFOFFSET3D(u,0,1,2),kadd(KRANC_GFOFFSET3D(u,0,-2,-1),KRANC_GFOFFSET3D(u,0,2,1)))),ToReal(-405),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,2),KRANC_GFOFFSET3D(u,0,2,-2)),ToReal(-81),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,3),kadd(KRANC_GFOFFSET3D(u,0,1,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,1),KRANC_GFOFFSET3D(u,0,3,-1)))),ToReal(-45),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-3),kadd(KRANC_GFOFFSET3D(u,0,2,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-2),KRANC_GFOFFSET3D(u,0,3,2)))),ToReal(-9),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,3),kadd(KRANC_GFOFFSET3D(u,0,2,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,2),KRANC_GFOFFSET3D(u,0,3,-2)))),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-3),kadd(KRANC_GFOFFSET3D(u,0,1,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-1),KRANC_GFOFFSET3D(u,0,3,1)))),ToReal(45),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-2),KRANC_GFOFFSET3D(u,0,2,2)),ToReal(81),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,2),kadd(KRANC_GFOFFSET3D(u,0,1,-2),kadd(KRANC_GFOFFSET3D(u,0,-2,1),KRANC_GFOFFSET3D(u,0,2,-1)))),ToReal(405),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,-1),KRANC_GFOFFSET3D(u,0,1,1)),ToReal(2025)))))),KRANC_GFOFFSET3D(u,0,3,-3)),KRANC_GFOFFSET3D(u,0,-3,3)))))))))))
#else
#  define PDstandardNthfdOrder632(u) (PDstandardNthfdOrder632_impl(u,p1o3600dydz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder632_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o3600dydz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder632_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o3600dydz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o3600dydz,kadd(KRANC_GFOFFSET3D(u,0,-3,-3),kadd(KRANC_GFOFFSET3D(u,0,3,3),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,1),KRANC_GFOFFSET3D(u,0,1,-1)),ToReal(-2025),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-2),kadd(KRANC_GFOFFSET3D(u,0,1,2),kadd(KRANC_GFOFFSET3D(u,0,-2,-1),KRANC_GFOFFSET3D(u,0,2,1)))),ToReal(-405),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,2),KRANC_GFOFFSET3D(u,0,2,-2)),ToReal(-81),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,3),kadd(KRANC_GFOFFSET3D(u,0,1,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,1),KRANC_GFOFFSET3D(u,0,3,-1)))),ToReal(-45),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-3),kadd(KRANC_GFOFFSET3D(u,0,2,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-2),KRANC_GFOFFSET3D(u,0,3,2)))),ToReal(-9),ksub(ksub(kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,3),kadd(KRANC_GFOFFSET3D(u,0,2,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,2),KRANC_GFOFFSET3D(u,0,3,-2)))),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-3),kadd(KRANC_GFOFFSET3D(u,0,1,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-1),KRANC_GFOFFSET3D(u,0,3,1)))),ToReal(45),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-2),KRANC_GFOFFSET3D(u,0,2,2)),ToReal(81),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,2),kadd(KRANC_GFOFFSET3D(u,0,1,-2),kadd(KRANC_GFOFFSET3D(u,0,-2,1),KRANC_GFOFFSET3D(u,0,2,-1)))),ToReal(405),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,-1),KRANC_GFOFFSET3D(u,0,1,1)),ToReal(2025)))))),KRANC_GFOFFSET3D(u,0,3,-3)),KRANC_GFOFFSET3D(u,0,-3,3))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder812(u) (kmul(p1o705600dxdy,kmadd(kadd(KRANC_GFOFFSET3D(u,-1,1,0),KRANC_GFOFFSET3D(u,1,-1,0)),ToReal(-451584),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-2,0),kadd(KRANC_GFOFFSET3D(u,1,2,0),kadd(KRANC_GFOFFSET3D(u,-2,-1,0),KRANC_GFOFFSET3D(u,2,1,0)))),ToReal(-112896),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,2,0),KRANC_GFOFFSET3D(u,2,-2,0)),ToReal(-28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,3,0),kadd(KRANC_GFOFFSET3D(u,1,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,1,0),KRANC_GFOFFSET3D(u,3,-1,0)))),ToReal(-21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-3,0),kadd(KRANC_GFOFFSET3D(u,2,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-2,0),KRANC_GFOFFSET3D(u,3,2,0)))),ToReal(-5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-4,0),kadd(KRANC_GFOFFSET3D(u,1,4,0),kadd(KRANC_GFOFFSET3D(u,-4,-1,0),KRANC_GFOFFSET3D(u,4,1,0)))),ToReal(-2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,3,0),KRANC_GFOFFSET3D(u,3,-3,0)),ToReal(-1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,4,0),kadd(KRANC_GFOFFSET3D(u,2,-4,0),kadd(KRANC_GFOFFSET3D(u,-4,2,0),KRANC_GFOFFSET3D(u,4,-2,0)))),ToReal(-504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,-4,0),kadd(KRANC_GFOFFSET3D(u,3,4,0),kadd(KRANC_GFOFFSET3D(u,-4,-3,0),KRANC_GFOFFSET3D(u,4,3,0)))),ToReal(-96),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,4,0),KRANC_GFOFFSET3D(u,4,-4,0)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,-4,0),KRANC_GFOFFSET3D(u,4,4,0)),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,4,0),kadd(KRANC_GFOFFSET3D(u,3,-4,0),kadd(KRANC_GFOFFSET3D(u,-4,3,0),KRANC_GFOFFSET3D(u,4,-3,0)))),ToReal(96),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-4,0),kadd(KRANC_GFOFFSET3D(u,2,4,0),kadd(KRANC_GFOFFSET3D(u,-4,-2,0),KRANC_GFOFFSET3D(u,4,2,0)))),ToReal(504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,-3,0),KRANC_GFOFFSET3D(u,3,3,0)),ToReal(1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,4,0),kadd(KRANC_GFOFFSET3D(u,1,-4,0),kadd(KRANC_GFOFFSET3D(u,-4,1,0),KRANC_GFOFFSET3D(u,4,-1,0)))),ToReal(2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,3,0),kadd(KRANC_GFOFFSET3D(u,2,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,2,0),KRANC_GFOFFSET3D(u,3,-2,0)))),ToReal(5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-3,0),kadd(KRANC_GFOFFSET3D(u,1,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-1,0),KRANC_GFOFFSET3D(u,3,1,0)))),ToReal(21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-2,0),KRANC_GFOFFSET3D(u,2,2,0)),ToReal(28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,2,0),kadd(KRANC_GFOFFSET3D(u,1,-2,0),kadd(KRANC_GFOFFSET3D(u,-2,1,0),KRANC_GFOFFSET3D(u,2,-1,0)))),ToReal(112896),kmul(kadd(KRANC_GFOFFSET3D(u,-1,-1,0),KRANC_GFOFFSET3D(u,1,1,0)),ToReal(451584)))))))))))))))))))))))
#else
#  define PDstandardNthfdOrder812(u) (PDstandardNthfdOrder812_impl(u,p1o705600dxdy,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder812_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o705600dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder812_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o705600dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o705600dxdy,kmadd(kadd(KRANC_GFOFFSET3D(u,-1,1,0),KRANC_GFOFFSET3D(u,1,-1,0)),ToReal(-451584),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-2,0),kadd(KRANC_GFOFFSET3D(u,1,2,0),kadd(KRANC_GFOFFSET3D(u,-2,-1,0),KRANC_GFOFFSET3D(u,2,1,0)))),ToReal(-112896),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,2,0),KRANC_GFOFFSET3D(u,2,-2,0)),ToReal(-28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,3,0),kadd(KRANC_GFOFFSET3D(u,1,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,1,0),KRANC_GFOFFSET3D(u,3,-1,0)))),ToReal(-21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-3,0),kadd(KRANC_GFOFFSET3D(u,2,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-2,0),KRANC_GFOFFSET3D(u,3,2,0)))),ToReal(-5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-4,0),kadd(KRANC_GFOFFSET3D(u,1,4,0),kadd(KRANC_GFOFFSET3D(u,-4,-1,0),KRANC_GFOFFSET3D(u,4,1,0)))),ToReal(-2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,3,0),KRANC_GFOFFSET3D(u,3,-3,0)),ToReal(-1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,4,0),kadd(KRANC_GFOFFSET3D(u,2,-4,0),kadd(KRANC_GFOFFSET3D(u,-4,2,0),KRANC_GFOFFSET3D(u,4,-2,0)))),ToReal(-504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,-4,0),kadd(KRANC_GFOFFSET3D(u,3,4,0),kadd(KRANC_GFOFFSET3D(u,-4,-3,0),KRANC_GFOFFSET3D(u,4,3,0)))),ToReal(-96),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,4,0),KRANC_GFOFFSET3D(u,4,-4,0)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,-4,0),KRANC_GFOFFSET3D(u,4,4,0)),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,4,0),kadd(KRANC_GFOFFSET3D(u,3,-4,0),kadd(KRANC_GFOFFSET3D(u,-4,3,0),KRANC_GFOFFSET3D(u,4,-3,0)))),ToReal(96),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-4,0),kadd(KRANC_GFOFFSET3D(u,2,4,0),kadd(KRANC_GFOFFSET3D(u,-4,-2,0),KRANC_GFOFFSET3D(u,4,2,0)))),ToReal(504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,-3,0),KRANC_GFOFFSET3D(u,3,3,0)),ToReal(1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,4,0),kadd(KRANC_GFOFFSET3D(u,1,-4,0),kadd(KRANC_GFOFFSET3D(u,-4,1,0),KRANC_GFOFFSET3D(u,4,-1,0)))),ToReal(2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,3,0),kadd(KRANC_GFOFFSET3D(u,2,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,2,0),KRANC_GFOFFSET3D(u,3,-2,0)))),ToReal(5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-3,0),kadd(KRANC_GFOFFSET3D(u,1,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-1,0),KRANC_GFOFFSET3D(u,3,1,0)))),ToReal(21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-2,0),KRANC_GFOFFSET3D(u,2,2,0)),ToReal(28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,2,0),kadd(KRANC_GFOFFSET3D(u,1,-2,0),kadd(KRANC_GFOFFSET3D(u,-2,1,0),KRANC_GFOFFSET3D(u,2,-1,0)))),ToReal(112896),kmul(kadd(KRANC_GFOFFSET3D(u,-1,-1,0),KRANC_GFOFFSET3D(u,1,1,0)),ToReal(451584))))))))))))))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder813(u) (kmul(p1o705600dxdz,kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,1),KRANC_GFOFFSET3D(u,1,0,-1)),ToReal(-451584),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-2),kadd(KRANC_GFOFFSET3D(u,1,0,2),kadd(KRANC_GFOFFSET3D(u,-2,0,-1),KRANC_GFOFFSET3D(u,2,0,1)))),ToReal(-112896),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,2),KRANC_GFOFFSET3D(u,2,0,-2)),ToReal(-28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,3),kadd(KRANC_GFOFFSET3D(u,1,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,1),KRANC_GFOFFSET3D(u,3,0,-1)))),ToReal(-21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-3),kadd(KRANC_GFOFFSET3D(u,2,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-2),KRANC_GFOFFSET3D(u,3,0,2)))),ToReal(-5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-4),kadd(KRANC_GFOFFSET3D(u,1,0,4),kadd(KRANC_GFOFFSET3D(u,-4,0,-1),KRANC_GFOFFSET3D(u,4,0,1)))),ToReal(-2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,3),KRANC_GFOFFSET3D(u,3,0,-3)),ToReal(-1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,4),kadd(KRANC_GFOFFSET3D(u,2,0,-4),kadd(KRANC_GFOFFSET3D(u,-4,0,2),KRANC_GFOFFSET3D(u,4,0,-2)))),ToReal(-504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,-4),kadd(KRANC_GFOFFSET3D(u,3,0,4),kadd(KRANC_GFOFFSET3D(u,-4,0,-3),KRANC_GFOFFSET3D(u,4,0,3)))),ToReal(-96),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,4),KRANC_GFOFFSET3D(u,4,0,-4)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,-4),KRANC_GFOFFSET3D(u,4,0,4)),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,4),kadd(KRANC_GFOFFSET3D(u,3,0,-4),kadd(KRANC_GFOFFSET3D(u,-4,0,3),KRANC_GFOFFSET3D(u,4,0,-3)))),ToReal(96),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-4),kadd(KRANC_GFOFFSET3D(u,2,0,4),kadd(KRANC_GFOFFSET3D(u,-4,0,-2),KRANC_GFOFFSET3D(u,4,0,2)))),ToReal(504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,-3),KRANC_GFOFFSET3D(u,3,0,3)),ToReal(1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,4),kadd(KRANC_GFOFFSET3D(u,1,0,-4),kadd(KRANC_GFOFFSET3D(u,-4,0,1),KRANC_GFOFFSET3D(u,4,0,-1)))),ToReal(2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,3),kadd(KRANC_GFOFFSET3D(u,2,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,2),KRANC_GFOFFSET3D(u,3,0,-2)))),ToReal(5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-3),kadd(KRANC_GFOFFSET3D(u,1,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-1),KRANC_GFOFFSET3D(u,3,0,1)))),ToReal(21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-2),KRANC_GFOFFSET3D(u,2,0,2)),ToReal(28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,2),kadd(KRANC_GFOFFSET3D(u,1,0,-2),kadd(KRANC_GFOFFSET3D(u,-2,0,1),KRANC_GFOFFSET3D(u,2,0,-1)))),ToReal(112896),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,-1),KRANC_GFOFFSET3D(u,1,0,1)),ToReal(451584)))))))))))))))))))))))
#else
#  define PDstandardNthfdOrder813(u) (PDstandardNthfdOrder813_impl(u,p1o705600dxdz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder813_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o705600dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder813_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o705600dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o705600dxdz,kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,1),KRANC_GFOFFSET3D(u,1,0,-1)),ToReal(-451584),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-2),kadd(KRANC_GFOFFSET3D(u,1,0,2),kadd(KRANC_GFOFFSET3D(u,-2,0,-1),KRANC_GFOFFSET3D(u,2,0,1)))),ToReal(-112896),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,2),KRANC_GFOFFSET3D(u,2,0,-2)),ToReal(-28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,3),kadd(KRANC_GFOFFSET3D(u,1,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,1),KRANC_GFOFFSET3D(u,3,0,-1)))),ToReal(-21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-3),kadd(KRANC_GFOFFSET3D(u,2,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-2),KRANC_GFOFFSET3D(u,3,0,2)))),ToReal(-5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-4),kadd(KRANC_GFOFFSET3D(u,1,0,4),kadd(KRANC_GFOFFSET3D(u,-4,0,-1),KRANC_GFOFFSET3D(u,4,0,1)))),ToReal(-2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,3),KRANC_GFOFFSET3D(u,3,0,-3)),ToReal(-1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,4),kadd(KRANC_GFOFFSET3D(u,2,0,-4),kadd(KRANC_GFOFFSET3D(u,-4,0,2),KRANC_GFOFFSET3D(u,4,0,-2)))),ToReal(-504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,-4),kadd(KRANC_GFOFFSET3D(u,3,0,4),kadd(KRANC_GFOFFSET3D(u,-4,0,-3),KRANC_GFOFFSET3D(u,4,0,3)))),ToReal(-96),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,4),KRANC_GFOFFSET3D(u,4,0,-4)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,-4),KRANC_GFOFFSET3D(u,4,0,4)),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,4),kadd(KRANC_GFOFFSET3D(u,3,0,-4),kadd(KRANC_GFOFFSET3D(u,-4,0,3),KRANC_GFOFFSET3D(u,4,0,-3)))),ToReal(96),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-4),kadd(KRANC_GFOFFSET3D(u,2,0,4),kadd(KRANC_GFOFFSET3D(u,-4,0,-2),KRANC_GFOFFSET3D(u,4,0,2)))),ToReal(504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,-3),KRANC_GFOFFSET3D(u,3,0,3)),ToReal(1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,4),kadd(KRANC_GFOFFSET3D(u,1,0,-4),kadd(KRANC_GFOFFSET3D(u,-4,0,1),KRANC_GFOFFSET3D(u,4,0,-1)))),ToReal(2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,3),kadd(KRANC_GFOFFSET3D(u,2,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,2),KRANC_GFOFFSET3D(u,3,0,-2)))),ToReal(5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-3),kadd(KRANC_GFOFFSET3D(u,1,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-1),KRANC_GFOFFSET3D(u,3,0,1)))),ToReal(21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-2),KRANC_GFOFFSET3D(u,2,0,2)),ToReal(28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,2),kadd(KRANC_GFOFFSET3D(u,1,0,-2),kadd(KRANC_GFOFFSET3D(u,-2,0,1),KRANC_GFOFFSET3D(u,2,0,-1)))),ToReal(112896),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,-1),KRANC_GFOFFSET3D(u,1,0,1)),ToReal(451584))))))))))))))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder821(u) (kmul(p1o705600dxdy,kmadd(kadd(KRANC_GFOFFSET3D(u,-1,1,0),KRANC_GFOFFSET3D(u,1,-1,0)),ToReal(-451584),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-2,0),kadd(KRANC_GFOFFSET3D(u,1,2,0),kadd(KRANC_GFOFFSET3D(u,-2,-1,0),KRANC_GFOFFSET3D(u,2,1,0)))),ToReal(-112896),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,2,0),KRANC_GFOFFSET3D(u,2,-2,0)),ToReal(-28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,3,0),kadd(KRANC_GFOFFSET3D(u,1,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,1,0),KRANC_GFOFFSET3D(u,3,-1,0)))),ToReal(-21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-3,0),kadd(KRANC_GFOFFSET3D(u,2,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-2,0),KRANC_GFOFFSET3D(u,3,2,0)))),ToReal(-5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-4,0),kadd(KRANC_GFOFFSET3D(u,1,4,0),kadd(KRANC_GFOFFSET3D(u,-4,-1,0),KRANC_GFOFFSET3D(u,4,1,0)))),ToReal(-2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,3,0),KRANC_GFOFFSET3D(u,3,-3,0)),ToReal(-1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,4,0),kadd(KRANC_GFOFFSET3D(u,2,-4,0),kadd(KRANC_GFOFFSET3D(u,-4,2,0),KRANC_GFOFFSET3D(u,4,-2,0)))),ToReal(-504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,-4,0),kadd(KRANC_GFOFFSET3D(u,3,4,0),kadd(KRANC_GFOFFSET3D(u,-4,-3,0),KRANC_GFOFFSET3D(u,4,3,0)))),ToReal(-96),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,4,0),KRANC_GFOFFSET3D(u,4,-4,0)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,-4,0),KRANC_GFOFFSET3D(u,4,4,0)),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,4,0),kadd(KRANC_GFOFFSET3D(u,3,-4,0),kadd(KRANC_GFOFFSET3D(u,-4,3,0),KRANC_GFOFFSET3D(u,4,-3,0)))),ToReal(96),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-4,0),kadd(KRANC_GFOFFSET3D(u,2,4,0),kadd(KRANC_GFOFFSET3D(u,-4,-2,0),KRANC_GFOFFSET3D(u,4,2,0)))),ToReal(504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,-3,0),KRANC_GFOFFSET3D(u,3,3,0)),ToReal(1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,4,0),kadd(KRANC_GFOFFSET3D(u,1,-4,0),kadd(KRANC_GFOFFSET3D(u,-4,1,0),KRANC_GFOFFSET3D(u,4,-1,0)))),ToReal(2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,3,0),kadd(KRANC_GFOFFSET3D(u,2,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,2,0),KRANC_GFOFFSET3D(u,3,-2,0)))),ToReal(5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-3,0),kadd(KRANC_GFOFFSET3D(u,1,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-1,0),KRANC_GFOFFSET3D(u,3,1,0)))),ToReal(21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-2,0),KRANC_GFOFFSET3D(u,2,2,0)),ToReal(28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,2,0),kadd(KRANC_GFOFFSET3D(u,1,-2,0),kadd(KRANC_GFOFFSET3D(u,-2,1,0),KRANC_GFOFFSET3D(u,2,-1,0)))),ToReal(112896),kmul(kadd(KRANC_GFOFFSET3D(u,-1,-1,0),KRANC_GFOFFSET3D(u,1,1,0)),ToReal(451584)))))))))))))))))))))))
#else
#  define PDstandardNthfdOrder821(u) (PDstandardNthfdOrder821_impl(u,p1o705600dxdy,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder821_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o705600dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder821_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o705600dxdy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o705600dxdy,kmadd(kadd(KRANC_GFOFFSET3D(u,-1,1,0),KRANC_GFOFFSET3D(u,1,-1,0)),ToReal(-451584),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-2,0),kadd(KRANC_GFOFFSET3D(u,1,2,0),kadd(KRANC_GFOFFSET3D(u,-2,-1,0),KRANC_GFOFFSET3D(u,2,1,0)))),ToReal(-112896),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,2,0),KRANC_GFOFFSET3D(u,2,-2,0)),ToReal(-28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,3,0),kadd(KRANC_GFOFFSET3D(u,1,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,1,0),KRANC_GFOFFSET3D(u,3,-1,0)))),ToReal(-21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-3,0),kadd(KRANC_GFOFFSET3D(u,2,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-2,0),KRANC_GFOFFSET3D(u,3,2,0)))),ToReal(-5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-4,0),kadd(KRANC_GFOFFSET3D(u,1,4,0),kadd(KRANC_GFOFFSET3D(u,-4,-1,0),KRANC_GFOFFSET3D(u,4,1,0)))),ToReal(-2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,3,0),KRANC_GFOFFSET3D(u,3,-3,0)),ToReal(-1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,4,0),kadd(KRANC_GFOFFSET3D(u,2,-4,0),kadd(KRANC_GFOFFSET3D(u,-4,2,0),KRANC_GFOFFSET3D(u,4,-2,0)))),ToReal(-504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,-4,0),kadd(KRANC_GFOFFSET3D(u,3,4,0),kadd(KRANC_GFOFFSET3D(u,-4,-3,0),KRANC_GFOFFSET3D(u,4,3,0)))),ToReal(-96),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,4,0),KRANC_GFOFFSET3D(u,4,-4,0)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,-4,0),KRANC_GFOFFSET3D(u,4,4,0)),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,4,0),kadd(KRANC_GFOFFSET3D(u,3,-4,0),kadd(KRANC_GFOFFSET3D(u,-4,3,0),KRANC_GFOFFSET3D(u,4,-3,0)))),ToReal(96),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-4,0),kadd(KRANC_GFOFFSET3D(u,2,4,0),kadd(KRANC_GFOFFSET3D(u,-4,-2,0),KRANC_GFOFFSET3D(u,4,2,0)))),ToReal(504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,-3,0),KRANC_GFOFFSET3D(u,3,3,0)),ToReal(1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,4,0),kadd(KRANC_GFOFFSET3D(u,1,-4,0),kadd(KRANC_GFOFFSET3D(u,-4,1,0),KRANC_GFOFFSET3D(u,4,-1,0)))),ToReal(2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,3,0),kadd(KRANC_GFOFFSET3D(u,2,-3,0),kadd(KRANC_GFOFFSET3D(u,-3,2,0),KRANC_GFOFFSET3D(u,3,-2,0)))),ToReal(5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,-3,0),kadd(KRANC_GFOFFSET3D(u,1,3,0),kadd(KRANC_GFOFFSET3D(u,-3,-1,0),KRANC_GFOFFSET3D(u,3,1,0)))),ToReal(21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,-2,0),KRANC_GFOFFSET3D(u,2,2,0)),ToReal(28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,2,0),kadd(KRANC_GFOFFSET3D(u,1,-2,0),kadd(KRANC_GFOFFSET3D(u,-2,1,0),KRANC_GFOFFSET3D(u,2,-1,0)))),ToReal(112896),kmul(kadd(KRANC_GFOFFSET3D(u,-1,-1,0),KRANC_GFOFFSET3D(u,1,1,0)),ToReal(451584))))))))))))))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder823(u) (kmul(p1o705600dydz,kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,1),KRANC_GFOFFSET3D(u,0,1,-1)),ToReal(-451584),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-2),kadd(KRANC_GFOFFSET3D(u,0,1,2),kadd(KRANC_GFOFFSET3D(u,0,-2,-1),KRANC_GFOFFSET3D(u,0,2,1)))),ToReal(-112896),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,2),KRANC_GFOFFSET3D(u,0,2,-2)),ToReal(-28224),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,3),kadd(KRANC_GFOFFSET3D(u,0,1,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,1),KRANC_GFOFFSET3D(u,0,3,-1)))),ToReal(-21504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-3),kadd(KRANC_GFOFFSET3D(u,0,2,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-2),KRANC_GFOFFSET3D(u,0,3,2)))),ToReal(-5376),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-4),kadd(KRANC_GFOFFSET3D(u,0,1,4),kadd(KRANC_GFOFFSET3D(u,0,-4,-1),KRANC_GFOFFSET3D(u,0,4,1)))),ToReal(-2016),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,3),KRANC_GFOFFSET3D(u,0,3,-3)),ToReal(-1024),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,4),kadd(KRANC_GFOFFSET3D(u,0,2,-4),kadd(KRANC_GFOFFSET3D(u,0,-4,2),KRANC_GFOFFSET3D(u,0,4,-2)))),ToReal(-504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,-4),kadd(KRANC_GFOFFSET3D(u,0,3,4),kadd(KRANC_GFOFFSET3D(u,0,-4,-3),KRANC_GFOFFSET3D(u,0,4,3)))),ToReal(-96),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,4),KRANC_GFOFFSET3D(u,0,4,-4)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,-4),KRANC_GFOFFSET3D(u,0,4,4)),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,4),kadd(KRANC_GFOFFSET3D(u,0,3,-4),kadd(KRANC_GFOFFSET3D(u,0,-4,3),KRANC_GFOFFSET3D(u,0,4,-3)))),ToReal(96),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-4),kadd(KRANC_GFOFFSET3D(u,0,2,4),kadd(KRANC_GFOFFSET3D(u,0,-4,-2),KRANC_GFOFFSET3D(u,0,4,2)))),ToReal(504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,-3),KRANC_GFOFFSET3D(u,0,3,3)),ToReal(1024),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,4),kadd(KRANC_GFOFFSET3D(u,0,1,-4),kadd(KRANC_GFOFFSET3D(u,0,-4,1),KRANC_GFOFFSET3D(u,0,4,-1)))),ToReal(2016),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,3),kadd(KRANC_GFOFFSET3D(u,0,2,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,2),KRANC_GFOFFSET3D(u,0,3,-2)))),ToReal(5376),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-3),kadd(KRANC_GFOFFSET3D(u,0,1,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-1),KRANC_GFOFFSET3D(u,0,3,1)))),ToReal(21504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-2),KRANC_GFOFFSET3D(u,0,2,2)),ToReal(28224),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,2),kadd(KRANC_GFOFFSET3D(u,0,1,-2),kadd(KRANC_GFOFFSET3D(u,0,-2,1),KRANC_GFOFFSET3D(u,0,2,-1)))),ToReal(112896),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,-1),KRANC_GFOFFSET3D(u,0,1,1)),ToReal(451584)))))))))))))))))))))))
#else
#  define PDstandardNthfdOrder823(u) (PDstandardNthfdOrder823_impl(u,p1o705600dydz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder823_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o705600dydz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder823_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o705600dydz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o705600dydz,kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,1),KRANC_GFOFFSET3D(u,0,1,-1)),ToReal(-451584),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-2),kadd(KRANC_GFOFFSET3D(u,0,1,2),kadd(KRANC_GFOFFSET3D(u,0,-2,-1),KRANC_GFOFFSET3D(u,0,2,1)))),ToReal(-112896),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,2),KRANC_GFOFFSET3D(u,0,2,-2)),ToReal(-28224),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,3),kadd(KRANC_GFOFFSET3D(u,0,1,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,1),KRANC_GFOFFSET3D(u,0,3,-1)))),ToReal(-21504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-3),kadd(KRANC_GFOFFSET3D(u,0,2,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-2),KRANC_GFOFFSET3D(u,0,3,2)))),ToReal(-5376),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-4),kadd(KRANC_GFOFFSET3D(u,0,1,4),kadd(KRANC_GFOFFSET3D(u,0,-4,-1),KRANC_GFOFFSET3D(u,0,4,1)))),ToReal(-2016),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,3),KRANC_GFOFFSET3D(u,0,3,-3)),ToReal(-1024),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,4),kadd(KRANC_GFOFFSET3D(u,0,2,-4),kadd(KRANC_GFOFFSET3D(u,0,-4,2),KRANC_GFOFFSET3D(u,0,4,-2)))),ToReal(-504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,-4),kadd(KRANC_GFOFFSET3D(u,0,3,4),kadd(KRANC_GFOFFSET3D(u,0,-4,-3),KRANC_GFOFFSET3D(u,0,4,3)))),ToReal(-96),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,4),KRANC_GFOFFSET3D(u,0,4,-4)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,-4),KRANC_GFOFFSET3D(u,0,4,4)),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,4),kadd(KRANC_GFOFFSET3D(u,0,3,-4),kadd(KRANC_GFOFFSET3D(u,0,-4,3),KRANC_GFOFFSET3D(u,0,4,-3)))),ToReal(96),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-4),kadd(KRANC_GFOFFSET3D(u,0,2,4),kadd(KRANC_GFOFFSET3D(u,0,-4,-2),KRANC_GFOFFSET3D(u,0,4,2)))),ToReal(504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,-3),KRANC_GFOFFSET3D(u,0,3,3)),ToReal(1024),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,4),kadd(KRANC_GFOFFSET3D(u,0,1,-4),kadd(KRANC_GFOFFSET3D(u,0,-4,1),KRANC_GFOFFSET3D(u,0,4,-1)))),ToReal(2016),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,3),kadd(KRANC_GFOFFSET3D(u,0,2,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,2),KRANC_GFOFFSET3D(u,0,3,-2)))),ToReal(5376),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-3),kadd(KRANC_GFOFFSET3D(u,0,1,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-1),KRANC_GFOFFSET3D(u,0,3,1)))),ToReal(21504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-2),KRANC_GFOFFSET3D(u,0,2,2)),ToReal(28224),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,2),kadd(KRANC_GFOFFSET3D(u,0,1,-2),kadd(KRANC_GFOFFSET3D(u,0,-2,1),KRANC_GFOFFSET3D(u,0,2,-1)))),ToReal(112896),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,-1),KRANC_GFOFFSET3D(u,0,1,1)),ToReal(451584))))))))))))))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder831(u) (kmul(p1o705600dxdz,kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,1),KRANC_GFOFFSET3D(u,1,0,-1)),ToReal(-451584),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-2),kadd(KRANC_GFOFFSET3D(u,1,0,2),kadd(KRANC_GFOFFSET3D(u,-2,0,-1),KRANC_GFOFFSET3D(u,2,0,1)))),ToReal(-112896),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,2),KRANC_GFOFFSET3D(u,2,0,-2)),ToReal(-28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,3),kadd(KRANC_GFOFFSET3D(u,1,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,1),KRANC_GFOFFSET3D(u,3,0,-1)))),ToReal(-21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-3),kadd(KRANC_GFOFFSET3D(u,2,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-2),KRANC_GFOFFSET3D(u,3,0,2)))),ToReal(-5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-4),kadd(KRANC_GFOFFSET3D(u,1,0,4),kadd(KRANC_GFOFFSET3D(u,-4,0,-1),KRANC_GFOFFSET3D(u,4,0,1)))),ToReal(-2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,3),KRANC_GFOFFSET3D(u,3,0,-3)),ToReal(-1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,4),kadd(KRANC_GFOFFSET3D(u,2,0,-4),kadd(KRANC_GFOFFSET3D(u,-4,0,2),KRANC_GFOFFSET3D(u,4,0,-2)))),ToReal(-504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,-4),kadd(KRANC_GFOFFSET3D(u,3,0,4),kadd(KRANC_GFOFFSET3D(u,-4,0,-3),KRANC_GFOFFSET3D(u,4,0,3)))),ToReal(-96),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,4),KRANC_GFOFFSET3D(u,4,0,-4)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,-4),KRANC_GFOFFSET3D(u,4,0,4)),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,4),kadd(KRANC_GFOFFSET3D(u,3,0,-4),kadd(KRANC_GFOFFSET3D(u,-4,0,3),KRANC_GFOFFSET3D(u,4,0,-3)))),ToReal(96),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-4),kadd(KRANC_GFOFFSET3D(u,2,0,4),kadd(KRANC_GFOFFSET3D(u,-4,0,-2),KRANC_GFOFFSET3D(u,4,0,2)))),ToReal(504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,-3),KRANC_GFOFFSET3D(u,3,0,3)),ToReal(1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,4),kadd(KRANC_GFOFFSET3D(u,1,0,-4),kadd(KRANC_GFOFFSET3D(u,-4,0,1),KRANC_GFOFFSET3D(u,4,0,-1)))),ToReal(2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,3),kadd(KRANC_GFOFFSET3D(u,2,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,2),KRANC_GFOFFSET3D(u,3,0,-2)))),ToReal(5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-3),kadd(KRANC_GFOFFSET3D(u,1,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-1),KRANC_GFOFFSET3D(u,3,0,1)))),ToReal(21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-2),KRANC_GFOFFSET3D(u,2,0,2)),ToReal(28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,2),kadd(KRANC_GFOFFSET3D(u,1,0,-2),kadd(KRANC_GFOFFSET3D(u,-2,0,1),KRANC_GFOFFSET3D(u,2,0,-1)))),ToReal(112896),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,-1),KRANC_GFOFFSET3D(u,1,0,1)),ToReal(451584)))))))))))))))))))))))
#else
#  define PDstandardNthfdOrder831(u) (PDstandardNthfdOrder831_impl(u,p1o705600dxdz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder831_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o705600dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder831_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o705600dxdz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o705600dxdz,kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,1),KRANC_GFOFFSET3D(u,1,0,-1)),ToReal(-451584),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-2),kadd(KRANC_GFOFFSET3D(u,1,0,2),kadd(KRANC_GFOFFSET3D(u,-2,0,-1),KRANC_GFOFFSET3D(u,2,0,1)))),ToReal(-112896),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,2),KRANC_GFOFFSET3D(u,2,0,-2)),ToReal(-28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,3),kadd(KRANC_GFOFFSET3D(u,1,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,1),KRANC_GFOFFSET3D(u,3,0,-1)))),ToReal(-21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-3),kadd(KRANC_GFOFFSET3D(u,2,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-2),KRANC_GFOFFSET3D(u,3,0,2)))),ToReal(-5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-4),kadd(KRANC_GFOFFSET3D(u,1,0,4),kadd(KRANC_GFOFFSET3D(u,-4,0,-1),KRANC_GFOFFSET3D(u,4,0,1)))),ToReal(-2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,3),KRANC_GFOFFSET3D(u,3,0,-3)),ToReal(-1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,4),kadd(KRANC_GFOFFSET3D(u,2,0,-4),kadd(KRANC_GFOFFSET3D(u,-4,0,2),KRANC_GFOFFSET3D(u,4,0,-2)))),ToReal(-504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,-4),kadd(KRANC_GFOFFSET3D(u,3,0,4),kadd(KRANC_GFOFFSET3D(u,-4,0,-3),KRANC_GFOFFSET3D(u,4,0,3)))),ToReal(-96),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,4),KRANC_GFOFFSET3D(u,4,0,-4)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,-4),KRANC_GFOFFSET3D(u,4,0,4)),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,4),kadd(KRANC_GFOFFSET3D(u,3,0,-4),kadd(KRANC_GFOFFSET3D(u,-4,0,3),KRANC_GFOFFSET3D(u,4,0,-3)))),ToReal(96),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-4),kadd(KRANC_GFOFFSET3D(u,2,0,4),kadd(KRANC_GFOFFSET3D(u,-4,0,-2),KRANC_GFOFFSET3D(u,4,0,2)))),ToReal(504),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,-3),KRANC_GFOFFSET3D(u,3,0,3)),ToReal(1024),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,4),kadd(KRANC_GFOFFSET3D(u,1,0,-4),kadd(KRANC_GFOFFSET3D(u,-4,0,1),KRANC_GFOFFSET3D(u,4,0,-1)))),ToReal(2016),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,3),kadd(KRANC_GFOFFSET3D(u,2,0,-3),kadd(KRANC_GFOFFSET3D(u,-3,0,2),KRANC_GFOFFSET3D(u,3,0,-2)))),ToReal(5376),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,-3),kadd(KRANC_GFOFFSET3D(u,1,0,3),kadd(KRANC_GFOFFSET3D(u,-3,0,-1),KRANC_GFOFFSET3D(u,3,0,1)))),ToReal(21504),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,-2),KRANC_GFOFFSET3D(u,2,0,2)),ToReal(28224),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,2),kadd(KRANC_GFOFFSET3D(u,1,0,-2),kadd(KRANC_GFOFFSET3D(u,-2,0,1),KRANC_GFOFFSET3D(u,2,0,-1)))),ToReal(112896),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,-1),KRANC_GFOFFSET3D(u,1,0,1)),ToReal(451584))))))))))))))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDstandardNthfdOrder832(u) (kmul(p1o705600dydz,kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,1),KRANC_GFOFFSET3D(u,0,1,-1)),ToReal(-451584),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-2),kadd(KRANC_GFOFFSET3D(u,0,1,2),kadd(KRANC_GFOFFSET3D(u,0,-2,-1),KRANC_GFOFFSET3D(u,0,2,1)))),ToReal(-112896),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,2),KRANC_GFOFFSET3D(u,0,2,-2)),ToReal(-28224),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,3),kadd(KRANC_GFOFFSET3D(u,0,1,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,1),KRANC_GFOFFSET3D(u,0,3,-1)))),ToReal(-21504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-3),kadd(KRANC_GFOFFSET3D(u,0,2,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-2),KRANC_GFOFFSET3D(u,0,3,2)))),ToReal(-5376),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-4),kadd(KRANC_GFOFFSET3D(u,0,1,4),kadd(KRANC_GFOFFSET3D(u,0,-4,-1),KRANC_GFOFFSET3D(u,0,4,1)))),ToReal(-2016),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,3),KRANC_GFOFFSET3D(u,0,3,-3)),ToReal(-1024),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,4),kadd(KRANC_GFOFFSET3D(u,0,2,-4),kadd(KRANC_GFOFFSET3D(u,0,-4,2),KRANC_GFOFFSET3D(u,0,4,-2)))),ToReal(-504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,-4),kadd(KRANC_GFOFFSET3D(u,0,3,4),kadd(KRANC_GFOFFSET3D(u,0,-4,-3),KRANC_GFOFFSET3D(u,0,4,3)))),ToReal(-96),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,4),KRANC_GFOFFSET3D(u,0,4,-4)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,-4),KRANC_GFOFFSET3D(u,0,4,4)),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,4),kadd(KRANC_GFOFFSET3D(u,0,3,-4),kadd(KRANC_GFOFFSET3D(u,0,-4,3),KRANC_GFOFFSET3D(u,0,4,-3)))),ToReal(96),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-4),kadd(KRANC_GFOFFSET3D(u,0,2,4),kadd(KRANC_GFOFFSET3D(u,0,-4,-2),KRANC_GFOFFSET3D(u,0,4,2)))),ToReal(504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,-3),KRANC_GFOFFSET3D(u,0,3,3)),ToReal(1024),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,4),kadd(KRANC_GFOFFSET3D(u,0,1,-4),kadd(KRANC_GFOFFSET3D(u,0,-4,1),KRANC_GFOFFSET3D(u,0,4,-1)))),ToReal(2016),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,3),kadd(KRANC_GFOFFSET3D(u,0,2,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,2),KRANC_GFOFFSET3D(u,0,3,-2)))),ToReal(5376),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-3),kadd(KRANC_GFOFFSET3D(u,0,1,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-1),KRANC_GFOFFSET3D(u,0,3,1)))),ToReal(21504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-2),KRANC_GFOFFSET3D(u,0,2,2)),ToReal(28224),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,2),kadd(KRANC_GFOFFSET3D(u,0,1,-2),kadd(KRANC_GFOFFSET3D(u,0,-2,1),KRANC_GFOFFSET3D(u,0,2,-1)))),ToReal(112896),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,-1),KRANC_GFOFFSET3D(u,0,1,1)),ToReal(451584)))))))))))))))))))))))
#else
#  define PDstandardNthfdOrder832(u) (PDstandardNthfdOrder832_impl(u,p1o705600dydz,cdj,cdk))
static CCTK_REAL_VEC PDstandardNthfdOrder832_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o705600dydz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDstandardNthfdOrder832_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o705600dydz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o705600dydz,kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,1),KRANC_GFOFFSET3D(u,0,1,-1)),ToReal(-451584),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-2),kadd(KRANC_GFOFFSET3D(u,0,1,2),kadd(KRANC_GFOFFSET3D(u,0,-2,-1),KRANC_GFOFFSET3D(u,0,2,1)))),ToReal(-112896),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,2),KRANC_GFOFFSET3D(u,0,2,-2)),ToReal(-28224),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,3),kadd(KRANC_GFOFFSET3D(u,0,1,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,1),KRANC_GFOFFSET3D(u,0,3,-1)))),ToReal(-21504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-3),kadd(KRANC_GFOFFSET3D(u,0,2,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-2),KRANC_GFOFFSET3D(u,0,3,2)))),ToReal(-5376),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-4),kadd(KRANC_GFOFFSET3D(u,0,1,4),kadd(KRANC_GFOFFSET3D(u,0,-4,-1),KRANC_GFOFFSET3D(u,0,4,1)))),ToReal(-2016),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,3),KRANC_GFOFFSET3D(u,0,3,-3)),ToReal(-1024),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,4),kadd(KRANC_GFOFFSET3D(u,0,2,-4),kadd(KRANC_GFOFFSET3D(u,0,-4,2),KRANC_GFOFFSET3D(u,0,4,-2)))),ToReal(-504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,-4),kadd(KRANC_GFOFFSET3D(u,0,3,4),kadd(KRANC_GFOFFSET3D(u,0,-4,-3),KRANC_GFOFFSET3D(u,0,4,3)))),ToReal(-96),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,4),KRANC_GFOFFSET3D(u,0,4,-4)),ToReal(-9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,-4),KRANC_GFOFFSET3D(u,0,4,4)),ToReal(9),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,4),kadd(KRANC_GFOFFSET3D(u,0,3,-4),kadd(KRANC_GFOFFSET3D(u,0,-4,3),KRANC_GFOFFSET3D(u,0,4,-3)))),ToReal(96),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-4),kadd(KRANC_GFOFFSET3D(u,0,2,4),kadd(KRANC_GFOFFSET3D(u,0,-4,-2),KRANC_GFOFFSET3D(u,0,4,2)))),ToReal(504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,-3),KRANC_GFOFFSET3D(u,0,3,3)),ToReal(1024),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,4),kadd(KRANC_GFOFFSET3D(u,0,1,-4),kadd(KRANC_GFOFFSET3D(u,0,-4,1),KRANC_GFOFFSET3D(u,0,4,-1)))),ToReal(2016),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,3),kadd(KRANC_GFOFFSET3D(u,0,2,-3),kadd(KRANC_GFOFFSET3D(u,0,-3,2),KRANC_GFOFFSET3D(u,0,3,-2)))),ToReal(5376),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,-3),kadd(KRANC_GFOFFSET3D(u,0,1,3),kadd(KRANC_GFOFFSET3D(u,0,-3,-1),KRANC_GFOFFSET3D(u,0,3,1)))),ToReal(21504),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,-2),KRANC_GFOFFSET3D(u,0,2,2)),ToReal(28224),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,2),kadd(KRANC_GFOFFSET3D(u,0,1,-2),kadd(KRANC_GFOFFSET3D(u,0,-2,1),KRANC_GFOFFSET3D(u,0,2,-1)))),ToReal(112896),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,-1),KRANC_GFOFFSET3D(u,0,1,1)),ToReal(451584))))))))))))))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDdissipationNthfdOrder21(u) (kmul(p1o16dx,kadd(KRANC_GFOFFSET3D(u,-2,0,0),kadd(KRANC_GFOFFSET3D(u,2,0,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(6)))))))
#else
#  define PDdissipationNthfdOrder21(u) (PDdissipationNthfdOrder21_impl(u,p1o16dx,cdj,cdk))
static CCTK_REAL_VEC PDdissipationNthfdOrder21_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o16dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDdissipationNthfdOrder21_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o16dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o16dx,kadd(KRANC_GFOFFSET3D(u,-2,0,0),kadd(KRANC_GFOFFSET3D(u,2,0,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(6))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDdissipationNthfdOrder22(u) (kmul(p1o16dy,kadd(KRANC_GFOFFSET3D(u,0,-2,0),kadd(KRANC_GFOFFSET3D(u,0,2,0),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(6)))))))
#else
#  define PDdissipationNthfdOrder22(u) (PDdissipationNthfdOrder22_impl(u,p1o16dy,cdj,cdk))
static CCTK_REAL_VEC PDdissipationNthfdOrder22_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o16dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDdissipationNthfdOrder22_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o16dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o16dy,kadd(KRANC_GFOFFSET3D(u,0,-2,0),kadd(KRANC_GFOFFSET3D(u,0,2,0),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(6))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDdissipationNthfdOrder23(u) (kmul(p1o16dz,kadd(KRANC_GFOFFSET3D(u,0,0,-2),kadd(KRANC_GFOFFSET3D(u,0,0,2),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(6)))))))
#else
#  define PDdissipationNthfdOrder23(u) (PDdissipationNthfdOrder23_impl(u,p1o16dz,cdj,cdk))
static CCTK_REAL_VEC PDdissipationNthfdOrder23_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o16dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDdissipationNthfdOrder23_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o16dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o16dz,kadd(KRANC_GFOFFSET3D(u,0,0,-2),kadd(KRANC_GFOFFSET3D(u,0,0,2),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(6))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDdissipationNthfdOrder41(u) (kmul(p1o64dx,kadd(KRANC_GFOFFSET3D(u,-3,0,0),kadd(KRANC_GFOFFSET3D(u,3,0,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-20),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(-6),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(15))))))))
#else
#  define PDdissipationNthfdOrder41(u) (PDdissipationNthfdOrder41_impl(u,p1o64dx,cdj,cdk))
static CCTK_REAL_VEC PDdissipationNthfdOrder41_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o64dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDdissipationNthfdOrder41_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o64dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o64dx,kadd(KRANC_GFOFFSET3D(u,-3,0,0),kadd(KRANC_GFOFFSET3D(u,3,0,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-20),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(-6),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(15)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDdissipationNthfdOrder42(u) (kmul(p1o64dy,kadd(KRANC_GFOFFSET3D(u,0,-3,0),kadd(KRANC_GFOFFSET3D(u,0,3,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-20),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(-6),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(15))))))))
#else
#  define PDdissipationNthfdOrder42(u) (PDdissipationNthfdOrder42_impl(u,p1o64dy,cdj,cdk))
static CCTK_REAL_VEC PDdissipationNthfdOrder42_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o64dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDdissipationNthfdOrder42_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o64dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o64dy,kadd(KRANC_GFOFFSET3D(u,0,-3,0),kadd(KRANC_GFOFFSET3D(u,0,3,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-20),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(-6),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(15)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDdissipationNthfdOrder43(u) (kmul(p1o64dz,kadd(KRANC_GFOFFSET3D(u,0,0,-3),kadd(KRANC_GFOFFSET3D(u,0,0,3),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-20),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(-6),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(15))))))))
#else
#  define PDdissipationNthfdOrder43(u) (PDdissipationNthfdOrder43_impl(u,p1o64dz,cdj,cdk))
static CCTK_REAL_VEC PDdissipationNthfdOrder43_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o64dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDdissipationNthfdOrder43_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o64dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o64dz,kadd(KRANC_GFOFFSET3D(u,0,0,-3),kadd(KRANC_GFOFFSET3D(u,0,0,3),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-20),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(-6),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(15)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDdissipationNthfdOrder61(u) (kmul(p1o256dx,kadd(KRANC_GFOFFSET3D(u,-4,0,0),kadd(KRANC_GFOFFSET3D(u,4,0,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(-56),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,0),KRANC_GFOFFSET3D(u,3,0,0)),ToReal(-8),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(28),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(70)))))))))
#else
#  define PDdissipationNthfdOrder61(u) (PDdissipationNthfdOrder61_impl(u,p1o256dx,cdj,cdk))
static CCTK_REAL_VEC PDdissipationNthfdOrder61_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o256dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDdissipationNthfdOrder61_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o256dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o256dx,kadd(KRANC_GFOFFSET3D(u,-4,0,0),kadd(KRANC_GFOFFSET3D(u,4,0,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(-56),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,0),KRANC_GFOFFSET3D(u,3,0,0)),ToReal(-8),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(28),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(70))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDdissipationNthfdOrder62(u) (kmul(p1o256dy,kadd(KRANC_GFOFFSET3D(u,0,-4,0),kadd(KRANC_GFOFFSET3D(u,0,4,0),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(-56),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,0),KRANC_GFOFFSET3D(u,0,3,0)),ToReal(-8),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(28),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(70)))))))))
#else
#  define PDdissipationNthfdOrder62(u) (PDdissipationNthfdOrder62_impl(u,p1o256dy,cdj,cdk))
static CCTK_REAL_VEC PDdissipationNthfdOrder62_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o256dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDdissipationNthfdOrder62_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o256dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o256dy,kadd(KRANC_GFOFFSET3D(u,0,-4,0),kadd(KRANC_GFOFFSET3D(u,0,4,0),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(-56),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,0),KRANC_GFOFFSET3D(u,0,3,0)),ToReal(-8),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(28),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(70))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDdissipationNthfdOrder63(u) (kmul(p1o256dz,kadd(KRANC_GFOFFSET3D(u,0,0,-4),kadd(KRANC_GFOFFSET3D(u,0,0,4),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(-56),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-3),KRANC_GFOFFSET3D(u,0,0,3)),ToReal(-8),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(28),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(70)))))))))
#else
#  define PDdissipationNthfdOrder63(u) (PDdissipationNthfdOrder63_impl(u,p1o256dz,cdj,cdk))
static CCTK_REAL_VEC PDdissipationNthfdOrder63_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o256dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDdissipationNthfdOrder63_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o256dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o256dz,kadd(KRANC_GFOFFSET3D(u,0,0,-4),kadd(KRANC_GFOFFSET3D(u,0,0,4),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(-56),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-3),KRANC_GFOFFSET3D(u,0,0,3)),ToReal(-8),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(28),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(70))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDdissipationNthfdOrder81(u) (kmul(p1o1024dx,kadd(KRANC_GFOFFSET3D(u,-5,0,0),kadd(KRANC_GFOFFSET3D(u,5,0,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-252),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(-120),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,0),KRANC_GFOFFSET3D(u,4,0,0)),ToReal(-10),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,0),KRANC_GFOFFSET3D(u,3,0,0)),ToReal(45),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(210))))))))))
#else
#  define PDdissipationNthfdOrder81(u) (PDdissipationNthfdOrder81_impl(u,p1o1024dx,cdj,cdk))
static CCTK_REAL_VEC PDdissipationNthfdOrder81_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o1024dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDdissipationNthfdOrder81_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o1024dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o1024dx,kadd(KRANC_GFOFFSET3D(u,-5,0,0),kadd(KRANC_GFOFFSET3D(u,5,0,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-252),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(-120),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,0),KRANC_GFOFFSET3D(u,4,0,0)),ToReal(-10),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,0),KRANC_GFOFFSET3D(u,3,0,0)),ToReal(45),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(210)))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDdissipationNthfdOrder82(u) (kmul(p1o1024dy,kadd(KRANC_GFOFFSET3D(u,0,-5,0),kadd(KRANC_GFOFFSET3D(u,0,5,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-252),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(-120),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,0),KRANC_GFOFFSET3D(u,0,4,0)),ToReal(-10),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,0),KRANC_GFOFFSET3D(u,0,3,0)),ToReal(45),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(210))))))))))
#else
#  define PDdissipationNthfdOrder82(u) (PDdissipationNthfdOrder82_impl(u,p1o1024dy,cdj,cdk))
static CCTK_REAL_VEC PDdissipationNthfdOrder82_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o1024dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDdissipationNthfdOrder82_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o1024dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o1024dy,kadd(KRANC_GFOFFSET3D(u,0,-5,0),kadd(KRANC_GFOFFSET3D(u,0,5,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-252),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(-120),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,0),KRANC_GFOFFSET3D(u,0,4,0)),ToReal(-10),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,0),KRANC_GFOFFSET3D(u,0,3,0)),ToReal(45),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(210)))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDdissipationNthfdOrder83(u) (kmul(p1o1024dz,kadd(KRANC_GFOFFSET3D(u,0,0,-5),kadd(KRANC_GFOFFSET3D(u,0,0,5),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-252),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(-120),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-4),KRANC_GFOFFSET3D(u,0,0,4)),ToReal(-10),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-3),KRANC_GFOFFSET3D(u,0,0,3)),ToReal(45),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(210))))))))))
#else
#  define PDdissipationNthfdOrder83(u) (PDdissipationNthfdOrder83_impl(u,p1o1024dz,cdj,cdk))
static CCTK_REAL_VEC PDdissipationNthfdOrder83_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o1024dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDdissipationNthfdOrder83_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o1024dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o1024dz,kadd(KRANC_GFOFFSET3D(u,0,0,-5),kadd(KRANC_GFOFFSET3D(u,0,0,5),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-252),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(-120),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-4),KRANC_GFOFFSET3D(u,0,0,4)),ToReal(-10),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-3),KRANC_GFOFFSET3D(u,0,0,3)),ToReal(45),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(210)))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthfdOrder21(u) (kmul(pm1o2dx,kmul(dir1,kadd(KRANC_GFOFFSET3D(u,2,0,0),kmadd(KRANC_GFOFFSET3D(u,1,0,0),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(3)))))))
#else
#  define PDupwindNthfdOrder21(u) (PDupwindNthfdOrder21_impl(u,pm1o2dx,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthfdOrder21_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o2dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthfdOrder21_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o2dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthfdOrder41(u) (kmul(p1o12dx,kmul(dir1,kadd(KRANC_GFOFFSET3D(u,3,0,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-10),kmadd(KRANC_GFOFFSET3D(u,2,0,0),ToReal(-6),kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-3),kmul(KRANC_GFOFFSET3D(u,1,0,0),ToReal(18)))))))))
#else
#  define PDupwindNthfdOrder41(u) (PDupwindNthfdOrder41_impl(u,p1o12dx,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthfdOrder41_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o12dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthfdOrder41_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o12dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthfdOrder61(u) (kmul(pm1o60dx,kmul(dir1,kadd(KRANC_GFOFFSET3D(u,4,0,0),kmadd(KRANC_GFOFFSET3D(u,1,0,0),ToReal(-80),kmadd(KRANC_GFOFFSET3D(u,3,0,0),ToReal(-8),kmadd(KRANC_GFOFFSET3D(u,-2,0,0),ToReal(-2),kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(24),kmadd(KRANC_GFOFFSET3D(u,2,0,0),ToReal(30),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(35)))))))))))
#else
#  define PDupwindNthfdOrder61(u) (PDupwindNthfdOrder61_impl(u,pm1o60dx,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthfdOrder61_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o60dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthfdOrder61_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o60dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthfdOrder81(u) (kmul(p1o840dx,kmul(dir1,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-378),kmadd(KRANC_GFOFFSET3D(u,5,0,0),ToReal(3),kmul(ToReal(-5),kadd(KRANC_GFOFFSET3D(u,-3,0,0),kmadd(KRANC_GFOFFSET3D(u,1,0,0),ToReal(-210),kmadd(KRANC_GFOFFSET3D(u,3,0,0),ToReal(-28),kmadd(KRANC_GFOFFSET3D(u,-2,0,0),ToReal(-12),kmadd(KRANC_GFOFFSET3D(u,4,0,0),ToReal(6),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(84)))))))))))))
#else
#  define PDupwindNthfdOrder81(u) (PDupwindNthfdOrder81_impl(u,p1o840dx,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthfdOrder81_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o840dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthfdOrder81_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o840dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthAntifdOrder21(u) (kmul(p1o4dx,kadd(KRANC_GFOFFSET3D(u,-2,0,0),kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-4),kmsub(KRANC_GFOFFSET3D(u,1,0,0),ToReal(4),KRANC_GFOFFSET3D(u,2,0,0))))))
#else
#  define PDupwindNthAntifdOrder21(u) (PDupwindNthAntifdOrder21_impl(u,p1o4dx,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthAntifdOrder21_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthAntifdOrder21_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o4dx,kadd(KRANC_GFOFFSET3D(u,-2,0,0),kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-4),kmsub(KRANC_GFOFFSET3D(u,1,0,0),ToReal(4),KRANC_GFOFFSET3D(u,2,0,0)))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthAntifdOrder41(u) (kmul(p1o24dx,kadd(KRANC_GFOFFSET3D(u,3,0,0),kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-21),kmadd(KRANC_GFOFFSET3D(u,2,0,0),ToReal(-6),ksub(kmadd(KRANC_GFOFFSET3D(u,-2,0,0),ToReal(6),kmul(KRANC_GFOFFSET3D(u,1,0,0),ToReal(21))),KRANC_GFOFFSET3D(u,-3,0,0)))))))
#else
#  define PDupwindNthAntifdOrder41(u) (PDupwindNthAntifdOrder41_impl(u,p1o24dx,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthAntifdOrder41_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o24dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthAntifdOrder41_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o24dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o24dx,kadd(KRANC_GFOFFSET3D(u,3,0,0),kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-21),kmadd(KRANC_GFOFFSET3D(u,2,0,0),ToReal(-6),ksub(kmadd(KRANC_GFOFFSET3D(u,-2,0,0),ToReal(6),kmul(KRANC_GFOFFSET3D(u,1,0,0),ToReal(21))),KRANC_GFOFFSET3D(u,-3,0,0))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthAntifdOrder61(u) (kmul(p1o120dx,kadd(KRANC_GFOFFSET3D(u,-4,0,0),kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-104),kmadd(KRANC_GFOFFSET3D(u,2,0,0),ToReal(-32),kmadd(KRANC_GFOFFSET3D(u,-3,0,0),ToReal(-8),ksub(kmadd(KRANC_GFOFFSET3D(u,3,0,0),ToReal(8),kmadd(KRANC_GFOFFSET3D(u,-2,0,0),ToReal(32),kmul(KRANC_GFOFFSET3D(u,1,0,0),ToReal(104)))),KRANC_GFOFFSET3D(u,4,0,0))))))))
#else
#  define PDupwindNthAntifdOrder61(u) (PDupwindNthAntifdOrder61_impl(u,p1o120dx,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthAntifdOrder61_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o120dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthAntifdOrder61_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o120dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o120dx,kadd(KRANC_GFOFFSET3D(u,-4,0,0),kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-104),kmadd(KRANC_GFOFFSET3D(u,2,0,0),ToReal(-32),kmadd(KRANC_GFOFFSET3D(u,-3,0,0),ToReal(-8),ksub(kmadd(KRANC_GFOFFSET3D(u,3,0,0),ToReal(8),kmadd(KRANC_GFOFFSET3D(u,-2,0,0),ToReal(32),kmul(KRANC_GFOFFSET3D(u,1,0,0),ToReal(104)))),KRANC_GFOFFSET3D(u,4,0,0)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthAntifdOrder81(u) (kmul(p1o1680dx,kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-1470),kmadd(KRANC_GFOFFSET3D(u,2,0,0),ToReal(-480),kmadd(KRANC_GFOFFSET3D(u,-3,0,0),ToReal(-145),kmadd(KRANC_GFOFFSET3D(u,4,0,0),ToReal(-30),kmadd(KRANC_GFOFFSET3D(u,-5,0,0),ToReal(-3),kmadd(KRANC_GFOFFSET3D(u,5,0,0),ToReal(3),kmadd(KRANC_GFOFFSET3D(u,-4,0,0),ToReal(30),kmadd(KRANC_GFOFFSET3D(u,3,0,0),ToReal(145),kmadd(KRANC_GFOFFSET3D(u,-2,0,0),ToReal(480),kmul(KRANC_GFOFFSET3D(u,1,0,0),ToReal(1470)))))))))))))
#else
#  define PDupwindNthAntifdOrder81(u) (PDupwindNthAntifdOrder81_impl(u,p1o1680dx,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthAntifdOrder81_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o1680dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthAntifdOrder81_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o1680dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o1680dx,kmadd(KRANC_GFOFFSET3D(u,-1,0,0),ToReal(-1470),kmadd(KRANC_GFOFFSET3D(u,2,0,0),ToReal(-480),kmadd(KRANC_GFOFFSET3D(u,-3,0,0),ToReal(-145),kmadd(KRANC_GFOFFSET3D(u,4,0,0),ToReal(-30),kmadd(KRANC_GFOFFSET3D(u,-5,0,0),ToReal(-3),kmadd(KRANC_GFOFFSET3D(u,5,0,0),ToReal(3),kmadd(KRANC_GFOFFSET3D(u,-4,0,0),ToReal(30),kmadd(KRANC_GFOFFSET3D(u,3,0,0),ToReal(145),kmadd(KRANC_GFOFFSET3D(u,-2,0,0),ToReal(480),kmul(KRANC_GFOFFSET3D(u,1,0,0),ToReal(1470))))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthSymmfdOrder21(u) (kmul(pm1o4dx,kadd(KRANC_GFOFFSET3D(u,-2,0,0),kadd(KRANC_GFOFFSET3D(u,2,0,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(6)))))))
#else
#  define PDupwindNthSymmfdOrder21(u) (PDupwindNthSymmfdOrder21_impl(u,pm1o4dx,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthSymmfdOrder21_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o4dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthSymmfdOrder21_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o4dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(pm1o4dx,kadd(KRANC_GFOFFSET3D(u,-2,0,0),kadd(KRANC_GFOFFSET3D(u,2,0,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(6))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthSymmfdOrder41(u) (kmul(p1o24dx,kadd(KRANC_GFOFFSET3D(u,-3,0,0),kadd(KRANC_GFOFFSET3D(u,3,0,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-20),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(-6),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(15))))))))
#else
#  define PDupwindNthSymmfdOrder41(u) (PDupwindNthSymmfdOrder41_impl(u,p1o24dx,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthSymmfdOrder41_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o24dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthSymmfdOrder41_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o24dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o24dx,kadd(KRANC_GFOFFSET3D(u,-3,0,0),kadd(KRANC_GFOFFSET3D(u,3,0,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-20),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(-6),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(15)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthSymmfdOrder61(u) (kmul(pm1o120dx,kadd(KRANC_GFOFFSET3D(u,-4,0,0),kadd(KRANC_GFOFFSET3D(u,4,0,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(-56),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,0),KRANC_GFOFFSET3D(u,3,0,0)),ToReal(-8),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(28),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(70)))))))))
#else
#  define PDupwindNthSymmfdOrder61(u) (PDupwindNthSymmfdOrder61_impl(u,pm1o120dx,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthSymmfdOrder61_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o120dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthSymmfdOrder61_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o120dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(pm1o120dx,kadd(KRANC_GFOFFSET3D(u,-4,0,0),kadd(KRANC_GFOFFSET3D(u,4,0,0),kmadd(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(-56),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,0),KRANC_GFOFFSET3D(u,3,0,0)),ToReal(-8),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(28),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(70))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthSymmfdOrder81(u) (kmul(p1o560dx,kadd(KRANC_GFOFFSET3D(u,-5,0,0),kadd(KRANC_GFOFFSET3D(u,5,0,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-252),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(-120),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,0),KRANC_GFOFFSET3D(u,4,0,0)),ToReal(-10),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,0),KRANC_GFOFFSET3D(u,3,0,0)),ToReal(45),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(210))))))))))
#else
#  define PDupwindNthSymmfdOrder81(u) (PDupwindNthSymmfdOrder81_impl(u,p1o560dx,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthSymmfdOrder81_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o560dx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthSymmfdOrder81_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o560dx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o560dx,kadd(KRANC_GFOFFSET3D(u,-5,0,0),kadd(KRANC_GFOFFSET3D(u,5,0,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-252),kmadd(kadd(KRANC_GFOFFSET3D(u,-2,0,0),KRANC_GFOFFSET3D(u,2,0,0)),ToReal(-120),kmadd(kadd(KRANC_GFOFFSET3D(u,-4,0,0),KRANC_GFOFFSET3D(u,4,0,0)),ToReal(-10),kmadd(kadd(KRANC_GFOFFSET3D(u,-3,0,0),KRANC_GFOFFSET3D(u,3,0,0)),ToReal(45),kmul(kadd(KRANC_GFOFFSET3D(u,-1,0,0),KRANC_GFOFFSET3D(u,1,0,0)),ToReal(210)))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDonesided1(u) (kmul(p1odx,kmul(dir1,ksub(KRANC_GFOFFSET3D(u,1,0,0),KRANC_GFOFFSET3D(u,0,0,0)))))
#else
#  define PDonesided1(u) (PDonesided1_impl(u,p1odx,cdj,cdk))
static CCTK_REAL_VEC PDonesided1_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1odx, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDonesided1_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1odx, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthfdOrder22(u) (kmul(pm1o2dy,kmul(dir2,kadd(KRANC_GFOFFSET3D(u,0,2,0),kmadd(KRANC_GFOFFSET3D(u,0,1,0),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(3)))))))
#else
#  define PDupwindNthfdOrder22(u) (PDupwindNthfdOrder22_impl(u,pm1o2dy,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthfdOrder22_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o2dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthfdOrder22_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o2dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthfdOrder42(u) (kmul(p1o12dy,kmul(dir2,kadd(KRANC_GFOFFSET3D(u,0,3,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-10),kmadd(KRANC_GFOFFSET3D(u,0,2,0),ToReal(-6),kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-3),kmul(KRANC_GFOFFSET3D(u,0,1,0),ToReal(18)))))))))
#else
#  define PDupwindNthfdOrder42(u) (PDupwindNthfdOrder42_impl(u,p1o12dy,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthfdOrder42_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o12dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthfdOrder42_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o12dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthfdOrder62(u) (kmul(pm1o60dy,kmul(dir2,kadd(KRANC_GFOFFSET3D(u,0,4,0),kmadd(KRANC_GFOFFSET3D(u,0,1,0),ToReal(-80),kmadd(KRANC_GFOFFSET3D(u,0,3,0),ToReal(-8),kmadd(KRANC_GFOFFSET3D(u,0,-2,0),ToReal(-2),kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(24),kmadd(KRANC_GFOFFSET3D(u,0,2,0),ToReal(30),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(35)))))))))))
#else
#  define PDupwindNthfdOrder62(u) (PDupwindNthfdOrder62_impl(u,pm1o60dy,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthfdOrder62_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o60dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthfdOrder62_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o60dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthfdOrder82(u) (kmul(p1o840dy,kmul(dir2,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-378),kmadd(KRANC_GFOFFSET3D(u,0,5,0),ToReal(3),kmul(ToReal(-5),kadd(KRANC_GFOFFSET3D(u,0,-3,0),kmadd(KRANC_GFOFFSET3D(u,0,1,0),ToReal(-210),kmadd(KRANC_GFOFFSET3D(u,0,3,0),ToReal(-28),kmadd(KRANC_GFOFFSET3D(u,0,-2,0),ToReal(-12),kmadd(KRANC_GFOFFSET3D(u,0,4,0),ToReal(6),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(84)))))))))))))
#else
#  define PDupwindNthfdOrder82(u) (PDupwindNthfdOrder82_impl(u,p1o840dy,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthfdOrder82_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o840dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthfdOrder82_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o840dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthAntifdOrder22(u) (kmul(p1o4dy,kadd(KRANC_GFOFFSET3D(u,0,-2,0),kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-4),kmsub(KRANC_GFOFFSET3D(u,0,1,0),ToReal(4),KRANC_GFOFFSET3D(u,0,2,0))))))
#else
#  define PDupwindNthAntifdOrder22(u) (PDupwindNthAntifdOrder22_impl(u,p1o4dy,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthAntifdOrder22_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthAntifdOrder22_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o4dy,kadd(KRANC_GFOFFSET3D(u,0,-2,0),kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-4),kmsub(KRANC_GFOFFSET3D(u,0,1,0),ToReal(4),KRANC_GFOFFSET3D(u,0,2,0)))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthAntifdOrder42(u) (kmul(p1o24dy,kadd(KRANC_GFOFFSET3D(u,0,3,0),kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-21),kmadd(KRANC_GFOFFSET3D(u,0,2,0),ToReal(-6),ksub(kmadd(KRANC_GFOFFSET3D(u,0,-2,0),ToReal(6),kmul(KRANC_GFOFFSET3D(u,0,1,0),ToReal(21))),KRANC_GFOFFSET3D(u,0,-3,0)))))))
#else
#  define PDupwindNthAntifdOrder42(u) (PDupwindNthAntifdOrder42_impl(u,p1o24dy,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthAntifdOrder42_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o24dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthAntifdOrder42_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o24dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o24dy,kadd(KRANC_GFOFFSET3D(u,0,3,0),kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-21),kmadd(KRANC_GFOFFSET3D(u,0,2,0),ToReal(-6),ksub(kmadd(KRANC_GFOFFSET3D(u,0,-2,0),ToReal(6),kmul(KRANC_GFOFFSET3D(u,0,1,0),ToReal(21))),KRANC_GFOFFSET3D(u,0,-3,0))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthAntifdOrder62(u) (kmul(p1o120dy,kadd(KRANC_GFOFFSET3D(u,0,-4,0),kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-104),kmadd(KRANC_GFOFFSET3D(u,0,2,0),ToReal(-32),kmadd(KRANC_GFOFFSET3D(u,0,-3,0),ToReal(-8),ksub(kmadd(KRANC_GFOFFSET3D(u,0,3,0),ToReal(8),kmadd(KRANC_GFOFFSET3D(u,0,-2,0),ToReal(32),kmul(KRANC_GFOFFSET3D(u,0,1,0),ToReal(104)))),KRANC_GFOFFSET3D(u,0,4,0))))))))
#else
#  define PDupwindNthAntifdOrder62(u) (PDupwindNthAntifdOrder62_impl(u,p1o120dy,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthAntifdOrder62_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o120dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthAntifdOrder62_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o120dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o120dy,kadd(KRANC_GFOFFSET3D(u,0,-4,0),kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-104),kmadd(KRANC_GFOFFSET3D(u,0,2,0),ToReal(-32),kmadd(KRANC_GFOFFSET3D(u,0,-3,0),ToReal(-8),ksub(kmadd(KRANC_GFOFFSET3D(u,0,3,0),ToReal(8),kmadd(KRANC_GFOFFSET3D(u,0,-2,0),ToReal(32),kmul(KRANC_GFOFFSET3D(u,0,1,0),ToReal(104)))),KRANC_GFOFFSET3D(u,0,4,0)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthAntifdOrder82(u) (kmul(p1o1680dy,kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-1470),kmadd(KRANC_GFOFFSET3D(u,0,2,0),ToReal(-480),kmadd(KRANC_GFOFFSET3D(u,0,-3,0),ToReal(-145),kmadd(KRANC_GFOFFSET3D(u,0,4,0),ToReal(-30),kmadd(KRANC_GFOFFSET3D(u,0,-5,0),ToReal(-3),kmadd(KRANC_GFOFFSET3D(u,0,5,0),ToReal(3),kmadd(KRANC_GFOFFSET3D(u,0,-4,0),ToReal(30),kmadd(KRANC_GFOFFSET3D(u,0,3,0),ToReal(145),kmadd(KRANC_GFOFFSET3D(u,0,-2,0),ToReal(480),kmul(KRANC_GFOFFSET3D(u,0,1,0),ToReal(1470)))))))))))))
#else
#  define PDupwindNthAntifdOrder82(u) (PDupwindNthAntifdOrder82_impl(u,p1o1680dy,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthAntifdOrder82_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o1680dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthAntifdOrder82_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o1680dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o1680dy,kmadd(KRANC_GFOFFSET3D(u,0,-1,0),ToReal(-1470),kmadd(KRANC_GFOFFSET3D(u,0,2,0),ToReal(-480),kmadd(KRANC_GFOFFSET3D(u,0,-3,0),ToReal(-145),kmadd(KRANC_GFOFFSET3D(u,0,4,0),ToReal(-30),kmadd(KRANC_GFOFFSET3D(u,0,-5,0),ToReal(-3),kmadd(KRANC_GFOFFSET3D(u,0,5,0),ToReal(3),kmadd(KRANC_GFOFFSET3D(u,0,-4,0),ToReal(30),kmadd(KRANC_GFOFFSET3D(u,0,3,0),ToReal(145),kmadd(KRANC_GFOFFSET3D(u,0,-2,0),ToReal(480),kmul(KRANC_GFOFFSET3D(u,0,1,0),ToReal(1470))))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthSymmfdOrder22(u) (kmul(pm1o4dy,kadd(KRANC_GFOFFSET3D(u,0,-2,0),kadd(KRANC_GFOFFSET3D(u,0,2,0),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(6)))))))
#else
#  define PDupwindNthSymmfdOrder22(u) (PDupwindNthSymmfdOrder22_impl(u,pm1o4dy,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthSymmfdOrder22_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o4dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthSymmfdOrder22_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o4dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(pm1o4dy,kadd(KRANC_GFOFFSET3D(u,0,-2,0),kadd(KRANC_GFOFFSET3D(u,0,2,0),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(6))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthSymmfdOrder42(u) (kmul(p1o24dy,kadd(KRANC_GFOFFSET3D(u,0,-3,0),kadd(KRANC_GFOFFSET3D(u,0,3,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-20),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(-6),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(15))))))))
#else
#  define PDupwindNthSymmfdOrder42(u) (PDupwindNthSymmfdOrder42_impl(u,p1o24dy,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthSymmfdOrder42_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o24dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthSymmfdOrder42_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o24dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o24dy,kadd(KRANC_GFOFFSET3D(u,0,-3,0),kadd(KRANC_GFOFFSET3D(u,0,3,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-20),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(-6),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(15)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthSymmfdOrder62(u) (kmul(pm1o120dy,kadd(KRANC_GFOFFSET3D(u,0,-4,0),kadd(KRANC_GFOFFSET3D(u,0,4,0),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(-56),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,0),KRANC_GFOFFSET3D(u,0,3,0)),ToReal(-8),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(28),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(70)))))))))
#else
#  define PDupwindNthSymmfdOrder62(u) (PDupwindNthSymmfdOrder62_impl(u,pm1o120dy,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthSymmfdOrder62_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o120dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthSymmfdOrder62_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o120dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(pm1o120dy,kadd(KRANC_GFOFFSET3D(u,0,-4,0),kadd(KRANC_GFOFFSET3D(u,0,4,0),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(-56),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,0),KRANC_GFOFFSET3D(u,0,3,0)),ToReal(-8),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(28),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(70))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthSymmfdOrder82(u) (kmul(p1o560dy,kadd(KRANC_GFOFFSET3D(u,0,-5,0),kadd(KRANC_GFOFFSET3D(u,0,5,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-252),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(-120),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,0),KRANC_GFOFFSET3D(u,0,4,0)),ToReal(-10),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,0),KRANC_GFOFFSET3D(u,0,3,0)),ToReal(45),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(210))))))))))
#else
#  define PDupwindNthSymmfdOrder82(u) (PDupwindNthSymmfdOrder82_impl(u,p1o560dy,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthSymmfdOrder82_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o560dy, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthSymmfdOrder82_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o560dy, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o560dy,kadd(KRANC_GFOFFSET3D(u,0,-5,0),kadd(KRANC_GFOFFSET3D(u,0,5,0),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-252),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-2,0),KRANC_GFOFFSET3D(u,0,2,0)),ToReal(-120),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-4,0),KRANC_GFOFFSET3D(u,0,4,0)),ToReal(-10),kmadd(kadd(KRANC_GFOFFSET3D(u,0,-3,0),KRANC_GFOFFSET3D(u,0,3,0)),ToReal(45),kmul(kadd(KRANC_GFOFFSET3D(u,0,-1,0),KRANC_GFOFFSET3D(u,0,1,0)),ToReal(210)))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDonesided2(u) (kmul(p1ody,kmul(dir2,ksub(KRANC_GFOFFSET3D(u,0,1,0),KRANC_GFOFFSET3D(u,0,0,0)))))
#else
#  define PDonesided2(u) (PDonesided2_impl(u,p1ody,cdj,cdk))
static CCTK_REAL_VEC PDonesided2_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1ody, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDonesided2_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1ody, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthfdOrder23(u) (kmul(pm1o2dz,kmul(dir3,kadd(KRANC_GFOFFSET3D(u,0,0,2),kmadd(KRANC_GFOFFSET3D(u,0,0,1),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(3)))))))
#else
#  define PDupwindNthfdOrder23(u) (PDupwindNthfdOrder23_impl(u,pm1o2dz,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthfdOrder23_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o2dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthfdOrder23_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o2dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthfdOrder43(u) (kmul(p1o12dz,kmul(dir3,kadd(KRANC_GFOFFSET3D(u,0,0,3),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-10),kmadd(KRANC_GFOFFSET3D(u,0,0,2),ToReal(-6),kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-3),kmul(KRANC_GFOFFSET3D(u,0,0,1),ToReal(18)))))))))
#else
#  define PDupwindNthfdOrder43(u) (PDupwindNthfdOrder43_impl(u,p1o12dz,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthfdOrder43_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o12dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthfdOrder43_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o12dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthfdOrder63(u) (kmul(pm1o60dz,kmul(dir3,kadd(KRANC_GFOFFSET3D(u,0,0,4),kmadd(KRANC_GFOFFSET3D(u,0,0,1),ToReal(-80),kmadd(KRANC_GFOFFSET3D(u,0,0,3),ToReal(-8),kmadd(KRANC_GFOFFSET3D(u,0,0,-2),ToReal(-2),kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(24),kmadd(KRANC_GFOFFSET3D(u,0,0,2),ToReal(30),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(35)))))))))))
#else
#  define PDupwindNthfdOrder63(u) (PDupwindNthfdOrder63_impl(u,pm1o60dz,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthfdOrder63_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o60dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthfdOrder63_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o60dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthfdOrder83(u) (kmul(p1o840dz,kmul(dir3,kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-378),kmadd(KRANC_GFOFFSET3D(u,0,0,5),ToReal(3),kmul(ToReal(-5),kadd(KRANC_GFOFFSET3D(u,0,0,-3),kmadd(KRANC_GFOFFSET3D(u,0,0,1),ToReal(-210),kmadd(KRANC_GFOFFSET3D(u,0,0,3),ToReal(-28),kmadd(KRANC_GFOFFSET3D(u,0,0,-2),ToReal(-12),kmadd(KRANC_GFOFFSET3D(u,0,0,4),ToReal(6),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(84)))))))))))))
#else
#  define PDupwindNthfdOrder83(u) (PDupwindNthfdOrder83_impl(u,p1o840dz,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthfdOrder83_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o840dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthfdOrder83_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o840dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthAntifdOrder23(u) (kmul(p1o4dz,kadd(KRANC_GFOFFSET3D(u,0,0,-2),kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-4),kmsub(KRANC_GFOFFSET3D(u,0,0,1),ToReal(4),KRANC_GFOFFSET3D(u,0,0,2))))))
#else
#  define PDupwindNthAntifdOrder23(u) (PDupwindNthAntifdOrder23_impl(u,p1o4dz,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthAntifdOrder23_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthAntifdOrder23_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o4dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o4dz,kadd(KRANC_GFOFFSET3D(u,0,0,-2),kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-4),kmsub(KRANC_GFOFFSET3D(u,0,0,1),ToReal(4),KRANC_GFOFFSET3D(u,0,0,2)))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthAntifdOrder43(u) (kmul(p1o24dz,kadd(KRANC_GFOFFSET3D(u,0,0,3),kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-21),kmadd(KRANC_GFOFFSET3D(u,0,0,2),ToReal(-6),ksub(kmadd(KRANC_GFOFFSET3D(u,0,0,-2),ToReal(6),kmul(KRANC_GFOFFSET3D(u,0,0,1),ToReal(21))),KRANC_GFOFFSET3D(u,0,0,-3)))))))
#else
#  define PDupwindNthAntifdOrder43(u) (PDupwindNthAntifdOrder43_impl(u,p1o24dz,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthAntifdOrder43_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o24dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthAntifdOrder43_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o24dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o24dz,kadd(KRANC_GFOFFSET3D(u,0,0,3),kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-21),kmadd(KRANC_GFOFFSET3D(u,0,0,2),ToReal(-6),ksub(kmadd(KRANC_GFOFFSET3D(u,0,0,-2),ToReal(6),kmul(KRANC_GFOFFSET3D(u,0,0,1),ToReal(21))),KRANC_GFOFFSET3D(u,0,0,-3))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthAntifdOrder63(u) (kmul(p1o120dz,kadd(KRANC_GFOFFSET3D(u,0,0,-4),kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-104),kmadd(KRANC_GFOFFSET3D(u,0,0,2),ToReal(-32),kmadd(KRANC_GFOFFSET3D(u,0,0,-3),ToReal(-8),ksub(kmadd(KRANC_GFOFFSET3D(u,0,0,3),ToReal(8),kmadd(KRANC_GFOFFSET3D(u,0,0,-2),ToReal(32),kmul(KRANC_GFOFFSET3D(u,0,0,1),ToReal(104)))),KRANC_GFOFFSET3D(u,0,0,4))))))))
#else
#  define PDupwindNthAntifdOrder63(u) (PDupwindNthAntifdOrder63_impl(u,p1o120dz,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthAntifdOrder63_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o120dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthAntifdOrder63_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o120dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o120dz,kadd(KRANC_GFOFFSET3D(u,0,0,-4),kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-104),kmadd(KRANC_GFOFFSET3D(u,0,0,2),ToReal(-32),kmadd(KRANC_GFOFFSET3D(u,0,0,-3),ToReal(-8),ksub(kmadd(KRANC_GFOFFSET3D(u,0,0,3),ToReal(8),kmadd(KRANC_GFOFFSET3D(u,0,0,-2),ToReal(32),kmul(KRANC_GFOFFSET3D(u,0,0,1),ToReal(104)))),KRANC_GFOFFSET3D(u,0,0,4)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthAntifdOrder83(u) (kmul(p1o1680dz,kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-1470),kmadd(KRANC_GFOFFSET3D(u,0,0,2),ToReal(-480),kmadd(KRANC_GFOFFSET3D(u,0,0,-3),ToReal(-145),kmadd(KRANC_GFOFFSET3D(u,0,0,4),ToReal(-30),kmadd(KRANC_GFOFFSET3D(u,0,0,-5),ToReal(-3),kmadd(KRANC_GFOFFSET3D(u,0,0,5),ToReal(3),kmadd(KRANC_GFOFFSET3D(u,0,0,-4),ToReal(30),kmadd(KRANC_GFOFFSET3D(u,0,0,3),ToReal(145),kmadd(KRANC_GFOFFSET3D(u,0,0,-2),ToReal(480),kmul(KRANC_GFOFFSET3D(u,0,0,1),ToReal(1470)))))))))))))
#else
#  define PDupwindNthAntifdOrder83(u) (PDupwindNthAntifdOrder83_impl(u,p1o1680dz,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthAntifdOrder83_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o1680dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthAntifdOrder83_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o1680dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o1680dz,kmadd(KRANC_GFOFFSET3D(u,0,0,-1),ToReal(-1470),kmadd(KRANC_GFOFFSET3D(u,0,0,2),ToReal(-480),kmadd(KRANC_GFOFFSET3D(u,0,0,-3),ToReal(-145),kmadd(KRANC_GFOFFSET3D(u,0,0,4),ToReal(-30),kmadd(KRANC_GFOFFSET3D(u,0,0,-5),ToReal(-3),kmadd(KRANC_GFOFFSET3D(u,0,0,5),ToReal(3),kmadd(KRANC_GFOFFSET3D(u,0,0,-4),ToReal(30),kmadd(KRANC_GFOFFSET3D(u,0,0,3),ToReal(145),kmadd(KRANC_GFOFFSET3D(u,0,0,-2),ToReal(480),kmul(KRANC_GFOFFSET3D(u,0,0,1),ToReal(1470))))))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthSymmfdOrder23(u) (kmul(pm1o4dz,kadd(KRANC_GFOFFSET3D(u,0,0,-2),kadd(KRANC_GFOFFSET3D(u,0,0,2),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(6)))))))
#else
#  define PDupwindNthSymmfdOrder23(u) (PDupwindNthSymmfdOrder23_impl(u,pm1o4dz,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthSymmfdOrder23_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o4dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthSymmfdOrder23_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o4dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(pm1o4dz,kadd(KRANC_GFOFFSET3D(u,0,0,-2),kadd(KRANC_GFOFFSET3D(u,0,0,2),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(-4),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(6))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthSymmfdOrder43(u) (kmul(p1o24dz,kadd(KRANC_GFOFFSET3D(u,0,0,-3),kadd(KRANC_GFOFFSET3D(u,0,0,3),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-20),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(-6),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(15))))))))
#else
#  define PDupwindNthSymmfdOrder43(u) (PDupwindNthSymmfdOrder43_impl(u,p1o24dz,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthSymmfdOrder43_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o24dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthSymmfdOrder43_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o24dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o24dz,kadd(KRANC_GFOFFSET3D(u,0,0,-3),kadd(KRANC_GFOFFSET3D(u,0,0,3),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-20),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(-6),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(15)))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthSymmfdOrder63(u) (kmul(pm1o120dz,kadd(KRANC_GFOFFSET3D(u,0,0,-4),kadd(KRANC_GFOFFSET3D(u,0,0,4),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(-56),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-3),KRANC_GFOFFSET3D(u,0,0,3)),ToReal(-8),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(28),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(70)))))))))
#else
#  define PDupwindNthSymmfdOrder63(u) (PDupwindNthSymmfdOrder63_impl(u,pm1o120dz,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthSymmfdOrder63_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o120dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthSymmfdOrder63_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const pm1o120dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(pm1o120dz,kadd(KRANC_GFOFFSET3D(u,0,0,-4),kadd(KRANC_GFOFFSET3D(u,0,0,4),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(-56),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-3),KRANC_GFOFFSET3D(u,0,0,3)),ToReal(-8),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(28),kmul(KRANC_GFOFFSET3D(u,0,0,0),ToReal(70))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDupwindNthSymmfdOrder83(u) (kmul(p1o560dz,kadd(KRANC_GFOFFSET3D(u,0,0,-5),kadd(KRANC_GFOFFSET3D(u,0,0,5),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-252),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(-120),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-4),KRANC_GFOFFSET3D(u,0,0,4)),ToReal(-10),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-3),KRANC_GFOFFSET3D(u,0,0,3)),ToReal(45),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(210))))))))))
#else
#  define PDupwindNthSymmfdOrder83(u) (PDupwindNthSymmfdOrder83_impl(u,p1o560dz,cdj,cdk))
static CCTK_REAL_VEC PDupwindNthSymmfdOrder83_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o560dz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDupwindNthSymmfdOrder83_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1o560dz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{
  ptrdiff_t const cdi=sizeof(CCTK_REAL);
  return kmul(p1o560dz,kadd(KRANC_GFOFFSET3D(u,0,0,-5),kadd(KRANC_GFOFFSET3D(u,0,0,5),kmadd(KRANC_GFOFFSET3D(u,0,0,0),ToReal(-252),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-2),KRANC_GFOFFSET3D(u,0,0,2)),ToReal(-120),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-4),KRANC_GFOFFSET3D(u,0,0,4)),ToReal(-10),kmadd(kadd(KRANC_GFOFFSET3D(u,0,0,-3),KRANC_GFOFFSET3D(u,0,0,3)),ToReal(45),kmul(kadd(KRANC_GFOFFSET3D(u,0,0,-1),KRANC_GFOFFSET3D(u,0,0,1)),ToReal(210)))))))));
}
#endif

#ifndef KRANC_DIFF_FUNCTIONS
#  define PDonesided3(u) (kmul(p1odz,kmul(dir3,ksub(KRANC_GFOFFSET3D(u,0,0,1),KRANC_GFOFFSET3D(u,0,0,0)))))
#else
#  define PDonesided3(u) (PDonesided3_impl(u,p1odz,cdj,cdk))
static CCTK_REAL_VEC PDonesided3_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1odz, ptrdiff_t const cdj, ptrdiff_t const cdk) CCTK_ATTRIBUTE_NOINLINE CCTK_ATTRIBUTE_UNUSED;
static CCTK_REAL_VEC PDonesided3_impl(CCTK_REAL const* restrict const u, CCTK_REAL_VEC const p1odz, ptrdiff_t const cdj, ptrdiff_t const cdk)
{ assert(0); return ToReal(1e30); /* ERROR */ }
#endif


// Kernel Function:
kernel
__attribute__((vec_type_hint(CCTK_REAL_VEC)))
__attribute__((reqd_work_group_size(GROUP_SIZE_I, GROUP_SIZE_J, GROUP_SIZE_K)))
void ML_BSSN_CL_RHS2
 (cGH constant *restrict const cctkGH,
   cctk_parameters_t constant *restrict const cctk_parameters,
   CCTK_REAL global *restrict const At11,
   CCTK_REAL global *restrict const At11_p,
   CCTK_REAL global *restrict const At11_p_p,
   CCTK_REAL global *restrict const At12,
   CCTK_REAL global *restrict const At12_p,
   CCTK_REAL global *restrict const At12_p_p,
   CCTK_REAL global *restrict const At13,
   CCTK_REAL global *restrict const At13_p,
   CCTK_REAL global *restrict const At13_p_p,
   CCTK_REAL global *restrict const At22,
   CCTK_REAL global *restrict const At22_p,
   CCTK_REAL global *restrict const At22_p_p,
   CCTK_REAL global *restrict const At23,
   CCTK_REAL global *restrict const At23_p,
   CCTK_REAL global *restrict const At23_p_p,
   CCTK_REAL global *restrict const At33,
   CCTK_REAL global *restrict const At33_p,
   CCTK_REAL global *restrict const At33_p_p,
   CCTK_REAL global *restrict const At11rhs,
   CCTK_REAL global *restrict const At12rhs,
   CCTK_REAL global *restrict const At13rhs,
   CCTK_REAL global *restrict const At22rhs,
   CCTK_REAL global *restrict const At23rhs,
   CCTK_REAL global *restrict const At33rhs,
   CCTK_REAL global *restrict const Xt1,
   CCTK_REAL global *restrict const Xt1_p,
   CCTK_REAL global *restrict const Xt1_p_p,
   CCTK_REAL global *restrict const Xt2,
   CCTK_REAL global *restrict const Xt2_p,
   CCTK_REAL global *restrict const Xt2_p_p,
   CCTK_REAL global *restrict const Xt3,
   CCTK_REAL global *restrict const Xt3_p,
   CCTK_REAL global *restrict const Xt3_p_p,
   CCTK_REAL global *restrict const alpha,
   CCTK_REAL global *restrict const alpha_p,
   CCTK_REAL global *restrict const alpha_p_p,
   CCTK_REAL global *restrict const phi,
   CCTK_REAL global *restrict const phi_p,
   CCTK_REAL global *restrict const phi_p_p,
   CCTK_REAL global *restrict const gt11,
   CCTK_REAL global *restrict const gt11_p,
   CCTK_REAL global *restrict const gt11_p_p,
   CCTK_REAL global *restrict const gt12,
   CCTK_REAL global *restrict const gt12_p,
   CCTK_REAL global *restrict const gt12_p_p,
   CCTK_REAL global *restrict const gt13,
   CCTK_REAL global *restrict const gt13_p,
   CCTK_REAL global *restrict const gt13_p_p,
   CCTK_REAL global *restrict const gt22,
   CCTK_REAL global *restrict const gt22_p,
   CCTK_REAL global *restrict const gt22_p_p,
   CCTK_REAL global *restrict const gt23,
   CCTK_REAL global *restrict const gt23_p,
   CCTK_REAL global *restrict const gt23_p_p,
   CCTK_REAL global *restrict const gt33,
   CCTK_REAL global *restrict const gt33_p,
   CCTK_REAL global *restrict const gt33_p_p,
   CCTK_REAL global *restrict const beta1,
   CCTK_REAL global *restrict const beta1_p,
   CCTK_REAL global *restrict const beta1_p_p,
   CCTK_REAL global *restrict const beta2,
   CCTK_REAL global *restrict const beta2_p,
   CCTK_REAL global *restrict const beta2_p_p,
   CCTK_REAL global *restrict const beta3,
   CCTK_REAL global *restrict const beta3_p,
   CCTK_REAL global *restrict const beta3_p_p,
   CCTK_REAL global *restrict const trK,
   CCTK_REAL global *restrict const trK_p,
   CCTK_REAL global *restrict const trK_p_p)
{
  DECLARE_CCTK_ARGUMENTS
  DECLARE_CCTK_PARAMETERS

  // The Kernel:

/* Include user-supplied include files */

/* Initialise finite differencing variables */
ptrdiff_t const di CCTK_ATTRIBUTE_UNUSED  = 1;
ptrdiff_t const dj CCTK_ATTRIBUTE_UNUSED  = CCTK_GFINDEX3D(cctkGH,0,1,0) - CCTK_GFINDEX3D(cctkGH,0,0,0);
ptrdiff_t const dk CCTK_ATTRIBUTE_UNUSED  = CCTK_GFINDEX3D(cctkGH,0,0,1) - CCTK_GFINDEX3D(cctkGH,0,0,0);
ptrdiff_t const cdi CCTK_ATTRIBUTE_UNUSED  = sizeof(CCTK_REAL) * di;
ptrdiff_t const cdj CCTK_ATTRIBUTE_UNUSED  = sizeof(CCTK_REAL) * dj;
ptrdiff_t const cdk CCTK_ATTRIBUTE_UNUSED  = sizeof(CCTK_REAL) * dk;
CCTK_REAL_VEC const dx CCTK_ATTRIBUTE_UNUSED  = ToReal(CCTK_DELTA_SPACE(0));
CCTK_REAL_VEC const dy CCTK_ATTRIBUTE_UNUSED  = ToReal(CCTK_DELTA_SPACE(1));
CCTK_REAL_VEC const dz CCTK_ATTRIBUTE_UNUSED  = ToReal(CCTK_DELTA_SPACE(2));
CCTK_REAL_VEC const dt CCTK_ATTRIBUTE_UNUSED  = ToReal(CCTK_DELTA_TIME);
CCTK_REAL_VEC const t CCTK_ATTRIBUTE_UNUSED  = ToReal(cctk_time);
CCTK_REAL_VEC const dxi CCTK_ATTRIBUTE_UNUSED  = INV(dx);
CCTK_REAL_VEC const dyi CCTK_ATTRIBUTE_UNUSED  = INV(dy);
CCTK_REAL_VEC const dzi CCTK_ATTRIBUTE_UNUSED  = INV(dz);
CCTK_REAL_VEC const khalf CCTK_ATTRIBUTE_UNUSED  = ToReal(0.5);
CCTK_REAL_VEC const kthird CCTK_ATTRIBUTE_UNUSED  = ToReal(1.0/3.0);
CCTK_REAL_VEC const ktwothird CCTK_ATTRIBUTE_UNUSED  = ToReal(2.0/3.0);
CCTK_REAL_VEC const kfourthird CCTK_ATTRIBUTE_UNUSED  = ToReal(4.0/3.0);
CCTK_REAL_VEC const keightthird CCTK_ATTRIBUTE_UNUSED  = ToReal(8.0/3.0);
CCTK_REAL_VEC const hdxi CCTK_ATTRIBUTE_UNUSED  = kmul(ToReal(0.5), dxi);
CCTK_REAL_VEC const hdyi CCTK_ATTRIBUTE_UNUSED  = kmul(ToReal(0.5), dyi);
CCTK_REAL_VEC const hdzi CCTK_ATTRIBUTE_UNUSED  = kmul(ToReal(0.5), dzi);

/* Initialize predefined quantities */
CCTK_REAL_VEC const p1o1024dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0009765625),dx);
CCTK_REAL_VEC const p1o1024dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0009765625),dy);
CCTK_REAL_VEC const p1o1024dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0009765625),dz);
CCTK_REAL_VEC const p1o120dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00833333333333333333333333333333),dx);
CCTK_REAL_VEC const p1o120dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00833333333333333333333333333333),dy);
CCTK_REAL_VEC const p1o120dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00833333333333333333333333333333),dz);
CCTK_REAL_VEC const p1o12dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0833333333333333333333333333333),dx);
CCTK_REAL_VEC const p1o12dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0833333333333333333333333333333),dy);
CCTK_REAL_VEC const p1o12dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0833333333333333333333333333333),dz);
CCTK_REAL_VEC const p1o144dxdy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00694444444444444444444444444444),kmul(dy,dx));
CCTK_REAL_VEC const p1o144dxdz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00694444444444444444444444444444),kmul(dz,dx));
CCTK_REAL_VEC const p1o144dydz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00694444444444444444444444444444),kmul(dz,dy));
CCTK_REAL_VEC const p1o1680dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.000595238095238095238095238095238),dx);
CCTK_REAL_VEC const p1o1680dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.000595238095238095238095238095238),dy);
CCTK_REAL_VEC const p1o1680dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.000595238095238095238095238095238),dz);
CCTK_REAL_VEC const p1o16dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0625),dx);
CCTK_REAL_VEC const p1o16dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0625),dy);
CCTK_REAL_VEC const p1o16dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0625),dz);
CCTK_REAL_VEC const p1o180dx2 CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00555555555555555555555555555556),kmul(dx,dx));
CCTK_REAL_VEC const p1o180dy2 CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00555555555555555555555555555556),kmul(dy,dy));
CCTK_REAL_VEC const p1o180dz2 CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00555555555555555555555555555556),kmul(dz,dz));
CCTK_REAL_VEC const p1o24dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0416666666666666666666666666667),dx);
CCTK_REAL_VEC const p1o24dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0416666666666666666666666666667),dy);
CCTK_REAL_VEC const p1o24dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0416666666666666666666666666667),dz);
CCTK_REAL_VEC const p1o256dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00390625),dx);
CCTK_REAL_VEC const p1o256dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00390625),dy);
CCTK_REAL_VEC const p1o256dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00390625),dz);
CCTK_REAL_VEC const p1o2dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.5),dx);
CCTK_REAL_VEC const p1o2dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.5),dy);
CCTK_REAL_VEC const p1o2dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.5),dz);
CCTK_REAL_VEC const p1o3600dxdy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.000277777777777777777777777777778),kmul(dy,dx));
CCTK_REAL_VEC const p1o3600dxdz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.000277777777777777777777777777778),kmul(dz,dx));
CCTK_REAL_VEC const p1o3600dydz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.000277777777777777777777777777778),kmul(dz,dy));
CCTK_REAL_VEC const p1o4dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.25),dx);
CCTK_REAL_VEC const p1o4dxdy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.25),kmul(dy,dx));
CCTK_REAL_VEC const p1o4dxdz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.25),kmul(dz,dx));
CCTK_REAL_VEC const p1o4dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.25),dy);
CCTK_REAL_VEC const p1o4dydz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.25),kmul(dz,dy));
CCTK_REAL_VEC const p1o4dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.25),dz);
CCTK_REAL_VEC const p1o5040dx2 CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.000198412698412698412698412698413),kmul(dx,dx));
CCTK_REAL_VEC const p1o5040dy2 CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.000198412698412698412698412698413),kmul(dy,dy));
CCTK_REAL_VEC const p1o5040dz2 CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.000198412698412698412698412698413),kmul(dz,dz));
CCTK_REAL_VEC const p1o560dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00178571428571428571428571428571),dx);
CCTK_REAL_VEC const p1o560dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00178571428571428571428571428571),dy);
CCTK_REAL_VEC const p1o560dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00178571428571428571428571428571),dz);
CCTK_REAL_VEC const p1o60dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0166666666666666666666666666667),dx);
CCTK_REAL_VEC const p1o60dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0166666666666666666666666666667),dy);
CCTK_REAL_VEC const p1o60dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.0166666666666666666666666666667),dz);
CCTK_REAL_VEC const p1o64dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.015625),dx);
CCTK_REAL_VEC const p1o64dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.015625),dy);
CCTK_REAL_VEC const p1o64dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.015625),dz);
CCTK_REAL_VEC const p1o705600dxdy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(1.41723356009070294784580498866e-6),kmul(dy,dx));
CCTK_REAL_VEC const p1o705600dxdz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(1.41723356009070294784580498866e-6),kmul(dz,dx));
CCTK_REAL_VEC const p1o705600dydz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(1.41723356009070294784580498866e-6),kmul(dz,dy));
CCTK_REAL_VEC const p1o840dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00119047619047619047619047619048),dx);
CCTK_REAL_VEC const p1o840dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00119047619047619047619047619048),dy);
CCTK_REAL_VEC const p1o840dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(0.00119047619047619047619047619048),dz);
CCTK_REAL_VEC const p1odx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(1),dx);
CCTK_REAL_VEC const p1odx2 CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(1),kmul(dx,dx));
CCTK_REAL_VEC const p1ody CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(1),dy);
CCTK_REAL_VEC const p1ody2 CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(1),kmul(dy,dy));
CCTK_REAL_VEC const p1odz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(1),dz);
CCTK_REAL_VEC const p1odz2 CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(1),kmul(dz,dz));
CCTK_REAL_VEC const pm1o120dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.00833333333333333333333333333333),dx);
CCTK_REAL_VEC const pm1o120dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.00833333333333333333333333333333),dy);
CCTK_REAL_VEC const pm1o120dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.00833333333333333333333333333333),dz);
CCTK_REAL_VEC const pm1o12dx2 CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.0833333333333333333333333333333),kmul(dx,dx));
CCTK_REAL_VEC const pm1o12dy2 CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.0833333333333333333333333333333),kmul(dy,dy));
CCTK_REAL_VEC const pm1o12dz2 CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.0833333333333333333333333333333),kmul(dz,dz));
CCTK_REAL_VEC const pm1o2dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.5),dx);
CCTK_REAL_VEC const pm1o2dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.5),dy);
CCTK_REAL_VEC const pm1o2dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.5),dz);
CCTK_REAL_VEC const pm1o4dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.25),dx);
CCTK_REAL_VEC const pm1o4dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.25),dy);
CCTK_REAL_VEC const pm1o4dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.25),dz);
CCTK_REAL_VEC const pm1o60dx CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.0166666666666666666666666666667),dx);
CCTK_REAL_VEC const pm1o60dy CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.0166666666666666666666666666667),dy);
CCTK_REAL_VEC const pm1o60dz CCTK_ATTRIBUTE_UNUSED  = kdiv(ToReal(-0.0166666666666666666666666666667),dz);

/* Jacobian variable pointers */
bool const use_jacobian = (!CCTK_IsFunctionAliased("MultiPatch_GetMap") || MultiPatch_GetMap(cctkGH) != jacobian_identity_map)
                     && strlen(jacobian_group) > 0;
bool const usejacobian = use_jacobian;
if (use_jacobian && (strlen(jacobian_determinant_group) == 0 || strlen(jacobian_inverse_group) == 0 || strlen(jacobian_derivative_group) == 0))
{
  CCTK_WARN (1, "GenericFD::jacobian_group, GenericFD::jacobian_determinant_group, GenericFD::jacobian_inverse_group, and GenericFD::jacobian_derivative_group must all be set to valid group names");
}

CCTK_REAL const *restrict jacobian_ptrs[9];
if (use_jacobian) GenericFD_GroupDataPointers(cctkGH, jacobian_group,
                                              9, jacobian_ptrs);

CCTK_REAL const *restrict const J11 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_ptrs[0] : 0;
CCTK_REAL const *restrict const J12 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_ptrs[1] : 0;
CCTK_REAL const *restrict const J13 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_ptrs[2] : 0;
CCTK_REAL const *restrict const J21 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_ptrs[3] : 0;
CCTK_REAL const *restrict const J22 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_ptrs[4] : 0;
CCTK_REAL const *restrict const J23 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_ptrs[5] : 0;
CCTK_REAL const *restrict const J31 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_ptrs[6] : 0;
CCTK_REAL const *restrict const J32 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_ptrs[7] : 0;
CCTK_REAL const *restrict const J33 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_ptrs[8] : 0;

CCTK_REAL const *restrict jacobian_determinant_ptrs[1] CCTK_ATTRIBUTE_UNUSED;
if (use_jacobian) GenericFD_GroupDataPointers(cctkGH, jacobian_determinant_group,
                                              1, jacobian_determinant_ptrs);

CCTK_REAL const *restrict const detJ CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_ptrs[0] : 0;

CCTK_REAL const *restrict jacobian_inverse_ptrs[9] CCTK_ATTRIBUTE_UNUSED;
if (use_jacobian) GenericFD_GroupDataPointers(cctkGH, jacobian_inverse_group,
                                              9, jacobian_inverse_ptrs);

CCTK_REAL const *restrict const iJ11 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_inverse_ptrs[0] : 0;
CCTK_REAL const *restrict const iJ12 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_inverse_ptrs[1] : 0;
CCTK_REAL const *restrict const iJ13 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_inverse_ptrs[2] : 0;
CCTK_REAL const *restrict const iJ21 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_inverse_ptrs[3] : 0;
CCTK_REAL const *restrict const iJ22 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_inverse_ptrs[4] : 0;
CCTK_REAL const *restrict const iJ23 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_inverse_ptrs[5] : 0;
CCTK_REAL const *restrict const iJ31 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_inverse_ptrs[6] : 0;
CCTK_REAL const *restrict const iJ32 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_inverse_ptrs[7] : 0;
CCTK_REAL const *restrict const iJ33 CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_inverse_ptrs[8] : 0;

CCTK_REAL const *restrict jacobian_derivative_ptrs[18] CCTK_ATTRIBUTE_UNUSED;
if (use_jacobian) GenericFD_GroupDataPointers(cctkGH, jacobian_derivative_group,
                                              18, jacobian_derivative_ptrs);

CCTK_REAL const *restrict const dJ111  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[0] : 0;
CCTK_REAL const *restrict const dJ112  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[1] : 0;
CCTK_REAL const *restrict const dJ113  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[2] : 0;
CCTK_REAL const *restrict const dJ122  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[3] : 0;
CCTK_REAL const *restrict const dJ123  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[4] : 0;
CCTK_REAL const *restrict const dJ133  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[5] : 0;
CCTK_REAL const *restrict const dJ211  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[6] : 0;
CCTK_REAL const *restrict const dJ212  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[7] : 0;
CCTK_REAL const *restrict const dJ213  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[8] : 0;
CCTK_REAL const *restrict const dJ222  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[9] : 0;
CCTK_REAL const *restrict const dJ223  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[10] : 0;
CCTK_REAL const *restrict const dJ233  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[11] : 0;
CCTK_REAL const *restrict const dJ311  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[12] : 0;
CCTK_REAL const *restrict const dJ312  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[13] : 0;
CCTK_REAL const *restrict const dJ313  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[14] : 0;
CCTK_REAL const *restrict const dJ322  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[15] : 0;
CCTK_REAL const *restrict const dJ323  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[16] : 0;
CCTK_REAL const *restrict const dJ333  CCTK_ATTRIBUTE_UNUSED = use_jacobian ? jacobian_derivative_ptrs[17] : 0;

/* Assign local copies of arrays functions */



/* Calculate temporaries and arrays functions */

/* Copy local copies back to grid functions */

/* Loop over the grid points */
#pragma omp parallel
LC_LOOP3VEC(ML_BSSN_CL_RHS2,
  i,j,k, imin[0],imin[1],imin[2], imax[0],imax[1],imax[2],
  cctk_ash[0],cctk_ash[1],cctk_ash[2],
  CCTK_REAL_VEC_SIZE)
{
  ptrdiff_t const index CCTK_ATTRIBUTE_UNUSED  = di*i + dj*j + dk*k;
  
  /* Assign local copies of grid functions */
  
  CCTK_REAL_VEC alphaL CCTK_ATTRIBUTE_UNUSED = vec_load(alpha[index]);
  CCTK_REAL_VEC At11L CCTK_ATTRIBUTE_UNUSED = vec_load(At11[index]);
  CCTK_REAL_VEC At12L CCTK_ATTRIBUTE_UNUSED = vec_load(At12[index]);
  CCTK_REAL_VEC At13L CCTK_ATTRIBUTE_UNUSED = vec_load(At13[index]);
  CCTK_REAL_VEC At22L CCTK_ATTRIBUTE_UNUSED = vec_load(At22[index]);
  CCTK_REAL_VEC At23L CCTK_ATTRIBUTE_UNUSED = vec_load(At23[index]);
  CCTK_REAL_VEC At33L CCTK_ATTRIBUTE_UNUSED = vec_load(At33[index]);
  CCTK_REAL_VEC beta1L CCTK_ATTRIBUTE_UNUSED = vec_load(beta1[index]);
  CCTK_REAL_VEC beta2L CCTK_ATTRIBUTE_UNUSED = vec_load(beta2[index]);
  CCTK_REAL_VEC beta3L CCTK_ATTRIBUTE_UNUSED = vec_load(beta3[index]);
  CCTK_REAL_VEC gt11L CCTK_ATTRIBUTE_UNUSED = vec_load(gt11[index]);
  CCTK_REAL_VEC gt12L CCTK_ATTRIBUTE_UNUSED = vec_load(gt12[index]);
  CCTK_REAL_VEC gt13L CCTK_ATTRIBUTE_UNUSED = vec_load(gt13[index]);
  CCTK_REAL_VEC gt22L CCTK_ATTRIBUTE_UNUSED = vec_load(gt22[index]);
  CCTK_REAL_VEC gt23L CCTK_ATTRIBUTE_UNUSED = vec_load(gt23[index]);
  CCTK_REAL_VEC gt33L CCTK_ATTRIBUTE_UNUSED = vec_load(gt33[index]);
  CCTK_REAL_VEC phiL CCTK_ATTRIBUTE_UNUSED = vec_load(phi[index]);
  CCTK_REAL_VEC trKL CCTK_ATTRIBUTE_UNUSED = vec_load(trK[index]);
  CCTK_REAL_VEC Xt1L CCTK_ATTRIBUTE_UNUSED = vec_load(Xt1[index]);
  CCTK_REAL_VEC Xt2L CCTK_ATTRIBUTE_UNUSED = vec_load(Xt2[index]);
  CCTK_REAL_VEC Xt3L CCTK_ATTRIBUTE_UNUSED = vec_load(Xt3[index]);
  
  CCTK_REAL_VEC eTxxL, eTxyL, eTxzL, eTyyL, eTyzL, eTzzL CCTK_ATTRIBUTE_UNUSED ;
  
  if (*stress_energy_state)
  {
    eTxxL = vec_load(eTxx[index]);
    eTxyL = vec_load(eTxy[index]);
    eTxzL = vec_load(eTxz[index]);
    eTyyL = vec_load(eTyy[index]);
    eTyzL = vec_load(eTyz[index]);
    eTzzL = vec_load(eTzz[index]);
  }
  else
  {
    eTxxL = ToReal(0.0);
    eTxyL = ToReal(0.0);
    eTxzL = ToReal(0.0);
    eTyyL = ToReal(0.0);
    eTyzL = ToReal(0.0);
    eTzzL = ToReal(0.0);
  }
  
  CCTK_REAL_VEC dJ111L, dJ112L, dJ113L, dJ122L, dJ123L, dJ133L, dJ211L, dJ212L, dJ213L, dJ222L, dJ223L, dJ233L, dJ311L, dJ312L, dJ313L, dJ322L, dJ323L, dJ333L, J11L, J12L, J13L, J21L, J22L, J23L, J31L, J32L, J33L CCTK_ATTRIBUTE_UNUSED ;
  
  if (use_jacobian)
  {
    dJ111L = vec_load(dJ111[index]);
    dJ112L = vec_load(dJ112[index]);
    dJ113L = vec_load(dJ113[index]);
    dJ122L = vec_load(dJ122[index]);
    dJ123L = vec_load(dJ123[index]);
    dJ133L = vec_load(dJ133[index]);
    dJ211L = vec_load(dJ211[index]);
    dJ212L = vec_load(dJ212[index]);
    dJ213L = vec_load(dJ213[index]);
    dJ222L = vec_load(dJ222[index]);
    dJ223L = vec_load(dJ223[index]);
    dJ233L = vec_load(dJ233[index]);
    dJ311L = vec_load(dJ311[index]);
    dJ312L = vec_load(dJ312[index]);
    dJ313L = vec_load(dJ313[index]);
    dJ322L = vec_load(dJ322[index]);
    dJ323L = vec_load(dJ323[index]);
    dJ333L = vec_load(dJ333[index]);
    J11L = vec_load(J11[index]);
    J12L = vec_load(J12[index]);
    J13L = vec_load(J13[index]);
    J21L = vec_load(J21[index]);
    J22L = vec_load(J22[index]);
    J23L = vec_load(J23[index]);
    J31L = vec_load(J31[index]);
    J32L = vec_load(J32[index]);
    J33L = vec_load(J33[index]);
  }
  
  /* Include user supplied include files */
  
  /* Precompute derivatives */
  CCTK_REAL_VEC PDstandardNth1alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth11alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth22alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth33alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth12alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth13alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth23alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1beta1 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2beta1 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3beta1 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1beta2 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2beta2 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3beta2 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1beta3 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2beta3 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3beta3 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth11gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth22gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth33gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth12gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth13gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth23gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth11gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth22gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth33gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth12gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth13gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth23gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth11gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth22gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth33gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth12gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth13gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth23gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth11gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth22gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth33gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth12gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth13gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth23gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth11gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth22gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth33gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth12gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth13gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth23gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth11gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth22gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth33gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth12gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth13gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth23gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth11phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth22phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth33phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth12phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth13phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth23phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1Xt1 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2Xt1 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3Xt1 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1Xt2 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2Xt2 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3Xt2 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth1Xt3 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth2Xt3 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC PDstandardNth3Xt3 CCTK_ATTRIBUTE_UNUSED ;
  
  switch(fdOrder)
  {
    case 2:
      PDstandardNth1alpha = PDstandardNthfdOrder21(&alpha[index]);
      PDstandardNth2alpha = PDstandardNthfdOrder22(&alpha[index]);
      PDstandardNth3alpha = PDstandardNthfdOrder23(&alpha[index]);
      PDstandardNth11alpha = PDstandardNthfdOrder211(&alpha[index]);
      PDstandardNth22alpha = PDstandardNthfdOrder222(&alpha[index]);
      PDstandardNth33alpha = PDstandardNthfdOrder233(&alpha[index]);
      PDstandardNth12alpha = PDstandardNthfdOrder212(&alpha[index]);
      PDstandardNth13alpha = PDstandardNthfdOrder213(&alpha[index]);
      PDstandardNth23alpha = PDstandardNthfdOrder223(&alpha[index]);
      PDstandardNth1beta1 = PDstandardNthfdOrder21(&beta1[index]);
      PDstandardNth2beta1 = PDstandardNthfdOrder22(&beta1[index]);
      PDstandardNth3beta1 = PDstandardNthfdOrder23(&beta1[index]);
      PDstandardNth1beta2 = PDstandardNthfdOrder21(&beta2[index]);
      PDstandardNth2beta2 = PDstandardNthfdOrder22(&beta2[index]);
      PDstandardNth3beta2 = PDstandardNthfdOrder23(&beta2[index]);
      PDstandardNth1beta3 = PDstandardNthfdOrder21(&beta3[index]);
      PDstandardNth2beta3 = PDstandardNthfdOrder22(&beta3[index]);
      PDstandardNth3beta3 = PDstandardNthfdOrder23(&beta3[index]);
      PDstandardNth1gt11 = PDstandardNthfdOrder21(&gt11[index]);
      PDstandardNth2gt11 = PDstandardNthfdOrder22(&gt11[index]);
      PDstandardNth3gt11 = PDstandardNthfdOrder23(&gt11[index]);
      PDstandardNth11gt11 = PDstandardNthfdOrder211(&gt11[index]);
      PDstandardNth22gt11 = PDstandardNthfdOrder222(&gt11[index]);
      PDstandardNth33gt11 = PDstandardNthfdOrder233(&gt11[index]);
      PDstandardNth12gt11 = PDstandardNthfdOrder212(&gt11[index]);
      PDstandardNth13gt11 = PDstandardNthfdOrder213(&gt11[index]);
      PDstandardNth23gt11 = PDstandardNthfdOrder223(&gt11[index]);
      PDstandardNth1gt12 = PDstandardNthfdOrder21(&gt12[index]);
      PDstandardNth2gt12 = PDstandardNthfdOrder22(&gt12[index]);
      PDstandardNth3gt12 = PDstandardNthfdOrder23(&gt12[index]);
      PDstandardNth11gt12 = PDstandardNthfdOrder211(&gt12[index]);
      PDstandardNth22gt12 = PDstandardNthfdOrder222(&gt12[index]);
      PDstandardNth33gt12 = PDstandardNthfdOrder233(&gt12[index]);
      PDstandardNth12gt12 = PDstandardNthfdOrder212(&gt12[index]);
      PDstandardNth13gt12 = PDstandardNthfdOrder213(&gt12[index]);
      PDstandardNth23gt12 = PDstandardNthfdOrder223(&gt12[index]);
      PDstandardNth1gt13 = PDstandardNthfdOrder21(&gt13[index]);
      PDstandardNth2gt13 = PDstandardNthfdOrder22(&gt13[index]);
      PDstandardNth3gt13 = PDstandardNthfdOrder23(&gt13[index]);
      PDstandardNth11gt13 = PDstandardNthfdOrder211(&gt13[index]);
      PDstandardNth22gt13 = PDstandardNthfdOrder222(&gt13[index]);
      PDstandardNth33gt13 = PDstandardNthfdOrder233(&gt13[index]);
      PDstandardNth12gt13 = PDstandardNthfdOrder212(&gt13[index]);
      PDstandardNth13gt13 = PDstandardNthfdOrder213(&gt13[index]);
      PDstandardNth23gt13 = PDstandardNthfdOrder223(&gt13[index]);
      PDstandardNth1gt22 = PDstandardNthfdOrder21(&gt22[index]);
      PDstandardNth2gt22 = PDstandardNthfdOrder22(&gt22[index]);
      PDstandardNth3gt22 = PDstandardNthfdOrder23(&gt22[index]);
      PDstandardNth11gt22 = PDstandardNthfdOrder211(&gt22[index]);
      PDstandardNth22gt22 = PDstandardNthfdOrder222(&gt22[index]);
      PDstandardNth33gt22 = PDstandardNthfdOrder233(&gt22[index]);
      PDstandardNth12gt22 = PDstandardNthfdOrder212(&gt22[index]);
      PDstandardNth13gt22 = PDstandardNthfdOrder213(&gt22[index]);
      PDstandardNth23gt22 = PDstandardNthfdOrder223(&gt22[index]);
      PDstandardNth1gt23 = PDstandardNthfdOrder21(&gt23[index]);
      PDstandardNth2gt23 = PDstandardNthfdOrder22(&gt23[index]);
      PDstandardNth3gt23 = PDstandardNthfdOrder23(&gt23[index]);
      PDstandardNth11gt23 = PDstandardNthfdOrder211(&gt23[index]);
      PDstandardNth22gt23 = PDstandardNthfdOrder222(&gt23[index]);
      PDstandardNth33gt23 = PDstandardNthfdOrder233(&gt23[index]);
      PDstandardNth12gt23 = PDstandardNthfdOrder212(&gt23[index]);
      PDstandardNth13gt23 = PDstandardNthfdOrder213(&gt23[index]);
      PDstandardNth23gt23 = PDstandardNthfdOrder223(&gt23[index]);
      PDstandardNth1gt33 = PDstandardNthfdOrder21(&gt33[index]);
      PDstandardNth2gt33 = PDstandardNthfdOrder22(&gt33[index]);
      PDstandardNth3gt33 = PDstandardNthfdOrder23(&gt33[index]);
      PDstandardNth11gt33 = PDstandardNthfdOrder211(&gt33[index]);
      PDstandardNth22gt33 = PDstandardNthfdOrder222(&gt33[index]);
      PDstandardNth33gt33 = PDstandardNthfdOrder233(&gt33[index]);
      PDstandardNth12gt33 = PDstandardNthfdOrder212(&gt33[index]);
      PDstandardNth13gt33 = PDstandardNthfdOrder213(&gt33[index]);
      PDstandardNth23gt33 = PDstandardNthfdOrder223(&gt33[index]);
      PDstandardNth1phi = PDstandardNthfdOrder21(&phi[index]);
      PDstandardNth2phi = PDstandardNthfdOrder22(&phi[index]);
      PDstandardNth3phi = PDstandardNthfdOrder23(&phi[index]);
      PDstandardNth11phi = PDstandardNthfdOrder211(&phi[index]);
      PDstandardNth22phi = PDstandardNthfdOrder222(&phi[index]);
      PDstandardNth33phi = PDstandardNthfdOrder233(&phi[index]);
      PDstandardNth12phi = PDstandardNthfdOrder212(&phi[index]);
      PDstandardNth13phi = PDstandardNthfdOrder213(&phi[index]);
      PDstandardNth23phi = PDstandardNthfdOrder223(&phi[index]);
      PDstandardNth1Xt1 = PDstandardNthfdOrder21(&Xt1[index]);
      PDstandardNth2Xt1 = PDstandardNthfdOrder22(&Xt1[index]);
      PDstandardNth3Xt1 = PDstandardNthfdOrder23(&Xt1[index]);
      PDstandardNth1Xt2 = PDstandardNthfdOrder21(&Xt2[index]);
      PDstandardNth2Xt2 = PDstandardNthfdOrder22(&Xt2[index]);
      PDstandardNth3Xt2 = PDstandardNthfdOrder23(&Xt2[index]);
      PDstandardNth1Xt3 = PDstandardNthfdOrder21(&Xt3[index]);
      PDstandardNth2Xt3 = PDstandardNthfdOrder22(&Xt3[index]);
      PDstandardNth3Xt3 = PDstandardNthfdOrder23(&Xt3[index]);
      break;
    
    case 4:
      PDstandardNth1alpha = PDstandardNthfdOrder41(&alpha[index]);
      PDstandardNth2alpha = PDstandardNthfdOrder42(&alpha[index]);
      PDstandardNth3alpha = PDstandardNthfdOrder43(&alpha[index]);
      PDstandardNth11alpha = PDstandardNthfdOrder411(&alpha[index]);
      PDstandardNth22alpha = PDstandardNthfdOrder422(&alpha[index]);
      PDstandardNth33alpha = PDstandardNthfdOrder433(&alpha[index]);
      PDstandardNth12alpha = PDstandardNthfdOrder412(&alpha[index]);
      PDstandardNth13alpha = PDstandardNthfdOrder413(&alpha[index]);
      PDstandardNth23alpha = PDstandardNthfdOrder423(&alpha[index]);
      PDstandardNth1beta1 = PDstandardNthfdOrder41(&beta1[index]);
      PDstandardNth2beta1 = PDstandardNthfdOrder42(&beta1[index]);
      PDstandardNth3beta1 = PDstandardNthfdOrder43(&beta1[index]);
      PDstandardNth1beta2 = PDstandardNthfdOrder41(&beta2[index]);
      PDstandardNth2beta2 = PDstandardNthfdOrder42(&beta2[index]);
      PDstandardNth3beta2 = PDstandardNthfdOrder43(&beta2[index]);
      PDstandardNth1beta3 = PDstandardNthfdOrder41(&beta3[index]);
      PDstandardNth2beta3 = PDstandardNthfdOrder42(&beta3[index]);
      PDstandardNth3beta3 = PDstandardNthfdOrder43(&beta3[index]);
      PDstandardNth1gt11 = PDstandardNthfdOrder41(&gt11[index]);
      PDstandardNth2gt11 = PDstandardNthfdOrder42(&gt11[index]);
      PDstandardNth3gt11 = PDstandardNthfdOrder43(&gt11[index]);
      PDstandardNth11gt11 = PDstandardNthfdOrder411(&gt11[index]);
      PDstandardNth22gt11 = PDstandardNthfdOrder422(&gt11[index]);
      PDstandardNth33gt11 = PDstandardNthfdOrder433(&gt11[index]);
      PDstandardNth12gt11 = PDstandardNthfdOrder412(&gt11[index]);
      PDstandardNth13gt11 = PDstandardNthfdOrder413(&gt11[index]);
      PDstandardNth23gt11 = PDstandardNthfdOrder423(&gt11[index]);
      PDstandardNth1gt12 = PDstandardNthfdOrder41(&gt12[index]);
      PDstandardNth2gt12 = PDstandardNthfdOrder42(&gt12[index]);
      PDstandardNth3gt12 = PDstandardNthfdOrder43(&gt12[index]);
      PDstandardNth11gt12 = PDstandardNthfdOrder411(&gt12[index]);
      PDstandardNth22gt12 = PDstandardNthfdOrder422(&gt12[index]);
      PDstandardNth33gt12 = PDstandardNthfdOrder433(&gt12[index]);
      PDstandardNth12gt12 = PDstandardNthfdOrder412(&gt12[index]);
      PDstandardNth13gt12 = PDstandardNthfdOrder413(&gt12[index]);
      PDstandardNth23gt12 = PDstandardNthfdOrder423(&gt12[index]);
      PDstandardNth1gt13 = PDstandardNthfdOrder41(&gt13[index]);
      PDstandardNth2gt13 = PDstandardNthfdOrder42(&gt13[index]);
      PDstandardNth3gt13 = PDstandardNthfdOrder43(&gt13[index]);
      PDstandardNth11gt13 = PDstandardNthfdOrder411(&gt13[index]);
      PDstandardNth22gt13 = PDstandardNthfdOrder422(&gt13[index]);
      PDstandardNth33gt13 = PDstandardNthfdOrder433(&gt13[index]);
      PDstandardNth12gt13 = PDstandardNthfdOrder412(&gt13[index]);
      PDstandardNth13gt13 = PDstandardNthfdOrder413(&gt13[index]);
      PDstandardNth23gt13 = PDstandardNthfdOrder423(&gt13[index]);
      PDstandardNth1gt22 = PDstandardNthfdOrder41(&gt22[index]);
      PDstandardNth2gt22 = PDstandardNthfdOrder42(&gt22[index]);
      PDstandardNth3gt22 = PDstandardNthfdOrder43(&gt22[index]);
      PDstandardNth11gt22 = PDstandardNthfdOrder411(&gt22[index]);
      PDstandardNth22gt22 = PDstandardNthfdOrder422(&gt22[index]);
      PDstandardNth33gt22 = PDstandardNthfdOrder433(&gt22[index]);
      PDstandardNth12gt22 = PDstandardNthfdOrder412(&gt22[index]);
      PDstandardNth13gt22 = PDstandardNthfdOrder413(&gt22[index]);
      PDstandardNth23gt22 = PDstandardNthfdOrder423(&gt22[index]);
      PDstandardNth1gt23 = PDstandardNthfdOrder41(&gt23[index]);
      PDstandardNth2gt23 = PDstandardNthfdOrder42(&gt23[index]);
      PDstandardNth3gt23 = PDstandardNthfdOrder43(&gt23[index]);
      PDstandardNth11gt23 = PDstandardNthfdOrder411(&gt23[index]);
      PDstandardNth22gt23 = PDstandardNthfdOrder422(&gt23[index]);
      PDstandardNth33gt23 = PDstandardNthfdOrder433(&gt23[index]);
      PDstandardNth12gt23 = PDstandardNthfdOrder412(&gt23[index]);
      PDstandardNth13gt23 = PDstandardNthfdOrder413(&gt23[index]);
      PDstandardNth23gt23 = PDstandardNthfdOrder423(&gt23[index]);
      PDstandardNth1gt33 = PDstandardNthfdOrder41(&gt33[index]);
      PDstandardNth2gt33 = PDstandardNthfdOrder42(&gt33[index]);
      PDstandardNth3gt33 = PDstandardNthfdOrder43(&gt33[index]);
      PDstandardNth11gt33 = PDstandardNthfdOrder411(&gt33[index]);
      PDstandardNth22gt33 = PDstandardNthfdOrder422(&gt33[index]);
      PDstandardNth33gt33 = PDstandardNthfdOrder433(&gt33[index]);
      PDstandardNth12gt33 = PDstandardNthfdOrder412(&gt33[index]);
      PDstandardNth13gt33 = PDstandardNthfdOrder413(&gt33[index]);
      PDstandardNth23gt33 = PDstandardNthfdOrder423(&gt33[index]);
      PDstandardNth1phi = PDstandardNthfdOrder41(&phi[index]);
      PDstandardNth2phi = PDstandardNthfdOrder42(&phi[index]);
      PDstandardNth3phi = PDstandardNthfdOrder43(&phi[index]);
      PDstandardNth11phi = PDstandardNthfdOrder411(&phi[index]);
      PDstandardNth22phi = PDstandardNthfdOrder422(&phi[index]);
      PDstandardNth33phi = PDstandardNthfdOrder433(&phi[index]);
      PDstandardNth12phi = PDstandardNthfdOrder412(&phi[index]);
      PDstandardNth13phi = PDstandardNthfdOrder413(&phi[index]);
      PDstandardNth23phi = PDstandardNthfdOrder423(&phi[index]);
      PDstandardNth1Xt1 = PDstandardNthfdOrder41(&Xt1[index]);
      PDstandardNth2Xt1 = PDstandardNthfdOrder42(&Xt1[index]);
      PDstandardNth3Xt1 = PDstandardNthfdOrder43(&Xt1[index]);
      PDstandardNth1Xt2 = PDstandardNthfdOrder41(&Xt2[index]);
      PDstandardNth2Xt2 = PDstandardNthfdOrder42(&Xt2[index]);
      PDstandardNth3Xt2 = PDstandardNthfdOrder43(&Xt2[index]);
      PDstandardNth1Xt3 = PDstandardNthfdOrder41(&Xt3[index]);
      PDstandardNth2Xt3 = PDstandardNthfdOrder42(&Xt3[index]);
      PDstandardNth3Xt3 = PDstandardNthfdOrder43(&Xt3[index]);
      break;
    
    case 6:
      PDstandardNth1alpha = PDstandardNthfdOrder61(&alpha[index]);
      PDstandardNth2alpha = PDstandardNthfdOrder62(&alpha[index]);
      PDstandardNth3alpha = PDstandardNthfdOrder63(&alpha[index]);
      PDstandardNth11alpha = PDstandardNthfdOrder611(&alpha[index]);
      PDstandardNth22alpha = PDstandardNthfdOrder622(&alpha[index]);
      PDstandardNth33alpha = PDstandardNthfdOrder633(&alpha[index]);
      PDstandardNth12alpha = PDstandardNthfdOrder612(&alpha[index]);
      PDstandardNth13alpha = PDstandardNthfdOrder613(&alpha[index]);
      PDstandardNth23alpha = PDstandardNthfdOrder623(&alpha[index]);
      PDstandardNth1beta1 = PDstandardNthfdOrder61(&beta1[index]);
      PDstandardNth2beta1 = PDstandardNthfdOrder62(&beta1[index]);
      PDstandardNth3beta1 = PDstandardNthfdOrder63(&beta1[index]);
      PDstandardNth1beta2 = PDstandardNthfdOrder61(&beta2[index]);
      PDstandardNth2beta2 = PDstandardNthfdOrder62(&beta2[index]);
      PDstandardNth3beta2 = PDstandardNthfdOrder63(&beta2[index]);
      PDstandardNth1beta3 = PDstandardNthfdOrder61(&beta3[index]);
      PDstandardNth2beta3 = PDstandardNthfdOrder62(&beta3[index]);
      PDstandardNth3beta3 = PDstandardNthfdOrder63(&beta3[index]);
      PDstandardNth1gt11 = PDstandardNthfdOrder61(&gt11[index]);
      PDstandardNth2gt11 = PDstandardNthfdOrder62(&gt11[index]);
      PDstandardNth3gt11 = PDstandardNthfdOrder63(&gt11[index]);
      PDstandardNth11gt11 = PDstandardNthfdOrder611(&gt11[index]);
      PDstandardNth22gt11 = PDstandardNthfdOrder622(&gt11[index]);
      PDstandardNth33gt11 = PDstandardNthfdOrder633(&gt11[index]);
      PDstandardNth12gt11 = PDstandardNthfdOrder612(&gt11[index]);
      PDstandardNth13gt11 = PDstandardNthfdOrder613(&gt11[index]);
      PDstandardNth23gt11 = PDstandardNthfdOrder623(&gt11[index]);
      PDstandardNth1gt12 = PDstandardNthfdOrder61(&gt12[index]);
      PDstandardNth2gt12 = PDstandardNthfdOrder62(&gt12[index]);
      PDstandardNth3gt12 = PDstandardNthfdOrder63(&gt12[index]);
      PDstandardNth11gt12 = PDstandardNthfdOrder611(&gt12[index]);
      PDstandardNth22gt12 = PDstandardNthfdOrder622(&gt12[index]);
      PDstandardNth33gt12 = PDstandardNthfdOrder633(&gt12[index]);
      PDstandardNth12gt12 = PDstandardNthfdOrder612(&gt12[index]);
      PDstandardNth13gt12 = PDstandardNthfdOrder613(&gt12[index]);
      PDstandardNth23gt12 = PDstandardNthfdOrder623(&gt12[index]);
      PDstandardNth1gt13 = PDstandardNthfdOrder61(&gt13[index]);
      PDstandardNth2gt13 = PDstandardNthfdOrder62(&gt13[index]);
      PDstandardNth3gt13 = PDstandardNthfdOrder63(&gt13[index]);
      PDstandardNth11gt13 = PDstandardNthfdOrder611(&gt13[index]);
      PDstandardNth22gt13 = PDstandardNthfdOrder622(&gt13[index]);
      PDstandardNth33gt13 = PDstandardNthfdOrder633(&gt13[index]);
      PDstandardNth12gt13 = PDstandardNthfdOrder612(&gt13[index]);
      PDstandardNth13gt13 = PDstandardNthfdOrder613(&gt13[index]);
      PDstandardNth23gt13 = PDstandardNthfdOrder623(&gt13[index]);
      PDstandardNth1gt22 = PDstandardNthfdOrder61(&gt22[index]);
      PDstandardNth2gt22 = PDstandardNthfdOrder62(&gt22[index]);
      PDstandardNth3gt22 = PDstandardNthfdOrder63(&gt22[index]);
      PDstandardNth11gt22 = PDstandardNthfdOrder611(&gt22[index]);
      PDstandardNth22gt22 = PDstandardNthfdOrder622(&gt22[index]);
      PDstandardNth33gt22 = PDstandardNthfdOrder633(&gt22[index]);
      PDstandardNth12gt22 = PDstandardNthfdOrder612(&gt22[index]);
      PDstandardNth13gt22 = PDstandardNthfdOrder613(&gt22[index]);
      PDstandardNth23gt22 = PDstandardNthfdOrder623(&gt22[index]);
      PDstandardNth1gt23 = PDstandardNthfdOrder61(&gt23[index]);
      PDstandardNth2gt23 = PDstandardNthfdOrder62(&gt23[index]);
      PDstandardNth3gt23 = PDstandardNthfdOrder63(&gt23[index]);
      PDstandardNth11gt23 = PDstandardNthfdOrder611(&gt23[index]);
      PDstandardNth22gt23 = PDstandardNthfdOrder622(&gt23[index]);
      PDstandardNth33gt23 = PDstandardNthfdOrder633(&gt23[index]);
      PDstandardNth12gt23 = PDstandardNthfdOrder612(&gt23[index]);
      PDstandardNth13gt23 = PDstandardNthfdOrder613(&gt23[index]);
      PDstandardNth23gt23 = PDstandardNthfdOrder623(&gt23[index]);
      PDstandardNth1gt33 = PDstandardNthfdOrder61(&gt33[index]);
      PDstandardNth2gt33 = PDstandardNthfdOrder62(&gt33[index]);
      PDstandardNth3gt33 = PDstandardNthfdOrder63(&gt33[index]);
      PDstandardNth11gt33 = PDstandardNthfdOrder611(&gt33[index]);
      PDstandardNth22gt33 = PDstandardNthfdOrder622(&gt33[index]);
      PDstandardNth33gt33 = PDstandardNthfdOrder633(&gt33[index]);
      PDstandardNth12gt33 = PDstandardNthfdOrder612(&gt33[index]);
      PDstandardNth13gt33 = PDstandardNthfdOrder613(&gt33[index]);
      PDstandardNth23gt33 = PDstandardNthfdOrder623(&gt33[index]);
      PDstandardNth1phi = PDstandardNthfdOrder61(&phi[index]);
      PDstandardNth2phi = PDstandardNthfdOrder62(&phi[index]);
      PDstandardNth3phi = PDstandardNthfdOrder63(&phi[index]);
      PDstandardNth11phi = PDstandardNthfdOrder611(&phi[index]);
      PDstandardNth22phi = PDstandardNthfdOrder622(&phi[index]);
      PDstandardNth33phi = PDstandardNthfdOrder633(&phi[index]);
      PDstandardNth12phi = PDstandardNthfdOrder612(&phi[index]);
      PDstandardNth13phi = PDstandardNthfdOrder613(&phi[index]);
      PDstandardNth23phi = PDstandardNthfdOrder623(&phi[index]);
      PDstandardNth1Xt1 = PDstandardNthfdOrder61(&Xt1[index]);
      PDstandardNth2Xt1 = PDstandardNthfdOrder62(&Xt1[index]);
      PDstandardNth3Xt1 = PDstandardNthfdOrder63(&Xt1[index]);
      PDstandardNth1Xt2 = PDstandardNthfdOrder61(&Xt2[index]);
      PDstandardNth2Xt2 = PDstandardNthfdOrder62(&Xt2[index]);
      PDstandardNth3Xt2 = PDstandardNthfdOrder63(&Xt2[index]);
      PDstandardNth1Xt3 = PDstandardNthfdOrder61(&Xt3[index]);
      PDstandardNth2Xt3 = PDstandardNthfdOrder62(&Xt3[index]);
      PDstandardNth3Xt3 = PDstandardNthfdOrder63(&Xt3[index]);
      break;
    
    case 8:
      PDstandardNth1alpha = PDstandardNthfdOrder81(&alpha[index]);
      PDstandardNth2alpha = PDstandardNthfdOrder82(&alpha[index]);
      PDstandardNth3alpha = PDstandardNthfdOrder83(&alpha[index]);
      PDstandardNth11alpha = PDstandardNthfdOrder811(&alpha[index]);
      PDstandardNth22alpha = PDstandardNthfdOrder822(&alpha[index]);
      PDstandardNth33alpha = PDstandardNthfdOrder833(&alpha[index]);
      PDstandardNth12alpha = PDstandardNthfdOrder812(&alpha[index]);
      PDstandardNth13alpha = PDstandardNthfdOrder813(&alpha[index]);
      PDstandardNth23alpha = PDstandardNthfdOrder823(&alpha[index]);
      PDstandardNth1beta1 = PDstandardNthfdOrder81(&beta1[index]);
      PDstandardNth2beta1 = PDstandardNthfdOrder82(&beta1[index]);
      PDstandardNth3beta1 = PDstandardNthfdOrder83(&beta1[index]);
      PDstandardNth1beta2 = PDstandardNthfdOrder81(&beta2[index]);
      PDstandardNth2beta2 = PDstandardNthfdOrder82(&beta2[index]);
      PDstandardNth3beta2 = PDstandardNthfdOrder83(&beta2[index]);
      PDstandardNth1beta3 = PDstandardNthfdOrder81(&beta3[index]);
      PDstandardNth2beta3 = PDstandardNthfdOrder82(&beta3[index]);
      PDstandardNth3beta3 = PDstandardNthfdOrder83(&beta3[index]);
      PDstandardNth1gt11 = PDstandardNthfdOrder81(&gt11[index]);
      PDstandardNth2gt11 = PDstandardNthfdOrder82(&gt11[index]);
      PDstandardNth3gt11 = PDstandardNthfdOrder83(&gt11[index]);
      PDstandardNth11gt11 = PDstandardNthfdOrder811(&gt11[index]);
      PDstandardNth22gt11 = PDstandardNthfdOrder822(&gt11[index]);
      PDstandardNth33gt11 = PDstandardNthfdOrder833(&gt11[index]);
      PDstandardNth12gt11 = PDstandardNthfdOrder812(&gt11[index]);
      PDstandardNth13gt11 = PDstandardNthfdOrder813(&gt11[index]);
      PDstandardNth23gt11 = PDstandardNthfdOrder823(&gt11[index]);
      PDstandardNth1gt12 = PDstandardNthfdOrder81(&gt12[index]);
      PDstandardNth2gt12 = PDstandardNthfdOrder82(&gt12[index]);
      PDstandardNth3gt12 = PDstandardNthfdOrder83(&gt12[index]);
      PDstandardNth11gt12 = PDstandardNthfdOrder811(&gt12[index]);
      PDstandardNth22gt12 = PDstandardNthfdOrder822(&gt12[index]);
      PDstandardNth33gt12 = PDstandardNthfdOrder833(&gt12[index]);
      PDstandardNth12gt12 = PDstandardNthfdOrder812(&gt12[index]);
      PDstandardNth13gt12 = PDstandardNthfdOrder813(&gt12[index]);
      PDstandardNth23gt12 = PDstandardNthfdOrder823(&gt12[index]);
      PDstandardNth1gt13 = PDstandardNthfdOrder81(&gt13[index]);
      PDstandardNth2gt13 = PDstandardNthfdOrder82(&gt13[index]);
      PDstandardNth3gt13 = PDstandardNthfdOrder83(&gt13[index]);
      PDstandardNth11gt13 = PDstandardNthfdOrder811(&gt13[index]);
      PDstandardNth22gt13 = PDstandardNthfdOrder822(&gt13[index]);
      PDstandardNth33gt13 = PDstandardNthfdOrder833(&gt13[index]);
      PDstandardNth12gt13 = PDstandardNthfdOrder812(&gt13[index]);
      PDstandardNth13gt13 = PDstandardNthfdOrder813(&gt13[index]);
      PDstandardNth23gt13 = PDstandardNthfdOrder823(&gt13[index]);
      PDstandardNth1gt22 = PDstandardNthfdOrder81(&gt22[index]);
      PDstandardNth2gt22 = PDstandardNthfdOrder82(&gt22[index]);
      PDstandardNth3gt22 = PDstandardNthfdOrder83(&gt22[index]);
      PDstandardNth11gt22 = PDstandardNthfdOrder811(&gt22[index]);
      PDstandardNth22gt22 = PDstandardNthfdOrder822(&gt22[index]);
      PDstandardNth33gt22 = PDstandardNthfdOrder833(&gt22[index]);
      PDstandardNth12gt22 = PDstandardNthfdOrder812(&gt22[index]);
      PDstandardNth13gt22 = PDstandardNthfdOrder813(&gt22[index]);
      PDstandardNth23gt22 = PDstandardNthfdOrder823(&gt22[index]);
      PDstandardNth1gt23 = PDstandardNthfdOrder81(&gt23[index]);
      PDstandardNth2gt23 = PDstandardNthfdOrder82(&gt23[index]);
      PDstandardNth3gt23 = PDstandardNthfdOrder83(&gt23[index]);
      PDstandardNth11gt23 = PDstandardNthfdOrder811(&gt23[index]);
      PDstandardNth22gt23 = PDstandardNthfdOrder822(&gt23[index]);
      PDstandardNth33gt23 = PDstandardNthfdOrder833(&gt23[index]);
      PDstandardNth12gt23 = PDstandardNthfdOrder812(&gt23[index]);
      PDstandardNth13gt23 = PDstandardNthfdOrder813(&gt23[index]);
      PDstandardNth23gt23 = PDstandardNthfdOrder823(&gt23[index]);
      PDstandardNth1gt33 = PDstandardNthfdOrder81(&gt33[index]);
      PDstandardNth2gt33 = PDstandardNthfdOrder82(&gt33[index]);
      PDstandardNth3gt33 = PDstandardNthfdOrder83(&gt33[index]);
      PDstandardNth11gt33 = PDstandardNthfdOrder811(&gt33[index]);
      PDstandardNth22gt33 = PDstandardNthfdOrder822(&gt33[index]);
      PDstandardNth33gt33 = PDstandardNthfdOrder833(&gt33[index]);
      PDstandardNth12gt33 = PDstandardNthfdOrder812(&gt33[index]);
      PDstandardNth13gt33 = PDstandardNthfdOrder813(&gt33[index]);
      PDstandardNth23gt33 = PDstandardNthfdOrder823(&gt33[index]);
      PDstandardNth1phi = PDstandardNthfdOrder81(&phi[index]);
      PDstandardNth2phi = PDstandardNthfdOrder82(&phi[index]);
      PDstandardNth3phi = PDstandardNthfdOrder83(&phi[index]);
      PDstandardNth11phi = PDstandardNthfdOrder811(&phi[index]);
      PDstandardNth22phi = PDstandardNthfdOrder822(&phi[index]);
      PDstandardNth33phi = PDstandardNthfdOrder833(&phi[index]);
      PDstandardNth12phi = PDstandardNthfdOrder812(&phi[index]);
      PDstandardNth13phi = PDstandardNthfdOrder813(&phi[index]);
      PDstandardNth23phi = PDstandardNthfdOrder823(&phi[index]);
      PDstandardNth1Xt1 = PDstandardNthfdOrder81(&Xt1[index]);
      PDstandardNth2Xt1 = PDstandardNthfdOrder82(&Xt1[index]);
      PDstandardNth3Xt1 = PDstandardNthfdOrder83(&Xt1[index]);
      PDstandardNth1Xt2 = PDstandardNthfdOrder81(&Xt2[index]);
      PDstandardNth2Xt2 = PDstandardNthfdOrder82(&Xt2[index]);
      PDstandardNth3Xt2 = PDstandardNthfdOrder83(&Xt2[index]);
      PDstandardNth1Xt3 = PDstandardNthfdOrder81(&Xt3[index]);
      PDstandardNth2Xt3 = PDstandardNthfdOrder82(&Xt3[index]);
      PDstandardNth3Xt3 = PDstandardNthfdOrder83(&Xt3[index]);
      break;
  }
  
  /* Calculate temporaries and grid functions */
  CCTK_REAL_VEC JacPDstandardNth11alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth11gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth11gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth11gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth11gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth11gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth11gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth11phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth12alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth12gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth12gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth12gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth12gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth12gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth12gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth12phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth13alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth13gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth13gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth13gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth13gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth13gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth13gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth13phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1beta1 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1beta2 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1beta3 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1Xt1 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1Xt2 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth1Xt3 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth21gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth21gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth21gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth21gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth21gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth21gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth22alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth22gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth22gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth22gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth22gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth22gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth22gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth22phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth23alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth23gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth23gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth23gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth23gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth23gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth23gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth23phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2beta1 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2beta2 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2beta3 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2Xt1 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2Xt2 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth2Xt3 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth31gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth31gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth31gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth31gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth31gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth31gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth32gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth32gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth32gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth32gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth32gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth32gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth33alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth33gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth33gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth33gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth33gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth33gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth33gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth33phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3alpha CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3beta1 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3beta2 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3beta3 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3gt11 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3gt12 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3gt13 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3gt22 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3gt23 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3gt33 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3phi CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3Xt1 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3Xt2 CCTK_ATTRIBUTE_UNUSED ;
  CCTK_REAL_VEC JacPDstandardNth3Xt3 CCTK_ATTRIBUTE_UNUSED ;
  
  if (use_jacobian)
  {
    JacPDstandardNth1alpha = 
      kmadd(J11L,PDstandardNth1alpha,kmadd(J21L,PDstandardNth2alpha,kmul(J31L,PDstandardNth3alpha)));
    
    JacPDstandardNth1beta1 = 
      kmadd(J11L,PDstandardNth1beta1,kmadd(J21L,PDstandardNth2beta1,kmul(J31L,PDstandardNth3beta1)));
    
    JacPDstandardNth1beta2 = 
      kmadd(J11L,PDstandardNth1beta2,kmadd(J21L,PDstandardNth2beta2,kmul(J31L,PDstandardNth3beta2)));
    
    JacPDstandardNth1beta3 = 
      kmadd(J11L,PDstandardNth1beta3,kmadd(J21L,PDstandardNth2beta3,kmul(J31L,PDstandardNth3beta3)));
    
    JacPDstandardNth1gt11 = 
      kmadd(J11L,PDstandardNth1gt11,kmadd(J21L,PDstandardNth2gt11,kmul(J31L,PDstandardNth3gt11)));
    
    JacPDstandardNth1gt12 = 
      kmadd(J11L,PDstandardNth1gt12,kmadd(J21L,PDstandardNth2gt12,kmul(J31L,PDstandardNth3gt12)));
    
    JacPDstandardNth1gt13 = 
      kmadd(J11L,PDstandardNth1gt13,kmadd(J21L,PDstandardNth2gt13,kmul(J31L,PDstandardNth3gt13)));
    
    JacPDstandardNth1gt22 = 
      kmadd(J11L,PDstandardNth1gt22,kmadd(J21L,PDstandardNth2gt22,kmul(J31L,PDstandardNth3gt22)));
    
    JacPDstandardNth1gt23 = 
      kmadd(J11L,PDstandardNth1gt23,kmadd(J21L,PDstandardNth2gt23,kmul(J31L,PDstandardNth3gt23)));
    
    JacPDstandardNth1gt33 = 
      kmadd(J11L,PDstandardNth1gt33,kmadd(J21L,PDstandardNth2gt33,kmul(J31L,PDstandardNth3gt33)));
    
    JacPDstandardNth1phi = 
      kmadd(J11L,PDstandardNth1phi,kmadd(J21L,PDstandardNth2phi,kmul(J31L,PDstandardNth3phi)));
    
    JacPDstandardNth1Xt1 = 
      kmadd(J11L,PDstandardNth1Xt1,kmadd(J21L,PDstandardNth2Xt1,kmul(J31L,PDstandardNth3Xt1)));
    
    JacPDstandardNth1Xt2 = 
      kmadd(J11L,PDstandardNth1Xt2,kmadd(J21L,PDstandardNth2Xt2,kmul(J31L,PDstandardNth3Xt2)));
    
    JacPDstandardNth1Xt3 = 
      kmadd(J11L,PDstandardNth1Xt3,kmadd(J21L,PDstandardNth2Xt3,kmul(J31L,PDstandardNth3Xt3)));
    
    JacPDstandardNth2alpha = 
      kmadd(J12L,PDstandardNth1alpha,kmadd(J22L,PDstandardNth2alpha,kmul(J32L,PDstandardNth3alpha)));
    
    JacPDstandardNth2beta1 = 
      kmadd(J12L,PDstandardNth1beta1,kmadd(J22L,PDstandardNth2beta1,kmul(J32L,PDstandardNth3beta1)));
    
    JacPDstandardNth2beta2 = 
      kmadd(J12L,PDstandardNth1beta2,kmadd(J22L,PDstandardNth2beta2,kmul(J32L,PDstandardNth3beta2)));
    
    JacPDstandardNth2beta3 = 
      kmadd(J12L,PDstandardNth1beta3,kmadd(J22L,PDstandardNth2beta3,kmul(J32L,PDstandardNth3beta3)));
    
    JacPDstandardNth2gt11 = 
      kmadd(J12L,PDstandardNth1gt11,kmadd(J22L,PDstandardNth2gt11,kmul(J32L,PDstandardNth3gt11)));
    
    JacPDstandardNth2gt12 = 
      kmadd(J12L,PDstandardNth1gt12,kmadd(J22L,PDstandardNth2gt12,kmul(J32L,PDstandardNth3gt12)));
    
    JacPDstandardNth2gt13 = 
      kmadd(J12L,PDstandardNth1gt13,kmadd(J22L,PDstandardNth2gt13,kmul(J32L,PDstandardNth3gt13)));
    
    JacPDstandardNth2gt22 = 
      kmadd(J12L,PDstandardNth1gt22,kmadd(J22L,PDstandardNth2gt22,kmul(J32L,PDstandardNth3gt22)));
    
    JacPDstandardNth2gt23 = 
      kmadd(J12L,PDstandardNth1gt23,kmadd(J22L,PDstandardNth2gt23,kmul(J32L,PDstandardNth3gt23)));
    
    JacPDstandardNth2gt33 = 
      kmadd(J12L,PDstandardNth1gt33,kmadd(J22L,PDstandardNth2gt33,kmul(J32L,PDstandardNth3gt33)));
    
    JacPDstandardNth2phi = 
      kmadd(J12L,PDstandardNth1phi,kmadd(J22L,PDstandardNth2phi,kmul(J32L,PDstandardNth3phi)));
    
    JacPDstandardNth2Xt1 = 
      kmadd(J12L,PDstandardNth1Xt1,kmadd(J22L,PDstandardNth2Xt1,kmul(J32L,PDstandardNth3Xt1)));
    
    JacPDstandardNth2Xt2 = 
      kmadd(J12L,PDstandardNth1Xt2,kmadd(J22L,PDstandardNth2Xt2,kmul(J32L,PDstandardNth3Xt2)));
    
    JacPDstandardNth2Xt3 = 
      kmadd(J12L,PDstandardNth1Xt3,kmadd(J22L,PDstandardNth2Xt3,kmul(J32L,PDstandardNth3Xt3)));
    
    JacPDstandardNth3alpha = 
      kmadd(J13L,PDstandardNth1alpha,kmadd(J23L,PDstandardNth2alpha,kmul(J33L,PDstandardNth3alpha)));
    
    JacPDstandardNth3beta1 = 
      kmadd(J13L,PDstandardNth1beta1,kmadd(J23L,PDstandardNth2beta1,kmul(J33L,PDstandardNth3beta1)));
    
    JacPDstandardNth3beta2 = 
      kmadd(J13L,PDstandardNth1beta2,kmadd(J23L,PDstandardNth2beta2,kmul(J33L,PDstandardNth3beta2)));
    
    JacPDstandardNth3beta3 = 
      kmadd(J13L,PDstandardNth1beta3,kmadd(J23L,PDstandardNth2beta3,kmul(J33L,PDstandardNth3beta3)));
    
    JacPDstandardNth3gt11 = 
      kmadd(J13L,PDstandardNth1gt11,kmadd(J23L,PDstandardNth2gt11,kmul(J33L,PDstandardNth3gt11)));
    
    JacPDstandardNth3gt12 = 
      kmadd(J13L,PDstandardNth1gt12,kmadd(J23L,PDstandardNth2gt12,kmul(J33L,PDstandardNth3gt12)));
    
    JacPDstandardNth3gt13 = 
      kmadd(J13L,PDstandardNth1gt13,kmadd(J23L,PDstandardNth2gt13,kmul(J33L,PDstandardNth3gt13)));
    
    JacPDstandardNth3gt22 = 
      kmadd(J13L,PDstandardNth1gt22,kmadd(J23L,PDstandardNth2gt22,kmul(J33L,PDstandardNth3gt22)));
    
    JacPDstandardNth3gt23 = 
      kmadd(J13L,PDstandardNth1gt23,kmadd(J23L,PDstandardNth2gt23,kmul(J33L,PDstandardNth3gt23)));
    
    JacPDstandardNth3gt33 = 
      kmadd(J13L,PDstandardNth1gt33,kmadd(J23L,PDstandardNth2gt33,kmul(J33L,PDstandardNth3gt33)));
    
    JacPDstandardNth3phi = 
      kmadd(J13L,PDstandardNth1phi,kmadd(J23L,PDstandardNth2phi,kmul(J33L,PDstandardNth3phi)));
    
    JacPDstandardNth3Xt1 = 
      kmadd(J13L,PDstandardNth1Xt1,kmadd(J23L,PDstandardNth2Xt1,kmul(J33L,PDstandardNth3Xt1)));
    
    JacPDstandardNth3Xt2 = 
      kmadd(J13L,PDstandardNth1Xt2,kmadd(J23L,PDstandardNth2Xt2,kmul(J33L,PDstandardNth3Xt2)));
    
    JacPDstandardNth3Xt3 = 
      kmadd(J13L,PDstandardNth1Xt3,kmadd(J23L,PDstandardNth2Xt3,kmul(J33L,PDstandardNth3Xt3)));
    
    JacPDstandardNth11alpha = 
      kmadd(dJ111L,PDstandardNth1alpha,kmadd(dJ211L,PDstandardNth2alpha,kmadd(dJ311L,PDstandardNth3alpha,kmadd(PDstandardNth11alpha,kmul(J11L,J11L),kmadd(PDstandardNth22alpha,kmul(J21L,J21L),kmadd(PDstandardNth33alpha,kmul(J31L,J31L),kmul(kmadd(J11L,kmadd(J21L,PDstandardNth12alpha,kmul(J31L,PDstandardNth13alpha)),kmul(J21L,kmul(J31L,PDstandardNth23alpha))),ToReal(2))))))));
    
    JacPDstandardNth11gt11 = 
      kmadd(dJ111L,PDstandardNth1gt11,kmadd(dJ211L,PDstandardNth2gt11,kmadd(dJ311L,PDstandardNth3gt11,kmadd(PDstandardNth11gt11,kmul(J11L,J11L),kmadd(PDstandardNth22gt11,kmul(J21L,J21L),kmadd(PDstandardNth33gt11,kmul(J31L,J31L),kmul(kmadd(J11L,kmadd(J21L,PDstandardNth12gt11,kmul(J31L,PDstandardNth13gt11)),kmul(J21L,kmul(J31L,PDstandardNth23gt11))),ToReal(2))))))));
    
    JacPDstandardNth11gt12 = 
      kmadd(dJ111L,PDstandardNth1gt12,kmadd(dJ211L,PDstandardNth2gt12,kmadd(dJ311L,PDstandardNth3gt12,kmadd(PDstandardNth11gt12,kmul(J11L,J11L),kmadd(PDstandardNth22gt12,kmul(J21L,J21L),kmadd(PDstandardNth33gt12,kmul(J31L,J31L),kmul(kmadd(J11L,kmadd(J21L,PDstandardNth12gt12,kmul(J31L,PDstandardNth13gt12)),kmul(J21L,kmul(J31L,PDstandardNth23gt12))),ToReal(2))))))));
    
    JacPDstandardNth11gt13 = 
      kmadd(dJ111L,PDstandardNth1gt13,kmadd(dJ211L,PDstandardNth2gt13,kmadd(dJ311L,PDstandardNth3gt13,kmadd(PDstandardNth11gt13,kmul(J11L,J11L),kmadd(PDstandardNth22gt13,kmul(J21L,J21L),kmadd(PDstandardNth33gt13,kmul(J31L,J31L),kmul(kmadd(J11L,kmadd(J21L,PDstandardNth12gt13,kmul(J31L,PDstandardNth13gt13)),kmul(J21L,kmul(J31L,PDstandardNth23gt13))),ToReal(2))))))));
    
    JacPDstandardNth11gt22 = 
      kmadd(dJ111L,PDstandardNth1gt22,kmadd(dJ211L,PDstandardNth2gt22,kmadd(dJ311L,PDstandardNth3gt22,kmadd(PDstandardNth11gt22,kmul(J11L,J11L),kmadd(PDstandardNth22gt22,kmul(J21L,J21L),kmadd(PDstandardNth33gt22,kmul(J31L,J31L),kmul(kmadd(J11L,kmadd(J21L,PDstandardNth12gt22,kmul(J31L,PDstandardNth13gt22)),kmul(J21L,kmul(J31L,PDstandardNth23gt22))),ToReal(2))))))));
    
    JacPDstandardNth11gt23 = 
      kmadd(dJ111L,PDstandardNth1gt23,kmadd(dJ211L,PDstandardNth2gt23,kmadd(dJ311L,PDstandardNth3gt23,kmadd(PDstandardNth11gt23,kmul(J11L,J11L),kmadd(PDstandardNth22gt23,kmul(J21L,J21L),kmadd(PDstandardNth33gt23,kmul(J31L,J31L),kmul(kmadd(J11L,kmadd(J21L,PDstandardNth12gt23,kmul(J31L,PDstandardNth13gt23)),kmul(J21L,kmul(J31L,PDstandardNth23gt23))),ToReal(2))))))));
    
    JacPDstandardNth11gt33 = 
      kmadd(dJ111L,PDstandardNth1gt33,kmadd(dJ211L,PDstandardNth2gt33,kmadd(dJ311L,PDstandardNth3gt33,kmadd(PDstandardNth11gt33,kmul(J11L,J11L),kmadd(PDstandardNth22gt33,kmul(J21L,J21L),kmadd(PDstandardNth33gt33,kmul(J31L,J31L),kmul(kmadd(J11L,kmadd(J21L,PDstandardNth12gt33,kmul(J31L,PDstandardNth13gt33)),kmul(J21L,kmul(J31L,PDstandardNth23gt33))),ToReal(2))))))));
    
    JacPDstandardNth11phi = 
      kmadd(dJ111L,PDstandardNth1phi,kmadd(dJ211L,PDstandardNth2phi,kmadd(dJ311L,PDstandardNth3phi,kmadd(PDstandardNth11phi,kmul(J11L,J11L),kmadd(PDstandardNth22phi,kmul(J21L,J21L),kmadd(PDstandardNth33phi,kmul(J31L,J31L),kmul(kmadd(J11L,kmadd(J21L,PDstandardNth12phi,kmul(J31L,PDstandardNth13phi)),kmul(J21L,kmul(J31L,PDstandardNth23phi))),ToReal(2))))))));
    
    JacPDstandardNth22alpha = 
      kmadd(dJ122L,PDstandardNth1alpha,kmadd(dJ222L,PDstandardNth2alpha,kmadd(dJ322L,PDstandardNth3alpha,kmadd(PDstandardNth11alpha,kmul(J12L,J12L),kmadd(PDstandardNth22alpha,kmul(J22L,J22L),kmadd(PDstandardNth33alpha,kmul(J32L,J32L),kmul(kmadd(J12L,kmadd(J22L,PDstandardNth12alpha,kmul(J32L,PDstandardNth13alpha)),kmul(J22L,kmul(J32L,PDstandardNth23alpha))),ToReal(2))))))));
    
    JacPDstandardNth22gt11 = 
      kmadd(dJ122L,PDstandardNth1gt11,kmadd(dJ222L,PDstandardNth2gt11,kmadd(dJ322L,PDstandardNth3gt11,kmadd(PDstandardNth11gt11,kmul(J12L,J12L),kmadd(PDstandardNth22gt11,kmul(J22L,J22L),kmadd(PDstandardNth33gt11,kmul(J32L,J32L),kmul(kmadd(J12L,kmadd(J22L,PDstandardNth12gt11,kmul(J32L,PDstandardNth13gt11)),kmul(J22L,kmul(J32L,PDstandardNth23gt11))),ToReal(2))))))));
    
    JacPDstandardNth22gt12 = 
      kmadd(dJ122L,PDstandardNth1gt12,kmadd(dJ222L,PDstandardNth2gt12,kmadd(dJ322L,PDstandardNth3gt12,kmadd(PDstandardNth11gt12,kmul(J12L,J12L),kmadd(PDstandardNth22gt12,kmul(J22L,J22L),kmadd(PDstandardNth33gt12,kmul(J32L,J32L),kmul(kmadd(J12L,kmadd(J22L,PDstandardNth12gt12,kmul(J32L,PDstandardNth13gt12)),kmul(J22L,kmul(J32L,PDstandardNth23gt12))),ToReal(2))))))));
    
    JacPDstandardNth22gt13 = 
      kmadd(dJ122L,PDstandardNth1gt13,kmadd(dJ222L,PDstandardNth2gt13,kmadd(dJ322L,PDstandardNth3gt13,kmadd(PDstandardNth11gt13,kmul(J12L,J12L),kmadd(PDstandardNth22gt13,kmul(J22L,J22L),kmadd(PDstandardNth33gt13,kmul(J32L,J32L),kmul(kmadd(J12L,kmadd(J22L,PDstandardNth12gt13,kmul(J32L,PDstandardNth13gt13)),kmul(J22L,kmul(J32L,PDstandardNth23gt13))),ToReal(2))))))));
    
    JacPDstandardNth22gt22 = 
      kmadd(dJ122L,PDstandardNth1gt22,kmadd(dJ222L,PDstandardNth2gt22,kmadd(dJ322L,PDstandardNth3gt22,kmadd(PDstandardNth11gt22,kmul(J12L,J12L),kmadd(PDstandardNth22gt22,kmul(J22L,J22L),kmadd(PDstandardNth33gt22,kmul(J32L,J32L),kmul(kmadd(J12L,kmadd(J22L,PDstandardNth12gt22,kmul(J32L,PDstandardNth13gt22)),kmul(J22L,kmul(J32L,PDstandardNth23gt22))),ToReal(2))))))));
    
    JacPDstandardNth22gt23 = 
      kmadd(dJ122L,PDstandardNth1gt23,kmadd(dJ222L,PDstandardNth2gt23,kmadd(dJ322L,PDstandardNth3gt23,kmadd(PDstandardNth11gt23,kmul(J12L,J12L),kmadd(PDstandardNth22gt23,kmul(J22L,J22L),kmadd(PDstandardNth33gt23,kmul(J32L,J32L),kmul(kmadd(J12L,kmadd(J22L,PDstandardNth12gt23,kmul(J32L,PDstandardNth13gt23)),kmul(J22L,kmul(J32L,PDstandardNth23gt23))),ToReal(2))))))));
    
    JacPDstandardNth22gt33 = 
      kmadd(dJ122L,PDstandardNth1gt33,kmadd(dJ222L,PDstandardNth2gt33,kmadd(dJ322L,PDstandardNth3gt33,kmadd(PDstandardNth11gt33,kmul(J12L,J12L),kmadd(PDstandardNth22gt33,kmul(J22L,J22L),kmadd(PDstandardNth33gt33,kmul(J32L,J32L),kmul(kmadd(J12L,kmadd(J22L,PDstandardNth12gt33,kmul(J32L,PDstandardNth13gt33)),kmul(J22L,kmul(J32L,PDstandardNth23gt33))),ToReal(2))))))));
    
    JacPDstandardNth22phi = 
      kmadd(dJ122L,PDstandardNth1phi,kmadd(dJ222L,PDstandardNth2phi,kmadd(dJ322L,PDstandardNth3phi,kmadd(PDstandardNth11phi,kmul(J12L,J12L),kmadd(PDstandardNth22phi,kmul(J22L,J22L),kmadd(PDstandardNth33phi,kmul(J32L,J32L),kmul(kmadd(J12L,kmadd(J22L,PDstandardNth12phi,kmul(J32L,PDstandardNth13phi)),kmul(J22L,kmul(J32L,PDstandardNth23phi))),ToReal(2))))))));
    
    JacPDstandardNth33alpha = 
      kmadd(dJ133L,PDstandardNth1alpha,kmadd(dJ233L,PDstandardNth2alpha,kmadd(dJ333L,PDstandardNth3alpha,kmadd(PDstandardNth11alpha,kmul(J13L,J13L),kmadd(PDstandardNth22alpha,kmul(J23L,J23L),kmadd(PDstandardNth33alpha,kmul(J33L,J33L),kmul(kmadd(J13L,kmadd(J23L,PDstandardNth12alpha,kmul(J33L,PDstandardNth13alpha)),kmul(J23L,kmul(J33L,PDstandardNth23alpha))),ToReal(2))))))));
    
    JacPDstandardNth33gt11 = 
      kmadd(dJ133L,PDstandardNth1gt11,kmadd(dJ233L,PDstandardNth2gt11,kmadd(dJ333L,PDstandardNth3gt11,kmadd(PDstandardNth11gt11,kmul(J13L,J13L),kmadd(PDstandardNth22gt11,kmul(J23L,J23L),kmadd(PDstandardNth33gt11,kmul(J33L,J33L),kmul(kmadd(J13L,kmadd(J23L,PDstandardNth12gt11,kmul(J33L,PDstandardNth13gt11)),kmul(J23L,kmul(J33L,PDstandardNth23gt11))),ToReal(2))))))));
    
    JacPDstandardNth33gt12 = 
      kmadd(dJ133L,PDstandardNth1gt12,kmadd(dJ233L,PDstandardNth2gt12,kmadd(dJ333L,PDstandardNth3gt12,kmadd(PDstandardNth11gt12,kmul(J13L,J13L),kmadd(PDstandardNth22gt12,kmul(J23L,J23L),kmadd(PDstandardNth33gt12,kmul(J33L,J33L),kmul(kmadd(J13L,kmadd(J23L,PDstandardNth12gt12,kmul(J33L,PDstandardNth13gt12)),kmul(J23L,kmul(J33L,PDstandardNth23gt12))),ToReal(2))))))));
    
    JacPDstandardNth33gt13 = 
      kmadd(dJ133L,PDstandardNth1gt13,kmadd(dJ233L,PDstandardNth2gt13,kmadd(dJ333L,PDstandardNth3gt13,kmadd(PDstandardNth11gt13,kmul(J13L,J13L),kmadd(PDstandardNth22gt13,kmul(J23L,J23L),kmadd(PDstandardNth33gt13,kmul(J33L,J33L),kmul(kmadd(J13L,kmadd(J23L,PDstandardNth12gt13,kmul(J33L,PDstandardNth13gt13)),kmul(J23L,kmul(J33L,PDstandardNth23gt13))),ToReal(2))))))));
    
    JacPDstandardNth33gt22 = 
      kmadd(dJ133L,PDstandardNth1gt22,kmadd(dJ233L,PDstandardNth2gt22,kmadd(dJ333L,PDstandardNth3gt22,kmadd(PDstandardNth11gt22,kmul(J13L,J13L),kmadd(PDstandardNth22gt22,kmul(J23L,J23L),kmadd(PDstandardNth33gt22,kmul(J33L,J33L),kmul(kmadd(J13L,kmadd(J23L,PDstandardNth12gt22,kmul(J33L,PDstandardNth13gt22)),kmul(J23L,kmul(J33L,PDstandardNth23gt22))),ToReal(2))))))));
    
    JacPDstandardNth33gt23 = 
      kmadd(dJ133L,PDstandardNth1gt23,kmadd(dJ233L,PDstandardNth2gt23,kmadd(dJ333L,PDstandardNth3gt23,kmadd(PDstandardNth11gt23,kmul(J13L,J13L),kmadd(PDstandardNth22gt23,kmul(J23L,J23L),kmadd(PDstandardNth33gt23,kmul(J33L,J33L),kmul(kmadd(J13L,kmadd(J23L,PDstandardNth12gt23,kmul(J33L,PDstandardNth13gt23)),kmul(J23L,kmul(J33L,PDstandardNth23gt23))),ToReal(2))))))));
    
    JacPDstandardNth33gt33 = 
      kmadd(dJ133L,PDstandardNth1gt33,kmadd(dJ233L,PDstandardNth2gt33,kmadd(dJ333L,PDstandardNth3gt33,kmadd(PDstandardNth11gt33,kmul(J13L,J13L),kmadd(PDstandardNth22gt33,kmul(J23L,J23L),kmadd(PDstandardNth33gt33,kmul(J33L,J33L),kmul(kmadd(J13L,kmadd(J23L,PDstandardNth12gt33,kmul(J33L,PDstandardNth13gt33)),kmul(J23L,kmul(J33L,PDstandardNth23gt33))),ToReal(2))))))));
    
    JacPDstandardNth33phi = 
      kmadd(dJ133L,PDstandardNth1phi,kmadd(dJ233L,PDstandardNth2phi,kmadd(dJ333L,PDstandardNth3phi,kmadd(PDstandardNth11phi,kmul(J13L,J13L),kmadd(PDstandardNth22phi,kmul(J23L,J23L),kmadd(PDstandardNth33phi,kmul(J33L,J33L),kmul(kmadd(J13L,kmadd(J23L,PDstandardNth12phi,kmul(J33L,PDstandardNth13phi)),kmul(J23L,kmul(J33L,PDstandardNth23phi))),ToReal(2))))))));
    
    JacPDstandardNth12alpha = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11alpha,kmadd(J21L,PDstandardNth12alpha,kmul(J31L,PDstandardNth13alpha))),kmadd(J11L,kmadd(J22L,PDstandardNth12alpha,kmul(J32L,PDstandardNth13alpha)),kmadd(dJ112L,PDstandardNth1alpha,kmadd(J22L,kmadd(J21L,PDstandardNth22alpha,kmul(J31L,PDstandardNth23alpha)),kmadd(dJ212L,PDstandardNth2alpha,kmadd(J32L,kmadd(J21L,PDstandardNth23alpha,kmul(J31L,PDstandardNth33alpha)),kmul(dJ312L,PDstandardNth3alpha)))))));
    
    JacPDstandardNth12gt11 = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11gt11,kmadd(J21L,PDstandardNth12gt11,kmul(J31L,PDstandardNth13gt11))),kmadd(J11L,kmadd(J22L,PDstandardNth12gt11,kmul(J32L,PDstandardNth13gt11)),kmadd(dJ112L,PDstandardNth1gt11,kmadd(J22L,kmadd(J21L,PDstandardNth22gt11,kmul(J31L,PDstandardNth23gt11)),kmadd(dJ212L,PDstandardNth2gt11,kmadd(J32L,kmadd(J21L,PDstandardNth23gt11,kmul(J31L,PDstandardNth33gt11)),kmul(dJ312L,PDstandardNth3gt11)))))));
    
    JacPDstandardNth12gt12 = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11gt12,kmadd(J21L,PDstandardNth12gt12,kmul(J31L,PDstandardNth13gt12))),kmadd(J11L,kmadd(J22L,PDstandardNth12gt12,kmul(J32L,PDstandardNth13gt12)),kmadd(dJ112L,PDstandardNth1gt12,kmadd(J22L,kmadd(J21L,PDstandardNth22gt12,kmul(J31L,PDstandardNth23gt12)),kmadd(dJ212L,PDstandardNth2gt12,kmadd(J32L,kmadd(J21L,PDstandardNth23gt12,kmul(J31L,PDstandardNth33gt12)),kmul(dJ312L,PDstandardNth3gt12)))))));
    
    JacPDstandardNth12gt13 = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11gt13,kmadd(J21L,PDstandardNth12gt13,kmul(J31L,PDstandardNth13gt13))),kmadd(J11L,kmadd(J22L,PDstandardNth12gt13,kmul(J32L,PDstandardNth13gt13)),kmadd(dJ112L,PDstandardNth1gt13,kmadd(J22L,kmadd(J21L,PDstandardNth22gt13,kmul(J31L,PDstandardNth23gt13)),kmadd(dJ212L,PDstandardNth2gt13,kmadd(J32L,kmadd(J21L,PDstandardNth23gt13,kmul(J31L,PDstandardNth33gt13)),kmul(dJ312L,PDstandardNth3gt13)))))));
    
    JacPDstandardNth12gt22 = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11gt22,kmadd(J21L,PDstandardNth12gt22,kmul(J31L,PDstandardNth13gt22))),kmadd(J11L,kmadd(J22L,PDstandardNth12gt22,kmul(J32L,PDstandardNth13gt22)),kmadd(dJ112L,PDstandardNth1gt22,kmadd(J22L,kmadd(J21L,PDstandardNth22gt22,kmul(J31L,PDstandardNth23gt22)),kmadd(dJ212L,PDstandardNth2gt22,kmadd(J32L,kmadd(J21L,PDstandardNth23gt22,kmul(J31L,PDstandardNth33gt22)),kmul(dJ312L,PDstandardNth3gt22)))))));
    
    JacPDstandardNth12gt23 = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11gt23,kmadd(J21L,PDstandardNth12gt23,kmul(J31L,PDstandardNth13gt23))),kmadd(J11L,kmadd(J22L,PDstandardNth12gt23,kmul(J32L,PDstandardNth13gt23)),kmadd(dJ112L,PDstandardNth1gt23,kmadd(J22L,kmadd(J21L,PDstandardNth22gt23,kmul(J31L,PDstandardNth23gt23)),kmadd(dJ212L,PDstandardNth2gt23,kmadd(J32L,kmadd(J21L,PDstandardNth23gt23,kmul(J31L,PDstandardNth33gt23)),kmul(dJ312L,PDstandardNth3gt23)))))));
    
    JacPDstandardNth12gt33 = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11gt33,kmadd(J21L,PDstandardNth12gt33,kmul(J31L,PDstandardNth13gt33))),kmadd(J11L,kmadd(J22L,PDstandardNth12gt33,kmul(J32L,PDstandardNth13gt33)),kmadd(dJ112L,PDstandardNth1gt33,kmadd(J22L,kmadd(J21L,PDstandardNth22gt33,kmul(J31L,PDstandardNth23gt33)),kmadd(dJ212L,PDstandardNth2gt33,kmadd(J32L,kmadd(J21L,PDstandardNth23gt33,kmul(J31L,PDstandardNth33gt33)),kmul(dJ312L,PDstandardNth3gt33)))))));
    
    JacPDstandardNth12phi = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11phi,kmadd(J21L,PDstandardNth12phi,kmul(J31L,PDstandardNth13phi))),kmadd(J11L,kmadd(J22L,PDstandardNth12phi,kmul(J32L,PDstandardNth13phi)),kmadd(dJ112L,PDstandardNth1phi,kmadd(J22L,kmadd(J21L,PDstandardNth22phi,kmul(J31L,PDstandardNth23phi)),kmadd(dJ212L,PDstandardNth2phi,kmadd(J32L,kmadd(J21L,PDstandardNth23phi,kmul(J31L,PDstandardNth33phi)),kmul(dJ312L,PDstandardNth3phi)))))));
    
    JacPDstandardNth13alpha = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11alpha,kmadd(J21L,PDstandardNth12alpha,kmul(J31L,PDstandardNth13alpha))),kmadd(J11L,kmadd(J23L,PDstandardNth12alpha,kmul(J33L,PDstandardNth13alpha)),kmadd(dJ113L,PDstandardNth1alpha,kmadd(J23L,kmadd(J21L,PDstandardNth22alpha,kmul(J31L,PDstandardNth23alpha)),kmadd(dJ213L,PDstandardNth2alpha,kmadd(J33L,kmadd(J21L,PDstandardNth23alpha,kmul(J31L,PDstandardNth33alpha)),kmul(dJ313L,PDstandardNth3alpha)))))));
    
    JacPDstandardNth13gt11 = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11gt11,kmadd(J21L,PDstandardNth12gt11,kmul(J31L,PDstandardNth13gt11))),kmadd(J11L,kmadd(J23L,PDstandardNth12gt11,kmul(J33L,PDstandardNth13gt11)),kmadd(dJ113L,PDstandardNth1gt11,kmadd(J23L,kmadd(J21L,PDstandardNth22gt11,kmul(J31L,PDstandardNth23gt11)),kmadd(dJ213L,PDstandardNth2gt11,kmadd(J33L,kmadd(J21L,PDstandardNth23gt11,kmul(J31L,PDstandardNth33gt11)),kmul(dJ313L,PDstandardNth3gt11)))))));
    
    JacPDstandardNth13gt12 = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11gt12,kmadd(J21L,PDstandardNth12gt12,kmul(J31L,PDstandardNth13gt12))),kmadd(J11L,kmadd(J23L,PDstandardNth12gt12,kmul(J33L,PDstandardNth13gt12)),kmadd(dJ113L,PDstandardNth1gt12,kmadd(J23L,kmadd(J21L,PDstandardNth22gt12,kmul(J31L,PDstandardNth23gt12)),kmadd(dJ213L,PDstandardNth2gt12,kmadd(J33L,kmadd(J21L,PDstandardNth23gt12,kmul(J31L,PDstandardNth33gt12)),kmul(dJ313L,PDstandardNth3gt12)))))));
    
    JacPDstandardNth13gt13 = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11gt13,kmadd(J21L,PDstandardNth12gt13,kmul(J31L,PDstandardNth13gt13))),kmadd(J11L,kmadd(J23L,PDstandardNth12gt13,kmul(J33L,PDstandardNth13gt13)),kmadd(dJ113L,PDstandardNth1gt13,kmadd(J23L,kmadd(J21L,PDstandardNth22gt13,kmul(J31L,PDstandardNth23gt13)),kmadd(dJ213L,PDstandardNth2gt13,kmadd(J33L,kmadd(J21L,PDstandardNth23gt13,kmul(J31L,PDstandardNth33gt13)),kmul(dJ313L,PDstandardNth3gt13)))))));
    
    JacPDstandardNth13gt22 = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11gt22,kmadd(J21L,PDstandardNth12gt22,kmul(J31L,PDstandardNth13gt22))),kmadd(J11L,kmadd(J23L,PDstandardNth12gt22,kmul(J33L,PDstandardNth13gt22)),kmadd(dJ113L,PDstandardNth1gt22,kmadd(J23L,kmadd(J21L,PDstandardNth22gt22,kmul(J31L,PDstandardNth23gt22)),kmadd(dJ213L,PDstandardNth2gt22,kmadd(J33L,kmadd(J21L,PDstandardNth23gt22,kmul(J31L,PDstandardNth33gt22)),kmul(dJ313L,PDstandardNth3gt22)))))));
    
    JacPDstandardNth13gt23 = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11gt23,kmadd(J21L,PDstandardNth12gt23,kmul(J31L,PDstandardNth13gt23))),kmadd(J11L,kmadd(J23L,PDstandardNth12gt23,kmul(J33L,PDstandardNth13gt23)),kmadd(dJ113L,PDstandardNth1gt23,kmadd(J23L,kmadd(J21L,PDstandardNth22gt23,kmul(J31L,PDstandardNth23gt23)),kmadd(dJ213L,PDstandardNth2gt23,kmadd(J33L,kmadd(J21L,PDstandardNth23gt23,kmul(J31L,PDstandardNth33gt23)),kmul(dJ313L,PDstandardNth3gt23)))))));
    
    JacPDstandardNth13gt33 = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11gt33,kmadd(J21L,PDstandardNth12gt33,kmul(J31L,PDstandardNth13gt33))),kmadd(J11L,kmadd(J23L,PDstandardNth12gt33,kmul(J33L,PDstandardNth13gt33)),kmadd(dJ113L,PDstandardNth1gt33,kmadd(J23L,kmadd(J21L,PDstandardNth22gt33,kmul(J31L,PDstandardNth23gt33)),kmadd(dJ213L,PDstandardNth2gt33,kmadd(J33L,kmadd(J21L,PDstandardNth23gt33,kmul(J31L,PDstandardNth33gt33)),kmul(dJ313L,PDstandardNth3gt33)))))));
    
    JacPDstandardNth13phi = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11phi,kmadd(J21L,PDstandardNth12phi,kmul(J31L,PDstandardNth13phi))),kmadd(J11L,kmadd(J23L,PDstandardNth12phi,kmul(J33L,PDstandardNth13phi)),kmadd(dJ113L,PDstandardNth1phi,kmadd(J23L,kmadd(J21L,PDstandardNth22phi,kmul(J31L,PDstandardNth23phi)),kmadd(dJ213L,PDstandardNth2phi,kmadd(J33L,kmadd(J21L,PDstandardNth23phi,kmul(J31L,PDstandardNth33phi)),kmul(dJ313L,PDstandardNth3phi)))))));
    
    JacPDstandardNth21gt11 = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11gt11,kmadd(J21L,PDstandardNth12gt11,kmul(J31L,PDstandardNth13gt11))),kmadd(J11L,kmadd(J22L,PDstandardNth12gt11,kmul(J32L,PDstandardNth13gt11)),kmadd(dJ112L,PDstandardNth1gt11,kmadd(J22L,kmadd(J21L,PDstandardNth22gt11,kmul(J31L,PDstandardNth23gt11)),kmadd(dJ212L,PDstandardNth2gt11,kmadd(J32L,kmadd(J21L,PDstandardNth23gt11,kmul(J31L,PDstandardNth33gt11)),kmul(dJ312L,PDstandardNth3gt11)))))));
    
    JacPDstandardNth21gt12 = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11gt12,kmadd(J21L,PDstandardNth12gt12,kmul(J31L,PDstandardNth13gt12))),kmadd(J11L,kmadd(J22L,PDstandardNth12gt12,kmul(J32L,PDstandardNth13gt12)),kmadd(dJ112L,PDstandardNth1gt12,kmadd(J22L,kmadd(J21L,PDstandardNth22gt12,kmul(J31L,PDstandardNth23gt12)),kmadd(dJ212L,PDstandardNth2gt12,kmadd(J32L,kmadd(J21L,PDstandardNth23gt12,kmul(J31L,PDstandardNth33gt12)),kmul(dJ312L,PDstandardNth3gt12)))))));
    
    JacPDstandardNth21gt13 = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11gt13,kmadd(J21L,PDstandardNth12gt13,kmul(J31L,PDstandardNth13gt13))),kmadd(J11L,kmadd(J22L,PDstandardNth12gt13,kmul(J32L,PDstandardNth13gt13)),kmadd(dJ112L,PDstandardNth1gt13,kmadd(J22L,kmadd(J21L,PDstandardNth22gt13,kmul(J31L,PDstandardNth23gt13)),kmadd(dJ212L,PDstandardNth2gt13,kmadd(J32L,kmadd(J21L,PDstandardNth23gt13,kmul(J31L,PDstandardNth33gt13)),kmul(dJ312L,PDstandardNth3gt13)))))));
    
    JacPDstandardNth21gt22 = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11gt22,kmadd(J21L,PDstandardNth12gt22,kmul(J31L,PDstandardNth13gt22))),kmadd(J11L,kmadd(J22L,PDstandardNth12gt22,kmul(J32L,PDstandardNth13gt22)),kmadd(dJ112L,PDstandardNth1gt22,kmadd(J22L,kmadd(J21L,PDstandardNth22gt22,kmul(J31L,PDstandardNth23gt22)),kmadd(dJ212L,PDstandardNth2gt22,kmadd(J32L,kmadd(J21L,PDstandardNth23gt22,kmul(J31L,PDstandardNth33gt22)),kmul(dJ312L,PDstandardNth3gt22)))))));
    
    JacPDstandardNth21gt23 = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11gt23,kmadd(J21L,PDstandardNth12gt23,kmul(J31L,PDstandardNth13gt23))),kmadd(J11L,kmadd(J22L,PDstandardNth12gt23,kmul(J32L,PDstandardNth13gt23)),kmadd(dJ112L,PDstandardNth1gt23,kmadd(J22L,kmadd(J21L,PDstandardNth22gt23,kmul(J31L,PDstandardNth23gt23)),kmadd(dJ212L,PDstandardNth2gt23,kmadd(J32L,kmadd(J21L,PDstandardNth23gt23,kmul(J31L,PDstandardNth33gt23)),kmul(dJ312L,PDstandardNth3gt23)))))));
    
    JacPDstandardNth21gt33 = 
      kmadd(J12L,kmadd(J11L,PDstandardNth11gt33,kmadd(J21L,PDstandardNth12gt33,kmul(J31L,PDstandardNth13gt33))),kmadd(J11L,kmadd(J22L,PDstandardNth12gt33,kmul(J32L,PDstandardNth13gt33)),kmadd(dJ112L,PDstandardNth1gt33,kmadd(J22L,kmadd(J21L,PDstandardNth22gt33,kmul(J31L,PDstandardNth23gt33)),kmadd(dJ212L,PDstandardNth2gt33,kmadd(J32L,kmadd(J21L,PDstandardNth23gt33,kmul(J31L,PDstandardNth33gt33)),kmul(dJ312L,PDstandardNth3gt33)))))));
    
    JacPDstandardNth23alpha = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11alpha,kmadd(J22L,PDstandardNth12alpha,kmul(J32L,PDstandardNth13alpha))),kmadd(J12L,kmadd(J23L,PDstandardNth12alpha,kmul(J33L,PDstandardNth13alpha)),kmadd(dJ123L,PDstandardNth1alpha,kmadd(J23L,kmadd(J22L,PDstandardNth22alpha,kmul(J32L,PDstandardNth23alpha)),kmadd(dJ223L,PDstandardNth2alpha,kmadd(J33L,kmadd(J22L,PDstandardNth23alpha,kmul(J32L,PDstandardNth33alpha)),kmul(dJ323L,PDstandardNth3alpha)))))));
    
    JacPDstandardNth23gt11 = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11gt11,kmadd(J22L,PDstandardNth12gt11,kmul(J32L,PDstandardNth13gt11))),kmadd(J12L,kmadd(J23L,PDstandardNth12gt11,kmul(J33L,PDstandardNth13gt11)),kmadd(dJ123L,PDstandardNth1gt11,kmadd(J23L,kmadd(J22L,PDstandardNth22gt11,kmul(J32L,PDstandardNth23gt11)),kmadd(dJ223L,PDstandardNth2gt11,kmadd(J33L,kmadd(J22L,PDstandardNth23gt11,kmul(J32L,PDstandardNth33gt11)),kmul(dJ323L,PDstandardNth3gt11)))))));
    
    JacPDstandardNth23gt12 = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11gt12,kmadd(J22L,PDstandardNth12gt12,kmul(J32L,PDstandardNth13gt12))),kmadd(J12L,kmadd(J23L,PDstandardNth12gt12,kmul(J33L,PDstandardNth13gt12)),kmadd(dJ123L,PDstandardNth1gt12,kmadd(J23L,kmadd(J22L,PDstandardNth22gt12,kmul(J32L,PDstandardNth23gt12)),kmadd(dJ223L,PDstandardNth2gt12,kmadd(J33L,kmadd(J22L,PDstandardNth23gt12,kmul(J32L,PDstandardNth33gt12)),kmul(dJ323L,PDstandardNth3gt12)))))));
    
    JacPDstandardNth23gt13 = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11gt13,kmadd(J22L,PDstandardNth12gt13,kmul(J32L,PDstandardNth13gt13))),kmadd(J12L,kmadd(J23L,PDstandardNth12gt13,kmul(J33L,PDstandardNth13gt13)),kmadd(dJ123L,PDstandardNth1gt13,kmadd(J23L,kmadd(J22L,PDstandardNth22gt13,kmul(J32L,PDstandardNth23gt13)),kmadd(dJ223L,PDstandardNth2gt13,kmadd(J33L,kmadd(J22L,PDstandardNth23gt13,kmul(J32L,PDstandardNth33gt13)),kmul(dJ323L,PDstandardNth3gt13)))))));
    
    JacPDstandardNth23gt22 = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11gt22,kmadd(J22L,PDstandardNth12gt22,kmul(J32L,PDstandardNth13gt22))),kmadd(J12L,kmadd(J23L,PDstandardNth12gt22,kmul(J33L,PDstandardNth13gt22)),kmadd(dJ123L,PDstandardNth1gt22,kmadd(J23L,kmadd(J22L,PDstandardNth22gt22,kmul(J32L,PDstandardNth23gt22)),kmadd(dJ223L,PDstandardNth2gt22,kmadd(J33L,kmadd(J22L,PDstandardNth23gt22,kmul(J32L,PDstandardNth33gt22)),kmul(dJ323L,PDstandardNth3gt22)))))));
    
    JacPDstandardNth23gt23 = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11gt23,kmadd(J22L,PDstandardNth12gt23,kmul(J32L,PDstandardNth13gt23))),kmadd(J12L,kmadd(J23L,PDstandardNth12gt23,kmul(J33L,PDstandardNth13gt23)),kmadd(dJ123L,PDstandardNth1gt23,kmadd(J23L,kmadd(J22L,PDstandardNth22gt23,kmul(J32L,PDstandardNth23gt23)),kmadd(dJ223L,PDstandardNth2gt23,kmadd(J33L,kmadd(J22L,PDstandardNth23gt23,kmul(J32L,PDstandardNth33gt23)),kmul(dJ323L,PDstandardNth3gt23)))))));
    
    JacPDstandardNth23gt33 = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11gt33,kmadd(J22L,PDstandardNth12gt33,kmul(J32L,PDstandardNth13gt33))),kmadd(J12L,kmadd(J23L,PDstandardNth12gt33,kmul(J33L,PDstandardNth13gt33)),kmadd(dJ123L,PDstandardNth1gt33,kmadd(J23L,kmadd(J22L,PDstandardNth22gt33,kmul(J32L,PDstandardNth23gt33)),kmadd(dJ223L,PDstandardNth2gt33,kmadd(J33L,kmadd(J22L,PDstandardNth23gt33,kmul(J32L,PDstandardNth33gt33)),kmul(dJ323L,PDstandardNth3gt33)))))));
    
    JacPDstandardNth23phi = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11phi,kmadd(J22L,PDstandardNth12phi,kmul(J32L,PDstandardNth13phi))),kmadd(J12L,kmadd(J23L,PDstandardNth12phi,kmul(J33L,PDstandardNth13phi)),kmadd(dJ123L,PDstandardNth1phi,kmadd(J23L,kmadd(J22L,PDstandardNth22phi,kmul(J32L,PDstandardNth23phi)),kmadd(dJ223L,PDstandardNth2phi,kmadd(J33L,kmadd(J22L,PDstandardNth23phi,kmul(J32L,PDstandardNth33phi)),kmul(dJ323L,PDstandardNth3phi)))))));
    
    JacPDstandardNth31gt11 = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11gt11,kmadd(J21L,PDstandardNth12gt11,kmul(J31L,PDstandardNth13gt11))),kmadd(J11L,kmadd(J23L,PDstandardNth12gt11,kmul(J33L,PDstandardNth13gt11)),kmadd(dJ113L,PDstandardNth1gt11,kmadd(J23L,kmadd(J21L,PDstandardNth22gt11,kmul(J31L,PDstandardNth23gt11)),kmadd(dJ213L,PDstandardNth2gt11,kmadd(J33L,kmadd(J21L,PDstandardNth23gt11,kmul(J31L,PDstandardNth33gt11)),kmul(dJ313L,PDstandardNth3gt11)))))));
    
    JacPDstandardNth31gt12 = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11gt12,kmadd(J21L,PDstandardNth12gt12,kmul(J31L,PDstandardNth13gt12))),kmadd(J11L,kmadd(J23L,PDstandardNth12gt12,kmul(J33L,PDstandardNth13gt12)),kmadd(dJ113L,PDstandardNth1gt12,kmadd(J23L,kmadd(J21L,PDstandardNth22gt12,kmul(J31L,PDstandardNth23gt12)),kmadd(dJ213L,PDstandardNth2gt12,kmadd(J33L,kmadd(J21L,PDstandardNth23gt12,kmul(J31L,PDstandardNth33gt12)),kmul(dJ313L,PDstandardNth3gt12)))))));
    
    JacPDstandardNth31gt13 = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11gt13,kmadd(J21L,PDstandardNth12gt13,kmul(J31L,PDstandardNth13gt13))),kmadd(J11L,kmadd(J23L,PDstandardNth12gt13,kmul(J33L,PDstandardNth13gt13)),kmadd(dJ113L,PDstandardNth1gt13,kmadd(J23L,kmadd(J21L,PDstandardNth22gt13,kmul(J31L,PDstandardNth23gt13)),kmadd(dJ213L,PDstandardNth2gt13,kmadd(J33L,kmadd(J21L,PDstandardNth23gt13,kmul(J31L,PDstandardNth33gt13)),kmul(dJ313L,PDstandardNth3gt13)))))));
    
    JacPDstandardNth31gt22 = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11gt22,kmadd(J21L,PDstandardNth12gt22,kmul(J31L,PDstandardNth13gt22))),kmadd(J11L,kmadd(J23L,PDstandardNth12gt22,kmul(J33L,PDstandardNth13gt22)),kmadd(dJ113L,PDstandardNth1gt22,kmadd(J23L,kmadd(J21L,PDstandardNth22gt22,kmul(J31L,PDstandardNth23gt22)),kmadd(dJ213L,PDstandardNth2gt22,kmadd(J33L,kmadd(J21L,PDstandardNth23gt22,kmul(J31L,PDstandardNth33gt22)),kmul(dJ313L,PDstandardNth3gt22)))))));
    
    JacPDstandardNth31gt23 = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11gt23,kmadd(J21L,PDstandardNth12gt23,kmul(J31L,PDstandardNth13gt23))),kmadd(J11L,kmadd(J23L,PDstandardNth12gt23,kmul(J33L,PDstandardNth13gt23)),kmadd(dJ113L,PDstandardNth1gt23,kmadd(J23L,kmadd(J21L,PDstandardNth22gt23,kmul(J31L,PDstandardNth23gt23)),kmadd(dJ213L,PDstandardNth2gt23,kmadd(J33L,kmadd(J21L,PDstandardNth23gt23,kmul(J31L,PDstandardNth33gt23)),kmul(dJ313L,PDstandardNth3gt23)))))));
    
    JacPDstandardNth31gt33 = 
      kmadd(J13L,kmadd(J11L,PDstandardNth11gt33,kmadd(J21L,PDstandardNth12gt33,kmul(J31L,PDstandardNth13gt33))),kmadd(J11L,kmadd(J23L,PDstandardNth12gt33,kmul(J33L,PDstandardNth13gt33)),kmadd(dJ113L,PDstandardNth1gt33,kmadd(J23L,kmadd(J21L,PDstandardNth22gt33,kmul(J31L,PDstandardNth23gt33)),kmadd(dJ213L,PDstandardNth2gt33,kmadd(J33L,kmadd(J21L,PDstandardNth23gt33,kmul(J31L,PDstandardNth33gt33)),kmul(dJ313L,PDstandardNth3gt33)))))));
    
    JacPDstandardNth32gt11 = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11gt11,kmadd(J22L,PDstandardNth12gt11,kmul(J32L,PDstandardNth13gt11))),kmadd(J12L,kmadd(J23L,PDstandardNth12gt11,kmul(J33L,PDstandardNth13gt11)),kmadd(dJ123L,PDstandardNth1gt11,kmadd(J23L,kmadd(J22L,PDstandardNth22gt11,kmul(J32L,PDstandardNth23gt11)),kmadd(dJ223L,PDstandardNth2gt11,kmadd(J33L,kmadd(J22L,PDstandardNth23gt11,kmul(J32L,PDstandardNth33gt11)),kmul(dJ323L,PDstandardNth3gt11)))))));
    
    JacPDstandardNth32gt12 = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11gt12,kmadd(J22L,PDstandardNth12gt12,kmul(J32L,PDstandardNth13gt12))),kmadd(J12L,kmadd(J23L,PDstandardNth12gt12,kmul(J33L,PDstandardNth13gt12)),kmadd(dJ123L,PDstandardNth1gt12,kmadd(J23L,kmadd(J22L,PDstandardNth22gt12,kmul(J32L,PDstandardNth23gt12)),kmadd(dJ223L,PDstandardNth2gt12,kmadd(J33L,kmadd(J22L,PDstandardNth23gt12,kmul(J32L,PDstandardNth33gt12)),kmul(dJ323L,PDstandardNth3gt12)))))));
    
    JacPDstandardNth32gt13 = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11gt13,kmadd(J22L,PDstandardNth12gt13,kmul(J32L,PDstandardNth13gt13))),kmadd(J12L,kmadd(J23L,PDstandardNth12gt13,kmul(J33L,PDstandardNth13gt13)),kmadd(dJ123L,PDstandardNth1gt13,kmadd(J23L,kmadd(J22L,PDstandardNth22gt13,kmul(J32L,PDstandardNth23gt13)),kmadd(dJ223L,PDstandardNth2gt13,kmadd(J33L,kmadd(J22L,PDstandardNth23gt13,kmul(J32L,PDstandardNth33gt13)),kmul(dJ323L,PDstandardNth3gt13)))))));
    
    JacPDstandardNth32gt22 = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11gt22,kmadd(J22L,PDstandardNth12gt22,kmul(J32L,PDstandardNth13gt22))),kmadd(J12L,kmadd(J23L,PDstandardNth12gt22,kmul(J33L,PDstandardNth13gt22)),kmadd(dJ123L,PDstandardNth1gt22,kmadd(J23L,kmadd(J22L,PDstandardNth22gt22,kmul(J32L,PDstandardNth23gt22)),kmadd(dJ223L,PDstandardNth2gt22,kmadd(J33L,kmadd(J22L,PDstandardNth23gt22,kmul(J32L,PDstandardNth33gt22)),kmul(dJ323L,PDstandardNth3gt22)))))));
    
    JacPDstandardNth32gt23 = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11gt23,kmadd(J22L,PDstandardNth12gt23,kmul(J32L,PDstandardNth13gt23))),kmadd(J12L,kmadd(J23L,PDstandardNth12gt23,kmul(J33L,PDstandardNth13gt23)),kmadd(dJ123L,PDstandardNth1gt23,kmadd(J23L,kmadd(J22L,PDstandardNth22gt23,kmul(J32L,PDstandardNth23gt23)),kmadd(dJ223L,PDstandardNth2gt23,kmadd(J33L,kmadd(J22L,PDstandardNth23gt23,kmul(J32L,PDstandardNth33gt23)),kmul(dJ323L,PDstandardNth3gt23)))))));
    
    JacPDstandardNth32gt33 = 
      kmadd(J13L,kmadd(J12L,PDstandardNth11gt33,kmadd(J22L,PDstandardNth12gt33,kmul(J32L,PDstandardNth13gt33))),kmadd(J12L,kmadd(J23L,PDstandardNth12gt33,kmul(J33L,PDstandardNth13gt33)),kmadd(dJ123L,PDstandardNth1gt33,kmadd(J23L,kmadd(J22L,PDstandardNth22gt33,kmul(J32L,PDstandardNth23gt33)),kmadd(dJ223L,PDstandardNth2gt33,kmadd(J33L,kmadd(J22L,PDstandardNth23gt33,kmul(J32L,PDstandardNth33gt33)),kmul(dJ323L,PDstandardNth3gt33)))))));
  }
  else
  {
    JacPDstandardNth1alpha = PDstandardNth1alpha;
    
    JacPDstandardNth1beta1 = PDstandardNth1beta1;
    
    JacPDstandardNth1beta2 = PDstandardNth1beta2;
    
    JacPDstandardNth1beta3 = PDstandardNth1beta3;
    
    JacPDstandardNth1gt11 = PDstandardNth1gt11;
    
    JacPDstandardNth1gt12 = PDstandardNth1gt12;
    
    JacPDstandardNth1gt13 = PDstandardNth1gt13;
    
    JacPDstandardNth1gt22 = PDstandardNth1gt22;
    
    JacPDstandardNth1gt23 = PDstandardNth1gt23;
    
    JacPDstandardNth1gt33 = PDstandardNth1gt33;
    
    JacPDstandardNth1phi = PDstandardNth1phi;
    
    JacPDstandardNth1Xt1 = PDstandardNth1Xt1;
    
    JacPDstandardNth1Xt2 = PDstandardNth1Xt2;
    
    JacPDstandardNth1Xt3 = PDstandardNth1Xt3;
    
    JacPDstandardNth2alpha = PDstandardNth2alpha;
    
    JacPDstandardNth2beta1 = PDstandardNth2beta1;
    
    JacPDstandardNth2beta2 = PDstandardNth2beta2;
    
    JacPDstandardNth2beta3 = PDstandardNth2beta3;
    
    JacPDstandardNth2gt11 = PDstandardNth2gt11;
    
    JacPDstandardNth2gt12 = PDstandardNth2gt12;
    
    JacPDstandardNth2gt13 = PDstandardNth2gt13;
    
    JacPDstandardNth2gt22 = PDstandardNth2gt22;
    
    JacPDstandardNth2gt23 = PDstandardNth2gt23;
    
    JacPDstandardNth2gt33 = PDstandardNth2gt33;
    
    JacPDstandardNth2phi = PDstandardNth2phi;
    
    JacPDstandardNth2Xt1 = PDstandardNth2Xt1;
    
    JacPDstandardNth2Xt2 = PDstandardNth2Xt2;
    
    JacPDstandardNth2Xt3 = PDstandardNth2Xt3;
    
    JacPDstandardNth3alpha = PDstandardNth3alpha;
    
    JacPDstandardNth3beta1 = PDstandardNth3beta1;
    
    JacPDstandardNth3beta2 = PDstandardNth3beta2;
    
    JacPDstandardNth3beta3 = PDstandardNth3beta3;
    
    JacPDstandardNth3gt11 = PDstandardNth3gt11;
    
    JacPDstandardNth3gt12 = PDstandardNth3gt12;
    
    JacPDstandardNth3gt13 = PDstandardNth3gt13;
    
    JacPDstandardNth3gt22 = PDstandardNth3gt22;
    
    JacPDstandardNth3gt23 = PDstandardNth3gt23;
    
    JacPDstandardNth3gt33 = PDstandardNth3gt33;
    
    JacPDstandardNth3phi = PDstandardNth3phi;
    
    JacPDstandardNth3Xt1 = PDstandardNth3Xt1;
    
    JacPDstandardNth3Xt2 = PDstandardNth3Xt2;
    
    JacPDstandardNth3Xt3 = PDstandardNth3Xt3;
    
    JacPDstandardNth11alpha = PDstandardNth11alpha;
    
    JacPDstandardNth11gt11 = PDstandardNth11gt11;
    
    JacPDstandardNth11gt12 = PDstandardNth11gt12;
    
    JacPDstandardNth11gt13 = PDstandardNth11gt13;
    
    JacPDstandardNth11gt22 = PDstandardNth11gt22;
    
    JacPDstandardNth11gt23 = PDstandardNth11gt23;
    
    JacPDstandardNth11gt33 = PDstandardNth11gt33;
    
    JacPDstandardNth11phi = PDstandardNth11phi;
    
    JacPDstandardNth22alpha = PDstandardNth22alpha;
    
    JacPDstandardNth22gt11 = PDstandardNth22gt11;
    
    JacPDstandardNth22gt12 = PDstandardNth22gt12;
    
    JacPDstandardNth22gt13 = PDstandardNth22gt13;
    
    JacPDstandardNth22gt22 = PDstandardNth22gt22;
    
    JacPDstandardNth22gt23 = PDstandardNth22gt23;
    
    JacPDstandardNth22gt33 = PDstandardNth22gt33;
    
    JacPDstandardNth22phi = PDstandardNth22phi;
    
    JacPDstandardNth33alpha = PDstandardNth33alpha;
    
    JacPDstandardNth33gt11 = PDstandardNth33gt11;
    
    JacPDstandardNth33gt12 = PDstandardNth33gt12;
    
    JacPDstandardNth33gt13 = PDstandardNth33gt13;
    
    JacPDstandardNth33gt22 = PDstandardNth33gt22;
    
    JacPDstandardNth33gt23 = PDstandardNth33gt23;
    
    JacPDstandardNth33gt33 = PDstandardNth33gt33;
    
    JacPDstandardNth33phi = PDstandardNth33phi;
    
    JacPDstandardNth12alpha = PDstandardNth12alpha;
    
    JacPDstandardNth12gt11 = PDstandardNth12gt11;
    
    JacPDstandardNth12gt12 = PDstandardNth12gt12;
    
    JacPDstandardNth12gt13 = PDstandardNth12gt13;
    
    JacPDstandardNth12gt22 = PDstandardNth12gt22;
    
    JacPDstandardNth12gt23 = PDstandardNth12gt23;
    
    JacPDstandardNth12gt33 = PDstandardNth12gt33;
    
    JacPDstandardNth12phi = PDstandardNth12phi;
    
    JacPDstandardNth13alpha = PDstandardNth13alpha;
    
    JacPDstandardNth13gt11 = PDstandardNth13gt11;
    
    JacPDstandardNth13gt12 = PDstandardNth13gt12;
    
    JacPDstandardNth13gt13 = PDstandardNth13gt13;
    
    JacPDstandardNth13gt22 = PDstandardNth13gt22;
    
    JacPDstandardNth13gt23 = PDstandardNth13gt23;
    
    JacPDstandardNth13gt33 = PDstandardNth13gt33;
    
    JacPDstandardNth13phi = PDstandardNth13phi;
    
    JacPDstandardNth21gt11 = PDstandardNth12gt11;
    
    JacPDstandardNth21gt12 = PDstandardNth12gt12;
    
    JacPDstandardNth21gt13 = PDstandardNth12gt13;
    
    JacPDstandardNth21gt22 = PDstandardNth12gt22;
    
    JacPDstandardNth21gt23 = PDstandardNth12gt23;
    
    JacPDstandardNth21gt33 = PDstandardNth12gt33;
    
    JacPDstandardNth23alpha = PDstandardNth23alpha;
    
    JacPDstandardNth23gt11 = PDstandardNth23gt11;
    
    JacPDstandardNth23gt12 = PDstandardNth23gt12;
    
    JacPDstandardNth23gt13 = PDstandardNth23gt13;
    
    JacPDstandardNth23gt22 = PDstandardNth23gt22;
    
    JacPDstandardNth23gt23 = PDstandardNth23gt23;
    
    JacPDstandardNth23gt33 = PDstandardNth23gt33;
    
    JacPDstandardNth23phi = PDstandardNth23phi;
    
    JacPDstandardNth31gt11 = PDstandardNth13gt11;
    
    JacPDstandardNth31gt12 = PDstandardNth13gt12;
    
    JacPDstandardNth31gt13 = PDstandardNth13gt13;
    
    JacPDstandardNth31gt22 = PDstandardNth13gt22;
    
    JacPDstandardNth31gt23 = PDstandardNth13gt23;
    
    JacPDstandardNth31gt33 = PDstandardNth13gt33;
    
    JacPDstandardNth32gt11 = PDstandardNth23gt11;
    
    JacPDstandardNth32gt12 = PDstandardNth23gt12;
    
    JacPDstandardNth32gt13 = PDstandardNth23gt13;
    
    JacPDstandardNth32gt22 = PDstandardNth23gt22;
    
    JacPDstandardNth32gt23 = PDstandardNth23gt23;
    
    JacPDstandardNth32gt33 = PDstandardNth23gt33;
  }
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED detgt = ToReal(1);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED gtu11 = 
    kdiv(kmsub(gt22L,gt33L,kmul(gt23L,gt23L)),detgt);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED gtu12 = 
    kdiv(kmsub(gt13L,gt23L,kmul(gt12L,gt33L)),detgt);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED gtu13 = 
    kdiv(kmsub(gt12L,gt23L,kmul(gt13L,gt22L)),detgt);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED gtu22 = 
    kdiv(kmsub(gt11L,gt33L,kmul(gt13L,gt13L)),detgt);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED gtu23 = 
    kdiv(kmsub(gt12L,gt13L,kmul(gt11L,gt23L)),detgt);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED gtu33 = 
    kdiv(kmsub(gt11L,gt22L,kmul(gt12L,gt12L)),detgt);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl111 = 
    kmul(JacPDstandardNth1gt11,ToReal(0.5));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl112 = 
    kmul(JacPDstandardNth2gt11,ToReal(0.5));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl113 = 
    kmul(JacPDstandardNth3gt11,ToReal(0.5));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl122 = 
    kmadd(JacPDstandardNth1gt22,ToReal(-0.5),JacPDstandardNth2gt12);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl123 = 
    kmul(kadd(JacPDstandardNth2gt13,ksub(JacPDstandardNth3gt12,JacPDstandardNth1gt23)),ToReal(0.5));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl133 = 
    kmadd(JacPDstandardNth1gt33,ToReal(-0.5),JacPDstandardNth3gt13);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl211 = 
    kmadd(JacPDstandardNth2gt11,ToReal(-0.5),JacPDstandardNth1gt12);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl212 = 
    kmul(JacPDstandardNth1gt22,ToReal(0.5));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl213 = 
    kmul(kadd(JacPDstandardNth1gt23,ksub(JacPDstandardNth3gt12,JacPDstandardNth2gt13)),ToReal(0.5));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl222 = 
    kmul(JacPDstandardNth2gt22,ToReal(0.5));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl223 = 
    kmul(JacPDstandardNth3gt22,ToReal(0.5));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl233 = 
    kmadd(JacPDstandardNth2gt33,ToReal(-0.5),JacPDstandardNth3gt23);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl311 = 
    kmadd(JacPDstandardNth3gt11,ToReal(-0.5),JacPDstandardNth1gt13);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl312 = 
    kmul(kadd(JacPDstandardNth1gt23,ksub(JacPDstandardNth2gt13,JacPDstandardNth3gt12)),ToReal(0.5));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl313 = 
    kmul(JacPDstandardNth1gt33,ToReal(0.5));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl322 = 
    kmadd(JacPDstandardNth3gt22,ToReal(-0.5),JacPDstandardNth2gt23);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl323 = 
    kmul(JacPDstandardNth2gt33,ToReal(0.5));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtl333 = 
    kmul(JacPDstandardNth3gt33,ToReal(0.5));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu111 = 
    kmadd(Gtl111,gtu11,kmadd(Gtl112,gtu12,kmul(Gtl113,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu112 = 
    kmadd(Gtl111,gtu12,kmadd(Gtl112,gtu22,kmul(Gtl113,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu113 = 
    kmadd(Gtl111,gtu13,kmadd(Gtl112,gtu23,kmul(Gtl113,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu121 = 
    kmadd(Gtl112,gtu11,kmadd(Gtl122,gtu12,kmul(Gtl123,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu122 = 
    kmadd(Gtl112,gtu12,kmadd(Gtl122,gtu22,kmul(Gtl123,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu123 = 
    kmadd(Gtl112,gtu13,kmadd(Gtl122,gtu23,kmul(Gtl123,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu131 = 
    kmadd(Gtl113,gtu11,kmadd(Gtl123,gtu12,kmul(Gtl133,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu132 = 
    kmadd(Gtl113,gtu12,kmadd(Gtl123,gtu22,kmul(Gtl133,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu133 = 
    kmadd(Gtl113,gtu13,kmadd(Gtl123,gtu23,kmul(Gtl133,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu211 = 
    kmadd(Gtl211,gtu11,kmadd(Gtl212,gtu12,kmul(Gtl213,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu212 = 
    kmadd(Gtl211,gtu12,kmadd(Gtl212,gtu22,kmul(Gtl213,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu213 = 
    kmadd(Gtl211,gtu13,kmadd(Gtl212,gtu23,kmul(Gtl213,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu221 = 
    kmadd(Gtl212,gtu11,kmadd(Gtl222,gtu12,kmul(Gtl223,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu222 = 
    kmadd(Gtl212,gtu12,kmadd(Gtl222,gtu22,kmul(Gtl223,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu223 = 
    kmadd(Gtl212,gtu13,kmadd(Gtl222,gtu23,kmul(Gtl223,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu231 = 
    kmadd(Gtl213,gtu11,kmadd(Gtl223,gtu12,kmul(Gtl233,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu232 = 
    kmadd(Gtl213,gtu12,kmadd(Gtl223,gtu22,kmul(Gtl233,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu233 = 
    kmadd(Gtl213,gtu13,kmadd(Gtl223,gtu23,kmul(Gtl233,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu311 = 
    kmadd(Gtl311,gtu11,kmadd(Gtl312,gtu12,kmul(Gtl313,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu312 = 
    kmadd(Gtl311,gtu12,kmadd(Gtl312,gtu22,kmul(Gtl313,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu313 = 
    kmadd(Gtl311,gtu13,kmadd(Gtl312,gtu23,kmul(Gtl313,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu321 = 
    kmadd(Gtl312,gtu11,kmadd(Gtl322,gtu12,kmul(Gtl323,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu322 = 
    kmadd(Gtl312,gtu12,kmadd(Gtl322,gtu22,kmul(Gtl323,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu323 = 
    kmadd(Gtl312,gtu13,kmadd(Gtl322,gtu23,kmul(Gtl323,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu331 = 
    kmadd(Gtl313,gtu11,kmadd(Gtl323,gtu12,kmul(Gtl333,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu332 = 
    kmadd(Gtl313,gtu12,kmadd(Gtl323,gtu22,kmul(Gtl333,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gtlu333 = 
    kmadd(Gtl313,gtu13,kmadd(Gtl323,gtu23,kmul(Gtl333,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt111 = 
    kmadd(Gtl111,gtu11,kmadd(Gtl211,gtu12,kmul(Gtl311,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt211 = 
    kmadd(Gtl111,gtu12,kmadd(Gtl211,gtu22,kmul(Gtl311,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt311 = 
    kmadd(Gtl111,gtu13,kmadd(Gtl211,gtu23,kmul(Gtl311,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt112 = 
    kmadd(Gtl112,gtu11,kmadd(Gtl212,gtu12,kmul(Gtl312,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt212 = 
    kmadd(Gtl112,gtu12,kmadd(Gtl212,gtu22,kmul(Gtl312,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt312 = 
    kmadd(Gtl112,gtu13,kmadd(Gtl212,gtu23,kmul(Gtl312,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt113 = 
    kmadd(Gtl113,gtu11,kmadd(Gtl213,gtu12,kmul(Gtl313,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt213 = 
    kmadd(Gtl113,gtu12,kmadd(Gtl213,gtu22,kmul(Gtl313,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt313 = 
    kmadd(Gtl113,gtu13,kmadd(Gtl213,gtu23,kmul(Gtl313,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt122 = 
    kmadd(Gtl122,gtu11,kmadd(Gtl222,gtu12,kmul(Gtl322,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt222 = 
    kmadd(Gtl122,gtu12,kmadd(Gtl222,gtu22,kmul(Gtl322,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt322 = 
    kmadd(Gtl122,gtu13,kmadd(Gtl222,gtu23,kmul(Gtl322,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt123 = 
    kmadd(Gtl123,gtu11,kmadd(Gtl223,gtu12,kmul(Gtl323,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt223 = 
    kmadd(Gtl123,gtu12,kmadd(Gtl223,gtu22,kmul(Gtl323,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt323 = 
    kmadd(Gtl123,gtu13,kmadd(Gtl223,gtu23,kmul(Gtl323,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt133 = 
    kmadd(Gtl133,gtu11,kmadd(Gtl233,gtu12,kmul(Gtl333,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt233 = 
    kmadd(Gtl133,gtu12,kmadd(Gtl233,gtu22,kmul(Gtl333,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Gt333 = 
    kmadd(Gtl133,gtu13,kmadd(Gtl233,gtu23,kmul(Gtl333,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Xtn1 = 
    kmadd(Gt111,gtu11,kmadd(Gt122,gtu22,kmadd(Gt133,gtu33,kmul(kmadd(Gt112,gtu12,kmadd(Gt113,gtu13,kmul(Gt123,gtu23))),ToReal(2)))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Xtn2 = 
    kmadd(Gt211,gtu11,kmadd(Gt222,gtu22,kmadd(Gt233,gtu33,kmul(kmadd(Gt212,gtu12,kmadd(Gt213,gtu13,kmul(Gt223,gtu23))),ToReal(2)))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Xtn3 = 
    kmadd(Gt311,gtu11,kmadd(Gt322,gtu22,kmadd(Gt333,gtu33,kmul(kmadd(Gt312,gtu12,kmadd(Gt313,gtu13,kmul(Gt323,gtu23))),ToReal(2)))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED e4phi = IfThen(conformalMethod == 
    1,kdiv(ToReal(1),kmul(phiL,phiL)),kexp(kmul(phiL,ToReal(4))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED em4phi = kdiv(ToReal(1),e4phi);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED g11 = kmul(gt11L,e4phi);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED g12 = kmul(gt12L,e4phi);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED g13 = kmul(gt13L,e4phi);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED g22 = kmul(gt22L,e4phi);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED g23 = kmul(gt23L,e4phi);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED g33 = kmul(gt33L,e4phi);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED gu11 = kmul(em4phi,gtu11);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED gu12 = kmul(em4phi,gtu12);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED gu13 = kmul(em4phi,gtu13);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED gu22 = kmul(em4phi,gtu22);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED gu23 = kmul(em4phi,gtu23);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED gu33 = kmul(em4phi,gtu33);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Rt11 = 
    kmul(ToReal(0.5),knmsub(gtu11,JacPDstandardNth11gt11,knmsub(gtu22,JacPDstandardNth22gt11,knmsub(gtu33,JacPDstandardNth33gt11,knmsub(gtu12,kadd(JacPDstandardNth21gt11,JacPDstandardNth12gt11),knmsub(gtu13,kadd(JacPDstandardNth31gt11,JacPDstandardNth13gt11),knmsub(gtu23,kadd(JacPDstandardNth32gt11,JacPDstandardNth23gt11),kmadd(kmadd(Gt211,Gtlu211,kmadd(Gt212,Gtlu212,kmadd(Gt213,Gtlu213,kmadd(Gt311,Gtlu311,kmadd(Gt312,Gtlu312,kmadd(Gt313,Gtlu313,kmul(gt11L,JacPDstandardNth1Xt1))))))),ToReal(2),kmadd(gt12L,kmul(JacPDstandardNth1Xt2,ToReal(2)),kmadd(gt13L,kmul(JacPDstandardNth1Xt3,ToReal(2)),kmadd(Gtl111,kmul(Xtn1,ToReal(2)),kmadd(Gtl112,kmul(Xtn2,ToReal(2)),kmadd(Gtl113,kmul(Xtn3,ToReal(2)),kmadd(kmadd(Gt211,Gtlu121,kmadd(Gt212,Gtlu122,kmadd(Gt213,Gtlu123,kmadd(Gt311,Gtlu131,kmadd(Gt312,Gtlu132,kmul(Gt313,Gtlu133)))))),ToReal(4),kmul(kmadd(Gt111,Gtlu111,kmadd(Gt112,Gtlu112,kmul(Gt113,Gtlu113))),ToReal(6))))))))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Rt12 = 
    kmul(ToReal(0.5),kmadd(gt12L,JacPDstandardNth1Xt1,kmadd(gt22L,JacPDstandardNth1Xt2,kmadd(gt23L,JacPDstandardNth1Xt3,kmadd(gt11L,JacPDstandardNth2Xt1,kmadd(gt12L,JacPDstandardNth2Xt2,kmadd(gt13L,JacPDstandardNth2Xt3,kmadd(Gtl112,Xtn1,kmadd(Gtl211,Xtn1,kmadd(Gtl122,Xtn2,kmadd(Gtl212,Xtn2,kmadd(Gtl123,Xtn3,kmadd(Gtl213,Xtn3,knmsub(gtu11,JacPDstandardNth11gt12,knmsub(gtu22,JacPDstandardNth22gt12,knmsub(gtu33,JacPDstandardNth33gt12,knmsub(gtu12,kadd(JacPDstandardNth21gt12,JacPDstandardNth12gt12),knmsub(gtu13,kadd(JacPDstandardNth31gt12,JacPDstandardNth13gt12),knmsub(gtu23,kadd(JacPDstandardNth32gt12,JacPDstandardNth23gt12),kmadd(kmadd(Gt122,Gtlu112,kmadd(Gt123,Gtlu113,kmadd(Gt111,Gtlu121,kmadd(Gt212,Gtlu121,kmadd(Gt222,Gtlu122,kmadd(Gt113,Gtlu123,kmadd(Gt223,Gtlu123,kmadd(Gt312,Gtlu131,kmadd(Gt322,Gtlu132,kmadd(Gt323,Gtlu133,kmadd(Gt111,Gtlu211,kmadd(Gt112,kadd(Gtlu111,kadd(Gtlu122,Gtlu212)),kmadd(Gt113,Gtlu213,kmadd(Gt311,Gtlu231,kmadd(Gt312,Gtlu232,kmadd(Gt313,Gtlu233,kmadd(Gt311,Gtlu321,kmadd(Gt312,Gtlu322,kmul(Gt313,Gtlu323))))))))))))))))))),ToReal(2),kmul(kmadd(Gt211,Gtlu221,kmadd(Gt212,Gtlu222,kmul(Gt213,Gtlu223))),ToReal(4))))))))))))))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Rt13 = 
    kmul(ToReal(0.5),kmadd(gt13L,JacPDstandardNth1Xt1,kmadd(gt23L,JacPDstandardNth1Xt2,kmadd(gt33L,JacPDstandardNth1Xt3,kmadd(gt11L,JacPDstandardNth3Xt1,kmadd(gt12L,JacPDstandardNth3Xt2,kmadd(gt13L,JacPDstandardNth3Xt3,kmadd(Gtl113,Xtn1,kmadd(Gtl311,Xtn1,kmadd(Gtl123,Xtn2,kmadd(Gtl312,Xtn2,kmadd(Gtl133,Xtn3,kmadd(Gtl313,Xtn3,knmsub(gtu11,JacPDstandardNth11gt13,knmsub(gtu22,JacPDstandardNth22gt13,knmsub(gtu33,JacPDstandardNth33gt13,knmsub(gtu12,kadd(JacPDstandardNth21gt13,JacPDstandardNth12gt13),knmsub(gtu13,kadd(JacPDstandardNth31gt13,JacPDstandardNth13gt13),knmsub(gtu23,kadd(JacPDstandardNth32gt13,JacPDstandardNth23gt13),kmadd(kmadd(Gt123,Gtlu112,kmadd(Gt133,Gtlu113,kmadd(Gt213,Gtlu121,kmadd(Gt223,Gtlu122,kmadd(Gt233,Gtlu123,kmadd(Gt111,Gtlu131,kmadd(Gt313,Gtlu131,kmadd(Gt112,Gtlu132,kmadd(Gt323,Gtlu132,kmadd(Gt333,Gtlu133,kmadd(Gt211,Gtlu231,kmadd(Gt212,Gtlu232,kmadd(Gt213,Gtlu233,kmadd(Gt111,Gtlu311,kmadd(Gt112,Gtlu312,kmadd(Gt113,kadd(Gtlu111,kadd(Gtlu133,Gtlu313)),kmadd(Gt211,Gtlu321,kmadd(Gt212,Gtlu322,kmul(Gt213,Gtlu323))))))))))))))))))),ToReal(2),kmul(kmadd(Gt311,Gtlu331,kmadd(Gt312,Gtlu332,kmul(Gt313,Gtlu333))),ToReal(4))))))))))))))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Rt22 = 
    kmul(ToReal(0.5),knmsub(gtu11,JacPDstandardNth11gt22,knmsub(gtu22,JacPDstandardNth22gt22,knmsub(gtu33,JacPDstandardNth33gt22,knmsub(gtu12,kadd(JacPDstandardNth21gt22,JacPDstandardNth12gt22),knmsub(gtu13,kadd(JacPDstandardNth31gt22,JacPDstandardNth13gt22),knmsub(gtu23,kadd(JacPDstandardNth32gt22,JacPDstandardNth23gt22),kmadd(gt22L,kmul(JacPDstandardNth2Xt2,ToReal(2)),kmadd(gt23L,kmul(JacPDstandardNth2Xt3,ToReal(2)),kmadd(Gtl212,kmul(Xtn1,ToReal(2)),kmadd(Gtl222,kmul(Xtn2,ToReal(2)),kmadd(Gtl223,kmul(Xtn3,ToReal(2)),kmadd(ToReal(2),kmadd(Gt123,Gtlu123,kmadd(Gt312,Gtlu321,kmadd(Gt322,Gtlu322,kmadd(Gt323,Gtlu323,kmadd(gt12L,JacPDstandardNth2Xt1,kmadd(Gt112,kmadd(Gtlu211,ToReal(2),Gtlu121),kmul(Gt122,kmadd(Gtlu212,ToReal(2),Gtlu122)))))))),kmadd(kmadd(Gt123,Gtlu213,kmadd(Gt312,Gtlu231,kmadd(Gt322,Gtlu232,kmul(Gt323,Gtlu233)))),ToReal(4),kmul(kmadd(Gt212,Gtlu221,kmadd(Gt222,Gtlu222,kmul(Gt223,Gtlu223))),ToReal(6))))))))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Rt23 = 
    kmul(ToReal(0.5),kmadd(gt13L,JacPDstandardNth2Xt1,kmadd(gt23L,JacPDstandardNth2Xt2,kmadd(gt33L,JacPDstandardNth2Xt3,kmadd(gt12L,JacPDstandardNth3Xt1,kmadd(gt22L,JacPDstandardNth3Xt2,kmadd(gt23L,JacPDstandardNth3Xt3,kmadd(Gtl213,Xtn1,kmadd(Gtl312,Xtn1,kmadd(Gtl223,Xtn2,kmadd(Gtl322,Xtn2,kmadd(Gtl233,Xtn3,kmadd(Gtl323,Xtn3,knmsub(gtu11,JacPDstandardNth11gt23,knmsub(gtu22,JacPDstandardNth22gt23,knmsub(gtu33,JacPDstandardNth33gt23,knmsub(gtu12,kadd(JacPDstandardNth21gt23,JacPDstandardNth12gt23),knmsub(gtu13,kadd(JacPDstandardNth31gt23,JacPDstandardNth13gt23),knmsub(gtu23,kadd(JacPDstandardNth32gt23,JacPDstandardNth23gt23),kmadd(kmadd(Gt123,Gtlu133,kmadd(Gt113,Gtlu211,kmadd(Gt123,Gtlu212,kmadd(Gt133,Gtlu213,kmadd(Gt213,Gtlu221,kmadd(Gt223,Gtlu222,kmadd(Gt233,Gtlu223,kmadd(Gt212,Gtlu231,kmadd(Gt313,Gtlu231,kmadd(Gt222,Gtlu232,kmadd(Gt323,Gtlu232,kmadd(Gt223,Gtlu233,kmadd(Gt333,Gtlu233,kmadd(Gt112,kadd(Gtlu131,Gtlu311),kmadd(Gt122,kadd(Gtlu132,Gtlu312),kmadd(Gt123,Gtlu313,kmadd(Gt212,Gtlu321,kmadd(Gt222,Gtlu322,kmul(Gt223,Gtlu323))))))))))))))))))),ToReal(2),kmul(kmadd(Gt312,Gtlu331,kmadd(Gt322,Gtlu332,kmul(Gt323,Gtlu333))),ToReal(4))))))))))))))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Rt33 = 
    kmul(ToReal(0.5),knmsub(gtu11,JacPDstandardNth11gt33,knmsub(gtu22,JacPDstandardNth22gt33,knmsub(gtu33,JacPDstandardNth33gt33,knmsub(gtu12,kadd(JacPDstandardNth21gt33,JacPDstandardNth12gt33),knmsub(gtu13,kadd(JacPDstandardNth31gt33,JacPDstandardNth13gt33),knmsub(gtu23,kadd(JacPDstandardNth32gt33,JacPDstandardNth23gt33),kmadd(gt23L,kmul(JacPDstandardNth3Xt2,ToReal(2)),kmadd(gt33L,kmul(JacPDstandardNth3Xt3,ToReal(2)),kmadd(Gtl313,kmul(Xtn1,ToReal(2)),kmadd(Gtl323,kmul(Xtn2,ToReal(2)),kmadd(Gtl333,kmul(Xtn3,ToReal(2)),kmadd(ToReal(2),kmadd(Gt133,Gtlu133,kmadd(Gt213,Gtlu231,kmadd(Gt223,Gtlu232,kmadd(Gt233,Gtlu233,kmadd(gt13L,JacPDstandardNth3Xt1,kmadd(Gt113,kmadd(Gtlu311,ToReal(2),Gtlu131),kmul(Gt123,kmadd(Gtlu312,ToReal(2),Gtlu132)))))))),kmadd(kmadd(Gt133,Gtlu313,kmadd(Gt213,Gtlu321,kmadd(Gt223,Gtlu322,kmul(Gt233,Gtlu323)))),ToReal(4),kmul(kmadd(Gt313,Gtlu331,kmadd(Gt323,Gtlu332,kmul(Gt333,Gtlu333))),ToReal(6))))))))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED fac1 = IfThen(conformalMethod == 
    1,kdiv(ToReal(-0.5),phiL),ToReal(1));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED cdphi1 = 
    kmul(fac1,JacPDstandardNth1phi);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED cdphi2 = 
    kmul(fac1,JacPDstandardNth2phi);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED cdphi3 = 
    kmul(fac1,JacPDstandardNth3phi);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED fac2 = IfThen(conformalMethod == 
    1,kdiv(ToReal(0.5),kmul(phiL,phiL)),ToReal(0));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED cdphi211 = 
    kmadd(fac2,kmul(JacPDstandardNth1phi,JacPDstandardNth1phi),kmul(fac1,ksub(JacPDstandardNth11phi,kmadd(Gt111,JacPDstandardNth1phi,kmadd(Gt311,JacPDstandardNth3phi,kmul(Gt211,JacPDstandardNth2phi))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED cdphi212 = 
    kmadd(fac2,kmul(JacPDstandardNth1phi,JacPDstandardNth2phi),kmul(fac1,ksub(JacPDstandardNth12phi,kmadd(Gt112,JacPDstandardNth1phi,kmadd(Gt312,JacPDstandardNth3phi,kmul(Gt212,JacPDstandardNth2phi))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED cdphi213 = 
    kmadd(fac2,kmul(JacPDstandardNth1phi,JacPDstandardNth3phi),kmul(fac1,ksub(JacPDstandardNth13phi,kmadd(Gt113,JacPDstandardNth1phi,kmadd(Gt313,JacPDstandardNth3phi,kmul(Gt213,JacPDstandardNth2phi))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED cdphi222 = 
    kmsub(fac2,kmul(JacPDstandardNth2phi,JacPDstandardNth2phi),kmul(fac1,kmadd(Gt122,JacPDstandardNth1phi,kmadd(Gt222,JacPDstandardNth2phi,kmsub(Gt322,JacPDstandardNth3phi,JacPDstandardNth22phi)))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED cdphi223 = 
    kmsub(fac2,kmul(JacPDstandardNth2phi,JacPDstandardNth3phi),kmul(fac1,kmadd(Gt123,JacPDstandardNth1phi,kmadd(Gt223,JacPDstandardNth2phi,kmsub(Gt323,JacPDstandardNth3phi,JacPDstandardNth23phi)))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED cdphi233 = 
    kmsub(fac2,kmul(JacPDstandardNth3phi,JacPDstandardNth3phi),kmul(fac1,kmadd(Gt133,JacPDstandardNth1phi,kmadd(Gt233,JacPDstandardNth2phi,kmsub(Gt333,JacPDstandardNth3phi,JacPDstandardNth33phi)))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Rphi11 = 
    kmul(ToReal(-2),kadd(cdphi211,kmadd(kmul(cdphi1,cdphi1),kmul(kmadd(gt11L,gtu11,ToReal(-1)),ToReal(2)),kmul(gt11L,kmadd(cdphi211,gtu11,kmadd(cdphi233,gtu33,kmadd(kmadd(cdphi212,gtu12,kmadd(cdphi213,gtu13,kmadd(cdphi223,gtu23,kmul(gtu33,kmul(cdphi3,cdphi3))))),ToReal(2),kmadd(gtu22,kmadd(kmul(cdphi2,cdphi2),ToReal(2),cdphi222),kmul(kmadd(cdphi1,kmadd(cdphi2,gtu12,kmul(cdphi3,gtu13)),kmul(cdphi2,kmul(cdphi3,gtu23))),ToReal(4))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Rphi12 = 
    kmul(ToReal(-2),kadd(cdphi212,kmadd(gt12L,kmadd(cdphi211,gtu11,kmadd(kmadd(cdphi212,gtu12,kmadd(cdphi213,gtu13,kmadd(cdphi223,gtu23,kmul(gtu11,kmul(cdphi1,cdphi1))))),ToReal(2),kmadd(gtu22,kmadd(kmul(cdphi2,cdphi2),ToReal(2),cdphi222),kmadd(gtu33,kmadd(kmul(cdphi3,cdphi3),ToReal(2),cdphi233),kmul(cdphi2,kmul(cdphi3,kmul(gtu23,ToReal(4)))))))),kmul(cdphi1,kmadd(gt12L,kmul(cdphi3,kmul(gtu13,ToReal(4))),kmul(cdphi2,kmadd(gt12L,kmul(gtu12,ToReal(4)),ToReal(-2))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Rphi13 = 
    kmul(ToReal(-2),kadd(cdphi213,kmadd(gt13L,kmadd(cdphi211,gtu11,kmadd(kmadd(cdphi212,gtu12,kmadd(cdphi213,gtu13,kmadd(cdphi223,gtu23,kmul(gtu11,kmul(cdphi1,cdphi1))))),ToReal(2),kmadd(gtu22,kmadd(kmul(cdphi2,cdphi2),ToReal(2),cdphi222),kmadd(gtu33,kmadd(kmul(cdphi3,cdphi3),ToReal(2),cdphi233),kmul(cdphi2,kmul(cdphi3,kmul(gtu23,ToReal(4)))))))),kmul(cdphi1,kmadd(gt13L,kmul(cdphi2,kmul(gtu12,ToReal(4))),kmul(cdphi3,kmadd(gt13L,kmul(gtu13,ToReal(4)),ToReal(-2))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Rphi22 = 
    kmul(ToReal(-2),kadd(cdphi222,kmadd(kmul(cdphi2,cdphi2),kmul(kmadd(gt22L,gtu22,ToReal(-1)),ToReal(2)),kmul(gt22L,kmadd(cdphi222,gtu22,kmadd(cdphi233,gtu33,kmadd(kmadd(cdphi212,gtu12,kmadd(cdphi213,gtu13,kmadd(cdphi223,gtu23,kmul(gtu33,kmul(cdphi3,cdphi3))))),ToReal(2),kmadd(gtu11,kmadd(kmul(cdphi1,cdphi1),ToReal(2),cdphi211),kmul(kmadd(cdphi1,kmul(cdphi3,gtu13),kmul(cdphi2,kmadd(cdphi1,gtu12,kmul(cdphi3,gtu23)))),ToReal(4))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Rphi23 = 
    kmul(ToReal(-2),kadd(cdphi223,kmadd(gt23L,kmadd(cdphi222,gtu22,kmadd(kmadd(cdphi212,gtu12,kmadd(cdphi213,gtu13,kmadd(cdphi223,gtu23,kmul(gtu22,kmul(cdphi2,cdphi2))))),ToReal(2),kmadd(gtu11,kmadd(kmul(cdphi1,cdphi1),ToReal(2),cdphi211),kmadd(gtu33,kmadd(kmul(cdphi3,cdphi3),ToReal(2),cdphi233),kmul(cdphi1,kmul(cdphi3,kmul(gtu13,ToReal(4)))))))),kmul(cdphi2,kmadd(gt23L,kmul(cdphi1,kmul(gtu12,ToReal(4))),kmul(cdphi3,kmadd(gt23L,kmul(gtu23,ToReal(4)),ToReal(-2))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Rphi33 = 
    kmul(ToReal(-2),kadd(cdphi233,kmadd(kmul(cdphi3,cdphi3),kmul(kmadd(gt33L,gtu33,ToReal(-1)),ToReal(2)),kmul(gt33L,kmadd(cdphi233,gtu33,kmadd(kmadd(cdphi213,gtu13,kmul(cdphi223,gtu23)),ToReal(2),kmadd(gtu11,kmadd(kmul(cdphi1,cdphi1),ToReal(2),cdphi211),kmadd(gtu22,kmadd(kmul(cdphi2,cdphi2),ToReal(2),cdphi222),kmadd(cdphi3,kmul(kmadd(cdphi1,gtu13,kmul(cdphi2,gtu23)),ToReal(4)),kmul(gtu12,kmadd(cdphi212,ToReal(2),kmul(cdphi1,kmul(cdphi2,ToReal(4))))))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Atm11 = 
    kmadd(At11L,gtu11,kmadd(At12L,gtu12,kmul(At13L,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Atm21 = 
    kmadd(At11L,gtu12,kmadd(At12L,gtu22,kmul(At13L,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Atm31 = 
    kmadd(At11L,gtu13,kmadd(At12L,gtu23,kmul(At13L,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Atm12 = 
    kmadd(At12L,gtu11,kmadd(At22L,gtu12,kmul(At23L,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Atm22 = 
    kmadd(At12L,gtu12,kmadd(At22L,gtu22,kmul(At23L,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Atm32 = 
    kmadd(At12L,gtu13,kmadd(At22L,gtu23,kmul(At23L,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Atm13 = 
    kmadd(At13L,gtu11,kmadd(At23L,gtu12,kmul(At33L,gtu13)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Atm23 = 
    kmadd(At13L,gtu12,kmadd(At23L,gtu22,kmul(At33L,gtu23)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Atm33 = 
    kmadd(At13L,gtu13,kmadd(At23L,gtu23,kmul(At33L,gtu33)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED R11 = kadd(Rphi11,Rt11);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED R12 = kadd(Rphi12,Rt12);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED R13 = kadd(Rphi13,Rt13);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED R22 = kadd(Rphi22,Rt22);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED R23 = kadd(Rphi23,Rt23);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED R33 = kadd(Rphi33,Rt33);
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED trS = 
    kmul(em4phi,kmadd(eTxxL,gtu11,kmadd(eTyyL,gtu22,kmadd(eTzzL,gtu33,kmul(kmadd(eTxyL,gtu12,kmadd(eTxzL,gtu13,kmul(eTyzL,gtu23))),ToReal(2))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Ats11 = 
    kmadd(Gt211,JacPDstandardNth2alpha,kmadd(Gt311,JacPDstandardNth3alpha,kmadd(alphaL,R11,kmsub(JacPDstandardNth1alpha,kmadd(cdphi1,ToReal(4),Gt111),JacPDstandardNth11alpha))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Ats12 = 
    kmadd(Gt312,JacPDstandardNth3alpha,kmadd(alphaL,R12,ksub(kmadd(JacPDstandardNth2alpha,kmadd(cdphi1,ToReal(2),Gt212),kmul(JacPDstandardNth1alpha,kmadd(cdphi2,ToReal(2),Gt112))),JacPDstandardNth12alpha)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Ats13 = 
    kmadd(Gt213,JacPDstandardNth2alpha,kmadd(alphaL,R13,ksub(kmadd(JacPDstandardNth3alpha,kmadd(cdphi1,ToReal(2),Gt313),kmul(JacPDstandardNth1alpha,kmadd(cdphi3,ToReal(2),Gt113))),JacPDstandardNth13alpha)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Ats22 = 
    kmadd(Gt122,JacPDstandardNth1alpha,kmadd(Gt322,JacPDstandardNth3alpha,kmadd(alphaL,R22,kmsub(JacPDstandardNth2alpha,kmadd(cdphi2,ToReal(4),Gt222),JacPDstandardNth22alpha))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Ats23 = 
    kmadd(Gt123,JacPDstandardNth1alpha,kmadd(alphaL,R23,ksub(kmadd(JacPDstandardNth3alpha,kmadd(cdphi2,ToReal(2),Gt323),kmul(JacPDstandardNth2alpha,kmadd(cdphi3,ToReal(2),Gt223))),JacPDstandardNth23alpha)));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED Ats33 = 
    kmadd(Gt133,JacPDstandardNth1alpha,kmadd(Gt233,JacPDstandardNth2alpha,kmadd(alphaL,R33,kmsub(JacPDstandardNth3alpha,kmadd(cdphi3,ToReal(4),Gt333),JacPDstandardNth33alpha))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED trAts = 
    kmadd(Ats11,gu11,kmadd(Ats22,gu22,kmadd(Ats33,gu33,kmul(kmadd(Ats12,gu12,kmadd(Ats13,gu13,kmul(Ats23,gu23))),ToReal(2)))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED At11rhsL = 
    kmadd(At11L,kmul(kadd(JacPDstandardNth1beta1,kadd(JacPDstandardNth2beta2,JacPDstandardNth3beta3)),ToReal(-0.666666666666666666666666666667)),kmadd(em4phi,kmadd(g11,kmul(trAts,ToReal(-0.333333333333333333333333333333)),Ats11),kmadd(kmadd(At11L,JacPDstandardNth1beta1,kmadd(At12L,JacPDstandardNth1beta2,kmul(At13L,JacPDstandardNth1beta3))),ToReal(2),kmul(alphaL,kmadd(kmadd(At12L,Atm21,kmul(At13L,Atm31)),ToReal(-2),kmadd(At11L,kmadd(Atm11,ToReal(-2),trKL),kmul(em4phi,kmul(ToReal(-8),kmul(kmadd(g11,kmul(trS,ToReal(-0.333333333333333333333333333333)),eTxxL),ToReal(Pi))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED At12rhsL = 
    kmadd(At22L,JacPDstandardNth1beta2,kmadd(At23L,JacPDstandardNth1beta3,kmadd(At11L,JacPDstandardNth2beta1,kmadd(At13L,JacPDstandardNth2beta3,kmadd(At12L,kadd(JacPDstandardNth1beta1,kmadd(kadd(JacPDstandardNth1beta1,kadd(JacPDstandardNth2beta2,JacPDstandardNth3beta3)),ToReal(-0.666666666666666666666666666667),JacPDstandardNth2beta2)),kmadd(em4phi,kmadd(g12,kmul(trAts,ToReal(-0.333333333333333333333333333333)),Ats12),kmul(alphaL,kmadd(kmadd(At11L,Atm12,kmul(At13L,Atm32)),ToReal(-2),kmadd(At12L,kmadd(Atm22,ToReal(-2),trKL),kmul(em4phi,kmul(ToReal(-8),kmul(kmadd(g12,kmul(trS,ToReal(-0.333333333333333333333333333333)),eTxyL),ToReal(Pi)))))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED At13rhsL = 
    kmadd(At23L,JacPDstandardNth1beta2,kmadd(At33L,JacPDstandardNth1beta3,kmadd(At11L,JacPDstandardNth3beta1,kmadd(At12L,JacPDstandardNth3beta2,kmadd(At13L,kadd(JacPDstandardNth1beta1,kmadd(kadd(JacPDstandardNth1beta1,kadd(JacPDstandardNth2beta2,JacPDstandardNth3beta3)),ToReal(-0.666666666666666666666666666667),JacPDstandardNth3beta3)),kmadd(em4phi,kmadd(g13,kmul(trAts,ToReal(-0.333333333333333333333333333333)),Ats13),kmul(alphaL,kmadd(kmadd(At11L,Atm13,kmul(At12L,Atm23)),ToReal(-2),kmadd(At13L,kmadd(Atm33,ToReal(-2),trKL),kmul(em4phi,kmul(ToReal(-8),kmul(kmadd(g13,kmul(trS,ToReal(-0.333333333333333333333333333333)),eTxzL),ToReal(Pi)))))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED At22rhsL = 
    kmadd(At22L,kmul(kadd(JacPDstandardNth1beta1,kadd(JacPDstandardNth2beta2,JacPDstandardNth3beta3)),ToReal(-0.666666666666666666666666666667)),kmadd(em4phi,kmadd(g22,kmul(trAts,ToReal(-0.333333333333333333333333333333)),Ats22),kmadd(kmadd(At12L,JacPDstandardNth2beta1,kmadd(At22L,JacPDstandardNth2beta2,kmul(At23L,JacPDstandardNth2beta3))),ToReal(2),kmul(alphaL,kmadd(kmadd(At12L,Atm12,kmul(At23L,Atm32)),ToReal(-2),kmadd(At22L,kmadd(Atm22,ToReal(-2),trKL),kmul(em4phi,kmul(ToReal(-8),kmul(kmadd(g22,kmul(trS,ToReal(-0.333333333333333333333333333333)),eTyyL),ToReal(Pi))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED At23rhsL = 
    kmadd(At13L,JacPDstandardNth2beta1,kmadd(At33L,JacPDstandardNth2beta3,kmadd(At12L,JacPDstandardNth3beta1,kmadd(At22L,JacPDstandardNth3beta2,kmadd(At23L,kadd(JacPDstandardNth2beta2,kmadd(kadd(JacPDstandardNth1beta1,kadd(JacPDstandardNth2beta2,JacPDstandardNth3beta3)),ToReal(-0.666666666666666666666666666667),JacPDstandardNth3beta3)),kmadd(em4phi,kmadd(g23,kmul(trAts,ToReal(-0.333333333333333333333333333333)),Ats23),kmul(alphaL,kmadd(kmadd(At12L,Atm13,kmul(At22L,Atm23)),ToReal(-2),kmadd(At23L,kmadd(Atm33,ToReal(-2),trKL),kmul(em4phi,kmul(ToReal(-8),kmul(kmadd(g23,kmul(trS,ToReal(-0.333333333333333333333333333333)),eTyzL),ToReal(Pi)))))))))))));
  
  CCTK_REAL_VEC CCTK_ATTRIBUTE_UNUSED At33rhsL = 
    kmadd(At33L,kmul(kadd(JacPDstandardNth1beta1,kadd(JacPDstandardNth2beta2,JacPDstandardNth3beta3)),ToReal(-0.666666666666666666666666666667)),kmadd(em4phi,kmadd(g33,kmul(trAts,ToReal(-0.333333333333333333333333333333)),Ats33),kmadd(kmadd(At13L,JacPDstandardNth3beta1,kmadd(At23L,JacPDstandardNth3beta2,kmul(At33L,JacPDstandardNth3beta3))),ToReal(2),kmul(alphaL,kmadd(kmadd(At13L,Atm13,kmul(At23L,Atm23)),ToReal(-2),kmadd(At33L,kmadd(Atm33,ToReal(-2),trKL),kmul(em4phi,kmul(ToReal(-8),kmul(kmadd(g33,kmul(trS,ToReal(-0.333333333333333333333333333333)),eTzzL),ToReal(Pi))))))))));
  
  /* Copy local copies back to grid functions */
  vec_store_partial_prepare(i,lc_imin,lc_imax);
  vec_store_nta_partial(At11rhs[index],At11rhsL);
  vec_store_nta_partial(At12rhs[index],At12rhsL);
  vec_store_nta_partial(At13rhs[index],At13rhsL);
  vec_store_nta_partial(At22rhs[index],At22rhsL);
  vec_store_nta_partial(At23rhs[index],At23rhsL);
  vec_store_nta_partial(At33rhs[index],At33rhsL);
}
LC_ENDLOOP3VEC(ML_BSSN_CL_RHS2);
}
