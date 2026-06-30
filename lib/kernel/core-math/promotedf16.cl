#include "common_types.h"

float _CL_OVERLOADABLE _cl_modf (float, private float *);
float _CL_OVERLOADABLE _cl_remquo (float, float, private int *);

/* FP16 overloads for math builtins without a native FP16 path -- neither a
   Clang/LLVM FP16 builtin nor a SLEEF FP16 routine. Unlike additionalf16.cl
   (the __builtin_elementwise_* overloads, which the vectorized generic <fn>.cl
   files already provide via __builtin_<fn>f16 when
   ENABLE_HOST_CPU_VECTORIZE_BUILTINS=ON), nothing else in the kernel library
   supplies these in either configuration, so this file is compiled in both the
   vectorized and non-vectorized builds.
 *
 * These overloads either promote to float and compute with pocl's own float
 * builtin before rounding back to half, or, for nextafter, compute directly on
 * the half bit pattern. This keeps the calls inside the kernel library instead
 * of lowering to unresolved host libm symbols.
 *
 * The float computation calls the OpenCL builtin (e.g. `logb`), which the
 * kernel-library rename (_builtin_renames.h) maps to pocl's own `_cl_logb`. It
 * must not use `__builtin_logbf`: that has no LLVM intrinsic, so at the kernel
 * library's -O0 it lowers to a call to the libm symbol `logbf`, which the
 * kernel library does not provide -- producing
 * "Cannot find symbol logbf in kernel library" when a half kernel is built.
 * (The half overload below renames to `_cl_logb(half)`; the `logb((float)a)`
 * call resolves by overload to `_cl_logb(float)`, so there is no recursion.) */

/* logb : half -> half */
half _CL_OVERLOADABLE logb (half a) { return (half)logb ((float)a); }
DEFINE_FP16_EXPR_V_V (logb)

/* ilogb : half -> int (scalar), halfN -> intN (vector) */
int _CL_OVERLOADABLE ilogb (half a) { return ilogb ((float)a); }
#define IMPLEMENT_FP16_ILOGB(ITYPE, HTYPE, NAMEN)                            \
  ITYPE _CL_OVERLOADABLE ilogb (HTYPE a) { return (ITYPE)NAMEN (ilogb); }
IMPLEMENT_FP16_ILOGB (int2, half2, NAME1_2)
IMPLEMENT_FP16_ILOGB (int3, half3, NAME1_3)
IMPLEMENT_FP16_ILOGB (int4, half4, NAME1_4)
IMPLEMENT_FP16_ILOGB (int8, half8, NAME1_8)
IMPLEMENT_FP16_ILOGB (int16, half16, NAME1_16)

/* ldexp : (half, int) -> half */
half _CL_OVERLOADABLE ldexp (half a, int b) { return (half)ldexp ((float)a, b); }
DEFINE_FP16_EXPR_V_VI (ldexp)

/* rootn : (half, int) -> half */
half _CL_OVERLOADABLE rootn (half a, int b) { return (half)rootn ((float)a, b); }
DEFINE_FP16_EXPR_V_VI (rootn)

/* pown : (half, int) -> half. (Relocated from powf16.cl, which is compiled only
   in the non-vectorized build; pown has no native FP16 builtin so it is needed
   in both.) */
half _CL_OVERLOADABLE pown (half a, int b) { return (half)pown ((float)a, b); }
DEFINE_FP16_EXPR_V_VI (pown)

/* remainder : (half, half) -> half */
half _CL_OVERLOADABLE remainder (half a, half b) { return (half)remainder ((float)a, (float)b); }
DEFINE_FP16_EXPR_V_VV (remainder)

/* modf : (halfN, address-space halfN*) -> halfN */
half _CL_OVERLOADABLE
modf (half x, private half *iptr)
{
  float fip;
  half r = (half)modf ((float)x, &fip);
  *iptr = (half)fip;
  return r;
}

#define IMPLEMENT_FP16_MODF_AS(TYPE, AS)                                     \
  TYPE _CL_OVERLOADABLE modf (TYPE x, AS TYPE *iptr)                         \
  { TYPE t; TYPE r = modf (x, &t); *iptr = t; return r; }

#define IMPLEMENT_FP16_MODF_VECTOR(TYPE, LTYPE, HTYPE, LO, HI)               \
  TYPE _CL_OVERLOADABLE modf (TYPE x, private TYPE *iptr)                    \
  {                                                                          \
    LTYPE l;                                                                 \
    HTYPE h;                                                                 \
    TYPE r = (TYPE)(modf (x.LO, &l), modf (x.HI, &h));                       \
    iptr->LO = l;                                                            \
    iptr->HI = h;                                                            \
    return r;                                                                \
  }                                                                          \
  IMPLEMENT_FP16_MODF_AS (TYPE, local)                                       \
  IMPLEMENT_FP16_MODF_AS (TYPE, global)                                      \
  IF_GEN_AS (IMPLEMENT_FP16_MODF_AS (TYPE, generic))

IMPLEMENT_FP16_MODF_AS (half, local)
IMPLEMENT_FP16_MODF_AS (half, global)
IF_GEN_AS (IMPLEMENT_FP16_MODF_AS (half, generic))
IMPLEMENT_FP16_MODF_VECTOR (half2, half, half, lo, hi)
IMPLEMENT_FP16_MODF_VECTOR (half3, half2, half, lo, s2)
IMPLEMENT_FP16_MODF_VECTOR (half4, half2, half2, lo, hi)
IMPLEMENT_FP16_MODF_VECTOR (half8, half4, half4, lo, hi)
IMPLEMENT_FP16_MODF_VECTOR (half16, half8, half8, lo, hi)

/* remquo : (halfN, halfN, address-space intN*) -> halfN */
half _CL_OVERLOADABLE
remquo (half x, half y, private int *quo)
{
  return (half)remquo ((float)x, (float)y, quo);
}

#define IMPLEMENT_FP16_REMQUO_AS(TYPE, ITYPE, AS)                            \
  TYPE _CL_OVERLOADABLE remquo (TYPE x, TYPE y, AS ITYPE *quo)               \
  { ITYPE t; TYPE r = remquo (x, y, &t); *quo = t; return r; }

#define IMPLEMENT_FP16_REMQUO_VECTOR(TYPE, ITYPE, LITYPE, HITYPE, LO, HI)    \
  TYPE _CL_OVERLOADABLE remquo (TYPE x, TYPE y, private ITYPE *quo)          \
  {                                                                          \
    LITYPE l;                                                                \
    HITYPE h;                                                                \
    TYPE r = (TYPE)(remquo (x.LO, y.LO, &l), remquo (x.HI, y.HI, &h));       \
    quo->LO = l;                                                             \
    quo->HI = h;                                                             \
    return r;                                                                \
  }                                                                          \
  IMPLEMENT_FP16_REMQUO_AS (TYPE, ITYPE, local)                              \
  IMPLEMENT_FP16_REMQUO_AS (TYPE, ITYPE, global)                             \
  IF_GEN_AS (IMPLEMENT_FP16_REMQUO_AS (TYPE, ITYPE, generic))

IMPLEMENT_FP16_REMQUO_AS (half, int, local)
IMPLEMENT_FP16_REMQUO_AS (half, int, global)
IF_GEN_AS (IMPLEMENT_FP16_REMQUO_AS (half, int, generic))
IMPLEMENT_FP16_REMQUO_VECTOR (half2, int2, int, int, lo, hi)
IMPLEMENT_FP16_REMQUO_VECTOR (half3, int3, int2, int, lo, s2)
IMPLEMENT_FP16_REMQUO_VECTOR (half4, int4, int2, int2, lo, hi)
IMPLEMENT_FP16_REMQUO_VECTOR (half8, int8, int4, int4, lo, hi)
IMPLEMENT_FP16_REMQUO_VECTOR (half16, int16, int8, int8, lo, hi)

/* powr : (half, half) -> half. powr(x,y) = pow(x,y) for x >= 0, NaN for x < 0.
   Built on the half `pow` builtin -- which is available in both configurations
   (CORE-Math powf16.cl when vectorization is off, the generic vectorized pow.cl
   when it is on) -- so this works in both. (Relocated from powf16.cl, which is
   compiled only in the non-vectorized build.) */
half _CL_OVERLOADABLE
powr (half x, half y)
{
  ushort xu = as_ushort (x);
  x = (xu == (ushort)0x8000) ? (half)0 : x;       /* -0 -> +0 */
  half r = pow (x, y);
  r = (xu > (ushort)0x8000) ? (half)NAN : r;       /* x < 0 -> NaN */
  r = isnan (x) ? (half)NAN : r;
  r = isnan (y) ? (half)NAN : r;
  return r;
}
DEFINE_FP16_EXPR_V_VV (powr)

/* nextafter : (half, half) -> half. Computed directly on the half bit pattern;
   float promotion is wrong here (it would step to the next float, not the next
   half). Mirrors sleef-pocl/nextafter.cl (pocl#2210). */
half _CL_OVERLOADABLE
nextafter (half x, half y)
{
  const short sign_bit = as_short ((ushort)0x8000);
  const short sign_bit_mask = 0x7fff;

  short ix = as_short (x);
  short ax = ix & sign_bit_mask;
  short mx = sign_bit - ix;
  mx = ix < (short)0 ? mx : ix;

  short iy = as_short (y);
  short ay = iy & sign_bit_mask;
  short my = sign_bit - iy;
  my = iy < (short)0 ? my : iy;

  short t = mx + (mx < my ? 1 : -1);
  short r = sign_bit - t;
  r = t < (short)0 ? r : t;

  if (t == 0 && ix < 0)
    r = sign_bit;

  r = isnan (x) ? ix : r;
  r = isnan (y) ? iy : r;
  r = (((ax | ay) == (short)0) | (ix == iy)) ? iy : r;
  return as_half (r);
}
DEFINE_FP16_EXPR_V_VV (nextafter)
