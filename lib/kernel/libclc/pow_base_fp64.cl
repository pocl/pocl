/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See ROCM_LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml_helpers.h"
#include "singlevec.h"

extern _CL_OVERLOADABLE v2type MATH_PRIVATE(epln)(vtype);
extern _CL_OVERLOADABLE vtype MATH_PRIVATE(expep)(v2type);

vtype
#if defined(COMPILING_POWR)
MATH_MANGLE(powr)(vtype x, vtype y)
#elif defined(COMPILING_POWN)
MATH_MANGLE(pown)(vtype x, inttype ny)
#elif defined(COMPILING_ROOTN)
MATH_MANGLE(rootn)(vtype x, inttype ny)
#else
MATH_MANGLE(pow)(vtype x, vtype y)
#endif
{

    vtype ax = BUILTIN_ABS_F64(x);

#if defined(COMPILING_POWN)
    vtype y = convert_vtype(ny);
#elif defined(COMPILING_ROOTN)
    v2type y = rcp(convert_vtype(ny));
#else
    vtype ay = BUILTIN_ABS_F64(y);
    // flush denorm y to zero
    itype is_denorm = (as_itype(ay) < (itype)IMPBIT_DP64);
    y = (is_denorm) ? as_vtype(as_itype(y) & (itype)SIGNBIT_DP64) : y;
#endif

    vtype ret = MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));

#if defined(COMPILING_POWN) || defined(COMPILING_ROOTN)
    itype nyi = convert_itype(ny);
    itype is_odd = (nyi << 63);
#else
    vtype tay = BUILTIN_TRUNC_F64(ay);
    itype is_int = (ay == tay);
    vtype unused;
    itype is_odd = is_int ? (fract(tay*0.5, &unused) != vZERO_DP64) : (itype)0;

#ifdef SINGLEVEC
    if (is_odd && (ax > x))
      ret = copysign(ret, -0.0);
#else
    ret = copysign(ret, as_vtype(is_odd & (ax > x)));
#endif

#endif


    // edge cases
#if defined(COMPILING_POWN) || defined(COMPILING_ROOTN)
    vtype ret0 = (nyi > (itype)0) ? (vtype)0.0 : vINFINITY_DP64;
    vtype retI = (nyi > (itype)0) ? vINFINITY_DP64 : (vtype)0.0f;
    ret = (ax > (vtype)0.0) ? ret : ret0;
    ret = (isinf(x)) ? retI : ret;
    ret = (isnan(x)) ? x : ret;

    itype xneg = as_itype(x) & (itype)SIGNBIT_DP64;
    ret = (is_odd & xneg) ? copysign(ret, x) : ret;

#if defined(COMPILING_POWN)
    ret = (nyi == (itype)0) ? (vtype)1.0 : ret;
#elif defined(COMPILING_ROOTN)
    ret = (SV_NOT(is_odd) SV_AND (x < (vtype)0.0)) ? vNAN_DP64 : ret;
    ret = (nyi == (itype)0) ? vNAN_DP64: ret;
#endif

    return ret;
#else  /* POW / POWR */
    itype ax_eq_0 = (as_itype(ax) == as_itype(vZERO_DP64));
    itype ay_eq_0 = (as_itype(ay) == as_itype(vZERO_DP64));

    itype ax_lt_1 = (ax < vONE_DP64);
    itype ax_gt_1 = (ax > vONE_DP64);

    itype ax_eq_nan = (isnan(ax));
    itype ay_eq_nan = (isnan(ay));
    itype ay_eq_pinf = (ay == vINFINITY_DP64);

#ifdef SINGLEVEC
    itype y_pos = (as_itype(y) & (itype)SIGNBIT_DP64) == 0;
    itype x_neg = (as_itype(x) & (itype)SIGNBIT_DP64) != 0;
    itype y_neg = !y_pos;
#else
    itype x_neg = (as_itype(x) & (itype)SIGNBIT_DP64);
    itype y_neg = (as_itype(y) & (itype)SIGNBIT_DP64);
    itype y_pos = (y_neg ^ (itype)SIGNBIT_DP64);
#endif

    itype y_eq_ninf = (y == vNINFINITY_DP64);
    itype y_eq_pinf = (y == vINFINITY_DP64);

#endif /* POW / POWR */


#if defined(COMPILING_POWR)

    itype ax_eq_pinf = (ax == vINFINITY_DP64);
    itype ax_eq_1 = (ax == vONE_DP64);
    itype ay_lt_inf = (ay < vINFINITY_DP64);
    itype ax_lt_pinf = (ax < vINFINITY_DP64);
    itype ax_ne_0 = SV_NOT(ax_eq_0);

    #ifndef FINITE_MATH_ONLY
        ret = (ax_lt_1 & y_eq_ninf) ? vINFINITY_DP64 : ret;
        ret = (ax_lt_1 & y_eq_pinf) ? vZERO_DP64 : ret;
        ret = (ax_eq_1 & ay_lt_inf) ? vONE_DP64 : ret;
        ret = (ax_eq_1 & ay_eq_pinf) ? vNAN_DP64 : ret;
        ret = (ax_gt_1 & y_eq_ninf) ? vZERO_DP64 : ret;
        ret = (ax_gt_1 & y_eq_pinf) ? vINFINITY_DP64 : ret;

        ret = (ax_lt_pinf & ay_eq_0) ? vONE_DP64 : ret;

        ret = (ax_eq_pinf & y_neg) ? vZERO_DP64 : ret;
        ret = (ax_eq_pinf & y_pos) ? vINFINITY_DP64 : ret;
        ret = (ax_eq_pinf & y_eq_pinf) ? vINFINITY_DP64 : ret;
        ret = (ax_eq_pinf & ay_eq_0) ? vNAN_DP64 : ret;

        ret = (ax_eq_0 & y_neg) ? vINFINITY_DP64 : ret;
        ret = (ax_eq_0 & y_pos) ? vZERO_DP64 : ret;
        ret = (ax_eq_0 & ay_eq_0) ? vNAN_DP64 : ret;
        ret = (ax_ne_0 & x_neg) ? vNAN_DP64 : ret;

        ret = (ax_eq_nan) ? x : ret;
        ret = (ay_eq_nan) ? y : ret;
    #else
        ret = (ax_eq_1) ? vONE_DP64 : ret;
        ret = (ay_eq_0) ? vONE_DP64 : ret;
        ret = (ax_eq_0 & y_pos) ? vZERO_DP64 : ret;
    #endif

    return ret;
#endif

#if defined(COMPILING_POW)

    itype is_not_int = SV_NOT(is_int);
    itype is_not_odd = SV_NOT(is_odd);

    itype x_eq_ninf = (x == vNINFINITY_DP64);
    itype x_eq_pinf = (x == vINFINITY_DP64);

    #ifndef FINITE_MATH_ONLY
        vtype xinf = copysign(vINFINITY_DP64, x);
        vtype xzero = copysign(vZERO_DP64, x);

        ret = (x_neg & is_not_int) ? vNAN_DP64 : ret;

        ret = (ax_lt_1 & y_eq_ninf) ? vINFINITY_DP64 : ret;
        ret = (ax_gt_1 & y_eq_ninf) ? vZERO_DP64 : ret;
        ret = (ax_lt_1 & y_eq_pinf) ? vZERO_DP64 : ret;
        ret = (ax_gt_1 & y_eq_pinf) ? vINFINITY_DP64 : ret;

        ret = (ax_eq_0 & y_neg & is_odd) ? xinf : ret;
        ret = (ax_eq_0 & y_neg & is_not_odd) ? vINFINITY_DP64 : ret;
        ret = (ax_eq_0 & y_pos & is_odd) ? xzero : ret;
        ret = (ax_eq_0 & y_pos & is_not_odd) ? vZERO_DP64 : ret;
        ret = (ax_eq_0 & y_eq_ninf) ? vINFINITY_DP64 : ret;

        ret = ((x == (vtype)(-1.0)) & ay_eq_pinf) ? vONE_DP64 : ret;

        ret = (x_eq_ninf & y_neg & is_odd) ? (vtype)(-0.0) : ret;
        ret = (x_eq_ninf & y_neg & is_not_odd) ? vZERO_DP64 : ret;
        ret = (x_eq_ninf & y_pos & is_odd) ? vNINFINITY_DP64 : ret;
        ret = (x_eq_ninf & y_pos & is_not_odd) ? vINFINITY_DP64 : ret;
        ret = (x_eq_pinf & y_neg) ? vZERO_DP64 : ret;
        ret = (x_eq_pinf & y_pos) ? vINFINITY_DP64 : ret;
        ret = (ax_eq_nan) ? x : ret;
        ret = (ay_eq_nan) ? y : ret;

    #else
        // XXX work around conformance test incorrectly checking these cases
        vtype xinf = copysign(vINFINITY_DP64, x);
        ret = (ax_eq_0 & y_neg & is_odd) ? xinf : ret;
        ret = (ax_eq_0 & y_neg & is_not_odd) ? vINFINITY_DP64 : ret;

        vtype xzero = copysign(0.0f, x);
        ret = (ax_eq_0 & y_pos & is_odd) ? xzero : ret;
        ret = (ax_eq_0 & y_pos & is_not_odd) ? vZERO_DP64 : ret;
    #endif

    ret = ay_eq_0 ? vONE_DP64 : ret;
    ret = (x == vONE_DP64) ? vONE_DP64 : ret;
    return ret;

#endif

}
