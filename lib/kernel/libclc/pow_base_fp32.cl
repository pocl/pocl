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
MATH_MANGLE(pown)(vtype x, itype ny)
#elif defined(COMPILING_ROOTN)
MATH_MANGLE(rootn)(vtype x, itype ny)
#else
MATH_MANGLE(pow)(vtype x, vtype y)
#endif
{

    vtype ax = BUILTIN_ABS_F32(x);

#if defined(COMPILING_POWN) || defined(COMPILING_ROOTN)
    itype nyh = ny & (itype)0xffff0000;
    v2type y = fadd(convert_vtype(nyh), convert_vtype(ny - nyh));
#if defined(COMPILING_ROOTN)
    y = rcp(y);
#endif
#else
    vtype ay = BUILTIN_ABS_F32(y);
    // flush denorm y to zero
    itype is_denorm = (as_itype(ay) < (itype)IMPBIT_SP32);
    y = (is_denorm) ? as_vtype(as_itype(y) & (itype)SIGNBIT_SP32) : y;
#endif

    vtype ret = MATH_PRIVATE(expep)(omul(y, MATH_PRIVATE(epln)(ax)));

#if defined(COMPILING_POWN) || defined(COMPILING_ROOTN)
    itype is_odd = (ny << 31);
#else

    vtype tay = BUILTIN_TRUNC_F32(ay);
    itype is_int = (ay == tay);
    vtype unused;
    itype is_odd = is_int ? (fract(tay*0.5f, &unused) != vZERO_SP32) : (itype)0;

#ifdef SINGLEVEC
    if (is_odd && (ax > x))
      ret = copysign(ret, -0.0f);
#else
    ret = copysign(ret, as_vtype(is_odd & (ax > x)));
#endif

#endif



    // edge cases
#if defined(COMPILING_POWN) || defined(COMPILING_ROOTN)
    vtype ret0 = (ny > (itype)0) ? vZERO_SP32 : vINFINITY_SP32;
    vtype retI = (ny > (itype)0) ? vINFINITY_SP32 : vZERO_SP32;
    ret = (ax == vZERO_SP32) ? ret0 : ret;
    ret = (isinf(x)) ? retI : ret;

    itype xneg = as_itype(x) & (itype)SIGNBIT_SP32;
    ret = (is_odd & xneg) ? copysign(ret, x) : ret;

#if defined(COMPILING_POWN)
    ret = (ny == (itype)0) ? vONE_SP32 : ret;
#elif defined(COMPILING_ROOTN)
    ret = (SV_NOT(is_odd) SV_AND (x < vZERO_SP32)) ? vNAN_SP32 : ret;
    ret = (ny == (itype)0) ? vNAN_SP32 : ret;
#endif

    return ret;

#else /* POW / POWR */

    itype ax_eq_0 = (as_itype(ax) == as_itype(vZERO_SP32));
    itype ay_eq_0 = (as_itype(ay) == as_itype(vZERO_SP32));

    itype ax_lt_1 = (ax < vONE_SP32);
    itype ax_gt_1 = (ax > vONE_SP32);

    itype ax_eq_nan = (isnan(ax));
    itype ay_eq_nan = (isnan(ay));
    itype ay_eq_pinf = (ay == vINFINITY_SP32);

#ifdef SINGLEVEC
    itype y_pos = (as_itype(y) & (itype)SIGNBIT_SP32) == 0;
    itype x_neg = (as_itype(x) & (itype)SIGNBIT_SP32) != 0;
    itype y_neg = !y_pos;
#else
    itype x_neg = (as_itype(x) & (itype)SIGNBIT_SP32);
    itype y_neg = (as_itype(y) & (itype)SIGNBIT_SP32);
    itype y_pos = (y_neg ^ (itype)SIGNBIT_SP32);
#endif

    itype y_eq_ninf = (y == vNINFINITY_SP32);
    itype y_eq_pinf = (y == vINFINITY_SP32);


#endif /* POW / POWR */

#if defined(COMPILING_POWR)

    itype ax_eq_pinf = (ax == vINFINITY_SP32);
    itype ax_eq_1 = (ax == vONE_SP32);
    itype ay_lt_inf = (ay < vINFINITY_SP32);
    itype ax_lt_pinf = (ax < vINFINITY_SP32);
    itype ax_ne_0 = SV_NOT(ax_eq_0);

    #ifndef FINITE_MATH_ONLY
        ret = (ax_lt_1 & y_eq_ninf) ? vINFINITY_SP32 : ret;
        ret = (ax_lt_1 & y_eq_pinf) ? vZERO_SP32 : ret;
        ret = (ax_eq_1 & ay_lt_inf) ? vONE_SP32 : ret;
        ret = (ax_eq_1 & ay_eq_pinf) ? vNAN_SP32 : ret;
        ret = (ax_gt_1 & y_eq_ninf) ? vZERO_SP32 : ret;
        ret = (ax_gt_1 & y_eq_pinf) ? vINFINITY_SP32 : ret;

        ret = (ax_lt_pinf & ay_eq_0) ? vONE_SP32 : ret;

        ret = (ax_eq_pinf & y_neg) ? vZERO_SP32 : ret;
        ret = (ax_eq_pinf & y_pos) ? vINFINITY_SP32 : ret;
        ret = (ax_eq_pinf & y_eq_pinf) ? vINFINITY_SP32 : ret;
        ret = (ax_eq_pinf & ay_eq_0) ? vNAN_SP32 : ret;

        ret = (ax_eq_0 & y_neg) ? vINFINITY_SP32 : ret;
        ret = (ax_eq_0 & y_pos) ? vZERO_SP32 : ret;
        ret = (ax_eq_0 & ay_eq_0) ? vNAN_SP32 : ret;
        ret = (ax_ne_0 & x_neg) ? vNAN_SP32 : ret;

        ret = (ax_eq_nan) ? x : ret;
        ret = (ay_eq_nan) ? y : ret;

    #else
        ret = (ax_eq_1) ? vONE_SP32 : ret;
        ret = (ay_eq_0) ? vONE_SP32 : ret;
        ret = (ax_eq_0 & y_pos) ? vZERO_SP32 : ret;
    #endif

    return ret;
#endif

#if defined(COMPILING_POW)

    itype is_not_int = SV_NOT(is_int);
    itype is_not_odd = SV_NOT(is_odd);

    itype x_eq_ninf = (x == vNINFINITY_SP32);
    itype x_eq_pinf = (x == vINFINITY_SP32);

    #ifndef FINITE_MATH_ONLY
        vtype xinf = copysign(vINFINITY_SP32, x);
        vtype xzero = copysign(vZERO_SP32, x);

        ret = (x_neg & is_not_int) ? vNAN_SP32 : ret;

        ret = (ax_lt_1 & y_eq_ninf) ? vINFINITY_SP32 : ret;
        ret = (ax_gt_1 & y_eq_ninf) ? vZERO_SP32 : ret;
        ret = (ax_lt_1 & y_eq_pinf) ? vZERO_SP32 : ret;
        ret = (ax_gt_1 & y_eq_pinf) ? vINFINITY_SP32 : ret;

        ret = (ax_eq_0 & y_neg & is_odd) ? xinf : ret;
        ret = (ax_eq_0 & y_neg & is_not_odd) ? vINFINITY_SP32 : ret;
        ret = (ax_eq_0 & y_pos & is_odd) ? xzero : ret;
        ret = (ax_eq_0 & y_pos & is_not_odd) ? vZERO_SP32 : ret;
        ret = (ax_eq_0 & y_eq_ninf) ? vINFINITY_SP32 : ret;

        ret = ((x == (vtype)-1.0f) & ay_eq_pinf) ? vONE_SP32 : ret;

        ret = (x_eq_ninf & y_neg & is_odd) ? (vtype)-0.0f : ret;
        ret = (x_eq_ninf & y_neg & is_not_odd) ? vZERO_SP32 : ret;
        ret = (x_eq_ninf & y_pos & is_odd) ? vNINFINITY_SP32 : ret;
        ret = (x_eq_ninf & y_pos & is_not_odd) ? vINFINITY_SP32 : ret;
        ret = (x_eq_pinf & y_neg) ? vZERO_SP32 : ret;
        ret = (x_eq_pinf & y_pos) ? vINFINITY_SP32 : ret;
        ret = (ax_eq_nan) ? x : ret;
        ret = (ay_eq_nan) ? y : ret;

    #else
        // XXX work around conformance test incorrectly checking these cases
        vtype xinf = copysign(vINFINITY_SP32, x);
        ret = (ax_eq_0 & y_neg & is_odd) ? xinf : ret;
        ret = (ax_eq_0 & y_neg & is_not_odd) ? vINFINITY_SP32 : ret;

        vtype xzero = copysign(0.0f, x);
        ret = (ax_eq_0 & y_pos & is_odd) ? xzero : ret;
        ret = (ax_eq_0 & y_pos & is_not_odd) ? vZERO_SP32 : ret;
    #endif

    ret = ay_eq_0 ? vONE_SP32 : ret;
    ret = (x == vONE_SP32) ? vONE_SP32 : ret;
    return ret;


#endif

}
