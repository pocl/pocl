/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See ROCM_LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml_helpers.h"

// How many bits of the quotient per iteration
#define X_GT_Y_BITS (itype)12
#define BITS 12

#define APPEND_AS(x, y) APPEND_AS2(x, y)
#define APPEND_AS2(x, y) x ## y

#if defined(COMPILING_REMQUO)
#define X_GT_Y APPEND_AS(x_gt_y_remquo_, ADDRSPACE)
#define X_LT_Y APPEND_AS(x_lt_y_remquo_, ADDRSPACE)
#elif defined(COMPILING_FMOD)
#define X_GT_Y x_gt_y_fmod
#define X_LT_Y x_lt_y_fmod
#else //remainder
#define X_GT_Y x_gt_y_remainder
#define X_LT_Y x_lt_y_remainder
#endif

#if defined(COMPILING_REMQUO)
OCML_ATTR vtype X_GT_Y(vtype x, vtype y, vtype ax, vtype ay, itype *q7) {
#else
OCML_ATTR vtype X_GT_Y(vtype x, vtype y, vtype ax, vtype ay) {
#endif
    itype ex, ey;

    ex = BUILTIN_FREXP_EXP_F32(ax) - (itype)1;
    ax = BUILTIN_FLDEXP_F32(BUILTIN_FREXP_MANT_F32(ax), X_GT_Y_BITS);
    ey = BUILTIN_FREXP_EXP_F32(ay) - (itype)1;
    ay = BUILTIN_FLDEXP_F32(BUILTIN_FREXP_MANT_F32(ay), 1);
    vtype axN = ax;

    itype nb = ex - ey;
    itype nbN = nb;
    vtype ayinv = MATH_FAST_RCP(ay);

#if !defined(COMPILING_FMOD)
    itype qacc = (itype)0;
#endif
#if defined(COMPILING_REMQUO)
    itype qaccN = qacc;
#endif

    while (SV_ANY(nb > X_GT_Y_BITS)) {
        vtype q = BUILTIN_RINT_F32(ax * ayinv);
        axN = fnma(q, ay, ax);
        itype clt = (axN < (vtype)0.0f);
        axN = clt ? (axN + ay) : axN;
#if defined(COMPILING_REMQUO)
        itype iq = convert_itype(q);
        iq = clt ? (iq-(itype)1) : iq;
        qaccN = (qacc << BITS) | iq;
#endif
        axN = BUILTIN_FLDEXP_F32(axN, X_GT_Y_BITS);
        nbN -= X_GT_Y_BITS;

        itype cond = (nb > X_GT_Y_BITS);
        nb = cond ? nbN : nb;
        ax = cond ? axN : ax;
#if defined(COMPILING_REMQUO)
        qacc = cond ? qaccN : qacc;
#endif
    }

    ax = BUILTIN_FLDEXP_F32(ax, (nb - X_GT_Y_BITS + (itype)1));

    // Final iteration
    {
        vtype q = BUILTIN_RINT_F32(ax * ayinv);
        ax = fnma(q, ay, ax);
        itype clt = (ax < (vtype)0.0f);
        ax = clt ? (ax + ay) : ax;
#if !defined(COMPILING_FMOD)
        itype iq = convert_itype(q);
        iq = clt ? (iq-(itype)1) : iq;
#if defined(COMPILING_REMQUO)
        qacc = (qacc << (nb+(itype)1)) | iq;
#else
        qacc = iq;
#endif
#endif
    }

#if !defined(COMPILING_FMOD)
    // Adjust ax so that it is the range (-y/2, y/2]
    // We need to choose the even integer when x/y is midway between two integer
    itype qacc_is_odd = SV_ODD32(qacc);
    itype aq = ((2.0f*ax > ay) SV_OR ((qacc_is_odd) SV_AND ((2.0f*ax) == ay)));
    ax = ax - (aq ? ay : (vtype)0.0f);
#if defined(COMPILING_REMQUO)
    qacc += (aq ? (itype)1 : (itype)0);
    itype qneg = (as_itype(x) ^ as_itype(y)) >> 31;
    *q7 = ((qacc & (itype)0x7f) ^ qneg) - qneg;
#endif
#endif

    ax = BUILTIN_FLDEXP_F32(ax, ey);
    return as_vtype( ((as_itype(x) & (itype)SIGNBIT_SP32) ^ as_itype(ax)) );
}

#undef X_GT_Y_BITS
#undef BITS

/***************************************************************************/
/***************************************************************************/
/***************************************************************************/

#if defined(COMPILING_REMQUO)
OCML_ATTR vtype X_LT_Y(vtype x, vtype y, vtype ax, vtype ay, itype *q7o) {
#else
OCML_ATTR vtype X_LT_Y(vtype x, vtype y, vtype ax, vtype ay) {
#endif
    vtype ret = x;
#if defined(COMPILING_REMQUO)
    itype q7 = (itype)0;
#endif

#if !defined(COMPILING_FMOD)
    itype c = ((ay < (vtype)0x1.0p+127f) & (2.0f*ax > ay)) | (ax > 0.5f*ay);

    itype qsgn = (itype)1 + (((as_itype(x) ^ as_itype(y)) >> 31) << 1);
    vtype t = MATH_MAD(y, convert_vtype(-qsgn), x);
    ret = c ? t : ret;
#if defined(COMPILING_REMQUO)
    q7 = c ? qsgn : q7;
#endif
#endif

    ret = (ax == ay) ? copysign((vtype)0.0f, x) : ret;
#if defined(COMPILING_REMQUO)
    q7 = (ax == ay) ? qsgn : q7;
    *q7o = q7;
#endif
    return ret;
}

/***************************************************************************/
/***************************************************************************/
/***************************************************************************/


#if defined(COMPILING_FMOD)
CONSTATTR vtype
MATH_MANGLE(fmod)(vtype x, vtype y)
#elif defined(COMPILING_REMQUO)
vtype
MATH_MANGLE(remquo)(vtype x, vtype y, ADDRSPACE itype *q7p)
#else
CONSTATTR vtype
MATH_MANGLE(remainder)(vtype x, vtype y)
#endif
{

    vtype ax = BUILTIN_ABS_F32(x);
    vtype ay = BUILTIN_ABS_F32(y);
    vtype ret;

#if defined(COMPILING_REMQUO)
    itype q7, q7_gt, q7_lt;
    vtype x_gt = X_GT_Y(x, y, ax, ay, &q7_gt);
    vtype x_lt = X_LT_Y(x, y, ax, ay, &q7_lt);
    q7 = (ax > ay) ? q7_gt : q7_lt;
#else
    vtype x_gt = X_GT_Y(x, y, ax, ay);
    vtype x_lt = X_LT_Y(x, y, ax, ay);
#endif
    ret = (ax > ay) ? x_gt : x_lt;

    itype c = (isnan(y) SV_OR isinf(x) SV_OR isnan(x) SV_OR (y == (vtype)0.0f));
    ret = c ? as_vtype((utype)QNANBITPATT_SP32) : ret;

#if defined(COMPILING_REMQUO)
    q7 = c ? (itype)0 : q7;
    *q7p = q7;
#endif

    return ret;
}

#undef APPEND_AS
