/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See ROCM_LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "ocml_helpers.h"

v2type _CL_OVERLOADABLE
MATH_PRIVATE(epln)(vtype a)
{
    vtype m = BUILTIN_FREXP_MANT_F32(a);
    itype b = (m < (vtype)(2.0f/3.0f)) ? (itype)1 : (itype)0;
    m = BUILTIN_FLDEXP_F32(m, b);
    itype e = BUILTIN_FREXP_EXP_F32(a) - b;

    v2type x = div(m - (vtype)1.0f, add(m, (vtype)1.0f));
    v2type s = sqr(x);
    vtype t = s.hi;
    vtype p = MATH_MAD(t, MATH_MAD(t, (vtype)0x1.ed89c2p-3f,
                      (vtype)0x1.23e988p-2f), (vtype)0x1.999bdep-2f);

    // ln(2)*e + 2*x + x^3(c3 + x^2*p)
    v2type r = add(mul(con((vtype)0x1.62e430p-1f, (vtype)-0x1.05c610p-29f),
                 convert_vtype(e)),
                   fadd(ldx(x,1),
                      mul(mul(s, x),
                        fadd(con((vtype)0x1.555554p-1f,
                                 (vtype)0x1.e72020p-29f),
                             mul(s, p)))));

    return r;
}


vtype _CL_OVERLOADABLE
MATH_PRIVATE(expep)(v2type x)
{
    vtype fn = BUILTIN_RINT_F32(x.hi * 0x1.715476p+0f);
    v2type t = fsub(fsub(sub(x, fn*0x1.62e400p-1f), fn*0x1.7f7800p-20f), fn*0x1.473de6p-34f);

    vtype th = t.hi;
    vtype p = MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th,
                  (vtype)0x1.6850e4p-10f, (vtype)0x1.123bccp-7f),
                  (vtype)0x1.555b98p-5f), (vtype)0x1.55548ep-3f),
                  (vtype)0x1.fffff8p-2f);

    v2type r = fadd(t, mul(sqr(t), p));
    vtype z = (vtype)1.0f + r.hi;

    z = BUILTIN_FLDEXP_F32(z, convert_inttype(fn));

    z = (x.hi > (vtype)89.0f) ? as_vtype((utype)PINFBITPATT_SP32) : z;
    z = (x.hi < (vtype)-104.0f) ? (vtype)0.0f : z;

    return z;
}
