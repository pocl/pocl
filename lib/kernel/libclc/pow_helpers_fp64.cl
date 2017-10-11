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
    vtype m = BUILTIN_FREXP_MANT_F64(a);
    itype b = (m < (vtype)(2.0/3.0)) ? (itype)1 : (itype)0;
    m = BUILTIN_FLDEXP_F64(m, convert_inttype(b));
    itype e = BUILTIN_FREXP_EXP_F64(a) - b;

    v2type x = div(m - (vtype)1.0, add(m, (vtype)1.0));
    v2type s = sqr(x);
    vtype t = s.hi;
    vtype p = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
               MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                   (vtype)0x1.dee674222de17p-4, (vtype)0x1.a6564968915a9p-4),
                   (vtype)0x1.e25e43abe935ap-4), (vtype)0x1.110ef47e6c9c2p-3),
                   (vtype)0x1.3b13bcfa74449p-3), (vtype)0x1.745d171bf3c30p-3),
                   (vtype)0x1.c71c71c7792cep-3), (vtype)0x1.24924924920dap-2),
                   (vtype)0x1.999999999999cp-2);

    // ln(2)*e + 2*x + x^3(c3 + x^2*p)
    v2type r = add(mul(con((vtype)0x1.62e42fefa39efp-1, (vtype)0x1.abc9e3b39803fp-56), convert_vtype(e)),
                    fadd(ldx(x,1),
                          mul(mul(s, x),
                              fadd(con((vtype)0x1.5555555555555p-1,(vtype)0x1.543b0d5df274dp-55),
                                   mul(s, p)))));

    return r;
}

vtype _CL_OVERLOADABLE
MATH_PRIVATE(expep)(v2type x)
{
    vtype dn = BUILTIN_RINT_F64(x.hi * 0x1.71547652b82fep+0);
    v2type t = fsub(fsub(sub(x, dn*0x1.62e42fefa3000p-1),
                  dn*0x1.3de6af278e000p-42), dn*0x1.9cc01f97b57a0p-83);

    vtype th = t.hi;
    vtype p = MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th,
               MATH_MAD(th, MATH_MAD(th, MATH_MAD(th, MATH_MAD(th,
               MATH_MAD(th,
                   (vtype)0x1.ade156a5dcb37p-26, (vtype)0x1.28af3fca7ab0cp-22),
                   (vtype)0x1.71dee623fde64p-19), (vtype)0x1.a01997c89e6b0p-16),
                   (vtype)0x1.a01a014761f6ep-13), (vtype)0x1.6c16c1852b7b0p-10),
                   (vtype)0x1.1111111122322p-7), (vtype)0x1.55555555502a1p-5),
                   (vtype)0x1.5555555555511p-3), (vtype)0x1.000000000000bp-1);

    v2type r = fadd(t, mul(sqr(t), p));
    vtype z = (vtype)1.0 + r.hi;

    z = BUILTIN_FLDEXP_F64(z, convert_inttype(dn));

    z = (x.hi > (vtype)710.0) ? as_vtype((utype)PINFBITPATT_DP64) : z;
    z = (x.hi < (vtype)-745.0) ? (vtype)0.0 : z;

    return z;
}
