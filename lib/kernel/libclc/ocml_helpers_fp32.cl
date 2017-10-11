/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See ROCM_LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#define FLOAT_SPECIALIZATION
#include "ocml_helpers.h"
#include "ocml_helpers_impl.cl"
#undef FLOAT_SPECIALIZATION

// The arguments must only be variable names
#define FULL_MUL(A, B, CHI, CLO) \
    do { \
        vtype __ha = as_vtype(as_utype(A) & (utype)0xfffff000U); \
        vtype __ta = A - __ha; \
        vtype __hb = as_vtype(as_utype(B) & (utype)0xfffff000U); \
        vtype __tb = B - __hb; \
        CHI = A * B; \
        CLO = MATH_MAD(__ta, __tb, MATH_MAD(__ta, __hb, MATH_MAD(__ha, __tb, MATH_MAD(__ha, __hb, -CHI)))); \
    } while (0)


OCML_ATTR vtype
fnma(vtype a, vtype b, vtype c)
{
    vtype d;
    if (HAVE_FMA32) {
        d = BUILTIN_FMA_F32(-a, b, c);
    } else {
        vtype h, t;
        FULL_MUL(a, b, h, t);
        d = c - h;
        d = (((c - d) - h) - t) + d;
    }
    return d;
}

#undef FULL_MUL
