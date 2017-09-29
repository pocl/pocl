/*
 * Copyright (c) 2014,2015 Advanced Micro Devices, Inc.
 *
 * Copyright (c) 2017 Michal Babej / Tampere University of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */




_CL_OVERLOADABLE vtype atanpi(vtype x) {
    const vtype pi = (vtype)M_PI_F;

    utype ux = as_utype(x);
    utype aux = ux & (utype)EXSIGNBIT_SP32;
    utype sx = ux ^ aux;

    vtype xbypi = MATH_DIVIDE(x, pi);
    vtype shalf = as_vtype(sx | as_utype((vtype)0.5f));

    vtype v = as_vtype(aux);

    // Return for NaN
    vtype ret = x;

    // 2^26 <= |x| <= Inf => atan(x) is close to piby2
    ret = (aux <= (utype)PINFBITPATT_SP32) ? shalf : ret;

    // Reduce arguments 2^-19 <= |x| < 2^26

    // 39/16 <= x < 2^26
    x = -MATH_RECIP(v);
    vtype c = (vtype)1.57079632679489655800f; // atan(infinity)

    // 19/16 <= x < 39/16
    itype l = (aux < (utype)0x401c0000);
    vtype xx = MATH_DIVIDE(v - (vtype)1.5f, pocl_fma(v, (vtype)1.5f, (vtype)1.0f));
    x = l ? xx : x;
    c = l ? (vtype)9.82793723247329054082e-1f : c; // atan(1.5)

    // 11/16 <= x < 19/16
    l = (aux < (utype)0x3f980000U);
    xx =  MATH_DIVIDE(v - (vtype)1.0f, (vtype)1.0f + v);
    x = l ? xx : x;
    c = l ? (vtype)7.85398163397448278999e-1f : c; // atan(1)

    // 7/16 <= x < 11/16
    l = (aux < (utype)0x3f300000);
    xx = MATH_DIVIDE(pocl_fma(v, (vtype)2.0f, (vtype)-1.0f), (vtype)2.0f + v);
    x = l ? xx : x;
    c = l ? (vtype)4.63647609000806093515e-1f: c; // atan(0.5)

    // 2^-19 <= x < 7/16
    l = (aux < (utype)0x3ee00000);
    x = l ? v : x;
    c = l ? (vtype)0.0f : c;

    // Core approximation: Remez(2,2) on [-7/16,7/16]

    vtype s = x * x;
    vtype a = pocl_fma(s,
                  pocl_fma(s,
                    (vtype)0.470677934286149214138357545549e-2f,
                    (vtype)0.192324546402108583211697690500f),
                  (vtype)0.296528598819239217902158651186f);

    vtype b = pocl_fma(s,
                  pocl_fma(s,
                    (vtype)0.299309699959659728404442796915f,
                    (vtype)0.111072499995399550138837673349e1f),
                  (vtype)0.889585796862432286486651434570f);

    vtype q = x * s * MATH_DIVIDE(a, b);

    vtype z = c - (q - x);
    z = MATH_DIVIDE(z, pi);
    vtype zs = as_vtype(sx | as_utype(z));

    ret  = (aux< (utype)0x4c800000) ? zs : ret;

    // |x| < 2^e
    ret = (aux< (utype)0x36000000) ? xbypi : ret;
    return ret;
}
