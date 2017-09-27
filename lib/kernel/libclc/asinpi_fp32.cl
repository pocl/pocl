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



_CL_OVERLOADABLE vtype asinpi(vtype x) {
    // Computes arcsin(x).
    // The argument is first reduced by noting that arcsin(x)
    // is invalid for abs(x) > 1 and arcsin(-x) = -arcsin(x).
    // For denormal and small arguments arcsin(x) = x to machine
    // accuracy. Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arcsin(x) = x + x^3*R(x^2)
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arcsin(x) = pi/2 - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.


    const vtype pi = (vtype)3.1415926535897933e+00f;
    const vtype piby2_tail = (vtype)7.5497894159e-08F;   /* 0x33a22168 */
    const vtype hpiby2_head = (vtype)7.8539812565e-01F;  /* 0x3f490fda */

    utype ux = as_utype(x);
    utype aux = ux & (utype)EXSIGNBIT_SP32;
    utype xs = ux ^ aux;
    vtype shalf = as_vtype(xs | as_utype((vtype)0.5f));
    itype xexp = as_itype(aux >> EXPSHIFTBITS_SP32) - (itype)EXPBIAS_SP32;
    vtype y = as_vtype(aux);

    // abs(x) >= 0.5
    itype transform = (xexp >= (itype)-1);

    vtype y2 = y * y;
    vtype rt = (vtype)0.5f * ((vtype)1.0f - y);
    vtype r = transform ? rt : y2;

    // Use a rational approximation for [0.0, 0.5]
    vtype a = pocl_fma(r,
                pocl_fma(r,
                  pocl_fma(r,
                    (vtype)-0.00396137437848476485201154797087F,
                    (vtype)-0.0133819288943925804214011424456F),
                  (vtype)-0.0565298683201845211985026327361F),
                (vtype)0.184161606965100694821398249421F);
    vtype b = pocl_fma(r,
                (vtype)-0.836411276854206731913362287293F,
                (vtype)1.10496961524520294485512696706F);
    vtype u = r * MATH_DIVIDE(a, b);

    vtype s = MATH_SQRT(r);
    vtype s1 = as_vtype(as_utype(s) & (utype)0xffff0000);
    vtype c = MATH_DIVIDE(pocl_fma(-s1, s1, r), s + s1);
    vtype p = pocl_fma((vtype)2.0f * s, u, -pocl_fma(c, (vtype)-2.0f, piby2_tail));
    vtype q = pocl_fma(s1, (vtype)-2.0f, hpiby2_head);
    vtype vt = hpiby2_head - (p - q);
    vtype v = pocl_fma(y, u, y);
    v = transform ? vt : v;
    v = MATH_DIVIDE(v, pi);
    vtype xbypi = MATH_DIVIDE(x, pi);

    vtype ret = as_vtype(xs | as_utype(v));
    ret = (aux > (utype)0x3f800000U) ? as_vtype((utype)QNANBITPATT_SP32) : ret;
    ret = (aux == (utype)0x3f800000U) ? shalf : ret;
    ret = (xexp < (itype)-14) ? xbypi : ret;

    return ret;
}
