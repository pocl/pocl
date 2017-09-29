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



_CL_OVERLOADABLE vtype acospi(vtype x) {
    // Computes arccos(x).
    // The argument is first reduced by noting that arccos(x)
    // is invalid for abs(x) > 1. For denormal and small
    // arguments arccos(x) = pi/2 to machine accuracy.
    // Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arccos(x) = pi/2 - arcsin(x)
    // = pi/2 - (x + x^3*R(x^2))
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.

    const vtype pi = (vtype)0x1.921fb54442d18p+1;
    const vtype piby2_tail = (vtype)6.12323399573676603587e-17;        /* 0x3c91a62633145c07 */

    vtype y = fabs(x);

    itype xneg = (as_itype(x) < (itype)0);
    itype xexp = (as_itype(y) >> 52) - (itype)EXPBIAS_DP64;

    // abs(x) >= 0.5
    itype transform = (xexp >= -1);

    // Transform y into the range [0,0.5)
    vtype r1 = (vtype)0.5 * ((vtype)1.0 - y);
    vtype s = sqrt(r1);
    vtype r = y * y;
    r = transform ? r1 : r;
    y = transform ? s : y;

    // Use a rational approximation for [0.0, 0.5]
    vtype un = pocl_fma(r,
                    pocl_fma(r,
                        pocl_fma(r,
                            pocl_fma(r,
                                pocl_fma(r, 0.0000482901920344786991880522822991,
                                       0.00109242697235074662306043804220),
                                -0.0549989809235685841612020091328),
                            0.275558175256937652532686256258),
                        -0.445017216867635649900123110649),
                    0.227485835556935010735943483075);

    vtype ud = pocl_fma(r,
                    pocl_fma(r,
                        pocl_fma(r,
                            pocl_fma(r, 0.105869422087204370341222318533,
                                   -0.943639137032492685763471240072),
                            2.76568859157270989520376345954),
                        -3.28431505720958658909889444194),
                    1.36491501334161032038194214209);

    vtype u = r * MATH_DIVIDE(un, ud);

    // Reconstruct acos carefully in transformed region
    vtype res1 = pocl_fma((vtype)-2.0,
                   MATH_DIVIDE(s + pocl_fma(y, u, -piby2_tail), pi),
                   (vtype)1.0);
    vtype s1 = as_vtype(as_utype(s) & (utype)0xffffffff00000000UL);
    vtype c = MATH_DIVIDE(pocl_fma(-s1, s1, r), s + s1);
    vtype res2 = MATH_DIVIDE(pocl_fma((vtype)2.0, s1, pocl_fma((vtype)2.0, c, (vtype)2.0 * y * u)), pi);
    res1 = xneg ? res1 : res2;
    res2 = (vtype)0.5 - pocl_fma(x, u, x) / pi;
    res1 = transform ? res1 : res2;

    const vtype qnan = as_vtype((utype)QNANBITPATT_DP64);
    res2 = (x == (vtype)1.0) ? (vtype)0.0 : qnan;
    res2 = (x == (vtype)-1.0) ? (vtype)1.0 : res2;
    res1 = (xexp >= (itype)0) ? res2 : res1;
    res1 = (xexp < (itype)-56) ? (vtype)0.5 : res1;

    return res1;
}
