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

    const vtype pi = (vtype)0x1.921fb54442d18p+1;
    const vtype piby2_tail = (vtype)6.1232339957367660e-17; /* 0x3c91a62633145c07 */
    const vtype hpiby2_head = (vtype)7.8539816339744831e-01;  /* 0x3fe921fb54442d18 */

    vtype y = fabs(x);
    itype xneg = (as_itype(x) < (itype)0);
    itype xexp = (as_itype(y) >> 52) - (itype)EXPBIAS_DP64;

    // abs(x) >= 0.5
    itype transform = (xexp >= (itype)-1);

    vtype rt = (vtype)0.5 * ((vtype)1.0 - y);
    vtype y2 = y * y;
    vtype r = transform ? rt : y2;

    // Use a rational approximation for [0.0, 0.5]
    vtype un = pocl_fma(r,
                    pocl_fma(r,
                        pocl_fma(r,
                            pocl_fma(r,
                                pocl_fma(r, (vtype)0.0000482901920344786991880522822991,
                                       (vtype)0.00109242697235074662306043804220),
                                (vtype)-0.0549989809235685841612020091328),
                            (vtype)0.275558175256937652532686256258),
                        (vtype)-0.445017216867635649900123110649),
                    (vtype)0.227485835556935010735943483075);

    vtype ud = pocl_fma(r,
                    pocl_fma(r,
                        pocl_fma(r,
                            pocl_fma(r, (vtype)0.105869422087204370341222318533,
                                   (vtype)-0.943639137032492685763471240072),
                            (vtype)2.76568859157270989520376345954),
                        (vtype)-3.28431505720958658909889444194),
                    (vtype)1.36491501334161032038194214209);

    vtype u = r * MATH_DIVIDE(un, ud);


    // Reconstruct asin carefully in transformed region
    vtype s = sqrt(r);
    vtype sh = as_vtype(as_utype(s) & (utype)0xffffffff00000000UL);
    vtype c = MATH_DIVIDE(pocl_fma(-sh, sh, r), s + sh);
    vtype p = pocl_fma((2.0 * s), u, -pocl_fma((vtype)-2.0, c, piby2_tail));
    vtype q = pocl_fma((vtype)-2.0, sh, hpiby2_head);
    vtype vt = hpiby2_head - (p - q);
    vtype v = pocl_fma(y, u, y);
    v = transform ? vt : v;

    v = (xexp < (itype)-28) ? y : v;
    v = MATH_DIVIDE(v, pi);
    v = (xexp >= (itype)0) ? as_vtype((utype)QNANBITPATT_DP64) : v;
    v = (y == (vtype)1.0) ? (vtype)0.5 : v;
    return xneg ? -v : v;
}
