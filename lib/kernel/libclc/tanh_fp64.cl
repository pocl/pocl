/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
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

_CL_OVERLOADABLE vtype tanh(vtype x)
{
    // The definition of tanh(x) is sinh(x)/cosh(x), which is also equivalent
    // to the following three formulae:
    // 1.  (exp(x) - exp(-x))/(exp(x) + exp(-x))
    // 2.  (1 - (2/(exp(2*x) + 1 )))
    // 3.  (exp(2*x) - 1)/(exp(2*x) + 1)
    // but computationally, some formulae are better on some ranges.

    // The point at which e^-x is insignificant compared to e^x = ln(2^27)
    const vtype large_threshold = (vtype)0x1.2b708872320e2p+4;

    utype ux = as_utype(x);
    utype ax = ux & (utype)EXSIGNBIT_DP64;
    utype sx = ux ^ ax;
    vtype y = as_vtype(ax);
    vtype y2 = y * y;

    // y < 0.9
    vtype znl = pocl_fma(y2,
                     pocl_fma(y2,
                       pocl_fma(y2,
                         (vtype)-0.142077926378834722618091e-7,
                         (vtype)-0.200047621071909498730453e-3),
                       (vtype)-0.176016349003044679402273e-1),
                     (vtype)-0.274030424656179760118928e0);

    vtype zdl = pocl_fma(y2,
                     pocl_fma(y2,
                       pocl_fma(y2,
                         (vtype)0.2091140262529164482568557e-3,
                         (vtype)0.201562166026937652780575e-1),
                       (vtype)0.381641414288328849317962e0),
                     (vtype)0.822091273968539282568011e0);

    // 0.9 <= y <= 1
    vtype znm = pocl_fma(y2,
                     pocl_fma(y2,
                         pocl_fma(y2,
                         (vtype)-0.115475878996143396378318e-7,
                         (vtype)-0.165597043903549960486816e-3),
                       (vtype)-0.146173047288731678404066e-1),
                     (vtype)-0.227793870659088295252442e0);

    vtype zdm = pocl_fma(y2,
                     pocl_fma(y2,
                       pocl_fma(y2,
                         (vtype)0.173076050126225961768710e-3,
                         (vtype)0.167358775461896562588695e-1),
                       (vtype)0.317204558977294374244770e0),
                     (vtype)0.683381611977295894959554e0);

    itype c = (y < (vtype)0.9);
    vtype zn = c ? znl : znm;
    vtype zd = c ? zdl : zdm;
    vtype z = y + y*y2 * MATH_DIVIDE(zn, zd);

    // y > 1
    vtype p = exp(2.0 * y) + (vtype)1.0;
    vtype zg = (vtype)1.0 - ((vtype)2.0 / p);

    z = (y > (vtype)1.0) ? zg : z;

    // Other cases
    z = (y < (vtype)0x1.0p-28) ? x : z;
    z = (ax > (utype)PINFBITPATT_DP64) ? x : z;

    z = (y > large_threshold) ? (vtype)1.0 : z;

    return as_vtype(sx | as_utype(z));
}
