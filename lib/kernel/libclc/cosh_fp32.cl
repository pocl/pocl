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

_CL_OVERLOADABLE vtype cosh(vtype x) {

    // After dealing with special cases the computation is split into regions as follows.
    // abs(x) >= max_cosh_arg:
    // cosh(x) = sign(x)*Inf
    // abs(x) >= small_threshold:
    // cosh(x) = sign(x)*exp(abs(x))/2 computed using the
    // splitexp and scaleDouble functions as for exp_amd().
    // abs(x) < small_threshold:
    // compute p = exp(y) - 1 and then z = 0.5*(p+(p/(p+1.0)))
    // cosh(x) is then z.

    const vtype max_cosh_arg = (vtype)0x1.65a9fap+6f;
    const vtype small_threshold = (vtype)0x1.0a2b24p+3f;

    utype ux = as_utype(x);
    utype aux = ux & (utype)EXSIGNBIT_SP32;
    vtype y = as_vtype(aux);

    // Find the integer part y0 of y and the increment dy = y - y0. We then compute
    // z = sinh(y) = sinh(y0)cosh(dy) + cosh(y0)sinh(dy)
    // z = cosh(y) = cosh(y0)cosh(dy) + sinh(y0)sinh(dy)
    // where sinh(y0) and cosh(y0) are tabulated above.

    vtype indv = trunc(y);
    utype indi = convert_utype(indv);
    indi = (indi > (utype)36) ? (utype)0 : indi;

    vtype dy = y - indv;
    vtype dy2 = dy * dy;

    vtype sdy = pocl_fma(dy2,
                    pocl_fma(dy2,
                        pocl_fma(dy2,
                            pocl_fma(dy2,
                                pocl_fma(dy2,
                                    pocl_fma(dy2,
                                      (vtype)0.7746188980094184251527126e-12f,
                                      (vtype)0.160576793121939886190847e-9f),
                                    (vtype)0.250521176994133472333666e-7f),
                                (vtype)0.275573191913636406057211e-5f),
                            (vtype)0.198412698413242405162014e-3f),
                        (vtype)0.833333333333329931873097e-2f),
                    (vtype)0.166666666666666667013899e0f);
    sdy = pocl_fma(sdy, dy*dy2, dy);

    vtype cdy = pocl_fma(dy2,
                    pocl_fma(dy2,
                        pocl_fma(dy2,
                            pocl_fma(dy2,
                                pocl_fma(dy2,
                                    pocl_fma(dy2,
                                      (vtype)0.1163921388172173692062032e-10f,
                                      (vtype)0.208744349831471353536305e-8f),
                                    (vtype)0.275573350756016588011357e-6f),
                                (vtype)0.248015872460622433115785e-4f),
                            (vtype)0.138888888889814854814536e-2f),
                        (vtype)0.416666666666660876512776e-1f),
                    (vtype)0.500000000000000005911074e0f);

    cdy = pocl_fma(cdy, dy2, (vtype)1.0f);

    v2type tv = USE_VTABLE(sinhcosh_tbl, indi);
    vtype z = pocl_fma(tv.lo, sdy, tv.hi * cdy);

    // When exp(-x) is insignificant compared to exp(x), return exp(x)/2
    vtype t = exp(y - (vtype)0x1.62e500p-1f);
    vtype zsmall = pocl_fma((vtype)0x1.a0210ep-18f, t, t);
    z = (y >= small_threshold) ? zsmall : z;

    // Corner cases
    z = (y >= max_cosh_arg) ? as_vtype((utype)PINFBITPATT_SP32) : z;
    z = (aux > (utype)PINFBITPATT_SP32) ? as_vtype((utype)QNANBITPATT_SP32) : z;
    z = (aux < (utype)0x38800000) ? (vtype)1.0f : z;

    return z;
}
