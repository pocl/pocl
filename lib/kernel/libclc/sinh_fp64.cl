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



_CL_OVERLOADABLE vtype sinh(vtype x)
{
    // After dealing with special cases the computation is split into
    // regions as follows:
    //
    // abs(x) >= max_sinh_arg:
    // sinh(x) = sign(x)*Inf
    //
    // abs(x) >= small_threshold:
    // sinh(x) = sign(x)*exp(abs(x))/2 computed using the
    // splitexp and scaleDouble functions as for exp_amd().
    //
    // abs(x) < small_threshold:
    // compute p = exp(y) - 1 and then z = 0.5*(p+(p/(p+1.0)))
    // sinh(x) is then sign(x)*z.

    const vtype max_sinh_arg = 7.10475860073943977113e+02; // 0x408633ce8fb9f87e

    // This is where exp(-x) is insignificant compared to exp(x) = ln(2^27)
    const vtype small_threshold = 0x1.2b708872320e2p+4;

    vtype y = fabs(x);

    // In this range we find the integer part y0 of y
    // and the increment dy = y - y0. We then compute
    // z = sinh(y) = sinh(y0)cosh(dy) + cosh(y0)sinh(dy)
    // where sinh(y0) and cosh(y0) are obtained from tables

    vtype indv = trunc(y);
    itype indi = convert_itype(indv);
    indi = min((itype)indi, (itype)36U);

    vtype dy = y - indv;
    vtype dy2 = dy * dy;

    vtype sdy = dy * dy2 *
            pocl_fma(dy2,
              pocl_fma(dy2,
                pocl_fma(dy2,
                  pocl_fma(dy2,
                    pocl_fma(dy2,
                      pocl_fma(dy2,
                 (vtype)0.7746188980094184251527126e-12,
                 (vtype)0.160576793121939886190847e-9),
               (vtype)0.250521176994133472333666e-7),
             (vtype)0.275573191913636406057211e-5),
           (vtype)0.198412698413242405162014e-3),
         (vtype)0.833333333333329931873097e-2),
       (vtype)0.166666666666666667013899e0);

    vtype cdy = dy2 *
            pocl_fma(dy2,
              pocl_fma(dy2,
                pocl_fma(dy2,
                  pocl_fma(dy2,
                    pocl_fma(dy2,
                      pocl_fma(dy2,
                 (vtype)0.1163921388172173692062032e-10,
                 (vtype)0.208744349831471353536305e-8),
               (vtype)0.275573350756016588011357e-6),
             (vtype)0.248015872460622433115785e-4),
           (vtype)0.138888888889814854814536e-2),
         (vtype)0.416666666666660876512776e-1),
       (vtype)0.500000000000000005911074e0);

    // At this point sinh(dy) is approximated by dy + sdy.
    // Shift some significant bits from dy to sdy.
    vtype sdy1 = as_vtype(as_utype(dy) & 0xfffffffff8000000UL);
    vtype sdy2 = sdy + (dy - sdy1);

    v2type tv = USE_VTABLE(cosh_tbl, convert_uinttype(indi));
    vtype cl = tv.lo;
    vtype ct = tv.hi;

    tv = USE_VTABLE(sinh_tbl, convert_uinttype(indi));
    vtype sl = tv.lo;
    vtype st = tv.hi;


    vtype z = pocl_fma(cl, sdy1,
                pocl_fma(sl, cdy,
                  pocl_fma(cl, sdy2,
                    pocl_fma(ct, sdy1,
                      pocl_fma(st, cdy, ct*sdy2)) + st))) + sl;

    // Other cases
    z = (y < 0x1.0p-28) | isnan(x) | isinf(x) ? y : z;

    vtype t = exp(y - 0x1.62e42fefa3800p-1);
    t = pocl_fma(t, (vtype)-0x1.ef35793c76641p-45, t);
    z = (y >= small_threshold) ? t : z;
    z = (y >= max_sinh_arg) ? as_vtype((utype)PINFBITPATT_DP64) : z;

    return copysign(z, x);
}
