/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
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


_CL_OVERLOADABLE vtype hypot(vtype x, vtype y)
{
    utype ux = as_utype(x) & (utype)(~SIGNBIT_DP64);
    itype xexp = as_itype(ux >> (utype)EXPSHIFTBITS_DP64);
    x = as_vtype(ux);

    utype uy = as_utype(y) & (utype)(~SIGNBIT_DP64);
    itype yexp = as_itype(uy >> (utype)EXPSHIFTBITS_DP64);
    y = as_vtype(uy);

    int c = (xexp > (itype)(EXPBIAS_DP64 + 500) | (yexp > (itype)(EXPBIAS_DP64 + 500));
    vtype preadjust = c ? (vtype)0x1.0p-600 : (vtype)1.0;
    vtype postadjust = c ? (vtype)0x1.0p+600 : (vtype)1.0;

    c = (xexp < (itype)(EXPBIAS_DP64 - 500) | (yexp < (itype)(EXPBIAS_DP64 - 500));
    preadjust = c ? (vtype)0x1.0p+600 : preadjust;
    postadjust = c ? (vtype)0x1.0p-600 : postadjust;

    vtype ax = x * preadjust;
    vtype ay = y * preadjust;

    // The post adjust may overflow, but this can't be avoided in any case
    vtype r = sqrt(fma(ax, ax, ay*ay)) * postadjust;

    // If the difference in exponents between x and y is large
    vtype s = x + y;
    c = abs(xexp - yexp) > (itype)(MANTLENGTH_DP64 + 1);
    r = c ? s : r;

    // Check for NaN
    //c = x != x | y != y;
    c = isnan(x) | isnan(y);
    r = c ? as_vtype((utype)QNANBITPATT_DP64) : r;

    // If either is Inf, we must return Inf
    c = x == as_vtype((utype)PINFBITPATT_DP64) | y == as_vtype((utype)PINFBITPATT_DP64);
    r = c ? as_vtype((utype)PINFBITPATT_DP64) : r;

    return r;
}
