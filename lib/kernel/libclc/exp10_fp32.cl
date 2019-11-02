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


//    Algorithm:
//
//    e^x = 2^(x/ln(2)) = 2^(x*(64/ln(2))/64)
//
//    x*(64/ln(2)) = n + f, |f| <= 0.5, n is integer
//    n = 64*m + j,   0 <= j < 64
//
//    e^x = 2^((64*m + j + f)/64)
//        = (2^m) * (2^(j/64)) * 2^(f/64)
//        = (2^m) * (2^(j/64)) * e^(f*(ln(2)/64))
//
//    f = x*(64/ln(2)) - n
//    r = f*(ln(2)/64) = x - n*(ln(2)/64)
//
//    e^x = (2^m) * (2^(j/64)) * e^r
//
//    (2^(j/64)) is precomputed
//
//    e^r = 1 + r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
//    e^r = 1 + q
//
//    q = r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
//
//    e^x = (2^m) * ( (2^(j/64)) + q*(2^(j/64)) )


_CL_OVERLOADABLE vtype exp10(vtype x)
{
    const vtype X_MAX = (vtype) 0x1.344134p+5f; // 128*log2/log10 : 38.53183944498959
    const vtype X_MIN = (vtype)-0x1.66d3e8p+5f; // -149*log2/log10 : -44.8534693539332

    const vtype R_64_BY_LOG10_2 = (vtype)0x1.a934f0p+7f; // 64*log10/log2 : 212.6033980727912
    const vtype R_LOG10_2_BY_64_LD = (vtype)-0x1.340000p-8f; // log2/(64 * log10) lead : 0.004699707
    const vtype R_LOG10_2_BY_64_TL = (vtype)-0x1.04d426p-18f; // log2/(64 * log10) tail : 0.00000388665057
    const vtype R_LN10 = (vtype)0x1.26bb1cp+1f;

    itype return_nan = isnan(x);
    itype return_inf = x > X_MAX;
    itype return_zero = x < X_MIN;

    itype n = convert_itype(x * R_64_BY_LOG10_2);

    vtype fn = convert_vtype(n);
    utype j = as_utype(n & (itype)0x3f);
    itype m = n >> 6;
    itype m2 = m << EXPSHIFTBITS_SP32;
    vtype r;

    r = R_LN10 * mad(fn, R_LOG10_2_BY_64_TL, mad(fn, R_LOG10_2_BY_64_LD, x));

    // Truncated Taylor series for e^r
    vtype z2 = mad(mad(mad(r, (vtype)0x1.555556p-5f, (vtype)0x1.555556p-3f), r, (vtype)0x1.000000p-1f), r*r, r);

    vtype two_to_jby64 = USE_VTABLE(exp_tbl, j);

    z2 = mad(two_to_jby64, z2, two_to_jby64);

    vtype z2s = z2 * as_vtype((itype)0x1 << (m + (itype)149));
    vtype z2n = as_vtype(as_itype(z2) + m2);
    z2 = (m <= (itype)-126) ? z2s : z2n;


    z2 = return_inf ? as_vtype((utype)PINFBITPATT_SP32) : z2;
    z2 = return_zero ? (vtype)0.0f : z2;
    z2 = return_nan ? x : z2;
    return z2;
}
