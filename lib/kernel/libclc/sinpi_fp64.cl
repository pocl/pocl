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


_CL_OVERLOADABLE vtype sinpi(vtype x)
{
    itype ix = as_itype(x);
    itype xsgn = ix & (itype)SIGNBIT_DP64;
    ix ^= xsgn;
    vtype ax = as_vtype(ix);
    vtype iaxv = trunc(ax);
    itype iaxi = convert_itype(iaxv);
    vtype r = ax - iaxv;
    itype xodd = xsgn ^ ((iaxi & (itype)1) << 63);

    // Initialize with return for +-Inf and NaN
    itype ir = (itype)QNANBITPATT_DP64;

    // 2^23 <= |x| < Inf, the result is always integer
    ir = (ix < (itype)EXPBITS_DP64) ? xsgn : ir;

    // 0x1.0p-7 <= |x| < 2^23, result depends on which 0.25 interval

    // r < 1.0
    vtype a = (vtype)1.0 - r;
    itype e = (itype)0;

    // r <= 0.75
    itype c = (r <= (vtype)0.75);
    a = c ? (r - (vtype)0.5) : a;
    e = c ? (itype)(-1) : e;

    // r < 0.5
    c = (r < (vtype)0.5);
    a = c ? ((vtype)0.5 - r) : a;

    // r <= 0.25
    c = (r <= 0.25);
    a = c ? r : a;
    e = c ? (itype)0 : e;

    v2type sc = __pocl_sincos_piby4(a * M_PI, (vtype)0.0);
    itype jr = xodd ^ as_itype(e ? sc.hi : sc.lo);

    ir = (ax < (vtype)0x1.0p+52) ? jr : ir;

    return as_vtype(ir);
}
