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

_CL_OVERLOADABLE vtype acosh(vtype x) {
    const vtype recrteps = (vtype)0x1.6a09e667f3bcdp+26;  // 1/sqrt(eps) = (vtype)9.49062656242515593767e+07
    //log2_lead and log2_tail sum to an extra-precise version of log(2)
    const vtype log2_lead = (vtype)0x1.62e42ep-1;
    const vtype log2_tail = (vtype)0x1.efa39ef35793cp-25;

    // Handle x >= 128 here
    itype xlarge = (x > recrteps);
    vtype r = x + sqrt(pocl_fma(x, x, (vtype)-1.0));
    r = xlarge ? x : r;

    itype xexp;
    vtype r1, r2;
    __pocl_ep_log(r, &xexp, &r1, &r2);

    itype xlarge2 = (x > recrteps) ? (itype)1 : (itype)0;
    vtype dxexp = convert_vtype(xexp + xlarge2);
    r1 = pocl_fma(dxexp, log2_lead, r1);
    r2 = pocl_fma(dxexp, log2_tail, r2);

    vtype ret1 = r1 + r2;

    // Handle 1 < x < 128 here
    // We compute the value
    // t = x - 1.0 + sqrt(2.0*(x - 1.0) + (x - 1.0)*(x - 1.0))
    // using simulated quad precision.
    vtype t = x - (vtype)1.0;
    vtype u1 = t * 2.0;

    // (t,0) * (t,0) -> (v1, v2)
    vtype v1 = t * t;
    vtype v2 = pocl_fma(t, t, -v1);

    // (u1,0) + (v1,v2) -> (w1,w2)
    r = u1 + v1;
    vtype s = (((u1 - r) + v1) + v2);
    vtype w1 = r + s;
    vtype w2 = (r - w1) + s;

    // sqrt(w1,w2) -> (u1,u2)
    vtype p1 = sqrt(w1);
    vtype a1 = p1*p1;
    vtype a2 = pocl_fma(p1, p1, -a1);
    vtype temp = (((w1 - a1) - a2) + w2);
    vtype p2 = MATH_DIVIDE(temp * 0.5, p1);
    u1 = p1 + p2;
    vtype u2 = (p1 - u1) + p2;

    // (u1,u2) + (t,0) -> (r1,r2)
    r = u1 + t;
    s = ((u1 - r) + t) + u2;
    // r1 = r + s;
    // r2 = (r - r1) + s;
    // t = r1 + r2;
    t = r + s;

    // For arguments 1.13 <= x <= 1.5 the log1p function is good enough
    vtype ret2 = log1p(t);

    utype ux = as_utype(x);
    vtype ret = (x >= (vtype)128.0) ? ret1 : ret2;

    ret = (x == (vtype)1.0) ? (vtype)0.0 : ret;

    ret = (ux >= (utype)EXPBITS_DP64) ? x : ret;

    vtype nans = as_vtype((utype)QNANBITPATT_DP64);
    itype retnans = ((ux & (utype)(SIGNBIT_DP64)) != 0);
    retnans |= ((itype)(x < (vtype)1.0));
    ret = retnans ? nans : ret;

    return ret;
}
