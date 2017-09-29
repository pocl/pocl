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



_CL_OVERLOADABLE vtype atan2pi(vtype y, vtype x) {
    const vtype pi = (vtype)3.1415926535897932e+00;          /* 0x400921fb54442d18 */
    const vtype pi_head = (vtype)3.1415926218032836e+00;     /* 0x400921fb50000000 */
    const vtype pi_tail = (vtype)3.1786509547056392e-08;     /* 0x3e6110b4611a6263 */
    const vtype piby2_head = (vtype)1.5707963267948965e+00;  /* 0x3ff921fb54442d18 */
    const vtype piby2_tail = (vtype)6.1232339957367660e-17;  /* 0x3c91a62633145c07 */

    vtype x2 = x;
    itype xneg = (as_itype(x) & (itype)SIGNBIT_DP64);
    itype xexp = ((as_itype(x) & (itype)EXPBITS_DP64) >> 52);
    vtype y2 = y;
    itype yneg = (as_itype(y) & (itype)SIGNBIT_DP64);
    itype yexp = ((as_itype(y) & (itype)EXPBITS_DP64) >> 52);
    itype diffexp = yexp - xexp;

    // Scale up both x and y if they are both below 1/4
    vtype x1 = ldexp(x, 1024);
    itype xexp1 = ((as_itype(x1) & (itype)EXPBITS_DP64) >> 52);
    vtype y1 = ldexp(y, 1024);
    itype yexp1 = ((as_itype(y1) & (itype)EXPBITS_DP64) >> 52);
    itype diffexp1 = yexp1 - xexp1;

    itype cond2 = (xexp < 1021) & (yexp < 1021);
    diffexp = cond2 ? diffexp1 : diffexp;
    x = cond2 ? x1 : x;
    y = cond2 ? y1 : y;

    // General case: take absolute values of arguments
    vtype u = fabs(x);
    vtype v = fabs(y);

    // Swap u and v if necessary to obtain 0 < v < u. Compute v/u.
    itype swap_vu = u < v;
    vtype uu = u;
    u = swap_vu ? v : u;
    v = swap_vu ? uu : v;

    vtype vbyu = v / u;
    vtype q1, q2;

    // General values of v/u. Use a look-up table and series expansion.
    {
        vtype val = (vbyu > (vtype)0.0625) ? vbyu : (vtype)0.063;
        itype index = convert_itype(pocl_fma((vtype)256.0, val, (vtype)0.5));
        v2type tv = USE_VTABLE(atan_jby256_tbl, convert_uinttype(index - 16));
        q1 = tv.lo; // s0
        q2 = tv.hi; // s1
        vtype c = convert_vtype(index) * 0x1.0p-8;

        // We're going to scale u and v by 2^(-u_exponent) to bring them close to 1
        // u_exponent could be EMAX so we have to do it in 2 steps
        itype m = -(as_itype(as_utype(u) >> EXPSHIFTBITS_DP64) - (itype)EXPBIAS_DP64);
        vtype um = ldexp(u, convert_inttype(m));
        vtype vm = ldexp(v, convert_inttype(m));

        // 26 leading bits of u
        vtype u1 = as_vtype(as_utype(um) & (utype)0xfffffffff8000000UL);
        vtype u2 = um - u1;

        vtype r = MATH_DIVIDE(pocl_fma(-c, u2, pocl_fma(-c, u1, vm)), pocl_fma(c, vm, um));

        // Polynomial approximation to atan(r)
        vtype s = r * r;
        q2 = q2 + pocl_fma((s * pocl_fma(-s, (vtype)0.19999918038989143496, (vtype)0.33333333333224095522)), -r, r);
    }


    vtype q3, q4;
    {
        q3 = 0.0;
        q4 = vbyu;
    }

    vtype q5, q6;
    {
        vtype u1 = as_vtype(as_utype(u) & (utype)0xffffffff00000000UL);
        vtype u2 = u - u1;
        vtype vu1 = as_vtype(as_utype(vbyu) & (utype)0xffffffff00000000UL);
        vtype vu2 = vbyu - vu1;

        q5 = 0.0;
        vtype s = vbyu * vbyu;
        q6 = vbyu + pocl_fma(-vbyu * s,
                        pocl_fma(-s,
                            pocl_fma(-s,
                                pocl_fma(-s,
                                    pocl_fma(-s, (vtype)0.90029810285449784439E-01,
                                        (vtype)0.11110736283514525407),
                                    (vtype)0.14285713561807169030),
                                (vtype)0.19999999999393223405),
                            (vtype)0.33333333333333170500),
       MATH_DIVIDE(pocl_fma(-u, vu2, pocl_fma(-u2, vu1, pocl_fma(-u1, vu1, v))), u));
    }


    q3 = vbyu < (vtype)0x1.d12ed0af1a27fp-27 ? q3 : q5;
    q4 = vbyu < (vtype)0x1.d12ed0af1a27fp-27 ? q4 : q6;

    q1 = vbyu > (vtype)0.0625 ? q1 : q3;
    q2 = vbyu > (vtype)0.0625 ? q2 : q4;

    // Tidy-up according to which quadrant the arguments lie in
    vtype res1, res2, res3, res4;
    q1 = swap_vu ? piby2_head - q1 : q1;
    q2 = swap_vu ? piby2_tail - q2 : q2;
    q1 = xneg ? pi_head - q1 : q1;
    q2 = xneg ? pi_tail - q2 : q2;
    q1 = MATH_DIVIDE(q1 + q2, pi);
    res4 = yneg ? -q1 : q1;

    res1 = yneg ? (vtype)-0.75 : (vtype)0.75;
    res2 = yneg ? (vtype)-0.25 : (vtype)0.25;
    res3 = xneg ? res1 : res2;

    res3 = isinf(y2) & isinf(x2) ? res3 : res4;
    res1 = yneg ? (vtype)-1.0 : (vtype)1.0;

    // abs(x)/abs(y) > 2^56 and x < 0
    res3 = ((diffexp < (itype)-56) && xneg) ? res1 : res3;

    res4 = MATH_DIVIDE(MATH_DIVIDE(y, x), pi);
    // x positive and dominant over y by a factor of 2^28
    itype xpos = xneg ^ (itype)SIGNBIT_DP64;
    res3 = ((diffexp < (itype)-28) & xpos) ? res4 : res3;

    // abs(y)/abs(x) > 2^56
    res4 = yneg ? (vtype)-0.5 : (vtype)0.5;        // atan(y/x) is insignificant compared to piby2
    res3 = (diffexp > (itype)56) ? res4 : res3;

    res3 = (x2 == (vtype)0.0) ? res4 : res3;  // Zero x gives +- pi/2 depending on sign of y
    res4 = xneg ? res1 : y2;

    res3 = (y2 == (vtype)0.0) ? res4 : res3;  // Zero y gives +-0 for positive x and +-pi for negative x
    res3 = isnan(y2) ? y2 : res3;
    res3 = isnan(x2) ? x2 : res3;

    return res3;
}
