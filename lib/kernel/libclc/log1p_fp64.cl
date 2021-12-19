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


_CL_OVERLOADABLE vtype log1p(vtype x)
{
    // Computes natural log(1+x). Algorithm based on:
    // Ping-Tak Peter Tang
    // "Table-driven implementation of the logarithm function in IEEE
    // floating-point arithmetic"
    // ACM Transactions on Mathematical Software (TOMS)
    // Volume 16, Issue 4 (December 1990)
    // Note that we use a lookup table of size 64 rather than 128,
    // and compensate by having extra terms in the minimax polynomial
    // for the kernel approximation.

    // Process Inside the threshold now
    utype ux = as_utype((vtype)1.0 + x);
    itype xexp = ((ux >> 52) & (utype)0x7ff) - (utype)EXPBIAS_DP64;
    vtype f = as_vtype((utype)ONEEXPBITS_DP64 | (ux & (utype)MANTBITS_DP64));

    itype j = ux >> 45;
    j = (((itype)0x80 | (j & (itype)0x7e)) >> 1) + (j & (itype)0x1);
    vtype f1 = (vtype)j * (vtype)0x1.0p-6;
    j -= (itype)64;

    vtype f2temp = f - f1;
    vtype m2 = as_vtype(convert_utype((itype)0x3ff - xexp) << EXPSHIFTBITS_DP64);
    vtype f2l = pocl_fma(m2, x, m2 - f1);
    vtype f2g = pocl_fma(m2, x, -f1) + m2;
    vtype f2 = (xexp <= (itype)(MANTLENGTH_DP64-1)) ? f2l : f2g;
    f2 = (xexp <= (itype)-2) | (xexp >= (itype)(MANTLENGTH_DP64+8)) ? f2temp : f2;

    v2type tv = USE_VTABLE(ln_tbl, j);
    vtype s0 = tv.lo;
    vtype s1 = tv.hi;
    vtype z1 = s0;
    vtype q = s1;

    vtype u = MATH_DIVIDE(f2, pocl_fma((vtype)0.5, f2, f1));
    vtype v = u * u;

    vtype poly = v * pocl_fma(v,
                          pocl_fma(v, (vtype)2.23219810758559851206e-03, (vtype)1.24999999978138668903e-02),
                          (vtype)8.33333333333333593622e-02);

    // log2_lead and log2_tail sum to an extra-precise version of log(2)
    const vtype log2_lead = (vtype)6.93147122859954833984e-01; /* 0x3fe62e42e0000000 */
    const vtype log2_tail = (vtype)5.76999904754328540596e-08; /* 0x3e6efa39ef35793c */

    vtype z2 = q + pocl_fma(u, poly, u);
    vtype dxexp = convert_vtype(xexp);
    vtype r1 = pocl_fma(dxexp, log2_lead, z1);
    vtype r2 = pocl_fma(dxexp, log2_tail, z2);
    vtype result1 = r1 + r2;

    // Process Outside the threshold now
    vtype r = x;
    u = r / ((vtype)2.0 + r);
    vtype correction = r * u;
    u = u + u;
    v = u * u;
    r1 = r;

    poly = pocl_fma(v,
               pocl_fma(v,
                   pocl_fma(v, (vtype)4.34887777707614552256e-04, (vtype)2.23213998791944806202e-03),
                   (vtype)1.25000000037717509602e-02),
               (vtype)8.33333333333317923934e-02);

    r2 = pocl_fma(u*v, poly, -correction);

    // The values exp(-1/16)-1 and exp(1/16)-1
    const vtype log1p_thresh1 = (vtype)-0x1.f0540438fd5c3p-5;
    const vtype log1p_thresh2 = (vtype)0x1.082b577d34ed8p-4;
    vtype result2 = r1 + r2;
    result2 = (x < log1p_thresh1) | (x > log1p_thresh2) ? result1 : result2;

    result2 = isinf(x) ? x : result2;
    result2 = (x < (vtype)-1.0) ? as_vtype((utype)QNANBITPATT_DP64) : result2;
    result2 = (x == (vtype)-1.0) ? as_vtype((utype)NINFBITPATT_DP64) : result2;
    return result2;
}
