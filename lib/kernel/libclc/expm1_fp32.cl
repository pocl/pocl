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


_CL_OVERLOADABLE vtype expm1(vtype x) {

    const vtype X_MAX = (vtype)0x1.62e42ep+6f; // 128*log2 : 88.722839111673
    const vtype X_MIN = (vtype)-0x1.9d1da0p+6f; // -149*log2 : -103.27892990343184

    const vtype R_64_BY_LOG2 = (vtype)0x1.715476p+6f;     // 64/log2 : 92.332482616893657
    const vtype R_LOG2_BY_64_LD = (vtype)0x1.620000p-7f;  // log2/64 lead: 0.0108032227
    const vtype R_LOG2_BY_64_TL = (vtype)0x1.c85fdep-16f; // log2/64 tail: 0.0000272020388

    utype xi = as_utype(x);
    itype n = convert_itype(x * R_64_BY_LOG2);
    vtype fn = convert_vtype(n);

    utype j = as_utype(n & (itype)0x3f);
    itype m = n >> 6;

    vtype r = mad(fn, -R_LOG2_BY_64_TL, mad(fn, -R_LOG2_BY_64_LD, x));

    // Truncated Taylor series
    vtype z2 = mad(r*r, mad(r, mad(r, (vtype)0x1.555556p-5f,  (vtype)0x1.555556p-3f), (vtype)0.5f), r);

    vtype m2 = as_vtype((m + (itype)EXPBIAS_SP32) << (itype)EXPSHIFTBITS_SP32);

    v2type tv = USE_VTABLE(exp_tbl_ep, j);
    vtype s0 = tv.lo;
    vtype s1 = tv.hi;

    vtype two_to_jby64_h = s0 * m2;
    vtype two_to_jby64_t = s1 * m2;
    vtype two_to_jby64 = two_to_jby64_h + two_to_jby64_t;

    z2 = mad(z2, two_to_jby64, two_to_jby64_t) + (two_to_jby64_h - (vtype)1.0f);

	//Make subnormals work

    z2 = (x == (vtype)0.f) ? x : z2;
    z2 = ((x < X_MIN) | (m < -24)) ? (vtype)-1.0f : z2;
    z2 = (x > X_MAX) ? (vtype)(as_float(PINFBITPATT_SP32)) : z2;
    z2 = isnan(x) ? x : z2;

    return z2;
}
