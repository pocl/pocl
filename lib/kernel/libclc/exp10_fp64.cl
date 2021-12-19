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




_CL_OVERLOADABLE vtype exp10(vtype x)
{
    const vtype X_MAX = (vtype)0x1.34413509f79ffp+8; // 1024*ln(2)/ln(10)
    const vtype X_MIN = (vtype)-0x1.434e6420f4374p+8; // -1074*ln(2)/ln(10)

    const vtype R_64_BY_LOG10_2 = (vtype)0x1.a934f0979a371p+7; // 64*ln(10)/ln(2)
    const vtype R_LOG10_2_BY_64_LD = (vtype)-0x1.3441350000000p-8; // head ln(2)/(64*ln(10))
    const vtype R_LOG10_2_BY_64_TL = (vtype)-0x1.3ef3fde623e25p-37; // tail ln(2)/(64*ln(10))
    const vtype R_LN10 = (vtype)0x1.26bb1bbb55516p+1; // ln(10)

    itype n = convert_itype(x * R_64_BY_LOG10_2);

    vtype dn = convert_vtype(n);

    uinttype j = convert_uinttype(n & (itype)0x3f);
    itype m = n >> 6;

    vtype r = R_LN10 * fma(R_LOG10_2_BY_64_TL, dn, fma(R_LOG10_2_BY_64_LD, dn, x));

    // 6 term tail of Taylor expansion of e^r
    vtype z2 = r * fma(r,
	                fma(r,
		            fma(r,
			        fma(r,
			            fma(r, (vtype)0x1.6c16c16c16c17p-10, (vtype)0x1.1111111111111p-7),
			            (vtype)0x1.5555555555555p-5),
			        (vtype)0x1.5555555555555p-3),
		            (vtype)0x1.0000000000000p-1),
		        (vtype)1.0);

    v2type tv = USE_VTABLE(two_to_jby64_ep_tbl, j);
    vtype s0 = tv.lo;
    vtype s1 = tv.hi;

    z2 = fma(s0 + s1, z2, s1) + s0;

    itype small_value = (m < (itype)-1022) || ((m == (itype)-1022) && (z2 < (vtype)1.0));

		itype n1 = m >> 2;
		itype n2 = m-n1;
		vtype z3= z2 * as_vtype((n1 + (itype)1023) << 52);
		z3 *= as_vtype((n2 + (itype)1023) << 52);

    z2 = ldexp(z2, m);
    z2 = small_value ? z3: z2;

    z2 = isnan(x) ? x : z2;

    z2 = (x > (vtype)X_MAX) ? as_vtype((utype)PINFBITPATT_DP64) : z2;
    z2 = (x < (vtype)X_MIN) ? (vtype)0.0 : z2;

    return z2;
}
