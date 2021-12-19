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

    const vtype max_expm1_arg = (vtype)709.8;
    const vtype min_expm1_arg = (vtype)-37.42994775023704;
    const vtype log_OnePlus_OneByFour = (vtype)0.22314355131420976;   //0x3FCC8FF7C79A9A22 = log(1+1/4)
    const vtype log_OneMinus_OneByFour = (vtype)-0.28768207245178096; //0xBFD269621134DB93 = log(1-1/4)
    const vtype sixtyfour_by_lnof2 = (vtype)92.33248261689366;        //0x40571547652b82fe
    const vtype lnof2_by_64_head = (vtype)0.010830424696223417;       //0x3f862e42fefa0000
    const vtype lnof2_by_64_tail = (vtype)2.5728046223276688e-14;     //0x3d1cf79abc9e3b39

    // First, assume log(1-1/4) < x < log(1+1/4) i.e  -0.28768 < x < 0.22314
    vtype u = as_vtype(as_utype(x) & (utype)0xffffffffff000000UL);
    vtype v = x - u;
    vtype y = u * u * 0.5;
    vtype z = v * (x + u) * 0.5;

    vtype q = fma(x,
	           fma(x,
		       fma(x,
			   fma(x,
			       fma(x,
				   fma(x,
				       fma(x,
					   fma(x, (vtype)2.4360682937111612e-8, (vtype)2.7582184028154370e-7),
					   (vtype)2.7558212415361945e-6),
				       (vtype)2.4801576918453420e-5),
				   (vtype)1.9841269447671544e-4),
			       (vtype)1.3888888890687830e-3),
			   (vtype)8.3333333334012270e-3),
		       (vtype)4.1666666666665560e-2),
		   (vtype)1.6666666666666632e-1);
    q *= x * x * x;

    vtype z1g = (u + y) + (q + (v + z));
    vtype z1 = x + (y + (q + z));
    z1 = (y >= (vtype)0x1.0p-7) ? z1g : z1;

    // Now assume outside interval around 0
    itype n = convert_itype(x * sixtyfour_by_lnof2);
    uinttype j = convert_uinttype(n & (itype)0x3f);
    itype m = n >> 6;

    v2type tv = USE_VTABLE(two_to_jby64_ep_tbl, j);
    vtype f1 = tv.lo;
    vtype f2 = tv.hi;
    vtype f = f1 + f2;

    vtype dn = -n;
    vtype r = fma(dn, lnof2_by_64_tail, fma(dn, lnof2_by_64_head, x));

    q = fma(r,
	    fma(r,
		fma(r,
		    fma(r, (vtype)1.38889490863777199667e-03, (vtype)8.33336798434219616221e-03),
		    (vtype)4.16666666662260795726e-02),
		(vtype)1.66666666665260878863e-01),
	     (vtype)5.00000000000000008883e-01);
    q = fma(r*r, q, r);

    vtype twopm = as_vtype(convert_itype(m + (itype)EXPBIAS_DP64) << EXPSHIFTBITS_DP64);
    vtype twopmm = as_vtype(convert_itype((itype)EXPBIAS_DP64 - m) << EXPSHIFTBITS_DP64);

    // Computations for m > 52, including where result is close to Inf
    utype uval = as_utype((vtype)0x1.0p+1023 * (f1 + (f * q + (f2))));
    itype e = convert_itype(uval >> EXPSHIFTBITS_DP64) + 1;

    vtype zme1024 = as_vtype(((itype)e << EXPSHIFTBITS_DP64) | (uval & (utype)MANTBITS_DP64));
    zme1024 = e == 2047 ? as_vtype((itype)PINFBITPATT_DP64) : zme1024;

    vtype zmg52 = twopm * (f1 + fma(f, q, f2 - twopmm));
    zmg52 = (m == (itype)1024) ? zme1024 : zmg52;

    // For m < 53
    vtype zml53 = twopm * ((f1 - twopmm) + fma(f1, q, f2*((vtype)1.0 + q)));

    // For m < -7
    vtype zmln7 = fma(twopm,  f1 + fma(f, q, f2), -1.0);

    z = (m < (itype)53) ? zml53 : zmg52;
    z = (m < (itype)-7) ? zmln7 : z;
    z = ((x > log_OneMinus_OneByFour) & (x < log_OnePlus_OneByFour)) ? z1 : z;
    z = (x > max_expm1_arg) ? as_vtype((itype)PINFBITPATT_DP64) : z;
    z = (x < min_expm1_arg) ? (vtype)-1.0 : z;

    return z;
}
