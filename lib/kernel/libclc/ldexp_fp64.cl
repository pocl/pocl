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



_CL_OVERLOADABLE vtype ldexp(vtype x, itype n) {

	itype l = as_itype(x);
	itype e = (l >> 52) & (itype)0x7ff;
	itype s = l & (itype)0x8000000000000000;

	utype ux = as_utype(x * (vtype)0x1.0p+53);
	itype de = (as_itype(ux >> 52) & (itype)0x7ff) - (itype)53;
	itype c = (e == (itype)0);
	e = c ? de: e;

	ux = c ? ux : l;

	itype v = e + n;
	v = clamp(v, (itype)-0x7ff, (itype)0x7ff);

	ux &= (utype)(~EXPBITS_DP64);

	vtype mr = as_vtype(ux | ((utype)(v+(itype)53) << 52));
	mr = mr * (vtype)0x1.0p-53;

	mr = (v > (itype)0)  ? as_vtype(ux | ((utype)v << 52)) : mr;

	mr = (v == (itype)0x7ff) ? as_vtype(s | (itype)PINFBITPATT_DP64) : mr;
	mr = (v < (itype)-53) ? as_vtype(s) : mr;

	mr  = ((n == (itype)0) | isinf(x) | (x == (vtype)0) ) ? x : mr;
	return mr;

}
