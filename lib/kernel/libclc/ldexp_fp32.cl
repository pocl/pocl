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


_CL_OVERLOADABLE vtype ldexp(vtype x, int n) {

//	if (!__clc_fp32_subnormals_supported()) {

		// This treats subnormals as zeros
		itype i = as_itype(x);
		itype e = (i >> 23) & (itype)0xff;
		itype m = i & (itype)0x007fffff;
		itype s = i & (itype)0x80000000;
		itype v = add_sat(e, n);
		v = clamp(v, (itype)0, (itype)0xff);
		itype mr = ((e == (itype)0) | (v == (itype)0) | (v == (itype)0xff)) ? (itype)0 : m;
		itype c = (e == (itype)0xff);
		mr = c ? m : mr;
		itype er = c ? e : v;
		er = e ? er : e;
		return as_vtype( s | (er << 23) | mr );


//	}

	/* supports denormal values */
/*

	const int multiplier = 24;
	vtype val_f;
	uint val_ui;
	uint sign;
	int exponent;
	val_ui = as_uint(x);
	sign = val_ui & 0x80000000;
	val_ui = val_ui & 0x7fffffff; // remove the sign bit
	int val_x = val_ui;

	exponent = val_ui >> 23; // get the exponent
	int dexp = exponent;

	// denormal support
	int fbh = 127 - (as_uint((vtype)(as_vtype(val_ui | 0x3f800000) - 1.0f)) >> 23);
	int dexponent = 25 - fbh;
	uint dval_ui = (( (val_ui << fbh) & 0x007fffff) | (dexponent << 23));
	int ex = dexponent + n - multiplier;
	dexponent = ex;
	uint val = sign | (ex << 23) | (dval_ui & 0x007fffff);
	int ex1 = dexponent + multiplier;
	ex1 = -ex1 +25;
	dval_ui = (((dval_ui & 0x007fffff )| 0x800000) >> ex1);
	dval_ui = dexponent > 0 ? val :dval_ui;
	dval_ui = dexponent > 254 ? 0x7f800000 :dval_ui;  // overflow
	dval_ui = dexponent < -multiplier ? 0 : dval_ui;  // underflow
	dval_ui = dval_ui | sign;
	val_f = as_vtype(dval_ui);

	exponent += n;

	val = sign | (exponent << 23) | (val_ui & 0x007fffff);
	ex1 = exponent + multiplier;
	ex1 = -ex1 +25;
	val_ui = (((val_ui & 0x007fffff )| 0x800000) >> ex1);
	val_ui = exponent > 0 ? val :val_ui;
	val_ui = exponent > 254 ? 0x7f800000 :val_ui;  // overflow
	val_ui = exponent < -multiplier ? 0 : val_ui;  // underflow
	val_ui = val_ui | sign;

	val_ui = dexp == 0? dval_ui : val_ui;
	val_f = as_vtype(val_ui);

	val_f = isnan(x) | isinf(x) | val_x == 0 ? x : val_f;
	return val_f;
*/

}
