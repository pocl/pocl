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

_CL_OVERLOADABLE vtype tan(vtype x) {

    /*
    * TODO this does not work, for some reason
    *
    *  vtype sinx, cosx;
    *  sinx = sincos(x, (private vtype *)&cosx);
    *  return sinx / cosx;
    */
    vtype y = fabs(x);

    vtype r, rr, r2, rr2;
    itype regn, regn2;

    __pocl_remainder_piby2_medium(y, &r, &rr, &regn);
    itype cond = (y >= (vtype)0x1.0p+47);
    if (SV_ANY(cond)) {
        __pocl_remainder_piby2_large(y, &r2, &rr2, &regn2);
        regn = cond ? regn2 : regn;
        r = cond ? r2 : r;
        rr = cond ? rr2 : rr;
    }
    v2type sc = __pocl_sincos_piby4(r, rr);

    cond = (regn << 63);

    vtype ss = sc.lo;
    vtype cc = sc.hi;

    itype s = cond ? as_itype(cc) : as_itype(ss);

    ss = -sc.lo;
    itype c = cond ? as_itype(ss) : as_itype(cc);

    itype sgn = ((regn >> 1) << 63);
    s ^= sgn;
    c ^= sgn;
    s ^= (as_itype(x) & (itype)SIGNBIT_DP64);

    vtype ret = as_vtype(s) / as_vtype(c);
    vtype nans = as_vtype( (as_utype(x) & (utype)SIGNBIT_DP64)
    | ((utype)QNANBITPATT_DP64) );

    return (isnan(x) | isinf(x)) ? nans : ret;
}
