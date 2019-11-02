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
    vtype w = x;
    utype ux = as_utype(x);
    utype ax = ux & (utype)EXSIGNBIT_SP32;

    // |x| < 2^-4
    vtype u2 = MATH_DIVIDE(x, (vtype)2.0f + x);
    vtype u = u2 + u2;
    vtype v = u * u;
    // 2/(5 * 2^5), 2/(3 * 2^3)
    vtype zsmall = pocl_fma(-u2, x, pocl_fma(v, (vtype)0x1.99999ap-7f, (vtype)0x1.555556p-4f) * v * u) + x;

    // |x| >= 2^-4
    ux = as_utype(x + 1.0f);

    itype m = as_itype((ux >> EXPSHIFTBITS_SP32) & (utype)0xff) - (itype)EXPBIAS_SP32;
    vtype mf = convert_vtype(m);
    utype indx = (ux & (utype)0x007f0000) + ((ux & (utype)0x00008000) << 1);
    vtype F = as_vtype(indx | (utype)0x3f000000);

    // x > 2^24
    vtype fg24 = F - as_vtype((utype)0x3f000000 | (ux & (utype)MANTBITS_SP32));

    // x <= 2^24
    utype xhi = ux & (utype)0xffff8000;
    vtype xh = as_vtype(xhi);
    vtype xt = ((vtype)1.0f - xh) + w;
    utype xnm = ((~(xhi & (utype)0x7f800000)) - (utype)0x00800000) & (utype)0x7f800000;
    xt = xt * as_vtype(xnm) * 0.5f;
    vtype fl24 = F - as_vtype((utype)0x3f000000 | (xhi & (utype)MANTBITS_SP32)) - xt;

    vtype f = (mf > (vtype)24.0f) ? fg24 : fl24;

    indx = indx >> 16;
    vtype r = f * USE_VTABLE(log_inv_tbl, indx);

    // 1/3, 1/2
    vtype poly = pocl_fma(pocl_fma(r, (vtype)0x1.555556p-2f, (vtype)0x1.0p-1f), r*r, r);

    const vtype LOG2_HEAD = (vtype)0x1.62e000p-1f;   // 0.693115234
    const vtype LOG2_TAIL = (vtype)0x1.0bfbe8p-15f;  // 0.0000319461833

    v2type tv = USE_VTABLE(loge_tbl, indx);
    vtype s0 = tv.lo;
    vtype s1 = tv.hi;

    vtype z1 = pocl_fma(mf, LOG2_HEAD, s0);
    vtype z2 = pocl_fma(mf, LOG2_TAIL, -poly) + s1;
    vtype z = z1 + z2;

    z = (ax < (utype)0x3d800000U) ? zsmall : z;

    // Edge cases
    z = (ax >= (utype)PINFBITPATT_SP32) ? w : z;
    z = (w  < (vtype)-1.0f) ? as_vtype((utype)QNANBITPATT_SP32) : z;
    z = (w == (vtype)-1.0f) ? as_vtype((utype)NINFBITPATT_SP32) : z;
    //fix subnormals
    z = (ax < (utype)0x33800000) ? x : z;

    return z;
}
