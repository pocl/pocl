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

#define LN0 (vtype)8.33333333333317923934e-02
#define LN1 (vtype)1.25000000037717509602e-02
#define LN2 (vtype)2.23213998791944806202e-03
#define LN3 (vtype)4.34887777707614552256e-04

#define LF0 (vtype)8.33333333333333593622e-02
#define LF1 (vtype)1.24999999978138668903e-02
#define LF2 (vtype)2.23219810758559851206e-03

_CL_OVERLOADABLE void __pocl_ep_log(vtype x, itype *xexp, vtype *r1, vtype *r2)
{
    // Computes natural log(x). Algorithm based on:
    // Ping-Tak Peter Tang
    // "Table-driven implementation of the logarithm function in IEEE
    // vtypeing-point arithmetic"
    // ACM Transactions on Mathematical Software (TOMS)
    // Volume 16, Issue 4 (December 1990)
    itype near_one = (x >= (vtype)0x1.e0faap-1) & (x <= (vtype)0x1.1082cp+0);

    utype ux = as_utype(x);
    utype uxs = as_utype(as_vtype(as_utype((utype)0x03d0000000000000UL) | ux) - (vtype)0x1.0p-962);
    itype c = (ux < (utype)IMPBIT_DP64);
    ux = c ? uxs : ux;
    itype expadjust = c ? (itype)60 : (itype)0;

    // Store the exponent of x in xexp and put f into the range [0.5,1)
    itype xexp1 = ((as_itype(ux) >> 52) & 0x7ff) - (itype)EXPBIAS_DP64 - expadjust;
    vtype f = as_vtype(HALFEXPBITS_DP64 | (ux & MANTBITS_DP64));
    *xexp = near_one ? (itype)0 : xexp1;

    vtype r = x - (vtype)1.0;
    vtype u1 = MATH_DIVIDE(r, (vtype)2.0 + r);
    vtype ru1 = -r * u1;
    u1 = u1 + u1;

    itype index = as_itype(ux) >> 45; // 13 + 32
    index = (((itype)0x80 | (index & (itype)0x7e)) >> 1) + (index & (itype)0x1);

    vtype f1 = convert_vtype(index) * 0x1.0p-7;
    vtype f2 = f - f1;
    vtype u2 = MATH_DIVIDE(f2, pocl_fma((vtype)0.5, f2, f1));

    v2type tv = USE_VTABLE(ln_tbl, convert_uinttype(index - (itype)64));
    vtype z1 = tv.lo;
    vtype q = tv.hi;

    z1 = near_one ? r : z1;
    q = near_one ? (vtype)0.0 : q;
    vtype u = near_one ? u1 : u2;
    vtype v = u*u;

    vtype cc = near_one ? ru1 : u2;

    vtype z21 = pocl_fma(v, pocl_fma(v, pocl_fma(v, LN3, LN2), LN1), LN0);
    vtype z22 = pocl_fma(v, pocl_fma(v, LF2, LF1), LF0);
    vtype z2 = near_one ? z21 : z22;
    z2 = pocl_fma(u*v, z2, cc) + q;

    *r1 = z1;
    *r2 = z2;
}
