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




_CL_OVERLOADABLE  vtype atan2pi(vtype y, vtype x) {
    const vtype pi = (vtype)0x1.921fb6p+1f;

    vtype ax = fabs(x);
    vtype ay = fabs(y);
    vtype v = min(ax, ay);
    vtype u = max(ax, ay);

    // Scale since u could be large, as in "regular" divide
    vtype s = (u > (vtype)0x1.0p+96f) ? (vtype)0x1.0p-32f : (vtype)1.0f;
    vtype vbyu = s * MATH_DIVIDE(v, s*u);

    vtype vbyu2 = vbyu * vbyu;

    vtype p = pocl_fma(vbyu2,
                pocl_fma(vbyu2,
                    (vtype)-0x1.7e1f78p-9f,
                    (vtype)-0x1.7d1b98p-3f),
                (vtype)-0x1.5554d0p-2f)
                * vbyu2 * vbyu;
    vtype q = pocl_fma(vbyu2,
                pocl_fma(vbyu2,
                  (vtype)0x1.1a714cp-2f,
                  (vtype)0x1.287c56p+0f),
                  (vtype)1.0f);

    // Octant 0 result
    vtype a = MATH_DIVIDE(pocl_fma(p, MATH_RECIP(q), vbyu), pi);

    // Fix up 3 other octants
    vtype at = (vtype)0.5f - a;
    a = (ay > ax) ? at : a;
    at = (vtype)1.0f - a;
    a = (x < (vtype)0.0f) ? at : a;

    // y == 0 => 0 for x >= 0, pi for x < 0
    at = (as_itype(x) & (itype)SIGNBIT_SP32) ? (vtype)1.0f : (vtype)0.0f;
    a = (y == (vtype)0.0f) ? at : a;

    // x and y are +- Inf
    at = (x > (vtype)0.0f) ? (vtype)0.25f : (vtype)0.75f;
    a = ((ax == (vtype)INFINITY) & (ay == (vtype)INFINITY)) ? at : a;

    // x or y is NaN
    a = (isnan(x) | isnan(y)) ? as_vtype((utype)QNANBITPATT_SP32) : a;

    // Fixup sign and return
    return copysign(a, y);
}
