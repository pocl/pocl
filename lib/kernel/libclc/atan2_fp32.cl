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




_CL_OVERLOADABLE vtype atan2(vtype y, vtype x)
{
    const vtype pi = (vtype)0x1.921fb6p+1f;
    const vtype piby2 = (vtype)0x1.921fb6p+0f;
    const vtype piby4 = (vtype)0x1.921fb6p-1f;
    const vtype threepiby4 = (vtype)0x1.2d97c8p+1f;

    vtype ax = fabs(x);
    vtype ay = fabs(y);
    vtype v = fmin(ax, ay);
    vtype u = fmax(ax, ay);

    // Scale since u could be large, as in "regular" divide
    vtype s = (u > (vtype)0x1.0p+96f) ? (vtype)0x1.0p-32f : (vtype)1.0f;
    vtype vbyu = s * MATH_DIVIDE(v, s*u);
    vtype vbyu2 = vbyu * vbyu;

#define USE_2_2_APPROXIMATION
#if defined USE_2_2_APPROXIMATION
    vtype p = mad(vbyu2, mad(vbyu2, (vtype)-0x1.7e1f78p-9f, (vtype)-0x1.7d1b98p-3f), (vtype)-0x1.5554d0p-2f) * vbyu2 * vbyu;
    vtype q = mad(vbyu2, mad(vbyu2, (vtype)0x1.1a714cp-2f, (vtype)0x1.287c56p+0f), 1.0f);
#else
    vtype p = mad(vbyu2, mad(vbyu2, (vtype)-0x1.55cd22p-5f, (vtype)-0x1.26cf76p-2f), (vtype)-0x1.55554ep-2f) * vbyu2 * vbyu;
    vtype q = mad(vbyu2, mad(vbyu2, mad(vbyu2, (vtype)0x1.9f1304p-5f, (vtype)0x1.2656fap-1f), (vtype)0x1.76b4b8p+0f), (vtype)1.0f);
#endif

    // Octant 0 result
    vtype a = mad(p, MATH_RECIP(q), vbyu);

    // Fix up 3 other octants
    vtype at = piby2 - a;
    a = (ay > ax) ? at : a;
    at = pi - a;
    a = (x < (vtype)0.0f) ? at : a;

    // y == 0 => 0 for x >= 0, pi for x < 0
    at = as_itype(x) < (itype)0 ? pi : (vtype)0.0f;
    a = (y == (vtype)0.0f) ? at : a;

    // if (!FINITE_ONLY()) {
        // x and y are +- Inf
        at = (x > (vtype)0.0f) ? piby4 : threepiby4;
        a = ((ax == (vtype)INFINITY) & (ay == (vtype)INFINITY)) ? at : a;

        // x or y is NaN
        a = (isnan(x) | isnan(y)) ? as_vtype((utype)QNANBITPATT_SP32) : a;
    // }

    // Fixup sign and return
    return copysign(a, y);
}
