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



_CL_OVERLOADABLE vtype atanh(vtype x) {
    utype ux = as_utype(x);
    utype ax = ux & (utype)EXSIGNBIT_SP32;
    utype xs = ux ^ ax;

    // |x| > 1 or NaN
    vtype z = as_vtype((utype)QNANBITPATT_SP32);

    // |x| == 1
    vtype t = as_vtype(xs | (utype)PINFBITPATT_SP32);
    z = (ax == (utype)0x3f800000U) ? t : z;

    // 1/2 <= |x| < 1
    t = as_vtype(ax);
    t = MATH_DIVIDE(2.0f*t, (vtype)1.0f - t);
    t = 0.5f * log1p(t);
    t = as_vtype(xs | as_utype(t));
    z = (ax < (utype)0x3f800000U) ? t : z;

    // |x| < 1/2
    t = x * x;
    vtype a = pocl_fma(
                pocl_fma((vtype)0.92834212715e-2f,
                  t, (vtype)-0.28120347286e0f),
                t, (vtype)0.39453629046e0f);
    vtype b = pocl_fma(
                pocl_fma((vtype)0.45281890445e0f,
                  t, (vtype)-0.15537744551e1f),
                t, (vtype)0.11836088638e1f);
    vtype p = MATH_DIVIDE(a, b);
    t = pocl_fma(x*t, p, x);
    z = (ax < (utype)0x3f000000) ? t : z;

    // |x| < 2^(vtype)-1.
    z = (ax < (utype)0x39000000U) ? x : z;

    return z;
}
