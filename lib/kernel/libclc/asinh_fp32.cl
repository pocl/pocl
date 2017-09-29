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


_CL_OVERLOADABLE vtype asinh(vtype x) {
    utype ux = as_utype(x);
    utype ax = ux & (utype)EXSIGNBIT_SP32;
    utype xsgn = ax ^ ux;

    // |x| <= 2
    vtype t = x * x;
    vtype a = pocl_fma(t,
                pocl_fma(t,
                  pocl_fma(t,
                    pocl_fma(t,
                      (vtype)-1.177198915954942694e-4f,
                      (vtype)-4.162727710583425360e-2f),
                    (vtype)-5.063201055468483248e-1f),
                  (vtype)-1.480204186473758321f),
                (vtype)-1.152965835871758072f);
    vtype b = pocl_fma(t,
                pocl_fma(t,
                  pocl_fma(t,
                    pocl_fma(t,
                      (vtype)6.284381367285534560e-2f,
                      (vtype)1.260024978680227945f),
                    (vtype)6.582362487198468066f),
                  (vtype)11.99423176003939087f),
                (vtype)6.917795026025976739f);

    vtype q = MATH_DIVIDE(a, b);
    vtype z1 = pocl_fma(x*t, q, x);

    // |x| > 2

    // Arguments greater than 1/sqrt(epsilon) in magnitude are
    // approximated by asinh(x) = ln(2) + ln(abs(x)), with sign of x
    // Arguments such that 4.0 <= abs(x) <= 1/sqrt(epsilon) are
    // approximated by asinhf(x) = ln(abs(x) + sqrt(x*x+1))
    // with the sign of x (see Abramowitz and Stegun 4.6.20)

    vtype absx = as_vtype(ax);
    itype hi = (ax > 0x46000000U);
    vtype y = MATH_SQRT(absx * absx + (vtype)1.0f) + absx;
    y = hi ? absx : y;
    vtype r = log(y) + (hi ? (vtype)0x1.62e430p-1f : (vtype)0.0f);
    vtype z2 = as_vtype(xsgn | as_utype(r));

    vtype z = (ax <= (utype)0x40000000) ? z1 : z2;
    z = ((ax < (utype)0x39800000U) | (ax >= (utype)PINFBITPATT_SP32)) ? x : z;

    return z;
}
