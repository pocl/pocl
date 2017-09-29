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


_CL_OVERLOADABLE  vtype acosh(vtype x) {
    utype ux = as_utype(x);

    // Arguments greater than 1/sqrt(epsilon) in magnitude are
    // approximated by acosh(x) = ln(2) + ln(x)
    // For 2.0 <= x <= 1/sqrt(epsilon) the approximation is
    // acosh(x) = ln(x + sqrt(x*x-1)) */
    itype high = (ux > (utype)0x46000000);
    itype med = (ux > (utype)0x40000000);

    vtype w = x - (vtype)1.0f;
    vtype s = w*w + 2.0f*w;
    vtype t = x*x - (vtype)1.0f;
    vtype r = sqrt(select(s, t, med)) + select(w, x, med);
    vtype v = select(r, x, high) - select((vtype)0.0f, (vtype)1.0f, med);
    vtype z = log1p(v) + select((vtype)0.0f, (vtype)0x1.62e430p-1f, high);

    z = select(z, x, (ux >= (utype)PINFBITPATT_SP32));
    z = select(z, as_vtype((utype)QNANBITPATT_SP32), (x < (vtype)1.0f));

    return z;
}
