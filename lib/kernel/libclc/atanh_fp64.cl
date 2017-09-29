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
    vtype absx = fabs(x);

    vtype ret = (absx == (vtype)1.0) ? as_vtype((utype)PINFBITPATT_DP64) : as_vtype((utype)QNANBITPATT_DP64);

    // |x| >= 0.5
    // Note that atanh(x) = 0.5 * ln((1+x)/(1-x))
    // For greater accuracy we use
    // ln((1+x)/(1-x)) = ln(1 + 2x/(1-x)) = log1p(2x/(1-x)).
    vtype r = 0.5 * log1p((2.0 * absx) / ((vtype)1.0 - absx));
    ret = (absx < (vtype)1.0) ? r : ret;

    r = -ret;
    ret = (x < (vtype)0.0) ? r : ret;

    // Arguments up to 0.5 in magnitude are
    // approximated by a [5,5] minimax polynomial
    vtype t = x * x;

    vtype pn = pocl_fma(t,
                    pocl_fma(t,
                        pocl_fma(t,
                            pocl_fma(t,
                                pocl_fma(t,
                                  (vtype)-0.10468158892753136958e-3,
                                  (vtype)0.28728638600548514553e-1),
                                (vtype)-0.28180210961780814148e0),
                            (vtype)0.88468142536501647470e0),
                        (vtype)-0.11028356797846341457e1),
                    (vtype)0.47482573589747356373e0);

    vtype pd = pocl_fma(t,
                    pocl_fma(t,
                        pocl_fma(t,
                            pocl_fma(t,
                                pocl_fma(t,
                                  (vtype)-0.35861554370169537512e-1,
                                  (vtype)0.49561196555503101989e0),
                                (vtype)-0.22608883748988489342e1),
                            (vtype)0.45414700626084508355e1),
                        (vtype)-0.41631933639693546274e1),
                    (vtype)0.14244772076924206909e1);

    r = pocl_fma(x*t, pn/pd, x);
    ret = (absx < (vtype)0.5) ? r : ret;

    return ret;
}
