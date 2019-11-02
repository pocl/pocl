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



_CL_OVERLOADABLE vtype tanh(vtype x)
{
    // The definition of tanh(x) is sinh(x)/cosh(x), which is also equivalent
    // to the following three formulae:
    // 1.  (exp(x) - exp(-x))/(exp(x) + exp(-x))
    // 2.  (1 - (2/(exp(2*x) + 1 )))
    // 3.  (exp(2*x) - 1)/(exp(2*x) + 1)
    // but computationally, some formulae are better on some ranges.

#ifdef MAX_PRECISION
    // 1.29 ULP
    const vtype large_threshold1 = (vtype)0x1.039346p+3;  //8.1117277
    const vtype retval_threshold1 = (vtype)0x1.fffffap-1;
    const vtype large_threshold2 = (vtype)0x1.0a101p+3;  // 8.3144608
    const vtype retval_threshold2 = (vtype)0x1.fffffcp-1;
    const vtype large_threshold3 = (vtype)0x1.15273p+3;  // 8.661034
    const vtype retval_threshold3 = (vtype)0x1.fffffep-1;
#else
    // 3.0 ULP
    const vtype large_threshold1 = (vtype)0x1.0a2b24p+3f;
    const vtype retval_threshold1 = (vtype)0x1.fffffep-1f;
#endif
    utype ux = as_utype(x);
    utype aux = ux & (utype)EXSIGNBIT_SP32;
    utype xs = ux ^ aux;

    vtype y = as_vtype(aux);
    vtype y2 = y*y;

    vtype a1 = pocl_fma(y2,
                   pocl_fma(y2,
                     (vtype)0.4891631088530669873e-4f,
                     (vtype)-0.14628356048797849e-2f),
                   (vtype)-0.28192806108402678f);
    vtype b1 = pocl_fma(y2,
                 (vtype)0.3427017942262751343f,
                 (vtype)0.845784192581041099f);

    vtype a2 = pocl_fma(y2,
                   pocl_fma(y2,
                     (vtype)0.3827534993599483396e-4f,
                     (vtype)-0.12325644183611929e-2f),
                   (vtype)-0.24069858695196524f);
    vtype b2 = pocl_fma(y2,
                 (vtype)0.292529068698052819f,
                 (vtype)0.72209738473684982f);
    itype c = (y < (vtype)0.9f);
    vtype a = c ? a1 : a2;
    vtype b = c ? b1 : b2;
    vtype zlo = pocl_fma(MATH_DIVIDE(a, b), y*y2, y);

    vtype p = exp(2.0f * y) + (vtype)1.0f;
    vtype zhi = (vtype)1.0F - MATH_DIVIDE((vtype)2.0F, p);

    vtype z = (y <= (vtype)1.0f) ? zlo : zhi;

    // Edge cases
    z = (y > large_threshold1) ? retval_threshold1 : z;
#ifdef MAX_PRECISION
    z = (y > large_threshold2) ? retval_threshold2 : z;
    z = (y > large_threshold3) ? retval_threshold3 : z;
#endif
    z = as_vtype(xs | as_utype(z));

    return z;
}
