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



_CL_OVERLOADABLE vtype atanpi(vtype x) {
    const vtype pi = (vtype)M_PI;

    vtype v = fabs(x);

    // 2^56 > v > 39/16
    vtype a = (vtype)-1.0;
    vtype b = v;
    // (chi + clo) = arctan(infinity)
    vtype chi = (vtype)1.57079632679489655800e+00;
    vtype clo = (vtype)6.12323399573676480327e-17;

    vtype ta = v - (vtype)1.5;
    vtype tb = (vtype)1.0 + (vtype)1.5 * v;
    itype l = (v <= (vtype)0x1.38p+1); // 39/16 > v > 19/16
    a = l ? ta : a;
    b = l ? tb : b;
    // (chi + clo) = arctan(1.5)
    chi = l ? (vtype)9.82793723247329054082e-01 : chi;
    clo = l ? (vtype)1.39033110312309953701e-17 : clo;

    ta = v - (vtype)1.0;
    tb = (vtype)1.0 + v;
    l = (v <= (vtype)0x1.3p+0); // 19/16 > v > 11/16
    a = l ? ta : a;
    b = l ? tb : b;
    // (chi + clo) = arctan(1.)
    chi = l ? (vtype)7.85398163397448278999e-01 : chi;
    clo = l ? (vtype)3.06161699786838240164e-17 : clo;

    ta = (vtype)2.0 * v - (vtype)1.0;
    tb = (vtype)2.0 + v;
    l = (v <= (vtype)0x1.6p-1); // 11/16 > v > 7/16
    a = l ? ta : a;
    b = l ? tb : b;
    // (chi + clo) = arctan(0.5)
    chi = l ? (vtype)4.63647609000806093515e-01 : chi;
    clo = l ? (vtype)2.26987774529616809294e-17 : clo;

    l = (v <= (vtype)0x1.cp-2); // v < 7/16
    a = l ? v : a;
    b = l ? (vtype)1.0 : b;;
    chi = l ? (vtype)0.0 : chi;
    clo = l ? (vtype)0.0 : clo;

    // Core approximation: Remez(4,4) on [-7/16,7/16]
    vtype r = a / b;
    vtype s = r * r;
    vtype qn = pocl_fma(s,
                    pocl_fma(s,
                        pocl_fma(s,
                            pocl_fma(s,
                                (vtype)0.142316903342317766e-3,
                                (vtype)0.304455919504853031e-1),
                            (vtype)0.220638780716667420e0),
                        (vtype)0.447677206805497472e0),
                    (vtype)0.268297920532545909e0);

    vtype qd = pocl_fma(s,
                 pocl_fma(s,
                   pocl_fma(s,
                     pocl_fma(s,
                       (vtype)0.389525873944742195e-1,
                       (vtype)0.424602594203847109e0),
                     (vtype)0.141254259931958921e1),
                   (vtype)0.182596787737507063e1),
                 (vtype)0.804893761597637733e0);

    vtype q = r * s * qn / qd;
    r = (chi - ((q - clo) - r)) / pi;
    vtype vp = v / pi;

    vtype z = isnan(x) ? x : (vtype)0.5;
    z = (v <= (vtype)0x1.0p+56) ? r : z;
    z = (v < (vtype)0x1.0p-26) ? vp : z;
    return x == v ? z : -z;
}
