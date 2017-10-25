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

_CL_OVERLOADABLE vtype
#if defined(COMPILING_LOG2)
log2(vtype x)
#elif defined(COMPILING_LOGB)
logb(vtype x)
#elif defined(COMPILING_LOG10)
log10(vtype x)
#else
log(vtype x)
#endif
{

#ifdef COMPILING_LOGB
#define COMPILING_LOG2
#endif

#if defined(COMPILING_LOG10)
    // log10e_lead and log10e_tail sum to an extra-precision version of log10(e) (19 bits in lead)
    const vtype log10e_lead = (vtype)4.34293746948242187500e-01;  /* 0x3fdbcb7800000000 */
    const vtype log10e_tail = (vtype)7.3495500964015109100644e-7; /* 0x3ea8a93728719535 */
#elif defined(COMPILING_LOG2)
    // log2e_lead and log2e_tail sum to an extra-precision version of log2(e) (19 bits in lead)
    const vtype log2e_lead = (vtype)1.44269180297851562500E+00; /* 0x3FF7154400000000 */
    const vtype log2e_tail = (vtype)3.23791044778235969970E-06; /* 0x3ECB295C17F0BBBE */
#endif

    // log_thresh1 = 9.39412117004394531250e-1 = 0x3fee0faa00000000
    // log_thresh2 = 1.06449508666992187500 = 0x3ff1082c00000000
    const vtype log_thresh1 = (vtype)0x1.e0faap-1;
    const vtype log_thresh2 = (vtype)0x1.1082cp+0;

    itype is_near = (x >= log_thresh1) & (x <= log_thresh2);

    // Near 1 code
    vtype r = x - (vtype)1.0;
    vtype u = r / ((vtype)2.0 + r);
    vtype correction = r * u;
    u = u + u;
    vtype v = u * u;
    vtype r1 = r;

    const vtype ca_1 = (vtype)8.33333333333317923934e-02; /* 0x3fb55555555554e6 */
    const vtype ca_2 = (vtype)1.25000000037717509602e-02; /* 0x3f89999999bac6d4 */
    const vtype ca_3 = (vtype)2.23213998791944806202e-03; /* 0x3f62492307f1519f */
    const vtype ca_4 = (vtype)4.34887777707614552256e-04; /* 0x3f3c8034c85dfff0 */

    vtype r2 = pocl_fma(u*v,
                 pocl_fma(v,
                   pocl_fma(v,
                     pocl_fma(v, ca_4, ca_3),
                     ca_2),
                   ca_1),
                 -correction);

#if defined(COMPILING_LOG10)
    r = r1;
    r1 = as_vtype(as_utype(r1) & (utype)0xffffffff00000000);
    r2 = r2 + (r - r1);
    vtype ret_near = pocl_fma(log10e_lead, r1,
                       pocl_fma(log10e_lead, r2,
                         pocl_fma(log10e_tail, r1, log10e_tail * r2)));
#elif defined(COMPILING_LOG2)
    r = r1;
    r1 = as_vtype(as_utype(r1) & (utype)0xffffffff00000000);
    r2 = r2 + (r - r1);
    vtype ret_near = pocl_fma(log2e_lead, r1,
                       pocl_fma(log2e_lead, r2,
                         pocl_fma(log2e_tail, r1, log2e_tail * r2)));
#else
    vtype ret_near = r1 + r2;
#endif

    // This is the far from 1 code

    // Deal with subnormal
    utype ux = as_utype(x);
    utype uxs = as_utype(
                as_vtype((utype)0x03d0000000000000UL | ux)
                - (vtype)0x1.0p-962);
    itype c = (ux < IMPBIT_DP64);
    ux = c ? uxs : ux;
    itype expadjust = c ? (itype)60 : (itype)0;

    itype xexp = ((as_itype(ux) >> 52) & 0x7ff) - (itype)EXPBIAS_DP64 - expadjust;
    vtype f = as_vtype((utype)HALFEXPBITS_DP64 | (ux & (utype)MANTBITS_DP64));
    uinttype index = convert_uinttype(ux >> 45);
    index = (((uinttype)0x80 | (index & (uinttype)0x7e)) >> 1)
             + (index & (uinttype)0x1);

    v2type tv = USE_VTABLE(ln_tbl, index - (uinttype)64);
    vtype z1 = tv.lo;
    vtype q = tv.hi;

    vtype f1 = convert_vtype(index) * 0x1.0p-7;
    vtype f2 = f - f1;
    u = f2 / pocl_fma(f2, (vtype)0.5, f1);
    v = u * u;

    const vtype cb_1 = (vtype)8.33333333333333593622e-02; /* 0x3fb5555555555557 */
    const vtype cb_2 = (vtype)1.24999999978138668903e-02; /* 0x3f89999999865ede */
    const vtype cb_3 = (vtype)2.23219810758559851206e-03; /* 0x3f6249423bd94741 */

    vtype poly = v * pocl_fma(v, pocl_fma(v, cb_3, cb_2), cb_1);
    vtype z2 = q + pocl_fma(u, poly, u);

    vtype dxexp = convert_vtype(xexp);
#if defined (COMPILING_LOG10)
    // Add xexp * log(2) to z1,z2 to get log(x)
    r1 = pocl_fma(dxexp, log2_lead, z1);
    r2 = pocl_fma(dxexp, log2_tail, z2);
    vtype ret_far = pocl_fma(log10e_lead, r1,
                      pocl_fma(log10e_lead, r2,
                        pocl_fma(log10e_tail, r1, log10e_tail*r2)));
#elif defined(COMPILING_LOG2)
    r1 = pocl_fma(log2e_lead, z1, dxexp);
    r2 = pocl_fma(log2e_lead, z2, pocl_fma(log2e_tail, z1, log2e_tail*z2));
    vtype ret_far = r1 + r2;
#else
    r1 = pocl_fma(dxexp, log2_lead, z1);
    r2 = pocl_fma(dxexp, log2_tail, z2);
    vtype ret_far = r1 + r2;
#endif

    vtype ret = is_near ? ret_near : ret_far;

    ret = isinf(x) ? as_vtype((utype)PINFBITPATT_DP64) : ret;
    ret = (isnan(x) | (x < (vtype)0.0))
           ? as_vtype((utype)QNANBITPATT_DP64) : ret;
    ret = (x == (vtype)0.0) ? as_vtype((utype)NINFBITPATT_DP64) : ret;
    return ret;
}
