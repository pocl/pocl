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

/*
   Algorithm:

   Based on:
   Ping-Tak Peter Tang
   "Table-driven implementation of the logarithm function in IEEE
   floating-point arithmetic"
   ACM Transactions on Mathematical Software (TOMS)
   Volume 16, Issue 4 (December 1990)

   x very close to 1.0 is handled differently, for x everywhere else
   a brief explanation is given below

   x = (2^m)*A
   x = (2^m)*(G+g) with (1 <= G < 2) and (g <= 2^(-8))
   x = (2^m)*2*(G/2+g/2)
   x = (2^m)*2*(F+f) with (0.5 <= F < 1) and (f <= 2^(-9))

   Y = (2^(-1))*(2^(-m))*(2^m)*A
   Now, range of Y is: 0.5 <= Y < 1

   F = 0x80 + (first 7 mantissa bits) + (8th mantissa bit)
   Now, range of F is: 128 <= F <= 256
   F = F / 256
   Now, range of F is: 0.5 <= F <= 1

   f = -(Y-F), with (f <= 2^(-9))

   log(x) = m*log(2) + log(2) + log(F-f)
   log(x) = m*log(2) + log(2) + log(F) + log(1-(f/F))
   log(x) = m*log(2) + log(2*F) + log(1-r)

   r = (f/F), with (r <= 2^(-8))
   r = f*(1/F) with (1/F) precomputed to avoid division

   log(x) = m*log(2) + log(G) - poly

   log(G) is precomputed
   poly = (r + (r^2)/2 + (r^3)/3 + (r^4)/4) + (r^5)/5))

   log(2) and log(G) need to be maintained in extra precision
   to avoid losing precision in the calculations


   For x close to 1.0, we employ the following technique to
   ensure faster convergence.

   log(x) = log((1+s)/(1-s)) = 2*s + (2/3)*s^3 + (2/5)*s^5 + (2/7)*s^7
   x = ((1+s)/(1-s))
   x = 1 + r
   s = r/(2+r)

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

#if defined(COMPILING_LOGB)
#define COMPILING_LOG2
#endif

#if defined(COMPILING_LOG2)
    const vtype LOG2E = (vtype)0x1.715476p+0f;      // 1.4426950408889634
    const vtype LOG2E_HEAD = (vtype)0x1.700000p+0f; // 1.4375
    const vtype LOG2E_TAIL = (vtype)0x1.547652p-8f; // 0.00519504072
#elif defined(COMPILING_LOG10)
    const vtype LOG10E = (vtype)0x1.bcb7b2p-2f;        // 0.43429448190325182
    const vtype LOG10E_HEAD = (vtype)0x1.bc0000p-2f;   // 0.43359375
    const vtype LOG10E_TAIL = (vtype)0x1.6f62a4p-11f;  // 0.0007007319
    const vtype LOG10_2_HEAD = (vtype)0x1.340000p-2f;  // 0.30078125
    const vtype LOG10_2_TAIL = (vtype)0x1.04d426p-12f; // 0.000248745637
#else
    const vtype LOG2_HEAD = (vtype)0x1.62e000p-1f;  // 0.693115234
    const vtype LOG2_TAIL = (vtype)0x1.0bfbe8p-15f; // 0.0000319461833
#endif

    utype xi = as_utype(x);
    utype ax = xi & (utype)EXSIGNBIT_SP32;

    // Calculations for |x-1| < 2^-4
    vtype r = x - (vtype)1.0f;
    itype near1 = (fabs(r) < (vtype)0x1.0p-4f);
    vtype u2 = MATH_DIVIDE(r, (vtype)2.0f + r);
    vtype corr = u2 * r;
    vtype u = u2 + u2;
    vtype v = u * u;
    vtype znear1, z1, z2;

    // 2/(5 * 2^5), 2/(3 * 2^3)
    z2 = pocl_fma(u,
           pocl_fma(v,
             (vtype)0x1.99999ap-7f,
             (vtype)0x1.555556p-4f)*v,
           -corr);

#if defined(COMPILING_LOG2)
    z1 = as_vtype(as_itype(r) & (itype)0xffff0000);
    z2 = z2 + (r - z1);
    znear1 = pocl_fma(z1, LOG2E_HEAD,
               pocl_fma(z2, LOG2E_HEAD,
                 pocl_fma(z1, LOG2E_TAIL, z2*LOG2E_TAIL)));
#elif defined(COMPILING_LOG10)
    z1 = as_vtype(as_itype(r) & (itype)0xffff0000);
    z2 = z2 + (r - z1);
    znear1 = pocl_fma(z1, LOG10E_HEAD,
               pocl_fma(z2, LOG10E_HEAD,
                 pocl_fma(z1, LOG10E_TAIL, z2*LOG10E_TAIL)));
#else
    znear1 = z2 + r;
#endif

    // Calculations for x not near 1
    itype m = as_itype(xi >> EXPSHIFTBITS_SP32) - (itype)EXPBIAS_SP32;

    // Normalize subnormal
    utype xis = as_utype(as_vtype(xi | (utype)0x3f800000) - (vtype)1.0f);
    itype ms = (as_itype(xis) >> EXPSHIFTBITS_SP32) - (itype)253;
    itype c = (m == -127);
    m = c ? ms : m;
    utype xin = c ? xis : xi;

    vtype mf = convert_vtype(m);
    utype indx = (xin & (utype)0x007f0000) + ((xin & (utype)0x00008000) << 1);

    // F - Y
    vtype f = as_vtype((utype)0x3f000000 | indx)
              - as_vtype((utype)0x3f000000 | (xin & MANTBITS_SP32));

    indx = indx >> 16;
    r = f * USE_VTABLE(log_inv_tbl, indx);

    // 1/3,  1/2
    vtype poly = pocl_fma(
                   pocl_fma(r, (vtype)0x1.555556p-2f, (vtype)0.5f),
                   r*r,
                   r);

#if defined(COMPILING_LOG2)
    v2type tv = USE_VTABLE(log2_tbl, indx);
    vtype s0 = tv.lo;
    vtype s1 = tv.hi;
    z1 = s0 + mf;
    z2 = pocl_fma(poly, -LOG2E, s1);
#elif defined(COMPILING_LOG10)
    v2type tv = USE_VTABLE(log10_tbl, indx);
    vtype s0 = tv.lo;
    vtype s1 = tv.hi;
    z1 = pocl_fma(mf, LOG10_2_HEAD, s0);
    z2 = pocl_fma(poly, -LOG10E, mf*LOG10_2_TAIL) + s1;
#else
    v2type tv = USE_VTABLE(loge_tbl, indx);
    vtype s0 = tv.lo;
    vtype s1 = tv.hi;
    z1 = pocl_fma(mf, LOG2_HEAD, s0);
    z2 = pocl_fma(mf, LOG2_TAIL, -poly) + s1;
#endif

    vtype z = z1 + z2;
    z = near1 ? znear1 : z;

    // Corner cases
    z = (ax >= (utype)PINFBITPATT_SP32) ? x : z;
    z = (xi != ax) ? as_vtype((utype)QNANBITPATT_SP32) : z;
    z = (ax == 0) ? as_vtype((utype)NINFBITPATT_SP32) : z;

    return z;
}
