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

#define bitalign(hi, lo, shift) \
  ((hi) << ((itype)32 - (shift))) | ((lo) >> (shift));

_CL_OVERLOADABLE void __pocl_fullMulS(vtype *hi, vtype *lo, vtype a, vtype b, vtype bh, vtype bt)
{
    if (HAVE_FMA32) {
        vtype ph = a * b;
        *hi = ph;
        *lo = fma(a, b, -ph);
    } else {
        vtype ah = as_vtype(as_utype(a) & (utype)0xfffff000U);
        vtype at = a - ah;
        vtype ph = a * b;
        vtype pt = pocl_fma(at, bt, pocl_fma(at, bh, pocl_fma(ah, bt, pocl_fma(ah, bh, -ph))));
        *hi = ph;
        *lo = pt;
    }
}

_CL_OVERLOADABLE vtype __pocl_removePi2S(vtype *hi, vtype *lo, vtype x)
{
    // 72 bits of pi/2
    const vtype fpiby2_1 = (vtype)( 0xC90FDA / 0x1.0p+23f);
    const vtype fpiby2_1_h = (vtype)( 0xC90 / 0x1.0p+11f);
    const vtype fpiby2_1_t = (vtype)( 0xFDA / 0x1.0p+23f);

    const vtype fpiby2_2 = (vtype)( 0xA22168 / 0x1.0p+47f);
    const vtype fpiby2_2_h = (vtype)( 0xA22 / 0x1.0p+35f);
    const vtype fpiby2_2_t = (vtype)( 0x168 / 0x1.0p+47f);

    const vtype fpiby2_3 = (vtype)( 0xC234C4 / 0x1.0p+71f);
    const vtype fpiby2_3_h = (vtype)( 0xC23 / 0x1.0p+59f);
    const vtype fpiby2_3_t = (vtype)( 0x4C4 / 0x1.0p+71f);

    const vtype twobypi = (vtype)0x1.45f306p-1f;

    vtype fnpi2 = trunc(pocl_fma(x, twobypi, (vtype)0.5f));

    // subtract n * pi/2 from x
    vtype rhead, rtail;
    __pocl_fullMulS(&rhead, &rtail, fnpi2, fpiby2_1, fpiby2_1_h, fpiby2_1_t);
    vtype v = x - rhead;
    vtype rem = v + (((x - v) - rhead) - rtail);

    vtype rhead2, rtail2;
    __pocl_fullMulS(&rhead2, &rtail2, fnpi2, fpiby2_2, fpiby2_2_h, fpiby2_2_t);
    v = rem - rhead2;
    rem = v + (((rem - v) - rhead2) - rtail2);

    vtype rhead3, rtail3;
    __pocl_fullMulS(&rhead3, &rtail3, fnpi2, fpiby2_3, fpiby2_3_h, fpiby2_3_t);
    v = rem - rhead3;

    *hi = v + ((rem - v) - rhead3);
    *lo = -rtail3;
    return fnpi2;
}

_CL_OVERLOADABLE itype __pocl_argReductionSmallS(vtype *r, vtype *rr, vtype x)
{
    vtype fnpi2 = __pocl_removePi2S(r, rr, x);
    return convert_itype(fnpi2) & (itype)0x3;
}

#define FULL_MUL(A, B, HI, LO) \
    LO = A * B; \
    HI = mul_hi(A, B)

#define FULL_MAD(A, B, C, HI, LO) \
    LO = ((A) * (B) + (C)); \
    HI = mul_hi(A, B); \
    HI += ((LO < C) ? (utype)1 : (utype)0)

#ifdef SINGLEVEC
#define SHIFT_MINUS_32 shift -= c << 5
#else
#define SHIFT_MINUS_32 shift -= c & (itype)32
#endif

_CL_OVERLOADABLE itype __pocl_argReductionLargeS(vtype *r, vtype *rr, vtype x)
{
    itype xe = (itype)(as_itype(x) >> 23) - (itype)127;
    utype xm = (utype)0x00800000U | (as_utype(x) & (utype)0x7fffffU);

    // 224 bits of 2/PI: . A2F9836E 4E441529 FC2757D1 F534DDC0 DB629599 3C439041 FE5163AB
    const utype b6 = (utype)0xA2F9836EU;
    const utype b5 = (utype)0x4E441529U;
    const utype b4 = (utype)0xFC2757D1U;
    const utype b3 = (utype)0xF534DDC0U;
    const utype b2 = (utype)0xDB629599U;
    const utype b1 = (utype)0x3C439041U;
    const utype b0 = (utype)0xFE5163ABU;

    utype p0, p1, p2, p3, p4, p5, p6, p7, c0, c1;

    FULL_MUL(xm, b0, c0, p0);
    FULL_MAD(xm, b1, c0, c1, p1);
    FULL_MAD(xm, b2, c1, c0, p2);
    FULL_MAD(xm, b3, c0, c1, p3);
    FULL_MAD(xm, b4, c1, c0, p4);
    FULL_MAD(xm, b5, c0, c1, p5);
    FULL_MAD(xm, b6, c1, p7, p6);

    itype fbits = (itype)224 + (itype)23 - xe;

    // shift amount to get 2 lsb of integer part at top 2 bits
    //   min: 25 (xe=18) max: 134 (xe=127)
    itype shift = (itype)254 - fbits;

    // Shift by up to 134/32 = 4 words
    itype c = (shift > 31);
    p7 = c ? p6 : p7;
    p6 = c ? p5 : p6;
    p5 = c ? p4 : p5;
    p4 = c ? p3 : p4;
    p3 = c ? p2 : p3;
    p2 = c ? p1 : p2;
    p1 = c ? p0 : p1;
    SHIFT_MINUS_32;

    c = (shift > 31);
    p7 = c ? p6 : p7;
    p6 = c ? p5 : p6;
    p5 = c ? p4 : p5;
    p4 = c ? p3 : p4;
    p3 = c ? p2 : p3;
    p2 = c ? p1 : p2;
    SHIFT_MINUS_32;

    c = (shift > 31);
    p7 = c ? p6 : p7;
    p6 = c ? p5 : p6;
    p5 = c ? p4 : p5;
    p4 = c ? p3 : p4;
    p3 = c ? p2 : p3;
    SHIFT_MINUS_32;

    c = (shift > 31);
    p7 = c ? p6 : p7;
    p6 = c ? p5 : p6;
    p5 = c ? p4 : p5;
    p4 = c ? p3 : p4;
    SHIFT_MINUS_32;

    // bitalign cannot handle a shift of 32
    c = (shift > 0);
    shift = (itype)32 - shift;
    utype t7 = bitalign(p7, p6, shift);
    utype t6 = bitalign(p6, p5, shift);
    utype t5 = bitalign(p5, p4, shift);
    p7 = c ? t7 : p7;
    p6 = c ? t6 : p6;
    p5 = c ? t5 : p5;

    // Get 2 lsb of itype part and msb of fraction
    itype i = as_itype(p7 >> 29);

    // Scoot up 2 more bits so only fraction remains
    p7 = bitalign(p7, p6, 30);
    p6 = bitalign(p6, p5, 30);
    p5 = bitalign(p5, p4, 30);

    // Subtract 1 if msb of fraction is 1, i.e. fraction >= 0.5
    utype flip = (i << 31) ? (utype)0xffffffffU : (utype)0U;
    utype sign = (i << 31) ? (utype)0x80000000U : (utype)0U;
    p7 = p7 ^ flip;
    p6 = p6 ^ flip;
    p5 = p5 ^ flip;

    // Find exponent and shift away leading zeroes and hidden bit
    xe = as_itype(clz(p7)) + (itype)1;
    shift = (itype)32 - xe;
    p7 = bitalign(p7, p6, shift);
    p6 = bitalign(p6, p5, shift);

    // Most significant part of fraction
    vtype q1 = as_vtype(as_itype(sign) | (((itype)127 - xe) << 23) | as_itype(p7 >> 9));

    // Shift out bits we captured on q1
    p7 = bitalign(p7, p6, 32-23);

    // Get 24 more bits of fraction in another vtype, there are not long strings of zeroes here
    itype xxe = as_itype(clz(p7)) + (itype)1;
    p7 = bitalign(p7, p6, (itype)32 - xxe);
    vtype q0 = as_vtype(as_itype(sign) | (((itype)127 - (xe + (itype)23 + xxe)) << 23) | as_itype(p7 >> 9));

    // At this point, the fraction q1 + q0 is correct to at least 48 bits
    // Now we need to multiply the fraction by pi/2
    // This loses us about 4 bits
    // pi/2 = C90 FDA A22 168 C23 4C4

    const vtype pio2h = (vtype)(0xc90fda / 0x1.0p+23f);
    const vtype pio2hh = (vtype)(0xc90 / 0x1.0p+11f);
    const vtype pio2ht = (vtype)(0xfda / 0x1.0p+23f);
    const vtype pio2t = (vtype)(0xa22168 / 0x1.0p+47f);

    vtype rh, rt;

    if (HAVE_FMA32) {
        rh = q1 * pio2h;
        rt = pocl_fma(q0, pio2h,
               pocl_fma(q1, pio2t,
                 pocl_fma(q1, pio2h, -rh)));
    } else {
        vtype q1h = as_vtype(as_utype(q1) & (utype)0xfffff000);
        vtype q1t = q1 - q1h;
        rh = q1 * pio2h;
        rt = pocl_fma(q1t, pio2ht,
               pocl_fma(q1t, pio2hh,
                 pocl_fma(q1h, pio2ht, pocl_fma(q1h, pio2hh, -rh))));
        rt = pocl_fma(q0, pio2h, pocl_fma(q1, pio2t, rt));
    }

    vtype t = rh + rt;
    rt = rt - (t - rh);

    *r = t;
    *rr = rt;
    return ((i >> 1) + (i & (itype)1)) & (itype)0x3;
}

#undef SHIFT_MINUS_32

_CL_OVERLOADABLE itype __pocl_argReductionS(vtype *r, vtype *rr, vtype x)
{
    itype retval = __pocl_argReductionSmallS(r, rr, x);
    itype cond = (x >= (vtype)0x1.0p+23f);
    if (SV_ANY(cond)) {
        retval = __pocl_argReductionLargeS(r, rr, x);
    }
    return retval;
}



// Evaluate single precisions in and cos of value in interval [-pi/4, pi/4]
_CL_OVERLOADABLE v2type __pocl_sincosf_piby4(vtype x)
{
    // Taylor series for sin(x) is x - x^3/3! + x^5/5! - x^7/7! ...
    // = x * (1 - x^2/3! + x^4/5! - x^6/7! ...
    // = x * f(w)
    // where w = x*x and f(w) = (1 - w/3! + w^2/5! - w^3/7! ...
    // We use a minimax approximation of (f(w) - 1) / w
    // because this produces an expansion in even powers of x.

    // Taylor series for cos(x) is 1 - x^2/2! + x^4/4! - x^6/6! ...
    // = f(w)
    // where w = x*x and f(w) = (1 - w/2! + w^2/4! - w^3/6! ...
    // We use a minimax approximation of (f(w) - 1 + w/2) / (w*w)
    // because this produces an expansion in even powers of x.

    const vtype sc1 = (vtype)-0.166666666638608441788607926e0F;
    const vtype sc2 = (vtype)0.833333187633086262120839299e-2F;
    const vtype sc3 = (vtype)-0.198400874359527693921333720e-3F;
    const vtype sc4 = (vtype)0.272500015145584081596826911e-5F;

    const vtype cc1 = (vtype)0.41666666664325175238031e-1F;
    const vtype cc2 = (vtype)-0.13888887673175665567647e-2F;
    const vtype cc3 = (vtype)0.24800600878112441958053e-4F;
    const vtype cc4 = (vtype)-0.27301013343179832472841e-6F;

    vtype x2 = x * x;

    v2type ret;
    ret.lo = pocl_fma(x*x2,
               pocl_fma(x2,
                 pocl_fma(x2,
                   pocl_fma(x2, sc4, sc3),
                   sc2),
                 sc1),
               x);
    ret.hi = pocl_fma(x2*x2,
               pocl_fma(x2,
                 pocl_fma(x2,
                   pocl_fma(x2, cc4, cc3),
                   cc2),
                 cc1),
                 pocl_fma(x2, (vtype)(-0.5f), (vtype)1.0f));
    return ret;
}


_CL_OVERLOADABLE vtype __pocl_cosf_piby4(vtype x, vtype y) {
    // Taylor series for cos(x) is 1 - x^2/2! + x^4/4! - x^6/6! ...
    // = f(w)
    // where w = x*x and f(w) = (1 - w/2! + w^2/4! - w^3/6! ...
    // We use a minimax approximation of (f(w) - 1 + w/2) / (w*w)
    // because this produces an expansion in even powers of x.

    const vtype c1 = (vtype)0.416666666e-1f;
    const vtype c2 = (vtype)-0.138888876e-2f;
    const vtype c3 = (vtype)0.248006008e-4f;
    const vtype c4 = (vtype)-0.2730101334e-6f;
    const vtype c5 = (vtype)2.0875723372e-09f;  // 0x310f74f6
    const vtype c6 = (vtype)-1.1359647598e-11f; // 0xad47d74e

    vtype z = x * x;
    vtype r = z * pocl_fma(z,
                    pocl_fma(z,
                      pocl_fma(z,
                        pocl_fma(z,
                          pocl_fma(z, c6,  c5),
                          c4),
                        c3),
                      c2),
                    c1);

    // if |x| < 0.3
    vtype qx = (vtype)0.0f;

    itype ix = as_itype(x) & (itype)EXSIGNBIT_SP32;

    //  0.78125 > |x| >= 0.3
    vtype xby4 = as_vtype(ix - (itype)0x01000000);
    qx = ((ix >= (itype)0x3e99999a) & (ix <= (itype)0x3f480000)) ? xby4 : qx;

    // x > 0.78125
    qx = (ix > (itype)0x3f480000) ? (vtype)0.28125f : qx;

    vtype hz = pocl_fma(z, (vtype)0.5f, -qx);
    vtype a = (vtype)1.0f - qx;
    vtype ret = a - (hz - pocl_fma(z, r, -x*y));
    return ret;
}


_CL_OVERLOADABLE vtype __pocl_sinf_piby4(vtype x, vtype y) {
    // Taylor series for sin(x) is x - x^3/3! + x^5/5! - x^7/7! ...
    // = x * (1 - x^2/3! + x^4/5! - x^6/7! ...
    // = x * f(w)
    // where w = x*x and f(w) = (1 - w/3! + w^2/5! - w^3/7! ...
    // We use a minimax approximation of (f(w) - 1) / w
    // because this produces an expansion in even powers of x.

    const vtype c1 = (vtype)-0.1666666666e0f;
    const vtype c2 = (vtype)0.8333331876e-2f;
    const vtype c3 = (vtype)-0.198400874e-3f;
    const vtype c4 = (vtype)0.272500015e-5f;
    const vtype c5 = (vtype)-2.5050759689e-08f; // 0xb2d72f34
    const vtype c6 = (vtype)1.5896910177e-10f;  // 0x2f2ec9d3

    vtype z = x * x;
    vtype v = z * x;
    vtype r = pocl_fma(z,
                pocl_fma(z,
                  pocl_fma(z,
                    pocl_fma(z, c6, c5),
                    c4),
                  c3),
                c2);
    vtype ret = x - pocl_fma(v, -c1,
                      pocl_fma(z,
                        pocl_fma(y, (vtype)0.5f, -v*r), -y));

    return ret;
}
