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


#define bytealign(src0, src1, src2) \
  ((( ((convert_utype(src0)) << 32) | convert_utype(src1)) >> (((src2) & 3)*8)))

// Reduction for medium sized arguments
_CL_OVERLOADABLE void __pocl_remainder_piby2_medium(vtype x, vtype *r, vtype *rr, itype *regn) {

    // How many pi/2 is x a multiple of?
    const vtype two_by_pi = (vtype)0x1.45f306dc9c883p-1;
    const vtype dnpi2 = trunc(pocl_fma(x, two_by_pi, (vtype)0.5));

    const vtype piby2_h = (vtype)(-7074237752028440.0 / 0x1.0p+52);
    const vtype piby2_m = (vtype)(-2483878800010755.0 / 0x1.0p+105);
    const vtype piby2_t = (vtype)(-3956492004828932.0 / 0x1.0p+158);

    // Compute product of npi2 with 159 bits of 2/pi
    vtype p_hh = piby2_h * dnpi2;
    vtype p_ht = pocl_fma(piby2_h, dnpi2, -p_hh);
    vtype p_mh = piby2_m * dnpi2;
    vtype p_mt = pocl_fma(piby2_m, dnpi2, -p_mh);
    vtype p_th = piby2_t * dnpi2;
    vtype p_tt = pocl_fma(piby2_t, dnpi2, -p_th);

    // Reduce to 159 bits
    vtype ph = p_hh;
    vtype pm = p_ht + p_mh;
    vtype t = p_mh - (pm - p_ht);
    vtype pt = p_th + t + p_mt + p_tt;
    t = ph + pm; pm = pm - (t - ph); ph = t;
    t = pm + pt; pt = pt - (t - pm); pm = t;

    // Subtract from x
    t = x + ph;
    vtype qh = t + pm;
    vtype qt = pm - (qh - t) + pt;

    *r = qh;
    *rr = qt;
    *regn = convert_itype(dnpi2) & (itype)0x3;

}

// Given positive argument x, reduce it to the range [-pi/4,pi/4] using
// extra precision, and return the result in r, rr.
// Return value "regn" tells how many lots of pi/2 were subtracted
// from x to put it in the range [-pi/4,pi/4], mod 4.
_CL_OVERLOADABLE void __pocl_remainder_piby2_large(vtype x, vtype *r, vtype *rr, itype *regn) {

    itype ux = as_itype(x);
    itype e = (ux >> 52) - (itype)1023;
    itype i = max((itype)23, (e >> 3) + (itype)17);
    itype j = (itype)150 - i;
    itype j16 = j & (itype)(~0xf);
    vtype fract_temp;

    // The following extracts 192 consecutive bits of 2/pi aligned on an arbitrary byte boundary
    uinttype j16i = convert_uinttype(j16);
    utype4 q0 = USE_VTABLE(pibits_tbl, j16i);
    utype4 q1 = USE_VTABLE(pibits_tbl, (j16i + (uinttype)16));
    utype4 q2 = USE_VTABLE(pibits_tbl, (j16i + (uinttype)32));

    itype k = (j >> 2) & (itype)0x3;
    itype4 c;
    c.s0 = convert_inttype(k == (itype)0);
    c.s1 = convert_inttype(k == (itype)1);
    c.s2 = convert_inttype(k == (itype)2);
    c.s3 = convert_inttype(k == (itype)3);

    uinttype u0, u1, u2, u3, u4, u5, u6;

    u0 = c.s1 ? q0.s1 : q0.s0;
    u0 = c.s2 ? q0.s2 : u0;
    u0 = c.s3 ? q0.s3 : u0;

    u1 = c.s1 ? q0.s2 : q0.s1;
    u1 = c.s2 ? q0.s3 : u1;
    u1 = c.s3 ? q1.s0 : u1;

    u2 = c.s1 ? q0.s3 : q0.s2;
    u2 = c.s2 ? q1.s0 : u2;
    u2 = c.s3 ? q1.s1 : u2;

    u3 = c.s1 ? q1.s0 : q0.s3;
    u3 = c.s2 ? q1.s1 : u3;
    u3 = c.s3 ? q1.s2 : u3;

    u4 = c.s1 ? q1.s1 : q1.s0;
    u4 = c.s2 ? q1.s2 : u4;
    u4 = c.s3 ? q1.s3 : u4;

    u5 = c.s1 ? q1.s2 : q1.s1;
    u5 = c.s2 ? q1.s3 : u5;
    u5 = c.s3 ? q2.s0 : u5;

    u6 = c.s1 ? q1.s3 : q1.s2;
    u6 = c.s2 ? q2.s0 : u6;
    u6 = c.s3 ? q2.s1 : u6;

    const utype lomask = (utype)(0xffffffff);
    const utype himask = lomask << 32;
    const utype himask2 = (utype)0xffff00000000UL;

    utype v0 = bytealign(u1, u0, j) & lomask;
    utype v1 = bytealign(u2, u1, j) & lomask;
    utype v2 = bytealign(u3, u2, j) & lomask;
    utype v3 = bytealign(u4, u3, j) & lomask;
    utype v4 = bytealign(u5, u4, j) & lomask;
    utype v5 = bytealign(u6, u5, j) & lomask;
    utype v1hi = v1 << 32;
    utype v2hi = v2 << 32;
    utype v4hi = v4 << 32;
    utype v5hi = v5 << 32;

    // Place those 192 bits in 4 48-bit vtypes along with correct exponent
    // If i > 1018 we would get subnormals so we scale p up and x down to get the same product
    i = (itype)2 + 8*i;
    x *= (i > (itype)1018) ? (vtype)0x1.0p-136 : (vtype)1.0;
    i -= (i > (itype)1018) ? (itype)136 : (itype)0;

    utype ua = as_utype(1023 + 52 - i) << 52;
    vtype a = as_vtype(ua);
    utype addi3 = (utype)0x0300000000000000U;
    vtype p0 = as_vtype(v0 | (ua | (v1hi & himask2))  ) - a;
    ua += addi3;
    a = as_vtype(ua & himask);
    vtype p1 = as_vtype( ((v2 << 16) | (v1 >> 16))
                         | ((ua | (v2hi >> 16)) & himask) ) - a;
    ua += addi3;
    a = as_vtype(ua & himask);
    vtype p2 = as_vtype(v3 | ((ua | (v4hi & himask2))) ) - a;
    ua += addi3;
    a = as_vtype(ua & himask);
    vtype p3 = as_vtype( ((v5 << 16) | (v4 >> 16))
                         | ((ua | (v5hi >> 16)) & himask) ) - a;

    // Exact multiply
    vtype f0h = p0 * x;
    vtype f0l = pocl_fma(p0, x, (vtype)-f0h);
    vtype f1h = p1 * x;
    vtype f1l = pocl_fma(p1, x, (vtype)-f1h);
    vtype f2h = p2 * x;
    vtype f2l = pocl_fma(p2, x, (vtype)-f2h);
    vtype f3h = p3 * x;
    vtype f3l = pocl_fma(p3, x, (vtype)-f3h);

    // Accumulate product into 4 vtypes
    vtype s, t;

    vtype f3 = f3h + f2h;
    t = f2h - (f3 - f3h);
    s = f3l + t;
    t = t - (s - f3l);

    vtype f2 = s + f1h;
    t = f1h - (f2 - s) + t;
    s = f2l + t;
    t = t - (s - f2l);

    vtype f1 = s + f0h;
    t = f0h - (f1 - s) + t;
    s = f1l + t;

    vtype f0 = s + f0l;

    // Strip off unwanted large integer bits
    f3 = (vtype)0x1.0p+10 * fract((f3 * 0x1.0p-10), &fract_temp);
    f3 += ((f3 + f2) < (vtype)0.0) ? (vtype)0x1.0p+10 : (vtype)0.0;

    // Compute least significant integer bits
    t = f3 + f2;
    vtype di = t - fract(t, &fract_temp);
    i = convert_itype(di);

    // Shift out remaining integer part
    f3 -= di;
    s = f3 + f2; t = f2 - (s - f3); f3 = s; f2 = t;
    s = f2 + f1; t = f1 - (s - f2); f2 = s; f1 = t;
    f1 += f0;

    // Subtract 1 if fraction is >= 0.5, and update regn
#ifdef SINGLEVEC
    itype g = (f3 >= (vtype)0.5);
    i += g;
#else
    utype g = (as_utype(f3 >= (vtype)0.5) >> 63);
    i += as_itype(g);
#endif
    f3 -= convert_vtype(g);

    // Shift up bits
    s = f3 + f2; t = f2 -(s - f3); f3 = s; f2 = t + f1;

    // Multiply precise fraction by pi/2 to get radians
    const vtype p2h = (vtype)(7074237752028440.0 / 0x1.0p+52);
    const vtype p2t = (vtype)(4967757600021510.0 / 0x1.0p+106);

    vtype rhi = f3 * p2h;
    vtype rlo = pocl_fma(f2, p2h, pocl_fma(f3, p2t, pocl_fma(f3, p2h, -rhi)));

    *r = rhi + rlo;
    *rr = rlo - (*r - rhi);
    *regn = i & (itype)0x3;
}



_CL_OVERLOADABLE v2type __pocl_sincos_piby4(vtype x, vtype xx) {
    // Taylor series for sin(x) is x - x^3/3! + x^5/5! - x^7/7! ...
    //                      = x * (1 - x^2/3! + x^4/5! - x^6/7! ...
    //                      = x * f(w)
    // where w = x*x and f(w) = (1 - w/3! + w^2/5! - w^3/7! ...
    // We use a minimax approximation of (f(w) - 1) / w
    // because this produces an expansion in even powers of x.
    // If xx (the tail of x) is non-zero, we add a correction
    // term g(x,xx) = (1-x*x/2)*xx to the result, where g(x,xx)
    // is an approximation to cos(x)*sin(xx) valid because
    // xx is tiny relative to x.

    // Taylor series for cos(x) is 1 - x^2/2! + x^4/4! - x^6/6! ...
    //                      = f(w)
    // where w = x*x and f(w) = (1 - w/2! + w^2/4! - w^3/6! ...
    // We use a minimax approximation of (f(w) - 1 + w/2) / (w*w)
    // because this produces an expansion in even powers of x.
    // If xx (the tail of x) is non-zero, we subtract a correction
    // term g(x,xx) = x*xx to the result, where g(x,xx)
    // is an approximation to sin(x)*sin(xx) valid because
    // xx is tiny relative to x.

    const vtype sc1 = (vtype)-0.166666666666666646259241729;
    const vtype sc2 = (vtype)0.833333333333095043065222816e-2;
    const vtype sc3 = (vtype)-0.19841269836761125688538679e-3;
    const vtype sc4 = (vtype)0.275573161037288022676895908448e-5;
    const vtype sc5 = (vtype)-0.25051132068021699772257377197e-7;
    const vtype sc6 = (vtype)0.159181443044859136852668200e-9;

    const vtype cc1 = (vtype)0.41666666666666665390037e-1;
    const vtype cc2 = (vtype)-0.13888888888887398280412e-2;
    const vtype cc3 = (vtype)0.248015872987670414957399e-4;
    const vtype cc4 = (vtype)-0.275573172723441909470836e-6;
    const vtype cc5 = (vtype)0.208761463822329611076335e-8;
    const vtype cc6 = (vtype)-0.113826398067944859590880e-10;

    vtype x2 = x * x;
    vtype x3 = x2 * x;
    vtype r = 0.5 * x2;
    vtype t = (vtype)1.0 - r;

    vtype sp = pocl_fma(
                 pocl_fma(
                   pocl_fma(
                     pocl_fma(sc6, x2, sc5),
                     x2, sc4),
                   x2, sc3),
                 x2, sc2);

    vtype cp = t + pocl_fma(
                     pocl_fma(
                       pocl_fma(
                         pocl_fma(
                           pocl_fma(
                             pocl_fma(cc6, x2, cc5),
                             x2, cc4),
                           x2, cc3),
                         x2, cc2),
                       x2, cc1),
                     x2*x2,
                     pocl_fma(x, xx, ((vtype)1.0 - t) - r));

    v2type ret;
    ret.lo = x - pocl_fma(-x3, sc1,
                   pocl_fma(
                     pocl_fma(-x3, sp, 0.5*xx),
                   x2,
                 -xx));
    ret.hi = cp;

    return ret;
}
