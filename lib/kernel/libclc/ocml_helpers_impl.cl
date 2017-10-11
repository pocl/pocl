/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See ROCM_LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/


#ifndef vtype
#error vtype must be defined
#endif

#ifndef v2type
#error v2type must be defined
#endif

#if defined(FLOAT_SPECIALIZATION)

#define HIGH(X) as_vtype(as_utype(X) & (utype)0xfffff000U)

#define USE_FMA HAVE_FMA32

#endif

#if defined(DOUBLE_SPECIALIZATION)

#define USE_FMA HAVE_FMA64

#define HIGH(X) as_vtype(as_utype(X) & (utype)0xfffffffff8000000UL)

#endif


OCML_ATTR v2type
con(vtype a, vtype b)
{
    v2type c; c.lo = b; c.hi = a;
    return c;
}

OCML_ATTR v2type
csgn(v2type a, vtype b)
{
    return con(COPYSIGN(a.hi, b), COPYSIGN(a.lo, b));
}

OCML_ATTR v2type
csgn(v2type a, v2type b)
{
    return con(COPYSIGN(a.hi, b.hi), COPYSIGN(a.lo, b.lo));
}

OCML_ATTR v2type
fadd(vtype a, vtype b)
{
    vtype s = a + b;
    return con(s, b - (s - a));
}

OCML_ATTR v2type
nrm(v2type a)
{
    return fadd(a.hi, a.lo);
}

OCML_ATTR v2type
onrm(v2type a)
{
    vtype s = a.hi + a.lo;
    vtype t = a.lo - (s - a.hi);
    s = ISINF(a.hi) ? a.hi : s;
    return con(s, ISINF(s) ? (vtype)0 : t);
}

OCML_ATTR v2type
fsub(vtype a, vtype b)
{
    vtype d = a - b;
    return con(d, (a - d) - b);
}

OCML_ATTR v2type
add(vtype a, vtype b)
{
    vtype s = a + b;
    vtype d = s - a;
    return con(s, (a - (s - d)) + (b - d));
}

OCML_ATTR v2type
sub(vtype a, vtype b)
{
    vtype d = a - b;
    vtype e = d - a;
    return con(d, (a - (d - e)) - (b + e));
}

OCML_ATTR v2type
mul(vtype a, vtype b)
{
    vtype p = a * b;
    if (USE_FMA) {
        return con(p, FMA(a, b, -p));
    } else {
        vtype ah = HIGH(a);
        vtype al = a - ah;
        vtype bh = HIGH(b);
        vtype bl = b - bh;
        vtype p = a * b;
        return con(p, ((ah*bh - p) + ah*bl + al*bh) + al*bl);
    }
}

OCML_ATTR v2type
sqr(vtype a)
{
    vtype p = a * a;
    if (USE_FMA) {
        return con(p, FMA(a, a, -p));
    } else {
        vtype ah = HIGH(a);
        vtype al = a - ah;
        return con(p, ((ah*ah - p) + 2.0f*ah*al) + al*al);
    }
}

OCML_ATTR v2type
add(v2type a, vtype b)
{
    v2type s = add(a.hi, b);
    s.lo += a.lo;
    return nrm(s);
}

OCML_ATTR v2type
fadd(v2type a, vtype b)
{
    v2type s = fadd(a.hi, b);
    s.lo += a.lo;
    return nrm(s);
}

OCML_ATTR v2type
add(vtype a, v2type b)
{
    v2type s = add(a, b.hi);
    s.lo += b.lo;
    return nrm(s);
}

OCML_ATTR v2type
fadd(vtype a, v2type b)
{
    v2type s = fadd(a, b.hi);
    s.lo += b.lo;
    return nrm(s);
}

OCML_ATTR v2type
add(v2type a, v2type b)
{
    v2type s = add(a.hi, b.hi);
    v2type t = add(a.lo, b.lo);
    s.lo += t.hi;
    s = nrm(s);
    s.lo += t.lo;
    return nrm(s);
}

OCML_ATTR v2type
fadd(v2type a, v2type b)
{
    v2type s = fadd(a.hi, b.hi);
    s.lo += a.lo + b.lo;
    return nrm(s);
}

OCML_ATTR v2type
sub(v2type a, vtype b)
{
    v2type d = sub(a.hi, b);
    d.lo += a.lo;
    return nrm(d);
}

OCML_ATTR v2type
fsub(v2type a, vtype b)
{
    v2type d = fsub(a.hi, b);
    d.lo += a.lo;
    return nrm(d);
}

OCML_ATTR v2type
sub(vtype a, v2type b)
{
    v2type d = sub(a, b.hi);
    d.lo -= b.lo;
    return nrm(d);
}

OCML_ATTR v2type
fsub(vtype a, v2type b)
{
    v2type d = fsub(a, b.hi);
    d.lo -= b.lo;
    return nrm(d);
}

OCML_ATTR v2type
sub(v2type a, v2type b)
{
    v2type d = sub(a.hi, b.hi);
    v2type e = sub(a.lo, b.lo);
    d.lo += e.hi;
    d = nrm(d);
    d.lo += e.lo;
    return nrm(d);
}

OCML_ATTR v2type
fsub(v2type a, v2type b)
{
    v2type d = fsub(a.hi, b.hi);
    d.lo = d.lo + a.lo - b.lo;
    return nrm(d);
}

OCML_ATTR v2type
ldx(v2type a, int e)
{
    return con(LDEXP(a.hi, e), LDEXP(a.lo, e));
}

OCML_ATTR v2type
mul(v2type a, vtype b)
{
    v2type p = mul(a.hi, b);
    if (USE_FMA) {
        p.lo = FMA(a.lo, b, p.lo);
    } else {
        p.lo += a.lo * b;
    }
    return nrm(p);
}

OCML_ATTR v2type
omul(v2type a, vtype b)
{
    v2type p = mul(a.hi, b);
    if (USE_FMA) {
        p.lo = FMA(a.lo, b, p.lo);
    } else {
        p.lo += a.lo * b;
    }
    return onrm(p);
}

OCML_ATTR v2type
mul(vtype a, v2type b)
{
    v2type p = mul(a, b.hi);
    if (USE_FMA) {
        p.lo = FMA(a, b.lo, p.lo);
    } else {
        p.lo += a * b.lo;
    }
    return nrm(p);
}

OCML_ATTR v2type
omul(vtype a, v2type b)
{
    v2type p = mul(a, b.hi);
    if (USE_FMA) {
        p.lo = FMA(a, b.lo, p.lo);
    } else {
        p.lo += a * b.lo;
    }
    return onrm(p);
}

OCML_ATTR v2type
mul(v2type a, v2type b)
{
    v2type p = mul(a.hi, b.hi);
    if (USE_FMA) {
        p.lo += FMA(a.hi, b.lo, a.lo*b.hi);
    } else {
        p.lo += a.hi*b.lo + a.lo*b.hi;
    }
    return nrm(p);
}

OCML_ATTR v2type
omul(v2type a, v2type b)
{
    v2type p = mul(a.hi, b.hi);
    if (USE_FMA) {
        p.lo += FMA(a.hi, b.lo, a.lo*b.hi);
    } else {
        p.lo += a.hi*b.lo + a.lo*b.hi;
    }
    return onrm(p);
}

OCML_ATTR v2type
div(vtype a, vtype b)
{
    vtype r = RCP(b);
    vtype qhi = a * r;
    v2type p = mul(qhi, b);
    v2type d = fsub(a, p.hi);
    d.lo -= p.lo;
    vtype qlo = (d.hi + d.lo) * r;
    return fadd(qhi, qlo);
}

OCML_ATTR v2type
div(v2type a, vtype b)
{
    vtype r = RCP(b);
    vtype qhi = a.hi * r;
    v2type p = mul(qhi, b);
    v2type d = fsub(a.hi, p.hi);
    d.lo = d.lo + a.lo - p.lo;
    vtype qlo = (d.hi + d.lo) * r;
    return fadd(qhi, qlo);
}

OCML_ATTR v2type
div(vtype a, v2type b)
{
    vtype r = RCP(b.hi);
    vtype qhi = a * r;
    v2type p = mul(qhi, b);
    v2type d = fsub(a, p.hi);
    d.lo -= p.lo;
    vtype qlo = (d.hi + d.lo) * r;
    return fadd(qhi, qlo);
}

OCML_ATTR v2type
fdiv(v2type a, v2type b)
{
    vtype r = RCP(b.hi);
    vtype qhi = a.hi * r;
    v2type p = mul(qhi, b);
    v2type d = fsub(a.hi, p.hi);
    d.lo = d.lo - p.lo + a.lo;
    vtype qlo = (d.hi + d.lo) * r;
    return fadd(qhi, qlo);
}

OCML_ATTR v2type
div(v2type a, v2type b)
{
    vtype y = RCP(b.hi);
    vtype qhi = a.hi * y;
    v2type r = fsub(a, mul(qhi, b));
    vtype qmi = r.hi * y;
    r = fsub(r, mul(qmi, b));
    vtype qlo = r.hi * y;
    v2type q = fadd(qhi, qmi);
    q.lo += qlo;
    return nrm(q);
}

OCML_ATTR v2type
rcp(vtype b)
{
    vtype qhi = RCP(b);
    v2type p = mul(qhi, b);
    v2type d = fsub((vtype)1, p.hi);
    d.lo -= p.lo;
    vtype qlo = (d.hi + d.lo) * qhi;
    return fadd(qhi, qlo);
}

OCML_ATTR v2type
frcp(v2type b)
{
    vtype qhi = RCP(b.hi);
    v2type p = mul(qhi, b);
    v2type d = fsub((vtype)1, p.hi);
    d.lo -= p.lo;
    vtype qlo = (d.hi + d.lo) * qhi;
    return fadd(qhi, qlo);
}

OCML_ATTR v2type
rcp(v2type b)
{
    vtype qhi = RCP(b.hi);
    v2type r = fsub((vtype)1, mul(qhi, b));
    vtype qmi = r.hi * qhi;
    r = fsub(r, mul(qmi, b));
    vtype qlo = r.hi * qhi;
    v2type q = fadd(qhi, qmi);
    q.lo += qlo;
    return nrm(q);
}

OCML_ATTR v2type
sqr(v2type a)
{
    v2type p = sqr(a.hi);
    if (USE_FMA) {
        p.lo = FMA(a.lo, a.lo, FMA(a.hi, (vtype)2*a.lo, p.lo));
    } else {
        p.lo = p.lo + a.hi * a.lo * (vtype)2 + a.lo * a.lo;
    }
    return fadd(p.hi, p.lo);
}

OCML_ATTR v2type
root2(vtype a)
{
    vtype shi = SQRT(a);
    v2type e = fsub(a, sqr(shi));
    vtype slo = DIV(e.hi, (vtype)2 * shi);
    return fadd(shi, slo);
}

OCML_ATTR v2type
root2(v2type a)
{
    vtype shi = SQRT(a.hi);
    v2type e = fsub(a, sqr(shi));
    vtype slo = DIV(e.hi, (vtype)2 * shi);
    return fadd(shi, slo);
}

#undef USE_FMA
#undef HIGH
