//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

typedef struct {
  vfloat x, y;
} vfloat2;

static INLINE CONST vfloat vupper_vf_vf(vfloat d) {
  return vreinterpret_vf_vi2(vand_vi2_vi2_vi2(vreinterpret_vi2_vf(d), vcast_vi2_i(0xfffff000)));
}

static INLINE CONST vfloat2 vcast_vf2_vf_vf(vfloat h, vfloat l) {
  vfloat2 ret = {h, l};
  return ret;
}

static INLINE CONST vfloat2 vcast_vf2_f_f(float h, float l) {
  vfloat2 ret = {vcast_vf_f(h), vcast_vf_f(l)};
  return ret;
}

static INLINE CONST vfloat2 vcast_vf2_d(double d) {
  vfloat2 ret = {vcast_vf_f(d), vcast_vf_f(d - (float)d)};
  return ret;
}

static INLINE CONST vfloat2 vsel_vf2_vo_vf2_vf2(vopmask m, vfloat2 x, vfloat2 y) {
  vfloat2 r;
  r.x = vsel_vf_vo_vf_vf(m, x.x, y.x);
  r.y = vsel_vf_vo_vf_vf(m, x.y, y.y);
  return r;
}

static INLINE CONST vfloat2 vsel_vf2_vo_f_f_f_f(vopmask o, float x1, float y1, float x0, float y0) {
  vfloat2 r;
  r.x = vsel_vf_vo_f_f(o, x1, x0);
  r.y = vsel_vf_vo_f_f(o, y1, y0);
  return r;
}

static INLINE CONST vfloat2 vsel_vf2_vo_vo_d_d_d(vopmask o0, vopmask o1, double d0, double d1, double d2) {
  return vsel_vf2_vo_vf2_vf2(o0, vcast_vf2_d(d0), vsel_vf2_vo_vf2_vf2(o1, vcast_vf2_d(d1), vcast_vf2_d(d2)));
}

static INLINE CONST vfloat2 vsel_vf2_vo_vo_vo_d_d_d_d(vopmask o0, vopmask o1, vopmask o2, double d0, double d1, double d2, double d3) {
  return vsel_vf2_vo_vf2_vf2(o0, vcast_vf2_d(d0), vsel_vf2_vo_vf2_vf2(o1, vcast_vf2_d(d1), vsel_vf2_vo_vf2_vf2(o2, vcast_vf2_d(d2), vcast_vf2_d(d3))));
}

static INLINE CONST vfloat2 vabs_vf2_vf2(vfloat2 x) {
  return vcast_vf2_vf_vf(vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0)), vreinterpret_vm_vf(x.x)), vreinterpret_vm_vf(x.x))),
			 vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0)), vreinterpret_vm_vf(x.x)), vreinterpret_vm_vf(x.y))));
}

static INLINE CONST vfloat vadd_vf_3vf(vfloat v0, vfloat v1, vfloat v2) {
  return vadd_vf_vf_vf(vadd_vf_vf_vf(v0, v1), v2);
}

static INLINE CONST vfloat vadd_vf_4vf(vfloat v0, vfloat v1, vfloat v2, vfloat v3) {
  return vadd_vf_3vf(vadd_vf_vf_vf(v0, v1), v2, v3);
}

static INLINE CONST vfloat vadd_vf_5vf(vfloat v0, vfloat v1, vfloat v2, vfloat v3, vfloat v4) {
  return vadd_vf_4vf(vadd_vf_vf_vf(v0, v1), v2, v3, v4);
}

static INLINE CONST vfloat vadd_vf_6vf(vfloat v0, vfloat v1, vfloat v2, vfloat v3, vfloat v4, vfloat v5) {
  return vadd_vf_5vf(vadd_vf_vf_vf(v0, v1), v2, v3, v4, v5);
}

static INLINE CONST vfloat vadd_vf_7vf(vfloat v0, vfloat v1, vfloat v2, vfloat v3, vfloat v4, vfloat v5, vfloat v6) {
  return vadd_vf_6vf(vadd_vf_vf_vf(v0, v1), v2, v3, v4, v5, v6);
}

static INLINE CONST vfloat vsub_vf_3vf(vfloat v0, vfloat v1, vfloat v2) {
  return vsub_vf_vf_vf(vsub_vf_vf_vf(v0, v1), v2);
}

static INLINE CONST vfloat vsub_vf_4vf(vfloat v0, vfloat v1, vfloat v2, vfloat v3) {
  return vsub_vf_3vf(vsub_vf_vf_vf(v0, v1), v2, v3);
}

static INLINE CONST vfloat vsub_vf_5vf(vfloat v0, vfloat v1, vfloat v2, vfloat v3, vfloat v4) {
  return vsub_vf_4vf(vsub_vf_vf_vf(v0, v1), v2, v3, v4);
}

//

static INLINE CONST vfloat2 dfneg_vf2_vf2(vfloat2 x) {
  return vcast_vf2_vf_vf(vneg_vf_vf(x.x), vneg_vf_vf(x.y));
}

static INLINE CONST vfloat2 dfabs_vf2_vf2(vfloat2 x) {
  return vcast_vf2_vf_vf(vabs_vf_vf(x.x),
			 vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(x.y), vand_vm_vm_vm(vreinterpret_vm_vf(x.x), vreinterpret_vm_vf(vcast_vf_f(-0.0f))))));
}

static INLINE CONST vfloat2 dfnormalize_vf2_vf2(vfloat2 t) {
  vfloat2 s;

  s.x = vadd_vf_vf_vf(t.x, t.y);
  s.y = vadd_vf_vf_vf(vsub_vf_vf_vf(t.x, s.x), t.y);

  return s;
}

static INLINE CONST vfloat2 dfscale_vf2_vf2_vf(vfloat2 d, vfloat s) {
  vfloat2 r = {vmul_vf_vf_vf(d.x, s), vmul_vf_vf_vf(d.y, s)};
  return r;
}

static INLINE CONST vfloat2 dfadd_vf2_vf_vf(vfloat x, vfloat y) {
  vfloat2 r;

  r.x = vadd_vf_vf_vf(x, y);
  r.y = vadd_vf_vf_vf(vsub_vf_vf_vf(x, r.x), y);

  return r;
}

static INLINE CONST vfloat2 dfadd2_vf2_vf_vf(vfloat x, vfloat y) {
  vfloat2 r;

  r.x = vadd_vf_vf_vf(x, y);
  vfloat v = vsub_vf_vf_vf(r.x, x);
  r.y = vadd_vf_vf_vf(vsub_vf_vf_vf(x, vsub_vf_vf_vf(r.x, v)), vsub_vf_vf_vf(y, v));

  return r;
}

static INLINE CONST vfloat2 dfadd2_vf2_vf_vf2(vfloat x, vfloat2 y) {
  vfloat2 r;

  r.x  = vadd_vf_vf_vf(x, y.x);
  vfloat v = vsub_vf_vf_vf(r.x, x);
  r.y = vadd_vf_vf_vf(vadd_vf_vf_vf(vsub_vf_vf_vf(x, vsub_vf_vf_vf(r.x, v)), vsub_vf_vf_vf(y.x, v)), y.y);

  return r;
}

static INLINE CONST vfloat2 dfadd_vf2_vf2_vf(vfloat2 x, vfloat y) {
  vfloat2 r;

  r.x = vadd_vf_vf_vf(x.x, y);
  r.y = vadd_vf_3vf(vsub_vf_vf_vf(x.x, r.x), y, x.y);

  return r;
}

static INLINE CONST vfloat2 dfsub_vf2_vf2_vf(vfloat2 x, vfloat y) {
  vfloat2 r;

  r.x = vsub_vf_vf_vf(x.x, y);
  r.y = vadd_vf_vf_vf(vsub_vf_vf_vf(vsub_vf_vf_vf(x.x, r.x), y), x.y);

  return r;
}

static INLINE CONST vfloat2 dfadd2_vf2_vf2_vf(vfloat2 x, vfloat y) {
  vfloat2 r;

  r.x = vadd_vf_vf_vf(x.x, y);
  vfloat v = vsub_vf_vf_vf(r.x, x.x);
  r.y = vadd_vf_vf_vf(vsub_vf_vf_vf(x.x, vsub_vf_vf_vf(r.x, v)), vsub_vf_vf_vf(y, v));
  r.y = vadd_vf_vf_vf(r.y, x.y);

  return r;
}

static INLINE CONST vfloat2 dfadd_vf2_vf_vf2(vfloat x, vfloat2 y) {
  vfloat2 r;

  r.x = vadd_vf_vf_vf(x, y.x);
  r.y = vadd_vf_3vf(vsub_vf_vf_vf(x, r.x), y.x, y.y);

  return r;
}

static INLINE CONST vfloat2 dfadd_vf2_vf2_vf2(vfloat2 x, vfloat2 y) {
  // |x| >= |y|

  vfloat2 r;

  r.x = vadd_vf_vf_vf(x.x, y.x);
  r.y = vadd_vf_4vf(vsub_vf_vf_vf(x.x, r.x), y.x, x.y, y.y);

  return r;
}

static INLINE CONST vfloat2 dfadd2_vf2_vf2_vf2(vfloat2 x, vfloat2 y) {
  vfloat2 r;

  r.x  = vadd_vf_vf_vf(x.x, y.x);
  vfloat v = vsub_vf_vf_vf(r.x, x.x);
  r.y = vadd_vf_vf_vf(vsub_vf_vf_vf(x.x, vsub_vf_vf_vf(r.x, v)), vsub_vf_vf_vf(y.x, v));
  r.y = vadd_vf_vf_vf(r.y, vadd_vf_vf_vf(x.y, y.y));

  return r;
}

static INLINE CONST vfloat2 dfsub_vf2_vf_vf(vfloat x, vfloat y) {
  // |x| >= |y|

  vfloat2 r;

  r.x = vsub_vf_vf_vf(x, y);
  r.y = vsub_vf_vf_vf(vsub_vf_vf_vf(x, r.x), y);

  return r;
}

static INLINE CONST vfloat2 dfsub_vf2_vf2_vf2(vfloat2 x, vfloat2 y) {
  // |x| >= |y|

  vfloat2 r;

  r.x = vsub_vf_vf_vf(x.x, y.x);
  r.y = vsub_vf_vf_vf(x.x, r.x);
  r.y = vsub_vf_vf_vf(r.y, y.x);
  r.y = vadd_vf_vf_vf(r.y, x.y);
  r.y = vsub_vf_vf_vf(r.y, y.y);

  return r;
}

#ifdef ENABLE_FMA_SP
static INLINE CONST vfloat2 dfdiv_vf2_vf2_vf2(vfloat2 n, vfloat2 d) {
  vfloat2 q;
  vfloat t = vrec_vf_vf(d.x), u;

  q.x = vmul_vf_vf_vf(n.x, t);
  u = vfmapn_vf_vf_vf_vf(t, n.x, q.x);
  q.y = vfmanp_vf_vf_vf_vf(d.y, t, vfmanp_vf_vf_vf_vf(d.x, t, vcast_vf_f(1)));
  q.y = vfma_vf_vf_vf_vf(q.x, q.y, vfma_vf_vf_vf_vf(n.y, t, u));

  return q;
}

static INLINE CONST vfloat2 dfmul_vf2_vf_vf(vfloat x, vfloat y) {
  vfloat2 r;

  r.x = vmul_vf_vf_vf(x, y);
  r.y = vfmapn_vf_vf_vf_vf(x, y, r.x);

  return r;
}

static INLINE CONST vfloat2 dfsqu_vf2_vf2(vfloat2 x) {
  vfloat2 r;

  r.x = vmul_vf_vf_vf(x.x, x.x);
  r.y = vfma_vf_vf_vf_vf(vadd_vf_vf_vf(x.x, x.x), x.y, vfmapn_vf_vf_vf_vf(x.x, x.x, r.x));

  return r;
}

static INLINE CONST vfloat dfsqu_vf_vf2(vfloat2 x) {
  return vfma_vf_vf_vf_vf(x.x, x.x, vadd_vf_vf_vf(vmul_vf_vf_vf(x.x, x.y), vmul_vf_vf_vf(x.x, x.y)));
}

static INLINE CONST vfloat2 dfmul_vf2_vf2_vf2(vfloat2 x, vfloat2 y) {
  vfloat2 r;

  r.x = vmul_vf_vf_vf(x.x, y.x);
  r.y = vfma_vf_vf_vf_vf(x.x, y.y, vfma_vf_vf_vf_vf(x.y, y.x, vfmapn_vf_vf_vf_vf(x.x, y.x, r.x)));

  return r;
}

static INLINE CONST vfloat dfmul_vf_vf2_vf2(vfloat2 x, vfloat2 y) {
  return vfma_vf_vf_vf_vf(x.x, y.x, vfma_vf_vf_vf_vf(x.y, y.x, vmul_vf_vf_vf(x.x, y.y)));
}

static INLINE CONST vfloat2 dfmul_vf2_vf2_vf(vfloat2 x, vfloat y) {
  vfloat2 r;

  r.x = vmul_vf_vf_vf(x.x, y);
  r.y = vfma_vf_vf_vf_vf(x.y, y, vfmapn_vf_vf_vf_vf(x.x, y, r.x));

  return r;
}

static INLINE CONST vfloat2 dfrec_vf2_vf(vfloat d) {
  vfloat2 q;

  q.x = vrec_vf_vf(d);
  q.y = vmul_vf_vf_vf(q.x, vfmanp_vf_vf_vf_vf(d, q.x, vcast_vf_f(1)));

  return q;
}

static INLINE CONST vfloat2 dfrec_vf2_vf2(vfloat2 d) {
  vfloat2 q;

  q.x = vrec_vf_vf(d.x);
  q.y = vmul_vf_vf_vf(q.x, vfmanp_vf_vf_vf_vf(d.y, q.x, vfmanp_vf_vf_vf_vf(d.x, q.x, vcast_vf_f(1))));

  return q;
}
#else
static INLINE CONST vfloat2 dfdiv_vf2_vf2_vf2(vfloat2 n, vfloat2 d) {
  vfloat t = vrec_vf_vf(d.x);
  vfloat dh  = vupper_vf_vf(d.x), dl  = vsub_vf_vf_vf(d.x,  dh);
  vfloat th  = vupper_vf_vf(t  ), tl  = vsub_vf_vf_vf(t  ,  th);
  vfloat nhh = vupper_vf_vf(n.x), nhl = vsub_vf_vf_vf(n.x, nhh);

  vfloat2 q;

  q.x = vmul_vf_vf_vf(n.x, t);

  vfloat u, w;
  w = vcast_vf_f(-1);
  w = vmla_vf_vf_vf_vf(dh, th, w);
  w = vmla_vf_vf_vf_vf(dh, tl, w);
  w = vmla_vf_vf_vf_vf(dl, th, w);
  w = vmla_vf_vf_vf_vf(dl, tl, w);
  w = vneg_vf_vf(w);

  u = vmla_vf_vf_vf_vf(nhh, th, vneg_vf_vf(q.x));
  u = vmla_vf_vf_vf_vf(nhh, tl, u);
  u = vmla_vf_vf_vf_vf(nhl, th, u);
  u = vmla_vf_vf_vf_vf(nhl, tl, u);
  u = vmla_vf_vf_vf_vf(q.x, w , u);

  q.y = vmla_vf_vf_vf_vf(t, vsub_vf_vf_vf(n.y, vmul_vf_vf_vf(q.x, d.y)), u);

  return q;
}

static INLINE CONST vfloat2 dfmul_vf2_vf_vf(vfloat x, vfloat y) {
  vfloat xh = vupper_vf_vf(x), xl = vsub_vf_vf_vf(x, xh);
  vfloat yh = vupper_vf_vf(y), yl = vsub_vf_vf_vf(y, yh);
  vfloat2 r;

  r.x = vmul_vf_vf_vf(x, y);

  vfloat t;
  t = vmla_vf_vf_vf_vf(xh, yh, vneg_vf_vf(r.x));
  t = vmla_vf_vf_vf_vf(xl, yh, t);
  t = vmla_vf_vf_vf_vf(xh, yl, t);
  t = vmla_vf_vf_vf_vf(xl, yl, t);
  r.y = t;

  return r;
}

static INLINE CONST vfloat2 dfmul_vf2_vf2_vf(vfloat2 x, vfloat y) {
  vfloat xh = vupper_vf_vf(x.x), xl = vsub_vf_vf_vf(x.x, xh);
  vfloat yh = vupper_vf_vf(y  ), yl = vsub_vf_vf_vf(y  , yh);
  vfloat2 r;

  r.x = vmul_vf_vf_vf(x.x, y);

  vfloat t;
  t = vmla_vf_vf_vf_vf(xh, yh, vneg_vf_vf(r.x));
  t = vmla_vf_vf_vf_vf(xl, yh, t);
  t = vmla_vf_vf_vf_vf(xh, yl, t);
  t = vmla_vf_vf_vf_vf(xl, yl, t);
  t = vmla_vf_vf_vf_vf(x.y, y, t);
  r.y = t;

  return r;
}

static INLINE CONST vfloat2 dfmul_vf2_vf2_vf2(vfloat2 x, vfloat2 y) {
  vfloat xh = vupper_vf_vf(x.x), xl = vsub_vf_vf_vf(x.x, xh);
  vfloat yh = vupper_vf_vf(y.x), yl = vsub_vf_vf_vf(y.x, yh);
  vfloat2 r;

  r.x = vmul_vf_vf_vf(x.x, y.x);

  vfloat t;
  t = vmla_vf_vf_vf_vf(xh, yh, vneg_vf_vf(r.x));
  t = vmla_vf_vf_vf_vf(xl, yh, t);
  t = vmla_vf_vf_vf_vf(xh, yl, t);
  t = vmla_vf_vf_vf_vf(xl, yl, t);
  t = vmla_vf_vf_vf_vf(x.x, y.y, t);
  t = vmla_vf_vf_vf_vf(x.y, y.x, t);
  r.y = t;

  return r;
}

static INLINE CONST vfloat dfmul_vf_vf2_vf2(vfloat2 x, vfloat2 y) {
  vfloat xh = vupper_vf_vf(x.x), xl = vsub_vf_vf_vf(x.x, xh);
  vfloat yh = vupper_vf_vf(y.x), yl = vsub_vf_vf_vf(y.x, yh);

  return vadd_vf_6vf(vmul_vf_vf_vf(x.y, yh), vmul_vf_vf_vf(xh, y.y), vmul_vf_vf_vf(xl, yl), vmul_vf_vf_vf(xh, yl), vmul_vf_vf_vf(xl, yh), vmul_vf_vf_vf(xh, yh));
}

static INLINE CONST vfloat2 dfsqu_vf2_vf2(vfloat2 x) {
  vfloat xh = vupper_vf_vf(x.x), xl = vsub_vf_vf_vf(x.x, xh);
  vfloat2 r;

  r.x = vmul_vf_vf_vf(x.x, x.x);

  vfloat t;
  t = vmla_vf_vf_vf_vf(xh, xh, vneg_vf_vf(r.x));
  t = vmla_vf_vf_vf_vf(vadd_vf_vf_vf(xh, xh), xl, t);
  t = vmla_vf_vf_vf_vf(xl, xl, t);
  t = vmla_vf_vf_vf_vf(x.x, vadd_vf_vf_vf(x.y, x.y), t);
  r.y = t;

  return r;
}

static INLINE CONST vfloat dfsqu_vf_vf2(vfloat2 x) {
  vfloat xh = vupper_vf_vf(x.x), xl = vsub_vf_vf_vf(x.x, xh);

  return vadd_vf_5vf(vmul_vf_vf_vf(xh, x.y), vmul_vf_vf_vf(xh, x.y), vmul_vf_vf_vf(xl, xl), vadd_vf_vf_vf(vmul_vf_vf_vf(xh, xl), vmul_vf_vf_vf(xh, xl)), vmul_vf_vf_vf(xh, xh));
}

static INLINE CONST vfloat2 dfrec_vf2_vf(vfloat d) {
  vfloat t = vrec_vf_vf(d);
  vfloat dh = vupper_vf_vf(d), dl = vsub_vf_vf_vf(d, dh);
  vfloat th = vupper_vf_vf(t), tl = vsub_vf_vf_vf(t, th);
  vfloat2 q;

  q.x = t;

  vfloat u = vcast_vf_f(-1);
  u = vmla_vf_vf_vf_vf(dh, th, u);
  u = vmla_vf_vf_vf_vf(dh, tl, u);
  u = vmla_vf_vf_vf_vf(dl, th, u);
  u = vmla_vf_vf_vf_vf(dl, tl, u);
  q.y = vmul_vf_vf_vf(vneg_vf_vf(t), u);

  return q;
}

static INLINE CONST vfloat2 dfrec_vf2_vf2(vfloat2 d) {
  vfloat t = vrec_vf_vf(d.x);
  vfloat dh = vupper_vf_vf(d.x), dl = vsub_vf_vf_vf(d.x, dh);
  vfloat th = vupper_vf_vf(t  ), tl = vsub_vf_vf_vf(t  , th);
  vfloat2 q;

  q.x = t;

  vfloat u = vcast_vf_f(-1);
  u = vmla_vf_vf_vf_vf(dh, th, u);
  u = vmla_vf_vf_vf_vf(dh, tl, u);
  u = vmla_vf_vf_vf_vf(dl, th, u);
  u = vmla_vf_vf_vf_vf(dl, tl, u);
  u = vmla_vf_vf_vf_vf(d.y, t, u);
  q.y = vmul_vf_vf_vf(vneg_vf_vf(t), u);

  return q;
}
#endif

static INLINE CONST vfloat2 dfsqrt_vf2_vf2(vfloat2 d) {
#ifdef ENABLE_RECSQRT_SP
  vfloat x = vrecsqrt_vf_vf(vadd_vf_vf_vf(d.x, d.y));
  vfloat2 r = dfmul_vf2_vf2_vf(d, x);
  return dfscale_vf2_vf2_vf(dfmul_vf2_vf2_vf2(r, dfadd2_vf2_vf2_vf(dfmul_vf2_vf2_vf(r, x), vcast_vf_f(-3.0))), vcast_vf_f(-0.5));
#else
  vfloat t = vsqrt_vf_vf(vadd_vf_vf_vf(d.x, d.y));
  return dfscale_vf2_vf2_vf(dfmul_vf2_vf2_vf2(dfadd2_vf2_vf2_vf2(d, dfmul_vf2_vf_vf(t, t)), dfrec_vf2_vf(t)), vcast_vf_f(0.5));
#endif
}

static INLINE CONST vfloat2 dfsqrt_vf2_vf(vfloat d) {
  vfloat t = vsqrt_vf_vf(d);
  return dfscale_vf2_vf2_vf(dfmul_vf2_vf2_vf2(dfadd2_vf2_vf_vf2(d, dfmul_vf2_vf_vf(t, t)), dfrec_vf2_vf(t)), vcast_vf_f(0.5f));
}

#ifndef SLEEF_SINGLE_MINMAXNUM_AVAILABLE
static INLINE CONST vfloat vmaxnum_vf_vf_vf(vfloat x, vfloat y) {
  return vsel_vf_vo_vf_vf(visnan_vo_vf(y), x, vsel_vf_vo_vf_vf(vgt_vo_vf_vf(x, y), x, y));
}

static INLINE CONST vfloat vminnum_vf_vf_vf(vfloat x, vfloat y) {
  return vsel_vf_vo_vf_vf(visnan_vo_vf(y), x, vsel_vf_vo_vf_vf(vgt_vo_vf_vf(y, x), x, y));
}
#endif
