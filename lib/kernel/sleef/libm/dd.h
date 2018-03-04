//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

typedef struct {
  vdouble x, y;
} vdouble2;

static INLINE CONST vdouble vupper_vd_vd(vdouble d) {
  return vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vcast_vm_i_i(0xffffffff, 0xf8000000)));
}

static INLINE CONST vdouble2 vcast_vd2_vd_vd(vdouble h, vdouble l) {
  vdouble2 ret = {h, l};
  return ret;
}

static INLINE CONST vdouble2 vcast_vd2_d_d(double h, double l) {
  vdouble2 ret = {vcast_vd_d(h), vcast_vd_d(l)};
  return ret;
}

static INLINE CONST vdouble2 vsel_vd2_vo_vd2_vd2(vopmask m, vdouble2 x, vdouble2 y) {
  vdouble2 r;
  r.x = vsel_vd_vo_vd_vd(m, x.x, y.x);
  r.y = vsel_vd_vo_vd_vd(m, x.y, y.y);
  return r;
}

static INLINE CONST vdouble2 vsel_vd2_vo_d_d_d_d(vopmask o, double x1, double y1, double x0, double y0) {
  vdouble2 r;
  r.x = vsel_vd_vo_d_d(o, x1, x0);
  r.y = vsel_vd_vo_d_d(o, y1, y0);
  return r;
}

static INLINE CONST vdouble vadd_vd_3vd(vdouble v0, vdouble v1, vdouble v2) {
  return vadd_vd_vd_vd(vadd_vd_vd_vd(v0, v1), v2);
}

static INLINE CONST vdouble vadd_vd_4vd(vdouble v0, vdouble v1, vdouble v2, vdouble v3) {
  return vadd_vd_3vd(vadd_vd_vd_vd(v0, v1), v2, v3);
}

static INLINE CONST vdouble vadd_vd_5vd(vdouble v0, vdouble v1, vdouble v2, vdouble v3, vdouble v4) {
  return vadd_vd_4vd(vadd_vd_vd_vd(v0, v1), v2, v3, v4);
}

static INLINE CONST vdouble vadd_vd_6vd(vdouble v0, vdouble v1, vdouble v2, vdouble v3, vdouble v4, vdouble v5) {
  return vadd_vd_5vd(vadd_vd_vd_vd(v0, v1), v2, v3, v4, v5);
}

static INLINE CONST vdouble vadd_vd_7vd(vdouble v0, vdouble v1, vdouble v2, vdouble v3, vdouble v4, vdouble v5, vdouble v6) {
  return vadd_vd_6vd(vadd_vd_vd_vd(v0, v1), v2, v3, v4, v5, v6);
}

static INLINE CONST vdouble vsub_vd_3vd(vdouble v0, vdouble v1, vdouble v2) {
  return vsub_vd_vd_vd(vsub_vd_vd_vd(v0, v1), v2);
}

static INLINE CONST vdouble vsub_vd_4vd(vdouble v0, vdouble v1, vdouble v2, vdouble v3) {
  return vsub_vd_3vd(vsub_vd_vd_vd(v0, v1), v2, v3);
}

static INLINE CONST vdouble vsub_vd_5vd(vdouble v0, vdouble v1, vdouble v2, vdouble v3, vdouble v4) {
  return vsub_vd_4vd(vsub_vd_vd_vd(v0, v1), v2, v3, v4);
}

static INLINE CONST vdouble vsub_vd_6vd(vdouble v0, vdouble v1, vdouble v2, vdouble v3, vdouble v4, vdouble v5) {
  return vsub_vd_5vd(vsub_vd_vd_vd(v0, v1), v2, v3, v4, v5);
}

//

static INLINE CONST vdouble2 ddneg_vd2_vd2(vdouble2 x) {
  return vcast_vd2_vd_vd(vneg_vd_vd(x.x), vneg_vd_vd(x.y));
}

static INLINE CONST vdouble2 ddabs_vd2_vd2(vdouble2 x) {
  return vcast_vd2_vd_vd(vabs_vd_vd(x.x),
			 vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(x.y), vand_vm_vm_vm(vreinterpret_vm_vd(x.x), vreinterpret_vm_vd(vcast_vd_d(-0.0))))));
}

static INLINE CONST vdouble2 ddnormalize_vd2_vd2(vdouble2 t) {
  vdouble2 s;

  s.x = vadd_vd_vd_vd(t.x, t.y);
  s.y = vadd_vd_vd_vd(vsub_vd_vd_vd(t.x, s.x), t.y);

  return s;
}

static INLINE CONST vdouble2 ddscale_vd2_vd2_vd(vdouble2 d, vdouble s) {
  vdouble2 r = {vmul_vd_vd_vd(d.x, s), vmul_vd_vd_vd(d.y, s)};
  return r;
}

static INLINE CONST vdouble2 ddadd_vd2_vd_vd(vdouble x, vdouble y) {
  vdouble2 r;

  r.x = vadd_vd_vd_vd(x, y);
  r.y = vadd_vd_vd_vd(vsub_vd_vd_vd(x, r.x), y);

  return r;
}

static INLINE CONST vdouble2 ddadd2_vd2_vd_vd(vdouble x, vdouble y) {
  vdouble2 r;

  r.x = vadd_vd_vd_vd(x, y);
  vdouble v = vsub_vd_vd_vd(r.x, x);
  r.y = vadd_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(r.x, v)), vsub_vd_vd_vd(y, v));

  return r;
}

static INLINE CONST vdouble2 ddadd_vd2_vd2_vd(vdouble2 x, vdouble y) {
  vdouble2 r;

  r.x = vadd_vd_vd_vd(x.x, y);
  r.y = vadd_vd_3vd(vsub_vd_vd_vd(x.x, r.x), y, x.y);

  return r;
}

static INLINE CONST vdouble2 ddsub_vd2_vd2_vd(vdouble2 x, vdouble y) {
  vdouble2 r;

  r.x = vsub_vd_vd_vd(x.x, y);
  r.y = vadd_vd_vd_vd(vsub_vd_vd_vd(vsub_vd_vd_vd(x.x, r.x), y), x.y);

  return r;
}

static INLINE CONST vdouble2 ddadd2_vd2_vd2_vd(vdouble2 x, vdouble y) {
  vdouble2 r;

  r.x = vadd_vd_vd_vd(x.x, y);
  vdouble v = vsub_vd_vd_vd(r.x, x.x);
  r.y = vadd_vd_vd_vd(vsub_vd_vd_vd(x.x, vsub_vd_vd_vd(r.x, v)), vsub_vd_vd_vd(y, v));
  r.y = vadd_vd_vd_vd(r.y, x.y);

  return r;
}

static INLINE CONST vdouble2 ddadd_vd2_vd_vd2(vdouble x, vdouble2 y) {
  vdouble2 r;

  r.x = vadd_vd_vd_vd(x, y.x);
  r.y = vadd_vd_3vd(vsub_vd_vd_vd(x, r.x), y.x, y.y);

  return r;
}

static INLINE CONST vdouble2 ddadd2_vd2_vd_vd2(vdouble x, vdouble2 y) {
  vdouble2 r;

  r.x  = vadd_vd_vd_vd(x, y.x);
  vdouble v = vsub_vd_vd_vd(r.x, x);
  r.y = vadd_vd_vd_vd(vadd_vd_vd_vd(vsub_vd_vd_vd(x, vsub_vd_vd_vd(r.x, v)), vsub_vd_vd_vd(y.x, v)), y.y);

  return r;
}

static INLINE CONST vdouble2 ddadd_vd2_vd2_vd2(vdouble2 x, vdouble2 y) {
  // |x| >= |y|

  vdouble2 r;

  r.x = vadd_vd_vd_vd(x.x, y.x);
  r.y = vadd_vd_4vd(vsub_vd_vd_vd(x.x, r.x), y.x, x.y, y.y);

  return r;
}

static INLINE CONST vdouble2 ddadd2_vd2_vd2_vd2(vdouble2 x, vdouble2 y) {
  vdouble2 r;

  r.x  = vadd_vd_vd_vd(x.x, y.x);
  vdouble v = vsub_vd_vd_vd(r.x, x.x);
  r.y = vadd_vd_vd_vd(vsub_vd_vd_vd(x.x, vsub_vd_vd_vd(r.x, v)), vsub_vd_vd_vd(y.x, v));
  r.y = vadd_vd_vd_vd(r.y, vadd_vd_vd_vd(x.y, y.y));

  return r;
}

static INLINE CONST vdouble2 ddsub_vd2_vd_vd(vdouble x, vdouble y) {
  // |x| >= |y|

  vdouble2 r;

  r.x = vsub_vd_vd_vd(x, y);
  r.y = vsub_vd_vd_vd(vsub_vd_vd_vd(x, r.x), y);

  return r;
}

static INLINE CONST vdouble2 ddsub_vd2_vd2_vd2(vdouble2 x, vdouble2 y) {
  // |x| >= |y|

  vdouble2 r;

  r.x = vsub_vd_vd_vd(x.x, y.x);
  r.y = vsub_vd_vd_vd(x.x, r.x);
  r.y = vsub_vd_vd_vd(r.y, y.x);
  r.y = vadd_vd_vd_vd(r.y, x.y);
  r.y = vsub_vd_vd_vd(r.y, y.y);

  return r;
}

#ifdef ENABLE_FMA_DP
static INLINE CONST vdouble2 dddiv_vd2_vd2_vd2(vdouble2 n, vdouble2 d) {
  vdouble2 q;
  vdouble t = vrec_vd_vd(d.x), u;

  q.x = vmul_vd_vd_vd(n.x, t);
  u = vfmapn_vd_vd_vd_vd(t, n.x, q.x);
  q.y = vfmanp_vd_vd_vd_vd(d.y, t, vfmanp_vd_vd_vd_vd(d.x, t, vcast_vd_d(1)));
  q.y = vfma_vd_vd_vd_vd(q.x, q.y, vfma_vd_vd_vd_vd(n.y, t, u));

  return q;
}

static INLINE CONST vdouble2 ddmul_vd2_vd_vd(vdouble x, vdouble y) {
  vdouble2 r;

  r.x = vmul_vd_vd_vd(x, y);
  r.y = vfmapn_vd_vd_vd_vd(x, y, r.x);

  return r;
}

static INLINE CONST vdouble2 ddsqu_vd2_vd2(vdouble2 x) {
  vdouble2 r;

  r.x = vmul_vd_vd_vd(x.x, x.x);
  r.y = vfma_vd_vd_vd_vd(vadd_vd_vd_vd(x.x, x.x), x.y, vfmapn_vd_vd_vd_vd(x.x, x.x, r.x));

  return r;
}

static INLINE CONST vdouble2 ddmul_vd2_vd2_vd2(vdouble2 x, vdouble2 y) {
  vdouble2 r;

  r.x = vmul_vd_vd_vd(x.x, y.x);
  r.y = vfma_vd_vd_vd_vd(x.x, y.y, vfma_vd_vd_vd_vd(x.y, y.x, vfmapn_vd_vd_vd_vd(x.x, y.x, r.x)));

  return r;
}

static INLINE CONST vdouble ddmul_vd_vd2_vd2(vdouble2 x, vdouble2 y) {
  return vfma_vd_vd_vd_vd(x.x, y.x, vfma_vd_vd_vd_vd(x.y, y.x, vmul_vd_vd_vd(x.x, y.y)));
}

static INLINE CONST vdouble ddsqu_vd_vd2(vdouble2 x) {
  return vfma_vd_vd_vd_vd(x.x, x.x, vadd_vd_vd_vd(vmul_vd_vd_vd(x.x, x.y), vmul_vd_vd_vd(x.x, x.y)));
}

static INLINE CONST vdouble2 ddmul_vd2_vd2_vd(vdouble2 x, vdouble y) {
  vdouble2 r;

  r.x = vmul_vd_vd_vd(x.x, y);
  r.y = vfma_vd_vd_vd_vd(x.y, y, vfmapn_vd_vd_vd_vd(x.x, y, r.x));

  return r;
}

static INLINE CONST vdouble2 ddrec_vd2_vd(vdouble d) {
  vdouble2 q;

  q.x = vrec_vd_vd(d);
  q.y = vmul_vd_vd_vd(q.x, vfmanp_vd_vd_vd_vd(d, q.x, vcast_vd_d(1)));

  return q;
}

static INLINE CONST vdouble2 ddrec_vd2_vd2(vdouble2 d) {
  vdouble2 q;

  q.x = vrec_vd_vd(d.x);
  q.y = vmul_vd_vd_vd(q.x, vfmanp_vd_vd_vd_vd(d.y, q.x, vfmanp_vd_vd_vd_vd(d.x, q.x, vcast_vd_d(1))));

  return q;
}
#else
static INLINE CONST vdouble2 dddiv_vd2_vd2_vd2(vdouble2 n, vdouble2 d) {
  vdouble t = vrec_vd_vd(d.x);
  vdouble dh  = vupper_vd_vd(d.x), dl  = vsub_vd_vd_vd(d.x,  dh);
  vdouble th  = vupper_vd_vd(t  ), tl  = vsub_vd_vd_vd(t  ,  th);
  vdouble nhh = vupper_vd_vd(n.x), nhl = vsub_vd_vd_vd(n.x, nhh);

  vdouble2 q;

  q.x = vmul_vd_vd_vd(n.x, t);

  vdouble u = vadd_vd_5vd(vsub_vd_vd_vd(vmul_vd_vd_vd(nhh, th), q.x), vmul_vd_vd_vd(nhh, tl), vmul_vd_vd_vd(nhl, th), vmul_vd_vd_vd(nhl, tl),
		    vmul_vd_vd_vd(q.x, vsub_vd_5vd(vcast_vd_d(1), vmul_vd_vd_vd(dh, th), vmul_vd_vd_vd(dh, tl), vmul_vd_vd_vd(dl, th), vmul_vd_vd_vd(dl, tl))));

  q.y = vmla_vd_vd_vd_vd(t, vsub_vd_vd_vd(n.y, vmul_vd_vd_vd(q.x, d.y)), u);

  return q;
}

static INLINE CONST vdouble2 ddmul_vd2_vd_vd(vdouble x, vdouble y) {
  vdouble xh = vupper_vd_vd(x), xl = vsub_vd_vd_vd(x, xh);
  vdouble yh = vupper_vd_vd(y), yl = vsub_vd_vd_vd(y, yh);
  vdouble2 r;

  r.x = vmul_vd_vd_vd(x, y);
  r.y = vadd_vd_5vd(vmul_vd_vd_vd(xh, yh), vneg_vd_vd(r.x), vmul_vd_vd_vd(xl, yh), vmul_vd_vd_vd(xh, yl), vmul_vd_vd_vd(xl, yl));

  return r;
}

static INLINE CONST vdouble2 ddmul_vd2_vd2_vd(vdouble2 x, vdouble y) {
  vdouble xh = vupper_vd_vd(x.x), xl = vsub_vd_vd_vd(x.x, xh);
  vdouble yh = vupper_vd_vd(y  ), yl = vsub_vd_vd_vd(y  , yh);
  vdouble2 r;

  r.x = vmul_vd_vd_vd(x.x, y);
  r.y = vadd_vd_6vd(vmul_vd_vd_vd(xh, yh), vneg_vd_vd(r.x), vmul_vd_vd_vd(xl, yh), vmul_vd_vd_vd(xh, yl), vmul_vd_vd_vd(xl, yl), vmul_vd_vd_vd(x.y, y));

  return r;
}

static INLINE CONST vdouble2 ddmul_vd2_vd2_vd2(vdouble2 x, vdouble2 y) {
  vdouble xh = vupper_vd_vd(x.x), xl = vsub_vd_vd_vd(x.x, xh);
  vdouble yh = vupper_vd_vd(y.x), yl = vsub_vd_vd_vd(y.x, yh);
  vdouble2 r;

  r.x = vmul_vd_vd_vd(x.x, y.x);
  r.y = vadd_vd_7vd(vmul_vd_vd_vd(xh, yh), vneg_vd_vd(r.x), vmul_vd_vd_vd(xl, yh), vmul_vd_vd_vd(xh, yl), vmul_vd_vd_vd(xl, yl), vmul_vd_vd_vd(x.x, y.y), vmul_vd_vd_vd(x.y, y.x));

  return r;
}

static INLINE CONST vdouble ddmul_vd_vd2_vd2(vdouble2 x, vdouble2 y) {
  vdouble xh = vupper_vd_vd(x.x), xl = vsub_vd_vd_vd(x.x, xh);
  vdouble yh = vupper_vd_vd(y.x), yl = vsub_vd_vd_vd(y.x, yh);

  return vadd_vd_6vd(vmul_vd_vd_vd(x.y, yh), vmul_vd_vd_vd(xh, y.y), vmul_vd_vd_vd(xl, yl), vmul_vd_vd_vd(xh, yl), vmul_vd_vd_vd(xl, yh), vmul_vd_vd_vd(xh, yh));
}

static INLINE CONST vdouble2 ddsqu_vd2_vd2(vdouble2 x) {
  vdouble xh = vupper_vd_vd(x.x), xl = vsub_vd_vd_vd(x.x, xh);
  vdouble2 r;

  r.x = vmul_vd_vd_vd(x.x, x.x);
  r.y = vadd_vd_5vd(vmul_vd_vd_vd(xh, xh), vneg_vd_vd(r.x), vmul_vd_vd_vd(vadd_vd_vd_vd(xh, xh), xl), vmul_vd_vd_vd(xl, xl), vmul_vd_vd_vd(x.x, vadd_vd_vd_vd(x.y, x.y)));

  return r;
}

static INLINE CONST vdouble ddsqu_vd_vd2(vdouble2 x) {
  vdouble xh = vupper_vd_vd(x.x), xl = vsub_vd_vd_vd(x.x, xh);

  return vadd_vd_5vd(vmul_vd_vd_vd(xh, x.y), vmul_vd_vd_vd(xh, x.y), vmul_vd_vd_vd(xl, xl), vadd_vd_vd_vd(vmul_vd_vd_vd(xh, xl), vmul_vd_vd_vd(xh, xl)), vmul_vd_vd_vd(xh, xh));
}

static INLINE CONST vdouble2 ddrec_vd2_vd(vdouble d) {
  vdouble t = vrec_vd_vd(d);
  vdouble dh = vupper_vd_vd(d), dl = vsub_vd_vd_vd(d, dh);
  vdouble th = vupper_vd_vd(t), tl = vsub_vd_vd_vd(t, th);
  vdouble2 q;

  q.x = t;
  q.y = vmul_vd_vd_vd(t, vsub_vd_5vd(vcast_vd_d(1), vmul_vd_vd_vd(dh, th), vmul_vd_vd_vd(dh, tl), vmul_vd_vd_vd(dl, th), vmul_vd_vd_vd(dl, tl)));

  return q;
}

static INLINE CONST vdouble2 ddrec_vd2_vd2(vdouble2 d) {
  vdouble t = vrec_vd_vd(d.x);
  vdouble dh = vupper_vd_vd(d.x), dl = vsub_vd_vd_vd(d.x, dh);
  vdouble th = vupper_vd_vd(t  ), tl = vsub_vd_vd_vd(t  , th);
  vdouble2 q;

  q.x = t;
  q.y = vmul_vd_vd_vd(t, vsub_vd_6vd(vcast_vd_d(1), vmul_vd_vd_vd(dh, th), vmul_vd_vd_vd(dh, tl), vmul_vd_vd_vd(dl, th), vmul_vd_vd_vd(dl, tl), vmul_vd_vd_vd(d.y, t)));

  return q;
}
#endif

static INLINE CONST vdouble2 ddsqrt_vd2_vd2(vdouble2 d) {
  vdouble t = vsqrt_vd_vd(vadd_vd_vd_vd(d.x, d.y));
  return ddscale_vd2_vd2_vd(ddmul_vd2_vd2_vd2(ddadd2_vd2_vd2_vd2(d, ddmul_vd2_vd_vd(t, t)), ddrec_vd2_vd(t)), vcast_vd_d(0.5));
}

static INLINE CONST vdouble2 ddsqrt_vd2_vd(vdouble d) {
  vdouble t = vsqrt_vd_vd(d);
  return ddscale_vd2_vd2_vd(ddmul_vd2_vd2_vd2(ddadd2_vd2_vd_vd2(d, ddmul_vd2_vd_vd(t, t)), ddrec_vd2_vd(t)), vcast_vd_d(0.5));
}

#ifndef SLEEF_DOUBLE_MINMAXNUM_AVAILABLE
static INLINE CONST vdouble vmaxnum_vd_vd_vd(vdouble x, vdouble y) {
  return vsel_vd_vo_vd_vd(visnan_vo_vd(y), x, vsel_vd_vo_vd_vd(vgt_vo_vd_vd(x, y), x, y));
}

static INLINE CONST vdouble vminnum_vd_vd_vd(vdouble x, vdouble y) {
  return vsel_vd_vo_vd_vd(visnan_vo_vd(y), x, vsel_vd_vo_vd_vd(vgt_vo_vd_vd(y, x), x, y));
}
#endif
