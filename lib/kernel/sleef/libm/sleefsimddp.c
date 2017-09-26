//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Always use -ffp-contract=off option to compile SLEEF.

#include <stdint.h>
#include <math.h>
#include <limits.h>
#include <float.h>

#include "misc.h"

#if (defined(_MSC_VER))
#pragma fp_contract (off)
#endif

#include "helpers.h"
#include "dd.h"

static INLINE vopmask vnot_vo64_vo64(vopmask x) {
  return vxor_vo_vo_vo(x, veq64_vo_vm_vm(vcast_vm_i_i(0, 0), vcast_vm_i_i(0, 0)));
}

static INLINE CONST vopmask vsignbit_vo_vd(vdouble d) {
  return veq64_vo_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(vcast_vd_d(-0.0)));
}

// return d0 < d1 ? x : y
static INLINE CONST vint vsel_vi_vd_vd_vi_vi(vdouble d0, vdouble d1, vint x, vint y) { return vsel_vi_vo_vi_vi(vcast_vo32_vo64(vlt_vo_vd_vd(d0, d1)), x, y); }

// return d0 < 0 ? x : 0
static INLINE CONST vint vsel_vi_vd_vi(vdouble d, vint x) { return vand_vi_vo_vi(vcast_vo32_vo64(vsignbit_vo_vd(d)), x); }

static INLINE CONST vopmask visnegzero_vo_vd(vdouble d) {
  return veq64_vo_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0)));
}

static INLINE CONST vopmask visnumber_vo_vd(vdouble x) {
  return vandnot_vo_vo_vo(visinf_vo_vd(x), veq_vo_vd_vd(x, x));
}

static INLINE CONST vmask vsignbit_vm_vd(vdouble d) {
  return vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0)));
}

static INLINE CONST vdouble vmulsign_vd_vd_vd(vdouble x, vdouble y) {
  return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(x), vsignbit_vm_vd(y)));
}

static INLINE CONST vdouble vcopysign_vd_vd_vd(vdouble x, vdouble y) {
  return vreinterpret_vd_vm(vxor_vm_vm_vm(vandnot_vm_vm_vm(vreinterpret_vm_vd(vcast_vd_d(-0.0)), vreinterpret_vm_vd(x)),
            vand_vm_vm_vm   (vreinterpret_vm_vd(vcast_vd_d(-0.0)), vreinterpret_vm_vd(y))));
}

static INLINE CONST vdouble vsign_vd_vd(vdouble d) {
  return vmulsign_vd_vd_vd(vcast_vd_d(1.0), d);
}

static INLINE CONST vdouble vpow2i_vd_vi(vint q) {
  q = vadd_vi_vi_vi(vcast_vi_i(0x3ff), q);
  vint2 r = vcastu_vi2_vi(q);
  return vreinterpret_vd_vi2(vsll_vi2_vi2_i(r, 20));
}

static INLINE CONST vdouble vldexp_vd_vd_vi(vdouble x, vint q) {
  vint m = vsra_vi_vi_i(q, 31);
  m = vsll_vi_vi_i(vsub_vi_vi_vi(vsra_vi_vi_i(vadd_vi_vi_vi(m, q), 9), m), 7);
  q = vsub_vi_vi_vi(q, vsll_vi_vi_i(m, 2));
  m = vadd_vi_vi_vi(vcast_vi_i(0x3ff), m);
  m = vandnot_vi_vo_vi(vgt_vo_vi_vi(vcast_vi_i(0), m), m);
  m = vsel_vi_vo_vi_vi(vgt_vo_vi_vi(m, vcast_vi_i(0x7ff)), vcast_vi_i(0x7ff), m);
  vint2 r = vcastu_vi2_vi(m);
  vdouble y = vreinterpret_vd_vi2(vsll_vi2_vi2_i(r, 20));
  return vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(x, y), y), y), y), vpow2i_vd_vi(q));
}

static INLINE CONST vdouble vldexp2_vd_vd_vi(vdouble d, vint e) {
  return vmul_vd_vd_vd(vmul_vd_vd_vd(d, vpow2i_vd_vi(vsra_vi_vi_i(e, 1))), vpow2i_vd_vi(vsub_vi_vi_vi(e, vsra_vi_vi_i(e, 1))));
}

static INLINE CONST vdouble vldexp3_vd_vd_vi(vdouble d, vint q) {
  return vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vd(d), vsll_vi2_vi2_i(vcastu_vi2_vi(q), 20)));
}

#ifndef ENABLE_AVX512F
static INLINE CONST vint vilogbk_vi_vd(vdouble d) {
  vopmask o = vlt_vo_vd_vd(d, vcast_vd_d(4.9090934652977266E-91));
  d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(vcast_vd_d(2.037035976334486E90), d), d);
  vint q = vcastu_vi_vi2(vreinterpret_vi2_vd(d));
  q = vand_vi_vi_vi(q, vcast_vi_i(((1 << 12)-1) << 20));
  q = vsrl_vi_vi_i(q, 20);
  q = vsub_vi_vi_vi(q, vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vcast_vi_i(300 + 0x3ff), vcast_vi_i(0x3ff)));
  return q;
}

static INLINE CONST vint vilogb2k_vi_vd(vdouble d) {
  vint q = vcastu_vi_vi2(vreinterpret_vi2_vd(d));
  q = vsrl_vi_vi_i(q, 20);
  q = vand_vi_vi_vi(q, vcast_vi_i(0x7ff));
  q = vsub_vi_vi_vi(q, vcast_vi_i(0x3ff));
  return q;
}
#endif

static INLINE CONST vopmask visint_vo_vd(vdouble d) {
  vdouble x = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0 / (1LL << 31))));
  x = vmla_vd_vd_vd_vd(vcast_vd_d(-(double)(1LL << 31)), x, d);
  return vor_vo_vo_vo(veq_vo_vd_vd(vtruncate_vd_vd(x), x),
          vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(1LL << 53)));
}

static INLINE CONST vopmask visodd_vo_vd(vdouble d) {
  vdouble x = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0 / (1LL << 31))));
  x = vmla_vd_vd_vd_vd(vcast_vd_d(-(double)(1LL << 31)), x, d);

  return vand_vo_vo_vo(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(vtruncate_vi_vd(x), vcast_vi_i(1)), vcast_vi_i(1))),
           vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(1LL << 53)));
}

EXPORT CONST vdouble xldexp(vdouble x, vint q) {
  // TODO this is probably possible to do more elegantly
  vint q_plus = vadd_vi_vi_vi(q, vcast_vi_i(10));
  q = vsel_vi_vo_vi_vi(
    veq_vo_vi_vi(q, vcast_vi_i(-2147483648)),
    q_plus,
    q);

  vdouble res = vldexp_vd_vd_vi(x, q);
  vdouble zero = vcast_vd_d(0.0);

  res = vsel_vd_vo_vd_vd(veq_vo_vd_vd(vabs_vd_vd(x), zero), x, res);
  res = vsel_vd_vo_vd_vd(visinf_vo_vd(x), x, res);
  res = vsel_vd_vo_vd_vd(visnan_vo_vd(x), x, res);

  return res;
}

EXPORT CONST vint xilogb(vdouble d) {
  vdouble e = vcast_vd_vi(vilogbk_vi_vd(vabs_vd_vd(d)));
  e = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0)), vcast_vd_d(FP_ILOGB0), e);
  e = vsel_vd_vo_vd_vd(visnan_vo_vd(d), vcast_vd_d(FP_ILOGBNAN), e);
  e = vsel_vd_vo_vd_vd(visinf_vo_vd(d), vcast_vd_d(INT_MAX), e);
  return vrint_vi_vd(e);
}

EXPORT CONST vdouble xsin(vdouble d) {
  vdouble u, s, r = d;
  vdouble dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI / (1 << 24))));
  dqh = vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24));
  vdouble dql = vrint_vd_vd(vmlapn_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI), dqh));
  vint ql = vrint_vi_vd(dql);

  d = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A), d);
  d = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A), d);
  d = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_B), d);
  d = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_B), d);
  d = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_C), d);
  d = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_C), d);
  d = vmla_vd_vd_vd_vd(vadd_vd_vd_vd(dqh, dql), vcast_vd_d(-PI_D), d);

  s = vmul_vd_vd_vd(d, d);

  d = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(1))), vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(d)));

  u = vcast_vd_d(-7.97255955009037868891952e-18);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.81009972710863200091251e-15));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-7.64712219118158833288484e-13));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(1.60590430605664501629054e-10));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.50521083763502045810755e-08));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.75573192239198747630416e-06));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.000198412698412696162806809));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00833333333333332974823815));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.166666666666666657414808));

  u = vadd_vd_vd_vd(vmul_vd_vd_vd(s, vmul_vd_vd_vd(u, d)), d);

  u = vsel_vd_vo_vd_vd(vandnot_vo_vo_vo(visinf_vo_vd(r),
          vor_vo_vo_vo(visnegzero_vo_vd(r),
                 vgt_vo_vd_vd(vabs_vd_vd(r), vcast_vd_d(TRIGRANGEMAX)))),
           vcast_vd_d(-0.0), u);

  return u;
}

EXPORT CONST vdouble xsin_u1(vdouble d) {
  vdouble u;
  vdouble2 s, t, x;
  vint ql;

  if (vtestallones_i_vo64(vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX2)))) {
    const vdouble dql = vrint_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)));
    ql = vrint_vi_vd(dql);
    u = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A2), d);
    s = ddadd_vd2_vd_vd (u, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B2)));
  } else {
    vdouble dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI / (1 << 24))));
    dqh = vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24));
    const vdouble dql = vrint_vd_vd(vmlapn_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI), dqh));
    ql = vrint_vi_vd(dql);

    u = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A), d);
    s = ddadd_vd2_vd_vd  (u, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_B)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_C)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_C)));
    s = ddadd_vd2_vd2_vd(s, vmul_vd_vd_vd(vadd_vd_vd_vd(dqh, dql), vcast_vd_d(-PI_D)));
  }

  t = s;
  s = ddsqu_vd2_vd2(s);

  u = vcast_vd_d(2.72052416138529567917983e-15);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-7.6429259411395447190023e-13));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(1.60589370117277896211623e-10));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-2.5052106814843123359368e-08));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.75573192104428224777379e-06));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.000198412698412046454654947));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00833333333333318056201922));

  x = ddadd_vd2_vd_vd2(vcast_vd_d(1), ddmul_vd2_vd2_vd2(ddadd_vd2_vd_vd(vcast_vd_d(-0.166666666666666657414808), vmul_vd_vd_vd(u, s.x)), s));

  u = ddmul_vd_vd2_vd2(t, x);

  u = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(1))),
                   vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(u)));
  u = vsel_vd_vo_vd_vd(vandnot_vo_vo_vo(visinf_vo_vd(d), vor_vo_vo_vo(visnegzero_vo_vd(d),
                      vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX)))),
           vcast_vd_d(-0.0), u);

  return u;
}

EXPORT CONST vdouble xcos(vdouble d) {
  vdouble u, s, r = d;
  vdouble dqh = vtruncate_vd_vd(vmla_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI / (1 << 23)), vcast_vd_d(-M_1_PI / (1 << 24))));
  vint ql = vrint_vi_vd(vadd_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)),
              vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-(1 << 23)), vcast_vd_d(-0.5))));
  dqh = vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24));
  ql = vadd_vi_vi_vi(vadd_vi_vi_vi(ql, ql), vcast_vi_i(1));
  vdouble dql = vcast_vd_vi(ql);

  d = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A * 0.5), d);
  d = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A * 0.5), d);
  d = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_B * 0.5), d);
  d = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_B * 0.5), d);
  d = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_C * 0.5), d);
  d = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_C * 0.5), d);
  d = vmla_vd_vd_vd_vd(vadd_vd_vd_vd(dqh, dql), vcast_vd_d(-PI_D * 0.5), d);

  s = vmul_vd_vd_vd(d, d);

  d = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(2)), vcast_vi_i(0))), vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(d)));

  u = vcast_vd_d(-7.97255955009037868891952e-18);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.81009972710863200091251e-15));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-7.64712219118158833288484e-13));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(1.60590430605664501629054e-10));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.50521083763502045810755e-08));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.75573192239198747630416e-06));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.000198412698412696162806809));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00833333333333332974823815));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.166666666666666657414808));

  u = vadd_vd_vd_vd(vmul_vd_vd_vd(s, vmul_vd_vd_vd(u, d)), d);

  u = vsel_vd_vo_vd_vd(vandnot_vo_vo_vo(visinf_vo_vd(r), vgt_vo_vd_vd(vabs_vd_vd(r), vcast_vd_d(TRIGRANGEMAX))), vcast_vd_d(1), u);

  return u;
}

EXPORT CONST vdouble xcos_u1(vdouble d) {
  vdouble u;
  vdouble2 s, t, x;
  vint ql;

  if (vtestallones_i_vo64(vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX2)))) {
    vdouble dql = vrint_vd_vd(vmla_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI), vcast_vd_d(-0.5)));
    dql = vmla_vd_vd_vd_vd(vcast_vd_d(2), dql, vcast_vd_d(1));
    ql = vrint_vi_vd(dql);
    s = ddadd2_vd2_vd_vd(d, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A2*0.5)));
    s = ddadd_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B2*0.5)));
  } else {
    vdouble dqh = vtruncate_vd_vd(vmla_vd_vd_vd_vd(d, vcast_vd_d(M_1_PI / (1 << 23)), vcast_vd_d(-M_1_PI / (1 << 24))));
    ql = vrint_vi_vd(vadd_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(M_1_PI)),
          vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-(1 << 23)), vcast_vd_d(-0.5))));
    dqh = vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24));
    ql = vadd_vi_vi_vi(vadd_vi_vi_vi(ql, ql), vcast_vi_i(1));
    const vdouble dql = vcast_vd_vi(ql);

    u = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A * 0.5), d);
    s = ddadd2_vd2_vd_vd(u, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A*0.5)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_B*0.5)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B*0.5)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_C*0.5)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_C*0.5)));
    s = ddadd_vd2_vd2_vd(s, vmul_vd_vd_vd(vadd_vd_vd_vd(dqh, dql), vcast_vd_d(-PI_D*0.5)));
  }

  t = s;
  s = ddsqu_vd2_vd2(s);

  u = vcast_vd_d(2.72052416138529567917983e-15);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-7.6429259411395447190023e-13));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(1.60589370117277896211623e-10));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-2.5052106814843123359368e-08));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.75573192104428224777379e-06));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.000198412698412046454654947));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00833333333333318056201922));

  x = ddadd_vd2_vd_vd2(vcast_vd_d(1), ddmul_vd2_vd2_vd2(ddadd_vd2_vd_vd(vcast_vd_d(-0.166666666666666657414808), vmul_vd_vd_vd(u, s.x)), s));

  u = ddmul_vd_vd2_vd2(t, x);

  u = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(2)), vcast_vi_i(0))), vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(u)));

  u = vsel_vd_vo_vd_vd(vandnot_vo_vo_vo(visinf_vo_vd(d), vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX))), vcast_vd_d(1), u);

  return u;
}

#ifdef ENABLE_GNUABI
#define TYPE2_FUNCATR static INLINE CONST
#define TYPE6_FUNCATR static INLINE CONST
#define XSINCOS sincosk
#define XSINCOS_U1 sincosk_u1
#define XSINCOSPI_U05 sincospik_u05
#define XSINCOSPI_U35 sincospik_u35
#define XMODF modfk
#else
#define TYPE2_FUNCATR EXPORT
#define TYPE6_FUNCATR EXPORT CONST
#define XSINCOS xsincos
#define XSINCOS_U1 xsincos_u1
#define XSINCOSPI_U05 xsincospi_u05
#define XSINCOSPI_U35 xsincospi_u35
#define XMODF xmodf
#endif

TYPE2_FUNCATR vdouble2 XSINCOS(vdouble d) {
  vopmask o;
  vdouble u, s, t, rx, ry;
  vdouble2 r;

  s = d;
  vdouble dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI / (1 << 24))));
  dqh = vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24));
  vdouble dql = vrint_vd_vd(vsub_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI)), dqh));
  vint ql = vrint_vi_vd(dql);

  s = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A * 0.5), s);
  s = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A * 0.5), s);
  s = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_B * 0.5), s);
  s = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_B * 0.5), s);
  s = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_C * 0.5), s);
  s = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_C * 0.5), s);
  s = vmla_vd_vd_vd_vd(vadd_vd_vd_vd(dqh, dql), vcast_vd_d(-PI_D * 0.5), s);

  t = s;

  s = vmul_vd_vd_vd(s, s);

  u = vcast_vd_d(1.58938307283228937328511e-10);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.50506943502539773349318e-08));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.75573131776846360512547e-06));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.000198412698278911770864914));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0083333333333191845961746));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.166666666666666130709393));

  rx = vmla_vd_vd_vd_vd(vmul_vd_vd_vd(u, s), t, t);
  rx = vsel_vd_vo_vd_vd(visnegzero_vo_vd(d), vcast_vd_d(-0.0), rx);

  u = vcast_vd_d(-1.13615350239097429531523e-11);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.08757471207040055479366e-09));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.75573144028847567498567e-07));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.48015872890001867311915e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.00138888888888714019282329));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0416666666666665519592062));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.5));

  ry = vmla_vd_vd_vd_vd(s, u, vcast_vd_d(1));

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(0)));
  r.x = vsel_vd_vo_vd_vd(o, rx, ry);
  r.y = vsel_vd_vo_vd_vd(o, ry, rx);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(2)), vcast_vi_i(2)));
  r.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.x)));

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(vadd_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(2)), vcast_vi_i(2)));
  r.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.y)));

  o = vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX));
  r.x = vreinterpret_vd_vm(vandnot_vm_vo64_vm(o, vreinterpret_vm_vd(r.x)));
  r.y = vsel_vd_vo_vd_vd(o, vcast_vd_d(1), r.y);

  o = visinf_vo_vd(d);
  r.x = vreinterpret_vd_vm(vor_vm_vo64_vm(o, vreinterpret_vm_vd(r.x)));
  r.y = vreinterpret_vd_vm(vor_vm_vo64_vm(o, vreinterpret_vm_vd(r.y)));

  return r;
}

TYPE2_FUNCATR vdouble2 XSINCOS_U1(vdouble d) {
  vopmask o;
  vdouble u, rx, ry;
  vdouble2 r, s, t, x;
  vint ql;

  if (vtestallones_i_vo64(vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX2)))) {
    const vdouble dql = vrint_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2 * M_1_PI)));
    ql = vrint_vi_vd(dql);
    u = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A2*0.5), d);
    s = ddadd_vd2_vd_vd (u, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B2*0.5)));
  } else {
    vdouble dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI / (1 << 24))));
    dqh = vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24));
    const vdouble dql = vrint_vd_vd(vsub_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI)), dqh));
    ql = vrint_vi_vd(dql);

    u = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A * 0.5), d);
    s = ddadd_vd2_vd_vd(u, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A*0.5)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_B*0.5)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B*0.5)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_C*0.5)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_C*0.5)));
    s = ddadd_vd2_vd2_vd(s, vmul_vd_vd_vd(vadd_vd_vd_vd(dqh, dql), vcast_vd_d(-PI_D*0.5)));
  }

  t = s;

  s.x = ddsqu_vd_vd2(s);

  u = vcast_vd_d(1.58938307283228937328511e-10);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-2.50506943502539773349318e-08));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.75573131776846360512547e-06));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.000198412698278911770864914));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0083333333333191845961746));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.166666666666666130709393));

  u = vmul_vd_vd_vd(u, vmul_vd_vd_vd(s.x, t.x));

  x = ddadd_vd2_vd2_vd(t, u);
  rx = vadd_vd_vd_vd(x.x, x.y);

  rx = vsel_vd_vo_vd_vd(visnegzero_vo_vd(d), vcast_vd_d(-0.0), rx);

  u = vcast_vd_d(-1.13615350239097429531523e-11);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.08757471207040055479366e-09));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-2.75573144028847567498567e-07));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.48015872890001867311915e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.00138888888888714019282329));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0416666666666665519592062));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-0.5));

  x = ddadd_vd2_vd_vd2(vcast_vd_d(1), ddmul_vd2_vd_vd(s.x, u));
  ry = vadd_vd_vd_vd(x.x, x.y);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(0)));
  r.x = vsel_vd_vo_vd_vd(o, rx, ry);
  r.y = vsel_vd_vo_vd_vd(o, ry, rx);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(2)), vcast_vi_i(2)));
  r.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.x)));

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(vadd_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(2)), vcast_vi_i(2)));
  r.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.y)));

  o = vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX));
  r.x = vreinterpret_vd_vm(vandnot_vm_vo64_vm(o, vreinterpret_vm_vd(r.x)));
  r.y = vsel_vd_vo_vd_vd(o, vcast_vd_d(1), r.y);

  o = visinf_vo_vd(d);
  r.x = vreinterpret_vd_vm(vor_vm_vo64_vm(o, vreinterpret_vm_vd(r.x)));
  r.y = vreinterpret_vd_vm(vor_vm_vo64_vm(o, vreinterpret_vm_vd(r.y)));

  return r;
}

TYPE2_FUNCATR vdouble2 XSINCOSPI_U05(vdouble d) {
  vopmask o;
  vdouble u, s, t, rx, ry;
  vdouble2 r, x, s2;

  u = vmul_vd_vd_vd(d, vcast_vd_d(4.0));
  vint q = vand_vi_vi_vi(vrint_vi_vd(vadd_vd_vd_vd(u, vcast_vd_d(0.5))), vcast_vi_i(~1));
  s = vsub_vd_vd_vd(u, vcast_vd_vi(q));

  t = s;
  s = vmul_vd_vd_vd(s, s);
  s2 = ddmul_vd2_vd_vd(t, t);

  //

  u = vcast_vd_d(-2.02461120785182399295868e-14);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(6.94821830580179461327784e-12));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-1.75724749952853179952664e-09));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(3.13361688966868392878422e-07));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-3.6576204182161551920361e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00249039457019271850274356));
  x = ddadd2_vd2_vd_vd2(vmul_vd_vd_vd(u, s), vcast_vd2_d_d(-0.0807455121882807852484731, 3.61852475067037104849987e-18));
  x = ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd2(s2, x), vcast_vd2_d_d(0.785398163397448278999491, 3.06287113727155002607105e-17));

  x = ddmul_vd2_vd2_vd(x, t);
  rx = vadd_vd_vd_vd(x.x, x.y);

  rx = vsel_vd_vo_vd_vd(visnegzero_vo_vd(d), vcast_vd_d(-0.0), rx);

  //

  u = vcast_vd_d(9.94480387626843774090208e-16);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-3.89796226062932799164047e-13));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(1.15011582539996035266901e-10));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-2.4611369501044697495359e-08));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(3.59086044859052754005062e-06));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.000325991886927389905997954));
  x = ddadd2_vd2_vd_vd2(vmul_vd_vd_vd(u, s), vcast_vd2_d_d(0.0158543442438155018914259, -1.04693272280631521908845e-18));
  x = ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd2(s2, x), vcast_vd2_d_d(-0.308425137534042437259529, -1.95698492133633550338345e-17));

  x = ddadd2_vd2_vd2_vd(ddmul_vd2_vd2_vd2(x, s2), vcast_vd_d(1));
  ry = vadd_vd_vd_vd(x.x, x.y);

  //

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(0)));
  r.x = vsel_vd_vo_vd_vd(o, rx, ry);
  r.y = vsel_vd_vo_vd_vd(o, ry, rx);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(4)), vcast_vi_i(4)));
  r.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.x)));

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(vadd_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(4)), vcast_vi_i(4)));
  r.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.y)));

  o = vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX3/4));
  r.x = vreinterpret_vd_vm(vandnot_vm_vo64_vm(o, vreinterpret_vm_vd(r.x)));
  r.y = vsel_vd_vo_vd_vd(o, vcast_vd_d(1), r.y);

  o = visinf_vo_vd(d);
  r.x = vreinterpret_vd_vm(vor_vm_vo64_vm(o, vreinterpret_vm_vd(r.x)));
  r.y = vreinterpret_vd_vm(vor_vm_vo64_vm(o, vreinterpret_vm_vd(r.y)));

  return r;
}

TYPE2_FUNCATR vdouble2 XSINCOSPI_U35(vdouble d) {
  vopmask o;
  vdouble u, s, t, rx, ry;
  vdouble2 r;

  u = vmul_vd_vd_vd(d, vcast_vd_d(4.0));
  vint q = vand_vi_vi_vi(vrint_vi_vd(vadd_vd_vd_vd(u, vcast_vd_d(0.5))), vcast_vi_i(~1));
  s = vsub_vd_vd_vd(u, vcast_vd_vi(q));

  t = s;
  s = vmul_vd_vd_vd(s, s);

  //

  u = vcast_vd_d(+0.6880638894766060136e-11);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.1757159564542310199e-8));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(+0.3133616327257867311e-6));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.3657620416388486452e-4));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(+0.2490394570189932103e-2));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.8074551218828056320e-1));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(+0.7853981633974482790e+0));

  rx = vmul_vd_vd_vd(u, t);

  //

  u = vcast_vd_d(-0.3860141213683794352e-12);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(+0.1150057888029681415e-9));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.2461136493006663553e-7));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(+0.3590860446623516713e-5));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.3259918869269435942e-3));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(+0.1585434424381541169e-1));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.3084251375340424373e+0));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(1));

  ry = u;

  //

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(0)));
  r.x = vsel_vd_vo_vd_vd(o, rx, ry);
  r.y = vsel_vd_vo_vd_vd(o, ry, rx);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(4)), vcast_vi_i(4)));
  r.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.x)));

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(vadd_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(4)), vcast_vi_i(4)));
  r.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(r.y)));

  o = vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX3/4));
  r.x = vreinterpret_vd_vm(vandnot_vm_vo64_vm(o, vreinterpret_vm_vd(r.x)));
  r.y = vreinterpret_vd_vm(vandnot_vm_vo64_vm(o, vreinterpret_vm_vd(r.y)));

  o = visinf_vo_vd(d);
  r.x = vreinterpret_vd_vm(vor_vm_vo64_vm(o, vreinterpret_vm_vd(r.x)));
  r.y = vreinterpret_vd_vm(vor_vm_vo64_vm(o, vreinterpret_vm_vd(r.y)));

  return r;
}

TYPE6_FUNCATR vdouble2 XMODF(vdouble x) {
  vdouble fr = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vcast_vd_d(1LL << 31), vcast_vd_vi(vtruncate_vi_vd(vmul_vd_vd_vd(x, vcast_vd_d(1.0 / (1LL << 31)))))));
  fr = vsub_vd_vd_vd(fr, vcast_vd_vi(vtruncate_vi_vd(fr)));
  fr = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(1LL << 52)), vcast_vd_d(0), fr);

  vdouble2 ret;

  ret.x = vcopysign_vd_vd_vd(fr, x);
  ret.y = vcopysign_vd_vd_vd(vsub_vd_vd_vd(x, fr), x);

  return ret;
}

#ifdef ENABLE_GNUABI
EXPORT void xsincos(vdouble a, double *ps, double *pc) {
  vdouble2 r = sincosk(a);
  vstoreu_v_p_vd(ps, r.x);
  vstoreu_v_p_vd(pc, r.y);
}

EXPORT void xsincos_u1(vdouble a, double *ps, double *pc) {
  vdouble2 r = sincosk_u1(a);
  vstoreu_v_p_vd(ps, r.x);
  vstoreu_v_p_vd(pc, r.y);
}

EXPORT void xsincospi_u05(vdouble a, double *ps, double *pc) {
  vdouble2 r = sincospik_u05(a);
  vstoreu_v_p_vd(ps, r.x);
  vstoreu_v_p_vd(pc, r.y);
}

EXPORT void xsincospi_u35(vdouble a, double *ps, double *pc) {
  vdouble2 r = sincospik_u35(a);
  vstoreu_v_p_vd(ps, r.x);
  vstoreu_v_p_vd(pc, r.y);
}

EXPORT CONST vdouble xmodf(vdouble a, double *iptr) {
  vdouble2 r = modfk(a);
  vstoreu_v_p_vd(iptr, r.y);
  return r.x;
}
#endif // #ifdef ENABLE_GNUABI

static INLINE CONST vdouble2 sinpik(vdouble d) {
  vopmask o;
  vdouble u, s, t;
  vdouble2 x, s2;

  u = vmul_vd_vd_vd(d, vcast_vd_d(4.0));
  vint q = vand_vi_vi_vi(vrint_vi_vd(vadd_vd_vd_vd(u, vcast_vd_d(0.5))), vcast_vi_i(~1));
  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(2)));

  s = vsub_vd_vd_vd(u, vcast_vd_vi(q));
  t = s;
  s = vmul_vd_vd_vd(s, s);
  s2 = ddmul_vd2_vd_vd(t, t);

  //

  u = vsel_vd_vo_d_d(o, 9.94480387626843774090208e-16, -2.02461120785182399295868e-14);
  u = vmla_vd_vd_vd_vd(u, s, vsel_vd_vo_d_d(o, -3.89796226062932799164047e-13, 6.948218305801794613277840e-12));
  u = vmla_vd_vd_vd_vd(u, s, vsel_vd_vo_d_d(o, 1.150115825399960352669010e-10, -1.75724749952853179952664e-09));
  u = vmla_vd_vd_vd_vd(u, s, vsel_vd_vo_d_d(o, -2.46113695010446974953590e-08, 3.133616889668683928784220e-07));
  u = vmla_vd_vd_vd_vd(u, s, vsel_vd_vo_d_d(o, 3.590860448590527540050620e-06, -3.65762041821615519203610e-05));
  u = vmla_vd_vd_vd_vd(u, s, vsel_vd_vo_d_d(o, -0.000325991886927389905997954, 0.0024903945701927185027435600));
  x = ddadd2_vd2_vd_vd2(vmul_vd_vd_vd(u, s),
      vsel_vd2_vo_d_d_d_d(o, 0.0158543442438155018914259, -1.04693272280631521908845e-18,
              -0.0807455121882807852484731, 3.61852475067037104849987e-18));
  x = ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd2(s2, x),
       vsel_vd2_vo_d_d_d_d(o, -0.308425137534042437259529, -1.95698492133633550338345e-17,
               0.785398163397448278999491, 3.06287113727155002607105e-17));

  x = ddmul_vd2_vd2_vd2(x, vsel_vd2_vo_vd2_vd2(o, s2, vcast_vd2_vd_vd(t, vcast_vd_d(0))));
  x = vsel_vd2_vo_vd2_vd2(o, ddadd2_vd2_vd2_vd(x, vcast_vd_d(1)), x);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(4)), vcast_vi_i(4)));
  x.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(x.x)));
  x.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(x.y)));

  return x;
}

EXPORT CONST vdouble xsinpi_u05(vdouble d) {
  vdouble2 x = sinpik(d);
  vdouble r = vadd_vd_vd_vd(x.x, x.y);

  r = vsel_vd_vo_vd_vd(visnegzero_vo_vd(d), vcast_vd_d(-0.0), r);
  r = vreinterpret_vd_vm(vandnot_vm_vo64_vm(vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX3/4)), vreinterpret_vm_vd(r)));
  r = vreinterpret_vd_vm(vor_vm_vo64_vm(visinf_vo_vd(d), vreinterpret_vm_vd(r)));

  return r;
}

static INLINE CONST vdouble2 cospik(vdouble d) {
  vopmask o;
  vdouble u, s, t;
  vdouble2 x, s2;

  u = vmul_vd_vd_vd(d, vcast_vd_d(4.0));
  vint q = vand_vi_vi_vi(vrint_vi_vd(vadd_vd_vd_vd(u, vcast_vd_d(0.5))), vcast_vi_i(~1));
  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(0)));

  s = vsub_vd_vd_vd(u, vcast_vd_vi(q));
  t = s;
  s = vmul_vd_vd_vd(s, s);
  s2 = ddmul_vd2_vd_vd(t, t);

  //

  u = vsel_vd_vo_d_d(o, 9.94480387626843774090208e-16, -2.02461120785182399295868e-14);
  u = vmla_vd_vd_vd_vd(u, s, vsel_vd_vo_d_d(o, -3.89796226062932799164047e-13, 6.948218305801794613277840e-12));
  u = vmla_vd_vd_vd_vd(u, s, vsel_vd_vo_d_d(o, 1.150115825399960352669010e-10, -1.75724749952853179952664e-09));
  u = vmla_vd_vd_vd_vd(u, s, vsel_vd_vo_d_d(o, -2.46113695010446974953590e-08, 3.133616889668683928784220e-07));
  u = vmla_vd_vd_vd_vd(u, s, vsel_vd_vo_d_d(o, 3.590860448590527540050620e-06, -3.65762041821615519203610e-05));
  u = vmla_vd_vd_vd_vd(u, s, vsel_vd_vo_d_d(o, -0.000325991886927389905997954, 0.0024903945701927185027435600));
  x = ddadd2_vd2_vd_vd2(vmul_vd_vd_vd(u, s),
      vsel_vd2_vo_d_d_d_d(o, 0.0158543442438155018914259, -1.04693272280631521908845e-18,
              -0.0807455121882807852484731, 3.61852475067037104849987e-18));
  x = ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd2(s2, x),
       vsel_vd2_vo_d_d_d_d(o, -0.308425137534042437259529, -1.95698492133633550338345e-17,
               0.785398163397448278999491, 3.06287113727155002607105e-17));

  x = ddmul_vd2_vd2_vd2(x, vsel_vd2_vo_vd2_vd2(o, s2, vcast_vd2_vd_vd(t, vcast_vd_d(0))));
  x = vsel_vd2_vo_vd2_vd2(o, ddadd2_vd2_vd2_vd(x, vcast_vd_d(1)), x);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(vadd_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(4)), vcast_vi_i(4)));
  x.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(x.x)));
  x.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(x.y)));

  return x;
}

EXPORT CONST vdouble xcospi_u05(vdouble d) {
  vdouble2 x = cospik(d);
  vdouble r = vadd_vd_vd_vd(x.x, x.y);

  r = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX3/4)), vcast_vd_d(1), r);
  r = vreinterpret_vd_vm(vor_vm_vo64_vm(visinf_vo_vd(d), vreinterpret_vm_vd(r)));

  return r;
}

EXPORT CONST vdouble xtan(vdouble d) {
  vdouble u, s, x;
  vopmask o;
  vdouble dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI / (1 << 24))));
  dqh = vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24));
  vdouble dql = vrint_vd_vd(vsub_vd_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI)), dqh));
  vint ql = vrint_vi_vd(dql);

  x = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A * 0.5), d);
  x = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A * 0.5), x);
  x = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_B * 0.5), x);
  x = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_B * 0.5), x);
  x = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_C * 0.5), x);
  x = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_C * 0.5), x);
  x = vmla_vd_vd_vd_vd(vadd_vd_vd_vd(dqh, dql), vcast_vd_d(-PI_D * 0.5), x);

  s = vmul_vd_vd_vd(x, x);

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(1)));
  x = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(x)));

  u = vcast_vd_d(9.99583485362149960784268e-06);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-4.31184585467324750724175e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000103573238391744000389851));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-0.000137892809714281708733524));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000157624358465342784274554));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(-6.07500301486087879295969e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000148898734751616411290179));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000219040550724571513561967));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000595799595197098359744547));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00145461240472358871965441));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0035923150771440177410343));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00886321546662684547901456));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0218694899718446938985394));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0539682539049961967903002));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.133333333334818976423364));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.333333333333320047664472));

  u = vmla_vd_vd_vd_vd(s, vmul_vd_vd_vd(u, x), x);

  u = vsel_vd_vo_vd_vd(o, vrec_vd_vd(u), u);

#ifndef ENABLE_AVX512F
  u = vreinterpret_vd_vm(vor_vm_vo64_vm(visinf_vo_vd(d), vreinterpret_vm_vd(u)));
  u = vsel_vd_vo_vd_vd(visnegzero_vo_vd(d), vcast_vd_d(-0.0), u);
#else
  u = vfixup_vd_vd_vd_vi2_i(u, d, vcast_vi2_i((3 << (4*4)) | (3 << (5*4))), 0);
#endif

  return u;
}

EXPORT CONST vdouble xtan_u1(vdouble d) {
  vdouble u;
  vdouble2 s, t, x;
  vopmask o;
  vint ql;

  if (vtestallones_i_vo64(vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX2)))) {
    vdouble dql = vrint_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2 * M_1_PI)));
    ql = vrint_vi_vd(dql);
    u = vmla_vd_vd_vd_vd(dql, vcast_vd_d(-PI_A2*0.5), d);
    s = ddadd_vd2_vd_vd (u, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B2*0.5)));
  } else {
    vdouble dqh = vtruncate_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(2*M_1_PI / (1 << 24))));
    dqh = vmul_vd_vd_vd(dqh, vcast_vd_d(1 << 24));
    s = ddadd2_vd2_vd2_vd(ddmul_vd2_vd2_vd(vcast_vd2_d_d(M_2_PI_H, M_2_PI_L), d),
        vsub_vd_vd_vd(vsel_vd_vo_vd_vd(vlt_vo_vd_vd(d, vcast_vd_d(0)),
               vcast_vd_d(-0.5), vcast_vd_d(0.5)), dqh));
    const vdouble dql = vtruncate_vd_vd(vadd_vd_vd_vd(s.x, s.y));
    ql = vrint_vi_vd(dql);

    u = vmla_vd_vd_vd_vd(dqh, vcast_vd_d(-PI_A * 0.5), d);
    s = ddadd_vd2_vd_vd(u, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_A*0.5            )));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_B*0.5)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_B*0.5            )));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dqh, vcast_vd_d(-PI_C*0.5)));
    s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dql, vcast_vd_d(-PI_C*0.5            )));
    s = ddadd_vd2_vd2_vd(s, vmul_vd_vd_vd(vadd_vd_vd_vd(dqh, dql), vcast_vd_d(-PI_D*0.5)));
  }

  o = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(ql, vcast_vi_i(1)), vcast_vi_i(1)));
  vmask n = vand_vm_vo64_vm(o, vreinterpret_vm_vd(vcast_vd_d(-0.0)));
  s.x = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(s.x), n));
  s.y = vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(s.y), n));

  t = s;
  s = ddsqu_vd2_vd2(s);

  u = vcast_vd_d(1.01419718511083373224408e-05);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-2.59519791585924697698614e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(5.23388081915899855325186e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(-3.05033014433946488225616e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(7.14707504084242744267497e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(8.09674518280159187045078e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.000244884931879331847054404));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.000588505168743587154904506));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00145612788922812427978848));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00359208743836906619142924));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00886323944362401618113356));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0218694882853846389592078));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0539682539781298417636002));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.133333333333125941821962));

  x = ddadd_vd2_vd_vd2(vcast_vd_d(1), ddmul_vd2_vd2_vd2(ddadd_vd2_vd_vd(vcast_vd_d(0.333333333333334980164153), vmul_vd_vd_vd(u, s.x)), s));
  x = ddmul_vd2_vd2_vd2(t, x);

  x = vsel_vd2_vo_vd2_vd2(o, ddrec_vd2_vd2(x), x);

  u = vadd_vd_vd_vd(x.x, x.y);

  u = vsel_vd_vo_vd_vd(vandnot_vo_vo_vo(visinf_vo_vd(d),
          vor_vo_vo_vo(visnegzero_vo_vd(d),
                 vgt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(TRIGRANGEMAX)))),
           vcast_vd_d(-0.0), u);

  return u;
}

static INLINE CONST vdouble atan2k(vdouble y, vdouble x) {
  vdouble s, t, u;
  vint q;
  vopmask p;

  q = vsel_vi_vd_vi(x, vcast_vi_i(-2));
  x = vabs_vd_vd(x);

  q = vsel_vi_vd_vd_vi_vi(x, y, vadd_vi_vi_vi(q, vcast_vi_i(1)), q);
  p = vlt_vo_vd_vd(x, y);
  s = vsel_vd_vo_vd_vd(p, vneg_vd_vd(x), y);
  t = vmax_vd_vd_vd(x, y);

  s = vdiv_vd_vd_vd(s, t);
  t = vmul_vd_vd_vd(s, s);

  u = vcast_vd_d(-1.88796008463073496563746e-05);
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.000209850076645816976906797));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.00110611831486672482563471));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.00370026744188713119232403));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.00889896195887655491740809));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.016599329773529201970117));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0254517624932312641616861));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0337852580001353069993897));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0407629191276836500001934));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0466667150077840625632675));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0523674852303482457616113));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0587666392926673580854313));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0666573579361080525984562));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0769219538311769618355029));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.090908995008245008229153));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.111111105648261418443745));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.14285714266771329383765));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.199999999996591265594148));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.333333333333311110369124));

  t = vmla_vd_vd_vd_vd(s, vmul_vd_vd_vd(t, u), s);
  t = vmla_vd_vd_vd_vd(vcast_vd_vi(q), vcast_vd_d(M_PI/2), t);

  return t;
}

static INLINE CONST vdouble2 atan2k_u1(vdouble2 y, vdouble2 x) {
  vdouble u;
  vdouble2 s, t;
  vint q;
  vopmask p;

  q = vsel_vi_vd_vi(x.x, vcast_vi_i(-2));
  p = vlt_vo_vd_vd(x.x, vcast_vd_d(0));
  vmask b = vand_vm_vo64_vm(p, vreinterpret_vm_vd(vcast_vd_d(-0.0)));
  x.x = vreinterpret_vd_vm(vxor_vm_vm_vm(b, vreinterpret_vm_vd(x.x)));
  x.y = vreinterpret_vd_vm(vxor_vm_vm_vm(b, vreinterpret_vm_vd(x.y)));

  q = vsel_vi_vd_vd_vi_vi(x.x, y.x, vadd_vi_vi_vi(q, vcast_vi_i(1)), q);
  p = vlt_vo_vd_vd(x.x, y.x);
  s = vsel_vd2_vo_vd2_vd2(p, ddneg_vd2_vd2(x), y);
  t = vsel_vd2_vo_vd2_vd2(p, y, x);

  s = dddiv_vd2_vd2_vd2(s, t);
  t = ddsqu_vd2_vd2(s);
  t = ddnormalize_vd2_vd2(t);

  u = vcast_vd_d(1.06298484191448746607415e-05);
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.000125620649967286867384336));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.00070557664296393412389774));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.00251865614498713360352999));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.00646262899036991172313504));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.0128281333663399031014274));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.0208024799924145797902497));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.0289002344784740315686289));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.0359785005035104590853656));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.041848579703592507506027));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.0470843011653283988193763));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.0524914210588448421068719));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.0587946590969581003860434));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.0666620884778795497194182));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.0769225330296203768654095));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.0909090442773387574781907));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.111111108376896236538123));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.142857142756268568062339));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(0.199999999997977351284817));
  u = vmla_vd_vd_vd_vd(u, t.x, vcast_vd_d(-0.333333333333317605173818));

  t = ddmul_vd2_vd2_vd(t, u);
  t = ddmul_vd2_vd2_vd2(s, ddadd_vd2_vd_vd2(vcast_vd_d(1), t));
  t = ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_d_d(1.570796326794896557998982, 6.12323399573676603586882e-17), vcast_vd_vi(q)), t);

  return t;
}

static INLINE CONST vdouble visinf2_vd_vd_vd(vdouble d, vdouble m) {
  return vreinterpret_vd_vm(vand_vm_vo64_vm(visinf_vo_vd(d), vor_vm_vm_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(m))));
}

EXPORT CONST vdouble xatan2(vdouble y, vdouble x) {
  vdouble r = atan2k(vabs_vd_vd(y), x);

  r = vmulsign_vd_vd_vd(r, x);
  r = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), veq_vo_vd_vd(x, vcast_vd_d(0))), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), visinf2_vd_vd_vd(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/2), x))), r);
  r = vsel_vd_vo_vd_vd(visinf_vo_vd(y), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), visinf2_vd_vd_vd(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/4), x))), r);
  r = vsel_vd_vo_vd_vd(veq_vo_vd_vd(y, vcast_vd_d(0.0)), vreinterpret_vd_vm(vand_vm_vo64_vm(vsignbit_vo_vd(x), vreinterpret_vm_vd(vcast_vd_d(M_PI)))), r);

  r = vreinterpret_vd_vm(vor_vm_vo64_vm(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), vreinterpret_vm_vd(vmulsign_vd_vd_vd(r, y))));
  return r;
}

EXPORT CONST vdouble xatan2_u1(vdouble y, vdouble x) {
  vopmask o = vlt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(5.5626846462680083984e-309)); // nexttoward((1.0 / DBL_MAX), 1)
  x = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(x, vcast_vd_d(1ULL << 53)), x);
  y = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(y, vcast_vd_d(1ULL << 53)), y);

  vdouble2 d = atan2k_u1(vcast_vd2_vd_vd(vabs_vd_vd(y), vcast_vd_d(0)), vcast_vd2_vd_vd(x, vcast_vd_d(0)));
  vdouble r = vadd_vd_vd_vd(d.x, d.y);

  r = vmulsign_vd_vd_vd(r, x);
  r = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), veq_vo_vd_vd(x, vcast_vd_d(0))), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), visinf2_vd_vd_vd(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/2), x))), r);
  r = vsel_vd_vo_vd_vd(visinf_vo_vd(y), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), visinf2_vd_vd_vd(x, vmulsign_vd_vd_vd(vcast_vd_d(M_PI/4), x))), r);
  r = vsel_vd_vo_vd_vd(veq_vo_vd_vd(y, vcast_vd_d(0.0)), vreinterpret_vd_vm(vand_vm_vo64_vm(vsignbit_vo_vd(x), vreinterpret_vm_vd(vcast_vd_d(M_PI)))), r);

  r = vreinterpret_vd_vm(vor_vm_vo64_vm(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), vreinterpret_vm_vd(vmulsign_vd_vd_vd(r, y))));
  return r;
}

EXPORT CONST vdouble xasin(vdouble d) {
  vopmask o = vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(0.707));
  vdouble x2 = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, d), vmul_vd_vd_vd(vsub_vd_vd_vd(vcast_vd_d(1), vabs_vd_vd(d)), vcast_vd_d(0.5)));
  vdouble x = vsel_vd_vo_vd_vd(o, vabs_vd_vd(d), vsqrt_vd_vd(x2)), u;

  u = vcast_vd_d(+0.1093739490681073789e+1);
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.4285569429699107147e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.7994496223397078438e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.9235122556732529020e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.7378804179755443116e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.4284050251387537145e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1880302070400576397e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.6197598013399243655e+0));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1685313813037700725e+0));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.2364585657055901999e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1465340362631515313e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1098373140913713568e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1401416641102466373e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1734964278032805410e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.2237229790045894284e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.3038194033558650267e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.4464285721743240648e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.7499999999927489669e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1666666666666695718e+0));
  u = vmla_vd_vd_vd_vd(vmul_vd_vd_vd(u, x), x2, x);

  vdouble r = vsel_vd_vo_vd_vd(o, u, vmla_vd_vd_vd_vd(u, vcast_vd_d(-2), vcast_vd_d(M_PI/2)));
  return vmulsign_vd_vd_vd(r, d);
}

EXPORT CONST vdouble xasin_u1(vdouble d) {
  vopmask o = vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(0.707));
  vdouble x2 = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, d), vmul_vd_vd_vd(vsub_vd_vd_vd(vcast_vd_d(1), vabs_vd_vd(d)), vcast_vd_d(0.5))), u;
  vdouble2 x = vsel_vd2_vo_vd2_vd2(o, vcast_vd2_vd_vd(vabs_vd_vd(d), vcast_vd_d(0)), ddsqrt_vd2_vd(x2));
  x = vsel_vd2_vo_vd2_vd2(veq_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(1.0)), vcast_vd2_d_d(0, 0), x);

  u = vcast_vd_d(+0.1093739490681073789e+1);
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.4285569429699107147e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.7994496223397078438e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.9235122556732529020e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.7378804179755443116e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.4284050251387537145e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1880302070400576397e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.6197598013399243655e+0));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1685313813037700725e+0));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.2364585657055901999e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1465340362631515313e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1098373140913713568e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1401416641102466373e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1734964278032805410e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.2237229790045894284e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.3038194033558650267e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.4464285721743240648e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.7499999999927489669e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1666666666666695718e+0));
  u = vmul_vd_vd_vd(vmul_vd_vd_vd(u, x2), x.x);

  vdouble2 y = ddsub_vd2_vd2_vd(ddsub_vd2_vd2_vd2(vcast_vd2_d_d(3.141592653589793116/4, 1.2246467991473532072e-16/4), x), u);

  vdouble r = vsel_vd_vo_vd_vd(o, vadd_vd_vd_vd(u, x.x),
             vmul_vd_vd_vd(vadd_vd_vd_vd(y.x, y.y), vcast_vd_d(2)));
  return vmulsign_vd_vd_vd(r, d);
}

EXPORT CONST vdouble xacos(vdouble d) {
  vdouble x2 = vmul_vd_vd_vd(vsub_vd_vd_vd(vcast_vd_d(1), vabs_vd_vd(d)), vcast_vd_d(0.5));
  vdouble x = vsqrt_vd_vd(x2), u;

  u = vcast_vd_d(+0.1093739490681073789e+1);
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.4285569429699107147e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.7994496223397078438e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.9235122556732529020e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.7378804179755443116e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.4284050251387537145e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1880302070400576397e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.6197598013399243655e+0));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1685313813037700725e+0));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.2364585657055901999e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1465340362631515313e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1098373140913713568e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1401416641102466373e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1734964278032805410e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.2237229790045894284e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.3038194033558650267e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.4464285721743240648e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.7499999999927489669e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1666666666666695718e+0));
  u = vmla_vd_vd_vd_vd(vmul_vd_vd_vd(u, x), x2, x);

  vdouble r = vmul_vd_vd_vd(vcast_vd_d(2), u);

  return vsel_vd_vo_vd_vd(vsignbit_vo_vd(d), vsub_vd_vd_vd(vcast_vd_d(M_PI), r), r);
}

EXPORT CONST vdouble xacos_u1(vdouble d) {
  vopmask o = vlt_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(0.707));
  vdouble x2 = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, d), vmul_vd_vd_vd(vsub_vd_vd_vd(vcast_vd_d(1), vabs_vd_vd(d)), vcast_vd_d(0.5))), u;
  vdouble2 x = vsel_vd2_vo_vd2_vd2(o, vcast_vd2_vd_vd(vabs_vd_vd(d), vcast_vd_d(0)), ddsqrt_vd2_vd(x2));
  x = vsel_vd2_vo_vd2_vd2(veq_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(1.0)), vcast_vd2_d_d(0, 0), x);

  u = vcast_vd_d(+0.1093739490681073789e+1);
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.4285569429699107147e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.7994496223397078438e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.9235122556732529020e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.7378804179755443116e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.4284050251387537145e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1880302070400576397e+1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.6197598013399243655e+0));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1685313813037700725e+0));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(-0.2364585657055901999e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1465340362631515313e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1098373140913713568e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1401416641102466373e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1734964278032805410e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.2237229790045894284e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.3038194033558650267e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.4464285721743240648e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.7499999999927489669e-1));
  u = vmla_vd_vd_vd_vd(u, x2, vcast_vd_d(+0.1666666666666695718e+0));
  u = vmul_vd_vd_vd(vmul_vd_vd_vd(u, x.x), x2);

  vdouble2 y = ddsub_vd2_vd2_vd2(vcast_vd2_d_d(3.141592653589793116/2, 1.2246467991473532072e-16/2),
         ddadd_vd2_vd_vd(vmulsign_vd_vd_vd(x.x, d), vmulsign_vd_vd_vd(u, d)));
  x = ddadd_vd2_vd2_vd(x, u);

  vdouble r = vsel_vd_vo_vd_vd(o, vadd_vd_vd_vd(y.x, y.y),
             vmul_vd_vd_vd(vadd_vd_vd_vd(x.x, x.y), vcast_vd_d(2)));

  return vsel_vd_vo_vd_vd(vandnot_vo_vo_vo(o, vsignbit_vo_vd(d)), vsub_vd_vd_vd(vcast_vd_d(M_PI), r), r);
}

EXPORT CONST vdouble xatan_u1(vdouble d) {
  vdouble2 d2 = atan2k_u1(vcast_vd2_vd_vd(vabs_vd_vd(d), vcast_vd_d(0)), vcast_vd2_d_d(1, 0));
  vdouble r = vadd_vd_vd_vd(d2.x, d2.y);
  r = vsel_vd_vo_vd_vd(visinf_vo_vd(d), vcast_vd_d(1.570796326794896557998982), r);
  return vmulsign_vd_vd_vd(r, d);
}

EXPORT CONST vdouble xatan(vdouble s) {
  vdouble t, u;
  vint q;

  q = vsel_vi_vd_vi(s, vcast_vi_i(2));
  s = vabs_vd_vd(s);

  q = vsel_vi_vd_vd_vi_vi(vcast_vd_d(1), s, vadd_vi_vi_vi(q, vcast_vi_i(1)), q);
  s = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(vcast_vd_d(1), s), vrec_vd_vd(s), s);

  t = vmul_vd_vd_vd(s, s);

  u = vcast_vd_d(-1.88796008463073496563746e-05);
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.000209850076645816976906797));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.00110611831486672482563471));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.00370026744188713119232403));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.00889896195887655491740809));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.016599329773529201970117));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0254517624932312641616861));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0337852580001353069993897));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0407629191276836500001934));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0466667150077840625632675));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0523674852303482457616113));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0587666392926673580854313));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.0666573579361080525984562));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.0769219538311769618355029));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.090908995008245008229153));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.111111105648261418443745));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.14285714266771329383765));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(0.199999999996591265594148));
  u = vmla_vd_vd_vd_vd(u, t, vcast_vd_d(-0.333333333333311110369124));

  t = vmla_vd_vd_vd_vd(s, vmul_vd_vd_vd(t, u), s);

  t = vsel_vd_vo_vd_vd(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(1)), vcast_vi_i(1))), vsub_vd_vd_vd(vcast_vd_d(M_PI/2), t), t);
  t = vreinterpret_vd_vm(vxor_vm_vm_vm(vand_vm_vo64_vm(vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(q, vcast_vi_i(2)), vcast_vi_i(2))), vreinterpret_vm_vd(vcast_vd_d(-0.0))), vreinterpret_vm_vd(t)));

  return t;
}

EXPORT CONST vdouble xlog(vdouble d) {
  vdouble x, x2;
  vdouble t, m;

#ifndef ENABLE_AVX512F
  vopmask o = vlt_vo_vd_vd(d, vcast_vd_d(DBL_MIN));
  d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, vcast_vd_d((double)(1LL << 32) * (double)(1LL << 32))), d);
  vint e = vilogb2k_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
  m = vldexp3_vd_vd_vi(d, vneg_vi_vi(e));
  e = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vsub_vi_vi_vi(e, vcast_vi_i(64)), e);
#else
  vdouble e = vgetexp_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
  e = vsel_vd_vo_vd_vd(vispinf_vo_vd(e), vcast_vd_d(1024.0), e);
  m = vgetmant_vd_vd(d);
#endif

  x = vdiv_vd_vd_vd(vadd_vd_vd_vd(vcast_vd_d(-1), m), vadd_vd_vd_vd(vcast_vd_d(1), m));
  x2 = vmul_vd_vd_vd(x, x);

  t = vcast_vd_d(0.153487338491425068243146);
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.152519917006351951593857));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.181863266251982985677316));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.222221366518767365905163));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.285714294746548025383248));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.399999999950799600689777));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.6666666666667778740063));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(2));

#ifndef ENABLE_AVX512F
  x = vmla_vd_vd_vd_vd(x, t, vmul_vd_vd_vd(vcast_vd_d(0.693147180559945286226764), vcast_vd_vi(e)));

  x = vsel_vd_vo_vd_vd(vispinf_vo_vd(d), vcast_vd_d(INFINITY), x);
  x = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vlt_vo_vd_vd(d, vcast_vd_d(0)), visnan_vo_vd(d)), vcast_vd_d(NAN), x);
  x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0)), vcast_vd_d(-INFINITY), x);
#else
  x = vmla_vd_vd_vd_vd(x, t, vmul_vd_vd_vd(vcast_vd_d(0.693147180559945286226764), e));
  x = vfixup_vd_vd_vd_vi2_i(x, d, vcast_vi2_i((5 << (5*4))), 0);
#endif

  return x;
}

EXPORT CONST vdouble xexp(vdouble d) {
  vdouble u = vrint_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(R_LN2))), s;
  vint q = vrint_vi_vd(u);

  s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-L2U), d);
  s = vmla_vd_vd_vd_vd(u, vcast_vd_d(-L2L), s);

#ifdef ENABLE_FMA_DP
  u = vcast_vd_d(+0.2081276378237164457e-8);
  u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(+0.2511210703042288022e-7));
  u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(+0.2755762628169491192e-6));
  u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(+0.2755723402025388239e-5));
  u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(+0.2480158687479686264e-4));
  u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(+0.1984126989855865850e-3));
  u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(+0.1388888888914497797e-2));
  u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(+0.8333333333314938210e-2));
  u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(+0.4166666666666602598e-1));
  u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(+0.1666666666666669072e+0));
  u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(+0.5000000000000000000e+0));
  u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(+0.1000000000000000000e+1));
  u = vfma_vd_vd_vd_vd(u, s, vcast_vd_d(+0.1000000000000000000e+1));
#else
  u = vcast_vd_d(2.08860621107283687536341e-09);
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.51112930892876518610661e-08));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.75573911234900471893338e-07));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.75572362911928827629423e-06));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(2.4801587159235472998791e-05));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.000198412698960509205564975));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00138888888889774492207962));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.00833333333331652721664984));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.0416666666666665047591422));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.166666666666666851703837));
  u = vmla_vd_vd_vd_vd(u, s, vcast_vd_d(0.5));

  u = vadd_vd_vd_vd(vcast_vd_d(1), vmla_vd_vd_vd_vd(vmul_vd_vd_vd(s, s), u, s));
#endif

  u = vldexp2_vd_vd_vi(u, q);

  u = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(d, vcast_vd_d(709.78271114955742909217217426)), vcast_vd_d(INFINITY), u);
  u = vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(d, vcast_vd_d(-1000)), vreinterpret_vm_vd(u)));

  return u;
}

static INLINE CONST vdouble2 logk(vdouble d) {
  vdouble2 x, x2;
  vdouble t, m;

#ifndef ENABLE_AVX512F
  vopmask o = vlt_vo_vd_vd(d, vcast_vd_d(DBL_MIN));
  d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, vcast_vd_d((double)(1LL << 32) * (double)(1LL << 32))), d);
  vint e = vilogb2k_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
  m = vldexp3_vd_vd_vi(d, vneg_vi_vi(e));
  e = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vsub_vi_vi_vi(e, vcast_vi_i(64)), e);
#else
  vdouble e = vgetexp_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
  e = vsel_vd_vo_vd_vd(vispinf_vo_vd(e), vcast_vd_d(1024.0), e);
  m = vgetmant_vd_vd(d);
#endif

  x = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd_vd(vcast_vd_d(-1), m), ddadd2_vd2_vd_vd(vcast_vd_d(1), m));
  x2 = ddsqu_vd2_vd2(x);

  t = vcast_vd_d(0.116255524079935043668677);
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.103239680901072952701192));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.117754809412463995466069));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.13332981086846273921509));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.153846227114512262845736));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.181818180850050775676507));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.222222222230083560345903));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.285714285714249172087875));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.400000000000000077715612));
  vdouble2 c = vcast_vd2_d_d(0.666666666666666629659233, 3.80554962542412056336616e-17);

#ifndef ENABLE_AVX512F
  return ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_d_d(0.693147180559945286226764, 2.319046813846299558417771e-17), vcast_vd_vi(e)),
          ddadd2_vd2_vd2_vd2(ddscale_vd2_vd2_vd(x, vcast_vd_d(2)),
                 ddmul_vd2_vd2_vd2(ddmul_vd2_vd2_vd2(x2, x),
                 ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(x2, t), c))));
#else
  return ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_vd_vd(vcast_vd_d(0.693147180559945286226764), vcast_vd_d(2.319046813846299558417771e-17)), e),
          ddadd2_vd2_vd2_vd2(ddscale_vd2_vd2_vd(x, vcast_vd_d(2)),
                 ddmul_vd2_vd2_vd2(ddmul_vd2_vd2_vd2(x2, x),
                 ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(x2, t), c))));
#endif
}

EXPORT CONST vdouble xlog_u1(vdouble d) {
  vdouble2 x;
  vdouble t, m, x2;

#ifndef ENABLE_AVX512F
  vopmask o = vlt_vo_vd_vd(d, vcast_vd_d(DBL_MIN));
  d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, vcast_vd_d((double)(1LL << 32) * (double)(1LL << 32))), d);
  vint e = vilogb2k_vi_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
  m = vldexp3_vd_vd_vi(d, vneg_vi_vi(e));
  e = vsel_vi_vo_vi_vi(vcast_vo32_vo64(o), vsub_vi_vi_vi(e, vcast_vi_i(64)), e);
#else
  vdouble e = vgetexp_vd_vd(vmul_vd_vd_vd(d, vcast_vd_d(1.0/0.75)));
  e = vsel_vd_vo_vd_vd(vispinf_vo_vd(e), vcast_vd_d(1024.0), e);
  m = vgetmant_vd_vd(d);
#endif

  x = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd_vd(vcast_vd_d(-1), m), ddadd2_vd2_vd_vd(vcast_vd_d(1), m));
  x2 = vmul_vd_vd_vd(x.x, x.x);

  t = vcast_vd_d(0.1532076988502701353e+0);
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.1525629051003428716e+0));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.1818605932937785996e+0));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.2222214519839380009e+0));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.2857142932794299317e+0));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.3999999999635251990e+0));
  t = vmla_vd_vd_vd_vd(t, x2, vcast_vd_d(0.6666666666667333541e+0));

#ifndef ENABLE_AVX512F
  vdouble2 s = ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_d_d(0.693147180559945286226764, 2.319046813846299558417771e-17), vcast_vd_vi(e)),
          ddadd2_vd2_vd2_vd(ddscale_vd2_vd2_vd(x, vcast_vd_d(2)), vmul_vd_vd_vd(vmul_vd_vd_vd(x2, x.x), t)));
#else
  vdouble2 s = ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_d_d(0.693147180559945286226764, 2.319046813846299558417771e-17), e),
          ddadd2_vd2_vd2_vd(ddscale_vd2_vd2_vd(x, vcast_vd_d(2)), vmul_vd_vd_vd(vmul_vd_vd_vd(x2, x.x), t)));
#endif

  vdouble r = vadd_vd_vd_vd(s.x, s.y);

#ifndef ENABLE_AVX512F
  r = vsel_vd_vo_vd_vd(vispinf_vo_vd(d), vcast_vd_d(INFINITY), r);
  r = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vlt_vo_vd_vd(d, vcast_vd_d(0)), visnan_vo_vd(d)), vcast_vd_d(NAN), r);
  r = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0)), vcast_vd_d(-INFINITY), r);
#else
  r = vfixup_vd_vd_vd_vi2_i(r, d, vcast_vi2_i((4 << (2*4)) | (3 << (4*4)) | (5 << (5*4)) | (2 << (6*4))), 0);
#endif

  return r;
}

static INLINE CONST vdouble expk(vdouble2 d) {
  vdouble u = vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(R_LN2));
  vdouble dq = vrint_vd_vd(u);
  vint q = vrint_vi_vd(dq);
  vdouble2 s, t;

  s = ddadd2_vd2_vd2_vd(d, vmul_vd_vd_vd(dq, vcast_vd_d(-L2U)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dq, vcast_vd_d(-L2L)));

  s = ddnormalize_vd2_vd2(s);

  u = vcast_vd_d(2.51069683420950419527139e-08);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.76286166770270649116855e-07));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.75572496725023574143864e-06));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(2.48014973989819794114153e-05));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.000198412698809069797676111));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0013888888939977128960529));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.00833333333332371417601081));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.0416666666665409524128449));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.166666666666666740681535));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(0.500000000000000999200722));

  t = ddadd_vd2_vd2_vd2(s, ddmul_vd2_vd2_vd(ddsqu_vd2_vd2(s), u));

  t = ddadd_vd2_vd_vd2(vcast_vd_d(1), t);
  u = vadd_vd_vd_vd(t.x, t.y);
  u = vldexp2_vd_vd_vi(u, q);

  u = vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(d.x, vcast_vd_d(-1000)), vreinterpret_vm_vd(u)));

  return u;
}

EXPORT CONST vdouble xpow(vdouble x, vdouble y) {
#if 1
  vopmask yisint = visint_vo_vd(y);
  vopmask yisodd = vand_vo_vo_vo(visodd_vo_vd(y), yisint);

  vdouble2 d = ddmul_vd2_vd2_vd(logk(vabs_vd_vd(x)), y);
  vdouble result = expk(d);
  result = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(d.x, vcast_vd_d(709.78271114955742909217217426)), vcast_vd_d(INFINITY), result);

  result = vmul_vd_vd_vd(result,
       vsel_vd_vo_vd_vd(vgt_vo_vd_vd(x, vcast_vd_d(0)),
            vcast_vd_d(1),
            vsel_vd_vo_vd_vd(yisint, vsel_vd_vo_vd_vd(yisodd, vcast_vd_d(-1.0), vcast_vd_d(1)), vcast_vd_d(NAN))));

  vdouble efx = vmulsign_vd_vd_vd(vsub_vd_vd_vd(vabs_vd_vd(x), vcast_vd_d(1)), y);

  result = vsel_vd_vo_vd_vd(visinf_vo_vd(y),
          vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(efx, vcast_vd_d(0.0)),
                  vreinterpret_vm_vd(vsel_vd_vo_vd_vd(veq_vo_vd_vd(efx, vcast_vd_d(0.0)),
                              vcast_vd_d(1.0),
                              vcast_vd_d(INFINITY))))),
          result);

  result = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), veq_vo_vd_vd(x, vcast_vd_d(0.0))),
          vmul_vd_vd_vd(vsel_vd_vo_vd_vd(yisodd, vsign_vd_vd(x), vcast_vd_d(1.0)),
            vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(vsel_vd_vo_vd_vd(veq_vo_vd_vd(x, vcast_vd_d(0.0)), vneg_vd_vd(y), y), vcast_vd_d(0.0)),
                    vreinterpret_vm_vd(vcast_vd_d(INFINITY))))),
          result);

  result = vreinterpret_vd_vm(vor_vm_vo64_vm(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), vreinterpret_vm_vd(result)));

  result = vsel_vd_vo_vd_vd(vor_vo_vo_vo(veq_vo_vd_vd(y, vcast_vd_d(0)), veq_vo_vd_vd(x, vcast_vd_d(1))), vcast_vd_d(1), result);

  return result;
#else
  return expk(ddmul_vd2_vd2_vd(logk(x), y));
#endif
}

static INLINE CONST vdouble2 expk2(vdouble2 d) {
  vdouble u = vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(R_LN2));
  vdouble dq = vrint_vd_vd(u);
  vint q = vrint_vi_vd(dq);
  vdouble2 s, t;

  s = ddadd2_vd2_vd2_vd(d, vmul_vd_vd_vd(dq, vcast_vd_d(-L2U)));
  s = ddadd2_vd2_vd2_vd(s, vmul_vd_vd_vd(dq, vcast_vd_d(-L2L)));

  u = vcast_vd_d(+0.1602472219709932072e-9);
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(+0.2092255183563157007e-8));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(+0.2505230023782644465e-7));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(+0.2755724800902135303e-6));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(+0.2755731892386044373e-5));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(+0.2480158735605815065e-4));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(+0.1984126984148071858e-3));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(+0.1388888888886763255e-2));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(+0.8333333333333347095e-2));
  u = vmla_vd_vd_vd_vd(u, s.x, vcast_vd_d(+0.4166666666666669905e-1));

  t = ddadd2_vd2_vd2_vd(ddmul_vd2_vd2_vd(s, u), vcast_vd_d(+0.1666666666666666574e+0));
  t = ddadd2_vd2_vd2_vd(ddmul_vd2_vd2_vd2(s, t), vcast_vd_d(0.5));
  t = ddadd2_vd2_vd2_vd2(s, ddmul_vd2_vd2_vd2(ddsqu_vd2_vd2(s), t));

  t = ddadd_vd2_vd_vd2(vcast_vd_d(1), t);

  t.x = vldexp2_vd_vd_vi(t.x, q);
  t.y = vldexp2_vd_vd_vi(t.y, q);

  t.x = vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(d.x, vcast_vd_d(-1000)), vreinterpret_vm_vd(t.x)));
  t.y = vreinterpret_vd_vm(vandnot_vm_vo64_vm(vlt_vo_vd_vd(d.x, vcast_vd_d(-1000)), vreinterpret_vm_vd(t.y)));

  return t;
}

EXPORT CONST vdouble xsinh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = expk2(vcast_vd2_vd_vd(y, vcast_vd_d(0)));
  d = ddsub_vd2_vd2_vd2(d, ddrec_vd2_vd2(d));
  y = vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(0.5));

  y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(710)), visnan_vo_vd(y)), vcast_vd_d(INFINITY), y);
  y = vmulsign_vd_vd_vd(y, x);
  y = vreinterpret_vd_vm(vor_vm_vo64_vm(visnan_vo_vd(x), vreinterpret_vm_vd(y)));

  return y;
}

EXPORT CONST vdouble xcosh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = expk2(vcast_vd2_vd_vd(y, vcast_vd_d(0)));
  d = ddadd_vd2_vd2_vd2(d, ddrec_vd2_vd2(d));
  y = vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(0.5));

  y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(710)), visnan_vo_vd(y)), vcast_vd_d(INFINITY), y);
  y = vreinterpret_vd_vm(vor_vm_vo64_vm(visnan_vo_vd(x), vreinterpret_vm_vd(y)));

  return y;
}

EXPORT CONST vdouble xtanh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = expk2(vcast_vd2_vd_vd(y, vcast_vd_d(0)));
  vdouble2 e = ddrec_vd2_vd2(d);
  d = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd2_vd2(d, ddneg_vd2_vd2(e)), ddadd2_vd2_vd2_vd2(d, e));
  y = vadd_vd_vd_vd(d.x, d.y);

  y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(18.714973875)), visnan_vo_vd(y)), vcast_vd_d(1.0), y);
  y = vmulsign_vd_vd_vd(y, x);
  y = vreinterpret_vd_vm(vor_vm_vo64_vm(visnan_vo_vd(x), vreinterpret_vm_vd(y)));

  return y;
}

static INLINE CONST vdouble2 logk2(vdouble2 d) {
  vdouble2 x, x2, m;
  vdouble t;
  vint e;

  e = vilogbk_vi_vd(vmul_vd_vd_vd(d.x, vcast_vd_d(1.0/0.75)));

  m.x = vldexp2_vd_vd_vi(d.x, vneg_vi_vi(e));
  m.y = vldexp2_vd_vd_vi(d.y, vneg_vi_vi(e));

  x = dddiv_vd2_vd2_vd2(ddadd2_vd2_vd2_vd(m, vcast_vd_d(-1)), ddadd2_vd2_vd2_vd(m, vcast_vd_d(1)));
  x2 = ddsqu_vd2_vd2(x);

  t = vcast_vd_d(0.13860436390467167910856);
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.131699838841615374240845));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.153914168346271945653214));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.181816523941564611721589));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.22222224632662035403996));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.285714285511134091777308));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.400000000000914013309483));
  t = vmla_vd_vd_vd_vd(t, x2.x, vcast_vd_d(0.666666666666664853302393));

  return ddadd2_vd2_vd2_vd2(ddmul_vd2_vd2_vd(vcast_vd2_d_d(0.693147180559945286226764, 2.319046813846299558417771e-17),
               vcast_vd_vi(e)),
          ddadd2_vd2_vd2_vd2(ddscale_vd2_vd2_vd(x, vcast_vd_d(2)), ddmul_vd2_vd2_vd(ddmul_vd2_vd2_vd2(x2, x), t)));
}

EXPORT CONST vdouble xasinh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vopmask o = vgt_vo_vd_vd(y, vcast_vd_d(1));
  vdouble2 d;

  d = vsel_vd2_vo_vd2_vd2(o, ddrec_vd2_vd(x), vcast_vd2_vd_vd(y, vcast_vd_d(0)));
  d = ddsqrt_vd2_vd2(ddadd2_vd2_vd2_vd(ddsqu_vd2_vd2(d), vcast_vd_d(1)));
  d = vsel_vd2_vo_vd2_vd2(o, ddmul_vd2_vd2_vd(d, y), d);

  d = logk2(ddnormalize_vd2_vd2(ddadd2_vd2_vd2_vd(d, x)));
  y = vadd_vd_vd_vd(d.x, d.y);

  y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(SQRT_DBL_MAX)),
            visnan_vo_vd(y)),
           vmulsign_vd_vd_vd(vcast_vd_d(INFINITY), x), y);

  y = vreinterpret_vd_vm(vor_vm_vo64_vm(visnan_vo_vd(x), vreinterpret_vm_vd(y)));
  y = vsel_vd_vo_vd_vd(visnegzero_vo_vd(x), vcast_vd_d(-0.0), y);

  return y;
}

EXPORT CONST vdouble xacosh(vdouble x) {
  vdouble2 d = logk2(ddadd2_vd2_vd2_vd(ddmul_vd2_vd2_vd2(ddsqrt_vd2_vd2(ddadd2_vd2_vd_vd(x, vcast_vd_d(1))), ddsqrt_vd2_vd2(ddadd2_vd2_vd_vd(x, vcast_vd_d(-1)))), x));
  vdouble y = vadd_vd_vd_vd(d.x, d.y);

  y = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vgt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(SQRT_DBL_MAX)),
            visnan_vo_vd(y)),
           vcast_vd_d(INFINITY), y);
  y = vreinterpret_vd_vm(vandnot_vm_vo64_vm(veq_vo_vd_vd(x, vcast_vd_d(1.0)), vreinterpret_vm_vd(y)));

  y = vreinterpret_vd_vm(vor_vm_vo64_vm(vlt_vo_vd_vd(x, vcast_vd_d(1.0)), vreinterpret_vm_vd(y)));
  y = vreinterpret_vd_vm(vor_vm_vo64_vm(visnan_vo_vd(x), vreinterpret_vm_vd(y)));

  return y;
}

EXPORT CONST vdouble xatanh(vdouble x) {
  vdouble y = vabs_vd_vd(x);
  vdouble2 d = logk2(dddiv_vd2_vd2_vd2(ddadd2_vd2_vd_vd(vcast_vd_d(1), y), ddadd2_vd2_vd_vd(vcast_vd_d(1), vneg_vd_vd(y))));
  y = vreinterpret_vd_vm(vor_vm_vo64_vm(vgt_vo_vd_vd(y, vcast_vd_d(1.0)), vreinterpret_vm_vd(vsel_vd_vo_vd_vd(veq_vo_vd_vd(y, vcast_vd_d(1.0)), vcast_vd_d(INFINITY), vmul_vd_vd_vd(vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(0.5))))));

  y = vmulsign_vd_vd_vd(y, x);
  y = vreinterpret_vd_vm(vor_vm_vo64_vm(vor_vo_vo_vo(visinf_vo_vd(x), visnan_vo_vd(y)), vreinterpret_vm_vd(y)));
  y = vreinterpret_vd_vm(vor_vm_vo64_vm(visnan_vo_vd(x), vreinterpret_vm_vd(y)));

  return y;
}

EXPORT CONST vdouble xcbrt(vdouble d) {
  vdouble x, y, q = vcast_vd_d(1.0);
  vint e, qu, re;
  vdouble t;

#ifdef ENABLE_AVX512F
  vdouble s = d;
#endif
  e = vadd_vi_vi_vi(vilogbk_vi_vd(vabs_vd_vd(d)), vcast_vi_i(1));
  d = vldexp2_vd_vd_vi(d, vneg_vi_vi(e));

  t = vadd_vd_vd_vd(vcast_vd_vi(e), vcast_vd_d(6144));
  qu = vtruncate_vi_vd(vmul_vd_vd_vd(t, vcast_vd_d(1.0/3.0)));
  re = vtruncate_vi_vd(vsub_vd_vd_vd(t, vmul_vd_vd_vd(vcast_vd_vi(qu), vcast_vd_d(3))));

  q = vsel_vd_vo_vd_vd(vcast_vo64_vo32(veq_vo_vi_vi(re, vcast_vi_i(1))), vcast_vd_d(1.2599210498948731647672106), q);
  q = vsel_vd_vo_vd_vd(vcast_vo64_vo32(veq_vo_vi_vi(re, vcast_vi_i(2))), vcast_vd_d(1.5874010519681994747517056), q);
  q = vldexp2_vd_vd_vi(q, vsub_vi_vi_vi(qu, vcast_vi_i(2048)));

  q = vmulsign_vd_vd_vd(q, d);

  d = vabs_vd_vd(d);

  x = vcast_vd_d(-0.640245898480692909870982);
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(2.96155103020039511818595));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(-5.73353060922947843636166));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(6.03990368989458747961407));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(-3.85841935510444988821632));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(2.2307275302496609725722));

  y = vmul_vd_vd_vd(x, x); y = vmul_vd_vd_vd(y, y); x = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vmlapn_vd_vd_vd_vd(d, y, x), vcast_vd_d(1.0 / 3.0)));
  y = vmul_vd_vd_vd(vmul_vd_vd_vd(d, x), x);
  y = vmul_vd_vd_vd(vsub_vd_vd_vd(y, vmul_vd_vd_vd(vmul_vd_vd_vd(vcast_vd_d(2.0 / 3.0), y), vmla_vd_vd_vd_vd(y, x, vcast_vd_d(-1.0)))), q);

#ifdef ENABLE_AVX512F
  y = vsel_vd_vo_vd_vd(visinf_vo_vd(s), vmulsign_vd_vd_vd(vcast_vd_d(INFINITY), s), y);
  y = vsel_vd_vo_vd_vd(veq_vo_vd_vd(s, vcast_vd_d(0)), vmulsign_vd_vd_vd(vcast_vd_d(0), s), y);
#endif

  return y;
}

EXPORT CONST vdouble xcbrt_u1(vdouble d) {
  vdouble x, y, z, t;
  vdouble2 q2 = vcast_vd2_d_d(1, 0), u, v;
  vint e, qu, re;

#ifdef ENABLE_AVX512F
  vdouble s = d;
#endif
  e = vadd_vi_vi_vi(vilogbk_vi_vd(vabs_vd_vd(d)), vcast_vi_i(1));
  d = vldexp2_vd_vd_vi(d, vneg_vi_vi(e));

  t = vadd_vd_vd_vd(vcast_vd_vi(e), vcast_vd_d(6144));
  qu = vtruncate_vi_vd(vmul_vd_vd_vd(t, vcast_vd_d(1.0/3.0)));
  re = vtruncate_vi_vd(vsub_vd_vd_vd(t, vmul_vd_vd_vd(vcast_vd_vi(qu), vcast_vd_d(3))));

  q2 = vsel_vd2_vo_vd2_vd2(vcast_vo64_vo32(veq_vo_vi_vi(re, vcast_vi_i(1))), vcast_vd2_d_d(1.2599210498948731907, -2.5899333753005069177e-17), q2);
  q2 = vsel_vd2_vo_vd2_vd2(vcast_vo64_vo32(veq_vo_vi_vi(re, vcast_vi_i(2))), vcast_vd2_d_d(1.5874010519681995834, -1.0869008194197822986e-16), q2);

  q2.x = vmulsign_vd_vd_vd(q2.x, d); q2.y = vmulsign_vd_vd_vd(q2.y, d);
  d = vabs_vd_vd(d);

  x = vcast_vd_d(-0.640245898480692909870982);
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(2.96155103020039511818595));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(-5.73353060922947843636166));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(6.03990368989458747961407));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(-3.85841935510444988821632));
  x = vmla_vd_vd_vd_vd(x, d, vcast_vd_d(2.2307275302496609725722));

  y = vmul_vd_vd_vd(x, x); y = vmul_vd_vd_vd(y, y); x = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vmlapn_vd_vd_vd_vd(d, y, x), vcast_vd_d(1.0 / 3.0)));

  z = x;

  u = ddmul_vd2_vd_vd(x, x);
  u = ddmul_vd2_vd2_vd2(u, u);
  u = ddmul_vd2_vd2_vd(u, d);
  u = ddadd2_vd2_vd2_vd(u, vneg_vd_vd(x));
  y = vadd_vd_vd_vd(u.x, u.y);

  y = vmul_vd_vd_vd(vmul_vd_vd_vd(vcast_vd_d(-2.0 / 3.0), y), z);
  v = ddadd2_vd2_vd2_vd(ddmul_vd2_vd_vd(z, z), y);
  v = ddmul_vd2_vd2_vd(v, d);
  v = ddmul_vd2_vd2_vd2(v, q2);
  z = vldexp2_vd_vd_vi(vadd_vd_vd_vd(v.x, v.y), vsub_vi_vi_vi(qu, vcast_vi_i(2048)));

#ifndef ENABLE_AVX512F
  z = vsel_vd_vo_vd_vd(visinf_vo_vd(d), vmulsign_vd_vd_vd(vcast_vd_d(INFINITY), q2.x), z);
  z = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0)), vreinterpret_vd_vm(vsignbit_vm_vd(q2.x)), z);
#else
  z = vsel_vd_vo_vd_vd(visinf_vo_vd(s), vmulsign_vd_vd_vd(vcast_vd_d(INFINITY), s), z);
  z = vsel_vd_vo_vd_vd(veq_vo_vd_vd(s, vcast_vd_d(0)), vmulsign_vd_vd_vd(vcast_vd_d(0), s), z);
#endif

  return z;
}

EXPORT CONST vdouble xexp2(vdouble a) {
  vdouble u = expk(ddmul_vd2_vd2_vd(vcast_vd2_vd_vd(vcast_vd_d(0.69314718055994528623), vcast_vd_d(2.3190468138462995584e-17)), a));
  u = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(a, vcast_vd_d(1024)), vcast_vd_d(INFINITY), u);
  u = vreinterpret_vd_vm(vandnot_vm_vo64_vm(visminf_vo_vd(a), vreinterpret_vm_vd(u)));
  return u;
}

EXPORT CONST vdouble xexp10(vdouble a) {
  vdouble u = expk(ddmul_vd2_vd2_vd(vcast_vd2_vd_vd(vcast_vd_d(2.3025850929940459011), vcast_vd_d(-2.1707562233822493508e-16)), a));
  u = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(a, vcast_vd_d(308.254715559916743850652254)), vcast_vd_d(INFINITY), u);
  u = vreinterpret_vd_vm(vandnot_vm_vo64_vm(visminf_vo_vd(a), vreinterpret_vm_vd(u)));
  return u;
}

EXPORT CONST vdouble xexpm1(vdouble a) {
  vdouble2 d = ddadd2_vd2_vd2_vd(expk2(vcast_vd2_vd_vd(a, vcast_vd_d(0))), vcast_vd_d(-1.0));
  vdouble x = vadd_vd_vd_vd(d.x, d.y);
  x = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(a, vcast_vd_d(709.782712893383996732223)), vcast_vd_d(INFINITY), x);
  x = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(a, vcast_vd_d(-36.736800569677101399113302437)), vcast_vd_d(-1), x);
  x = vsel_vd_vo_vd_vd(visnegzero_vo_vd(a), vcast_vd_d(-0.0), x);
  return x;
}

EXPORT CONST vdouble xlog10(vdouble a) {
  vdouble2 d = ddmul_vd2_vd2_vd2(logk(a), vcast_vd2_vd_vd(vcast_vd_d(0.43429448190325176116), vcast_vd_d(6.6494347733425473126e-17)));
  vdouble x = vadd_vd_vd_vd(d.x, d.y);

#ifndef ENABLE_AVX512F
  x = vsel_vd_vo_vd_vd(vispinf_vo_vd(a), vcast_vd_d(INFINITY), x);
  x = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vlt_vo_vd_vd(a, vcast_vd_d(0)), visnan_vo_vd(a)), vcast_vd_d(NAN), x);
  x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(a, vcast_vd_d(0)), vcast_vd_d(-INFINITY), x);
#else
  x = vfixup_vd_vd_vd_vi2_i(x, a, vcast_vi2_i((4 << (2*4)) | (3 << (4*4)) | (5 << (5*4)) | (2 << (6*4))), 0);
#endif

  return x;
}

EXPORT CONST vdouble xlog1p(vdouble a) {
  vdouble2 d = logk2(ddadd2_vd2_vd_vd(a, vcast_vd_d(1)));
  vdouble x = vadd_vd_vd_vd(d.x, d.y);

  x = vsel_vd_vo_vd_vd(vgt_vo_vd_vd(a, vcast_vd_d(1e+307)), vcast_vd_d(INFINITY), x);
  x = vreinterpret_vd_vm(vor_vm_vo64_vm(vgt_vo_vd_vd(vcast_vd_d(-1.0), a), vreinterpret_vm_vd(x)));
  x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(a, vcast_vd_d(-1)), vcast_vd_d(-INFINITY), x);
  x = vsel_vd_vo_vd_vd(visnegzero_vo_vd(a), vcast_vd_d(-0.0), x);

  return x;
}

//

static INLINE CONST vint2 vcast_vi2_i_i(int i0, int i1) { return vcast_vi2_vm(vcast_vm_i_i(i0, i1)); }
static INLINE CONST vint2 vrev21_vi2_vi2(vint2 i) { return vreinterpret_vi2_vf(vrev21_vf_vf(vreinterpret_vf_vi2(i))); }

static INLINE CONST vdouble vnexttoward0(vdouble x) {
  vint2 xi2 = vreinterpret_vi2_vd(x);

  xi2 = vsub_vi2_vi2_vi2(xi2, vcast_vi2_i_i(0, 1));
  xi2 = vadd_vi2_vi2_vi2(xi2, vrev21_vi2_vi2(veq_vi2_vi2_vi2(xi2, vcast_vi2_i_i(-1, -1))));

  vdouble ret = vreinterpret_vd_vi2(xi2);
  ret = vsel_vd_vo_vd_vd(veq_vo_vd_vd(x, vcast_vd_d(0)), vcast_vd_d(0), ret);

  return ret;
}

static INLINE CONST vdouble upper2(vdouble d) {
  return vreinterpret_vd_vm(vand_vm_vm_vm(vreinterpret_vm_vd(d), vcast_vm_i_i(0xffffffff, 0xfffffffe)));
}

EXPORT CONST vdouble xfabs(vdouble x) { return vabs_vd_vd(x); }

EXPORT CONST vdouble xcopysign(vdouble x, vdouble y) { return vcopysign_vd_vd_vd(x, y); }

EXPORT CONST vdouble xfmax(vdouble x, vdouble y) {
  return vsel_vd_vo_vd_vd(visnan_vo_vd(y), x, vsel_vd_vo_vd_vd(vgt_vo_vd_vd(x, y), x, y));
}

EXPORT CONST vdouble xfmin(vdouble x, vdouble y) {
  return vsel_vd_vo_vd_vd(visnan_vo_vd(y), x, vsel_vd_vo_vd_vd(vgt_vo_vd_vd(y, x), x, y));
}

EXPORT CONST vdouble xfdim(vdouble x, vdouble y) {
  vdouble ret = vsub_vd_vd_vd(x, y);
  ret = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vlt_vo_vd_vd(ret, vcast_vd_d(0)), veq_vo_vd_vd(x, y)), vcast_vd_d(0), ret);
  return ret;
}

EXPORT CONST vdouble xtrunc(vdouble x) {
  vdouble fr = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vcast_vd_d(1LL << 31), vcast_vd_vi(vtruncate_vi_vd(vmul_vd_vd_vd(x, vcast_vd_d(1.0 / (1LL << 31)))))));
  fr = vsub_vd_vd_vd(fr, vcast_vd_vi(vtruncate_vi_vd(fr)));
  return vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), vge_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(1LL << 52))), x, vcopysign_vd_vd_vd(vsub_vd_vd_vd(x, fr), x));
}

EXPORT CONST vdouble xfloor(vdouble x) {
  vdouble fr = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vcast_vd_d(1LL << 31), vcast_vd_vi(vtruncate_vi_vd(vmul_vd_vd_vd(x, vcast_vd_d(1.0 / (1LL << 31)))))));
  fr = vsub_vd_vd_vd(fr, vcast_vd_vi(vtruncate_vi_vd(fr)));
  fr = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(fr, vcast_vd_d(0)), vadd_vd_vd_vd(fr, vcast_vd_d(1.0)), fr);
  return vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), vge_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(1LL << 52))), x, vcopysign_vd_vd_vd(vsub_vd_vd_vd(x, fr), x));
}

EXPORT CONST vdouble xceil(vdouble x) {
  vdouble fr = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vcast_vd_d(1LL << 31), vcast_vd_vi(vtruncate_vi_vd(vmul_vd_vd_vd(x, vcast_vd_d(1.0 / (1LL << 31)))))));
  fr = vsub_vd_vd_vd(fr, vcast_vd_vi(vtruncate_vi_vd(fr)));
  fr = vsel_vd_vo_vd_vd(vle_vo_vd_vd(fr, vcast_vd_d(0)), fr, vsub_vd_vd_vd(fr, vcast_vd_d(1.0)));
  return vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(x), vge_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(1LL << 52))), x, vcopysign_vd_vd_vd(vsub_vd_vd_vd(x, fr), x));
}

EXPORT CONST vdouble xround(vdouble d) {
  vdouble x = vadd_vd_vd_vd(d, vcast_vd_d(0.5));
  vdouble fr = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vcast_vd_d(1LL << 31), vcast_vd_vi(vtruncate_vi_vd(vmul_vd_vd_vd(x, vcast_vd_d(1.0 / (1LL << 31)))))));
  fr = vsub_vd_vd_vd(fr, vcast_vd_vi(vtruncate_vi_vd(fr)));
  x = vsel_vd_vo_vd_vd(vand_vo_vo_vo(vle_vo_vd_vd(x, vcast_vd_d(0)), veq_vo_vd_vd(fr, vcast_vd_d(0))), vsub_vd_vd_vd(x, vcast_vd_d(1.0)), x);
  fr = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(fr, vcast_vd_d(0)), vadd_vd_vd_vd(fr, vcast_vd_d(1.0)), fr);
  x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0.49999999999999994449)), vcast_vd_d(0), x);
  return vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(d), vge_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(1LL << 52))), d, vcopysign_vd_vd_vd(vsub_vd_vd_vd(x, fr), d));
}

EXPORT CONST vdouble xrint(vdouble d) {
  vdouble x = vadd_vd_vd_vd(d, vcast_vd_d(0.5));
  vdouble fr = vsub_vd_vd_vd(x, vmul_vd_vd_vd(vcast_vd_d(1LL << 31), vcast_vd_vi(vtruncate_vi_vd(vmul_vd_vd_vd(x, vcast_vd_d(1.0 / (1LL << 31)))))));
  vopmask isodd = vcast_vo64_vo32(veq_vo_vi_vi(vand_vi_vi_vi(vcast_vi_i(1), vtruncate_vi_vd(fr)), vcast_vi_i(1)));
  fr = vsub_vd_vd_vd(fr, vcast_vd_vi(vtruncate_vi_vd(fr)));
  fr = vsel_vd_vo_vd_vd(vor_vo_vo_vo(vlt_vo_vd_vd(fr, vcast_vd_d(0)), vand_vo_vo_vo(veq_vo_vd_vd(fr, vcast_vd_d(0)), isodd)), vadd_vd_vd_vd(fr, vcast_vd_d(1.0)), fr);
  x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0.50000000000000011102)), vcast_vd_d(0), x);
  vdouble ret = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visinf_vo_vd(d), vge_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(1LL << 52))), d, vcopysign_vd_vd_vd(vsub_vd_vd_vd(x, fr), d));
  return ret;
}

EXPORT CONST vdouble xnextafter(vdouble x, vdouble y) {
  x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(x, vcast_vd_d(0)), vmulsign_vd_vd_vd(vcast_vd_d(0), y), x);
  vint2 t, xi2 = vreinterpret_vi2_vd(x);
  vopmask c = vxor_vo_vo_vo(vsignbit_vo_vd(x), vge_vo_vd_vd(y, x));

  t = vadd_vi2_vi2_vi2(vxor_vi2_vi2_vi2(xi2, vcast_vi2_i_i(0x7fffffff, 0xffffffff)), vcast_vi2_i_i(0, 1));
  t = vadd_vi2_vi2_vi2(t, vrev21_vi2_vi2(vand_vi2_vi2_vi2(vcast_vi2_i_i(0, 1), veq_vi2_vi2_vi2(t, vcast_vi2_i_i(-1, 0)))));
  xi2 = vreinterpret_vi2_vd(vsel_vd_vo_vd_vd(c, vreinterpret_vd_vi2(t), vreinterpret_vd_vi2(xi2)));

  xi2 = vsub_vi2_vi2_vi2(xi2, vcast_vi2_vm(vand_vm_vo64_vm(vneq_vo_vd_vd(x, y), vcast_vm_i_i(0, 1))));

  xi2 = vreinterpret_vi2_vd(vsel_vd_vo_vd_vd(vneq_vo_vd_vd(x, y),
               vreinterpret_vd_vi2(vadd_vi2_vi2_vi2(xi2, vrev21_vi2_vi2(vand_vi2_vi2_vi2(vcast_vi2_i_i(0, -1), veq_vi2_vi2_vi2(xi2, vcast_vi2_i_i(0, -1)))))),
               vreinterpret_vd_vi2(xi2)));

  t = vadd_vi2_vi2_vi2(vxor_vi2_vi2_vi2(xi2, vcast_vi2_i_i(0x7fffffff, 0xffffffff)), vcast_vi2_i_i(0, 1));
  t = vadd_vi2_vi2_vi2(t, vrev21_vi2_vi2(vand_vi2_vi2_vi2(vcast_vi2_i_i(0, 1), veq_vi2_vi2_vi2(t, vcast_vi2_i_i(-1, 0)))));
  xi2 = vreinterpret_vi2_vd(vsel_vd_vo_vd_vd(c, vreinterpret_vd_vi2(t), vreinterpret_vd_vi2(xi2)));

  vdouble ret = vreinterpret_vd_vi2(xi2);

  ret = vsel_vd_vo_vd_vd(vand_vo_vo_vo(veq_vo_vd_vd(ret, vcast_vd_d(0)), vneq_vo_vd_vd(x, vcast_vd_d(0))),
       vmulsign_vd_vd_vd(vcast_vd_d(0), x), ret);

  ret = vsel_vd_vo_vd_vd(vand_vo_vo_vo(veq_vo_vd_vd(x, vcast_vd_d(0)), veq_vo_vd_vd(y, vcast_vd_d(0))), y, ret);

  ret = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), vcast_vd_d(NAN), ret);

  return ret;
}

EXPORT CONST vdouble xfrfrexp(vdouble x) {
  vdouble j = x;
  x = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(DBL_MIN)),
                       vmul_vd_vd_vd(x, vcast_vd_d((double)(1ULL << 63))),
                       x);

  vmask xm = vreinterpret_vm_vd(x);
  xm = vand_vm_vm_vm(xm, vcast_vm_i_i(~0x7ff00000, ~0));
  xm = vor_vm_vm_vm (xm, vcast_vm_i_i( 0x3fe00000,  0));

  vdouble ret = vreinterpret_vd_vm(xm);

  ret = vsel_vd_vo_vd_vd(visinf_vo_vd(x),
                         vmulsign_vd_vd_vd(vcast_vd_d(INFINITY), x),
                         ret);
  ret = vsel_vd_vo_vd_vd(veq_vo_vd_vd(x, vcast_vd_d(0.0)), x, ret);

  ret = vsel_vd_vo_vd_vd(visnan_vo_vd(j), j, ret);
  return ret;
}

EXPORT CONST vmask xexpfrexp(vdouble x) {
  vopmask isnan = vor_vo_vo_vo(visinf_vo_vd(x), visnan_vo_vd(x));
  vdouble mul = vmul_vd_vd_vd(x, vcast_vd_d(0x1p+63));
  vopmask is_denorm = vlt_vo_vd_vd(vabs_vd_vd(x), vcast_vd_d(0x1p-1022));

  x = vsel_vd_vo_vd_vd(is_denorm, mul, x);
  const vint2 m63 = vcast_vi2_i64(-63);
  const vint2 zeros = vcast_vi2_i(0);
  vint2 correct = vsel_vi2_vo_vi2_vi2(is_denorm, m63, zeros);

  vint2 ret = vreinterpret_vi2_vd(x);

#if defined(ENABLE_NEON32) || defined(ENABLE_ADVSIMD)
  ret = vsrl64_vi2_vi_52(ret);
#else
  ret = vsrl64_vi2_vi(ret, 52);
#endif
  ret = vand_vi2_vi2_vi2(ret, vcast_vi2_i64(0x7ff));
  ret = vsub64_vi2_vi2_vi2(ret, vcast_vi2_i64(0x3fe));
  ret = vadd64_vi2_vi2_vi2(ret, correct);

  ret = vsel_vi2_vo_vi2_vi2(
            veq_vo_vd_vd(x, vreinterpret_vd_vi2(zeros)),
            zeros,
            ret);
  ret = vsel_vi2_vo_vi2_vi2(isnan, zeros, ret);

  return vcast_vm_vi2(ret);
}

EXPORT CONST vdouble xfma(vdouble x, vdouble y, vdouble z) {
#ifdef ENABLE_FMA_DP
  return vmla_vd_vd_vd_vd(x, y, z);
#else
  vdouble h2 = vadd_vd_vd_vd(vmul_vd_vd_vd(x, y), z), q = vcast_vd_d(1);
  vopmask o = vlt_vo_vd_vd(vabs_vd_vd(h2), vcast_vd_d(1e-300));
  {
    const double c0 = 1ULL << 54, c1 = c0 * c0, c2 = c1 * c1;
    x = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(x, vcast_vd_d(c1)), x);
    y = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(y, vcast_vd_d(c1)), y);
    z = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(z, vcast_vd_d(c2)), z);
    q = vsel_vd_vo_vd_vd(o, vcast_vd_d(1.0 / c2), q);
  }
  o = vgt_vo_vd_vd(vabs_vd_vd(h2), vcast_vd_d(1e+300));
  {
    const double c0 = 1ULL << 54, c1 = c0 * c0, c2 = c1 * c1;
    x = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(x, vcast_vd_d(1.0 / c1)), x);
    y = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(y, vcast_vd_d(1.0 / c1)), y);
    z = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(z, vcast_vd_d(1.0 / c2)), z);
    q = vsel_vd_vo_vd_vd(o, vcast_vd_d(c2), q);
  }
  vdouble2 d = ddmul_vd2_vd_vd(x, y);
  d = ddadd2_vd2_vd2_vd(d, z);
  vdouble ret = vsel_vd_vo_vd_vd(vor_vo_vo_vo(veq_vo_vd_vd(x, vcast_vd_d(0)), veq_vo_vd_vd(y, vcast_vd_d(0))), z, vadd_vd_vd_vd(d.x, d.y));
  o = visinf_vo_vd(z);
  o = vandnot_vo_vo_vo(visinf_vo_vd(x), o);
  o = vandnot_vo_vo_vo(visnan_vo_vd(x), o);
  o = vandnot_vo_vo_vo(visinf_vo_vd(y), o);
  o = vandnot_vo_vo_vo(visnan_vo_vd(y), o);
  h2 = vsel_vd_vo_vd_vd(o, z, h2);

  o = vor_vo_vo_vo(visinf_vo_vd(h2), visnan_vo_vd(h2));

  return vsel_vd_vo_vd_vd(o, h2, vmul_vd_vd_vd(ret, q));
#endif
}

EXPORT CONST vdouble xsqrt_u05(vdouble d) {
#if 1
  return vsqrt_vd_vd(d);
#else
  vdouble q;
  vopmask o;

  d = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(d, vcast_vd_d(0)), vcast_vd_d(NAN), d);

  o = vlt_vo_vd_vd(d, vcast_vd_d(8.636168555094445E-78));
  d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, vcast_vd_d(1.157920892373162E77)), d);
  q = vsel_vd_vo_vd_vd(o, vcast_vd_d(2.9387358770557188E-39*0.5), vcast_vd_d(0.5));

  o = vgt_vo_vd_vd(d, vcast_vd_d(1.3407807929942597e+154));
  d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, vcast_vd_d(7.4583407312002070e-155)), d);
  q = vsel_vd_vo_vd_vd(o, vcast_vd_d(1.1579208923731620e+77*0.5), q);

  vdouble x = vreinterpret_vd_vi2(vsub_vi2_vi2_vi2(vcast_vi2_i_i(0x5fe6ec86, 0), vsrl_vi2_vi2_i(vreinterpret_vi2_vd(vadd_vd_vd_vd(d, vcast_vd_d(1e-320))), 1)));

  x = vmul_vd_vd_vd(x, vsub_vd_vd_vd(vcast_vd_d(1.5), vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(vcast_vd_d(0.5), d), x), x)));
  x = vmul_vd_vd_vd(x, vsub_vd_vd_vd(vcast_vd_d(1.5), vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(vcast_vd_d(0.5), d), x), x)));
  x = vmul_vd_vd_vd(x, vsub_vd_vd_vd(vcast_vd_d(1.5), vmul_vd_vd_vd(vmul_vd_vd_vd(vmul_vd_vd_vd(vcast_vd_d(0.5), d), x), x)));
  x = vmul_vd_vd_vd(x, d);

  vdouble2 d2 = ddmul_vd2_vd2_vd2(ddadd2_vd2_vd_vd2(d, ddmul_vd2_vd_vd(x, x)), ddrec_vd2_vd(x));

  x = vmul_vd_vd_vd(vadd_vd_vd_vd(d2.x, d2.y), q);

  x = vsel_vd_vo_vd_vd(vispinf_vo_vd(d), vcast_vd_d(INFINITY), x);
  x = vsel_vd_vo_vd_vd(veq_vo_vd_vd(d, vcast_vd_d(0)), d, x);

  return x;
#endif
}

EXPORT CONST vdouble xsqrt_u35(vdouble d) { return xsqrt_u05(d); }

EXPORT CONST vdouble xhypot_u05(vdouble x, vdouble y) {
  x = vabs_vd_vd(x);
  y = vabs_vd_vd(y);
  vdouble min = vmin_vd_vd_vd(x, y), n = min;
  vdouble max = vmax_vd_vd_vd(x, y), d = max;

  vopmask o = vlt_vo_vd_vd(max, vcast_vd_d(DBL_MIN));
  n = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(n, vcast_vd_d(1ULL << 54)), n);
  d = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(d, vcast_vd_d(1ULL << 54)), d);

  vdouble2 t = dddiv_vd2_vd2_vd2(vcast_vd2_vd_vd(n, vcast_vd_d(0)), vcast_vd2_vd_vd(d, vcast_vd_d(0)));
  t = ddmul_vd2_vd2_vd(ddsqrt_vd2_vd2(ddadd2_vd2_vd2_vd(ddsqu_vd2_vd2(t), vcast_vd_d(1))), max);
  vdouble ret = vadd_vd_vd_vd(t.x, t.y);
  ret = vsel_vd_vo_vd_vd(visnan_vo_vd(ret), vcast_vd_d(INFINITY), ret);
  ret = vsel_vd_vo_vd_vd(veq_vo_vd_vd(min, vcast_vd_d(0)), max, ret);
  ret = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), vcast_vd_d(NAN), ret);
  ret = vsel_vd_vo_vd_vd(vor_vo_vo_vo(veq_vo_vd_vd(x, vcast_vd_d(INFINITY)), veq_vo_vd_vd(y, vcast_vd_d(INFINITY))), vcast_vd_d(INFINITY), ret);

  return ret;
}

EXPORT CONST vdouble xhypot_u35(vdouble x, vdouble y) {
  x = vabs_vd_vd(x);
  y = vabs_vd_vd(y);
  vdouble min = vmin_vd_vd_vd(x, y);
  vdouble max = vmax_vd_vd_vd(x, y);

  vdouble t = vdiv_vd_vd_vd(min, max);
  vdouble ret = vmul_vd_vd_vd(max, vsqrt_vd_vd(vmla_vd_vd_vd_vd(t, t, vcast_vd_d(1))));
  ret = vsel_vd_vo_vd_vd(veq_vo_vd_vd(min, vcast_vd_d(0)), max, ret);
  ret = vsel_vd_vo_vd_vd(vor_vo_vo_vo(visnan_vo_vd(x), visnan_vo_vd(y)), vcast_vd_d(NAN), ret);
  ret = vsel_vd_vo_vd_vd(vor_vo_vo_vo(veq_vo_vd_vd(x, vcast_vd_d(INFINITY)), veq_vo_vd_vd(y, vcast_vd_d(INFINITY))), vcast_vd_d(INFINITY), ret);

  return ret;
}

/* TODO AArch64: potential optimization by using `vfmad_lane_f64` */
EXPORT CONST vdouble xfmod(vdouble x, vdouble y) {
  vdouble nu = vabs_vd_vd(x), de = vabs_vd_vd(y), s = vcast_vd_d(1);
  vopmask o = vlt_vo_vd_vd(de, vcast_vd_d(DBL_MIN));
  nu = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(nu, vcast_vd_d(1ULL << 54)), nu);
  de = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(de, vcast_vd_d(1ULL << 54)), de);
  s  = vsel_vd_vo_vd_vd(o, vmul_vd_vd_vd(s , vcast_vd_d(1.0 / (1ULL << 54))), s);

  vdouble2 q, r = vcast_vd2_vd_vd(nu, vcast_vd_d(0));

  for(int i=0;i<20;i++) { // ceil(log2(DBL_MAX) / 52)
    q = ddnormalize_vd2_vd2(dddiv_vd2_vd2_vd2(r, vcast_vd2_vd_vd(de, vcast_vd_d(0))));
    r = ddnormalize_vd2_vd2(ddadd2_vd2_vd2_vd2(r, ddmul_vd2_vd_vd(upper2(xtrunc(vsel_vd_vo_vd_vd(vlt_vo_vd_vd(q.y, vcast_vd_d(0)), vnexttoward0(q.x), q.x))), vneg_vd_vd(de))));
    if (vtestallones_i_vo64(vlt_vo_vd_vd(r.x, y))) break;
  }

  vdouble ret = vmul_vd_vd_vd(r.x, s);
  ret = vsel_vd_vo_vd_vd(veq_vo_vd_vd(vadd_vd_vd_vd(r.x, r.y), de), vcast_vd_d(0), ret);

  ret = vmulsign_vd_vd_vd(ret, x);

  ret = vsel_vd_vo_vd_vd(vlt_vo_vd_vd(vabs_vd_vd(x), vabs_vd_vd(y)), x, ret);

  return ret;
}

typedef struct {
  vdouble2 a, b;
} dd2;

/* TODO AArch64: potential optimization by using `vfmad_lane_f64` */
static CONST dd2 gammak(vdouble a) {
  vdouble2 clc = vcast_vd2_d_d(0, 0), clln = vcast_vd2_d_d(1, 0), clld = vcast_vd2_d_d(1, 0);
  vdouble2 v = vcast_vd2_d_d(1, 0), x, y, z;
  vdouble t, u;

  vopmask otiny = vlt_vo_vd_vd(vabs_vd_vd(a), vcast_vd_d(1e-306)), oref = vlt_vo_vd_vd(a, vcast_vd_d(0.5));

  x = vsel_vd2_vo_vd2_vd2(otiny, vcast_vd2_d_d(0, 0),
        vsel_vd2_vo_vd2_vd2(oref, ddadd2_vd2_vd_vd(vcast_vd_d(1), vneg_vd_vd(a)),
                vcast_vd2_vd_vd(a, vcast_vd_d(0))));

  vopmask o0 = vand_vo_vo_vo(vle_vo_vd_vd(vcast_vd_d(0.5), x.x), vle_vo_vd_vd(x.x, vcast_vd_d(1.1)));
  vopmask o2 = vle_vo_vd_vd(vcast_vd_d(2.3), x.x);

  y = ddnormalize_vd2_vd2(ddmul_vd2_vd2_vd2(ddadd2_vd2_vd2_vd(x, vcast_vd_d(1)), x));
  y = ddnormalize_vd2_vd2(ddmul_vd2_vd2_vd2(ddadd2_vd2_vd2_vd(x, vcast_vd_d(2)), y));
  y = ddnormalize_vd2_vd2(ddmul_vd2_vd2_vd2(ddadd2_vd2_vd2_vd(x, vcast_vd_d(3)), y));
  y = ddnormalize_vd2_vd2(ddmul_vd2_vd2_vd2(ddadd2_vd2_vd2_vd(x, vcast_vd_d(4)), y));

  vopmask o = vand_vo_vo_vo(o2, vle_vo_vd_vd(x.x, vcast_vd_d(7)));
  clln = vsel_vd2_vo_vd2_vd2(o, y, clln);

  x = vsel_vd2_vo_vd2_vd2(o, ddadd2_vd2_vd2_vd(x, vcast_vd_d(5)), x);

  t = vsel_vd_vo_vd_vd(o2, vrec_vd_vd(x.x), ddnormalize_vd2_vd2(ddadd2_vd2_vd2_vd(x, vsel_vd_vo_d_d(o0, -1, -2))).x);

  u = vsel_vd_vo_vo_d_d_d(o2, o0, -156.801412704022726379848862, +0.2947916772827614196e+2, +0.7074816000864609279e-7);
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, +1.120804464289911606838558160000, +0.1281459691827820109e+3, +0.4009244333008730443e-6));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, +13.39798545514258921833306020000, +0.2617544025784515043e+3, +0.1040114641628246946e-5));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, -0.116546276599463200848033357000, +0.3287022855685790432e+3, +0.1508349150733329167e-5));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, -1.391801093265337481495562410000, +0.2818145867730348186e+3, +0.1288143074933901020e-5));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, +0.015056113040026424412918973400, +0.1728670414673559605e+3, +0.4744167749884993937e-6));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, +0.179540117061234856098844714000, +0.7748735764030416817e+2, -0.6554816306542489902e-7));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, -0.002481743600264997730942489280, +0.2512856643080930752e+2, -0.3189252471452599844e-6));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, -0.029527880945699120504851034100, +0.5766792106140076868e+1, +0.1358883821470355377e-6));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, +0.000540164767892604515196325186, +0.7270275473996180571e+0, -0.4343931277157336040e-6));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, +0.006403362833808069794787256200, +0.8396709124579147809e-1, +0.9724785897406779555e-6));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, -0.000162516262783915816896611252, -0.8211558669746804595e-1, -0.2036886057225966011e-5));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, -0.001914438498565477526465972390, +0.6828831828341884458e-1, +0.4373363141819725815e-5));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, +7.20489541602001055898311517e-05, -0.7712481339961671511e-1, -0.9439951268304008677e-5));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, +0.000839498720672087279971000786, +0.8337492023017314957e-1, +0.2050727030376389804e-4));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, -5.17179090826059219329394422e-05, -0.9094964931456242518e-1, -0.4492620183431184018e-4));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, -0.000592166437353693882857342347, +0.1000996313575929358e+0, +0.9945751236071875931e-4));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, +6.97281375836585777403743539e-05, -0.1113342861544207724e+0, -0.2231547599034983196e-3));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, +0.000784039221720066627493314301, +0.1255096673213020875e+0, +0.5096695247101967622e-3));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, -0.000229472093621399176949318732, -0.1440498967843054368e+0, -0.1192753911667886971e-2));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, -0.002681327160493827160473958490, +0.1695571770041949811e+0, +0.2890510330742210310e-2));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, +0.003472222222222222222175164840, -0.2073855510284092762e+0, -0.7385551028674461858e-2));
  u = vmla_vd_vd_vd_vd(u, t, vsel_vd_vo_vo_d_d_d(o2, o0, +0.083333333333333333335592087900, +0.2705808084277815939e+0, +0.2058080842778455335e-1));

  y = ddmul_vd2_vd2_vd2(ddadd2_vd2_vd2_vd(x, vcast_vd_d(-0.5)), logk2(x));
  y = ddadd2_vd2_vd2_vd2(y, ddneg_vd2_vd2(x));
  y = ddadd2_vd2_vd2_vd2(y, vcast_vd2_d_d(0.91893853320467278056, -3.8782941580672414498e-17)); // 0.5*log(2*M_PI)

  z = ddadd2_vd2_vd2_vd(ddmul_vd2_vd_vd (u, t), vsel_vd_vo_d_d(o0, -0.4006856343865314862e+0, -0.6735230105319810201e-1));
  z = ddadd2_vd2_vd2_vd(ddmul_vd2_vd2_vd(z, t), vsel_vd_vo_d_d(o0, +0.8224670334241132030e+0, +0.3224670334241132030e+0));
  z = ddadd2_vd2_vd2_vd(ddmul_vd2_vd2_vd(z, t), vsel_vd_vo_d_d(o0, -0.5772156649015328655e+0, +0.4227843350984671345e+0));
  z = ddmul_vd2_vd2_vd(z, t);

  clc = vsel_vd2_vo_vd2_vd2(o2, y, z);

  clld = vsel_vd2_vo_vd2_vd2(o2, ddadd2_vd2_vd2_vd(ddmul_vd2_vd_vd(u, t), vcast_vd_d(1)), clld);

  y = clln;

  clc = vsel_vd2_vo_vd2_vd2(otiny, vcast_vd2_d_d(83.1776616671934334590333, 3.67103459631568507221878e-15), // log(2^120)
          vsel_vd2_vo_vd2_vd2(oref, ddadd2_vd2_vd2_vd2(vcast_vd2_d_d(1.1447298858494001639, 1.026595116270782638e-17), ddneg_vd2_vd2(clc)), clc)); // log(M_PI)
  clln = vsel_vd2_vo_vd2_vd2(otiny, vcast_vd2_d_d(1, 0), vsel_vd2_vo_vd2_vd2(oref, clln, clld));

  if (!vtestallones_i_vo64(vnot_vo64_vo64(oref))) {
    t = vsub_vd_vd_vd(a, vmul_vd_vd_vd(vcast_vd_d(1LL << 28), vcast_vd_vi(vtruncate_vi_vd(vmul_vd_vd_vd(a, vcast_vd_d(1.0 / (1LL << 28)))))));
    x = ddmul_vd2_vd2_vd2(clld, sinpik(t));
  }

  clld = vsel_vd2_vo_vd2_vd2(otiny, vcast_vd2_vd_vd(vmul_vd_vd_vd(a, vcast_vd_d((1LL << 60)*(double)(1LL << 60))), vcast_vd_d(0)),
           vsel_vd2_vo_vd2_vd2(oref, x, y));

  dd2 ret = { clc, dddiv_vd2_vd2_vd2(clln, clld) };

  return ret;
}

EXPORT CONST vdouble xtgamma_u1(vdouble a) {
  dd2 d = gammak(a);
  vdouble2 y = ddmul_vd2_vd2_vd2(expk2(d.a), d.b);
  vdouble r = vadd_vd_vd_vd(y.x, y.y);
  vopmask o;

  o = vor_vo_vo_vo(vor_vo_vo_vo(veq_vo_vd_vd(a, vcast_vd_d(-INFINITY)),
        vand_vo_vo_vo(vlt_vo_vd_vd(a, vcast_vd_d(0)), visint_vo_vd(a))),
       vand_vo_vo_vo(vand_vo_vo_vo(visnumber_vo_vd(a), vlt_vo_vd_vd(a, vcast_vd_d(0))), visnan_vo_vd(r)));
  r = vsel_vd_vo_vd_vd(o, vcast_vd_d(NAN), r);

  o = vand_vo_vo_vo(vand_vo_vo_vo(vor_vo_vo_vo(veq_vo_vd_vd(a, vcast_vd_d(INFINITY)), visnumber_vo_vd(a)),
          vge_vo_vd_vd(a, vcast_vd_d(-DBL_MIN))),
        vor_vo_vo_vo(vor_vo_vo_vo(veq_vo_vd_vd(a, vcast_vd_d(0)), vgt_vo_vd_vd(a, vcast_vd_d(200))), visnan_vo_vd(r)));
  r = vsel_vd_vo_vd_vd(o, vmulsign_vd_vd_vd(vcast_vd_d(INFINITY), a), r);

  return r;
}

EXPORT CONST vdouble xlgamma_u1(vdouble a) {
  dd2 d = gammak(a);
  vdouble2 y = ddadd2_vd2_vd2_vd2(d.a, logk2(ddabs_vd2_vd2(d.b)));
  vdouble r = vadd_vd_vd_vd(y.x, y.y);
  vopmask o;

  o = vor_vo_vo_vo(visinf_vo_vd(a),
       vor_vo_vo_vo(vand_vo_vo_vo(vle_vo_vd_vd(a, vcast_vd_d(0)), visint_vo_vd(a)),
        vand_vo_vo_vo(visnumber_vo_vd(a), visnan_vo_vd(r))));
  r = vsel_vd_vo_vd_vd(o, vcast_vd_d(INFINITY), r);

  return r;
}

/* TODO AArch64: potential optimization by using `vfmad_lane_f64` */
EXPORT CONST vdouble xerf_u1(vdouble a) {
  vdouble s = a, t, u;
  vdouble2 d;

  a = vabs_vd_vd(a);
  vopmask o0 = vlt_vo_vd_vd(a, vcast_vd_d(1.0));
  vopmask o1 = vlt_vo_vd_vd(a, vcast_vd_d(3.7));
  vopmask o2 = vlt_vo_vd_vd(a, vcast_vd_d(6.0));
  u = vsel_vd_vo_vd_vd(o0, vmul_vd_vd_vd(a, a), a);

  t = vsel_vd_vo_vo_d_d_d(o0, o1, +0.6801072401395392157e-20, +0.2830954522087717660e-13, -0.5846750404269610493e-17);
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, -0.2161766247570056391e-18, -0.1509491946179481940e-11, +0.6076691048812607898e-15));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, +0.4695919173301598752e-17, +0.3827857177807173152e-10, -0.3007518609604893831e-13));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, -0.9049140419888010819e-16, -0.6139733921558987241e-09, +0.9427906260824646063e-12));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, +0.1634018903557411517e-14, +0.6985387934608038824e-08, -0.2100110908269393629e-10));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, -0.2783485786333455216e-13, -0.5988224513034371474e-07, +0.3534639523461223473e-09));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, +0.4463221276786412722e-12, +0.4005716952355346640e-06, -0.4664967728285395926e-08));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, -0.6711366622850138987e-11, -0.2132190104575784400e-05, +0.4943823283769000532e-07));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, +0.9422759050232658346e-10, +0.9092461304042630325e-05, -0.4271203394761148254e-06));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, -0.1229055530100228477e-08, -0.3079188080966205457e-04, +0.3034067677404915895e-05));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, +0.1480719281585085023e-07, +0.7971413443082370762e-04, -0.1776295289066871135e-04));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, -0.1636584469123402714e-06, -0.1387853215225442864e-03, +0.8524547630559505050e-04));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, +0.1646211436588923363e-05, +0.6469678026257590965e-04, -0.3290582944961784398e-03));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, -0.1492565035840624866e-04, +0.4996645280372945860e-03, +0.9696966068789101157e-03));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, +0.1205533298178966496e-03, -0.1622802482842520535e-02, -0.1812527628046986137e-02));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, -0.8548327023450851166e-03, +0.1615320557049377171e-03, -0.4725409828123619017e-03));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, +0.5223977625442188799e-02, +0.1915262325574875607e-01, +0.2090315427924229266e-01));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, -0.2686617064513125569e-01, -0.1027818298486033455e+00, -0.1052041921842776645e+00));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, +0.1128379167095512753e+00, -0.6366172819842503827e+00, -0.6345351808766568347e+00));
  t = vmla_vd_vd_vd_vd(t, u, vsel_vd_vo_vo_d_d_d(o0, o1, -0.3761263890318375380e+00, -0.1128379590648910469e+01, -0.1129442929103524396e+01));
  d = ddmul_vd2_vd_vd(t, u);

  d = ddadd2_vd2_vd2_vd2(d, vcast_vd2_vd_vd(vsel_vd_vo_vo_d_d_d(o0, o1, 1.1283791670955125586, 3.4110644736196137587e-08, 0.00024963035690526438285),
              vsel_vd_vo_vo_d_d_d(o0, o1, 1.5335459613165822674e-17, -2.4875650708323294246e-24, -5.4362665034856259795e-21)));
  d = vsel_vd2_vo_vd2_vd2(o0, ddmul_vd2_vd2_vd(d, a), ddadd_vd2_vd_vd2(vcast_vd_d(1.0), ddneg_vd2_vd2(expk2(d))));

  u = vmulsign_vd_vd_vd(vsel_vd_vo_vd_vd(o2, vadd_vd_vd_vd(d.x, d.y), vcast_vd_d(1)), s);
  u = vsel_vd_vo_vd_vd(visnan_vo_vd(a), vcast_vd_d(NAN), u);

  return u;
}

/* TODO AArch64: potential optimization by using `vfmad_lane_f64` */
EXPORT CONST vdouble xerfc_u15(vdouble a) {
  vdouble s = a, r = vcast_vd_d(0), t;
  vdouble2 u, d, x;
  a = vabs_vd_vd(a);
  vopmask o0 = vlt_vo_vd_vd(a, vcast_vd_d(1.0));
  vopmask o1 = vlt_vo_vd_vd(a, vcast_vd_d(2.2));
  vopmask o2 = vlt_vo_vd_vd(a, vcast_vd_d(4.2));
  vopmask o3 = vlt_vo_vd_vd(a, vcast_vd_d(27.3));

  u = vsel_vd2_vo_vd2_vd2(o0, ddmul_vd2_vd_vd(a, a), vsel_vd2_vo_vd2_vd2(o1, vcast_vd2_vd_vd(a, vcast_vd_d(0)), dddiv_vd2_vd2_vd2(vcast_vd2_d_d(1, 0), vcast_vd2_vd_vd(a, vcast_vd_d(0)))));

  t = vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, +0.6801072401395386139e-20, +0.3438010341362585303e-12, -0.5757819536420710449e+2, +0.2334249729638701319e+5);
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.2161766247570055669e-18, -0.1237021188160598264e-10, +0.4669289654498104483e+3, -0.4695661044933107769e+5));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, +0.4695919173301595670e-17, +0.2117985839877627852e-09, -0.1796329879461355858e+4, +0.3173403108748643353e+5));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.9049140419888007122e-16, -0.2290560929177369506e-08, +0.4355892193699575728e+4, +0.3242982786959573787e+4));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, +0.1634018903557410728e-14, +0.1748931621698149538e-07, -0.7456258884965764992e+4, -0.2014717999760347811e+5));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.2783485786333451745e-13, -0.9956602606623249195e-07, +0.9553977358167021521e+4, +0.1554006970967118286e+5));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, +0.4463221276786415752e-12, +0.4330010240640327080e-06, -0.9470019905444229153e+4, -0.6150874190563554293e+4));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.6711366622850136563e-11, -0.1435050600991763331e-05, +0.7387344321849855078e+4, +0.1240047765634815732e+4));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, +0.9422759050232662223e-10, +0.3460139479650695662e-05, -0.4557713054166382790e+4, -0.8210325475752699731e+2));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.1229055530100229098e-08, -0.4988908180632898173e-05, +0.2207866967354055305e+4, +0.3242443880839930870e+2));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, +0.1480719281585086512e-07, -0.1308775976326352012e-05, -0.8217975658621754746e+3, -0.2923418863833160586e+2));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.1636584469123399803e-06, +0.2825086540850310103e-04, +0.2268659483507917400e+3, +0.3457461732814383071e+0));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, +0.1646211436588923575e-05, -0.6393913713069986071e-04, -0.4633361260318560682e+2, +0.5489730155952392998e+1));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.1492565035840623511e-04, -0.2566436514695078926e-04, +0.9557380123733945965e+1, +0.1559934132251294134e-2));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, +0.1205533298178967851e-03, +0.5895792375659440364e-03, -0.2958429331939661289e+1, -0.1541741566831520638e+1));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.8548327023450850081e-03, -0.1695715579163588598e-02, +0.1670329508092765480e+0, +0.2823152230558364186e-5));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, +0.5223977625442187932e-02, +0.2089116434918055149e-03, +0.6096615680115419211e+0, +0.6249999184195342838e+0));
  t = vmla_vd_vd_vd_vd(t, u.x, vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.2686617064513125222e-01, +0.1912855949584917753e-01, +0.1059212443193543585e-2, +0.1741749416408701288e-8));

  d = ddmul_vd2_vd2_vd(u, t);
  d = ddadd2_vd2_vd2_vd2(d, vcast_vd2_vd_vd(vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, 0.11283791670955126141, -0.10277263343147646779, -0.50005180473999022439, -0.5000000000258444377),
              vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, -4.0175691625932118483e-18, -6.2338714083404900225e-18, 2.6362140569041995803e-17, -4.0074044712386992281e-17)));
  d = ddmul_vd2_vd2_vd2(d, u);
  d = ddadd2_vd2_vd2_vd2(d, vcast_vd2_vd_vd(vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.37612638903183753802, -0.63661976742916359662, 1.601106273924963368e-06, 2.3761973137523364792e-13),
              vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, 1.3391897206042552387e-17, 7.6321019159085724662e-18, 1.1974001857764476775e-23, -1.1670076950531026582e-29)));
  d = ddmul_vd2_vd2_vd2(d, u);
  d = ddadd2_vd2_vd2_vd2(d, vcast_vd2_vd_vd(vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, 1.1283791670955125586, -1.1283791674717296161, -0.57236496645145429341, -0.57236494292470108114),
              vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o2, 1.5335459613165822674e-17, 8.0896847755965377194e-17, 3.0704553245872027258e-17, -2.3984352208056898003e-17)));

  x = ddmul_vd2_vd2_vd(vsel_vd2_vo_vd2_vd2(o1, d, vcast_vd2_vd_vd(vneg_vd_vd(a), vcast_vd_d(0))), a);
  x = vsel_vd2_vo_vd2_vd2(o1, x, ddadd2_vd2_vd2_vd2(x, d));
  x = vsel_vd2_vo_vd2_vd2(o0, ddsub_vd2_vd2_vd2(vcast_vd2_d_d(1, 0), x), expk2(x));
  x = vsel_vd2_vo_vd2_vd2(o1, x, ddmul_vd2_vd2_vd2(x, u));

  r = vsel_vd_vo_vd_vd(o3, vadd_vd_vd_vd(x.x, x.y), vcast_vd_d(0));
  r = vsel_vd_vo_vd_vd(vsignbit_vo_vd(s), vsub_vd_vd_vd(vcast_vd_d(2), r), r);
  r = vsel_vd_vo_vd_vd(visnan_vo_vd(s), vcast_vd_d(NAN), r);
  return r;
}

#ifdef ENABLE_MAIN
// gcc -DENABLE_MAIN -Wno-attributes -I../common -I../arch -DENABLE_AVX2 -mavx2 -mfma sleefsimddp.c ../common/common.c -lm
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  vdouble d1 = vcast_vd_d(atof(argv[1]));
  vdouble d2 = vcast_vd_d(atof(argv[2]));
  //vdouble d3 = vcast_vd_d(atof(argv[3]));
  //vdouble r = xnextafter(d1, d2);
  //int i;
  //double fr = frexp(atof(argv[1]), &i);
  //printf("%.20g\n", xfma(d1, d2, d3)[0]);;
  //printf("test %.20g\n", xtgamma_u1(d1)[0]);
  //printf("corr %.20g\n", tgamma(d1[0]));
  //printf("test %.20g\n", xerf_u1(d1)[0]);
  //printf("corr %.20g\n", erf(d1[0]));
  //printf("test %.20g\n", xerfc_u15(d1)[0]);
  //printf("corr %.20g\n", erfc(d1[0]));
  //printf("%.20g\n", nextafter(d1[0], d2[0]));;
  //printf("%.20g\n", vcast_d_vd(xhypot_u05(d1, d2)));
  //printf("%.20g\n", fr);
  printf("%.20g\n", fmod(atof(argv[1]), atof(argv[2])));
  printf("%.20g\n", xfmod(d1, d2)[0]);
  //vdouble2 r = xsincospi_u35(a);
  //printf("%g, %g\n", vcast_d_vd(r.x), vcast_d_vd(r.y));
}
#endif
