//          Copyright Naoki Shibata 2010 - 2017.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

// Always use -ffp-contract=off option to compile SLEEF.

#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <limits.h>
#include <float.h>

#include "misc.h"

#if (defined(_MSC_VER))
#pragma fp_contract (off)
#endif

//

#include "helpers.h"
#include "df.h"

static INLINE CONST vopmask visnegzero_vo_vf(vfloat d) {
  return veq_vo_vi2_vi2(vreinterpret_vi2_vf(d), vreinterpret_vi2_vf(vcast_vf_f(-0.0)));
}

static INLINE vopmask vnot_vo32_vo32(vopmask x) {
  return vxor_vo_vo_vo(x, veq_vo_vi2_vi2(vcast_vi2_i(0), vcast_vi2_i(0)));
}

static INLINE CONST vmask vsignbit_vm_vf(vfloat f) {
  return vand_vm_vm_vm(vreinterpret_vm_vf(f), vreinterpret_vm_vf(vcast_vf_f(-0.0f)));
}

static INLINE CONST vfloat vmulsign_vf_vf_vf(vfloat x, vfloat y) {
  return vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(x), vsignbit_vm_vf(y)));
}

static INLINE CONST vfloat vcopysign_vf_vf_vf(vfloat x, vfloat y) {
  return vreinterpret_vf_vm(vxor_vm_vm_vm(vandnot_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0f)), vreinterpret_vm_vf(x)),
            vand_vm_vm_vm   (vreinterpret_vm_vf(vcast_vf_f(-0.0f)), vreinterpret_vm_vf(y))));
}

static INLINE CONST vfloat vsign_vf_vf(vfloat f) {
  return vreinterpret_vf_vm(vor_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(1.0f)), vand_vm_vm_vm(vreinterpret_vm_vf(vcast_vf_f(-0.0f)), vreinterpret_vm_vf(f))));
}

static INLINE CONST vopmask vsignbit_vo_vf(vfloat d) {
  return veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vreinterpret_vi2_vf(d), vcast_vi2_i(0x80000000)), vcast_vi2_i(0x80000000));
}

static INLINE CONST vint2 vsel_vi2_vf_vf_vi2_vi2(vfloat f0, vfloat f1, vint2 x, vint2 y) {
  return vsel_vi2_vo_vi2_vi2(vlt_vo_vf_vf(f0, f1), x, y);
}

static INLINE CONST vint2 vsel_vi2_vf_vi2(vfloat d, vint2 x) {
  return vand_vi2_vo_vi2(vsignbit_vo_vf(d), x);
}

static INLINE CONST vopmask visint_vo_vf(vfloat y) { return veq_vo_vf_vf(vtruncate_vf_vf(y), y); }

static INLINE CONST vopmask visnumber_vo_vf(vfloat x) { return vnot_vo32_vo32(vor_vo_vo_vo(visinf_vo_vf(x), visnan_vo_vf(x))); }

#ifndef ENABLE_AVX512F
static INLINE CONST vint2 vilogbk_vi2_vf(vfloat d) {
  vopmask o = vlt_vo_vf_vf(d, vcast_vf_f(5.421010862427522E-20f));
  d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(vcast_vf_f(1.8446744073709552E19f), d), d);
  vint2 q = vand_vi2_vi2_vi2(vsrl_vi2_vi2_i(vcast_vi2_vm(vreinterpret_vm_vf(d)), 23), vcast_vi2_i(0xff));
  q = vsub_vi2_vi2_vi2(q, vsel_vi2_vo_vi2_vi2(o, vcast_vi2_i(64 + 0x7f), vcast_vi2_i(0x7f)));
  return q;
}

static INLINE CONST vint2 vilogb2k_vi2_vf(vfloat d) {
  vint2 q = vreinterpret_vi2_vf(d);
  q = vsrl_vi2_vi2_i(q, 23);
  q = vand_vi2_vi2_vi2(q, vcast_vi2_i(0xff));
  q = vsub_vi2_vi2_vi2(q, vcast_vi2_i(0x7f));
  return q;
}
#endif

//

EXPORT CONST vmask xilogbf(vfloat d) {
  vint2 e = vilogbk_vi2_vf(vabs_vf_vf(d));
  e = vsel_vi2_vo_vi2_vi2(veq_vo_vf_vf(d, vcast_vf_f(0.0f)), vcast_vi2_i(FP_ILOGB0), e);
  e = vsel_vi2_vo_vi2_vi2(visnan_vo_vf(d), vcast_vi2_i(FP_ILOGBNAN), e);
  e = vsel_vi2_vo_vi2_vi2(visinf_vo_vf(d), vcast_vi2_i(INT_MAX), e);
  return vcast_vm_vi2(e);
}

static INLINE CONST vfloat vpow2i_vf_vi2(vint2 q) {
  return vreinterpret_vf_vm(vcast_vm_vi2(vsll_vi2_vi2_i(vadd_vi2_vi2_vi2(q, vcast_vi2_i(0x7f)), 23)));
}

static INLINE CONST vfloat vldexp_vf_vf_vi2(vfloat x, vint2 q) {
  vfloat u;
  vint2 m = vsra_vi2_vi2_i(q, 31);
  m = vsll_vi2_vi2_i(vsub_vi2_vi2_vi2(vsra_vi2_vi2_i(vadd_vi2_vi2_vi2(m, q), 6), m), 4);
  q = vsub_vi2_vi2_vi2(q, vsll_vi2_vi2_i(m, 2));
  m = vadd_vi2_vi2_vi2(m, vcast_vi2_i(0x7f));
  m = vand_vi2_vi2_vi2(vgt_vi2_vi2_vi2(m, vcast_vi2_i(0)), m);
  vint2 n = vgt_vi2_vi2_vi2(m, vcast_vi2_i(0xff));
  m = vor_vi2_vi2_vi2(vandnot_vi2_vi2_vi2(n, m), vand_vi2_vi2_vi2(n, vcast_vi2_i(0xff)));
  u = vreinterpret_vf_vm(vcast_vm_vi2(vsll_vi2_vi2_i(m, 23)));
  x = vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(x, u), u), u), u);
  u = vreinterpret_vf_vm(vcast_vm_vi2(vsll_vi2_vi2_i(vadd_vi2_vi2_vi2(q, vcast_vi2_i(0x7f)), 23)));
  return vmul_vf_vf_vf(x, u);
}

static INLINE CONST vfloat vldexp2_vf_vf_vi2(vfloat d, vint2 e) {
  return vmul_vf_vf_vf(vmul_vf_vf_vf(d, vpow2i_vf_vi2(vsra_vi2_vi2_i(e, 1))), vpow2i_vf_vi2(vsub_vi2_vi2_vi2(e, vsra_vi2_vi2_i(e, 1))));
}

static INLINE CONST vfloat vldexp3_vf_vf_vi2(vfloat d, vint2 q) {
  return vreinterpret_vf_vi2(vadd_vi2_vi2_vi2(vreinterpret_vi2_vf(d), vsll_vi2_vi2_i(q, 23)));
}

EXPORT CONST vfloat xldexpf(vfloat x, vmask qm) {
  vint2 q1 = vcast_vi2_vm(qm);
  vint2 min = vcast_vi2_i(-2000);
  vint2 mask = vgt_vi2_vi2_vi2(min, q1);
  vint2 q = vor_vi2_vi2_vi2(vand_vi2_vi2_vi2(mask, min),
                                vandnot_vi2_vi2_vi2(mask, q1));
  vfloat res = vldexp_vf_vf_vi2(x, q);

  res = vsel_vf_vo_vf_vf(veq_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(0.0f)), x, res);
  res = vsel_vf_vo_vf_vf(visinf_vo_vf(x), x, res);
  res = vsel_vf_vo_vf_vf(visnan_vo_vf(x), x, res);
  return res;
}

EXPORT CONST vfloat xsinf(vfloat d) {
  vint2 q;
  vfloat u, s, r = d;

  q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f((float)M_1_PI)));
  u = vcast_vf_vi2(q);

  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Af), d);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Bf), d);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Cf), d);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Df), d);

  s = vmul_vf_vf_vf(d, d);

  d = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1)), vreinterpret_vm_vf(vcast_vf_f(-0.0f))), vreinterpret_vm_vf(d)));

  u = vcast_vf_f(2.6083159809786593541503e-06f);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.0001981069071916863322258f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00833307858556509017944336f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.166666597127914428710938f));

  u = vadd_vf_vf_vf(vmul_vf_vf_vf(s, vmul_vf_vf_vf(u, d)), d);

  u = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visnegzero_vo_vf(r),
            vgt_vo_vf_vf(vabs_vf_vf(r), vcast_vf_f(TRIGRANGEMAXf))),
           vcast_vf_f(-0.0), u);

  u = vreinterpret_vf_vm(vor_vm_vo32_vm(visinf_vo_vf(d), vreinterpret_vm_vf(u)));

  return u;
}

EXPORT CONST vfloat xcosf(vfloat d) {
  vint2 q;
  vfloat u, s, r = d;

  q = vrint_vi2_vf(vsub_vf_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f((float)M_1_PI)), vcast_vf_f(0.5f)));
  q = vadd_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, q), vcast_vi2_i(1));

  u = vcast_vf_vi2(q);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Af*0.5f), d);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Bf*0.5f), d);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Cf*0.5f), d);
  d = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Df*0.5f), d);

  s = vmul_vf_vf_vf(d, d);

  d = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(0)), vreinterpret_vm_vf(vcast_vf_f(-0.0f))), vreinterpret_vm_vf(d)));

  u = vcast_vf_f(2.6083159809786593541503e-06f);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.0001981069071916863322258f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00833307858556509017944336f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.166666597127914428710938f));

  u = vadd_vf_vf_vf(vmul_vf_vf_vf(s, vmul_vf_vf_vf(u, d)), d);

  u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(vgt_vo_vf_vf(vabs_vf_vf(r), vcast_vf_f(TRIGRANGEMAXf)),
              vreinterpret_vm_vf(u)));

  u = vreinterpret_vf_vm(vor_vm_vo32_vm(visinf_vo_vf(d), vreinterpret_vm_vf(u)));

  return u;
}

EXPORT CONST vfloat xtanf(vfloat d) {
  vint2 q;
  vopmask o;
  vfloat u, s, x;

  q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f((float)(2 * M_1_PI))));

  x = d;

  u = vcast_vf_vi2(q);
  x = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Af*0.5f), x);
  x = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Bf*0.5f), x);
  x = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Cf*0.5f), x);
  x = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Df*0.5f), x);

  s = vmul_vf_vf_vf(x, x);

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1));
  x = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0f))), vreinterpret_vm_vf(x)));

  u = vcast_vf_f(0.00927245803177356719970703f);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00331984995864331722259521f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0242998078465461730957031f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0534495301544666290283203f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.133383005857467651367188f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.333331853151321411132812f));

  u = vmla_vf_vf_vf_vf(s, vmul_vf_vf_vf(u, x), x);

  u = vsel_vf_vo_vf_vf(o, vrec_vf_vf(u), u);

#ifndef ENABLE_AVX512F
  u = vreinterpret_vf_vm(vor_vm_vo32_vm(visinf_vo_vf(d), vreinterpret_vm_vf(u)));
#else
  u = vfixup_vf_vf_vf_vi2_i(u, d, vcast_vi2_i((3 << (4*4)) | (3 << (5*4))), 0);
#endif

  return u;
}

EXPORT CONST vfloat xsinf_u1(vfloat d) {
  vint2 q;
  vfloat u, v;
  vfloat2 s, t, x;

  if (vtestallones_i_vo32(vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX2f)))) {
    u = vrint_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(M_1_PI)));
    q = vrint_vi2_vf(u);
    v = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_A2f), d);
    s = dfadd2_vf2_vf_vf(v, vmul_vf_vf_vf(u, vcast_vf_f(-PI_B2f)));
    s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_C2f)));
  } else {
    vfloat2 dfq = dfmul_vf2_vf2_vf(vcast_vf2_f_f(M_1_PI, M_1_PI - (float)M_1_PI), d);
    vfloat t = vrint_vf_vf(vmul_vf_vf_vf(dfq.x, vcast_vf_f(1.0f / (1 << 16))));
    dfq.y = vrint_vf_vf(vadd_vf_vf_vf(vmla_vf_vf_vf_vf(t, vcast_vf_f(-(1 << 16)), dfq.x), dfq.y));
    q = vrint_vi2_vf(dfq.y);
    dfq.x = vmul_vf_vf_vf(t, vcast_vf_f(1 << 16));
    dfq = dfnormalize_vf2_vf2(dfq);

    s = dfadd2_vf2_vf_vf2 (d, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_A3f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_B3f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_C3f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_D3f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_E3f)));
    s = dfnormalize_vf2_vf2(s);
  }

  t = s;
  s = dfsqu_vf2_vf2(s);

  u = vcast_vf_f(2.6083159809786593541503e-06f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(-0.0001981069071916863322258f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.00833307858556509017944336f));

  x = dfadd_vf2_vf_vf2(vcast_vf_f(1), dfmul_vf2_vf2_vf2(dfadd_vf2_vf_vf(vcast_vf_f(-0.166666597127914428710938f), vmul_vf_vf_vf(u, s.x)), s));

  u = dfmul_vf_vf2_vf2(t, x);

  u = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1)), vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(u)));
  u = vsel_vf_vo_vf_vf(vandnot_vo_vo_vo(visinf_vo_vf(d), vor_vo_vo_vo(visnegzero_vo_vf(d),
                      vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX3f)))),
           vcast_vf_f(-0.0), u);

  return u;
}

EXPORT CONST vfloat xcosf_u1(vfloat d) {
  vint2 q;
  vfloat u;
  vfloat2 s, t, x;

  if (vtestallones_i_vo32(vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX2f)))) {
    vfloat dq = vmla_vf_vf_vf_vf(vrint_vf_vf(vmla_vf_vf_vf_vf(d, vcast_vf_f(M_1_PI), vcast_vf_f(-0.5f))),
         vcast_vf_f(2), vcast_vf_f(1));
    q = vrint_vi2_vf(dq);
    s = dfadd2_vf2_vf_vf (d, vmul_vf_vf_vf(dq, vcast_vf_f(-PI_A2f*0.5f)));
    s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(dq, vcast_vf_f(-PI_B2f*0.5f)));
    s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(dq, vcast_vf_f(-PI_C2f*0.5f)));
  } else {
    vfloat2 dfq = dfadd2_vf2_vf2_vf(dfmul_vf2_vf2_vf(vcast_vf2_f_f(M_1_PI, M_1_PI - (float)M_1_PI), d), vcast_vf_f(-0.5f));
    vfloat t = vrint_vf_vf(vmul_vf_vf_vf(dfq.x, vcast_vf_f(1.0f / (1 << 16))));
    dfq.y = vmla_vf_vf_vf_vf(vrint_vf_vf(vadd_vf_vf_vf(vmla_vf_vf_vf_vf(t, vcast_vf_f(-(1 << 16)), dfq.x), dfq.y)),
           vcast_vf_f(2), vcast_vf_f(1));
    q = vrint_vi2_vf(dfq.y);
    dfq.x = vmul_vf_vf_vf(t, vcast_vf_f(1 << 17));
    dfq = dfnormalize_vf2_vf2(dfq);

    s = dfadd2_vf2_vf_vf2 (d, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_A3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_B3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_C3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_D3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_E3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
  }

  t = s;
  s = dfsqu_vf2_vf2(s);

  u = vcast_vf_f(2.6083159809786593541503e-06f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(-0.0001981069071916863322258f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.00833307858556509017944336f));

  x = dfadd_vf2_vf_vf2(vcast_vf_f(1), dfmul_vf2_vf2_vf2(dfadd_vf2_vf_vf(vcast_vf_f(-0.166666597127914428710938f), vmul_vf_vf_vf(u, s.x)), s));

  u = dfmul_vf_vf2_vf2(t, x);

  u = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(0)), vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(u)));

  u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(vandnot_vo_vo_vo(visinf_vo_vf(d), vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX3f))),
              vreinterpret_vm_vf(u)));

  return u;
}

#ifdef ENABLE_GNUABI
#define TYPE2_FUNCATR static INLINE CONST
#define TYPE6_FUNCATR static INLINE CONST
#define XSINCOSF sincosfk
#define XSINCOSF_U1 sincosfk_u1
#define XSINCOSPIF_U05 sincospifk_u05
#define XSINCOSPIF_U35 sincospifk_u35
#define XMODFF modffk
#else
#define TYPE2_FUNCATR EXPORT CONST
#define TYPE6_FUNCATR EXPORT
#define XSINCOSF xsincosf
#define XSINCOSF_U1 xsincosf_u1
#define XSINCOSPIF_U05 xsincospif_u05
#define XSINCOSPIF_U35 xsincospif_u35
#define XMODFF xmodff
#endif

TYPE2_FUNCATR vfloat2 XSINCOSF(vfloat d) {
  vint2 q;
  vopmask o;
  vfloat u, s, t, rx, ry;
  vfloat2 r;

  q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f((float)M_2_PI)));

  s = d;

  u = vcast_vf_vi2(q);
  s = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Af*0.5f), s);
  s = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Bf*0.5f), s);
  s = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Cf*0.5f), s);
  s = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_Df*0.5f), s);

  t = s;

  s = vmul_vf_vf_vf(s, s);

  u = vcast_vf_f(-0.000195169282960705459117889f);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00833215750753879547119141f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.166666537523269653320312f));

  rx = vmla_vf_vf_vf_vf(vmul_vf_vf_vf(u, s), t, t);
  rx = vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), vcast_vf_f(-0.0f), rx);

  u = vcast_vf_f(-2.71811842367242206819355e-07f);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(2.47990446951007470488548e-05f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.00138888787478208541870117f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0416666641831398010253906f));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.5));

  ry = vmla_vf_vf_vf_vf(s, u, vcast_vf_f(1));

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(0));
  r.x = vsel_vf_vo_vf_vf(o, rx, ry);
  r.y = vsel_vf_vo_vf_vf(o, ry, rx);

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(2));
  r.x = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(r.x)));

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(2)), vcast_vi2_i(2));
  r.y = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(r.y)));

  o = vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAXf));
  r.x = vreinterpret_vf_vm(vandnot_vm_vo32_vm(o, vreinterpret_vm_vf(r.x)));
  r.y = vreinterpret_vf_vm(vandnot_vm_vo32_vm(o, vreinterpret_vm_vf(r.y)));

  o = visinf_vo_vf(d);
  r.x = vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(r.x)));
  r.y = vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(r.y)));

  return r;
}

TYPE2_FUNCATR vfloat2 XSINCOSF_U1(vfloat d) {
  vint2 q;
  vopmask o;
  vfloat u, v, rx, ry;
  vfloat2 r, s, t, x;

  if (vtestallones_i_vo32(vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX2f)))) {
    u = vrint_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(2 * M_1_PI)));
    q = vrint_vi2_vf(u);
    v = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_A2f*0.5f), d);
    s = dfadd2_vf2_vf_vf(v, vmul_vf_vf_vf(u, vcast_vf_f(-PI_B2f*0.5f)));
    s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_C2f*0.5f)));
  } else {
    vfloat2 dfq = dfmul_vf2_vf2_vf(vcast_vf2_f_f(2*M_1_PI, 2*M_1_PI - (float)(2*M_1_PI)), d);
    vfloat t = vrint_vf_vf(vmul_vf_vf_vf(dfq.x, vcast_vf_f(1.0f / (1 << 16))));
    dfq.y = vrint_vf_vf(vadd_vf_vf_vf(vmla_vf_vf_vf_vf(t, vcast_vf_f(-(1 << 16)), dfq.x), dfq.y));
    q = vrint_vi2_vf(dfq.y);
    dfq.x = vmul_vf_vf_vf(t, vcast_vf_f(1 << 16));
    dfq = dfnormalize_vf2_vf2(dfq);

    s = dfadd2_vf2_vf_vf2 (d, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_A3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_B3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_C3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_D3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_E3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
  }

  t = s;

  s.x = dfsqu_vf_vf2(s);

  u = vcast_vf_f(-0.000195169282960705459117889f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.00833215750753879547119141f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(-0.166666537523269653320312f));

  u = vmul_vf_vf_vf(u, vmul_vf_vf_vf(s.x, t.x));

  x = dfadd_vf2_vf2_vf(t, u);
  rx = vadd_vf_vf_vf(x.x, x.y);

  rx = vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), vcast_vf_f(-0.0f), rx);

  u = vcast_vf_f(-2.71811842367242206819355e-07f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(2.47990446951007470488548e-05f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(-0.00138888787478208541870117f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.0416666641831398010253906f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(-0.5));

  x = dfadd_vf2_vf_vf2(vcast_vf_f(1), dfmul_vf2_vf_vf(s.x, u));
  ry = vadd_vf_vf_vf(x.x, x.y);

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(0));
  r.x = vsel_vf_vo_vf_vf(o, rx, ry);
  r.y = vsel_vf_vo_vf_vf(o, ry, rx);

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(2));
  r.x = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(r.x)));

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(2)), vcast_vi2_i(2));
  r.y = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(r.y)));

  o = vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX3f));
  r.x = vreinterpret_vf_vm(vandnot_vm_vo32_vm(o, vreinterpret_vm_vf(r.x)));
  r.y = vreinterpret_vf_vm(vandnot_vm_vo32_vm(o, vreinterpret_vm_vf(r.y)));

  o = visinf_vo_vf(d);
  r.x = vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(r.x)));
  r.y = vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(r.y)));

  return r;
}

TYPE2_FUNCATR vfloat2 XSINCOSPIF_U05(vfloat d) {
  vopmask o;
  vfloat u, s, t, rx, ry;
  vfloat2 r, x, s2;

  u = vmul_vf_vf_vf(d, vcast_vf_f(4));
  vint2 q = vtruncate_vi2_vf(u);
  q = vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, vxor_vi2_vi2_vi2(vsrl_vi2_vi2_i(q, 31), vcast_vi2_i(1))), vcast_vi2_i(~1));
  s = vsub_vf_vf_vf(u, vcast_vf_vi2(q));

  t = s;
  s = vmul_vf_vf_vf(s, s);
  s2 = dfmul_vf2_vf_vf(t, t);

  //

  u = vcast_vf_f(+0.3093842054e-6);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.3657307388e-4));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.2490393585e-2));
  x = dfadd2_vf2_vf_vf2(vmul_vf_vf_vf(u, s), vcast_vf2_f_f(-0.080745510756969451904, -1.3373665339076936258e-09));
  x = dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf2(s2, x), vcast_vf2_f_f(0.78539818525314331055, -2.1857338617566484855e-08));

  x = dfmul_vf2_vf2_vf(x, t);
  rx = vadd_vf_vf_vf(x.x, x.y);

  rx = vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), vcast_vf_f(-0.0f), rx);

  //

  u = vcast_vf_f(-0.2430611801e-7);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.3590577080e-5));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.3259917721e-3));
  x = dfadd2_vf2_vf_vf2(vmul_vf_vf_vf(u, s), vcast_vf2_f_f(0.015854343771934509277, 4.4940051354032242811e-10));
  x = dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf2(s2, x), vcast_vf2_f_f(-0.30842512845993041992, -9.0728339030733922277e-09));

  x = dfadd2_vf2_vf2_vf(dfmul_vf2_vf2_vf2(x, s2), vcast_vf_f(1));
  ry = vadd_vf_vf_vf(x.x, x.y);

  //

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(0));
  r.x = vsel_vf_vo_vf_vf(o, rx, ry);
  r.y = vsel_vf_vo_vf_vf(o, ry, rx);

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(4)), vcast_vi2_i(4));
  r.x = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(r.x)));

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(4)), vcast_vi2_i(4));
  r.y = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(r.y)));

  o = vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAXf));
  r.x = vreinterpret_vf_vm(vandnot_vm_vo32_vm(o, vreinterpret_vm_vf(r.x)));
  r.y = vreinterpret_vf_vm(vandnot_vm_vo32_vm(o, vreinterpret_vm_vf(r.y)));

  o = visinf_vo_vf(d);
  r.x = vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(r.x)));
  r.y = vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(r.y)));

  return r;
}

TYPE2_FUNCATR vfloat2 XSINCOSPIF_U35(vfloat d) {
  vopmask o;
  vfloat u, s, t, rx, ry;
  vfloat2 r;

  u = vmul_vf_vf_vf(d, vcast_vf_f(4));
  vint2 q = vtruncate_vi2_vf(u);
  q = vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, vxor_vi2_vi2_vi2(vsrl_vi2_vi2_i(q, 31), vcast_vi2_i(1))), vcast_vi2_i(~1));
  s = vsub_vf_vf_vf(u, vcast_vf_vi2(q));

  t = s;
  s = vmul_vf_vf_vf(s, s);

  //

  u = vcast_vf_f(-0.3600925265e-4);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.2490088111e-2));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.8074551076e-1));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.7853981853e+0));

  rx = vmul_vf_vf_vf(u, t);

  //

  u = vcast_vf_f(+0.3539815225e-5);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.3259574005e-3));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.1585431583e-1));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(-0.3084251285e+0));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(1));

  ry = u;

  //

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(0));
  r.x = vsel_vf_vo_vf_vf(o, rx, ry);
  r.y = vsel_vf_vo_vf_vf(o, ry, rx);

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(4)), vcast_vi2_i(4));
  r.x = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(r.x)));

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(4)), vcast_vi2_i(4));
  r.y = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(r.y)));

  o = vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAXf));
  r.x = vreinterpret_vf_vm(vandnot_vm_vo32_vm(o, vreinterpret_vm_vf(r.x)));
  r.y = vreinterpret_vf_vm(vandnot_vm_vo32_vm(o, vreinterpret_vm_vf(r.y)));

  o = visinf_vo_vf(d);
  r.x = vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(r.x)));
  r.y = vreinterpret_vf_vm(vor_vm_vo32_vm(o, vreinterpret_vm_vf(r.y)));

  return r;
}

TYPE6_FUNCATR vfloat2 XMODFF(vfloat x) {
  vfloat fr = vsub_vf_vf_vf(x, vcast_vf_vi2(vtruncate_vi2_vf(x)));
  fr = vsel_vf_vo_vf_vf(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(1LL << 23)), vcast_vf_f(0), fr);

  vfloat2 ret;

  ret.x = vcopysign_vf_vf_vf(fr, x);
  ret.y = vcopysign_vf_vf_vf(vsub_vf_vf_vf(x, fr), x);

  return ret;
}

#ifdef ENABLE_GNUABI
EXPORT void xsincosf(vfloat a, float *ps, float *pc) {
  vfloat2 r = sincosfk(a);
  vstoreu_v_p_vf(ps, r.x);
  vstoreu_v_p_vf(pc, r.y);
}

EXPORT void xsincosf_u1(vfloat a, float *ps, float *pc) {
  vfloat2 r = sincosfk_u1(a);
  vstoreu_v_p_vf(ps, r.x);
  vstoreu_v_p_vf(pc, r.y);
}

EXPORT void xsincospif_u05(vfloat a, float *ps, float *pc) {
  vfloat2 r = sincospifk_u05(a);
  vstoreu_v_p_vf(ps, r.x);
  vstoreu_v_p_vf(pc, r.y);
}

EXPORT void xsincospif_u35(vfloat a, float *ps, float *pc) {
  vfloat2 r = sincospifk_u35(a);
  vstoreu_v_p_vf(ps, r.x);
  vstoreu_v_p_vf(pc, r.y);
}

EXPORT CONST vfloat xmodff(vfloat a, float *iptr) {
  vfloat2 r = modffk(a);
  vstoreu_v_p_vf(iptr, r.y);
  return r.x;
}
#endif // #ifdef ENABLE_GNUABI

EXPORT CONST vfloat xtanf_u1(vfloat d) {
  vint2 q;
  vfloat u, v;
  vfloat2 s, t, x;
  vopmask o;

  if (vtestallones_i_vo32(vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX2f)))) {
    u = vrint_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(2 * M_1_PI)));
    q = vrint_vi2_vf(u);
    v = vmla_vf_vf_vf_vf(u, vcast_vf_f(-PI_A2f*0.5f), d);
    s = dfadd2_vf2_vf_vf(v, vmul_vf_vf_vf(u, vcast_vf_f(-PI_B2f*0.5f)));
    s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(u, vcast_vf_f(-PI_C2f*0.5f)));
  } else {
    vfloat2 dfq = dfmul_vf2_vf2_vf(vcast_vf2_f_f(2*M_1_PI, 2*M_1_PI - (float)(2*M_1_PI)), d);
    vfloat t = vrint_vf_vf(vmul_vf_vf_vf(dfq.x, vcast_vf_f(1.0f / (1 << 16))));
    dfq.y = vrint_vf_vf(vadd_vf_vf_vf(vmla_vf_vf_vf_vf(t, vcast_vf_f(-(1 << 16)), dfq.x), dfq.y));
    q = vrint_vi2_vf(dfq.y);
    dfq.x = vmul_vf_vf_vf(t, vcast_vf_f(1 << 16));
    dfq = dfnormalize_vf2_vf2(dfq);

    s = dfadd2_vf2_vf_vf2 (d, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_A3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_B3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_C3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_D3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
    s = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfq, vcast_vf_f(-PI_E3f*0.5f)));
    s = dfnormalize_vf2_vf2(s);
  }

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1));
  vmask n = vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0)));
  s.x = vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(s.x), n));
  s.y = vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(s.y), n));

  t = s;
  s = dfsqu_vf2_vf2(s);
  s = dfnormalize_vf2_vf2(s);

  u = vcast_vf_f(0.00446636462584137916564941f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(-8.3920182078145444393158e-05f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.0109639242291450500488281f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.0212360303848981857299805f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.0540687143802642822265625f));

  x = dfadd_vf2_vf_vf(vcast_vf_f(0.133325666189193725585938f), vmul_vf_vf_vf(u, s.x));
  x = dfadd_vf2_vf_vf2(vcast_vf_f(1), dfmul_vf2_vf2_vf2(dfadd_vf2_vf_vf2(vcast_vf_f(0.33333361148834228515625f), dfmul_vf2_vf2_vf2(s, x)), s));
  x = dfmul_vf2_vf2_vf2(t, x);

  x = vsel_vf2_vo_vf2_vf2(o, dfrec_vf2_vf2(x), x);

  u = vadd_vf_vf_vf(x.x, x.y);

  u = vsel_vf_vo_vf_vf(vandnot_vo_vo_vo(visinf_vo_vf(d),
          vor_vo_vo_vo(visnegzero_vo_vf(d),
                 vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX3f)))),
           vcast_vf_f(-0.0f), u);

  return u;
}

EXPORT CONST vfloat xatanf(vfloat d) {
  vfloat s, t, u;
  vint2 q;

  q = vsel_vi2_vf_vi2(d, vcast_vi2_i(2));
  s = vabs_vf_vf(d);

  q = vsel_vi2_vf_vf_vi2_vi2(vcast_vf_f(1.0f), s, vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)), q);
  s = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(vcast_vf_f(1.0f), s), vrec_vf_vf(s), s);

  t = vmul_vf_vf_vf(s, s);

  u = vcast_vf_f(0.00282363896258175373077393f);
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.0159569028764963150024414f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(0.0425049886107444763183594f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.0748900920152664184570312f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(0.106347933411598205566406f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.142027363181114196777344f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(0.199926957488059997558594f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.333331018686294555664062f));

  t = vmla_vf_vf_vf_vf(s, vmul_vf_vf_vf(t, u), s);

  t = vsel_vf_vo_vf_vf(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(1)), vcast_vi2_i(1)), vsub_vf_vf_vf(vcast_vf_f((float)(M_PI/2)), t), t);

  t = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(2)), vreinterpret_vm_vf(vcast_vf_f(-0.0f))), vreinterpret_vm_vf(t)));

#ifdef ENABLE_NEON32
  t = vsel_vf_vo_vf_vf(visinf_vo_vf(d), vmulsign_vf_vf_vf(vcast_vf_f(1.5874010519681994747517056f), d), t);
#endif

  return t;
}

static INLINE CONST vfloat atan2kf(vfloat y, vfloat x) {
  vfloat s, t, u;
  vint2 q;
  vopmask p;

  q = vsel_vi2_vf_vi2(x, vcast_vi2_i(-2));
  x = vabs_vf_vf(x);

  q = vsel_vi2_vf_vf_vi2_vi2(x, y, vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)), q);
  p = vlt_vo_vf_vf(x, y);
  s = vsel_vf_vo_vf_vf(p, vneg_vf_vf(x), y);
  t = vmax_vf_vf_vf(x, y);

  s = vdiv_vf_vf_vf(s, t);
  t = vmul_vf_vf_vf(s, s);

  u = vcast_vf_f(0.00282363896258175373077393f);
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.0159569028764963150024414f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(0.0425049886107444763183594f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.0748900920152664184570312f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(0.106347933411598205566406f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.142027363181114196777344f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(0.199926957488059997558594f));
  u = vmla_vf_vf_vf_vf(u, t, vcast_vf_f(-0.333331018686294555664062f));

  t = vmla_vf_vf_vf_vf(s, vmul_vf_vf_vf(t, u), s);
  t = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f((float)(M_PI/2)), t);

  return t;
}

static INLINE CONST vfloat visinf2_vf_vf_vf(vfloat d, vfloat m) {
  return vreinterpret_vf_vm(vand_vm_vo32_vm(visinf_vo_vf(d), vor_vm_vm_vm(vsignbit_vm_vf(d), vreinterpret_vm_vf(m))));
}

EXPORT CONST vfloat xatan2f(vfloat y, vfloat x) {
  vfloat r = atan2kf(vabs_vf_vf(y), x);

  r = vmulsign_vf_vf_vf(r, x);
  r = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(x), veq_vo_vf_vf(x, vcast_vf_f(0.0f))), vsub_vf_vf_vf(vcast_vf_f((float)(M_PI/2)), visinf2_vf_vf_vf(x, vmulsign_vf_vf_vf(vcast_vf_f((float)(M_PI/2)), x))), r);
  r = vsel_vf_vo_vf_vf(visinf_vo_vf(y), vsub_vf_vf_vf(vcast_vf_f((float)(M_PI/2)), visinf2_vf_vf_vf(x, vmulsign_vf_vf_vf(vcast_vf_f((float)(M_PI/4)), x))), r);

  r = vsel_vf_vo_vf_vf(veq_vo_vf_vf(y, vcast_vf_f(0.0f)), vreinterpret_vf_vm(vand_vm_vo32_vm(vsignbit_vo_vf(x), vreinterpret_vm_vf(vcast_vf_f((float)M_PI)))), r);

  r = vreinterpret_vf_vm(vor_vm_vo32_vm(vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)), vreinterpret_vm_vf(vmulsign_vf_vf_vf(r, y))));
  return r;
}

EXPORT CONST vfloat xasinf(vfloat d) {
  vopmask o = vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(0.5f));
  vfloat x2 = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, d), vmul_vf_vf_vf(vsub_vf_vf_vf(vcast_vf_f(1), vabs_vf_vf(d)), vcast_vf_f(0.5f)));
  vfloat x = vsel_vf_vo_vf_vf(o, vabs_vf_vf(d), vsqrt_vf_vf(x2)), u;

  u = vcast_vf_f(+0.4197454825e-1);
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.2424046025e-1));
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.4547423869e-1));
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.7495029271e-1));
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.1666677296e+0));
  u = vmla_vf_vf_vf_vf(u, vmul_vf_vf_vf(x, x2), x);

  vfloat r = vsel_vf_vo_vf_vf(o, u, vmla_vf_vf_vf_vf(u, vcast_vf_f(-2), vcast_vf_f(M_PIf/2)));
  return vmulsign_vf_vf_vf(r, d);
}

EXPORT CONST vfloat xacosf(vfloat d) {
  vopmask o = vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(0.5f));
  vfloat x2 = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, d),
				vmul_vf_vf_vf(vsub_vf_vf_vf(vcast_vf_f(1), vabs_vf_vf(d)), vcast_vf_f(0.5f))), u;
  vfloat x = vsel_vf_vo_vf_vf(o, vabs_vf_vf(d), vsqrt_vf_vf(x2));
  x = vsel_vf_vo_vf_vf(veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(1.0f)), vcast_vf_f(0), x);

  u = vcast_vf_f(+0.4197454825e-1);
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.2424046025e-1));
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.4547423869e-1));
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.7495029271e-1));
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.1666677296e+0));
  u = vmul_vf_vf_vf(u, vmul_vf_vf_vf(x2, x));

  vfloat y = vsub_vf_vf_vf(vcast_vf_f(3.1415926535897932f/2), vadd_vf_vf_vf(vmulsign_vf_vf_vf(x, d), vmulsign_vf_vf_vf(u, d)));
  x = vadd_vf_vf_vf(x, u);
  vfloat r = vsel_vf_vo_vf_vf(o, y, vmul_vf_vf_vf(x, vcast_vf_f(2)));
  return vsel_vf_vo_vf_vf(vandnot_vo_vo_vo(o, vlt_vo_vf_vf(d, vcast_vf_f(0))),
			  dfadd_vf2_vf2_vf(vcast_vf2_f_f(3.1415927410125732422f,-8.7422776573475857731e-08f),
					   vneg_vf_vf(r)).x, r);
}

//

static INLINE CONST vfloat2 atan2kf_u1(vfloat2 y, vfloat2 x) {
  vfloat u;
  vfloat2 s, t;
  vint2 q;
  vopmask p;
  vmask r;

  q = vsel_vi2_vf_vf_vi2_vi2(x.x, vcast_vf_f(0), vcast_vi2_i(-2), vcast_vi2_i(0));
  p = vlt_vo_vf_vf(x.x, vcast_vf_f(0));
  r = vand_vm_vo32_vm(p, vreinterpret_vm_vf(vcast_vf_f(-0.0)));
  x.x = vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(x.x), r));
  x.y = vreinterpret_vf_vm(vxor_vm_vm_vm(vreinterpret_vm_vf(x.y), r));

  q = vsel_vi2_vf_vf_vi2_vi2(x.x, y.x, vadd_vi2_vi2_vi2(q, vcast_vi2_i(1)), q);
  p = vlt_vo_vf_vf(x.x, y.x);
  s = vsel_vf2_vo_vf2_vf2(p, dfneg_vf2_vf2(x), y);
  t = vsel_vf2_vo_vf2_vf2(p, y, x);

  s = dfdiv_vf2_vf2_vf2(s, t);
  t = dfsqu_vf2_vf2(s);
  t = dfnormalize_vf2_vf2(t);

  u = vcast_vf_f(-0.00176397908944636583328247f);
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(0.0107900900766253471374512f));
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(-0.0309564601629972457885742f));
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(0.0577365085482597351074219f));
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(-0.0838950723409652709960938f));
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(0.109463557600975036621094f));
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(-0.142626821994781494140625f));
  u = vmla_vf_vf_vf_vf(u, t.x, vcast_vf_f(0.199983194470405578613281f));

  t = dfmul_vf2_vf2_vf2(t, dfadd_vf2_vf_vf(vcast_vf_f(-0.333332866430282592773438f), vmul_vf_vf_vf(u, t.x)));
  t = dfmul_vf2_vf2_vf2(s, dfadd_vf2_vf_vf2(vcast_vf_f(1), t));
  t = dfadd_vf2_vf2_vf2(dfmul_vf2_vf2_vf(vcast_vf2_f_f(1.5707963705062866211f, -4.3711388286737928865e-08f), vcast_vf_vi2(q)), t);

  return t;
}

EXPORT CONST vfloat xatan2f_u1(vfloat y, vfloat x) {
  vopmask o = vlt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(2.9387372783541830947e-39f)); // nexttowardf((1.0 / FLT_MAX), 1)
  x = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(x, vcast_vf_f(1 << 24)), x);
  y = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(y, vcast_vf_f(1 << 24)), y);

  vfloat2 d = atan2kf_u1(vcast_vf2_vf_vf(vabs_vf_vf(y), vcast_vf_f(0)), vcast_vf2_vf_vf(x, vcast_vf_f(0)));
  vfloat r = vadd_vf_vf_vf(d.x, d.y);

  r = vmulsign_vf_vf_vf(r, x);
  r = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(x), veq_vo_vf_vf(x, vcast_vf_f(0))), vsub_vf_vf_vf(vcast_vf_f(M_PI/2), visinf2_vf_vf_vf(x, vmulsign_vf_vf_vf(vcast_vf_f(M_PI/2), x))), r);
  r = vsel_vf_vo_vf_vf(visinf_vo_vf(y), vsub_vf_vf_vf(vcast_vf_f(M_PI/2), visinf2_vf_vf_vf(x, vmulsign_vf_vf_vf(vcast_vf_f(M_PI/4), x))), r);
  r = vsel_vf_vo_vf_vf(veq_vo_vf_vf(y, vcast_vf_f(0.0f)), vreinterpret_vf_vm(vand_vm_vo32_vm(vsignbit_vo_vf(x), vreinterpret_vm_vf(vcast_vf_f((float)M_PI)))), r);

  r = vreinterpret_vf_vm(vor_vm_vo32_vm(vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)), vreinterpret_vm_vf(vmulsign_vf_vf_vf(r, y))));
  return r;
}

EXPORT CONST vfloat xasinf_u1(vfloat d) {
  vopmask o = vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(0.5f));
  vfloat x2 = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, d), vmul_vf_vf_vf(vsub_vf_vf_vf(vcast_vf_f(1), vabs_vf_vf(d)), vcast_vf_f(0.5f))), u;
  vfloat2 x = vsel_vf2_vo_vf2_vf2(o, vcast_vf2_vf_vf(vabs_vf_vf(d), vcast_vf_f(0)), dfsqrt_vf2_vf(x2));
  x = vsel_vf2_vo_vf2_vf2(veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(1.0f)), vcast_vf2_f_f(0, 0), x);

  u = vcast_vf_f(+0.4197454825e-1);
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.2424046025e-1));
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.4547423869e-1));
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.7495029271e-1));
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.1666677296e+0));
  u = vmul_vf_vf_vf(u, vmul_vf_vf_vf(x2, x.x));

  vfloat2 y = dfsub_vf2_vf2_vf(dfsub_vf2_vf2_vf2(vcast_vf2_f_f(3.1415927410125732422f/4,-8.7422776573475857731e-08f/4), x), u);

  vfloat r = vsel_vf_vo_vf_vf(o, vadd_vf_vf_vf(u, x.x),
             vmul_vf_vf_vf(vadd_vf_vf_vf(y.x, y.y), vcast_vf_f(2)));
  return vmulsign_vf_vf_vf(r, d);
}

EXPORT CONST vfloat xacosf_u1(vfloat d) {
  vopmask o = vlt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(0.5f));
  vfloat x2 = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, d), vmul_vf_vf_vf(vsub_vf_vf_vf(vcast_vf_f(1), vabs_vf_vf(d)), vcast_vf_f(0.5f))), u;
  vfloat2 x = vsel_vf2_vo_vf2_vf2(o, vcast_vf2_vf_vf(vabs_vf_vf(d), vcast_vf_f(0)), dfsqrt_vf2_vf(x2));
  x = vsel_vf2_vo_vf2_vf2(veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(1.0f)), vcast_vf2_f_f(0, 0), x);

  u = vcast_vf_f(+0.4197454825e-1);
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.2424046025e-1));
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.4547423869e-1));
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.7495029271e-1));
  u = vmla_vf_vf_vf_vf(u, x2, vcast_vf_f(+0.1666677296e+0));
  u = vmul_vf_vf_vf(u, vmul_vf_vf_vf(x2, x.x));

  vfloat2 y = dfsub_vf2_vf2_vf2(vcast_vf2_f_f(3.1415927410125732422f/2, -8.7422776573475857731e-08f/2),
                                 dfadd_vf2_vf_vf(vmulsign_vf_vf_vf(x.x, d), vmulsign_vf_vf_vf(u, d)));
  x = dfadd_vf2_vf2_vf(x, u);

  y = vsel_vf2_vo_vf2_vf2(o, y, dfscale_vf2_vf2_vf(x, vcast_vf_f(2)));

  y = vsel_vf2_vo_vf2_vf2(vandnot_vo_vo_vo(o, vlt_vo_vf_vf(d, vcast_vf_f(0))),
                          dfsub_vf2_vf2_vf2(vcast_vf2_f_f(3.1415927410125732422f, -8.7422776573475857731e-08f), y), y);

  return vadd_vf_vf_vf(y.x, y.y);
}

EXPORT CONST vfloat xatanf_u1(vfloat d) {
  vfloat2 d2 = atan2kf_u1(vcast_vf2_vf_vf(vabs_vf_vf(d), vcast_vf_f(0)), vcast_vf2_f_f(1, 0));
  vfloat r = vadd_vf_vf_vf(d2.x, d2.y);
  r = vsel_vf_vo_vf_vf(visinf_vo_vf(d), vcast_vf_f(1.570796326794896557998982), r);
  return vmulsign_vf_vf_vf(r, d);
}

//

EXPORT CONST vfloat xlogf(vfloat d) {
  vfloat x, x2, t, m;

#ifndef ENABLE_AVX512F
  vopmask o = vlt_vo_vf_vf(d, vcast_vf_f(FLT_MIN));
  d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f((float)(1LL << 32) * (float)(1LL << 32))), d);
  vint2 e = vilogb2k_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f/0.75f)));
  m = vldexp3_vf_vf_vi2(d, vneg_vi2_vi2(e));
  e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
#else
  vfloat e = vgetexp_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f/0.75f)));
  e = vsel_vf_vo_vf_vf(vispinf_vo_vf(e), vcast_vf_f(128.0f), e);
  m = vgetmant_vf_vf(d);
#endif

  x = vdiv_vf_vf_vf(vadd_vf_vf_vf(vcast_vf_f(-1.0f), m), vadd_vf_vf_vf(vcast_vf_f(1.0f), m));
  x2 = vmul_vf_vf_vf(x, x);

  t = vcast_vf_f(0.2392828464508056640625f);
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.28518211841583251953125f));
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.400005877017974853515625f));
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(0.666666686534881591796875f));
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(2.0f));

#ifndef ENABLE_AVX512F
  x = vmla_vf_vf_vf_vf(x, t, vmul_vf_vf_vf(vcast_vf_f(0.693147180559945286226764f), vcast_vf_vi2(e)));
  x = vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(INFINITYf), x);
  x = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vlt_vo_vf_vf(d, vcast_vf_f(0)), visnan_vo_vf(d)), vcast_vf_f(NANf), x);
  x = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0)), vcast_vf_f(-INFINITYf), x);
#else
  x = vmla_vf_vf_vf_vf(x, t, vmul_vf_vf_vf(vcast_vf_f(0.693147180559945286226764f), e));
  x = vfixup_vf_vf_vf_vi2_i(x, d, vcast_vi2_i((5 << (5*4))), 0);
#endif

  return x;
}

EXPORT CONST vfloat xexpf(vfloat d) {
  vint2 q = vrint_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(R_LN2f)));
  vfloat s, u;

  s = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Uf), d);
  s = vmla_vf_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Lf), s);

  u = vcast_vf_f(0.000198527617612853646278381);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00139304355252534151077271));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.00833336077630519866943359));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.0416664853692054748535156));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.166666671633720397949219));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(0.5));

  u = vadd_vf_vf_vf(vcast_vf_f(1.0f), vmla_vf_vf_vf_vf(vmul_vf_vf_vf(s, s), u, s));

  u = vldexp2_vf_vf_vi2(u, q);

  u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(d, vcast_vf_f(-104)), vreinterpret_vm_vf(u)));
  u = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(vcast_vf_f(100), d), vcast_vf_f(INFINITYf), u);

  return u;
}

#ifdef ENABLE_NEON32
EXPORT CONST vfloat xsqrtf_u35(vfloat d) {
  vfloat e = vreinterpret_vf_vi2(vadd_vi2_vi2_vi2(vcast_vi2_i(0x20000000), vand_vi2_vi2_vi2(vcast_vi2_i(0x7f000000), vsrl_vi2_vi2_i(vreinterpret_vi2_vf(d), 1))));
  vfloat m = vreinterpret_vf_vi2(vadd_vi2_vi2_vi2(vcast_vi2_i(0x3f000000), vand_vi2_vi2_vi2(vcast_vi2_i(0x01ffffff), vreinterpret_vi2_vf(d))));
  float32x4_t x = vrsqrteq_f32(m);
  x = vmulq_f32(x, vrsqrtsq_f32(m, vmulq_f32(x, x)));
  float32x4_t u = vmulq_f32(x, m);
  u = vmlaq_f32(u, vmlsq_f32(m, u, u), vmulq_f32(x, vdupq_n_f32(0.5)));
  e = vreinterpret_vf_vm(vandnot_vm_vo32_vm(veq_vo_vf_vf(d, vcast_vf_f(0)), vreinterpret_vm_vf(e)));
  u = vmul_vf_vf_vf(e, u);

  u = vsel_vf_vo_vf_vf(visinf_vo_vf(d), vcast_vf_f(INFINITYf), u);
  u = vreinterpret_vf_vm(vor_vm_vo32_vm(vor_vo_vo_vo(visnan_vo_vf(d), vlt_vo_vf_vf(d, vcast_vf_f(0))), vreinterpret_vm_vf(u)));
  u = vmulsign_vf_vf_vf(u, d);

  return u;
}
#elif defined(ENABLE_VECEXT)
EXPORT CONST vfloat xsqrtf_u35(vfloat d) {
  vfloat q = vsqrt_vf_vf(d);
  q = vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), vcast_vf_f(-0.0), q);
  return vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(INFINITYf), q);
}
#else
EXPORT CONST vfloat xsqrtf_u35(vfloat d) { return vsqrt_vf_vf(d); }
#endif

EXPORT CONST vfloat xcbrtf(vfloat d) {
  vfloat x, y, q = vcast_vf_f(1.0), t;
  vint2 e, qu, re;

#ifdef ENABLE_AVX512F
  vfloat s = d;
#endif
  e = vadd_vi2_vi2_vi2(vilogbk_vi2_vf(vabs_vf_vf(d)), vcast_vi2_i(1));
  d = vldexp2_vf_vf_vi2(d, vneg_vi2_vi2(e));

  t = vadd_vf_vf_vf(vcast_vf_vi2(e), vcast_vf_f(6144));
  qu = vtruncate_vi2_vf(vmul_vf_vf_vf(t, vcast_vf_f(1.0f/3.0f)));
  re = vtruncate_vi2_vf(vsub_vf_vf_vf(t, vmul_vf_vf_vf(vcast_vf_vi2(qu), vcast_vf_f(3))));

  q = vsel_vf_vo_vf_vf(veq_vo_vi2_vi2(re, vcast_vi2_i(1)), vcast_vf_f(1.2599210498948731647672106f), q);
  q = vsel_vf_vo_vf_vf(veq_vo_vi2_vi2(re, vcast_vi2_i(2)), vcast_vf_f(1.5874010519681994747517056f), q);
  q = vldexp2_vf_vf_vi2(q, vsub_vi2_vi2_vi2(qu, vcast_vi2_i(2048)));

  q = vmulsign_vf_vf_vf(q, d);
  d = vabs_vf_vf(d);

  x = vcast_vf_f(-0.601564466953277587890625f);
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(2.8208892345428466796875f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(-5.532182216644287109375f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(5.898262500762939453125f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(-3.8095417022705078125f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(2.2241256237030029296875f));

  y = vmul_vf_vf_vf(vmul_vf_vf_vf(d, x), x);
  y = vmul_vf_vf_vf(vsub_vf_vf_vf(y, vmul_vf_vf_vf(vmul_vf_vf_vf(vcast_vf_f(2.0f / 3.0f), y), vmla_vf_vf_vf_vf(y, x, vcast_vf_f(-1.0f)))), q);

#ifdef ENABLE_AVX512F
  y = vsel_vf_vo_vf_vf(visinf_vo_vf(s), vmulsign_vf_vf_vf(vcast_vf_f(INFINITYf), s), y);
  y = vsel_vf_vo_vf_vf(veq_vo_vf_vf(s, vcast_vf_f(0)), vmulsign_vf_vf_vf(vcast_vf_f(0), s), y);
#endif

  y = vsel_vf_vo_vf_vf(visnan_vo_vf(d), d, y);

  return y;
}

EXPORT CONST vfloat xcbrtf_u1(vfloat d) {
  vfloat x, y, z, t;
  vfloat2 q2 = vcast_vf2_f_f(1, 0), u, v;
  vint2 e, qu, re;

#ifdef ENABLE_AVX512F
  vfloat s = d;
#endif
  e = vadd_vi2_vi2_vi2(vilogbk_vi2_vf(vabs_vf_vf(d)), vcast_vi2_i(1));
  d = vldexp2_vf_vf_vi2(d, vneg_vi2_vi2(e));

  t = vadd_vf_vf_vf(vcast_vf_vi2(e), vcast_vf_f(6144));
  qu = vtruncate_vi2_vf(vmul_vf_vf_vf(t, vcast_vf_f(1.0/3.0)));
  re = vtruncate_vi2_vf(vsub_vf_vf_vf(t, vmul_vf_vf_vf(vcast_vf_vi2(qu), vcast_vf_f(3))));

  q2 = vsel_vf2_vo_vf2_vf2(veq_vo_vi2_vi2(re, vcast_vi2_i(1)), vcast_vf2_f_f(1.2599210739135742188f, -2.4018701694217270415e-08), q2);
  q2 = vsel_vf2_vo_vf2_vf2(veq_vo_vi2_vi2(re, vcast_vi2_i(2)), vcast_vf2_f_f(1.5874010324478149414f,  1.9520385308169352356e-08), q2);

  q2.x = vmulsign_vf_vf_vf(q2.x, d); q2.y = vmulsign_vf_vf_vf(q2.y, d);
  d = vabs_vf_vf(d);

  x = vcast_vf_f(-0.601564466953277587890625f);
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(2.8208892345428466796875f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(-5.532182216644287109375f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(5.898262500762939453125f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(-3.8095417022705078125f));
  x = vmla_vf_vf_vf_vf(x, d, vcast_vf_f(2.2241256237030029296875f));

  y = vmul_vf_vf_vf(x, x); y = vmul_vf_vf_vf(y, y); x = vsub_vf_vf_vf(x, vmul_vf_vf_vf(vmlanp_vf_vf_vf_vf(d, y, x), vcast_vf_f(-1.0 / 3.0)));

  z = x;

  u = dfmul_vf2_vf_vf(x, x);
  u = dfmul_vf2_vf2_vf2(u, u);
  u = dfmul_vf2_vf2_vf(u, d);
  u = dfadd2_vf2_vf2_vf(u, vneg_vf_vf(x));
  y = vadd_vf_vf_vf(u.x, u.y);

  y = vmul_vf_vf_vf(vmul_vf_vf_vf(vcast_vf_f(-2.0 / 3.0), y), z);
  v = dfadd2_vf2_vf2_vf(dfmul_vf2_vf_vf(z, z), y);
  v = dfmul_vf2_vf2_vf(v, d);
  v = dfmul_vf2_vf2_vf2(v, q2);
  z = vldexp2_vf_vf_vi2(vadd_vf_vf_vf(v.x, v.y), vsub_vi2_vi2_vi2(qu, vcast_vi2_i(2048)));

  z = vsel_vf_vo_vf_vf(visinf_vo_vf(d), vmulsign_vf_vf_vf(vcast_vf_f(INFINITYf), q2.x), z);
  z = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0)), vreinterpret_vf_vm(vsignbit_vm_vf(q2.x)), z);

#ifdef ENABLE_AVX512F
  z = vsel_vf_vo_vf_vf(visinf_vo_vf(s), vmulsign_vf_vf_vf(vcast_vf_f(INFINITYf), s), z);
  z = vsel_vf_vo_vf_vf(veq_vo_vf_vf(s, vcast_vf_f(0)), vmulsign_vf_vf_vf(vcast_vf_f(0), s), z);
#endif

  return z;
}

static INLINE CONST vfloat2 logkf(vfloat d) {
  vfloat2 x, x2;
  vfloat t, m;

#ifndef ENABLE_AVX512F
  vopmask o = vlt_vo_vf_vf(d, vcast_vf_f(FLT_MIN));
  d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f((float)(1LL << 32) * (float)(1LL << 32))), d);
  vint2 e = vilogb2k_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f/0.75f)));
  m = vldexp3_vf_vf_vi2(d, vneg_vi2_vi2(e));
  e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
#else
  vfloat e = vgetexp_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f/0.75f)));
  e = vsel_vf_vo_vf_vf(vispinf_vo_vf(e), vcast_vf_f(128.0f), e);
  m = vgetmant_vf_vf(d);
#endif

  x = dfdiv_vf2_vf2_vf2(dfadd2_vf2_vf_vf(vcast_vf_f(-1), m), dfadd2_vf2_vf_vf(vcast_vf_f(1), m));
  x2 = dfsqu_vf2_vf2(x);

  t = vcast_vf_f(0.240320354700088500976562);
  t = vmla_vf_vf_vf_vf(t, x2.x, vcast_vf_f(0.285112679004669189453125));
  t = vmla_vf_vf_vf_vf(t, x2.x, vcast_vf_f(0.400007992982864379882812));
  vfloat2 c = vcast_vf2_f_f(0.66666662693023681640625f, 3.69183861259614332084311e-09f);

#ifndef ENABLE_AVX512F
  vfloat2 s = dfmul_vf2_vf2_vf(vcast_vf2_f_f(0.69314718246459960938f, -1.904654323148236017e-09f), vcast_vf_vi2(e));
#else
  vfloat2 s = dfmul_vf2_vf2_vf(vcast_vf2_f_f(0.69314718246459960938f, -1.904654323148236017e-09f), e);
#endif

  s = dfadd_vf2_vf2_vf2(s, dfscale_vf2_vf2_vf(x, vcast_vf_f(2)));
  s = dfadd_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf2(dfmul_vf2_vf2_vf2(x2, x),
                                            dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf(x2, t), c)));
  return s;
}

EXPORT CONST vfloat xlogf_u1(vfloat d) {
  vfloat2 x;
  vfloat t, m, x2;

#ifndef ENABLE_AVX512F
  vopmask o = vlt_vo_vf_vf(d, vcast_vf_f(FLT_MIN));
  d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f((float)(1LL << 32) * (float)(1LL << 32))), d);
  vint2 e = vilogb2k_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f/0.75f)));
  m = vldexp3_vf_vf_vi2(d, vneg_vi2_vi2(e));
  e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
  vfloat2 s = dfmul_vf2_vf2_vf(vcast_vf2_f_f(0.69314718246459960938f, -1.904654323148236017e-09f), vcast_vf_vi2(e));
#else
  vfloat e = vgetexp_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0f/0.75f)));
  e = vsel_vf_vo_vf_vf(vispinf_vo_vf(e), vcast_vf_f(128.0f), e);
  m = vgetmant_vf_vf(d);
  vfloat2 s = dfmul_vf2_vf2_vf(vcast_vf2_f_f(0.69314718246459960938f, -1.904654323148236017e-09f), e);
#endif

  x = dfdiv_vf2_vf2_vf2(dfadd2_vf2_vf_vf(vcast_vf_f(-1), m), dfadd2_vf2_vf_vf(vcast_vf_f(1), m));
  x2 = vmul_vf_vf_vf(x.x, x.x);

  t = vcast_vf_f(+0.3027294874e+0f);
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(+0.3996108174e+0f));
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(+0.6666694880e+0f));

  s = dfadd_vf2_vf2_vf2(s, dfscale_vf2_vf2_vf(x, vcast_vf_f(2)));
  s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(vmul_vf_vf_vf(x2, x.x), t));

  vfloat r = vadd_vf_vf_vf(s.x, s.y);

#ifndef ENABLE_AVX512F
  r = vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(INFINITYf), r);
  r = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vlt_vo_vf_vf(d, vcast_vf_f(0)), visnan_vo_vf(d)), vcast_vf_f(NANf), r);
  r = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0)), vcast_vf_f(-INFINITYf), r);
#else
  r = vfixup_vf_vf_vf_vi2_i(r, d, vcast_vi2_i((4 << (2*4)) | (3 << (4*4)) | (5 << (5*4)) | (2 << (6*4))), 0);
#endif

  return r;
}

static INLINE CONST vfloat expkf(vfloat2 d) {
  vfloat u = vmul_vf_vf_vf(vadd_vf_vf_vf(d.x, d.y), vcast_vf_f(R_LN2f));
  vint2 q = vrint_vi2_vf(u);
  vfloat2 s, t;

  s = dfadd2_vf2_vf2_vf(d, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Uf)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Lf)));

  s = dfnormalize_vf2_vf2(s);

  u = vcast_vf_f(0.00136324646882712841033936f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.00836596917361021041870117f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.0416710823774337768554688f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.166665524244308471679688f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(0.499999850988388061523438f));

  t = dfadd_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfsqu_vf2_vf2(s), u));

  t = dfadd_vf2_vf_vf2(vcast_vf_f(1), t);
  u = vadd_vf_vf_vf(t.x, t.y);
  u = vldexp_vf_vf_vi2(u, q);

  u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(d.x, vcast_vf_f(-104)), vreinterpret_vm_vf(u)));

  return u;
}

EXPORT CONST vfloat xpowf(vfloat x, vfloat y) {
#if 1
  vopmask yisint = vor_vo_vo_vo(veq_vo_vf_vf(vtruncate_vf_vf(y), y), vgt_vo_vf_vf(vabs_vf_vf(y), vcast_vf_f(1 << 24)));
  vopmask yisodd = vand_vo_vo_vo(vand_vo_vo_vo(veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vtruncate_vi2_vf(y), vcast_vi2_i(1)), vcast_vi2_i(1)), yisint),
         vlt_vo_vf_vf(vabs_vf_vf(y), vcast_vf_f(1 << 24)));

#ifdef ENABLE_NEON32
  yisodd = vandnot_vm_vo32_vm(visinf_vo_vf(y), yisodd);
#endif

  vfloat result = expkf(dfmul_vf2_vf2_vf(logkf(vabs_vf_vf(x)), y));

  result = vsel_vf_vo_vf_vf(visnan_vo_vf(result), vcast_vf_f(INFINITYf), result);

  result = vmul_vf_vf_vf(result,
       vsel_vf_vo_vf_vf(vgt_vo_vf_vf(x, vcast_vf_f(0)),
            vcast_vf_f(1),
            vsel_vf_vo_vf_vf(yisint, vsel_vf_vo_vf_vf(yisodd, vcast_vf_f(-1.0f), vcast_vf_f(1)), vcast_vf_f(NANf))));

  vfloat efx = vmulsign_vf_vf_vf(vsub_vf_vf_vf(vabs_vf_vf(x), vcast_vf_f(1)), y);

  result = vsel_vf_vo_vf_vf(visinf_vo_vf(y),
          vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(efx, vcast_vf_f(0.0f)),
                  vreinterpret_vm_vf(vsel_vf_vo_vf_vf(veq_vo_vf_vf(efx, vcast_vf_f(0.0f)),
                              vcast_vf_f(1.0f),
                              vcast_vf_f(INFINITYf))))),
          result);

  result = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(x), veq_vo_vf_vf(x, vcast_vf_f(0))),
          vmul_vf_vf_vf(vsel_vf_vo_vf_vf(yisodd, vsign_vf_vf(x), vcast_vf_f(1)),
            vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(vsel_vf_vo_vf_vf(veq_vo_vf_vf(x, vcast_vf_f(0)), vneg_vf_vf(y), y), vcast_vf_f(0)),
                    vreinterpret_vm_vf(vcast_vf_f(INFINITYf))))),
          result);

  result = vreinterpret_vf_vm(vor_vm_vo32_vm(vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)), vreinterpret_vm_vf(result)));

  result = vsel_vf_vo_vf_vf(vor_vo_vo_vo(veq_vo_vf_vf(y, vcast_vf_f(0)), veq_vo_vf_vf(x, vcast_vf_f(1))), vcast_vf_f(1), result);

  return result;
#else
  return expkf(dfmul_vf2_vf2_vf(logkf(x), y));
#endif
}

EXPORT CONST vfloat xpownf(vfloat x, vmask ym) {
    vint2 y = vcast_vi2_vm(ym);
    vfloat res = xpowf(x, vcast_vf_vi2(y));

    vint2 is_odd = vand_vi2_vi2_vi2(y, vcast_vi2_i(1));
    vopmask is_odd_o = vgt_vo_vi2_vi2(is_odd, vcast_vi2_i(0));

    // pown ( -x, odd y) == -res
    vfloat neg = vcopysign_vf_vf_vf(res, vcast_vf_f(-0.0f));

    res = vsel_vf_vo_vf_vf(
              vand_vo_vo_vo(
                vlt_vo_vf_vf(x, vcast_vf_f(0.0f)),
                is_odd_o),
              neg,
              res);

    //pown ( ±0, n ) is ±∞ for odd n < 0.
    //pown ( ±0, n ) is +∞ for even n < 0.
    //pown ( ±0, n ) is +0 for even n > 0.
    //pown ( ±0, n ) is ±0 for odd n > 0.

    vfloat xiszero = vsel_vf_vo_vf_vf(
                  vgt_vo_vi2_vi2(y, vcast_vi2_i(0)),
                  vcast_vf_f(0.0f),
                  vcast_vf_f(INFINITYf));

    vfloat with_sig = vcopysign_vf_vf_vf(xiszero, x);

    xiszero = vsel_vf_vo_vf_vf(is_odd_o, with_sig, xiszero);

    res = vsel_vf_vo_vf_vf(
              veq_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(0.0f)),
              xiszero,
              res);

    // pown ( x, 0 ) is 1 for any x
    res = vsel_vf_vo_vf_vf(
              veq_vo_vi2_vi2(y, vcast_vi2_i(0)),
              vcast_vf_f(1.0f),
              res);

    return res;
}

EXPORT CONST vfloat xpowrf(vfloat x, vfloat y) {
    vfloat res = xpowf(x, y);

    vfloat ax = vabs_vf_vf(x);
    vfloat ay = vabs_vf_vf(y);
    vfloat zeroes = vcast_vf_f(0.0f);

    //powr ( ±0, y ) is +0 for y > 0.
    //powr ( ±0, y ) is +∞ for finite y < 0.
    //powr ( ±0, -∞) is +∞.
    vfloat r_Xzero = vsel_vf_vo_vf_vf(
                       vlt_vo_vf_vf(y, zeroes),
                       vcast_vf_f(INFINITYf),
                       zeroes);
    r_Xzero = vsel_vf_vo_vf_vf(
                veq_vo_vf_vf(y, vcast_vf_f(-INFINITYf)),
                vcast_vf_f(INFINITYf),
                r_Xzero);

    res = vsel_vf_vo_vf_vf(
            veq_vo_vf_vf(ax, zeroes),
            r_Xzero,
            res);

    //powr ( ±0, ±0 ) returns NaN.
    vfloat r_Yzero = vsel_vf_vo_vf_vf(
                        veq_vo_vf_vf(ax, zeroes),
                        vcast_vf_f(NANf),
                        zeroes);
    //powr ( x, ±0 ) is 1 for finite x > 0.
    r_Yzero = vsel_vf_vo_vf_vf(
                vgt_vo_vf_vf(x, zeroes),
                vcast_vf_f(1.0f),
                r_Yzero);

    //powr ( +∞, ±0 ) returns NaN.
    r_Yzero = vsel_vf_vo_vf_vf(
                veq_vo_vf_vf(x, vcast_vf_f(INFINITYf)),
                vcast_vf_f(NANf),
                r_Yzero);

    res = vsel_vf_vo_vf_vf(
            veq_vo_vf_vf(ay, zeroes),
            r_Yzero,
            res);

    //powr ( +1, y ) is 1 for finite y.
    //powr ( +1, ±∞ ) returns NaN.
    vfloat r_Xone = vsel_vf_vo_vf_vf(
                      veq_vo_vf_vf(ay, vcast_vf_f(INFINITYf)),
                      vcast_vf_f(NANf),
                      vcast_vf_f(1.0f));

    res = vsel_vf_vo_vf_vf(
            veq_vo_vf_vf(x, vcast_vf_f(1.0f)),
            r_Xone,
            res);

    //powr ( x, y ) returns NaN for x < 0.
    res = vsel_vf_vo_vf_vf(
            vlt_vo_vf_vf(x, zeroes),
            vcast_vf_f(NANf),
            res);

    //powr ( NaN, y ) returns the NaN
    res = vsel_vf_vo_vf_vf(
            visnan_vo_vf(x),
            x,
            res);

    //powr ( x, NaN ) returns the NaN for x >= 0.
    res = vsel_vf_vo_vf_vf(
            visnan_vo_vf(y),
            y,
            res);
    return res;

}


static INLINE CONST vfloat2 expk2f(vfloat2 d) {
  vfloat u = vmul_vf_vf_vf(vadd_vf_vf_vf(d.x, d.y), vcast_vf_f(R_LN2f));
  vint2 q = vrint_vi2_vf(u);
  vfloat2 s, t;

  s = dfadd2_vf2_vf2_vf(d, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Uf)));
  s = dfadd2_vf2_vf2_vf(s, vmul_vf_vf_vf(vcast_vf_vi2(q), vcast_vf_f(-L2Lf)));

  u = vcast_vf_f(+0.1980960224e-3f);
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(+0.1394256484e-2f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(+0.8333456703e-2f));
  u = vmla_vf_vf_vf_vf(u, s.x, vcast_vf_f(+0.4166637361e-1f));

  t = dfadd2_vf2_vf2_vf(dfmul_vf2_vf2_vf(s, u), vcast_vf_f(+0.166666659414234244790680580464e+0f));
  t = dfadd2_vf2_vf2_vf(dfmul_vf2_vf2_vf2(s, t), vcast_vf_f(0.5));
  t = dfadd2_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf2(dfsqu_vf2_vf2(s), t));

  t = dfadd_vf2_vf_vf2(vcast_vf_f(1), t);

  t.x = vldexp2_vf_vf_vi2(t.x, q);
  t.y = vldexp2_vf_vf_vi2(t.y, q);

  t.x = vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(d.x, vcast_vf_f(-104)), vreinterpret_vm_vf(t.x)));
  t.y = vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(d.x, vcast_vf_f(-104)), vreinterpret_vm_vf(t.y)));

  return t;
}

EXPORT CONST vfloat xsinhf(vfloat x) {
  vfloat y = vabs_vf_vf(x);
  vfloat2 d = expk2f(vcast_vf2_vf_vf(y, vcast_vf_f(0)));
  d = dfsub_vf2_vf2_vf2(d, dfrec_vf2_vf2(d));
  y = vmul_vf_vf_vf(vadd_vf_vf_vf(d.x, d.y), vcast_vf_f(0.5));

  y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(89)),
            visnan_vo_vf(y)), vcast_vf_f(INFINITYf), y);
  y = vmulsign_vf_vf_vf(y, x);
  y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));

  return y;
}

EXPORT CONST vfloat xcoshf(vfloat x) {
  vfloat y = vabs_vf_vf(x);
  vfloat2 d = expk2f(vcast_vf2_vf_vf(y, vcast_vf_f(0)));
  d = dfadd_vf2_vf2_vf2(d, dfrec_vf2_vf2(d));
  y = vmul_vf_vf_vf(vadd_vf_vf_vf(d.x, d.y), vcast_vf_f(0.5));

  y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(89)),
            visnan_vo_vf(y)), vcast_vf_f(INFINITYf), y);
  y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));

  return y;
}

EXPORT CONST vfloat xtanhf(vfloat x) {
  vfloat y = vabs_vf_vf(x);
  vfloat2 d = expk2f(vcast_vf2_vf_vf(y, vcast_vf_f(0)));
  vfloat2 e = dfrec_vf2_vf2(d);
  d = dfdiv_vf2_vf2_vf2(dfadd_vf2_vf2_vf2(d, dfneg_vf2_vf2(e)), dfadd_vf2_vf2_vf2(d, e));
  y = vadd_vf_vf_vf(d.x, d.y);

  y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(8.664339742f)),
            visnan_vo_vf(y)), vcast_vf_f(1.0f), y);
  y = vmulsign_vf_vf_vf(y, x);
  y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));

  return y;
}

static INLINE CONST vfloat2 logk2f(vfloat2 d) {
  vfloat2 x, x2, m, s;
  vfloat t;
  vint2 e;

#ifndef ENABLE_AVX512F
  e = vilogbk_vi2_vf(vmul_vf_vf_vf(d.x, vcast_vf_f(1.0f/0.75f)));
#else
  e = vrint_vi2_vf(vgetexp_vf_vf(vmul_vf_vf_vf(d.x, vcast_vf_f(1.0f/0.75f))));
#endif
  m = dfscale_vf2_vf2_vf(d, vpow2i_vf_vi2(vneg_vi2_vi2(e)));

  x = dfdiv_vf2_vf2_vf2(dfadd2_vf2_vf2_vf(m, vcast_vf_f(-1)), dfadd2_vf2_vf2_vf(m, vcast_vf_f(1)));
  x2 = dfsqu_vf2_vf2(x);

  t = vcast_vf_f(0.2392828464508056640625f);
  t = vmla_vf_vf_vf_vf(t, x2.x, vcast_vf_f(0.28518211841583251953125f));
  t = vmla_vf_vf_vf_vf(t, x2.x, vcast_vf_f(0.400005877017974853515625f));
  t = vmla_vf_vf_vf_vf(t, x2.x, vcast_vf_f(0.666666686534881591796875f));

  s = dfmul_vf2_vf2_vf(vcast_vf2_vf_vf(vcast_vf_f(0.69314718246459960938f), vcast_vf_f(-1.904654323148236017e-09f)), vcast_vf_vi2(e));
  s = dfadd_vf2_vf2_vf2(s, dfscale_vf2_vf2_vf(x, vcast_vf_f(2)));
  s = dfadd_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf(dfmul_vf2_vf2_vf2(x2, x), t));

  return s;
}

EXPORT CONST vfloat xasinhf(vfloat x) {
  vfloat y = vabs_vf_vf(x);
  vopmask o = vgt_vo_vf_vf(y, vcast_vf_f(1));
  vfloat2 d;

  d = vsel_vf2_vo_vf2_vf2(o, dfrec_vf2_vf(x), vcast_vf2_vf_vf(y, vcast_vf_f(0)));
  d = dfsqrt_vf2_vf2(dfadd2_vf2_vf2_vf(dfsqu_vf2_vf2(d), vcast_vf_f(1)));
  d = vsel_vf2_vo_vf2_vf2(o, dfmul_vf2_vf2_vf(d, y), d);

  d = logk2f(dfnormalize_vf2_vf2(dfadd2_vf2_vf2_vf(d, x)));
  y = vadd_vf_vf_vf(d.x, d.y);

  y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(SQRT_FLT_MAX)),
            visnan_vo_vf(y)),
           vmulsign_vf_vf_vf(vcast_vf_f(INFINITYf), x), y);
  y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));
  y = vsel_vf_vo_vf_vf(visnegzero_vo_vf(x), vcast_vf_f(-0.0), y);

  return y;
}

EXPORT CONST vfloat xacoshf(vfloat x) {
  vfloat2 d = logk2f(dfadd2_vf2_vf2_vf(dfmul_vf2_vf2_vf2(dfsqrt_vf2_vf2(dfadd2_vf2_vf_vf(x, vcast_vf_f(1))), dfsqrt_vf2_vf2(dfadd2_vf2_vf_vf(x, vcast_vf_f(-1)))), x));
  vfloat y = vadd_vf_vf_vf(d.x, d.y);

  y = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vgt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(SQRT_FLT_MAX)),
            visnan_vo_vf(y)),
           vcast_vf_f(INFINITYf), y);

  y = vreinterpret_vf_vm(vandnot_vm_vo32_vm(veq_vo_vf_vf(x, vcast_vf_f(1.0f)), vreinterpret_vm_vf(y)));

  y = vreinterpret_vf_vm(vor_vm_vo32_vm(vlt_vo_vf_vf(x, vcast_vf_f(1.0f)), vreinterpret_vm_vf(y)));
  y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));

  return y;
}

EXPORT CONST vfloat xatanhf(vfloat x) {
  vfloat y = vabs_vf_vf(x);
  vfloat2 d = logk2f(dfdiv_vf2_vf2_vf2(dfadd2_vf2_vf_vf(vcast_vf_f(1), y), dfadd2_vf2_vf_vf(vcast_vf_f(1), vneg_vf_vf(y))));
  y = vreinterpret_vf_vm(vor_vm_vo32_vm(vgt_vo_vf_vf(y, vcast_vf_f(1.0)), vreinterpret_vm_vf(vsel_vf_vo_vf_vf(veq_vo_vf_vf(y, vcast_vf_f(1.0)), vcast_vf_f(INFINITYf), vmul_vf_vf_vf(vadd_vf_vf_vf(d.x, d.y), vcast_vf_f(0.5))))));

  y = vreinterpret_vf_vm(vor_vm_vo32_vm(vor_vo_vo_vo(visinf_vo_vf(x), visnan_vo_vf(y)), vreinterpret_vm_vf(y)));
  y = vmulsign_vf_vf_vf(y, x);
  y = vreinterpret_vf_vm(vor_vm_vo32_vm(visnan_vo_vf(x), vreinterpret_vm_vf(y)));

  return y;
}

EXPORT CONST vfloat xexp2f(vfloat d) {
  vfloat u = vrint_vf_vf(d), s;
  vint2 q = vrint_vi2_vf(u);

  s = vsub_vf_vf_vf(d, u);

  u = vcast_vf_f(+0.1535920892e-3);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.1339262701e-2));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.9618384764e-2));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.5550347269e-1));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.2402264476e+0));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.6931471825e+0));

#ifdef ENABLE_FMA_SP
  u = vfma_vf_vf_vf_vf(u, s, vcast_vf_f(1));
#else
  u = dfnormalize_vf2_vf2(dfadd_vf2_vf_vf2(vcast_vf_f(1), dfmul_vf2_vf_vf(u, s))).x;
#endif
  
  u = vldexp2_vf_vf_vi2(u, q);

  u = vsel_vf_vo_vf_vf(vge_vo_vf_vf(d, vcast_vf_f(128)), vcast_vf_f(INFINITY), u);
  u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(d, vcast_vf_f(-150)), vreinterpret_vm_vf(u)));

  return u;
}

EXPORT CONST vfloat xexp10f(vfloat d) {
  vfloat u = vrint_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(LOG10_2))), s;
  vint2 q = vrint_vi2_vf(u);

  s = vmla_vf_vf_vf_vf(u, vcast_vf_f(-L10Uf), d);
  s = vmla_vf_vf_vf_vf(u, vcast_vf_f(-L10Lf), s);

  u = vcast_vf_f(+0.2064004987e+0);
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.5417877436e+0));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.1171286821e+1));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.2034656048e+1));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.2650948763e+1));
  u = vmla_vf_vf_vf_vf(u, s, vcast_vf_f(+0.2302585125e+1));

#ifdef ENABLE_FMA_SP
  u = vfma_vf_vf_vf_vf(u, s, vcast_vf_f(1));
#else
  u = dfnormalize_vf2_vf2(dfadd_vf2_vf_vf2(vcast_vf_f(1), dfmul_vf2_vf_vf(u, s))).x;
#endif
  
  u = vldexp2_vf_vf_vi2(u, q);

  u = vsel_vf_vo_vf_vf(vgt_vo_vf_vf(d, vcast_vf_f(38.5318394191036238941387f)), vcast_vf_f(INFINITYf), u);
  u = vreinterpret_vf_vm(vandnot_vm_vo32_vm(vlt_vo_vf_vf(d, vcast_vf_f(-50)), vreinterpret_vm_vf(u)));

  return u;
}

EXPORT CONST vfloat xexpm1f(vfloat a) {
  vfloat2 d = dfadd2_vf2_vf2_vf(expk2f(vcast_vf2_vf_vf(a, vcast_vf_f(0))), vcast_vf_f(-1.0));
  vfloat x = vadd_vf_vf_vf(d.x, d.y);
  x = vsel_vf_vo_vf_vf(vgt_vo_vf_vf(a, vcast_vf_f(88.72283172607421875f)), vcast_vf_f(INFINITYf), x);
  x = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(a, vcast_vf_f(-16.635532333438687426013570f)), vcast_vf_f(-1), x);
  x = vsel_vf_vo_vf_vf(visnegzero_vo_vf(a), vcast_vf_f(-0.0f), x);
  return x;
}

EXPORT CONST vfloat xlog10f(vfloat d) {
  vfloat2 x;
  vfloat t, m, x2;

#ifndef ENABLE_AVX512F
  vopmask o = vlt_vo_vf_vf(d, vcast_vf_f(FLT_MIN));
  d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f((float)(1LL << 32) * (float)(1LL << 32))), d);
  vint2 e = vilogb2k_vi2_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0/0.75)));
  m = vldexp3_vf_vf_vi2(d, vneg_vi2_vi2(e));
  e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
#else
  vfloat e = vgetexp_vf_vf(vmul_vf_vf_vf(d, vcast_vf_f(1.0/0.75)));
  e = vsel_vf_vo_vf_vf(vispinf_vo_vf(e), vcast_vf_f(128.0f), e);
  m = vgetmant_vf_vf(d);
#endif

  x = dfdiv_vf2_vf2_vf2(dfadd2_vf2_vf_vf(vcast_vf_f(-1), m), dfadd2_vf2_vf_vf(vcast_vf_f(1), m));
  x2 = vmul_vf_vf_vf(x.x, x.x);

  t = vcast_vf_f(+0.1314289868e+0);
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f( +0.1735493541e+0));
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f( +0.2895309627e+0));
  
#ifndef ENABLE_AVX512F
  vfloat2 s = dfmul_vf2_vf2_vf(vcast_vf2_f_f(0.30103001, -1.432098889e-08), vcast_vf_vi2(e));
#else
  vfloat2 s = dfmul_vf2_vf2_vf(vcast_vf2_f_f(0.30103001, -1.432098889e-08), e);
#endif

  s = dfadd_vf2_vf2_vf2(s, dfmul_vf2_vf2_vf2(x, vcast_vf2_f_f(0.868588984, -2.170757285e-08)));
  s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(vmul_vf_vf_vf(x2, x.x), t));

  vfloat r = vadd_vf_vf_vf(s.x, s.y);

#ifndef ENABLE_AVX512F
  r = vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(INFINITY), r);
  r = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vlt_vo_vf_vf(d, vcast_vf_f(0)), visnan_vo_vf(d)), vcast_vf_f(NAN), r);
  r = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0)), vcast_vf_f(-INFINITY), r);
#else
  r = vfixup_vf_vf_vf_vi2_i(r, d, vcast_vi2_i((4 << (2*4)) | (3 << (4*4)) | (5 << (5*4)) | (2 << (6*4))), 0);
#endif
  
  return r;
}

EXPORT CONST vfloat xlog1pf_fast(vfloat d) {
  vfloat2 x;
  vfloat t, m, x2;

  vfloat dp1 = vadd_vf_vf_vf(d, vcast_vf_f(1));

#ifndef ENABLE_AVX512F
  vopmask o = vlt_vo_vf_vf(dp1, vcast_vf_f(FLT_MIN));
  dp1 = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(dp1, vcast_vf_f((float)(1LL << 32) * (float)(1LL << 32))), dp1);
  vint2 e = vilogb2k_vi2_vf(vmul_vf_vf_vf(dp1, vcast_vf_f(1.0f/0.75f)));
  t = vldexp3_vf_vf_vi2(vcast_vf_f(1), vneg_vi2_vi2(e));
  m = vmla_vf_vf_vf_vf(d, t, vsub_vf_vf_vf(t, vcast_vf_f(1)));
  e = vsel_vi2_vo_vi2_vi2(o, vsub_vi2_vi2_vi2(e, vcast_vi2_i(64)), e);
  vfloat2 s = dfmul_vf2_vf2_vf(vcast_vf2_f_f(0.69314718246459960938f, -1.904654323148236017e-09f), vcast_vf_vi2(e));
#else
  vfloat e = vgetexp_vf_vf(vmul_vf_vf_vf(dp1, vcast_vf_f(1.0f/0.75f)));
  e = vsel_vf_vo_vf_vf(vispinf_vo_vf(e), vcast_vf_f(128.0f), e);
  t = vldexp3_vf_vf_vi2(vcast_vf_f(1), vneg_vi2_vi2(vrint_vi2_vf(e)));
  m = vmla_vf_vf_vf_vf(d, t, vsub_vf_vf_vf(t, vcast_vf_f(1)));
  vfloat2 s = dfmul_vf2_vf2_vf(vcast_vf2_f_f(0.69314718246459960938f, -1.904654323148236017e-09f), e);
#endif

  x = dfdiv_vf2_vf2_vf2(vcast_vf2_vf_vf(m, vcast_vf_f(0)), dfadd_vf2_vf_vf(vcast_vf_f(2), m));
  x2 = vmul_vf_vf_vf(x.x, x.x);

  t = vcast_vf_f(+0.3027294874e+0f);
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(+0.3996108174e+0f));
  t = vmla_vf_vf_vf_vf(t, x2, vcast_vf_f(+0.6666694880e+0f));

  s = dfadd_vf2_vf2_vf2(s, dfscale_vf2_vf2_vf(x, vcast_vf_f(2)));
  s = dfadd_vf2_vf2_vf(s, vmul_vf_vf_vf(vmul_vf_vf_vf(x2, x.x), t));

  vfloat r = vadd_vf_vf_vf(s.x, s.y);

  r = vsel_vf_vo_vf_vf(vgt_vo_vf_vf(d, vcast_vf_f(1e+38)), vcast_vf_f(INFINITYf), r);
  r = vreinterpret_vf_vm(vor_vm_vo32_vm(vgt_vo_vf_vf(vcast_vf_f(-1), d), vreinterpret_vm_vf(r)));
  r = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(-1)), vcast_vf_f(-INFINITYf), r);
  r = vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), vcast_vf_f(-0.0f), r);

  return r;
}

EXPORT CONST vfloat xlog1pf(vfloat a) {
  vfloat log1_small = xlog1pf_fast(a);

  vfloat cutoff = vcast_vf_f( (float)1.0e23 );
  if (vall_lte32_i_vf_vf(a, cutoff))
    return log1_small;

  vopmask gt_cutoff = vgt_vo_vf_vf(a, cutoff);
  vfloat log1_big = xlogf(a);
  return vsel_vf_vo_vf_vf(gt_cutoff, log1_big, log1_small);
}

//

EXPORT CONST vfloat xfabsf(vfloat x) { return vabs_vf_vf(x); }

EXPORT CONST vfloat xcopysignf(vfloat x, vfloat y) { return vcopysign_vf_vf_vf(x, y); }

EXPORT CONST vfloat xfmaxf(vfloat x, vfloat y) { return vmaxnum_vf_vf_vf(x, y); }

EXPORT CONST vfloat xfminf(vfloat x, vfloat y) { return vminnum_vf_vf_vf(x, y); }

EXPORT CONST vfloat xfdimf(vfloat x, vfloat y) {
  vfloat ret = vsub_vf_vf_vf(x, y);
  ret = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vlt_vo_vf_vf(ret, vcast_vf_f(0)), veq_vo_vf_vf(x, y)), vcast_vf_f(0), ret);
  return ret;
}

EXPORT CONST vfloat xtruncf(vfloat x) {
  vfloat fr = vsub_vf_vf_vf(x, vcast_vf_vi2(vtruncate_vi2_vf(x)));
  return vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(x), vge_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(1LL << 23))), x, vcopysign_vf_vf_vf(vsub_vf_vf_vf(x, fr), x));
}

EXPORT CONST vfloat xfloorf(vfloat x) {
  vfloat fr = vsub_vf_vf_vf(x, vcast_vf_vi2(vtruncate_vi2_vf(x)));
  fr = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(fr, vcast_vf_f(0)), vadd_vf_vf_vf(fr, vcast_vf_f(1.0f)), fr);
  return vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(x), vge_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(1LL << 23))), x, vcopysign_vf_vf_vf(vsub_vf_vf_vf(x, fr), x));
}

EXPORT CONST vfloat xceilf(vfloat x) {
  vfloat fr = vsub_vf_vf_vf(x, vcast_vf_vi2(vtruncate_vi2_vf(x)));
  fr = vsel_vf_vo_vf_vf(vle_vo_vf_vf(fr, vcast_vf_f(0)), fr, vsub_vf_vf_vf(fr, vcast_vf_f(1.0f)));
  return vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(x), vge_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(1LL << 23))), x, vcopysign_vf_vf_vf(vsub_vf_vf_vf(x, fr), x));
}

EXPORT CONST vfloat xroundf(vfloat d) {
  vfloat x = vadd_vf_vf_vf(d, vcast_vf_f(0.5f));
  vfloat fr = vsub_vf_vf_vf(x, vcast_vf_vi2(vtruncate_vi2_vf(x)));
  x = vsel_vf_vo_vf_vf(vand_vo_vo_vo(vle_vo_vf_vf(x, vcast_vf_f(0)), veq_vo_vf_vf(fr, vcast_vf_f(0))), vsub_vf_vf_vf(x, vcast_vf_f(1.0f)), x);
  fr = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(fr, vcast_vf_f(0)), vadd_vf_vf_vf(fr, vcast_vf_f(1.0f)), fr);
  x = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0.4999999701976776123f)), vcast_vf_f(0), x);
  return vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(d), vge_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(1LL << 23))), d, vcopysign_vf_vf_vf(vsub_vf_vf_vf(x, fr), d));
}

EXPORT CONST vfloat xrintf(vfloat d) {
  vfloat x = vadd_vf_vf_vf(d, vcast_vf_f(0.5f));
  vopmask isodd = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vcast_vi2_i(1), vtruncate_vi2_vf(x)), vcast_vi2_i(1));
  vfloat fr = vsub_vf_vf_vf(x, vcast_vf_vi2(vtruncate_vi2_vf(x)));
  fr = vsel_vf_vo_vf_vf(vor_vo_vo_vo(vlt_vo_vf_vf(fr, vcast_vf_f(0)), vand_vo_vo_vo(veq_vo_vf_vf(fr, vcast_vf_f(0)), isodd)), vadd_vf_vf_vf(fr, vcast_vf_f(1.0f)), fr);
  x = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0.50000005960464477539f)), vcast_vf_f(0), x);
  vfloat ret = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visinf_vo_vf(d), vge_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(1LL << 23))), d, vcopysign_vf_vf_vf(vsub_vf_vf_vf(x, fr), d));
  return ret;
}

EXPORT CONST vfloat xfmaf(vfloat x, vfloat y, vfloat z) {
#ifdef ENABLE_FMA_SP
  return vmla_vf_vf_vf_vf(x, y, z);
#else
  vfloat h2 = vadd_vf_vf_vf(vmul_vf_vf_vf(x, y), z), q = vcast_vf_f(1);
  vopmask o = vlt_vo_vf_vf(vabs_vf_vf(h2), vcast_vf_f(1e-38f));
  {
    const float c0 = 1ULL << 25, c1 = c0 * c0, c2 = c1 * c1;
    x = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(x, vcast_vf_f(c1)), x);
    y = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(y, vcast_vf_f(c1)), y);
    z = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(z, vcast_vf_f(c2)), z);
    q = vsel_vf_vo_vf_vf(o, vcast_vf_f(1.0f / c2), q);
  }
  o = vgt_vo_vf_vf(vabs_vf_vf(h2), vcast_vf_f(1e+38f));
  {
    const float c0 = 1ULL << 25, c1 = c0 * c0, c2 = c1 * c1;
    x = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(x, vcast_vf_f(1.0f / c1)), x);
    y = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(y, vcast_vf_f(1.0f / c1)), y);
    z = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(z, vcast_vf_f(1.0f / c2)), z);
    q = vsel_vf_vo_vf_vf(o, vcast_vf_f(c2), q);
  }
  vfloat2 d = dfmul_vf2_vf_vf(x, y);
  d = dfadd2_vf2_vf2_vf(d, z);
  vfloat ret = vsel_vf_vo_vf_vf(vor_vo_vo_vo(veq_vo_vf_vf(x, vcast_vf_f(0)), veq_vo_vf_vf(y, vcast_vf_f(0))), z, vadd_vf_vf_vf(d.x, d.y));
  o = visinf_vo_vf(z);
  o = vandnot_vo_vo_vo(visinf_vo_vf(x), o);
  o = vandnot_vo_vo_vo(visnan_vo_vf(x), o);
  o = vandnot_vo_vo_vo(visinf_vo_vf(y), o);
  o = vandnot_vo_vo_vo(visnan_vo_vf(y), o);
  h2 = vsel_vf_vo_vf_vf(o, z, h2);

  o = vor_vo_vo_vo(visinf_vo_vf(h2), visnan_vo_vf(h2));

  return vsel_vf_vo_vf_vf(o, h2, vmul_vf_vf_vf(ret, q));
#endif
}

static INLINE CONST vint2 vcast_vi2_i_i(int i0, int i1) { return vcast_vi2_vm(vcast_vm_i_i(i0, i1)); }

EXPORT CONST vfloat xsqrtf_u05(vfloat d) {
#if 1
  return vsqrt_vf_vf(d);
#else
  vfloat q;
  vopmask o;

  d = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(d, vcast_vf_f(0)), vcast_vf_f(NANf), d);

  o = vlt_vo_vf_vf(d, vcast_vf_f(5.2939559203393770e-23f));
  d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f(1.8889465931478580e+22f)), d);
  q = vsel_vf_vo_vf_vf(o, vcast_vf_f(7.2759576141834260e-12f*0.5f), vcast_vf_f(0.5f));

  o = vgt_vo_vf_vf(d, vcast_vf_f(1.8446744073709552e+19f));
  d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f(5.4210108624275220e-20f)), d);
  q = vsel_vf_vo_vf_vf(o, vcast_vf_f(4294967296.0f * 0.5f), q);

  vfloat x = vreinterpret_vf_vi2(vsub_vi2_vi2_vi2(vcast_vi2_i(0x5f375a86), vsrl_vi2_vi2_i(vreinterpret_vi2_vf(vadd_vf_vf_vf(d, vcast_vf_f(1e-45f))), 1)));

  x = vmul_vf_vf_vf(x, vsub_vf_vf_vf(vcast_vf_f(1.5f), vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(vcast_vf_f(0.5f), d), x), x)));
  x = vmul_vf_vf_vf(x, vsub_vf_vf_vf(vcast_vf_f(1.5f), vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(vcast_vf_f(0.5f), d), x), x)));
  x = vmul_vf_vf_vf(x, vsub_vf_vf_vf(vcast_vf_f(1.5f), vmul_vf_vf_vf(vmul_vf_vf_vf(vmul_vf_vf_vf(vcast_vf_f(0.5f), d), x), x)));
  x = vmul_vf_vf_vf(x, d);

  vfloat2 d2 = dfmul_vf2_vf2_vf2(dfadd2_vf2_vf_vf2(d, dfmul_vf2_vf_vf(x, x)), dfrec_vf2_vf(x));

  x = vmul_vf_vf_vf(vadd_vf_vf_vf(d2.x, d2.y), q);

  x = vsel_vf_vo_vf_vf(vispinf_vo_vf(d), vcast_vf_f(INFINITYf), x);
  x = vsel_vf_vo_vf_vf(veq_vo_vf_vf(d, vcast_vf_f(0)), d, x);

  return x;
#endif
}

EXPORT CONST vfloat xhypotf_u05(vfloat x, vfloat y) {
  x = vabs_vf_vf(x);
  y = vabs_vf_vf(y);
  vfloat min = vmin_vf_vf_vf(x, y), n = min;
  vfloat max = vmax_vf_vf_vf(x, y), d = max;

  vopmask o = vlt_vo_vf_vf(max, vcast_vf_f(FLT_MIN));
  n = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(n, vcast_vf_f(1ULL << 24)), n);
  d = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(d, vcast_vf_f(1ULL << 24)), d);

  vfloat2 t = dfdiv_vf2_vf2_vf2(vcast_vf2_vf_vf(n, vcast_vf_f(0)), vcast_vf2_vf_vf(d, vcast_vf_f(0)));
  t = dfmul_vf2_vf2_vf(dfsqrt_vf2_vf2(dfadd2_vf2_vf2_vf(dfsqu_vf2_vf2(t), vcast_vf_f(1))), max);
  vfloat ret = vadd_vf_vf_vf(t.x, t.y);
  ret = vsel_vf_vo_vf_vf(visnan_vo_vf(ret), vcast_vf_f(INFINITYf), ret);
  ret = vsel_vf_vo_vf_vf(veq_vo_vf_vf(min, vcast_vf_f(0)), max, ret);
  ret = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)), vcast_vf_f(NANf), ret);
  ret = vsel_vf_vo_vf_vf(vor_vo_vo_vo(veq_vo_vf_vf(x, vcast_vf_f(INFINITYf)), veq_vo_vf_vf(y, vcast_vf_f(INFINITYf))), vcast_vf_f(INFINITYf), ret);

  return ret;
}

EXPORT CONST vfloat xhypotf_u35(vfloat x, vfloat y) {
  x = vabs_vf_vf(x);
  y = vabs_vf_vf(y);
  vfloat min = vmin_vf_vf_vf(x, y), n = min;
  vfloat max = vmax_vf_vf_vf(x, y), d = max;

  vfloat t = vdiv_vf_vf_vf(min, max);
  vfloat ret = vmul_vf_vf_vf(max, vsqrt_vf_vf(vmla_vf_vf_vf_vf(t, t, vcast_vf_f(1))));
  ret = vsel_vf_vo_vf_vf(veq_vo_vf_vf(min, vcast_vf_f(0)), max, ret);
  ret = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)), vcast_vf_f(NANf), ret);
  ret = vsel_vf_vo_vf_vf(vor_vo_vo_vo(veq_vo_vf_vf(x, vcast_vf_f(INFINITYf)), veq_vo_vf_vf(y, vcast_vf_f(INFINITYf))), vcast_vf_f(INFINITYf), ret);

  return ret;
}

EXPORT CONST vfloat xnextafterf(vfloat x, vfloat y) {
  x = vsel_vf_vo_vf_vf(veq_vo_vf_vf(x, vcast_vf_f(0)), vmulsign_vf_vf_vf(vcast_vf_f(0), y), x);
  vint2 t, xi2 = vreinterpret_vi2_vf(x);
  vopmask c = vxor_vo_vo_vo(vsignbit_vo_vf(x), vge_vo_vf_vf(y, x));

  xi2 = vsel_vi2_vo_vi2_vi2(c, vsub_vi2_vi2_vi2(vcast_vi2_i(0), vxor_vi2_vi2_vi2(xi2, vcast_vi2_i(1 << 31))), xi2);

  xi2 = vsel_vi2_vo_vi2_vi2(vneq_vo_vf_vf(x, y), vsub_vi2_vi2_vi2(xi2, vcast_vi2_i(1)), xi2);

  xi2 = vsel_vi2_vo_vi2_vi2(c, vsub_vi2_vi2_vi2(vcast_vi2_i(0), vxor_vi2_vi2_vi2(xi2, vcast_vi2_i(1 << 31))), xi2);

  vfloat ret = vreinterpret_vf_vi2(xi2);

  ret = vsel_vf_vo_vf_vf(vand_vo_vo_vo(veq_vo_vf_vf(ret, vcast_vf_f(0)), vneq_vo_vf_vf(x, vcast_vf_f(0))),
       vmulsign_vf_vf_vf(vcast_vf_f(0), x), ret);

  ret = vsel_vf_vo_vf_vf(vand_vo_vo_vo(veq_vo_vf_vf(x, vcast_vf_f(0)), veq_vo_vf_vf(y, vcast_vf_f(0))), y, ret);

  ret = vsel_vf_vo_vf_vf(vor_vo_vo_vo(visnan_vo_vf(x), visnan_vo_vf(y)), vcast_vf_f(NANf), ret);

  return ret;
}

EXPORT CONST vfloat xfrfrexpf(vfloat x) {
  vfloat j = x;
  x = vsel_vf_vo_vf_vf(
        vlt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(FLT_MIN)),
        vmul_vf_vf_vf(x, vcast_vf_f((float)(1U << 30))),
        x);


  vmask xm = vreinterpret_vm_vf(x);
  xm = vand_vm_vm_vm(xm, vcast_vm_i_i(~0x7f800000U, ~0x7f800000U));
  xm = vor_vm_vm_vm (xm, vcast_vm_i_i( 0x3f000000U,  0x3f000000U));

  vfloat ret = vreinterpret_vf_vm(xm);

  ret = vsel_vf_vo_vf_vf(visinf_vo_vf(x),
                         vmulsign_vf_vf_vf(vcast_vf_f(INFINITYf), x),
                         ret);
  ret = vsel_vf_vo_vf_vf(veq_vo_vf_vf(x, vcast_vf_f(0.0f)), x, ret);

  ret = vsel_vf_vo_vf_vf(visnan_vo_vf(j), j, ret);

  return ret;
}

EXPORT CONST vmask xexpfrexpf(vfloat x) {
  vopmask isnan = vor_vo_vo_vo(visinf_vo_vf(x), visnan_vo_vf(x));
  vfloat mul = vmul_vf_vf_vf(x, vcast_vf_f(0x1p+30));   //(float)(1U << 30)
  vopmask is_denorm = vlt_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(0x1p-126)); //FLT_MIN

  x = vsel_vf_vo_vf_vf(is_denorm, mul, x);
  const vint2 zeros = vcast_vi2_i(0);
  vint2 correct = vsel_vi2_vo_vi2_vi2(is_denorm, vcast_vi2_i(-30), zeros);

  vint2 ret = vreinterpret_vi2_vf(x);

  ret = vsrl_vi2_vi2_i(ret, 23);
  ret = vand_vi2_vi2_vi2(ret, vcast_vi2_i(0xff));
  ret = vsub_vi2_vi2_vi2(ret, vcast_vi2_i(0x7e));
  ret = vadd_vi2_vi2_vi2(ret, correct);

  ret = vsel_vi2_vo_vi2_vi2(
            veq_vo_vf_vf(x, vreinterpret_vf_vi2(zeros)),
            zeros,
            ret);

  ret = vsel_vi2_vo_vi2_vi2(isnan, zeros, ret);

  return vcast_vm_vi2(ret);;
}

static INLINE CONST vfloat vtoward0f(vfloat x) {
  vfloat t = vreinterpret_vf_vi2(vsub_vi2_vi2_vi2(vreinterpret_vi2_vf(x), vcast_vi2_i(1)));
  return vsel_vf_vo_vf_vf(veq_vo_vf_vf(x, vcast_vf_f(0)), vcast_vf_f(0), t);
}

static INLINE CONST vfloat vptruncf(vfloat x) {
#ifdef FULL_FP_ROUNDING
  return vtruncate_vf_vf(x);
#else
  vfloat fr = vsub_vf_vf_vf(x, vcast_vf_vi2(vtruncate_vi2_vf(x)));
  return vsel_vf_vo_vf_vf(vge_vo_vf_vf(vabs_vf_vf(x), vcast_vf_f(1LL << 23)), x, vsub_vf_vf_vf(x, fr));
#endif
}

EXPORT CONST vfloat xfmodf(vfloat x, vfloat y) {
  vfloat nu = vabs_vf_vf(x), de = vabs_vf_vf(y), s = vcast_vf_f(1), q;
  vopmask o = vlt_vo_vf_vf(de, vcast_vf_f(FLT_MIN));
  nu = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(nu, vcast_vf_f(1ULL << 25)), nu);
  de = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(de, vcast_vf_f(1ULL << 25)), de);
  s  = vsel_vf_vo_vf_vf(o, vmul_vf_vf_vf(s , vcast_vf_f(1.0f / (1ULL << 25))), s);
  vfloat rde = vtoward0f(vrec_vf_vf(de));
#ifdef ENABLE_NEON32
  rde = vtoward0f(rde);
#endif
  vfloat2 r = vcast_vf2_vf_vf(nu, vcast_vf_f(0));

  for(int i=0;i<8;i++) { // ceil(log2(FLT_MAX) / 22)+1
    q = vsel_vf_vo_vf_vf(vand_vo_vo_vo(vgt_vo_vf_vf(vadd_vf_vf_vf(de, de), r.x),
                                       vge_vo_vf_vf(r.x, de)),
                         vcast_vf_f(1), vmul_vf_vf_vf(vtoward0f(r.x), rde));
    r = dfnormalize_vf2_vf2(dfadd2_vf2_vf2_vf2(r, dfmul_vf2_vf_vf(vptruncf(q), vneg_vf_vf(de))));
    if (vtestallones_i_vo32(vlt_vo_vf_vf(r.x, de))) break;
  }

  vfloat ret = vmul_vf_vf_vf(vadd_vf_vf_vf(r.x, r.y), s);
  ret = vsel_vf_vo_vf_vf(veq_vo_vf_vf(vadd_vf_vf_vf(r.x, r.y), de), vcast_vf_f(0), ret);

  ret = vmulsign_vf_vf_vf(ret, x);

  ret = vsel_vf_vo_vf_vf(vlt_vo_vf_vf(nu, de), x, ret);
  ret = vsel_vf_vo_vf_vf(veq_vo_vf_vf(de, vcast_vf_f(0)), vcast_vf_f(NANf), ret);

  return ret;
}

//

static INLINE CONST vfloat2 sinpifk(vfloat d) {
  vopmask o;
  vfloat u, s, t;
  vfloat2 x, s2;

  u = vmul_vf_vf_vf(d, vcast_vf_f(4.0));
  vint2 q = vtruncate_vi2_vf(u);
  q = vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, vxor_vi2_vi2_vi2(vsrl_vi2_vi2_i(q, 31), vcast_vi2_i(1))), vcast_vi2_i(~1));
  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(2));

  s = vsub_vf_vf_vf(u, vcast_vf_vi2(q));
  t = s;
  s = vmul_vf_vf_vf(s, s);
  s2 = dfmul_vf2_vf_vf(t, t);

  //

  u = vsel_vf_vo_f_f(o, -0.2430611801e-7f, +0.3093842054e-6f);
  u = vmla_vf_vf_vf_vf(u, s, vsel_vf_vo_f_f(o, +0.3590577080e-5f, -0.3657307388e-4f));
  u = vmla_vf_vf_vf_vf(u, s, vsel_vf_vo_f_f(o, -0.3259917721e-3f, +0.2490393585e-2f));
  x = dfadd2_vf2_vf_vf2(vmul_vf_vf_vf(u, s),
      vsel_vf2_vo_f_f_f_f(o, 0.015854343771934509277, 4.4940051354032242811e-10,
              -0.080745510756969451904, -1.3373665339076936258e-09));
  x = dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf2(s2, x),
       vsel_vf2_vo_f_f_f_f(o, -0.30842512845993041992, -9.0728339030733922277e-09,
               0.78539818525314331055, -2.1857338617566484855e-08));

  x = dfmul_vf2_vf2_vf2(x, vsel_vf2_vo_vf2_vf2(o, s2, vcast_vf2_vf_vf(t, vcast_vf_f(0))));
  x = vsel_vf2_vo_vf2_vf2(o, dfadd2_vf2_vf2_vf(x, vcast_vf_f(1)), x);

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(4)), vcast_vi2_i(4));
  x.x = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(x.x)));
  x.y = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(x.y)));

  return x;
}

EXPORT CONST vfloat xsinpif_u05(vfloat d) {
  vfloat2 x = sinpifk(d);
  vfloat r = vadd_vf_vf_vf(x.x, x.y);

  r = vsel_vf_vo_vf_vf(visnegzero_vo_vf(d), vcast_vf_f(-0.0), r);
  r = vreinterpret_vf_vm(vandnot_vm_vo32_vm(vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX4f)), vreinterpret_vm_vf(r)));
  r = vreinterpret_vf_vm(vor_vm_vo32_vm(visinf_vo_vf(d), vreinterpret_vm_vf(r)));

  return r;
}

static INLINE CONST vfloat2 cospifk(vfloat d) {
  vopmask o;
  vfloat u, s, t;
  vfloat2 x, s2;

  u = vmul_vf_vf_vf(d, vcast_vf_f(4.0));
  vint2 q = vtruncate_vi2_vf(u);
  q = vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, vxor_vi2_vi2_vi2(vsrl_vi2_vi2_i(q, 31), vcast_vi2_i(1))), vcast_vi2_i(~1));
  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(0));

  s = vsub_vf_vf_vf(u, vcast_vf_vi2(q));
  t = s;
  s = vmul_vf_vf_vf(s, s);
  s2 = dfmul_vf2_vf_vf(t, t);

  //

  u = vsel_vf_vo_f_f(o, -0.2430611801e-7f, +0.3093842054e-6f);
  u = vmla_vf_vf_vf_vf(u, s, vsel_vf_vo_f_f(o, +0.3590577080e-5f, -0.3657307388e-4f));
  u = vmla_vf_vf_vf_vf(u, s, vsel_vf_vo_f_f(o, -0.3259917721e-3f, +0.2490393585e-2f));
  x = dfadd2_vf2_vf_vf2(vmul_vf_vf_vf(u, s),
      vsel_vf2_vo_f_f_f_f(o, 0.015854343771934509277, 4.4940051354032242811e-10,
              -0.080745510756969451904, -1.3373665339076936258e-09));
  x = dfadd2_vf2_vf2_vf2(dfmul_vf2_vf2_vf2(s2, x),
       vsel_vf2_vo_f_f_f_f(o, -0.30842512845993041992, -9.0728339030733922277e-09,
               0.78539818525314331055, -2.1857338617566484855e-08));

  x = dfmul_vf2_vf2_vf2(x, vsel_vf2_vo_vf2_vf2(o, s2, vcast_vf2_vf_vf(t, vcast_vf_f(0))));
  x = vsel_vf2_vo_vf2_vf2(o, dfadd2_vf2_vf2_vf(x, vcast_vf_f(1)), x);

  o = veq_vo_vi2_vi2(vand_vi2_vi2_vi2(vadd_vi2_vi2_vi2(q, vcast_vi2_i(2)), vcast_vi2_i(4)), vcast_vi2_i(4));
  x.x = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(x.x)));
  x.y = vreinterpret_vf_vm(vxor_vm_vm_vm(vand_vm_vo32_vm(o, vreinterpret_vm_vf(vcast_vf_f(-0.0))), vreinterpret_vm_vf(x.y)));

  return x;
}

EXPORT CONST vfloat xcospif_u05(vfloat d) {
  vfloat2 x = cospifk(d);
  vfloat r = vadd_vf_vf_vf(x.x, x.y);

  r = vsel_vf_vo_vf_vf(vgt_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(TRIGRANGEMAX4f)), vcast_vf_f(1), r);
  r = vreinterpret_vf_vm(vor_vm_vo32_vm(visinf_vo_vf(d), vreinterpret_vm_vf(r)));

  return r;
}

typedef struct {
  vfloat2 a, b;
} df2;

/* TODO AArch64: potential optimization by using `vfmad_lane_f64` */
static CONST df2 gammafk(vfloat a) {
  vfloat2 clc = vcast_vf2_f_f(0, 0), clln = vcast_vf2_f_f(1, 0), clld = vcast_vf2_f_f(1, 0);
  vfloat2 v = vcast_vf2_f_f(1, 0), x, y, z;
  vfloat t, u;

  vopmask otiny = vlt_vo_vf_vf(vabs_vf_vf(a), vcast_vf_f(1e-30f)), oref = vlt_vo_vf_vf(a, vcast_vf_f(0.5));

  x = vsel_vf2_vo_vf2_vf2(otiny, vcast_vf2_f_f(0, 0),
        vsel_vf2_vo_vf2_vf2(oref, dfadd2_vf2_vf_vf(vcast_vf_f(1), vneg_vf_vf(a)),
                vcast_vf2_vf_vf(a, vcast_vf_f(0))));

  vopmask o0 = vand_vo_vo_vo(vle_vo_vf_vf(vcast_vf_f(0.5), x.x), vle_vo_vf_vf(x.x, vcast_vf_f(1.2)));
  vopmask o2 = vle_vo_vf_vf(vcast_vf_f(2.3), x.x);

  y = dfnormalize_vf2_vf2(dfmul_vf2_vf2_vf2(dfadd2_vf2_vf2_vf(x, vcast_vf_f(1)), x));
  y = dfnormalize_vf2_vf2(dfmul_vf2_vf2_vf2(dfadd2_vf2_vf2_vf(x, vcast_vf_f(2)), y));

  vopmask o = vand_vo_vo_vo(o2, vle_vo_vf_vf(x.x, vcast_vf_f(7)));
  clln = vsel_vf2_vo_vf2_vf2(o, y, clln);

  x = vsel_vf2_vo_vf2_vf2(o, dfadd2_vf2_vf2_vf(x, vcast_vf_f(3)), x);
  t = vsel_vf_vo_vf_vf(o2, vrec_vf_vf(x.x), dfnormalize_vf2_vf2(dfadd2_vf2_vf2_vf(x, vsel_vf_vo_f_f(o0, -1, -2))).x);

  u = vsel_vf_vo_vo_f_f_f(o2, o0, +0.000839498720672087279971000786, +0.9435157776e+0f, +0.1102489550e-3f);
  u = vmla_vf_vf_vf_vf(u, t, vsel_vf_vo_vo_f_f_f(o2, o0, -5.17179090826059219329394422e-05, +0.8670063615e+0f, +0.8160019934e-4f));
  u = vmla_vf_vf_vf_vf(u, t, vsel_vf_vo_vo_f_f_f(o2, o0, -0.000592166437353693882857342347, +0.4826702476e+0f, +0.1528468856e-3f));
  u = vmla_vf_vf_vf_vf(u, t, vsel_vf_vo_vo_f_f_f(o2, o0, +6.97281375836585777403743539e-05, -0.8855129778e-1f, -0.2355068718e-3f));
  u = vmla_vf_vf_vf_vf(u, t, vsel_vf_vo_vo_f_f_f(o2, o0, +0.000784039221720066627493314301, +0.1013825238e+0f, +0.4962242092e-3f));
  u = vmla_vf_vf_vf_vf(u, t, vsel_vf_vo_vo_f_f_f(o2, o0, -0.000229472093621399176949318732, -0.1493408978e+0f, -0.1193488017e-2f));
  u = vmla_vf_vf_vf_vf(u, t, vsel_vf_vo_vo_f_f_f(o2, o0, -0.002681327160493827160473958490, +0.1697509140e+0f, +0.2891599433e-2f));
  u = vmla_vf_vf_vf_vf(u, t, vsel_vf_vo_vo_f_f_f(o2, o0, +0.003472222222222222222175164840, -0.2072454542e+0f, -0.7385451812e-2f));
  u = vmla_vf_vf_vf_vf(u, t, vsel_vf_vo_vo_f_f_f(o2, o0, +0.083333333333333333335592087900, +0.2705872357e+0f, +0.2058077045e-1f));

  y = dfmul_vf2_vf2_vf2(dfadd2_vf2_vf2_vf(x, vcast_vf_f(-0.5)), logk2f(x));
  y = dfadd2_vf2_vf2_vf2(y, dfneg_vf2_vf2(x));
  y = dfadd2_vf2_vf2_vf2(y, vcast_vf2_d(0.91893853320467278056)); // 0.5*log(2*M_PI)

  z = dfadd2_vf2_vf2_vf(dfmul_vf2_vf_vf (u, t), vsel_vf_vo_f_f(o0, -0.400686534596170958447352690395e+0f, -0.673523028297382446749257758235e-1f));
  z = dfadd2_vf2_vf2_vf(dfmul_vf2_vf2_vf(z, t), vsel_vf_vo_f_f(o0, +0.822466960142643054450325495997e+0f, +0.322467033928981157743538726901e+0f));
  z = dfadd2_vf2_vf2_vf(dfmul_vf2_vf2_vf(z, t), vsel_vf_vo_f_f(o0, -0.577215665946766039837398973297e+0f, +0.422784335087484338986941629852e+0f));
  z = dfmul_vf2_vf2_vf(z, t);

  clc = vsel_vf2_vo_vf2_vf2(o2, y, z);

  clld = vsel_vf2_vo_vf2_vf2(o2, dfadd2_vf2_vf2_vf(dfmul_vf2_vf_vf(u, t), vcast_vf_f(1)), clld);

  y = clln;

  clc = vsel_vf2_vo_vf2_vf2(otiny, vcast_vf2_d(41.58883083359671856503), // log(2^60)
          vsel_vf2_vo_vf2_vf2(oref, dfadd2_vf2_vf2_vf2(vcast_vf2_d(1.1447298858494001639), dfneg_vf2_vf2(clc)), clc)); // log(M_PI)
  clln = vsel_vf2_vo_vf2_vf2(otiny, vcast_vf2_f_f(1, 0), vsel_vf2_vo_vf2_vf2(oref, clln, clld));

  if (!vtestallones_i_vo32(vnot_vo32_vo32(oref))) {
    t = vsub_vf_vf_vf(a, vmul_vf_vf_vf(vcast_vf_f(1LL << 12), vcast_vf_vi2(vtruncate_vi2_vf(vmul_vf_vf_vf(a, vcast_vf_f(1.0 / (1LL << 12)))))));
    x = dfmul_vf2_vf2_vf2(clld, sinpifk(t));
  }

  clld = vsel_vf2_vo_vf2_vf2(otiny, vcast_vf2_vf_vf(vmul_vf_vf_vf(a, vcast_vf_f((1LL << 30)*(float)(1LL << 30))), vcast_vf_f(0)),
           vsel_vf2_vo_vf2_vf2(oref, x, y));

  df2 ret = { clc, dfdiv_vf2_vf2_vf2(clln, clld) };

  return ret;
}

EXPORT CONST vfloat xtgammaf_u1(vfloat a) {
  df2 d = gammafk(a);
  vfloat2 y = dfmul_vf2_vf2_vf2(expk2f(d.a), d.b);
  vfloat r = vadd_vf_vf_vf(y.x, y.y);
  vopmask o;

  o = vor_vo_vo_vo(vor_vo_vo_vo(veq_vo_vf_vf(a, vcast_vf_f(-INFINITYf)),
        vand_vo_vo_vo(vlt_vo_vf_vf(a, vcast_vf_f(0)), visint_vo_vf(a))),
       vand_vo_vo_vo(vand_vo_vo_vo(visnumber_vo_vf(a), vlt_vo_vf_vf(a, vcast_vf_f(0))), visnan_vo_vf(r)));
  r = vsel_vf_vo_vf_vf(o, vcast_vf_f(NANf), r);

  o = vand_vo_vo_vo(vand_vo_vo_vo(vor_vo_vo_vo(veq_vo_vf_vf(a, vcast_vf_f(INFINITYf)), visnumber_vo_vf(a)),
          vge_vo_vf_vf(a, vcast_vf_f(-FLT_MIN))),
        vor_vo_vo_vo(vor_vo_vo_vo(veq_vo_vf_vf(a, vcast_vf_f(0)), vgt_vo_vf_vf(a, vcast_vf_f(36))), visnan_vo_vf(r)));
  r = vsel_vf_vo_vf_vf(o, vmulsign_vf_vf_vf(vcast_vf_f(INFINITYf), a), r);

  return r;
}

EXPORT CONST vfloat xlgammaf_u1(vfloat a) {
  df2 d = gammafk(a);
  vfloat2 y = dfadd2_vf2_vf2_vf2(d.a, logk2f(dfabs_vf2_vf2(d.b)));
  vfloat r = vadd_vf_vf_vf(y.x, y.y);
  vopmask o;

  o = vor_vo_vo_vo(visinf_vo_vf(a),
       vor_vo_vo_vo(vand_vo_vo_vo(vle_vo_vf_vf(a, vcast_vf_f(0)), visint_vo_vf(a)),
        vand_vo_vo_vo(visnumber_vo_vf(a), visnan_vo_vf(r))));
  r = vsel_vf_vo_vf_vf(o, vcast_vf_f(INFINITYf), r);

  return r;
}

EXPORT CONST vfloat2 xlgamma_rf_u1(vfloat a) {
  df2 d = gammafk(a);
  vfloat2 y = dfadd2_vf2_vf2_vf2(d.a, logk2f(dfabs_vf2_vf2(d.b)));
  vfloat r = vadd_vf_vf_vf(y.x, y.y);
  vopmask o;

  o = vor_vo_vo_vo(visinf_vo_vf(a),
       vor_vo_vo_vo(vand_vo_vo_vo(vle_vo_vf_vf(a, vcast_vf_f(0)), visint_vo_vf(a)),
        vand_vo_vo_vo(visnumber_vo_vf(a), visnan_vo_vf(r))));
  r = vsel_vf_vo_vf_vf(o, vcast_vf_f(INFINITYf), r);

  vfloat2 ret;
  ret.x = r;
  ret.y = vreinterpret_vf_vm(vor_vm_vm_vm(
                               vand_vm_vm_vm(vreinterpret_vm_vf(d.b.x),
                                 vreinterpret_vm_vf(vcast_vf_f(-0.0f))),
                               vreinterpret_vm_vf(vcast_vf_f(1.0f)))
                            );

  return ret;
}

/* TODO AArch64: potential optimization by using `vfmad_lane_f64` */
EXPORT CONST vfloat xerff_u1(vfloat a) {
  vfloat s = a, t, u;
  vfloat2 d;

  a = vabs_vf_vf(a);
  vopmask o0 = vlt_vo_vf_vf(a, vcast_vf_f(1.1));
  vopmask o1 = vlt_vo_vf_vf(a, vcast_vf_f(2.4));
  vopmask o2 = vlt_vo_vf_vf(a, vcast_vf_f(4.0));
  u = vsel_vf_vo_vf_vf(o0, vmul_vf_vf_vf(a, a), a);

  t = vsel_vf_vo_vo_f_f_f(o0, o1, +0.7089292194e-4f, -0.1792667899e-4f, -0.9495757695e-5f);
  t = vmla_vf_vf_vf_vf(t, u, vsel_vf_vo_vo_f_f_f(o0, o1, -0.7768311189e-3f, +0.3937633010e-3f, +0.2481465926e-3f));
  t = vmla_vf_vf_vf_vf(t, u, vsel_vf_vo_vo_f_f_f(o0, o1, +0.5159463733e-2f, -0.3949181177e-2f, -0.2918176819e-2f));
  t = vmla_vf_vf_vf_vf(t, u, vsel_vf_vo_vo_f_f_f(o0, o1, -0.2683781274e-1f, +0.2445474640e-1f, +0.2059706673e-1f));
  t = vmla_vf_vf_vf_vf(t, u, vsel_vf_vo_vo_f_f_f(o0, o1, +0.1128318012e+0f, -0.1070996150e+0f, -0.9901899844e-1f));
  d = dfmul_vf2_vf_vf(t, u);
  d = dfadd2_vf2_vf2_vf2(d, vsel_vf2_vo_vo_d_d_d(o0, o1, -0.376125876000657465175213237214e+0, -0.634588905908410389971210809210e+0, -0.643598050547891613081201721633e+0));
  d = dfmul_vf2_vf2_vf(d, u);
  d = dfadd2_vf2_vf2_vf2(d, vsel_vf2_vo_vo_d_d_d(o0, o1, +0.112837916021059138255978217023e+1, -0.112879855826694507209862753992e+1, -0.112461487742845562801052956293e+1));
  d = dfmul_vf2_vf2_vf(d, a);
  d = vsel_vf2_vo_vf2_vf2(o0, d, dfadd_vf2_vf_vf2(vcast_vf_f(1.0), dfneg_vf2_vf2(expk2f(d))));
  u = vmulsign_vf_vf_vf(vsel_vf_vo_vf_vf(o2, vadd_vf_vf_vf(d.x, d.y), vcast_vf_f(1)), s);
  u = vsel_vf_vo_vf_vf(visnan_vo_vf(a), vcast_vf_f(NANf), u);

  return u;
}

/* TODO AArch64: potential optimization by using `vfmad_lane_f64` */
EXPORT CONST vfloat xerfcf_u15(vfloat a) {
  vfloat s = a, r = vcast_vf_f(0), t;
  vfloat2 u, d, x;
  a = vabs_vf_vf(a);
  vopmask o0 = vlt_vo_vf_vf(a, vcast_vf_f(1.0));
  vopmask o1 = vlt_vo_vf_vf(a, vcast_vf_f(2.2));
  vopmask o2 = vlt_vo_vf_vf(a, vcast_vf_f(4.3));
  vopmask o3 = vlt_vo_vf_vf(a, vcast_vf_f(10.1));

  u = vsel_vf2_vo_vf2_vf2(o1, vcast_vf2_vf_vf(a, vcast_vf_f(0)), dfdiv_vf2_vf2_vf2(vcast_vf2_f_f(1, 0), vcast_vf2_vf_vf(a, vcast_vf_f(0))));

  t = vsel_vf_vo_vo_vo_f_f_f_f(o0, o1, o2, -0.8638041618e-4f, -0.6236977242e-5f, -0.3869504035e+0f, +0.1115344167e+1f);
  t = vmla_vf_vf_vf_vf(t, u.x, vsel_vf_vo_vo_vo_f_f_f_f(o0, o1, o2, +0.6000166177e-3f, +0.5749821503e-4f, +0.1288077235e+1f, -0.9454904199e+0f));
  t = vmla_vf_vf_vf_vf(t, u.x, vsel_vf_vo_vo_vo_f_f_f_f(o0, o1, o2, -0.1665703603e-2f, +0.6002851478e-5f, -0.1816803217e+1f, -0.3667259514e+0f));
  t = vmla_vf_vf_vf_vf(t, u.x, vsel_vf_vo_vo_vo_f_f_f_f(o0, o1, o2, +0.1795156277e-3f, -0.2851036377e-2f, +0.1249150872e+1f, +0.7155663371e+0f));
  t = vmla_vf_vf_vf_vf(t, u.x, vsel_vf_vo_vo_vo_f_f_f_f(o0, o1, o2, +0.1914106123e-1f, +0.2260518074e-1f, -0.1328857988e+0f, -0.1262947265e-1f));

  d = dfmul_vf2_vf2_vf(u, t);
  d = dfadd2_vf2_vf2_vf2(d, vsel_vf2_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.102775359343930288081655368891e+0, -0.105247583459338632253369014063e+0, -0.482365310333045318680618892669e+0, -0.498961546254537647970305302739e+0));
  d = dfmul_vf2_vf2_vf2(d, u);
  d = dfadd2_vf2_vf2_vf2(d, vsel_vf2_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.636619483208481931303752546439e+0, -0.635609463574589034216723775292e+0, -0.134450203224533979217859332703e-2, -0.471199543422848492080722832666e-4));
  d = dfmul_vf2_vf2_vf2(d, u);
  d = dfadd2_vf2_vf2_vf2(d, vsel_vf2_vo_vo_vo_d_d_d_d(o0, o1, o2, -0.112837917790537404939545770596e+1, -0.112855987376668622084547028949e+1, -0.572319781150472949561786101080e+0, -0.572364030327966044425932623525e+0));

  x = dfmul_vf2_vf2_vf(vsel_vf2_vo_vf2_vf2(o1, d, vcast_vf2_vf_vf(vneg_vf_vf(a), vcast_vf_f(0))), a);
  x = vsel_vf2_vo_vf2_vf2(o1, x, dfadd2_vf2_vf2_vf2(x, d));

  x = expk2f(x);
  x = vsel_vf2_vo_vf2_vf2(o1, x, dfmul_vf2_vf2_vf2(x, u));

  r = vsel_vf_vo_vf_vf(o3, vadd_vf_vf_vf(x.x, x.y), vcast_vf_f(0));
  r = vsel_vf_vo_vf_vf(vsignbit_vo_vf(s), vsub_vf_vf_vf(vcast_vf_f(2), r), r);
  r = vsel_vf_vo_vf_vf(visnan_vo_vf(s), vcast_vf_f(NANf), r);
  return r;
}

#ifdef ENABLE_MAIN
// gcc -DENABLE_MAIN -Wno-attributes -I../common -I../arch -DENABLE_AVX2 -mavx2 -mfma sleefsimdsp.c ../common/common.c -lm
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  vfloat vf1 = vcast_vf_f(atof(argv[1]));
  //vfloat vf2 = vcast_vf_f(atof(argv[2]));

  //vfloat r = xpowf(vf1, vf2);
  //vfloat r = xsqrtf_u05(vf1);
  //printf("%g\n", xnextafterf(vf1, vf2)[0]);
  //printf("%g\n", nextafterf(atof(argv[1]), atof(argv[2])));
  printf("t = %.20g\n", xlogf_u1(vf1)[0]);
  printf("c = %.20g\n", logf(atof(argv[1])));

}
#endif
