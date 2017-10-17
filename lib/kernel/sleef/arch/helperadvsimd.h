/*********************************************************************/
/*          Copyright ARM Ltd. 2010 - 2017.                          */
/* Distributed under the Boost Software License, Version 1.0.        */
/*    (See accompanying file LICENSE.txt or copy at                  */
/*          http://www.boost.org/LICENSE_1_0.txt)                    */
/*********************************************************************/

#ifndef __ARM_NEON
#error Please specify advsimd flags.
#endif

#include <arm_neon.h>
#include <stdint.h>

#include "misc.h"

#define ENABLE_DP
#define LOG2VECTLENDP 1
#define VECTLENDP (1 << LOG2VECTLENDP)
#define ENABLE_FMA_DP

#define ENABLE_SP
#define LOG2VECTLENSP 2
#define VECTLENSP (1 << LOG2VECTLENSP)
#define ENABLE_FMA_SP

#define ISANAME "AArch64 AdvSIMD"

// Mask definition
typedef uint32x4_t vmask;
typedef uint32x4_t vopmask;

// Single precision definitions
typedef float32x4_t vfloat;
typedef int32x4_t vint2;

// Double precision definitions
typedef float64x2_t vdouble;
typedef int32x2_t vint;

#define DFTPRIORITY 10

static INLINE int vavailability_i(int name) { return 3; }
static INLINE void vprefetch_v_p(const void *ptr) { }

static INLINE int vtestallones_i_vo32(vopmask g) {
  uint32x2_t x0 = vand_u32(vget_low_u32(g), vget_high_u32(g));
  uint32x2_t x1 = vpmin_u32(x0, x0);
  return vget_lane_u32(x1, 0);
}

static INLINE int vtestallones_i_vo64(vopmask g) {
  uint32x2_t x0 = vand_u32(vget_low_u32(g), vget_high_u32(g));
  uint32x2_t x1 = vpmin_u32(x0, x0);
  return vget_lane_u32(x1, 0);
}

// Vector load / store
static INLINE vdouble vload_vd_p(const double *ptr) { return vld1q_f64(ptr); }
static INLINE vdouble vloadu_vd_p(const double *ptr) { return vld1q_f64(ptr); }
static INLINE void vstore_v_p_vd(double *ptr, vdouble v) { vst1q_f64(ptr, v); }
static INLINE void vstoreu_v_p_vd(double *ptr, vdouble v) { vst1q_f64(ptr, v); }
static INLINE vfloat vload_vf_p(const float *ptr) { return vld1q_f32(ptr); }
static INLINE vfloat vloadu_vf_p(const float *ptr) { return vld1q_f32(ptr); }
static INLINE void vstore_v_p_vf(float *ptr, vfloat v) { vst1q_f32(ptr, v); }
static INLINE void vstoreu_v_p_vf(float *ptr, vfloat v) { vst1q_f32(ptr, v); }
static INLINE vint2 vloadu_vi2_p(int32_t *p) { return vld1q_s32(p); }
static INLINE void vstoreu_v_p_vi2(int32_t *p, vint2 v) { vst1q_s32(p, v); }
static INLINE vint vloadu_vi_p(int32_t *p) { return vld1_s32(p); }
static INLINE void vstoreu_v_p_vi(int32_t *p, vint v) { vst1_s32(p, v); }

// Basic logical operations for mask
static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) { return vandq_u32(x, y); }
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) {
  return vbicq_u32(y, x);
}
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) { return vorrq_u32(x, y); }
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) { return veorq_u32(x, y); }

// Mask <--> single precision reinterpret
static INLINE vmask vreinterpret_vm_vf(vfloat vf) {
  return vreinterpretq_u32_f32(vf);
}
static INLINE vfloat vreinterpret_vf_vm(vmask vm) {
  return vreinterpretq_f32_u32(vm);
}
static INLINE vint2 vcast_vi2_vm(vmask vm) { return vreinterpretq_s32_u32(vm); }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return vreinterpretq_u32_s32(vi); }

// Mask <--> double precision reinterpret
static INLINE vmask vreinterpret_vm_vd(vdouble vd) {
  return vreinterpretq_u32_f64(vd);
}
static INLINE vdouble vreinterpret_vd_vm(vmask vm) {
  return vreinterpretq_f64_u32(vm);
}
static INLINE vfloat vreinterpret_vf_vi2(vint2 vm) {
  return vreinterpretq_f32_s32(vm);
}
static INLINE vint2 vreinterpret_vi2_vf(vfloat vf) {
  return vreinterpretq_s32_f32(vf);
}
static INLINE vint2 vreinterpret_vi2_vd(vdouble vd) {
  return vreinterpretq_s32_f64(vd);
}

/****************************************/
/* Single precision FP operations */
/****************************************/
// Broadcast
static INLINE vfloat vcast_vf_f(float f) { return vdupq_n_f32(f); }

// Add, Sub, Mul, Reciprocal 1/x, Division, Square root
static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) {
  return vaddq_f32(x, y);
}
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) {
  return vsubq_f32(x, y);
}
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) {
  return vmulq_f32(x, y);
}
static INLINE vfloat vrec_vf_vf(vfloat d) {
  return vdivq_f32(vcast_vf_f(1.0f), d);
}
static INLINE vfloat vdiv_vf_vf_vf(vfloat n, vfloat d) {
  return vdivq_f32(n, d);
}
static INLINE vfloat vsqrt_vf_vf(vfloat d) { return vsqrtq_f32(d); }

// Multiply accumulate: z = z + x * y
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) {
  return vfmaq_f32(z, x, y);
}
// Multiply subtract: z = z = x * y
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) {
  return vfmsq_f32(z, x, y);
}

// |x|, -x
static INLINE vfloat vabs_vf_vf(vfloat f) { return vabsq_f32(f); }
static INLINE vfloat vneg_vf_vf(vfloat f) { return vnegq_f32(f); }

// max, min
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) {
  return vmaxq_f32(x, y);
}
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) {
  return vminq_f32(x, y);
}

// Comparisons
static INLINE vmask veq_vm_vf_vf(vfloat x, vfloat y) { return vceqq_f32(x, y); }
static INLINE vmask vneq_vm_vf_vf(vfloat x, vfloat y) {
  return vmvnq_u32(vceqq_f32(x, y));
}
static INLINE vmask vlt_vm_vf_vf(vfloat x, vfloat y) { return vcltq_f32(x, y); }
static INLINE vmask vle_vm_vf_vf(vfloat x, vfloat y) { return vcleq_f32(x, y); }
static INLINE vmask vgt_vm_vf_vf(vfloat x, vfloat y) { return vcgtq_f32(x, y); }
static INLINE vmask vge_vm_vf_vf(vfloat x, vfloat y) { return vcgeq_f32(x, y); }

// Conditional select
static INLINE vfloat vsel_vf_vm_vf_vf(vmask mask, vfloat x, vfloat y) {
  return vbslq_f32(mask, x, y);
}

// int <--> float conversions
static INLINE vint2 vtruncate_vi2_vf(vfloat vf) { return vcvtq_s32_f32(vf); }
static INLINE vfloat vcast_vf_vi2(vint2 vi) { return vcvtq_f32_s32(vi); }
static INLINE vint2 vcast_vi2_i(int i) { return vdupq_n_s32(i); }
static INLINE vint2 vrint_vi2_vf(vfloat d) {
  return vcvtq_s32_f32(vrndaq_f32(d));
}

/***************************************/
/* Single precision integer operations */
/***************************************/

// Add, Sub, Neg (-x)
static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) {
  return vaddq_s32(x, y);
}
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) {
  return vsubq_s32(x, y);
}
static INLINE vint2 vneg_vi2_vi2(vint2 e) { return vnegq_s32(e); }

// Logical operations
static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) {
  return vandq_s32(x, y);
}
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) {
  return vbicq_s32(y, x);
}
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) {
  return vorrq_s32(x, y);
}
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) {
  return veorq_s32(x, y);
}

// Shifts
#if defined(__clang__)
#define vsll_vi2_vi2_i(x, c) vshlq_n_s32(x, c)
#define vsrl_vi2_vi2_i(x, c)                                                   \
  vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(x), c))

#define vsra_vi2_vi2_i(x, c) vshrq_n_s32(x, c)
#define vsra_vi_vi_i(x, c) vshr_n_s32(x, c)
#define vsll_vi_vi_i(x, c) vshl_n_s32(x, c)
#define vsrl_vi_vi_i(x, c)                                                     \
  vreinterpret_s32_u32(vshr_n_u32(vreinterpret_u32_s32(x), c))

static INLINE vint2 vsrl64_vi2_vi_52(vint2 x) {
  return vreinterpretq_s32_u64(vshrq_n_u64(vreinterpretq_u64_s32(x), 52));
}

static INLINE vint2 vadd64_vi2_vi2_vi2(vint2 a, vint2 b) {
  return vreinterpretq_s32_s64(vaddq_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)));
}

static INLINE vint2 vsub64_vi2_vi2_vi2(vint2 a, vint2 b) {
  return vreinterpretq_s32_s64(vsubq_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)));
}

static INLINE vint2 vcast_vi2_i64(int64_t l) {
  return vreinterpretq_s32_s64(vdupq_n_s64(l));
}

static INLINE vmask vgt_cvt_vo_vi_vi(vint a, vint b) {
  return vreinterpretq_u32_s64(vmovl_s32(vreinterpret_s32_u32(vcgt_s32(a, b))));
}

static INLINE vmask veq_cvt_vo_vi_vi(vint a, vint b) {
  return vreinterpretq_u32_s64(vmovl_s32(vreinterpret_s32_u32(vceq_s32(a, b))));
}

#else
static INLINE vint2 vsll_vi2_vi2_i(vint2 x, const int c) {
  return vshlq_n_s32(x, c);
}
static INLINE vint vsll_vi_vi_i(vint x, const int c) {
  return vshl_n_s32(x, c);
}
static INLINE vint2 vsrl_vi2_vi2_i(vint2 x, const int c) {
  return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(x), c));
}
static INLINE vint vsrl_vi_vi_i(vint x, const int c) {
  return vreinterpret_s32_u32(vshr_n_u32(vreinterpret_u32_s32(x), c));
}
static INLINE vint2 vsra_vi2_vi2_i(vint2 x, const int c) {
  return vshrq_n_s32(x, c);
}
static INLINE vint vsra_vi_vi_i(vint x, const int c) {
  return vshr_n_s32(x, c);
}
#endif



// Comparison returning masks
static INLINE vmask veq_vm_vi2_vi2(vint2 x, vint2 y) { return vceqq_s32(x, y); }
static INLINE vmask vgt_vm_vi2_vi2(vint2 x, vint2 y) { return vcgeq_s32(x, y); }
// Comparison returning integers
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) {
  return vreinterpretq_s32_u32(vcgeq_s32(x, y));
}
static INLINE vint2 veq_vi2_vi2_vi2(vint2 x, vint2 y) {
  return vreinterpretq_s32_u32(vceqq_s32(x, y));
}

// Conditional select
static INLINE vint2 vsel_vi2_vm_vi2_vi2(vmask m, vint2 x, vint2 y) {
  return vbslq_s32(m, x, y);
}

/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- */

/****************************************/
/* Double precision FP operations */
/****************************************/
// Broadcast
static INLINE vdouble vcast_vd_d(double f) { return vdupq_n_f64(f); }

// Add, Sub, Mul, Reciprocal 1/x, Division, Square root
static INLINE vdouble vadd_vd_vd_vd(vdouble x, vdouble y) {
  return vaddq_f64(x, y);
}
static INLINE vdouble vsub_vd_vd_vd(vdouble x, vdouble y) {
  return vsubq_f64(x, y);
}
static INLINE vdouble vmul_vd_vd_vd(vdouble x, vdouble y) {
  return vmulq_f64(x, y);
}
static INLINE vdouble vrec_vd_vd(vdouble d) {
  return vdivq_f64(vcast_vd_d(1.0f), d);
}
static INLINE vdouble vdiv_vd_vd_vd(vdouble n, vdouble d) {
  return vdivq_f64(n, d);
}
static INLINE vdouble vsqrt_vd_vd(vdouble d) { return vsqrtq_f64(d); }

// |x|, -x
static INLINE vdouble vabs_vd_vd(vdouble f) { return vabsq_f64(f); }
static INLINE vdouble vneg_vd_vd(vdouble f) { return vnegq_f64(f); }

// max, min
static INLINE vdouble vmax_vd_vd_vd(vdouble x, vdouble y) {
  return vmaxq_f64(x, y);
}
static INLINE vdouble vmin_vd_vd_vd(vdouble x, vdouble y) {
  return vminq_f64(x, y);
}

// Multiply accumulate: z = z + x * y
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) {
  return vfmaq_f64(z, x, y);
}

static INLINE vdouble vmlanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) {
  return vfmsq_f64(z, x, y);
}

static INLINE vdouble vfma_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { // z + x * y
  return vfmaq_f64(z, x, y);
}

static INLINE vdouble vfmanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { // z - x * y
  return vfmsq_f64(z, x, y);
}

//[z = x * y - z]
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) {
  return vneg_vd_vd(vfmanp_vd_vd_vd_vd(x, y, z));
}

static INLINE vdouble vfmapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { // x * y - z
  return vneg_vd_vd(vfmanp_vd_vd_vd_vd(x, y, z));
}

static INLINE vfloat vfma_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { // z + x * y
  return vfmaq_f32(z, x, y);
}

static INLINE vfloat vfmanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { // z - x * y
  return vfmsq_f32(z, x, y);
}

static INLINE vfloat vfmapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { // x * y - z
  return vneg_vf_vf(vfmanp_vf_vf_vf_vf(x, y, z));
}

/* Comparisons */
static INLINE vopmask veq_vo_vd_vd(vdouble x, vdouble y) {
  return vreinterpretq_u32_u64(vceqq_f64(x, y));
}
static INLINE vopmask vneq_vo_vd_vd(vdouble x, vdouble y) {
  return vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(x, y)));
}
static INLINE vopmask vlt_vo_vd_vd(vdouble x, vdouble y) {
  return vreinterpretq_u32_u64(vcltq_f64(x, y));
}
static INLINE vopmask vgt_vo_vd_vd(vdouble x, vdouble y) {
  return vreinterpretq_u32_u64(vcgtq_f64(x, y));
}
static INLINE vopmask vle_vo_vd_vd(vdouble x, vdouble y) {
  return vreinterpretq_u32_u64(vcleq_f64(x, y));
}
static INLINE vopmask vge_vo_vd_vd(vdouble x, vdouble y) {
  return vreinterpretq_u32_u64(vcgeq_f64(x, y));
}

// Conditional select
static INLINE vdouble vsel_vd_vo_vd_vd(vopmask mask, vdouble x, vdouble y) {
  return vbslq_f64(vreinterpretq_u64_u32(mask), x, y);
}

#if 1
static INLINE CONST vdouble vsel_vd_vo_d_d(vopmask o, double v1, double v0) {
  return vsel_vd_vo_vd_vd(o, vcast_vd_d(v1), vcast_vd_d(v0));
}

static INLINE vdouble vsel_vd_vo_vo_d_d_d(vopmask o0, vopmask o1, double d0, double d1, double d2) {
  return vsel_vd_vo_vd_vd(o0, vcast_vd_d(d0), vsel_vd_vo_d_d(o1, d1, d2));
}

static INLINE vdouble vsel_vd_vo_vo_vo_d_d_d_d(vopmask o0, vopmask o1, vopmask o2, double d0, double d1, double d2, double d3) {
  return vsel_vd_vo_vd_vd(o0, vcast_vd_d(d0), vsel_vd_vo_vd_vd(o1, vcast_vd_d(d1), vsel_vd_vo_d_d(o2, d2, d3)));
}
#else
// This implementation is slower on the current CPU models (as of May 2017.)
// I(Naoki Shibata) expect that on future CPU models with hardware similar to Super Shuffle Engine, this implementation will be faster.
static INLINE CONST vdouble vsel_vd_vo_d_d(vopmask o, double d0, double d1) {
  uint8x16_t idx = vbslq_u8(vreinterpretq_u8_u32(o), (uint8x16_t) { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 },
          (uint8x16_t) { 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15 });

  uint8x16_t tab = (uint8x16_t) (float64x2_t) { d0, d1 };
  return (vdouble) vqtbl1q_u8(tab, idx);
}

static INLINE vdouble vsel_vd_vo_vo_vo_d_d_d_d(vopmask o0, vopmask o1, vopmask o2, double d0, double d1, double d2, double d3) {
  uint8x16_t idx = vbslq_u8(vreinterpretq_u8_u32(o0), (uint8x16_t) { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7 },
          vbslq_u8(vreinterpretq_u8_u32(o1), (uint8x16_t) { 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15 },
             vbslq_u8(vreinterpretq_u8_u32(o2), (uint8x16_t) { 16, 17, 18, 19, 20, 21, 22, 23, 16, 17, 18, 19, 20, 21, 22, 23 },
                (uint8x16_t) { 24, 25, 26, 27, 28, 29, 30, 31, 24, 25, 26, 27, 28, 29, 30, 31 })));

  uint8x16x2_t tab = { { (uint8x16_t) (float64x2_t) { d0, d1 }, (uint8x16_t) (float64x2_t) { d2, d3 } } };
  return (vdouble) vqtbl2q_u8(tab, idx);
}

static INLINE vdouble vsel_vd_vo_vo_d_d_d(vopmask o0, vopmask o1, double d0, double d1, double d2) {
  return vsel_vd_vo_vo_vo_d_d_d_d(o0, o1, o1, d0, d1, d2, d2);
}
#endif

static INLINE vdouble vrint_vd_vd(vdouble d) { return vrndnq_f64(d); }
static INLINE vfloat vrint_vf_vf(vfloat d) { return vrndnq_f32(d); }

/****************************************/
/* int <--> float conversions           */
/****************************************/
static INLINE vint vtruncate_vi_vd(vdouble vf) {
  return vmovn_s64(vcvtq_s64_f64(vf));
}
static INLINE vdouble vcast_vd_vi(vint vi) {
  return vcvtq_f64_s64(vmovl_s32(vi));
}
static INLINE vint vcast_vi_i(int i) { return vdup_n_s32(i); }
static INLINE vint vrint_vi_vd(vdouble d) {
  return vqmovn_s64(vcvtq_s64_f64(vrndaq_f64(d)));
}

/***************************************/
/* Integer operations */
/***************************************/

// Add, Sub, Neg (-x)
static INLINE vint vadd_vi_vi_vi(vint x, vint y) { return vadd_s32(x, y); }
static INLINE vint vsub_vi_vi_vi(vint x, vint y) { return vsub_s32(x, y); }
static INLINE vint vneg_vi_vi(vint e) { return vneg_s32(e); }

// Logical operations
static INLINE vint vand_vi_vi_vi(vint x, vint y) { return vand_s32(x, y); }
static INLINE vint vandnot_vi_vi_vi(vint x, vint y) { return vbic_s32(y, x); }
static INLINE vint vor_vi_vi_vi(vint x, vint y) { return vorr_s32(x, y); }
static INLINE vint vxor_vi_vi_vi(vint x, vint y) { return veor_s32(x, y); }

// Comparison returning masks
static INLINE vopmask veq_vo_vi_vi(vint x, vint y) {
  return vcombine_u32(vceq_s32(x, y), vdup_n_u32(0));
}

// Conditional select
static INLINE vint vsel_vi_vm_vi_vi(vmask m, vint x, vint y) {
  return vbsl_s32(vget_low_u32(m), x, y);
}

/***************************************/
/* Predicates                          */
/***************************************/
static INLINE vopmask visinf_vo_vd(vdouble d) {
  const float64x2_t inf = vdupq_n_f64(INFINITY);
  const float64x2_t neg_inf = vdupq_n_f64(-INFINITY);
  uint64x2_t cmp = vorrq_u64(vceqq_f64(d, inf), vceqq_f64(d, neg_inf));
  return vreinterpretq_u32_u64(cmp);
}

static INLINE vopmask visnan_vo_vd(vdouble d) {
  return vmvnq_u32(vreinterpretq_u32_u64(vceqq_f64(d, d)));
}

static INLINE vopmask vispinf_vo_vd(vdouble d) {
  return vreinterpretq_u32_u64(vceqq_f64(d, vdupq_n_f64(INFINITY)));
}

static INLINE vopmask visminf_vo_vd(vdouble d) {
  return vreinterpretq_u32_u64(vceqq_f64(d, vdupq_n_f64(-INFINITY)));
}

static INLINE vfloat vsel_vf_vo_vf_vf(vopmask mask, vfloat x, vfloat y) {
  return vbslq_f32(mask, x, y);
}

static INLINE CONST vfloat vsel_vf_vo_f_f(vopmask o, float v1, float v0) {
  return vsel_vf_vo_vf_vf(o, vcast_vf_f(v1), vcast_vf_f(v0));
}

static INLINE vfloat vsel_vf_vo_vo_f_f_f(vopmask o0, vopmask o1, float d0, float d1, float d2) {
  return vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_f_f(o1, d1, d2));
}

static INLINE vfloat vsel_vf_vo_vo_vo_f_f_f_f(vopmask o0, vopmask o1, vopmask o2, float d0, float d1, float d2, float d3) {
  return vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_vf_vf(o1, vcast_vf_f(d1), vsel_vf_vo_f_f(o2, d2, d3)));
}

static INLINE vopmask veq_vo_vf_vf(vfloat x, vfloat y) {
  return vceqq_f32(x, y);
}
static INLINE vopmask vneq_vo_vf_vf(vfloat x, vfloat y) {
  return vmvnq_u32(vceqq_f32(x, y));
}
static INLINE vopmask vlt_vo_vf_vf(vfloat x, vfloat y) {
  return vcltq_f32(x, y);
}
static INLINE vopmask vle_vo_vf_vf(vfloat x, vfloat y) {
  return vcleq_f32(x, y);
}
static INLINE vopmask vgt_vo_vf_vf(vfloat x, vfloat y) {
  return vcgtq_f32(x, y);
}
static INLINE vopmask vge_vo_vf_vf(vfloat x, vfloat y) {
  return vcgeq_f32(x, y);
}

static INLINE vopmask veq_vo_vi2_vi2(vint2 x, vint2 y) {
  return vceqq_s32(x, y);
}
static INLINE vopmask vgt_vo_vi2_vi2(vint2 x, vint2 y) {
  return vcgeq_s32(x, y);
}
static INLINE vopmask vgt_vo_vi_vi(vint x, vint y) {
  return vcombine_u32(vcge_s32(x, y), vdup_n_u32(0));
}
static INLINE vopmask visinf_vo_vf(vfloat d) {
  return veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(INFINITYf));
}
static INLINE vopmask vispinf_vo_vf(vfloat d) {
  return veq_vo_vf_vf(d, vcast_vf_f(INFINITYf));
}
static INLINE vopmask visminf_vo_vf(vfloat d) {
  return veq_vo_vf_vf(d, vcast_vf_f(-INFINITYf));
}
static INLINE vopmask visnan_vo_vf(vfloat d) { return vneq_vo_vf_vf(d, d); }

static INLINE vopmask vcast_vo32_vo64(vopmask m) {
  return vuzpq_u32(m, m).val[0];
}
static INLINE vopmask vcast_vo64_vo32(vopmask m) {
  return vzipq_u32(m, m).val[0];
}

static INLINE vopmask vand_vo_vo_vo(vopmask x, vopmask y) {
  return vandq_u32(x, y);
}
static INLINE vopmask vandnot_vo_vo_vo(vopmask x, vopmask y) {
  return vbicq_u32(y, x);
}
static INLINE vopmask vor_vo_vo_vo(vopmask x, vopmask y) {
  return vorrq_u32(x, y);
}
static INLINE vopmask vxor_vo_vo_vo(vopmask x, vopmask y) {
  return veorq_u32(x, y);
}

static INLINE vint2 vsel_vi2_vo_vi2_vi2(vopmask m, vint2 x, vint2 y) {
  return vbslq_s32(m, x, y);
}
static INLINE vint2 vand_vi2_vo_vi2(vopmask x, vint2 y) {
  return vandq_s32(vreinterpretq_s32_u32(x), y);
}
static INLINE vint2 vandnot_vi2_vo_vi2(vopmask x, vint2 y) {
  return vbicq_s32(y, vreinterpretq_s32_u32(x));
}
static INLINE vint vandnot_vi_vo_vi(vopmask x, vint y) {
  return vbic_s32(y, vget_low_s32(vreinterpretq_s32_u32(x)));
}
static INLINE vmask vand_vm_vo32_vm(vopmask x, vmask y) {
  return vandq_u32(x, y);
}
static INLINE vmask vand_vm_vo64_vm(vopmask x, vmask y) {
  return vandq_u32(x, y);
}
static INLINE vmask vandnot_vm_vo32_vm(vopmask x, vmask y) {
  return vbicq_u32(y, x);
}
static INLINE vmask vandnot_vm_vo64_vm(vopmask x, vmask y) {
  return vbicq_u32(y, x);
}
static INLINE vmask vor_vm_vo32_vm(vopmask x, vmask y) {
  return vorrq_u32(x, y);
}
static INLINE vmask vor_vm_vo64_vm(vopmask x, vmask y) {
  return vorrq_u32(x, y);
}
static INLINE vmask vxor_vm_vo32_vm(vopmask x, vmask y) {
  return veorq_u32(x, y);
}

static INLINE vfloat vtruncate_vf_vf(vfloat vd) { return vrndq_f32(vd); }

static INLINE vmask vcast_vm_i_i(int i0, int i1) {
  return vreinterpretq_u32_u64(vdupq_n_u64((0xffffffff & (uint64_t)i1) | (((uint64_t)i0) << 32)));
}

static INLINE vopmask veq64_vo_vm_vm(vmask x, vmask y) {
  return vreinterpretq_u32_u64(
      vceqq_s64(vreinterpretq_s64_u32(x), vreinterpretq_s64_u32(y)));
}

static INLINE vint vsel_vi_vo_vi_vi(vopmask m, vint x, vint y) {
  return vbsl_s32(vget_low_u32(m), x, y);
}

// Logical operations
static INLINE vint vand_vi_vo_vi(vopmask x, vint y) {
  return vand_s32(vreinterpret_s32_u32(vget_low_u32(x)), y);
}

static INLINE vint2 vcastu_vi2_vi(vint vi) {
  return vreinterpretq_s32_u32(vrev64q_u32(vreinterpretq_u32_u64(vmovl_u32(vreinterpret_u32_s32(vi)))));
}
static INLINE vint vcastu_vi_vi2(vint2 vi2) {
  return vreinterpret_s32_u32(vmovn_u64(vreinterpretq_u64_u32(vrev64q_u32(vreinterpretq_u32_s32(vi2)))));
}
static INLINE vdouble vreinterpret_vd_vi2(vint2 vi) {
  return vreinterpretq_f64_s32(vi);
}
static INLINE vdouble vtruncate_vd_vd(vdouble vd) { return vrndq_f64(vd); }

//

#define PNMASK ((vdouble) { +0.0, -0.0 })
#define NPMASK ((vdouble) { -0.0, +0.0 })
#define PNMASKf ((vfloat) { +0.0f, -0.0f, +0.0f, -0.0f })
#define NPMASKf ((vfloat) { -0.0f, +0.0f, -0.0f, +0.0f })

static INLINE vdouble vposneg_vd_vd(vdouble d) { return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(PNMASK))); }
static INLINE vdouble vnegpos_vd_vd(vdouble d) { return vreinterpret_vd_vm(vxor_vm_vm_vm(vreinterpret_vm_vd(d), vreinterpret_vm_vd(NPMASK))); }
static INLINE vfloat vposneg_vf_vf(vfloat d) { return (vfloat)vxor_vm_vm_vm((vmask)d, (vmask)PNMASKf); }
static INLINE vfloat vnegpos_vf_vf(vfloat d) { return (vfloat)vxor_vm_vm_vm((vmask)d, (vmask)NPMASKf); }

static INLINE vdouble vsubadd_vd_vd_vd(vdouble x, vdouble y) { return vadd_vd_vd_vd(x, vnegpos_vd_vd(y)); }
static INLINE vfloat vsubadd_vf_vf_vf(vfloat d0, vfloat d1) { return vadd_vf_vf_vf(d0, vnegpos_vf_vf(d1)); }
static INLINE vdouble vmlsubadd_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vsubadd_vd_vd_vd(vmul_vd_vd_vd(x, y), z); }
static INLINE vfloat vmlsubadd_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vsubadd_vf_vf_vf(vmul_vf_vf_vf(x, y), z); }

static INLINE vdouble vrev21_vd_vd(vdouble d0) { return (float64x2_t)vcombine_u64(vget_high_u64((uint64x2_t)d0), vget_low_u64((uint64x2_t)d0)); }
static INLINE vdouble vreva2_vd_vd(vdouble vd) { return vd; }

static INLINE void vstream_v_p_vd(double *ptr, vdouble v) { vstore_v_p_vd(ptr, v); }
static INLINE void vscatter2_v_p_i_i_vd(double *ptr, int offset, int step, vdouble v) { vstore_v_p_vd((double *)(&ptr[2*offset]), v); }
static INLINE void vsscatter2_v_p_i_i_vd(double *ptr, int offset, int step, vdouble v) { vstore_v_p_vd((double *)(&ptr[2*offset]), v); }

static INLINE vfloat vrev21_vf_vf(vfloat d0) { return vrev64q_f32(d0); }
static INLINE vfloat vreva2_vf_vf(vfloat d0) { return vcombine_f32(vget_high_f32(d0), vget_low_f32(d0)); }

static INLINE void vstream_v_p_vf(float *ptr, vfloat v) { vstore_v_p_vf(ptr, v); }

static INLINE void vscatter2_v_p_i_i_vf(float *ptr, int offset, int step, vfloat v) {
  vst1_f32((float *)(ptr+(offset + step * 0)*2), vget_low_f32(v));
  vst1_f32((float *)(ptr+(offset + step * 1)*2), vget_high_f32(v));
}

static INLINE void vsscatter2_v_p_i_i_vf(float *ptr, int offset, int step, vfloat v) {
  vst1_f32((float *)(ptr+(offset + step * 0)*2), vget_low_f32(v));
  vst1_f32((float *)(ptr+(offset + step * 1)*2), vget_high_f32(v));
}

// TODO

static INLINE int vall_lte64_i_vd_vd(vdouble x, vdouble lim) {
  return 0;
}

static INLINE int vall_lte32_i_vf_vf(vfloat x, vfloat lim) {
  return 0;
}
