// -*-C++-*-

// <http://gcc.gnu.org/onlinedocs/gcc/ARM-NEON-Intrinsics.html>

#ifndef VEC_NEON_FLOAT4_H
#define VEC_NEON_FLOAT4_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <cmath>

// Neon intrinsics
#include <arm_neon.h>



namespace vecmathlib {
  
#define VECMATHLIB_HAVE_VEC_FLOAT_4
  template<> struct boolvec<float,4>;
  template<> struct intvec<float,4>;
  template<> struct realvec<float,4>;
  
  
  
  template<>
  struct boolvec<float,4>: floatprops<float>
  {
    static int const size = 4;
    typedef bool scalar_t;
    typedef uint32x4_t bvector_t;
    static int const alignment = sizeof(bvector_t);
    
    static_assert(size * sizeof(real_t) == sizeof(bvector_t),
                  "vector size is wrong");
    
  private:
    // true values are -1, false values are 0
    static uint_t from_bool(bool a) { return -int_t(a); }
    static bool to_bool(uint_t a) { return a; }
  public:
    
    typedef boolvec boolvec_t;
    typedef intvec<real_t, size> intvec_t;
    typedef realvec<real_t, size> realvec_t;
    
    // Short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    typedef floatprops<real_t> FP;
    typedef mathfuncs<realvec_t> MF;
    
    
    
    bvector_t v;
    
    boolvec() {}
    // Can't have a non-trivial copy constructor; if so, objects won't
    // be passed in registers
    // boolvec(boolvec const& x): v(x.v) {}
    // boolvec& operator=(boolvec const& x) { return v=x.v, *this; }
    boolvec(bvector_t x): v(x) {}
    boolvec(bool a): v(vdupq_n_u32(from_bool(a))) {}
    boolvec(bool const* as)
    {
      for (int d=0; d<size; ++d) set_elt(d, as[d]);
    }
    
    operator bvector_t() const { return v; }
    bool operator[](int n) const
    {
      return to_bool(vecmathlib::get_elt<BV,bvector_t,uint_t>(v, n));
    }
    boolvec& set_elt(int n, bool a)
    {
      return
        vecmathlib::set_elt<BV,bvector_t,uint_t>(v, n, from_bool(a)), *this;
    }
    
    
    
    intvec_t as_int() const;      // defined after intvec
    intvec_t convert_int() const; // defined after intvec
    
    
    
    boolvec operator!() const { return vmvnq_u32(v); }
    
    boolvec operator&&(boolvec x) const { return vandq_u32(v, x.v); }
    boolvec operator||(boolvec x) const { return vorrq_u32(v, x.v); }
    boolvec operator==(boolvec x) const { return vceqq_u32(v, x.v); }
    boolvec operator!=(boolvec x) const { return veorq_u32(v, x.v); }
    
    bool all() const
    {
      uint32x2_t x = vpmin_u32(vget_low_u32(v), vget_high_u32(v));
      uint32x2_t y = vpmin_u32(x, x);
      uint32_t z = vget_lane_u32(y, 0);
      return to_bool(z);
    }
    bool any() const
    {
      uint32x2_t x = vpmax_u32(vget_low_u32(v), vget_high_u32(v));
      uint32x2_t y = vpmax_u32(x, x);
      uint32_t z = vget_lane_u32(y, 0);
      return to_bool(z);
    }
    
    
    
    // ifthen(condition, then-value, else-value)
    boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
    intvec_t ifthen(intvec_t x, intvec_t y) const; // defined after intvec
    realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realvec
  };
  
  
  
  template<>
  struct intvec<float,4>: floatprops<float>
  {
    static int const size = 4;
    typedef int_t scalar_t;
    typedef int32x4_t ivector_t;
    static int const alignment = sizeof(ivector_t);
    
    static_assert(size * sizeof(real_t) == sizeof(ivector_t),
                  "vector size is wrong");
    
    typedef boolvec<real_t, size> boolvec_t;
    typedef intvec intvec_t;
    typedef realvec<real_t, size> realvec_t;
    
    // Short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    typedef floatprops<real_t> FP;
    typedef mathfuncs<realvec_t> MF;
    
    
    
    ivector_t v;
    
    intvec() {}
    // Can't have a non-trivial copy constructor; if so, objects won't
    // be passed in registers
    // intvec(intvec const& x): v(x.v) {}
    // intvec& operator=(intvec const& x) { return v=x.v, *this; }
    intvec(ivector_t x): v(x) {}
    intvec(int_t a): v(vdupq_n_s32(a)) {}
    intvec(int_t const* as)
    {
      for (int d=0; d<size; ++d) set_elt(d, as[d]);
    }
    static intvec iota()
    {
      return
        vcombine_s32(vcreate_s32((uint64_t(1) << uint64_t(32)) | uint64_t(0)),
                     vcreate_s32((uint64_t(3) << uint64_t(32)) | uint64_t(2)));
    }
    
    operator ivector_t() const { return v; }
    int_t operator[](int n) const
    {
      return vecmathlib::get_elt<IV,ivector_t,int_t>(v, n);
    }
    intvec_t& set_elt(int n, int_t a)
    {
      return vecmathlib::set_elt<IV,ivector_t,int_t>(v, n, a), *this;
    }
    
    
    
    // Vector casts do not change the bit battern
    boolvec_t as_bool() const { return vreinterpretq_u32_s32(v); }
    boolvec_t convert_bool() const { return *this != IV(0); }
    realvec_t as_float() const;      // defined after realvec
    realvec_t convert_float() const; // defined after realvec
    
    
    
    intvec operator+() const { return *this; }
    intvec operator-() const { return vnegq_s32(v); }
    
    intvec operator+(intvec x) const { return vaddq_s32(v, x.v); }
    intvec operator-(intvec x) const { return vsubq_s32(v, x.v); }
    intvec operator*(intvec x) const { return vmulq_s32(v, x.v); }
    
    intvec& operator+=(intvec const& x) { return *this=*this+x; }
    intvec& operator-=(intvec const& x) { return *this=*this-x; }
    intvec& operator*=(intvec const& x) { return *this=*this*x; }
    
    
    
    intvec operator~() const { return vmvnq_s32(v); }
    
    intvec operator&(intvec x) const { return vandq_s32(v, x.v); }
    intvec operator|(intvec x) const { return vorrq_s32(v, x.v); }
    intvec operator^(intvec x) const { return veorq_s32(v, x.v); }
    
    intvec& operator&=(intvec const& x) { return *this=*this&x; }
    intvec& operator|=(intvec const& x) { return *this=*this|x; }
    intvec& operator^=(intvec const& x) { return *this=*this^x; }
    
    intvec_t bitifthen(intvec_t x, intvec_t y) const
    {
      return vbslq_s32(vreinterpretq_u32_s32(v), x.v, y.v)
    }
    
    
    
    intvec_t lsr(int_t n) const { return lsr(IV(n)); }
    intvec_t rotate(int_t n) const;
    intvec operator>>(int_t n) const { return *this >> IV(n); }
    intvec operator<<(int_t n) const { return *this << IV(n); }
    intvec& operator>>=(int_t n) { return *this=*this>>n; }
    intvec& operator<<=(int_t n) { return *this=*this<<n; }
    
    intvec_t lsr(intvec_t n) const
    {
      return vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(v), (-n).v));
    }
    intvec_t rotate(intvec_t n) const;
    intvec operator>>(intvec n) const
    {
      return vshlq_s32(v, (-n).v);
    }
    intvec operator<<(intvec n) const
    {
      return vshlq_s32(v, n.v);
    }
    intvec& operator>>=(intvec n) { return *this=*this>>n; }
    intvec& operator<<=(intvec n) { return *this=*this<<n; }
    
    intvec_t clz() const { return vclzq_s32(v); }
    intvec_t popcount() const
    {
      return vpaddlq_s16(vpaddlq_s8(vcntq_s8(vreinterpretq_s8_s32(v))));
    }
    
    
    
    boolvec_t operator==(intvec const& x) const { return vceqq_s32(v, x.v); }
    boolvec_t operator!=(intvec const& x) const { return !(*this == x); }
    boolvec_t operator<(intvec const& x) const { return vcltq_s32(v, x.v); }
    boolvec_t operator<=(intvec const& x) const { return vcleq_s32(v, x.v); }
    boolvec_t operator>(intvec const& x) const { return vcgtq_s32(v, x.v); }
    boolvec_t operator>=(intvec const& x) const { return vcgeq_s32(v, x.v); }
    
    intvec_t abs() const { return vabsq_s32(v); }
    boolvec_t isignbit() const
    {
      //return *this < IV(I(0));
      return intvec(vshrq_n_s32(v, FP::bits-1)).as_bool();
    }
    intvec_t max(intvec_t x) const { return vmaxq_s32(v, x.v); }
    intvec_t min(intvec_t x) const { return vminq_s32(v, x.v); }
  };
  
  
  
  template<>
  struct realvec<float,4>: floatprops<float>
  {
    static int const size = 4;
    typedef real_t scalar_t;
    typedef float32x4_t vector_t;
    static int const alignment = sizeof(vector_t);
    
    static char const* name() { return "<NEON:4*float>"; }
    void barrier() { __asm__("": "+w"(v)); }
    
    static_assert(size * sizeof(real_t) == sizeof(vector_t),
                  "vector size is wrong");
    
    typedef boolvec<real_t, size> boolvec_t;
    typedef intvec<real_t, size> intvec_t;
    typedef realvec realvec_t;
    
    // Short names for type casts
    typedef real_t R;
    typedef int_t I;
    typedef uint_t U;
    typedef realvec_t RV;
    typedef intvec_t IV;
    typedef boolvec_t BV;
    typedef floatprops<real_t> FP;
    typedef mathfuncs<realvec_t> MF;
    
    
    
    vector_t v;
    
    realvec() {}
    // Can't have a non-trivial copy constructor; if so, objects won't
    // be passed in registers
    // realvec(realvec const& x): v(x.v) {}
    // realvec& operator=(realvec const& x) { return v=x.v, *this; }
    realvec(vector_t x): v(x) {}
    realvec(real_t a): v(vdupq_n_f32(a)) {}
    realvec(real_t const* as)
    {
      for (int d=0; d<size; ++d) set_elt(d, as[d]);
    }
    
    operator vector_t() const { return v; }
    real_t operator[](int n) const
    {
      return vecmathlib::get_elt<RV,vector_t,real_t>(v, n);
    }
    realvec_t& set_elt(int n, real_t a)
    {
      return vecmathlib::set_elt<RV,vector_t,real_t>(v, n, a), *this;
    }
    
    
    
    typedef vecmathlib::mask_t<realvec_t> mask_t;
    
    static realvec_t loada(real_t const* p)
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      return vld1q_f32(p);
    }
    static realvec_t loadu(real_t const* p)
    {
#if defined __ARM_FEATURE_UNALIGNED
      return vld1q_f32(p);
#else
      realvec_t r;
      r.set_elt(0, p[0]);
      r.set_elt(1, p[1]);
      r.set_elt(2, p[2]);
      r.set_elt(3, p[3]);
      return r;
#endif
    }
    static realvec_t loadu(real_t const* p, std::ptrdiff_t ioff)
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (ioff % realvec::size == 0) return loada(p+ioff);
      return loadu(p+ioff);
    }
    realvec_t loada(real_t const* p, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (__builtin_expect(all(m.m), true)) {
        return loada(p);
      } else {
        return m.m.ifthen(loada(p), *this);
      }
    }
    realvec_t loadu(real_t const* p, mask_t const& m) const
    {
      if (__builtin_expect(m.all_m, true)) {
        return loadu(p);
      } else {
        return m.m.ifthen(loadu(p), *this);
      }
    }
    realvec_t loadu(real_t const* p, std::ptrdiff_t ioff, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (ioff % realvec::size == 0) return loada(p+ioff, m);
      return loadu(p+ioff, m);
    }
    
    void storea(real_t* p) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      vst1q_f32(p, v);
    }
    void storeu(real_t* p) const
    {
      // Vector stores would require vector loads, which would need to
      // be atomic
#if defined __ARM_FEATURE_UNALIGNED
      vst1q_f32(p, v);
#else
      p[0] = (*this)[0];
      p[1] = (*this)[1];
      p[2] = (*this)[2];
      p[3] = (*this)[3];
#endif
    }
    void storeu(real_t* p, std::ptrdiff_t ioff) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (ioff % realvec::size == 0) return storea(p+ioff);
      storeu(p+ioff);
    }
    void storea(real_t* p, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (__builtin_expect(m.all_m, true)) {
        storea(p);
      } else {
        if (m.m[0]) p[0] = (*this)[0];
        if (m.m[1]) p[1] = (*this)[1];
        if (m.m[2]) p[2] = (*this)[2];
        if (m.m[3]) p[3] = (*this)[3];
      }
    }
    void storeu(real_t* p, mask_t const& m) const
    {
      if (__builtin_expect(m.all_m, true)) {
        storeu(p);
      } else {
        if (m.m[0]) p[0] = (*this)[0];
        if (m.m[1]) p[1] = (*this)[1];
        if (m.m[2]) p[2] = (*this)[2];
        if (m.m[3]) p[3] = (*this)[3];
      }
    }
    void storeu(real_t* p, std::ptrdiff_t ioff, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (ioff % realvec::size == 0) return storea(p+ioff, m);
      storeu(p+ioff, m);
    }
    
    
    
    intvec_t as_int() const { return vreinterpretq_s32_f32(v); }
    intvec_t convert_int() const { return vcvtq_s32_f32(v); }
    
    
    
    realvec operator+() const { return *this; }
    realvec operator-() const { return vnegq_f32(v); }
    
    realvec operator+(realvec x) const { return vaddq_f32(v, x.v); }
    realvec operator-(realvec x) const { return vsubq_f32(v, x.v); }
    realvec operator*(realvec x) const { return vmulq_f32(v, x.v); }
    realvec operator/(realvec x) const { return *this * x.rcp(); }
    
    realvec& operator+=(realvec const& x) { return *this=*this+x; }
    realvec& operator-=(realvec const& x) { return *this=*this-x; }
    realvec& operator*=(realvec const& x) { return *this=*this*x; }
    realvec& operator/=(realvec const& x) { return *this=*this/x; }
    
    real_t maxval() const
    {
      float32x2_t x = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
      float32x2_t y = vpmax_f32(x, x);
      float32_t z = vget_lane_f32(y, 0);
      return z;
    }
    real_t minval() const
    {
      float32x2_t x = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
      float32x2_t y = vpmin_f32(x, x);
      float32_t z = vget_lane_f32(y, 0);
      return z;
    }
    real_t prod() const
    {
      // TODO: multiply pairwise with 2-vectors
      return (*this)[0] * (*this)[1] * (*this)[2] * (*this)[3];
    }
    real_t sum() const
    {
      float32x2_t x = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
      float32x2_t y = vpadd_f32(x, x);
      float32_t z = vget_lane_f32(y, 0);
      return z;
    }
    
    
    
    boolvec_t operator==(realvec const& x) const { return vceqq_f32(v, x.v); }
    boolvec_t operator!=(realvec const& x) const { return !(*this == x); }
    boolvec_t operator<(realvec const& x) const { return vcltq_f32(v, x.v); }
    boolvec_t operator<=(realvec const& x) const { return vcleq_f32(v, x.v); }
    boolvec_t operator>(realvec const& x) const { return vcgtq_f32(v, x.v); }
    boolvec_t operator>=(realvec const& x) const { return vcgeq_f32(v, x.v); }
    
    
    
    realvec acos() const { return MF::vml_acos(*this); }
    realvec acosh() const { return MF::vml_acosh(*this); }
    realvec asin() const { return MF::vml_asin(*this); }
    realvec asinh() const { return MF::vml_asinh(*this); }
    realvec atan() const { return MF::vml_atan(*this); }
    realvec atan2(realvec y) const { return MF::vml_atan2(*this, y); }
    realvec atanh() const { return MF::vml_atanh(*this); }
    realvec cbrt() const { return MF::vml_cbrt(*this); }
    realvec ceil() const
    {
      // return vrndpq_f32(v);
      return MF::vml_ceil(*this);
    }
    realvec copysign(realvec y) const
    {
      return vbslq_f32(vdupq_n_u32(FP::signbit_mask), y.v, v);
    }
    realvec cos() const { return MF::vml_cos(*this); }
    realvec cosh() const { return MF::vml_cosh(*this); }
    realvec exp() const { return MF::vml_exp(*this); }
    realvec exp10() const { return MF::vml_exp10(*this); }
    realvec exp2() const { return MF::vml_exp2(*this); }
    realvec expm1() const { return MF::vml_expm1(*this); }
    realvec fabs() const { return vabsq_f32(v); }
    realvec fdim(realvec y) const { return MF::vml_fdim(*this, y); }
    realvec floor() const
    {
      // return vrndmq_f32(v);
      return MF::vml_floor(*this);
    }
    realvec fma(realvec y, realvec z) const
    {
      // TODO: vfmaq_f32
      return vmlaq_f32(z.v, v, y.v);
    }
    realvec fmax(realvec y) const { return vmaxq_f32(v, y.v); }
    realvec fmin(realvec y) const { return vminq_f32(v, y.v); }
    realvec fmod(realvec y) const { return MF::vml_fmod(*this, y); }
    realvec frexp(intvec_t* r) const { return MF::vml_frexp(*this, r); }
    realvec hypot(realvec y) const { return MF::vml_hypot(*this, y); }
    intvec_t ilogb() const { return MF::vml_ilogb(*this); }
    boolvec_t isfinite() const { return MF::vml_isfinite(*this); }
    boolvec_t isinf() const { return MF::vml_isinf(*this); }
    boolvec_t isnan() const { return MF::vml_isnan(*this); }
    boolvec_t isnormal() const { return MF::vml_isnormal(*this); }
    realvec ldexp(int_t n) const { return MF::vml_ldexp(*this, n); }
    realvec ldexp(intvec_t n) const { return MF::vml_ldexp(*this, n); }
    realvec log() const { return MF::vml_log(*this); }
    realvec log10() const { return MF::vml_log10(*this); }
    realvec log1p() const { return MF::vml_log1p(*this); }
    realvec log2() const { return MF::vml_log2(*this); }
    realvec nextafter(realvec y) const { return MF::vml_nextafter(*this, y); }
    realvec pow(realvec y) const { return MF::vml_pow(*this, y); }
    realvec rcp() const
    {
      realvec r = vrecpeq_f32(v);
      r *= vrecpsq_f32(v, r);
      r *= vrecpsq_f32(v, r);
      return r;
    }
    realvec remainder(realvec y) const { return MF::vml_remainder(*this, y); }
    realvec rint() const
    {
      // return vrndnq_f32(v);
      return MF::vml_rint(*this);
    }
    realvec round() const
    {
      // return vrndaq_f32(v);
      return MF::vml_round(*this);
    }
    realvec rsqrt() const
    {
      realvec r = vrsqrteq_f32(v);
      r *= vrsqrtsq_f32(v, r*r);
      r *= vrsqrtsq_f32(v, r*r);
      return r;
    }
    boolvec_t signbit() const { return MF::vml_signbit(*this); }
    realvec sin() const { return MF::vml_sin(*this); }
    realvec sinh() const { return MF::vml_sinh(*this); }
    realvec sqrt() const { return *this * rsqrt(); }
    realvec tan() const { return MF::vml_tan(*this); }
    realvec tanh() const { return MF::vml_tanh(*this); }
    realvec trunc() const
    {
      // return vrndq_f32(v);
      return MF::vml_trunc(*this);
    }
  };
  
  
  
  // boolvec definitions
  
  inline intvec<float,4> boolvec<float,4>::as_int() const
  {
    return vreinterpretq_s32_u32(v);
  }
  
  inline intvec<float,4> boolvec<float,4>::convert_int() const
  {
    return - as_int();
  }
  
  inline
  boolvec<float,4> boolvec<float,4>::ifthen(boolvec_t x, boolvec_t y) const
  {
    return vbslq_u32(v, x.v, y.v);
  }
  
  inline intvec<float,4> boolvec<float,4>::ifthen(intvec_t x, intvec_t y) const
  {
    return vbslq_s32(v, x.v, y.v);
  }
  
  inline
  realvec<float,4> boolvec<float,4>::ifthen(realvec_t x, realvec_t y) const
  {
    return vbslq_f32(v, x.v, y.v);
  }
  
  
  
  // intvec definitions
  
  inline realvec<float,4> intvec<float,4>::as_float() const
  {
    return vreinterpretq_f32_s32(v);
  }
  
  inline realvec<float,4> intvec<float,4>::convert_float() const
  {
    return vcvtq_f32_s32(v);
  }
  
  inline intvec<float,4> intvec<float,4>::rotate(int_t n) const
  {
    return MF::vml_rotate(*this, n);
  }
  
  inline intvec<float,4> intvec<float,4>::rotate(intvec_t n) const
  {
    return MF::vml_rotate(*this, n);
  }
  
} // namespace vecmathlib

#endif  // #ifndef VEC_NEON_FLOAT4_H
