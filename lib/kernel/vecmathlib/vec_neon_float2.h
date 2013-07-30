// -*-C++-*-

// <http://gcc.gnu.org/onlinedocs/gcc/ARM-NEON-Intrinsics.html>

#ifndef VEC_NEON_FLOAT2_H
#define VEC_NEON_FLOAT2_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <cmath>

// Neon intrinsics
#include <arm_neon.h>



namespace vecmathlib {
  
#define VECMATHLIB_HAVE_VEC_FLOAT_2
  template<> struct boolvec<float,2>;
  template<> struct intvec<float,2>;
  template<> struct realvec<float,2>;
  
  
  
  template<>
  struct boolvec<float,2>: floatprops<float>
  {
    static int const size = 2;
    typedef bool scalar_t;
    typedef uint32x2_t bvector_t;
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
    boolvec(bool a): v(vdup_n_u32(from_bool(a))) {}
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
    
    
    
    boolvec operator!() const { return vmvn_u32(v); }
    
    boolvec operator&&(boolvec x) const { return vand_u32(v, x.v); }
    boolvec operator||(boolvec x) const { return vorr_u32(v, x.v); }
    boolvec operator==(boolvec x) const { return vceq_u32(v, x.v); }
    boolvec operator!=(boolvec x) const { return veor_u32(v, x.v); }
    
    bool all() const
    {
      boolvec r = vpmin_u32(v, v);
      return r[0];
    }
    bool any() const
    {
      boolvec r = vpmax_u32(v, v);
      return r[0];
    }
    
    
    
    // ifthen(condition, then-value, else-value)
    boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
    intvec_t ifthen(intvec_t x, intvec_t y) const; // defined after intvec
    realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realvec
  };
  
  
  
  template<>
  struct intvec<float,2>: floatprops<float>
  {
    static int const size = 2;
    typedef int_t scalar_t;
    typedef int32x2_t ivector_t;
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
    intvec(int_t a): v(vdup_n_s32(a)) {}
    intvec(int_t const* as)
    {
      for (int d=0; d<size; ++d) set_elt(d, as[d]);
    }
    static intvec iota()
    {
      return vcreate_s32((uint64_t(1) << uint64_t(32)) | uint64_t(0));
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
    boolvec_t as_bool() const { return vreinterpret_u32_s32(v); }
    boolvec_t convert_bool() const { return *this != IV(0); }
    realvec_t as_float() const;      // defined after realvec
    realvec_t convert_float() const; // defined after realvec
    
    
    
    intvec operator+() const { return *this; }
    intvec operator-() const { return vneg_s32(v); }
    
    intvec operator+(intvec x) const { return vadd_s32(v, x.v); }
    intvec operator-(intvec x) const { return vsub_s32(v, x.v); }
    intvec operator*(intvec x) const { return vmul_s32(v, x.v); }
    
    intvec& operator+=(intvec const& x) { return *this=*this+x; }
    intvec& operator-=(intvec const& x) { return *this=*this-x; }
    intvec& operator*=(intvec const& x) { return *this=*this*x; }
    
    
    
    intvec operator~() const { return vmvn_s32(v); }
    
    intvec operator&(intvec x) const { return vand_s32(v, x.v); }
    intvec operator|(intvec x) const { return vorr_s32(v, x.v); }
    intvec operator^(intvec x) const { return veor_s32(v, x.v); }
    
    intvec& operator&=(intvec const& x) { return *this=*this&x; }
    intvec& operator|=(intvec const& x) { return *this=*this|x; }
    intvec& operator^=(intvec const& x) { return *this=*this^x; }
    
    
    
    intvec lsr(int_t n) const { return lsr(IV(n)); }
    intvec operator>>(int_t n) const { return *this >> IV(n); }
    intvec operator<<(int_t n) const { return *this << IV(n); }
    intvec& operator>>=(int_t n) { return *this=*this>>n; }
    intvec& operator<<=(int_t n) { return *this=*this<<n; }
    
    intvec lsr(intvec n) const
    {
      return vreinterpret_s32_u32(vshl_u32(vreinterpret_u32_s32(v), (-n).v));
    }
    intvec operator>>(intvec n) const
    {
      return vshl_s32(v, (-n).v);
    }
    intvec operator<<(intvec n) const
    {
      return vshl_s32(v, n.v);
    }
    intvec& operator>>=(intvec n) { return *this=*this>>n; }
    intvec& operator<<=(intvec n) { return *this=*this<<n; }
    
    
    
    boolvec_t signbit() const
    {
      //return *this < IV(I(0));
      return intvec(vshr_n_s32(v, FP::bits-1)).as_bool();
    }
    
    boolvec_t operator==(intvec const& x) const { return vceq_s32(v, x.v); }
    boolvec_t operator!=(intvec const& x) const { return !(*this == x); }
    boolvec_t operator<(intvec const& x) const { return vclt_s32(v, x.v); }
    boolvec_t operator<=(intvec const& x) const { return vcle_s32(v, x.v); }
    boolvec_t operator>(intvec const& x) const { return vcgt_s32(v, x.v); }
    boolvec_t operator>=(intvec const& x) const { return vcge_s32(v, x.v); }
  };
  
  
  
  template<>
  struct realvec<float,2>: floatprops<float>
  {
    static int const size = 2;
    typedef real_t scalar_t;
    typedef float32x2_t vector_t;
    static int const alignment = sizeof(vector_t);
    
    static char const* name() { return "<NEON:2*float>"; }
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
    realvec(real_t a): v(vdup_n_f32(a)) {}
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
      return vld1_f32(p);
    }
    static realvec_t loadu(real_t const* p)
    {
#if defined __ARM_FEATURE_UNALIGNED
      return vld1_f32(p);
#else
#  error "unaligned NEON loads not implemented"
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
      vst1_f32(p, v);
    }
    void storeu(real_t* p) const
    {
      // Vector stores would require vector loads, which would need to
      // be atomic
      // p[0] = (*this)[0];
      // p[1] = (*this)[1];
#if defined __ARM_FEATURE_UNALIGNED
      vst1_f32(p, v);
#else
#  error "unaligned NEON stores not implemented"
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
      }
    }
    void storeu(real_t* p, mask_t const& m) const
    {
      if (__builtin_expect(m.all_m, true)) {
        storeu(p);
      } else {
        if (m.m[0]) p[0] = (*this)[0];
        if (m.m[1]) p[1] = (*this)[1];
      }
    }
    void storeu(real_t* p, std::ptrdiff_t ioff, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (ioff % realvec::size == 0) return storea(p+ioff, m);
      storeu(p+ioff, m);
    }
    
    
    
    intvec_t as_int() const { return vreinterpret_s32_f32(v); }
    intvec_t convert_int() const { return vcvt_s32_f32(v); }
    
    
    
    realvec operator+() const { return *this; }
    realvec operator-() const { return vneg_f32(v); }
    
    realvec operator+(realvec x) const { return vadd_f32(v, x.v); }
    realvec operator-(realvec x) const { return vsub_f32(v, x.v); }
    realvec operator*(realvec x) const { return vmul_f32(v, x.v); }
    realvec operator/(realvec x) const { return *this * x.rcp(); }
    
    realvec& operator+=(realvec const& x) { return *this=*this+x; }
    realvec& operator-=(realvec const& x) { return *this=*this-x; }
    realvec& operator*=(realvec const& x) { return *this=*this*x; }
    realvec& operator/=(realvec const& x) { return *this=*this/x; }
    
    real_t maxval() const
    {
      realvec r = vpmax_f32(v, v);
      return r[0];
    }
    real_t minval() const
    {
      realvec r = vpmin_f32(v, v);
      return r[0];
    }
    real_t prod() const
    {
      return (*this)[0] * (*this)[1];
    }
    real_t sum() const
    {
      realvec r = vpadd_f32(v, v);
      return r[0];
    }
    
    
    
    boolvec_t operator==(realvec const& x) const { return vceq_f32(v, x.v); }
    boolvec_t operator!=(realvec const& x) const { return !(*this == x); }
    boolvec_t operator<(realvec const& x) const { return vclt_f32(v, x.v); }
    boolvec_t operator<=(realvec const& x) const { return vcle_f32(v, x.v); }
    boolvec_t operator>(realvec const& x) const { return vcgt_f32(v, x.v); }
    boolvec_t operator>=(realvec const& x) const { return vcge_f32(v, x.v); }
    
    
    
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
      // return vrndp_f32(v);
      return MF::vml_ceil(*this);
    }
    realvec copysign(realvec y) const
    {
      return vbsl_f32(vdup_n_u32(FP::signbit_mask), y.v, v);
    }
    realvec cos() const { return MF::vml_cos(*this); }
    realvec cosh() const { return MF::vml_cosh(*this); }
    realvec exp() const { return MF::vml_exp(*this); }
    realvec exp10() const { return MF::vml_exp10(*this); }
    realvec exp2() const { return MF::vml_exp2(*this); }
    realvec expm1() const { return MF::vml_expm1(*this); }
    realvec fabs() const { return vabs_f32(v); }
    realvec fdim(realvec y) const { return MF::vml_fdim(*this, y); }
    realvec floor() const
    {
      // return vrndm_f32(v);
      return MF::vml_floor(*this);
    }
    realvec fma(realvec y, realvec z) const
    {
      // TODO: vfma_f32
      return vmla_f32(z.v, v, y.v);
    }
    realvec fmax(realvec y) const { return vmax_f32(v, y.v); }
    realvec fmin(realvec y) const { return vmin_f32(v, y.v); }
    realvec fmod(realvec y) const { return MF::vml_fmod(*this, y); }
    realvec frexp(intvec_t& r) const { return MF::vml_frexp(*this, r); }
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
      realvec r = vrecpe_f32(v);
      r *= vrecps_f32(v, r);
      r *= vrecps_f32(v, r);
      return r;
    }
    realvec remainder(realvec y) const { return MF::vml_remainder(*this, y); }
    realvec rint() const
    {
      // return vrndn_f32(v);
      return MF::vml_rint(*this);
    }
    realvec round() const
    {
      // return vrnda_f32(v);
      return MF::vml_round(*this);
    }
    realvec rsqrt() const
    {
      realvec r = vrsqrte_f32(v);
      r *= vrsqrts_f32(v, r*r);
      r *= vrsqrts_f32(v, r*r);
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
      // return vrnd_f32(v);
      return MF::vml_trunc(*this);
    }
  };
  
  
  
  // boolvec definitions
  
  inline intvec<float,2> boolvec<float,2>::as_int() const
  {
    return vreinterpret_s32_u32(v);
  }
  
  inline intvec<float,2> boolvec<float,2>::convert_int() const
  {
    return - as_int();
  }
  
  inline
  boolvec<float,2> boolvec<float,2>::ifthen(boolvec_t x, boolvec_t y) const
  {
    return vbsl_u32(v, x.v, y.v);
  }
  
  inline intvec<float,2> boolvec<float,2>::ifthen(intvec_t x, intvec_t y) const
  {
    return vbsl_s32(v, x.v, y.v);
  }
  
  inline
  realvec<float,2> boolvec<float,2>::ifthen(realvec_t x, realvec_t y) const
  {
    return vbsl_f32(v, x.v, y.v);
  }
  
  
  
  // intvec definitions
  
  inline realvec<float,2> intvec<float,2>::as_float() const
  {
    return vreinterpret_f32_s32(v);
  }
  
  inline realvec<float,2> intvec<float,2>::convert_float() const
  {
    return vcvt_f32_s32(v);
  }
  
} // namespace vecmathlib

#endif  // #ifndef VEC_NEON_FLOAT2_H
