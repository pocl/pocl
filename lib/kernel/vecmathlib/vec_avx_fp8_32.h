// -*-C++-*-

#ifndef VEC_AVX_FP8_32_H
#define VEC_AVX_FP8_32_H

#include "floatprops.h"
#include "mathfuncs.h"
#include "vec_base.h"

#include <cmath>

// AVX intrinsics
#include <immintrin.h>



namespace vecmathlib {
  
#define VECMATHLIB_HAVE_VEC_FP8_32
  template<> struct boolvec<fp8,32>;
  template<> struct intvec<fp8,32>;
  template<> struct realvec<fp8,32>;
  
  
  
  template<>
  struct boolvec<fp8,32>: floatprops<fp8>
  {
    static int const size = 32;
    typedef bool scalar_t;
    typedef __m256i bvector_t;
    static int const alignment = sizeof(bvector_t);
    
    static_assert(size * sizeof(real_t) == sizeof(bvector_t),
                  "vector size is wrong");
    
  private:
    // true values have the sign bit set, false values have it unset
    static uint_t from_bool(bool a) { return - uint_t(a); }
    static bool to_bool(uint_t a) { return int_t(a) < int_t(0); }
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
    boolvec(bool a): v(_mm256_set1_epi8(from_bool(a))) {}
    boolvec(bool const* as):
    v(_mm256_set_epi8(from_bool(as[31]),
                      from_bool(as[30]),
                      from_bool(as[29]),
                      from_bool(as[28]),
                      from_bool(as[27]),
                      from_bool(as[26]),
                      from_bool(as[25]),
                      from_bool(as[24]),
                      from_bool(as[23]),
                      from_bool(as[22]),
                      from_bool(as[21]),
                      from_bool(as[20]),
                      from_bool(as[19]),
                      from_bool(as[18]),
                      from_bool(as[17]),
                      from_bool(as[16]),
                      from_bool(as[15]),
                      from_bool(as[14]),
                      from_bool(as[13]),
                      from_bool(as[12]),
                      from_bool(as[11]),
                      from_bool(as[10]),
                      from_bool(as[ 9]),
                      from_bool(as[ 8]),
                      from_bool(as[ 7]),
                      from_bool(as[ 6]),
                      from_bool(as[ 5]),
                      from_bool(as[ 4]),
                      from_bool(as[ 3]),
                      from_bool(as[ 2]),
                      from_bool(as[ 1]),
                      from_bool(as[ 0]))) {}
    
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
    
    
    
    boolvec operator!() const { return *this != boolvec(true); }
    
    boolvec operator&&(boolvec x) const 
    {
      return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v),
                                               _mm256_castsi256_ps(x.v)));
    }
    boolvec operator||(boolvec x) const
    {
      return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(v),
                                              _mm256_castsi256_ps(x.v)));
    }
    boolvec operator==(boolvec x) const { return !(*this!=x); }
    boolvec operator!=(boolvec x) const
    {
      return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(v),
                                               _mm256_castsi256_ps(x.v)));
    }
    
    bool all() const
    {
      bool r = (*this)[0];
      for (int n=1; n<size; ++n) r = r && (*this)[n];
      return r;
    }
    bool any() const
    {
      bool r = (*this)[0];;
      for (int n=1; n<size; ++n) r = r || (*this)[n];
      return r;
    }
    
    
    
    // ifthen(condition, then-value, else-value)
    boolvec_t ifthen(boolvec_t x, boolvec_t y) const;
    intvec_t ifthen(intvec_t x, intvec_t y) const; // defined after intvec
    realvec_t ifthen(realvec_t x, realvec_t y) const; // defined after realvec
  };
  
  
  
  template<>
  struct intvec<fp8,32>: floatprops<fp8>
  {
    static int const size = 32;
    typedef int_t scalar_t;
    typedef __m256i ivector_t;
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
    intvec(int_t a): v(_mm256_set1_epi8(a)) {}
    intvec(int_t const* as):
    v(_mm256_set_epi8(as[31],
                      as[30],
                      as[29],
                      as[28],
                      as[27],
                      as[26],
                      as[25],
                      as[24],
                      as[23],
                      as[22],
                      as[21],
                      as[20],
                      as[19],
                      as[18],
                      as[17],
                      as[16],
                      as[15],
                      as[14],
                      as[13],
                      as[12],
                      as[11],
                      as[10],
                      as[ 9],
                      as[ 8],
                      as[ 7],
                      as[ 6],
                      as[ 5],
                      as[ 4],
                      as[ 3],
                      as[ 2],
                      as[ 1],
                      as[ 0])) {}
    static intvec iota()
    {
      return _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24,
                             23, 22, 21, 20, 19, 18, 17, 16,
                             15, 14, 13, 12, 11, 10, 9, 8,
                             7, 6, 5, 4, 3, 2, 1, 0);
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
    
    
    
    boolvec_t as_bool() const { return v; }
    boolvec_t convert_bool() const
    {
      // Result: convert_bool(0)=false, convert_bool(else)=true
      // There is no intrinsic to compare with zero. Instead, we check
      // whether x is positive and x-1 is negative.
      intvec x = *this;
      // We know that boolvec values depend only on the sign bit
      // return (~(x-1) | x).as_bool();
      // return x.as_bool() || !(x-1).as_bool();
      return x.as_bool() || (x + (FP::signbit_mask - 1)).as_bool();
    }
    realvec_t as_float() const;      // defined after realvec
    realvec_t convert_float() const; // defined after realvec
    
    
    
    // Note: not all arithmetic operations are supported!
    
    intvec operator+() const { return *this; }
    intvec operator-() const { return IV(I(0)) - *this; }
    
    intvec operator+(intvec x) const
    {
      __m128i vlo = _mm256_castsi256_si128(v);
      __m128i vhi = _mm256_extractf128_si256(v, 1);
      __m128i xvlo = _mm256_castsi256_si128(x.v);
      __m128i xvhi = _mm256_extractf128_si256(x.v, 1);
      vlo = _mm_add_epi8(vlo, xvlo);
      vhi = _mm_add_epi8(vhi, xvhi);
      return _mm256_insertf128_si256(_mm256_castsi128_si256(vlo), vhi, 1);
    }
    intvec operator-(intvec x) const
    {
      __m128i vlo = _mm256_castsi256_si128(v);
      __m128i vhi = _mm256_extractf128_si256(v, 1);
      __m128i xvlo = _mm256_castsi256_si128(x.v);
      __m128i xvhi = _mm256_extractf128_si256(x.v, 1);
      vlo = _mm_sub_epi8(vlo, xvlo);
      vhi = _mm_sub_epi8(vhi, xvhi);
      return _mm256_insertf128_si256(_mm256_castsi128_si256(vlo), vhi, 1);
    }
    
    intvec& operator+=(intvec const& x) { return *this=*this+x; }
    intvec& operator-=(intvec const& x) { return *this=*this-x; }
    
    
    
    intvec operator~() const { return IV(~U(0)) ^ *this; }
    
    intvec operator&(intvec x) const
    {
      return _mm256_castps_si256(_mm256_and_ps(_mm256_castsi256_ps(v),
                                               _mm256_castsi256_ps(x.v)));
    }
    intvec operator|(intvec x) const
    {
      return _mm256_castps_si256(_mm256_or_ps(_mm256_castsi256_ps(v),
                                              _mm256_castsi256_ps(x.v)));
    }
    intvec operator^(intvec x) const
    {
      return _mm256_castps_si256(_mm256_xor_ps(_mm256_castsi256_ps(v),
                                               _mm256_castsi256_ps(x.v)));
    }
    
    intvec& operator&=(intvec const& x) { return *this=*this&x; }
    intvec& operator|=(intvec const& x) { return *this=*this|x; }
    intvec& operator^=(intvec const& x) { return *this=*this^x; }
    
    
    
    intvec lsr(int_t n) const
    {
      __m128i vlo = _mm256_castsi256_si128(v);
      __m128i vhi = _mm256_extractf128_si256(v, 1);
      uint_t masklo = U(0x00ffU) >> U(n);
      uint_t maskhi = U(0xff00U);
      __m128i mask = _mm_set1_epi16(masklo | maskhi);
      vlo = _mm_and_si128(_mm_srli_epi16(vlo, n), mask);
      vhi = _mm_and_si128(_mm_srli_epi16(vhi, n), mask);
      return _mm256_insertf128_si256(_mm256_castsi128_si256(vlo), vhi, 1);
    }
    intvec operator>>(int_t n) const
    {
      __m128i vlo = _mm256_castsi256_si128(v);
      __m128i vhi = _mm256_extractf128_si256(v, 1);
      uint_t masklo = U(0x00ffU);
      uint_t maskhi = U(0xff00U);
      __m128i vlolo = _mm_and_si128(_mm_srai_epi16(_mm_slli_epi16(vlo, 8), n+8),
                                    _mm_set1_epi16(masklo));
      __m128i vlohi = _mm_and_si128(_mm_srai_epi16(vlo, n),
                                    _mm_set1_epi16(maskhi));
      vlo = _mm_or_si128(vlolo, vlohi);
      __m128i vhilo = _mm_and_si128(_mm_srai_epi16(_mm_slli_epi16(vhi, 8), n+8),
                                    _mm_set1_epi16(masklo));
      __m128i vhihi = _mm_and_si128(_mm_srai_epi16(vhi, n),
                                    _mm_set1_epi16(maskhi));
      vhi = _mm_or_si128(vhilo, vhihi);
      return _mm256_insertf128_si256(_mm256_castsi128_si256(vlo), vhi, 1);
    }
    intvec operator<<(int_t n) const
    {
      __m128i vlo = _mm256_castsi256_si128(v);
      __m128i vhi = _mm256_extractf128_si256(v, 1);
      uint_t masklo = U(0x00ffU);
      uint_t maskhi = U(0xff00U) << U(n);
      __m128i mask = _mm_set1_epi16(masklo | maskhi);
      vlo = _mm_and_si128(_mm_slli_epi16(vlo, n), mask);
      vhi = _mm_and_si128(_mm_slli_epi16(vhi, n), mask);
      return _mm256_insertf128_si256(_mm256_castsi128_si256(vlo), vhi, 1);
    }
    intvec& operator>>=(int_t n) { return *this=*this>>n; }
    intvec& operator<<=(int_t n) { return *this=*this<<n; }
    
    intvec lsr(intvec n) const
    {
      intvec r;
      for (int i=0; i<size; ++i) {
        r.set_elt(i, U((*this)[i]) >> U(n[i]));
      }
      return r;
    }
    intvec operator>>(intvec n) const
    {
      intvec r;
      for (int i=0; i<size; ++i) {
        r.set_elt(i, (*this)[i] >> n[i]);
      }
      return r;
    }
    intvec operator<<(intvec n) const
    {
      intvec r;
      for (int i=0; i<size; ++i) {
        r.set_elt(i, (*this)[i] << n[i]);
      }
      return r;
    }
    intvec& operator>>=(intvec n) { return *this=*this>>n; }
    intvec& operator<<=(intvec n) { return *this=*this<<n; }
    
    
    
    boolvec_t operator==(intvec const& x) const
    {
      return ! (*this != x);
    }
    boolvec_t operator!=(intvec const& x) const
    {
      return (*this ^ x).convert_bool();
    }
  };
  
  
  
  template<>
  struct realvec<fp8,32>: floatprops<fp8>
  {
    static int const size = 32;
    typedef real_t scalar_t;
    typedef __m256i vector_t;
    static int const alignment = sizeof(vector_t);
    
    static char const* name() { return "<AVX:32*fp8>"; }
    void barrier() { __asm__("": "+x"(v)); }
    
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
    realvec(real_t a): v(_mm256_set1_epi8(FP::as_int(a))) {}
    realvec(real_t const* as):
    v(_mm256_set_epi8(FP::as_int(as[31]),
                      FP::as_int(as[30]),
                      FP::as_int(as[29]),
                      FP::as_int(as[28]),
                      FP::as_int(as[27]),
                      FP::as_int(as[26]),
                      FP::as_int(as[25]),
                      FP::as_int(as[24]),
                      FP::as_int(as[23]),
                      FP::as_int(as[22]),
                      FP::as_int(as[21]),
                      FP::as_int(as[20]),
                      FP::as_int(as[19]),
                      FP::as_int(as[18]),
                      FP::as_int(as[17]),
                      FP::as_int(as[16]),
                      FP::as_int(as[15]),
                      FP::as_int(as[14]),
                      FP::as_int(as[13]),
                      FP::as_int(as[12]),
                      FP::as_int(as[11]),
                      FP::as_int(as[10]),
                      FP::as_int(as[ 9]),
                      FP::as_int(as[ 8]),
                      FP::as_int(as[ 7]),
                      FP::as_int(as[ 6]),
                      FP::as_int(as[ 5]),
                      FP::as_int(as[ 4]),
                      FP::as_int(as[ 3]),
                      FP::as_int(as[ 2]),
                      FP::as_int(as[ 1]),
                      FP::as_int(as[ 0]))) {}
    
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
      return _mm256_load_si256((__m256i const*)p);
    }
    static realvec_t loadu(real_t const* p)
    {
      return _mm256_loadu_si256((__m256i const*)p);
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
      _mm256_store_si256((__m256i*)p, v);
    }
    void storeu(real_t* p) const
    {
      return _mm256_storeu_si256((__m256i*)p, v);
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
        // TODO: this is expensive
        for (int n=0; n<size; ++n) if (m.m[n]) p[n] = (*this)[n];
      }
    }
    void storeu(real_t* p, mask_t const& m) const
    {
      if (__builtin_expect(m.all_m, true)) {
        storeu(p);
      } else {
        // TODO: this is expensive
        for (int n=0; n<size; ++n) if (m.m[n]) p[n] = (*this)[n];
      }
    }
    void storeu(real_t* p, std::ptrdiff_t ioff, mask_t const& m) const
    {
      VML_ASSERT(intptr_t(p) % alignment == 0);
      if (ioff % realvec::size == 0) return storea(p+ioff, m);
      storeu(p+ioff, m);
    }
    
    
    
    intvec_t as_int() const { return v; }
    intvec_t convert_int() const { __builtin_unreachable(); }
    
    
    
    realvec operator+() const { __builtin_unreachable(); }
    realvec operator-() const { __builtin_unreachable(); }
    
    realvec operator+(realvec x) const { __builtin_unreachable(); }
    realvec operator-(realvec x) const { __builtin_unreachable(); }
    realvec operator*(realvec x) const { __builtin_unreachable(); }
    realvec operator/(realvec x) const { __builtin_unreachable(); }
    
    realvec& operator+=(realvec const& x) { return *this=*this+x; }
    realvec& operator-=(realvec const& x) { return *this=*this-x; }
    realvec& operator*=(realvec const& x) { return *this=*this*x; }
    realvec& operator/=(realvec const& x) { return *this=*this/x; }
    
    real_t maxval() const { __builtin_unreachable(); }
    real_t minval() const { __builtin_unreachable(); }
    real_t prod() const { __builtin_unreachable(); }
    real_t sum() const { __builtin_unreachable(); }
    
    
    
    boolvec_t operator==(realvec const& x) const { __builtin_unreachable(); }
    boolvec_t operator!=(realvec const& x) const { __builtin_unreachable(); }
    boolvec_t operator<(realvec const& x) const { __builtin_unreachable(); }
    boolvec_t operator<=(realvec const& x) const { __builtin_unreachable(); }
    boolvec_t operator>(realvec const& x) const { __builtin_unreachable(); }
    boolvec_t operator>=(realvec const& x) const { __builtin_unreachable(); }
    
    
    
    realvec copysign(realvec y) const { return MF::vml_copysign(*this, y); }
    realvec fabs() const { return MF::vml_fabs(*this); }
    intvec_t ilogb() const { return MF::vml_ilogb(*this); }
    boolvec_t isfinite() const { return MF::vml_isfinite(*this); }
    boolvec_t isinf() const { return MF::vml_isinf(*this); }
    boolvec_t isnan() const { return MF::vml_isnan(*this); }
    boolvec_t isnormal() const { return MF::vml_isnormal(*this); }
    realvec ldexp(int_t n) const { return MF::vml_ldexp(*this, n); }
    realvec ldexp(intvec_t n) const { return MF::vml_ldexp(*this, n); }
    boolvec_t signbit() const { return v; }
  };
  
  
  
  // boolvec definitions
  
  inline intvec<fp8,32> boolvec<fp8,32>::as_int() const
  {
    return v;
  }
  
  inline intvec<fp8,32> boolvec<fp8,32>::convert_int() const
  {
    return lsr(as_int(), bits-1);
  }
  
  inline
  boolvec<fp8,32> boolvec<fp8,32>::ifthen(boolvec_t x, boolvec_t y) const
  {
    return ifthen(x.as_int(), y.as_int()).as_bool();
  }
  
  inline intvec<fp8,32> boolvec<fp8,32>::ifthen(intvec_t x, intvec_t y) const
  {
    return ifthen(x.as_float(), y.as_float()).as_int();
  }
  
  inline
  realvec<fp8,32> boolvec<fp8,32>::ifthen(realvec_t x, realvec_t y) const
  {
    return (( -convert_int() & x.as_int()) |
            (~-convert_int() & y.as_int())).as_float();
  }

  
  
  // intvec definitions
  
  inline realvec<fp8,32> intvec<fp8,32>::as_float() const
  {
    return v;
  }
  
  inline realvec<fp8,32> intvec<fp8,32>::convert_float() const
  {
    __builtin_unreachable();
  }
  
} // namespace vecmathlib

#endif  // #ifndef VEC_AVX_FP8_32_H
