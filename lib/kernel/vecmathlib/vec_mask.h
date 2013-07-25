// -*-C++-*-

#ifndef VEC_MASK_H
#define VEC_MASK_H

#include <cstdlib>



namespace vecmathlib {
  
  template<typename realvec_t>
  class mask_t {
    
    typedef typename realvec_t::boolvec_t boolvec_t;
    typedef typename realvec_t::intvec_t intvec_t;
    static const int size = realvec_t::size;
    
  public:
    std::ptrdiff_t imin, imax;
    std::ptrdiff_t i;
    boolvec_t m;
    bool all_m;
    
  public:
    
    // Construct a mask from a boolvec
    mask_t(boolvec_t m_): m(m_), all_m(all(m)) {}
    
    // Construct a mask for a particular location i
    mask_t(std::ptrdiff_t i_,
           std::ptrdiff_t imin_, std::ptrdiff_t imax_, std::ptrdiff_t ioff):
      imin(imin_), imax(imax_), i(i_)
    {
      all_m = i-imin >= 0 && i+size-1-imax < 0;
      if (__builtin_expect(all_m, true)) {
        m = true;
      } else {
        m = (! signbit(intvec_t(i          - imin) + intvec_t::iota()) &&
               signbit(intvec_t(i + size-1 - imax) + intvec_t::iota()));
      }
    }
    
    // Construct a mask for a loop starting at imin, aligned down
    mask_t(std::ptrdiff_t imin_, std::ptrdiff_t imax_, std::ptrdiff_t ioff):
      imin(imin_), imax(imax_), i(imin_ - (ioff + imin_) % size)
    {
      all_m = i-imin >= 0 && i+size-1-imax < 0;
      if (__builtin_expect(all_m, true)) {
        m = true;
      } else {
        m = (! signbit(intvec_t(i          - imin) + intvec_t::iota()) &&
               signbit(intvec_t(i + size-1 - imax) + intvec_t::iota()));
      }
    }
    
    // Get current index
    std::ptrdiff_t index() const { return i; }
    
    // Looping condition
    operator bool() const { return i<imax; }
    
    // Loop stepper
    void operator++()
    {
      i += size;
      all_m = i + size-1 - imax < 0;
      if (__builtin_expect(all_m, true)) {
        m = true;
      } else {
        m = signbit(intvec_t(i + size-1 - imax) + intvec_t::iota());
      }
    }
  };
  
} // namespace vecmathlib

#endif  // #ifndef VEC_MASK_H
