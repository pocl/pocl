// -*-C++-*-

#ifndef MATHFUNCS_CONVERT_H
#define MATHFUNCS_CONVERT_H

#include "mathfuncs_base.h"

#include <cmath>



namespace vecmathlib {
  
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_convert_float(intvec_t x)
  {
    // Convert in two passes. Convert as much as possible during the
    // first pass (lobits), so that the second pass (hibits) may be
    // omitted if the high bits are known to be zero.
    int_t lobits = FP::mantissa_bits;
    // int_t hibits = FP::bits - lobits;
    
    // Convert lower bits
    intvec_t xlo = x & IV((U(1) << lobits) - 1);
    // exponent for the equivalent floating point number
    int_t exponent_lo = (FP::exponent_offset + lobits) << FP::mantissa_bits;
    xlo |= exponent_lo;
    // subtract hidden mantissa bit
    realvec_t flo = as_float(xlo) - RV(FP::as_float(exponent_lo));
    
    // Convert upper bits
    // make unsigned by subtracting largest negative number
    // (only do this for the high bits, since they have sufficient
    // precision to handle the overflow)
    x ^= FP::signbit_mask;
    intvec_t xhi = lsr(x, lobits);
    // exponent for the equivalent floating point number
    int_t exponent_hi = (FP::exponent_offset + 2*lobits) << FP::mantissa_bits;
    xhi |= exponent_hi;
    // subtract hidden mantissa bit
    realvec_t fhi = as_float(xhi) - RV(FP::as_float(exponent_hi));
    // add largest negative number again
    fhi -= RV(R(FP::signbit_mask));
    // Ensure that the converted low and high bits are calculated
    // separately, since a real_t doesn't have enough precision to
    // hold all the bits of an int_t
    fhi.barrier();
    
    // Combine results
    return flo + fhi;
  }
  
  
  
  template<typename realvec_t>
  typename realvec_t::intvec_t
  mathfuncs<realvec_t>::vml_convert_int(realvec_t x)
  {
    // Handle overflow
    // int_t min_int = FP::signbit_mask;
    // int_t max_int = ~FP::signbit_mask;
    // boolvec_t is_overflow = x < RV(R(min_int)) || x > RV(R(max_int));
    // Handle negative numbers
    boolvec_t is_negative = signbit(x);
    x = fabs(x);
    // Handle small numbers
    boolvec_t issmall = x < RV(1.0);
    
    intvec_t shift = ilogb(x) - IV(FP::mantissa_bits);
    boolvec_t shift_left = x > RV(std::ldexp(R(1.0), FP::mantissa_bits)); 
    intvec_t ix = as_int(x) & IV(FP::mantissa_mask);
    // add hidden mantissa bit
    ix |= U(1) << FP::mantissa_bits;
    // shift according to exponent (which may truncate)
    ix = ifthen(shift_left, ix << shift, ix >> -shift);
    
    // Handle small numbers
    ix = ifthen(issmall, IV(I(0)), ix);
    // Handle negative numbers
    ix = ifthen(is_negative, -ix, ix);
    // Handle overflow
    // ix = ifthen(is_overflow, IV(min_int), ix);
    
    return ix;
  }
  
  
  
  // Round to nearest integer, breaking ties using prevailing rounding
  // mode (default: round to even)
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_rint(realvec_t x)
  {
    realvec_t r = x;
    // Round by adding a large number, destroying all excess precision
    realvec_t offset = copysign(RV(std::ldexp(R(1.0), FP::mantissa_bits)), x);
    r += offset;
    // Ensure the rounding is not optimised away
    r.barrier();
    r -= offset;
    return r;
  }
  
  // Round to next integer above
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_ceil(realvec_t x)
  {
    // boolvec_t iszero = x == RV(0.0);
    // realvec_t offset = RV(0.5) - ldexp(fabs(x), I(-FP::mantissa_bits));
    // return ifthen(iszero, x, rint(x + offset));
    return ifthen(x<RV(0.0), trunc(x), vml_antitrunc(x));
  }
  
  // Round to next integer below
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_floor(realvec_t x)
  {
    // boolvec_t iszero = x == RV(0.0);
    // realvec_t offset = RV(0.5) - ldexp(fabs(x), I(-FP::mantissa_bits));
    // return ifthen(iszero, x, rint(x - offset));
    return ifthen(x<RV(0.0), vml_antitrunc(x), trunc(x));
  }
  
  // Round to nearest integer, breaking ties away from zero
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_round(realvec_t x)
  {
    // return copysign(floor(fabs(x)+RV(0.5)), x);
    return trunc(x + copysign(RV(0.5), x));
  }
  
  // Round to next integer towards zero
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_trunc(realvec_t x)
  {
    realvec_t x0 = x;
    x = fabs(x);
    boolvec_t istoosmall = x < RV(1.0);
    boolvec_t istoolarge = x >= RV(std::ldexp(R(1.0), FP::mantissa_bits));
    // Number of mantissa bits to keep
    intvec_t nbits = ilogb(x);
    // This is probably faster than a shift operation
    realvec_t mask = ldexp(RV(2.0), nbits) - RV(1.0);
    intvec_t imask = IV(FP::signbit_mask | FP::exponent_mask) | as_int(mask);
    realvec_t y = as_float(as_int(x) & imask);
    realvec_t r =
      copysign(ifthen(istoosmall, RV(0.0), ifthen(istoolarge, x, y)), x0);
    return r;
  }
  
  // Round to next integer away from zero
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_antitrunc(realvec_t x)
  {
    realvec_t x0 = x;
    x = fabs(x);
    boolvec_t iszero = x == RV(0.0);
    boolvec_t issmall = x <= RV(1.0);
    boolvec_t istoolarge =
      x > RV(std::ldexp(R(1.0), FP::mantissa_bits) - R(1.0));
    // Number of mantissa bits to keep
    intvec_t nbits = ilogb(x);
    // This is probably faster than a shift operation
    realvec_t mask = ldexp(RV(2.0), nbits) - RV(1.0);
    intvec_t imask = IV(FP::signbit_mask | FP::exponent_mask) | as_int(mask);
    realvec_t offset = RV(1.0) - ldexp(RV(1.0), nbits - IV(FP::mantissa_bits));
    offset.barrier();
    realvec_t y = as_float(as_int(x + offset) & imask);
    realvec_t r =
      copysign(ifthen(iszero, RV(0.0),
                      ifthen(issmall, RV(1.0),
                             ifthen(istoolarge, x, y))), x0);
    return r;
  }
  
  // Next machine representable number from x in direction y
  template<typename realvec_t>
  realvec_t mathfuncs<realvec_t>::vml_nextafter(realvec_t x, realvec_t y)
  {
    realvec_t dir = y - x;
    realvec_t offset = ldexp(RV(FP::epsilon()), ilogb(x));
    offset = copysign(offset, dir);
    offset = ifthen(convert_bool(as_int(x) & IV(FP::mantissa_mask)) ||
                    signbit(x) == signbit(offset),
                    offset,
                    offset * RV(0.5));
    realvec_t r = x + offset;
    real_t smallest_pos = std::ldexp(FP::min(), -FP::mantissa_bits);
    return ifthen(dir==RV(0.0), y,
                  ifthen(x==RV(0.0), copysign(RV(smallest_pos), dir), r));
  }
  
}; // namespace vecmathlib

#endif  // #ifndef MATHFUNCS_CONVERT_H
