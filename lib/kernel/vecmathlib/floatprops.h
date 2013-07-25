// -*-C++-*-

#ifndef FLOATPROPS_H
#define FLOATPROPS_H

#include "floattypes.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>



namespace vecmathlib {
  
  // A structure describing various properties of a floating point
  // type. Most properties are already described in numeric_limits, so
  // we inherit it.
  template<typename real_t>
  struct floatprops {
    // Some interesting properties are:
    //    min
    //    max
    //    digits
    //    epsilon
    //    min_exponent
    //    max_exponent
  };
  
  
  
  // Properties of fp8
  template<>
  struct floatprops<fp8> {
    typedef fp8 real_t;
    typedef std::int8_t int_t;
    typedef std::uint8_t uint_t;
    
    static char const* name() { return "fp8"; }
    
    // Definitions that might come from numeric_limits<> instead:
    static int min() { __builtin_unreachable(); }
    static int max() { __builtin_unreachable(); }
    static int const digits = 4;
    static int epsilon() { __builtin_unreachable(); }
    static int const min_exponent = -6;
    static int const max_exponent = 7;
    
    // Ensure the sizes match
    static_assert(sizeof(real_t) == sizeof(int_t), "int_t has wrong size");
    static_assert(sizeof(real_t) == sizeof(uint_t), "uint_t has wrong size");
    
    // Number of bits in internal representation
    static int const bits = 8 * sizeof(real_t);
    static int const mantissa_bits = digits - 1;
    static int const signbit_bits = 1;
    static int const exponent_bits = bits - mantissa_bits - signbit_bits;
    static int const exponent_offset = 2 - min_exponent;
    static_assert(mantissa_bits + exponent_bits + signbit_bits == bits,
                  "error in bit counts");
    static uint_t const mantissa_mask = (uint_t(1) << mantissa_bits) - 1;
    static uint_t const exponent_mask =
      ((uint_t(1) << exponent_bits) - 1) << mantissa_bits;
    static uint_t const signbit_mask = uint_t(1) << (bits-1);
    static_assert((mantissa_mask & exponent_mask & signbit_mask) == uint_t(0),
                  "error in masks");
    static_assert((mantissa_mask | exponent_mask | signbit_mask) ==
                  uint_t(~uint_t(0)),
                  "error in masks");
    
    // Re-interpret bit patterns
    static real_t as_float(int_t x)
    {
      real_t res;
      std::memcpy(&res, &x, sizeof res);
      return res;
    }
    static int_t as_int(real_t x)
    {
      int_t res;
      std::memcpy(&res, &x, sizeof res);
      return res;
    }
    
    // Convert values (truncate)
    static real_t convert_float(int_t x) { __builtin_unreachable(); }
    static int_t convert_int(real_t x) { __builtin_unreachable(); }
  };
  
  
  
  // Properties of fp16
  template<>
  struct floatprops<fp16> {
    typedef fp16 real_t;
    typedef std::int16_t int_t;
    typedef std::uint16_t uint_t;
    
    static char const* name() { return "fp16"; }
    
    // Definitions that might come from numeric_limits<> instead:
    static int min() { __builtin_unreachable(); }
    static int max() { __builtin_unreachable(); }
    static int const digits = 11;
    static int epsilon() { __builtin_unreachable(); }
    static int const min_exponent = -14;
    static int const max_exponent = 15;
    
    // Ensure the sizes match
    static_assert(sizeof(real_t) == sizeof(int_t), "int_t has wrong size");
    static_assert(sizeof(real_t) == sizeof(uint_t), "uint_t has wrong size");
    
    // Number of bits in internal representation
    static int const bits = 8 * sizeof(real_t);
    static int const mantissa_bits = digits - 1;
    static int const signbit_bits = 1;
    static int const exponent_bits = bits - mantissa_bits - signbit_bits;
    static int const exponent_offset = 2 - min_exponent;
    static_assert(mantissa_bits + exponent_bits + signbit_bits == bits,
                  "error in bit counts");
    static uint_t const mantissa_mask = (uint_t(1) << mantissa_bits) - 1;
    static uint_t const exponent_mask =
      ((uint_t(1) << exponent_bits) - 1) << mantissa_bits;
    static uint_t const signbit_mask = uint_t(1) << (bits-1);
    static_assert((mantissa_mask & exponent_mask & signbit_mask) == uint_t(0),
                  "error in masks");
    static_assert((mantissa_mask | exponent_mask | signbit_mask) ==
                  uint_t(~uint_t(0)),
                  "error in masks");
    
    // Re-interpret bit patterns
    static real_t as_float(int_t x)
    {
      real_t res;
      std::memcpy(&res, &x, sizeof res);
      return res;
    }
    static int_t as_int(real_t x)
    {
      int_t res;
      std::memcpy(&res, &x, sizeof res);
      return res;
    }
    
    // Convert values (truncate)
    static real_t convert_float(int_t x) { __builtin_unreachable(); }
    static int_t convert_int(real_t x) { __builtin_unreachable(); }
  };
  
  
  
  // Properties of float
  template<>
  struct floatprops<float>: std::numeric_limits<float> {
    typedef float real_t;
    typedef std::int32_t int_t;
    typedef std::uint32_t uint_t;
    
    static char const* name() { return "float"; }
    
    // Ensure the internal representation is what we expect
    static_assert(is_signed, "real_t is not signed");
    static_assert(radix==2, "real_t is not binary");
    
    // Ensure the sizes match
    static_assert(sizeof(real_t) == sizeof(int_t), "int_t has wrong size");
    static_assert(sizeof(real_t) == sizeof(uint_t), "uint_t has wrong size");
    
    // Number of bits in internal representation
    static int const bits = 8 * sizeof(real_t);
    static int const mantissa_bits = digits - 1;
    static int const signbit_bits = 1;
    static int const exponent_bits = bits - mantissa_bits - signbit_bits;
    static int const exponent_offset = 2 - min_exponent;
    static_assert(mantissa_bits + exponent_bits + signbit_bits == bits,
                  "error in bit counts");
    static uint_t const mantissa_mask = (uint_t(1) << mantissa_bits) - 1;
    static uint_t const exponent_mask =
      ((uint_t(1) << exponent_bits) - 1) << mantissa_bits;
    static uint_t const signbit_mask = uint_t(1) << (bits-1);
    static_assert((mantissa_mask & exponent_mask & signbit_mask) == uint_t(0),
                  "error in masks");
    static_assert((mantissa_mask | exponent_mask | signbit_mask) == ~uint_t(0),
                  "error in masks");
    
    // Re-interpret bit patterns
    static real_t as_float(int_t x)
    {
      real_t res;
      std::memcpy(&res, &x, sizeof res);
      return res;
    }
    static int_t as_int(real_t x)
    {
      int_t res;
      std::memcpy(&res, &x, sizeof res);
      return res;
    }
    
    // Convert values (truncate)
    static real_t convert_float(int_t x) { return real_t(x); }
    static int_t convert_int(real_t x) { return int_t(x); }
  };
  
  
  
  // Properties of double
  template<>
  struct floatprops<double>: std::numeric_limits<double> {
    typedef double real_t;
    typedef std::int64_t int_t;
    typedef std::uint64_t uint_t;
    
    static char const* name() { return "double"; }
    
    // Ensure the internal representation is what we expect
    static_assert(is_signed, "real_t is not signed");
    static_assert(radix==2, "real_t is not binary");
    
    // Ensure the sizes match
    static_assert(sizeof(real_t) == sizeof(int_t), "int_t has wrong size");
    static_assert(sizeof(real_t) == sizeof(uint_t), "uint_t has wrong size");
    
    // Number of bits in internal representation
    static int const bits = 8 * sizeof(real_t);
    static int const mantissa_bits = digits - 1;
    static int const signbit_bits = 1;
    static int const exponent_bits = bits - mantissa_bits - signbit_bits;
    static int const exponent_offset = 2 - min_exponent;
    static_assert(mantissa_bits + exponent_bits + signbit_bits == bits,
                  "error in bit counts");
    static uint_t const mantissa_mask = (uint_t(1) << mantissa_bits) - 1;
    static uint_t const exponent_mask =
      ((uint_t(1) << exponent_bits) - 1) << mantissa_bits;
    static uint_t const signbit_mask = uint_t(1) << (bits-1);
    static_assert((mantissa_mask & exponent_mask & signbit_mask) == uint_t(0),
                  "error in masks");
    static_assert((mantissa_mask | exponent_mask | signbit_mask) == ~uint_t(0),
                  "error in masks");
    
    // Re-interpret bit patterns
    static real_t as_float(int_t x)
    {
      real_t res;
      std::memcpy(&res, &x, sizeof res);
      return res;
    }
    static int_t as_int(real_t x)
    {
      int_t res;
      std::memcpy(&res, &x, sizeof res);
      return res;
    }
    
    // Convert values (truncate)
    static real_t convert_float(int_t x) { return real_t(x); }
    static int_t convert_int(real_t x) { return int_t(x); }
  };
  
  
  
  // We are adding the (unused) type RV here to avoid name mangling
  // problems. On some systems, the vector size does not enter into
  // the mangled name (!), leading to duplicate function definitions.
  template<typename RV, typename V, typename E>
  E get_elt(const V& v, const int n)
  {
    const size_t s = sizeof(E);
    E e;
    // assert(n>=0 and s*n<sizeof(V));
    std::memcpy(&e, &((const char*)&v)[s*n], s);
    return e;
  }
  
  template<typename RV, typename V, typename E>
  V& set_elt(V& v, const int n, const E e)
  {
    const size_t s = sizeof(E);
    // assert(n>=0 and s*n<sizeof(V));
    std::memcpy(&((char*)&v)[s*n], &e, s);
    return v;
  }
  
} // namespace vecmathlib

#endif  // #ifndef FLOATPROPS_H
