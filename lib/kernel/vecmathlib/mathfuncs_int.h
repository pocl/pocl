// -*-C++-*-

#ifndef MATHFUNCS_INT_H
#define MATHFUNCS_INT_H

#include "mathfuncs_base.h"

#include <climits>

namespace vecmathlib {

template <typename realvec_t>
typename realvec_t::intvec_t mathfuncs<realvec_t>::vml_abs(intvec_t x) {
  return ifthen(isignbit(x), -x, x);
}

template <typename realvec_t>
typename realvec_t::intvec_t
mathfuncs<realvec_t>::vml_bitifthen(intvec_t x, intvec_t y, intvec_t z) {
  return (x & y) | (~x & z);
}

template <typename realvec_t>
typename realvec_t::intvec_t mathfuncs<realvec_t>::vml_clz(intvec_t x) {
  // These implementations return 8*sizeof(TYPE) when the input is 0

  // These explicit implementations are taken from
  // <http://aggregate.org/MAGIC/>:
  //
  // @techreport{magicalgorithms,
  //   author={Henry Gordon Dietz},
  //   title={{The Aggregate Magic Algorithms}},
  //   institution={University of Kentucky},
  //   howpublished={Aggregate.Org online technical report},
  //   date={2013-03-25},
  //   URL={http://aggregate.org/MAGIC/}
  // }

  int_t bits = CHAR_BIT * sizeof(int_t);
  if (bits > 1)
    x |= lsr(x, 1);
  if (bits > 2)
    x |= lsr(x, 2);
  if (bits > 4)
    x |= lsr(x, 4);
  if (bits > 8)
    x |= lsr(x, 8);
  if (bits > 16)
    x |= lsr(x, 16);
  if (bits > 32)
    x |= lsr(x, 32);
  if (bits > 64)
    x |= lsr(x, 64);
  assert(bits <= 128);
  return IV(I(bits)) - popcount(x);
}

template <typename realvec_t>
typename realvec_t::boolvec_t mathfuncs<realvec_t>::vml_isignbit(intvec_t x) {
  return x < IV(I(0));
}

template <typename realvec_t>
typename realvec_t::intvec_t mathfuncs<realvec_t>::vml_max(intvec_t x,
                                                           intvec_t y) {
  return ifthen(x >= y, x, y);
}

template <typename realvec_t>
typename realvec_t::intvec_t mathfuncs<realvec_t>::vml_min(intvec_t x,
                                                           intvec_t y) {
  return ifthen(x < y, x, y);
}

template <typename realvec_t>
typename realvec_t::intvec_t mathfuncs<realvec_t>::vml_popcount(intvec_t x) {
  // These explicit implementations are taken from
  // <http://aggregate.org/MAGIC/>:
  //
  // @techreport{magicalgorithms,
  //   author={Henry Gordon Dietz},
  //   title={{The Aggregate Magic Algorithms}},
  //   institution={University of Kentucky},
  //   howpublished={Aggregate.Org online technical report},
  //   date={2013-03-25},
  //   URL={http://aggregate.org/MAGIC/}
  // }

  int_t bits = CHAR_BIT * sizeof(int_t);

  // intvec_t x55 = IV(FP::replicate_byte(0x55));
  // intvec_t x33 = IV(FP::replicate_byte(0x33));
  // intvec_t x0f = IV(FP::replicate_byte(0x0f));
  intvec_t x55 = I(~U(0) / U(3));  // 0x0101...
  intvec_t x33 = I(~U(0) / U(5));  // 0x00110011...
  intvec_t x0f = I(~U(0) / U(17)); // 0b0000111100001111...

  x -= lsr(x, I(1)) & x55;
  x = (x & x33) + (lsr(x, I(2)) & x33);
  x += lsr(x, I(4));
  x &= x0f;
  if (bits > 8)
    x += lsr(x, I(8));
  if (bits > 16)
    x += lsr(x, I(16));
  if (bits > 32)
    x += lsr(x, I(32));
  if (bits > 64)
    x += lsr(x, I(64));
  assert(bits <= 128);
  return x & IV(I(0xff));
}

template <typename realvec_t>
typename realvec_t::intvec_t mathfuncs<realvec_t>::vml_rotate(intvec_t x,
                                                              int_t n) {
  int_t mask = CHAR_BIT * sizeof(int_t) - 1;
  intvec_t left = x << (n & mask);
  intvec_t right = lsr(x, -n & mask);
  return left | right;
}

template <typename realvec_t>
typename realvec_t::intvec_t mathfuncs<realvec_t>::vml_rotate(intvec_t x,
                                                              intvec_t n) {
  intvec_t mask = IV(I(CHAR_BIT * sizeof(int_t) - 1));
  intvec_t left = x << (n & mask);
  intvec_t right = lsr(x, -n & mask);
  return left | right;
}

}; // namespace vecmathlib

#endif // #ifndef MATHFUNCS_ASIN_H
