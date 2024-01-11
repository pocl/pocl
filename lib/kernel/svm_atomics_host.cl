/* OpenCL built-in library: OpenCL 2.0 Atomics (C11 subset) implementation for host device

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

   This relies on Clang's C11 atomic builtins.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#ifndef _SVM_ATOMICS_H
#include "svm_atomics.h"
#define _SVM_ATOMICS_H
#endif





#if !defined(Q)

#  define Q __global
#  define QUAL(f) f ## __global
#  define ARG2_AS private
#  include "svm_atomics_host.cl"
#  undef ARG2_AS
#  undef Q
#  undef QUAL

#  define Q __local
#  define QUAL(f) f ## __local
#  define ARG2_AS private
#  include "svm_atomics_host.cl"
#  undef ARG2_AS
#  undef Q
#  undef QUAL

#ifdef __opencl_c_generic_address_space

#  define Q __generic
#  define QUAL(f) f ## __generic
#  define ARG2_AS generic
#  include "svm_atomics_host.cl"
#  undef ARG2_AS
#  undef Q
#  undef QUAL

#endif

#elif !defined(ATOMIC_TYPE)

bool _CL_OVERLOADABLE QUAL(__pocl_atomic_flag_test_and_set) ( volatile Q atomic_int  *object ,
  memory_order order,
  memory_scope scope)
{
  return __opencl_atomic_exchange(object, 1, order, scope);
}

void _CL_OVERLOADABLE QUAL(__pocl_atomic_flag_clear) ( volatile Q atomic_int  *object ,
  memory_order order,
  memory_scope scope)
{
  __opencl_atomic_store(object, 0, order, scope);
}

#  define ATOMIC_TYPE atomic_int
#  define NONATOMIC_TYPE int
#  include "svm_atomics_host.cl"
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#  define ATOMIC_TYPE atomic_uint
#  define NONATOMIC_TYPE uint
#  include "svm_atomics_host.cl"
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#  define ATOMIC_TYPE atomic_float
#  define NONATOMIC_TYPE float
#  define NON_INTEGER
#  define ATOMIC_LOOP(OP, ADDR, OPERAND, ORDER, SCOPE) \
  union \
  { \
    uint u32; \
    float f32; \
  } next, expected, current; \
  __builtin_memcpy_inline(&current.f32, (const Q void *)ADDR, sizeof(uint)); \
  do \
    { \
      expected.f32 = current.f32;    \
      next.f32 = OP(expected.f32, OPERAND); \
      current.u32                      \
          = QUAL(__pocl_atomic_compare_exchange_strong) ((volatile Q atomic_uint *)ADDR, \
                                     (private uint *)&expected.u32, \
                                     next.u32, \
                                     ORDER, ORDER, SCOPE); \
    } \
  while (current.u32 != expected.u32); \
  return current.f32;
#  include "svm_atomics_host.cl"
#  undef ATOMIC_LOOP
#  undef NON_INTEGER
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#if defined(cl_khr_int64_base_atomics) && defined(cl_khr_int64_extended_atomics)

#  define ATOMIC_TYPE atomic_long
#  define NONATOMIC_TYPE long
#  include "svm_atomics_host.cl"
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#  define ATOMIC_TYPE atomic_ulong
#  define NONATOMIC_TYPE ulong
#  include "svm_atomics_host.cl"
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#endif

#ifdef cl_khr_fp64

#  define ATOMIC_TYPE atomic_double
#  define NONATOMIC_TYPE double
#  define NON_INTEGER
#  define ATOMIC_LOOP(OP, ADDR, OPERAND, ORDER, SCOPE) \
  union \
  { \
    ulong u64; \
    double f64; \
  } next, expected, current; \
  __builtin_memcpy_inline(&current.f64, (const Q void *)ADDR, sizeof(ulong)); \
  do \
    { \
      expected.f64 = current.f64;    \
      next.f64 = OP(expected.f64, OPERAND); \
      current.u64                      \
          = QUAL(__pocl_atomic_compare_exchange_strong) ((volatile Q atomic_ulong *)ADDR, \
                                     (private ulong *)&expected.u64, \
                                     next.u64, \
                                     ORDER, ORDER, SCOPE); \
    } \
  while (current.u64 != expected.u64); \
  return current.f64;
#  include "svm_atomics_host.cl"
#  undef ATOMIC_LOOP
#  undef NON_INTEGER
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#endif

#else

/************************************************************************/

_CL_OVERLOADABLE void QUAL(__pocl_atomic_store)( volatile Q ATOMIC_TYPE  *object,
                              NONATOMIC_TYPE  desired,
                              memory_order order,
                              memory_scope scope)
{
  __opencl_atomic_store(object, desired, order, scope);
}

_CL_OVERLOADABLE NONATOMIC_TYPE QUAL(__pocl_atomic_load) ( volatile Q ATOMIC_TYPE  *object,
                                        memory_order order,
                                        memory_scope scope)
{
  return __opencl_atomic_load(object, order, scope);
}


_CL_OVERLOADABLE NONATOMIC_TYPE QUAL(__pocl_atomic_exchange) ( volatile Q ATOMIC_TYPE  *object,
                                            NONATOMIC_TYPE  desired,
                                            memory_order order,
                                            memory_scope scope)
{
  return __opencl_atomic_exchange(object, desired, order, scope);
}

bool _CL_OVERLOADABLE QUAL(__pocl_atomic_compare_exchange_strong) ( volatile Q ATOMIC_TYPE  *object,
  ARG2_AS NONATOMIC_TYPE  *expected,
  NONATOMIC_TYPE  desired,
  memory_order success,
  memory_order failure,
  memory_scope scope)
{
  return __opencl_atomic_compare_exchange_strong(object,  expected, desired, success, failure, scope);
}

bool _CL_OVERLOADABLE QUAL(__pocl_atomic_compare_exchange_weak) ( volatile Q ATOMIC_TYPE  *object,
  ARG2_AS NONATOMIC_TYPE  *expected,
  NONATOMIC_TYPE  desired,
  memory_order success,
  memory_order failure,
  memory_scope scope)
{
  return __opencl_atomic_compare_exchange_weak(object,  expected, desired, success, failure, scope);
}

/* available on integers, but also floats with cl_ext_float_atomics;
 * these might need different implementation depending on LLVM version;
 * atomic add/sub on floats is available since LLVM 13;
 * atomic min/max on floats is available since LLVM 17; */
#if (defined(NON_INTEGER) && defined(cl_ext_float_atomics))

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_fadd)(NONATOMIC_TYPE a, NONATOMIC_TYPE b) { return a+b; }
NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_fsub)(NONATOMIC_TYPE a, NONATOMIC_TYPE b) { return a-b; }

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_add) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __opencl_atomic_fetch_add(object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_sub) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __opencl_atomic_fetch_sub(object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_min) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
#if (__clang_major__ >= 17)
  return __opencl_atomic_fetch_min(object, operand, order, scope);
#else
  ATOMIC_LOOP(fmin, object, operand, order, scope);
#endif
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_max) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
#if (__clang_major__ >= 17)
  return __opencl_atomic_fetch_max(object, operand, order, scope);
#else
  ATOMIC_LOOP(fmax, object, operand, order, scope);
#endif
}

#endif

#ifndef NON_INTEGER

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_add) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __opencl_atomic_fetch_add(object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_sub) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __opencl_atomic_fetch_sub(object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_or) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __opencl_atomic_fetch_or(object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_xor) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __opencl_atomic_fetch_xor(object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_and) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __opencl_atomic_fetch_and(object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_min) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __opencl_atomic_fetch_min(object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_max) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __opencl_atomic_fetch_max(object, operand, order, scope);
}

#endif

/************************************************************************/


#endif
