/* OpenCL built-in library: OpenCL 2.0 Atomics (C11 subset) implementation for host device

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

   This relies on Clang's C11 atomic builtins.

   Note: for some architectures, the host-specific llvm bitcode is used instead
   of this file (since Clang doesn't have proper builtins for 64bit min/max atomics,
   yet LLVM's atomicrmw can do them; using this file gives only limited min/max
   atomics).

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

#define CONV_ORDER(mo) ((mo==memory_order_relaxed) ? __ATOMIC_RELAXED : \
                       ((mo==memory_order_acquire) ? __ATOMIC_ACQUIRE : \
                       ((mo==memory_order_release) ? __ATOMIC_RELEASE : \
                       ((mo==memory_order_acq_rel) ? __ATOMIC_ACQ_REL : \
                                                     __ATOMIC_SEQ_CST ))))
#define _SVM_ATOMICS_H
#endif





#if !defined(Q)

#  define Q __global
#  define QUAL(f) f ## __global
#  include "svm_atomics_host.cl"
#  undef Q
#  undef QUAL

#  define Q __local
#  define QUAL(f) f ## __local
#  include "svm_atomics_host.cl"
#  undef Q
#  undef QUAL

#elif !defined(ATOMIC_TYPE)

bool _CL_OVERLOADABLE QUAL(__pocl_atomic_flag_test_and_set) ( volatile Q atomic_int  *object ,
  memory_order order,
  memory_scope scope)
{
  return __c11_atomic_exchange(object, 1, CONV_ORDER(order));
}

void _CL_OVERLOADABLE QUAL(__pocl_atomic_flag_clear) ( volatile Q atomic_int  *object ,
  memory_order order,
  memory_scope scope)
{
  __c11_atomic_store(object, 0, CONV_ORDER(order));
}

#  define ATOMIC_TYPE atomic_int
#  define NONATOMIC_TYPE int
#  define IS_INT
#  include "svm_atomics_host.cl"
#  undef IS_INT
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#  define ATOMIC_TYPE atomic_uint
#  define NONATOMIC_TYPE uint
#  define IS_UINT
#  include "svm_atomics_host.cl"
#  undef IS_UINT
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#  define ATOMIC_TYPE atomic_float
#  define NONATOMIC_TYPE float
#  define NON_INTEGER
#  include "svm_atomics_host.cl"
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
#  include "svm_atomics_host.cl"
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
  __c11_atomic_store(object, desired, CONV_ORDER(order));
}

_CL_OVERLOADABLE NONATOMIC_TYPE QUAL(__pocl_atomic_load) ( volatile Q ATOMIC_TYPE  *object,
                                        memory_order order,
                                        memory_scope scope)
{
  return __c11_atomic_load(object, CONV_ORDER(order));
}


_CL_OVERLOADABLE NONATOMIC_TYPE QUAL(__pocl_atomic_exchange) ( volatile Q ATOMIC_TYPE  *object,
                                            NONATOMIC_TYPE  desired,
                                            memory_order order,
                                            memory_scope scope)
{
  return __c11_atomic_exchange(object, desired, CONV_ORDER(order));
}

bool _CL_OVERLOADABLE QUAL(__pocl_atomic_compare_exchange_strong) ( volatile Q ATOMIC_TYPE  *object,
  private NONATOMIC_TYPE  *expected,
  NONATOMIC_TYPE  desired,
  memory_order success,
  memory_order failure,
  memory_scope scope)
{
  return __c11_atomic_compare_exchange_strong(object,  expected, desired, CONV_ORDER(success), CONV_ORDER(failure));
}

bool _CL_OVERLOADABLE QUAL(__pocl_atomic_compare_exchange_weak) ( volatile Q ATOMIC_TYPE  *object,
  private NONATOMIC_TYPE  *expected,
  NONATOMIC_TYPE  desired,
  memory_order success,
  memory_order failure,
  memory_scope scope)
{
  return __c11_atomic_compare_exchange_weak(object,  expected, desired, CONV_ORDER(success), CONV_ORDER(failure));
}

#ifndef NON_INTEGER

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_add) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __c11_atomic_fetch_add(object, operand, CONV_ORDER(order));
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_sub) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __c11_atomic_fetch_sub(object, operand, CONV_ORDER(order));
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_or) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __c11_atomic_fetch_or(object, operand, CONV_ORDER(order));
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_xor) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __c11_atomic_fetch_xor(object, operand, CONV_ORDER(order));
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_and) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
  return __c11_atomic_fetch_and(object, operand, CONV_ORDER(order));
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_min) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
#if defined(IS_INT)
  return __sync_fetch_and_min((volatile Q NONATOMIC_TYPE *)object, operand);
#elif defined(IS_UINT)
  return __sync_fetch_and_umin((volatile Q NONATOMIC_TYPE *)object, operand);
#else
  __builtin_trap();
  return 0;
#endif
}

NONATOMIC_TYPE _CL_OVERLOADABLE QUAL(__pocl_atomic_fetch_max) ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order,
  memory_scope scope)
{
#if defined(IS_INT)
  return __sync_fetch_and_max((volatile Q NONATOMIC_TYPE *)object, operand);
#elif defined(IS_UINT)
  return __sync_fetch_and_umax((volatile Q NONATOMIC_TYPE *)object, operand);
#else
  __builtin_trap();
  return 0;
#endif
}

#endif

/************************************************************************/


#endif
