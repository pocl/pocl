/* OpenCL built-in library: OpenCL 2.0 Atomics (C11 subset)

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

   These implementations merely lower the call to a device-specific
   call (prefixed with "__pocl_atomic_")

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
#  include "svm_atomics.cl"
#  undef ARG2_AS
#  undef Q
#  undef QUAL

#  define Q __local
#  define QUAL(f) f ## __local
#  define ARG2_AS private
#  include "svm_atomics.cl"
#  undef ARG2_AS
#  undef Q
#  undef QUAL

#ifdef __opencl_c_generic_address_space

#  define Q __generic
#  define QUAL(f) f ## __generic
#  define ARG2_AS generic
#  include "svm_atomics.cl"
#  undef ARG2_AS
#  undef Q
#  undef QUAL

#endif

#elif !defined(ATOMIC_TYPE)

bool _CL_OVERLOADABLE
atomic_flag_test_and_set_explicit (volatile Q atomic_flag *object,
                                   memory_order order, memory_scope scope)
{
  return QUAL (__pocl_atomic_flag_test_and_set) (object, order, scope);
}

bool _CL_OVERLOADABLE atomic_flag_test_and_set_explicit ( volatile Q atomic_flag  *object ,
  memory_order order)
{
  return atomic_flag_test_and_set_explicit(object, order, memory_scope_device);
}

bool _CL_OVERLOADABLE
atomic_flag_test_and_set (volatile Q atomic_flag *object)
{
  return atomic_flag_test_and_set_explicit (object, memory_order_seq_cst);
}

void _CL_OVERLOADABLE
atomic_flag_clear_explicit (volatile Q atomic_flag *object, memory_order order,
                            memory_scope scope)
{
  return QUAL (__pocl_atomic_flag_clear) (object, order, scope);
}

void _CL_OVERLOADABLE atomic_flag_clear_explicit ( volatile Q atomic_flag  *object ,
  memory_order order)
{
  atomic_flag_clear_explicit(object, order, memory_scope_device);
}

void _CL_OVERLOADABLE
atomic_flag_clear (volatile Q atomic_flag *object)
{
  atomic_flag_clear_explicit (object, memory_order_seq_cst);
}

#  define ATOMIC_TYPE atomic_int
#  define NONATOMIC_TYPE int
#  define IS_INT
#  include "svm_atomics.cl"
#  undef IS_INT
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#  define ATOMIC_TYPE atomic_uint
#  define NONATOMIC_TYPE uint
#  define IS_UINT
#  include "svm_atomics.cl"
#  undef IS_UINT
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#  define ATOMIC_TYPE atomic_float
#  define NONATOMIC_TYPE float
#  define NON_INTEGER
#  include "svm_atomics.cl"
#  undef NON_INTEGER
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#if defined(cl_khr_int64_base_atomics) && defined(cl_khr_int64_extended_atomics)

#  define ATOMIC_TYPE atomic_long
#  define NONATOMIC_TYPE long
#  define INV_TYPE ulong
#  include "svm_atomics.cl"
#  undef INV_TYPE
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#  define ATOMIC_TYPE atomic_ulong
#  define NONATOMIC_TYPE ulong
#  define INV_TYPE long
#  include "svm_atomics.cl"
#  undef INV_TYPE
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#endif

#ifdef cl_khr_fp64

#  define ATOMIC_TYPE atomic_double
#  define NONATOMIC_TYPE double
#  define NON_INTEGER
#  include "svm_atomics.cl"
#  undef NON_INTEGER
#  undef ATOMIC_TYPE
#  undef NONATOMIC_TYPE

#endif

#else

void _CL_OVERLOADABLE
atomic_store_explicit (volatile Q ATOMIC_TYPE *object, NONATOMIC_TYPE desired,
                       memory_order order, memory_scope scope)
{
  QUAL (__pocl_atomic_store) (object, desired, order, scope);
}

void _CL_OVERLOADABLE atomic_store_explicit (  volatile Q ATOMIC_TYPE  *object,
                              NONATOMIC_TYPE  desired,
                              memory_order order)
{
  atomic_store_explicit(object, desired, order, memory_scope_device);
}

void _CL_OVERLOADABLE
atomic_store (volatile Q ATOMIC_TYPE *object, NONATOMIC_TYPE desired)
{
  atomic_store_explicit (object, desired, memory_order_seq_cst);
}

void _CL_OVERLOADABLE atomic_init (volatile Q ATOMIC_TYPE *object, NONATOMIC_TYPE value)
{
  atomic_store_explicit(object, value, memory_order_seq_cst);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_load_explicit (volatile Q ATOMIC_TYPE *object, memory_order order,
                      memory_scope scope)
{
  return QUAL (__pocl_atomic_load) (object, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_load_explicit ( volatile Q ATOMIC_TYPE  *object,
  memory_order order)
{
  return atomic_load_explicit(object, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_load (volatile Q ATOMIC_TYPE *object)
{
  return atomic_load_explicit (object, memory_order_seq_cst);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_exchange_explicit (volatile Q ATOMIC_TYPE *object,
                          NONATOMIC_TYPE desired, memory_order order,
                          memory_scope scope)
{
  return QUAL (__pocl_atomic_exchange) (object, desired, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_exchange_explicit (volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  desired,
  memory_order order)
{
  return atomic_exchange_explicit(object, desired, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_exchange (volatile Q ATOMIC_TYPE *object, NONATOMIC_TYPE desired)
{
  return atomic_exchange_explicit (object, desired, memory_order_seq_cst);
}

bool _CL_OVERLOADABLE
atomic_compare_exchange_strong_explicit (volatile Q ATOMIC_TYPE *object,
                                         ARG2_AS NONATOMIC_TYPE *expected,
                                         NONATOMIC_TYPE desired,
                                         memory_order success,
                                         memory_order failure,
                                         memory_scope scope)
{
  return QUAL (__pocl_atomic_compare_exchange_strong) (object, expected, desired,
                                                     success, failure, scope);
}

bool _CL_OVERLOADABLE atomic_compare_exchange_strong_explicit ( volatile Q ATOMIC_TYPE  *object,
  ARG2_AS NONATOMIC_TYPE  *expected,
  NONATOMIC_TYPE  desired,
  memory_order success,
  memory_order failure)
{
  return atomic_compare_exchange_strong_explicit(
        object, expected, desired, success, failure, memory_scope_device);
}

bool _CL_OVERLOADABLE
atomic_compare_exchange_strong (volatile Q ATOMIC_TYPE *object,
                                ARG2_AS NONATOMIC_TYPE *expected,
                                NONATOMIC_TYPE desired)
{
  return atomic_compare_exchange_strong_explicit (
      object, expected, desired, memory_order_seq_cst, memory_order_seq_cst);
}

bool _CL_OVERLOADABLE
atomic_compare_exchange_weak_explicit (volatile Q ATOMIC_TYPE *object,
                                       ARG2_AS NONATOMIC_TYPE *expected,
                                       NONATOMIC_TYPE desired,
                                       memory_order success,
                                       memory_order failure,
                                       memory_scope scope)
{
  return QUAL (__pocl_atomic_compare_exchange_weak) (object, expected, desired,
                                                   success, failure, scope);
}

bool _CL_OVERLOADABLE atomic_compare_exchange_weak_explicit ( volatile Q ATOMIC_TYPE  *object,
  ARG2_AS NONATOMIC_TYPE  *expected,
  NONATOMIC_TYPE  desired,
  memory_order success,
  memory_order failure)
{
  return atomic_compare_exchange_weak_explicit(
        object, expected, desired, success, failure, memory_scope_device);
}

bool _CL_OVERLOADABLE
atomic_compare_exchange_weak (volatile Q ATOMIC_TYPE *object,
                              ARG2_AS NONATOMIC_TYPE *expected,
                              NONATOMIC_TYPE desired)
{
  return atomic_compare_exchange_weak_explicit (
      object, expected, desired, memory_order_seq_cst, memory_order_seq_cst);
}

#ifndef NON_INTEGER

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_add_explicit (volatile Q ATOMIC_TYPE *object,
                           NONATOMIC_TYPE operand, memory_order order,
                           memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_add) (object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_add_explicit ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_add_explicit(object, operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_add (volatile Q ATOMIC_TYPE *object, NONATOMIC_TYPE operand)
{
  return atomic_fetch_add_explicit (object, operand, memory_order_seq_cst);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_sub_explicit (volatile Q ATOMIC_TYPE *object,
                           NONATOMIC_TYPE operand, memory_order order,
                           memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_sub) (object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_sub_explicit ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_sub_explicit(object, operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_sub (volatile Q ATOMIC_TYPE *object, NONATOMIC_TYPE operand)
{
  return atomic_fetch_sub_explicit (object, operand, memory_order_seq_cst);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_or_explicit (volatile Q ATOMIC_TYPE *object,
                          NONATOMIC_TYPE operand, memory_order order,
                          memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_or) (object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_or_explicit ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_or_explicit(object, operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_or (volatile Q ATOMIC_TYPE *object, NONATOMIC_TYPE operand)
{
  return atomic_fetch_or_explicit (object, operand, memory_order_seq_cst);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_xor_explicit (volatile Q ATOMIC_TYPE *object,
                           NONATOMIC_TYPE operand, memory_order order,
                           memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_xor) (object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_xor_explicit ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_xor_explicit(object, operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_xor (volatile Q ATOMIC_TYPE *object, NONATOMIC_TYPE operand)
{
  return atomic_fetch_xor_explicit (object, operand, memory_order_seq_cst);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_and_explicit (volatile Q ATOMIC_TYPE *object,
                           NONATOMIC_TYPE operand, memory_order order,
                           memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_and) (object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_and_explicit ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_and_explicit(object, operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_and (volatile Q ATOMIC_TYPE *object, NONATOMIC_TYPE operand)
{
  return atomic_fetch_and_explicit (object, operand, memory_order_seq_cst);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_min_explicit (volatile Q ATOMIC_TYPE *object,
                           NONATOMIC_TYPE operand, memory_order order,
                           memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_min) (object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_min_explicit ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_min_explicit(object, operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_min (volatile Q ATOMIC_TYPE *object, NONATOMIC_TYPE operand)
{
  return atomic_fetch_min_explicit (object, operand, memory_order_seq_cst);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_max_explicit (volatile Q ATOMIC_TYPE *object,
                           NONATOMIC_TYPE operand, memory_order order,
                           memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_max) (object, operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_max_explicit ( volatile Q ATOMIC_TYPE  *object,
  NONATOMIC_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_max_explicit(object, operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_max (volatile Q ATOMIC_TYPE *object, NONATOMIC_TYPE operand)
{
  return atomic_fetch_max_explicit (object, operand, memory_order_seq_cst);
}


#ifdef INV_TYPE

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_add_explicit (volatile Q ATOMIC_TYPE *object,
                           INV_TYPE operand, memory_order order,
                           memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_add) (object, (NONATOMIC_TYPE)operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_add_explicit ( volatile Q ATOMIC_TYPE  *object,
  INV_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_add_explicit(object, (NONATOMIC_TYPE)operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_add (volatile Q ATOMIC_TYPE *object, INV_TYPE operand)
{
  return atomic_fetch_add_explicit (object, (NONATOMIC_TYPE)operand, memory_order_seq_cst);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_sub_explicit (volatile Q ATOMIC_TYPE *object,
                           INV_TYPE operand, memory_order order,
                           memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_sub) (object, (NONATOMIC_TYPE)operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_sub_explicit ( volatile Q ATOMIC_TYPE  *object,
  INV_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_sub_explicit(object, (NONATOMIC_TYPE)operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_sub (volatile Q ATOMIC_TYPE *object, INV_TYPE operand)
{
  return atomic_fetch_sub_explicit (object, (NONATOMIC_TYPE)operand, memory_order_seq_cst);
}


NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_or_explicit (volatile Q ATOMIC_TYPE *object,
                          INV_TYPE operand, memory_order order,
                          memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_or) (object, (NONATOMIC_TYPE)operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_or_explicit ( volatile Q ATOMIC_TYPE  *object,
  INV_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_or_explicit(object, (NONATOMIC_TYPE)operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_or (volatile Q ATOMIC_TYPE *object, INV_TYPE operand)
{
  return atomic_fetch_or_explicit (object, (NONATOMIC_TYPE)operand, memory_order_seq_cst);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_xor_explicit (volatile Q ATOMIC_TYPE *object,
                           INV_TYPE operand, memory_order order,
                           memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_xor) (object, (NONATOMIC_TYPE)operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_xor_explicit ( volatile Q ATOMIC_TYPE  *object,
  INV_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_xor_explicit(object, (NONATOMIC_TYPE)operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_xor (volatile Q ATOMIC_TYPE *object, INV_TYPE operand)
{
  return atomic_fetch_xor_explicit (object, (NONATOMIC_TYPE)operand, memory_order_seq_cst);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_and_explicit (volatile Q ATOMIC_TYPE *object,
                           INV_TYPE operand, memory_order order,
                           memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_and) (object, (NONATOMIC_TYPE)operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_and_explicit ( volatile Q ATOMIC_TYPE  *object,
  INV_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_and_explicit(object, (NONATOMIC_TYPE)operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_and (volatile Q ATOMIC_TYPE *object, INV_TYPE operand)
{
  return atomic_fetch_and_explicit (object, (NONATOMIC_TYPE)operand, memory_order_seq_cst);
}


NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_min_explicit (volatile Q ATOMIC_TYPE *object,
                           INV_TYPE operand, memory_order order,
                           memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_min) (object, (NONATOMIC_TYPE)operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_min_explicit ( volatile Q ATOMIC_TYPE  *object,
  INV_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_min_explicit(object, (NONATOMIC_TYPE)operand, order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_min (volatile Q ATOMIC_TYPE *object, INV_TYPE operand)
{
  return atomic_fetch_min_explicit (object, (NONATOMIC_TYPE)operand, memory_order_seq_cst);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_max_explicit (volatile Q ATOMIC_TYPE *object,
                           INV_TYPE operand, memory_order order,
                           memory_scope scope)
{
  return QUAL (__pocl_atomic_fetch_max) (object, (NONATOMIC_TYPE)operand, order, scope);
}

NONATOMIC_TYPE _CL_OVERLOADABLE atomic_fetch_max_explicit ( volatile Q ATOMIC_TYPE  *object,
  INV_TYPE  operand,
  memory_order order)
{
  return atomic_fetch_max_explicit(object, operand, (NONATOMIC_TYPE)order, memory_scope_device);
}

NONATOMIC_TYPE _CL_OVERLOADABLE
atomic_fetch_max (volatile Q ATOMIC_TYPE *object, INV_TYPE operand)
{
  return atomic_fetch_max_explicit (object, (NONATOMIC_TYPE)operand, memory_order_seq_cst);
}

#endif

#endif

#endif
