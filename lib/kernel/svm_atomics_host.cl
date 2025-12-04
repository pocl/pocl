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

#if defined(__opencl_c_ext_fp32_global_atomic_add)
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
#endif

#if defined(__opencl_c_ext_fp32_global_atomic_min_max)
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
