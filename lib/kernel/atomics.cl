/* OpenCL built-in library: atomic operations

   Copyright (c) 2012 Universidad Rey Juan Carlos
   
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

/*
This file implements old-style (pre OpenCL 2.0) atomics using 3.0 atomics.

Note: as explained in https://github.com/KhronosGroup/OpenCL-Docs/issues/353:

"In atomics prior to OpenCL 2.0 there was no definition of memory ordering and
scope and therefore old atomics behaved in implementation defined manner wrt
to these."

The reason for using explicit versions of 3.0 atomics below, is that
non-explicit versions require support for generic AS, memory_order_seq_cst, and
device-scope, while the explicit versions only require support for
device-scope.
*/

// Repeat the content of this file several times with different values
// for Q, T, and U:
#if !defined(Q)

#  define Q __global
#  include "atomics.cl"
#  undef Q

#  define Q __local
#  include "atomics.cl"
#  undef Q

#elif !defined(T)

#  define T int
#  include "atomics.cl"
#  undef T

#  define T uint
#  include "atomics.cl"
#  undef T

#ifdef cl_khr_int64_base_atomics
#  define T long
#  include "atomics.cl"
#  undef T

#  define T ulong
#  include "atomics.cl"
#  undef T
#endif




// xchg is also supported for float as a special case
__attribute__((overloadable))
float atomic_xchg(volatile Q float *p, float val)
{
  int retval = atomic_xchg ((volatile Q int *)p, as_int(val));
  return as_float(retval);
}



#else

// basic

// read, add, store
__attribute__((overloadable))
T atomic_add(volatile Q T *p, T val)
{
  return __atomic_fetch_add (p, val, __ATOMIC_ACQ_REL);
}

__attribute__ ((overloadable)) T
atom_add (volatile Q T *p, T val)
{
  return __atomic_fetch_add (p, val, __ATOMIC_ACQ_REL);
}

// read, subtract, store
__attribute__((overloadable))
T atomic_sub(volatile Q T *p, T val)
{
  return __atomic_fetch_sub (p, val, __ATOMIC_ACQ_REL);
}

__attribute__ ((overloadable)) T
atom_sub (volatile Q T *p, T val)
{
  return __atomic_fetch_sub (p, val, __ATOMIC_ACQ_REL);
}

// read, increment, store
__attribute__((overloadable))
T atomic_inc(volatile Q T *p)
{
  return __atomic_fetch_add (p, (T)1, __ATOMIC_ACQ_REL);
}

__attribute__ ((overloadable)) T
atom_inc (volatile Q T *p)
{
  return __atomic_fetch_add (p, (T)1, __ATOMIC_ACQ_REL);
}

// read, decrement, store
__attribute__((overloadable))
T atomic_dec(volatile Q T *p)
{
  return __atomic_fetch_sub (p, (T)1, __ATOMIC_ACQ_REL);
}

__attribute__ ((overloadable)) T
atom_dec (volatile Q T *p)
{
  return __atomic_fetch_sub (p, (T)1, __ATOMIC_ACQ_REL);
}

__attribute__ ((overloadable)) T
atomic_and (volatile Q T *p, T val)
{
  return __atomic_fetch_and (p, val, __ATOMIC_ACQ_REL);
}

__attribute__ ((overloadable)) T
atom_and (volatile Q T *p, T val)
{
  return __atomic_fetch_and (p, val, __ATOMIC_ACQ_REL);
}

__attribute__ ((overloadable)) T
atomic_or (volatile Q T *p, T val)
{
  return __atomic_fetch_or (p, val, __ATOMIC_ACQ_REL);
}

__attribute__ ((overloadable)) T
atom_or (volatile Q T *p, T val)
{
  return __atomic_fetch_or (p, val, __ATOMIC_ACQ_REL);
}

__attribute__ ((overloadable)) T
atomic_xor (volatile Q T *p, T val)
{
  return __atomic_fetch_xor (p, val, __ATOMIC_ACQ_REL);
}

__attribute__ ((overloadable)) T
atom_xor (volatile Q T *p, T val)
{
  return __atomic_fetch_xor (p, val, __ATOMIC_ACQ_REL);
}

/**********************************************************************/

// read, swap, store
__attribute__ ((overloadable)) T
atomic_xchg (volatile Q T *p, T val)
{
  return __atomic_exchange_n (p, val, __ATOMIC_ACQ_REL);
}

__attribute__ ((overloadable)) T
atom_xchg (volatile Q T *p, T val)
{
  return __atomic_exchange_n (p, val, __ATOMIC_ACQ_REL);
}

// read, store
__attribute__((overloadable))
T atomic_cmpxchg(volatile Q T *p, T cmp, T val)
{
  __atomic_compare_exchange_n (p, &cmp, val, false, __ATOMIC_ACQ_REL,
                               __ATOMIC_ACQUIRE);
  return cmp;
}

__attribute__ ((overloadable)) T
atom_cmpxchg (volatile Q T *p, T cmp, T val)
{
  __atomic_compare_exchange_n (p, &cmp, val, false, __ATOMIC_ACQ_REL,
                               __ATOMIC_ACQUIRE);
  return cmp;
}

#if (__clang_major__ < 10)

__attribute__((overloadable))
T atomic_min (volatile Q T *p, T val)
{
  T min,old;
  do {
    old = min = *p;
    if (val < min)
      old = atomic_cmpxchg(p, min, val);
  } while (old != min);
  return old;
}

__attribute__((overloadable))
T atomic_max (volatile Q T *p, T val)
{
  T max,old;
  do {
    old = max = *p;
    if (val > max)
      old = atomic_cmpxchg(p, max, val);
  } while (old != max);
  return old;
}

#else

__attribute__ ((overloadable)) T
atomic_min (volatile Q T *p, T val)
{
  return __atomic_fetch_min (p, val, __ATOMIC_ACQ_REL);
}

__attribute__ ((overloadable)) T
atomic_max (volatile Q T *p, T val)
{
  return __atomic_fetch_max (p, val, __ATOMIC_ACQ_REL);
}

#endif

__attribute__ ((overloadable)) T
atom_min (volatile Q T *p, T val)
{
  return atomic_min (p, val);
}

__attribute__ ((overloadable)) T
atom_max (volatile Q T *p, T val)
{
  return atomic_max (p, val);
}

#endif
