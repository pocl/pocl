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
#  define MIN __sync_fetch_and_min
#  define MAX __sync_fetch_and_max
#  include "atomics.cl"
#  undef T
#  undef MIN
#  undef MAX

#  define T uint
#  define MIN __sync_fetch_and_umin
#  define MAX __sync_fetch_and_umax
#  include "atomics.cl"
#  undef T
#  undef MIN
#  undef MAX


// xchg is also supported for float as a special case
__attribute__((overloadable))
float atomic_xchg(volatile Q float *p, float val)
{
  // NOTE: We compare the float as int here...
  return __atomic_exchange_n((volatile int*)p, val, __ATOMIC_RELAXED);
}

#else



// basic

// read, add, store
__attribute__((overloadable))
T atomic_add(volatile Q T *p, T val)
{
  return __sync_fetch_and_add((volatile T*)p, val, __ATOMIC_RELAXED);
}

// read, subtract, store
__attribute__((overloadable))
T atomic_sub(volatile Q T *p, T val)
{
  return __sync_fetch_and_sub(p, val, __ATOMIC_RELAXED);
}

// read, swap, store
__attribute__((overloadable))
T atomic_xchg(volatile Q T *p, T val)
{
  return __atomic_exchange_n(p, val, __ATOMIC_RELAXED);
}

// read, increment, store
__attribute__((overloadable))
T atomic_inc(volatile Q T *p)
{
  return atomic_add(p, (T)1);
}

// read, decrement, store
__attribute__((overloadable))
T atomic_dec(volatile Q T *p)
{
  return atomic_sub(p, (T)1);
}

// read, store
__attribute__((overloadable))
T atomic_cmpxchg(volatile Q T *p, T cmp, T val)
{
  __atomic_compare_exchange_n(p, &cmp, val, false,
                              __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  return cmp;
}

// extended

__attribute__((overloadable))
T atomic_min(volatile Q T *p, T val)
{
  return MIN((volatile T*)p, val);
}

__attribute__((overloadable))
T atomic_max(volatile Q T *p, T val)
{
  return MAX((volatile T*)p, val);
}

__attribute__((overloadable))
T atomic_and(volatile Q T *p, T val)
{
  return __sync_fetch_and_and(p, val, __ATOMIC_RELAXED);
}

__attribute__((overloadable))
T atomic_or(volatile Q T *p, T val)
{
  return __sync_fetch_and_or(p, val, __ATOMIC_RELAXED);
}

__attribute__((overloadable))
T atomic_xor(volatile Q T *p, T val)
{
  return __sync_fetch_and_xor(p, val, __ATOMIC_RELAXED);
}



#endif
