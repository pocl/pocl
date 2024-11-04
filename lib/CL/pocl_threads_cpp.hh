/* OpenCL runtime library: header for utility functions for thread operations
   implemented using C++11 standard library

   Copyright (c) 2024 Michal Babej / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#ifndef POCL_THREADS_CPP_H
#define POCL_THREADS_CPP_H

#include "pocl_export.h"

typedef struct _pocl_barrier_t *pocl_barrier_t;
typedef struct _pocl_lock_t *pocl_lock_t;
typedef struct _pocl_cond_t *pocl_cond_t;
typedef struct _pocl_thread_t *pocl_thread_t;

#if defined(__GNUC__) || defined(__clang__)

#define POCL_ATOMIC_ADD(x, val) __atomic_add_fetch(&x, val, __ATOMIC_SEQ_CST);
#define POCL_ATOMIC_INC(x) __atomic_add_fetch(&x, 1, __ATOMIC_SEQ_CST)
#define POCL_ATOMIC_DEC(x) __atomic_sub_fetch(&x, 1, __ATOMIC_SEQ_CST)
#define POCL_ATOMIC_LOAD(x) __atomic_load_n(&x, __ATOMIC_SEQ_CST)
#define POCL_ATOMIC_STORE(x, val) __atomic_store_n(&x, val, __ATOMIC_SEQ_CST)
#define POCL_ATOMIC_CAS(ptr, oldval, newval)                                   \
  __sync_val_compare_and_swap(ptr, oldval, newval)

#elif defined(_WIN32)
#define POCL_ATOMIC_ADD(x, val) InterlockedAdd64(&x, val);
#define POCL_ATOMIC_INC(x) InterlockedIncrement64(&x)
#define POCL_ATOMIC_DEC(x) InterlockedDecrement64(&x)
#define POCL_ATOMIC_LOAD(x) InterlockedOr64(&x, 0)
#define POCL_ATOMIC_STORE(x, val) InterlockedExchange64(&x, val)
#define POCL_ATOMIC_CAS(ptr, oldval, newval)                                   \
  InterlockedCompareExchange64(ptr, newval, oldval)
#else
#error Need atomic_inc() builtin for this compiler
#endif

#define POCL_SET_THREAD_STACK_SIZE(N) (void *)0
#define POCL_GET_THREAD_STACK_SIZE() 0

#ifdef __cplusplus
extern "C" {
#endif

POCL_EXPORT
void pocl_mutex_init(pocl_lock_t *L);
POCL_EXPORT
void pocl_mutex_destroy(pocl_lock_t *L);
POCL_EXPORT
void pocl_mutex_lock(pocl_lock_t L);
POCL_EXPORT
void pocl_mutex_unlock(pocl_lock_t L);

POCL_EXPORT
void pocl_cond_init(pocl_cond_t *C);
POCL_EXPORT
void pocl_cond_destroy(pocl_cond_t *C);
POCL_EXPORT
void pocl_cond_signal(pocl_cond_t C);
POCL_EXPORT
void pocl_cond_broadcast(pocl_cond_t C);
POCL_EXPORT
void pocl_cond_wait(pocl_cond_t C, pocl_lock_t L);
POCL_EXPORT
void pocl_cond_timedwait(pocl_cond_t C, pocl_lock_t L, unsigned long msec);

POCL_EXPORT
void pocl_thread_create(pocl_thread_t *T, void *(*F)(void *), void *Arg);
POCL_EXPORT
pocl_thread_t pocl_thread_self();
POCL_EXPORT
void pocl_thread_join(pocl_thread_t T);

POCL_EXPORT
void pocl_barrier_init(pocl_barrier_t *B, unsigned long N);
POCL_EXPORT
void pocl_barrier_wait(pocl_barrier_t B);
POCL_EXPORT
void pocl_barrier_destroy(pocl_barrier_t *B);

#ifdef __cplusplus
}
#endif

#define POCL_LOCK(__LOCK__) pocl_mutex_lock(__LOCK__)
#define POCL_UNLOCK(__LOCK__) pocl_mutex_unlock(__LOCK__)
#define POCL_INIT_LOCK(__LOCK__) pocl_mutex_init(&__LOCK__)
#define POCL_DESTROY_LOCK(__LOCK__) pocl_mutex_destroy(&__LOCK__)

#define POCL_INIT_COND(c) pocl_cond_init(&c)
#define POCL_DESTROY_COND(c) pocl_cond_destroy(&c)
#define POCL_SIGNAL_COND(c) pocl_cond_signal(c)
#define POCL_BROADCAST_COND(c) pocl_cond_broadcast(c)
#define POCL_WAIT_COND(c, m) pocl_cond_wait(c, m)

// unsigned long t = time in milliseconds to wait
// TODO: should ignore ETIMEDOUT
#define POCL_TIMEDWAIT_COND(c, m, t) pocl_cond_timedwait(c, m, t)

#define POCL_CREATE_THREAD(thr, func, arg) pocl_thread_create(&thr, func, arg)
#define POCL_JOIN_THREAD(thr) pocl_thread_join(thr)
#define POCL_THREAD_SELF() pocl_thread_self()

#define POCL_INIT_BARRIER(bar, number) pocl_barrier_init(&bar, number)
#define POCL_WAIT_BARRIER(bar) pocl_barrier_wait(bar)
#define POCL_DESTROY_BARRIER(bar) pocl_barrier_destroy(&bar)

#endif
