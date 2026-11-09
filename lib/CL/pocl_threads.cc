/* OpenCL runtime library: utility functions for thread operations,
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

#include "pocl_debug.h"
#include "pocl_threads_cpp.hh"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>

// #define DEBUG_THREADS
#ifdef DEBUG_THREADS
#include <iostream>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <process.h>
#include <windows.h>
#endif

struct _pocl_barrier_t {
public:
  _pocl_barrier_t(unsigned long ctr) : counter(ctr){};
  ~_pocl_barrier_t() = default;
  void wait();

private:
  unsigned long counter;
  std::mutex lock;
  std::condition_variable cond;
};

struct _pocl_lock_t {
  std::mutex lock;
  _pocl_lock_t() = default;
  ~_pocl_lock_t() = default;
  _pocl_lock_t(_pocl_lock_t &&oth) = delete;
  _pocl_lock_t(const _pocl_lock_t &oth) = delete;
};

struct _pocl_cond_t {
  std::condition_variable cond;
};

struct _pocl_thread_t {
#ifdef _WIN32
  HANDLE Handle;
#else
  std::thread T;
#endif
  void *(*Func)(void *);
  void *Arg;
};

static thread_local pocl_thread_t PoclThreadSelf = nullptr;

static _pocl_lock_t pocl_init_lock_m;
// extern "C" is needed because MSVC mangles global variables.
extern "C" pocl_lock_t pocl_init_lock = &pocl_init_lock_m;


void pocl_mutex_lock(pocl_lock_t L) {
  L->lock.lock();
#ifdef DEBUG_THREADS
  std::cerr << "MUTEX LOCKED: " << (void *)L << std::endl;
#endif
}

void pocl_mutex_unlock(pocl_lock_t L) {
  L->lock.unlock();
#ifdef DEBUG_THREADS
  std::cerr << "MUTEX UNLOCKED: " << (void *)L << std::endl;
#endif
}

void pocl_mutex_init(pocl_lock_t *L) {
  *L = new _pocl_lock_t;
#ifdef DEBUG_THREADS
  std::cerr << "MUTEX INIT: " << (void *)*L << std::endl;
#endif
}

void pocl_mutex_destroy(pocl_lock_t *L) {
  if (*L != nullptr) {
    delete *L;
  }
  *L = nullptr;
#ifdef DEBUG_THREADS
  std::cerr << "MUTEX DESTROY: " << (void *)*L << std::endl;
#endif
}

void pocl_cond_init(pocl_cond_t *C) {
  *C = new _pocl_cond_t;
#ifdef DEBUG_THREADS
  std::cerr << "CREATE COND: " << (void *)*C << std::endl;
#endif
}

void pocl_cond_destroy(pocl_cond_t *C) {
  if (*C != nullptr) {
    delete *C;
  }
  *C = nullptr;
#ifdef DEBUG_THREADS
  std::cerr << "DESTROY COND: " << (void *)*C << std::endl;
#endif
}

void pocl_cond_signal(pocl_cond_t C) {
  C->cond.notify_one();
#ifdef DEBUG_THREADS
  std::cerr << "COND SIGNAL: " << (void *)C << std::endl;
#endif
}

void pocl_cond_broadcast(pocl_cond_t C) {
  C->cond.notify_all();
#ifdef DEBUG_THREADS
  std::cerr << "COND BROAD: " << (void *)C << std::endl;
#endif
}

void pocl_cond_wait(pocl_cond_t C, pocl_lock_t L) {
#ifdef DEBUG_THREADS
  std::cerr << "COND WAIT: " << (void *)C << " | LOCK " << (void *)L
            << std::endl;
#endif
  // the lock is expected to be locked by the user outside this call
  std::unique_lock<std::mutex> UL{L->lock, std::adopt_lock};
  C->cond.wait(UL);
  // we must return with the lock still locked
  UL.release();
}

void pocl_cond_timedwait(pocl_cond_t C, pocl_lock_t L, unsigned long msec) {
#ifdef DEBUG_THREADS
  std::cerr << "TIMED COND WAIT: " << (void *)C << " | LOCK " << (void *)L
            << std::endl;
#endif
  // the lock is expected to be locked by the user outside this call
  std::unique_lock<std::mutex> UL{L->lock, std::adopt_lock};
  C->cond.wait_for(UL, std::chrono::milliseconds(msec));
  // we must return with the lock still locked
  UL.release();
}

static void runThread(pocl_thread_t T) {
  PoclThreadSelf = T;
  T->Func(T->Arg);
}

#ifdef _WIN32
static unsigned __stdcall runThreadWin32(void *T) {
  runThread((pocl_thread_t)T);
  return 0;
}

// The stack reservation of threads created with the default size, as
// recorded in the executable.
static size_t defaultStackReserve() {
  const char *Image = (const char *)GetModuleHandleW(nullptr);
  auto *Dos = (const IMAGE_DOS_HEADER *)Image;
  auto *Nt = (const IMAGE_NT_HEADERS *)(Image + Dos->e_lfanew);
  return Nt->OptionalHeader.SizeOfStackReserve;
}
#endif

void pocl_thread_create(pocl_thread_t *T, void *(*F)(void *), void *Arg) {
  pocl_thread_create_with_stack_size(T, F, Arg, 0);
}

void pocl_thread_create_with_stack_size(pocl_thread_t *T, void *(*F)(void *),
                                        void *Arg, size_t MinStackSize) {
  pocl_thread_t Thr = new _pocl_thread_t;
  Thr->Func = F;
  Thr->Arg = Arg;
  *T = Thr;
#ifdef _WIN32
  // Only reserve the address space, like POSIX threads do; without the flag,
  // the size would be committed up front.
  unsigned StackSize = 0;
  unsigned Flags = 0;
  if (MinStackSize > defaultStackReserve()) {
    StackSize = (unsigned)MinStackSize;
    Flags = STACK_SIZE_PARAM_IS_A_RESERVATION;
  }
  Thr->Handle = (HANDLE)_beginthreadex(nullptr, StackSize, runThreadWin32, Thr,
                                       Flags, nullptr);
  if (Thr->Handle == 0)
    POCL_ABORT("_beginthreadex failed: %s\n", strerror(errno));
#else
  // std::thread can't set the stack size; threads get the platform default.
  Thr->T = std::thread(runThread, Thr);
#endif
}

void pocl_thread_join(pocl_thread_t T) {
#ifdef _WIN32
  if (WaitForSingleObject(T->Handle, INFINITE) != WAIT_OBJECT_0)
    POCL_ABORT("WaitForSingleObject failed: %lu\n", GetLastError());
  CloseHandle(T->Handle);
#else
  T->T.join();
#endif
  delete T;
}

pocl_thread_t pocl_thread_self() { return PoclThreadSelf; }

size_t pocl_get_thread_stack_size() {
#ifdef _WIN32
  // The stack spans from the base of the reservation a local lives in to the
  // top recorded in the thread information block. (GetCurrentThreadStackLimits
  // needs Windows 8 headers, which not all toolchains target.)
  MEMORY_BASIC_INFORMATION Info;
  if (VirtualQuery(&Info, &Info, sizeof(Info)) == 0)
    return 0;
  const NT_TIB *Tib = (const NT_TIB *)NtCurrentTeb();
  return (const char *)Tib->StackBase - (const char *)Info.AllocationBase;
#else
  return 0;
#endif
}

void _pocl_barrier_t::wait() {
  std::unique_lock<std::mutex> L(lock);
  --counter;
  if (counter == 0)
    cond.notify_all();
  while (counter > 0) {
    cond.wait(L);
  }
}

void pocl_barrier_init(pocl_barrier_t *B, unsigned long N) {
  _pocl_barrier_t *L = new _pocl_barrier_t(N);
  *B = L;
}

void pocl_barrier_wait(pocl_barrier_t B) {
  //  assert(B);
  B->wait();
}

void pocl_barrier_destroy(pocl_barrier_t *B) {
  if (*B != nullptr) {
    delete *B;
  }
  *B = nullptr;
}
