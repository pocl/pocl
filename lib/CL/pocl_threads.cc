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

#include "pocl_threads_cpp.hh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <map>

// #define DEBUG_THREADS
#ifdef DEBUG_THREADS
#include <iostream>
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
  std::thread T;
};

static _pocl_lock_t pocl_init_lock_m;
// extern "C" is needed because MSVC mangles global variables.
extern "C" pocl_lock_t pocl_init_lock = &pocl_init_lock_m;

static std::map<std::thread::id, pocl_thread_t> PoclThreadMap;

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

void pocl_thread_create(pocl_thread_t *T, void *(*F)(void *), void *Arg) {
  pocl_thread_t ThrPtr = new _pocl_thread_t;
  //assert (NT);
  ThrPtr->T = std::thread(F, Arg);
  *T = ThrPtr;
  PoclThreadMap.insert(std::make_pair(ThrPtr->T.get_id(), ThrPtr));
}

void pocl_thread_join(pocl_thread_t T) {
  // assert (T);
  T->T.join();
  delete T;
  PoclThreadMap.erase(T->T.get_id());
}


pocl_thread_t pocl_thread_self() {
  auto It = PoclThreadMap.find(std::this_thread::get_id());
  if (It == PoclThreadMap.end())
    return nullptr;
  else
    return It->second;
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
