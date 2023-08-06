/* quarded_queue.hh - a simple thread-safe queue

   Copyright (c) 2019-2023 Jan Solanti / Tampere University

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

#ifndef POCL_REMOTE_REQUEST_QUEUE_HH
#define POCL_REMOTE_REQUEST_QUEUE_HH

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

template <class T> class GuardedQueue {
  std::deque<T> q;
  std::mutex m;
  std::condition_variable cond;

public:
  void reset() {
    std::unique_lock<std::mutex> lock(m);
    q = std::deque<T>();
  }
  void push(T item) {
    {
      std::unique_lock<std::mutex> lock(m);
      q.push_front(item);
    }
    cond.notify_one();
  }
  T pop() {
    std::unique_lock<std::mutex> lock(m);
    if (q.empty())
      return nullptr;
    else {
      T tmp = q.back();
      q.pop_back();
      return tmp;
    }
  }
  void wait_cond() {
    auto now = std::chrono::system_clock::now();
    std::chrono::duration<unsigned long> d(3);
    now += d;
    std::unique_lock<std::mutex> lock(m);
    cond.wait_until(lock, now);
  }
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
