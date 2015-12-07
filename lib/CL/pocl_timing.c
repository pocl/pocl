/* OpenCL runtime library: OS-dependent time routines

   Copyright (c) 2015 Michal Babej / Tampere University of Technology

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

#include "pocl_timing.h"


#ifdef HAVE_CLOCK_GETTIME
// clock_gettime is (at best) nanosec res
const size_t pocl_timer_resolution = 1;
#else
#  ifndef _MSC_VER
// gettimeofday() has (at best) microsec res
const size_t pocl_timer_resolution = 1000;
#  else
// the resolution of windows clock is "it depends"...
const size_t pocl_timer_resolution = 1000;
#  endif
#endif


uint64_t pocl_gettime_ns() {
  uint64_t res = 0;

#ifndef _MSC_VER

# ifdef HAVE_CLOCK_GETTIME
  struct timespec timespec;

#  ifdef __linux__
  clock_gettime(CLOCK_MONOTONIC_RAW, &timespec);
#  elif defined(__DragonFly__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
  clock_gettime(CLOCK_UPTIME_FAST, &timespec);
#  elif defined(__MACH__)
  /* TODO test */
  /* clock_get_system_nanotime() in kern/clock.h
   * nanotime(struct timespec) in mach/mach_time.h */
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  timespec.tv_sec = mts.tv_sec;
  timespec.tv_nsec = mts.tv_nsec;
#  else
  clock_gettime(CLOCK_REALTIME, &ts);
#  endif

  res = timespec.tv_sec * 1000000000UL;
  return (res + timespec.tv_nsec);
# else /* HAVE_CLOCK_GETTIME */
  struct timeval current;
  gettimeofday(&current, NULL);
  return ((uint64_t)current.tv_sec * 1000000 + current.tv_usec)*1000;
# endif

#else  /* MSVC */
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  res |= ft.dwHighDateTime;
  res <<= 32;
  res |= ft.dwLowDateTime;
  res -= 11644473600000000Ui64;
  res /= 10;
  return res;
#endif
}
