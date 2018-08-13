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

#include "config.h"

#ifndef _MSC_VER
#  define _DEFAULT_SOURCE
#  define __POSIX_VISIBLE 200112L
#  ifndef _POSIX_C_SOURCE
#    define _POSIX_C_SOURCE 200112L
#  endif
#  include <inttypes.h>
#  if defined(HAVE_CLOCK_GETTIME) || defined(__APPLE__)
#    include <time.h>
#  else
#    include <sys/time.h>
#  endif
#  ifdef __MACH__
#    include <mach/clock.h>
#    include <mach/mach.h>
#  endif
#  include <sys/resource.h>
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#  include <stdint.h>
#  include <stddef.h> // size_t
#endif

#include "pocl_timing.h"

#ifdef HAVE_CLOCK_GETTIME
// clock_gettime is (at best) nanosec res
const unsigned pocl_timer_resolution = 1;
#else
#  ifndef _MSC_VER
// gettimeofday() has (at best) microsec res
const unsigned pocl_timer_resolution = 1000;
#  else
// the resolution of windows clock is "it depends"...
const unsigned pocl_timer_resolution = 1000;
#  endif
#endif


uint64_t pocl_gettimemono_ns() {

#ifdef HAVE_CLOCK_GETTIME
  struct timespec timespec;
# ifdef __linux__
#  ifdef CLOCK_MONOTONIC_RAW 
  clock_gettime(CLOCK_MONOTONIC_RAW, &timespec);
#  else
#   warning Using clock_gettime with CLOCK_MONOTONIC for monotonic clocks
  clock_gettime(CLOCK_MONOTONIC, &timespec);
#  endif
# elif defined(__DragonFly__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__FreeBSD_kernel__)
  clock_gettime(CLOCK_UPTIME_FAST, &timespec);
# else
# warning Using clock_gettime with CLOCK_REALTIME for monotonic clocks
  clock_gettime(CLOCK_REALTIME, &timespec);
# endif
  return (((uint64_t)timespec.tv_sec * 1000000000UL) + timespec.tv_nsec);


#elif defined(__APPLE__)
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  return ((mts.tv_sec * 1000000000UL) + mts.tv_nsec);

#elif defined(_WIN32)
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  res |= ft.dwHighDateTime;
  res <<= 32;
  res |= ft.dwLowDateTime;
  res -= 11644473600000000Ui64;
  res /= 10;
  return res;

#else
  struct timeval current;
  gettimeofday(&current, NULL);
  return ((uint64_t)current.tv_sec * 1000000 + current.tv_usec)*1000;

#endif
}

int pocl_gettimereal(int *year, int *mon, int *day, int *hour, int *min, int *sec, int* nanosec)
{
#if defined(HAVE_CLOCK_GETTIME) || defined(__APPLE__) || defined(HAVE_GETTIMEOFDAY)
  struct tm t;
  struct timespec timespec;
  time_t sec_input;

#if defined(HAVE_CLOCK_GETTIME)
  clock_gettime(CLOCK_REALTIME, &timespec);
  *nanosec = timespec.tv_nsec;
  sec_input = timespec.tv_sec;
#elif defined(__APPLE__)
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  *nanosec = mts.tv_nsec;
  sec_input = mts.tv_sec;
#else /* gettimeofday */
  struct timeval current;
  gettimeofday(&current, NULL);
  *nanosec = (uint64_t)current.tv_sec * 1000000;
  sec_input = current.tv_usec;
#endif
  gmtime_r(&sec_input, &t);
  *year = (t.tm_year + 1900);
  *mon = t.tm_mon;
  *day = t.tm_mday;
  *hour = t.tm_hour;
  *min = t.tm_min;
  *sec = t.tm_sec;
  return 0;

#elif defined(_WIN32)
  FILETIME ft;
  GetSystemTimeAsFileTime(&ft);
  res |= ft.dwHighDateTime;
  res <<= 32;
  res |= ft.dwLowDateTime;
  res -= 11644473600000000Ui64;
  res /= 10;
  // TODO finish this
  return 1;
#else
#error Unknown system variant
#endif

}
