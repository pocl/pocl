/* OpenCL runtime library: time measurement utility functions
   implemented using platform-specific C APIs

   Copyright (c) 2015 Michal Babej / Tampere University of Technology
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

#include "config.h"

#ifndef _WIN32
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
#  ifdef __APPLE__
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

uint64_t
pocl_gettimer_resolution ()
{
#ifdef HAVE_CLOCK_GETTIME
// clock_gettime is (at best) nanosec res
return 1;
#else
#  ifndef _MSC_VER
// gettimeofday() has (at best) microsec res
return 1000;
#  else
// the resolution of windows clock is "it depends"...
return 10000;
#  endif
#endif
}

uint64_t pocl_gettimemono_ns() {

#ifdef HAVE_CLOCK_GETTIME
  struct timespec timespec;
# ifdef CLOCK_MONOTONIC_RAW /* Linux */
  clock_gettime(CLOCK_MONOTONIC_RAW, &timespec);
# elif defined(CLOCK_UPTIME_FAST) /* FreeBSD, DragonFlyBSD, etc */
  clock_gettime(CLOCK_UPTIME_FAST, &timespec);
# elif defined(CLOCK_MONOTONIC) /* POSIX 2008, NetBSD, etc */
  clock_gettime(CLOCK_MONOTONIC, &timespec);
# else /* older POSIX didn't define CLOCK_MONOTONIC */
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
  uint64_t res = 0;
  GetSystemTimeAsFileTime(&ft);
  res |= ft.dwHighDateTime;
  res <<= 32;
  res |= ft.dwLowDateTime;
  res -= 11644473600000000ULL;
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
  *mon = t.tm_mon + 1;
  *day = t.tm_mday;
  *hour = t.tm_hour;
  *min = t.tm_min;
  *sec = t.tm_sec;
  return 0;

#elif defined(_WIN32)
  SYSTEMTIME st;
  FILETIME t;
  GetSystemTimeAsFileTime (&t);
  FileTimeToSystemTime (&t, &st);
  uint64_t st_nanosec = (t.dwLowDateTime % 10000000) * 100;

  *year = (int)st.wYear;
  *mon = (int)st.wMonth;
  *day = (int)st.wDay;
  *hour = (int)st.wHour;
  *min = (int)st.wMinute;
  *sec = (int)st.wSecond;
  *nanosec = (long)st_nanosec;
  return 0;
#else
#error Unknown system variant
#endif

}
