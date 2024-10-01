/* OpenCL runtime library: utility functions for thread operations
   implemented using POSIX threads (pthread.h)

   Copyright (c) 2023 Jan Solanti / Tampere University
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

#define _GNU_SOURCE
#include "pocl_debug.h"
#include "pocl_threads_c.h"

#include <errno.h>
#include <time.h>

pocl_lock_t pocl_init_lock = POCL_LOCK_INITIALIZER;

void
pocl_abort_on_pthread_error (int status, unsigned line, const char *func)
{
  if (status != 0)
    {
      POCL_MSG_PRINT2 (ERROR, func, line, "Error from pthread call:\n");
      POCL_ABORT ("PTHREAD ERROR in %s():%u: %s (%d)\n", func, line,
                  strerror (status), status);
    }
}

void
pocl_timed_wait (pocl_cond_t *c, pocl_lock_t *m, unsigned long usec)
{
  struct timespec now = { 0, 0 };
  clock_gettime (CLOCK_REALTIME, &now);

  unsigned long nsec = usec * 1000UL;
  if (now.tv_nsec + nsec < 1000000000UL)
    {
      now.tv_nsec += nsec;
    }
  else
    {
      now.tv_nsec = now.tv_nsec + nsec - 1000000000UL;
      now.tv_sec += 1;
    }

  PTHREAD_CHECK2 (ETIMEDOUT, pthread_cond_timedwait (c, m, &now));
}
