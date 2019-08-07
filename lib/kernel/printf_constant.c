/* OpenCL built-in library: printf_constant()

   Copyright (c) 2013 Pekka Jääskeläinen

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
/**
 * Implementation of printf where the format string
 * (note, later also %s) reside in the constant address space.
 *
 * Implemented as a wrapper that copies the format string to
 * the private space (0) and calls a system vprintf.
 */

//#include <stddef.h>
#include <stdarg.h>

#define NULL ((void*)0)

#ifdef __TCE_V1__
/* TCE includes need tceops.h to be generated just for _TCE_STDOUT in
   the stdio.h header. Work around this by hiding the __TCE_V1__ macro. */
#undef __TCE_V1__

/* The newlib headers of TCE expect to see valid long and double (which in 32bit
   TCE are defined to be 32bit). */
#undef long
#define long long
#undef double
#define double double

#endif

/* AS 0 is required for the prototypes, otherwise they get assigned
 * the generic AS (#4) */

#define OCL_C_AS __attribute__((address_space(0)))

int vprintf(const char *, __builtin_va_list);
int fflush(void *stream);

#undef printf
#define MAX_FORMAT_STR_SIZE 2048
int
printf(__attribute__((address_space(POCL_ADDRESS_SPACE_CONSTANT)))
           char* restrict fmt, ...)
{
  /* http://www.pagetable.com/?p=298 */
  int retval = 0;
  va_list ap;
  va_start(ap, fmt);
  char format_private[MAX_FORMAT_STR_SIZE];
  for (int i = 0; i < MAX_FORMAT_STR_SIZE; ++i)
    {
      format_private[i] = fmt[i];
      if (fmt[i] == '\0')
        {
	  break;
        }
    }
  retval = vprintf(format_private, ap);
  fflush(NULL);
  va_end(ap);
  return retval;
}
