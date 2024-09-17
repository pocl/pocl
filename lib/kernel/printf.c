/* OpenCL built-in library: printf()

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

#include <stdint.h>

/**************************************************************************/

#ifdef PRINTF_BUFFER_AS_ID
#define PRINTF_BUFFER_AS __attribute__ ((address_space (PRINTF_BUFFER_AS_ID)))
#else
#define PRINTF_BUFFER_AS
#endif

/* avoid prematurely inlining these functions;
 * they must exist until the Workgroup pass */
#define ATTRS __attribute__ ((noinline)) __attribute__ ((optnone))

extern PRINTF_BUFFER_AS char *_printf_buffer;
extern PRINTF_BUFFER_AS uint32_t *_printf_buffer_position;
extern uint32_t _printf_buffer_capacity;

#ifdef ENABLE_PRINTF_IMMEDIATE_FLUSH

/* flush the data in printf buffer to STDOUT */
/* definition exist on the host side only */

extern void __printf_flush_buffer (PRINTF_BUFFER_AS void *buffer,
                                   uint32_t bytes);

#endif

/* required by emitPrintfCall to allocate the storage for printf args.
 * this is the actual implementation */

PRINTF_BUFFER_AS void *ATTRS
__pocl_printf_alloc (PRINTF_BUFFER_AS char *__buffer,
                     PRINTF_BUFFER_AS uint32_t *__buffer_position,
                     uint32_t __buffer_capacity,
                     uint32_t bytes)
{
  if (*__buffer_position + bytes > __buffer_capacity)
    return (void *)0;

  PRINTF_BUFFER_AS char *retval = __buffer + *__buffer_position;
  *__buffer_position += bytes;
  return retval;
}

/* required by emitPrintfCall to allocate the storage for printf args.
 * this is a stub that will be replaced by __pocl_printf_alloc in Workgroup.cc
 */
PRINTF_BUFFER_AS void *ATTRS
__printf_alloc (uint32_t bytes)
{
  /* this ensures __pocl_printf_alloc is not optimized away */
  PRINTF_BUFFER_AS void *retval = __pocl_printf_alloc (
    _printf_buffer, _printf_buffer_position, _printf_buffer_capacity, bytes);

  /* this ensures the extern declaration is not optimized away*/
#ifdef ENABLE_PRINTF_IMMEDIATE_FLUSH
  __printf_flush_buffer (_printf_buffer, bytes);
#endif

  return retval;
}
