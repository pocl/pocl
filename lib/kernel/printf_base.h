/* OpenCL built-in library: printf_base.h

   Copyright (c) 2018 Michal Babej / Tampere University of Technology

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

#include <stdint.h>

/* printing largest double with %f formatter requires about ~1024 digits for
 * integral part, plus precision digits for decimal part, plus some space for
 * modifiers. */
#define BUFSIZE 1200
#define SMALL_BUF_SIZE 64

#define NULL ((void*)0)

/* This should be queried from the machine in the non-TAS case.
 * For now assume that if we are not using the fake address space
 * ids then we have a single address space. This version of printf
 * doesn't work with multiple address spaces anyways. */
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
#define OCL_CONSTANT_AS __attribute__ ((address_space (3)))
#else
#define OCL_CONSTANT_AS
#endif

typedef intptr_t ssize_t;

#ifdef cl_khr_int64

#define INT_T int64_t
#define UINT_T uint64_t
#define INT_T_MIN INT64_MIN
#define INT_T_MAX INT64_MAX

#else

#define INT_T int32_t
#define UINT_T uint32_t
#define INT_T_MIN INT32_MIN
#define INT_T_MAX INT32_MAX

#endif


#ifdef cl_khr_fp64
#define FLOAT_T double
#define FLOAT_INT_T int64_t
#define FLOAT_UINT_T uint64_t

#define NAN __builtin_nan ("1")
#define INFINITY (__builtin_inf())
#define SIGNBIT __builtin_signbit

#define EXPBITS      0x7ff0000000000000L
#define EXPBIAS      1023
#define EXPSHIFTBITS 52
#define MANTBITS     0x000fffffffffffffUL
#define LEADBIT      0x0010000000000000UL
#define MSB_NIBBLE   0x000f000000000000UL
#define MAX_NIBBLES  13

#define L_EXPONENT(x) (((x & EXPBITS) >> EXPSHIFTBITS) - EXPBIAS)
#define L_MANTISSA(x) (x & MANTBITS)
#define L_SIGNBIT(x) (x >> 63)
#define L_MSB_NIBBLE(x) (x & MSB_NIBBLE)

#else

#define FLOAT_T float
#define FLOAT_INT_T int32_t
#define FLOAT_UINT_T uint32_t

#define NAN __builtin_nanf ("1")
#define INFINITY (__builtin_inff())
#define SIGNBIT __builtin_signbitf

#define EXPBITS      0x7f800000
#define EXPBIAS      127
#define EXPSHIFTBITS 23
#define MANTBITS     0x007fffff
#define LEADBIT      0x00800000
#define MSB_NIBBLE   0x00780000
#define MAX_NIBBLES  6

#define L_EXPONENT(x) (((x & EXPBITS) >> EXPSHIFTBITS) - EXPBIAS)
#define L_MANTISSA(x) (x & MANTBITS)
#define L_SIGNBIT(x) (x >> 31)
#define L_MSB_NIBBLE(x) (x & MSB_NIBBLE)

#endif

/* Conversion flags */
typedef struct
{
  char zero : 1;        /**  Leading zeros */
  char alt : 1;         /**  alternate form */
  char align_left : 1;  /**  0 == align right (default), 1 == align left */
  char uc : 1;          /**  Upper case (for base16 only) */
  char always_sign : 1; /**  plus flag (always display sign) */
  char sign : 1;        /**   the actual sign **/
  char space : 1;       /** If the first character of a signed conversion is not
                          a sign, print a space */
  char nonzeroparam : 1; /** number to print is not zero */
} flags_t;

typedef struct
{
  char *bf;             /**  Buffer to output */
  char *restrict printf_buffer;
  size_t printf_buffer_index;
  size_t printf_buffer_capacity;
  int precision;       /**  field precision */
  unsigned width;      /**  field width */
  unsigned base;
  flags_t flags;
  char conv;
} param_t;

void __pocl_printf_putchw (param_t *p);

void __pocl_printf_putcf (param_t *p, char c);

void __pocl_printf_puts (param_t *p, const char *string);

void __pocl_printf_nonfinite (param_t *p, const char *ptr);

void __pocl_printf_puts_ljust (param_t *p, const char *string, size_t width,
                               ssize_t max_width);

void __pocl_printf_puts_rjust (param_t *p, const char *string, size_t width,
                               ssize_t max_width);

void __pocl_printf_ptr (param_t *p, const void *ptr);

void __pocl_printf_ulong (param_t *p, UINT_T u);

void __pocl_printf_long (param_t *p, INT_T i);

void __pocl_printf_float (param_t *p, FLOAT_T f);
