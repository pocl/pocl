/* OpenCL built-in library: printf()

   Copyright (c) 2013 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                      Perimeter Institute for Theoretical Physics

   Copyright (c) 2018 Michal Babej / Tampere University of Technology

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

#include "printf_base.h"

#include <stdarg.h>

#define OCL_C_AS

/* The OpenCL printf routine.
 *
 * For debugging
 * Use as: DEBUG_PRINTF((fmt, args...)) -- note double parentheses!

int printf(const char *format, ...);
#define DEBUG_PRINTF(args) (printf args)
*/

#define DEBUG_PRINTF(args) ((void)0)

/**************************************************************************/

#define DEFINE_PRINT_INTS(NAME, INT_TYPE, UINT_TYPE)                          \
  void __pocl_print_ints_##NAME (param_t *p, OCL_C_AS const void *vals,       \
                                 int n, int is_unsigned)                      \
  {                                                                           \
    DEBUG_PRINTF (("[printf:ints:n=%df]\n", n));                              \
    flags_t saved_user_flags = p->flags;                                      \
    for (size_t d = 0; d < n; ++d)                                            \
      {                                                                       \
        DEBUG_PRINTF (("[printf:ints:d=%d]\n", d));                           \
        p->flags = saved_user_flags;                                          \
        if (d != 0)                                                           \
          __pocl_printf_putcf (p, ',');                                       \
        if (is_unsigned)                                                      \
          __pocl_printf_ulong (                                               \
              p, (UINT_T) (((OCL_C_AS const UINT_TYPE *)vals)[d]));           \
        else                                                                  \
          __pocl_printf_long (                                                \
              p, (INT_T) (((OCL_C_AS const INT_TYPE *)vals)[d]));             \
      }                                                                       \
    DEBUG_PRINTF (("[printf:ints:done]\n"));                                  \
  }

DEFINE_PRINT_INTS (uchar, int8_t, uint8_t)
DEFINE_PRINT_INTS (ushort, int16_t, uint16_t)
DEFINE_PRINT_INTS (uint, int32_t, uint32_t)
#ifdef cl_khr_int64
DEFINE_PRINT_INTS (ulong, int64_t, uint64_t)
#endif

#undef DEFINE_PRINT_INTS

/**************************************************************************/

/* Note: NANs are printed always positive.
 * This is required to pass 1.2 conformance test. */
#define DEFINE_PRINT_FLOATS(FLOAT_TYPE)                                       \
  void __pocl_print_floats_##FLOAT_TYPE (param_t *p,                          \
                                         OCL_C_AS const void *vals, int n)    \
  {                                                                           \
    const char *NANs[2] = { "nan", "NAN" };                                   \
    const char *INFs[2] = { "inf", "INF" };                                   \
                                                                              \
    DEBUG_PRINTF (("[printf:floats:n=%d]\n", n));                             \
    flags_t saved_user_flags = p->flags;                                      \
    for (int d = 0; d < n; ++d)                                               \
      {                                                                       \
        DEBUG_PRINTF (("[printf:floats:d=%d]\n", d));                         \
        p->flags = saved_user_flags;                                          \
        if (d != 0)                                                           \
          __pocl_printf_putcf (p, ',');                                       \
        FLOAT_T val = *((OCL_C_AS const FLOAT_TYPE *)vals + d);               \
        const char *other = NULL;                                             \
        if (val != val)                                                       \
          other = NANs[p->flags.uc ? 1 : 0];                                  \
        if (val == (-INFINITY))                                               \
          {                                                                   \
            val = INFINITY;                                                   \
            p->flags.sign = 1;                                                \
          }                                                                   \
        if (val == (INFINITY))                                                \
          other = INFs[p->flags.uc ? 1 : 0];                                  \
        if (other)                                                            \
          __pocl_printf_nonfinite (p, other);                                 \
        else                                                                  \
          __pocl_printf_float (p, val);                                       \
      }                                                                       \
    DEBUG_PRINTF (("[printf:floats:done]\n"));                                \
  }

#ifdef cl_khr_fp16
DEFINE_PRINT_FLOATS (half)
#endif

DEFINE_PRINT_FLOATS (float)

#ifdef cl_khr_fp64
DEFINE_PRINT_FLOATS (double)
#endif

#undef DEFINE_PRINT_FLOATS

/**************************************************************************/
/**************************************************************************/
/**************************************************************************/

#define ERROR_STRING " printf format string error: 0x"

#define ERROR_NULL_AFTER_FORMAT_SIGN 0x11
#define ERROR_REPEATED_FLAG_MINUS 0x12
#define ERROR_REPEATED_FLAG_PLUS 0x13
#define ERROR_REPEATED_FLAG_SPACE 0x14
#define ERROR_REPEATED_FLAG_SHARP 0x15
#define ERROR_REPEATED_FLAG_ZERO 0x16

#define ERROR_FIELD_WIDTH_ZERO 0x17
#define ERROR_FIELD_WIDTH_OVERFLOW 0x18

#define ERROR_PRECISION_OVERFLOW 0x19

#define ERROR_VECTOR_LENGTH_ZERO 0x20
#define ERROR_VECTOR_LENGTH_OVERFLOW 0x21
#define ERROR_VECTOR_LENGTH_UNKNOWN 0x22

#define ERROR_VECTOR_LENGTH_WITHOUT_ELEMENT_SIZE 0x23
#define ERROR_HL_MODIFIER_USED_WITHOUT_VECTOR_LENGTH 0x24

#define ERROR_FLAGS_WITH_C_CONVERSION_SPECIFIER 0x25
#define ERROR_PRECISION_WITH_C_CONVERSION_SPECIFIER 0x25
#define ERROR_VECTOR_LENGTH_WITH_C_CONVERSION_SPECIFIER 0x25
#define ERROR_LENGTH_MODIFIER_WITH_C_CONVERSION_SPECIFIER 0x25

#define ERROR_FLAGS_WITH_S_CONVERSION_SPECIFIER 0x26
#define ERROR_VECTOR_LENGTH_WITH_S_CONVERSION_SPECIFIER 0x27
#define ERROR_LENGTH_MODIFIER_WITH_S_CONVERSION_SPECIFIER 0x28

#define ERROR_FLAGS_WITH_P_CONVERSION_SPECIFIER 0x29
#define ERROR_PRECISION_WITH_P_CONVERSION_SPECIFIER 0x30
#define ERROR_VECTOR_LENGTH_WITH_P_CONVERSION_SPECIFIER 0x31
#define ERROR_LENGTH_MODIFIER_WITH_P_CONVERSION_SPECIFIER 0x32

#define ERROR_UNKNOWN_CONVERSION_SPECIFIER 0x33

int
__pocl_printf_format_full (const PRINTF_FMT_STR_AS char *restrict format,
                           param_t *p, va_list ap)
{
  DEBUG_PRINTF (("[printf:format=%s]\n", format));
  char bf[BUFSIZE];
  p->bf = bf;
  char ch;
  unsigned errcode;

  while ((ch = *format++))
    {
      if (ch == '%')
        {

          ch = *format++;
          if (ch == 0)
            {
              errcode = ERROR_NULL_AFTER_FORMAT_SIGN;
              goto error;
            }

          if (ch == '%')
            {
              DEBUG_PRINTF (("[printf:%%]\n"));
              __pocl_printf_putcf (p, '%'); /* literal % */
            }
          else
            {
              DEBUG_PRINTF (("[printf:arg]\n"));
              // Flags
              flags_t flags;
              flags.align_left = 0;
              flags.sign = 0;
              flags.space = 0;
              flags.alt = 0;
              flags.zero = 0;
              flags.uc = 0;
              flags.always_sign = 0;
              for (;;)
                {
                  switch (ch)
                    {
                    case '-':
                      if (flags.align_left)
                        {
                          errcode = ERROR_REPEATED_FLAG_MINUS;
                          goto error;
                        }
                      flags.align_left = 1;
                      break;
                    case '+':
                      if (flags.always_sign)
                        {
                          errcode = ERROR_REPEATED_FLAG_PLUS;
                          goto error;
                        }
                      flags.always_sign = 1;
                      break;
                    case ' ':
                      if (flags.space)
                        {
                          errcode = ERROR_REPEATED_FLAG_SPACE;
                          goto error;
                        }
                      flags.space = 1;
                      break;
                    case '#':
                      if (flags.alt)
                        {
                          errcode = ERROR_REPEATED_FLAG_SHARP;
                          goto error;
                        }
                      flags.alt = 1;
                      break;
                    case '0':
                      if (flags.zero)
                        {
                          errcode = ERROR_REPEATED_FLAG_ZERO;
                          goto error;
                        }
                      if (flags.align_left == 0)
                        flags.zero = 1;
                      break;
                    default:
                      goto flags_done;
                    }
                  ch = *format++;
                }

            flags_done:;
              DEBUG_PRINTF (
                  ("[printf:flags:left=%d,plus=%d,space=%d,alt=%d,zero=%d]\n",
                   flags.align_left, flags.sign, flags.space, flags.alt,
                   flags.zero));

              /* Field width */
              size_t field_width = 0;
              while (ch >= '0' && ch <= '9')
                {
                  if (ch == '0' && field_width == 0)
                    {
                      errcode = ERROR_FIELD_WIDTH_ZERO;
                      goto error;
                    }
                  if (field_width > (INT_MAX - 9) / 10)
                    {
                      errcode = ERROR_FIELD_WIDTH_OVERFLOW;
                      goto error;
                    }
                  field_width = 10 * field_width + (ch - '0');
                  ch = *format++;
                }
              DEBUG_PRINTF (("[printf:width=%d]\n", field_width));

              /* Precision */
              int precision = -1;
              if (ch == '.')
                {
                  precision = 0;
                  ch = *format++;
                  while (ch >= '0' && ch <= '9')
                    {
                      if (precision > (INT_MAX - 9) / 10)
                        {
                          errcode = ERROR_PRECISION_OVERFLOW;
                          goto error;
                        }
                      precision = 10 * precision + (ch - '0');
                      ch = *format++;
                    }
                }
              DEBUG_PRINTF (("[printf:precision=%d]\n", precision));

              // Vector specifier
              size_t vector_length = 0;
              if (ch == 'v')
                {
                  ch = *format++;
                  while (ch >= '0' && ch <= '9')
                    {
                      if (ch == '0' && vector_length == 0)
                        {
                          errcode = ERROR_VECTOR_LENGTH_ZERO;
                          goto error;
                        }
                      if (vector_length > (INT_MAX - 9) / 10)
                        {
                          errcode = ERROR_VECTOR_LENGTH_OVERFLOW;
                          goto error;
                        }
                      vector_length = 10 * vector_length + (ch - '0');
                      ch = *format++;
                    }
                  if (!(vector_length == 2 || vector_length == 3
                        || vector_length == 4 || vector_length == 8
                        || vector_length == 16))
                    {
                      errcode = ERROR_VECTOR_LENGTH_UNKNOWN;
                      goto error;
                    }
                }
              DEBUG_PRINTF (("[printf:vector_length=%d]\n", vector_length));

              /* Length modifier */
              size_t length = 0;
              if (ch == 'h')
                {
                  ch = *format++;
                  if (ch == 'h')
                    {
                      ch = *format++;
                      length = 1; /* "hh" -> char */
                    }
                  else if (ch == 'l')
                    {
                      ch = *format++;
                      length = 4; /* "hl" -> int */
                    }
                  else
                    {
                      length = 2; /* "h" -> short */
                    }
                }
              else if (ch == 'l')
                {
                  ch = *format++;
                  length = 8; /* "l" -> long */
                }
              if (vector_length > 0 && length == 0)
                {
                  errcode = ERROR_VECTOR_LENGTH_WITHOUT_ELEMENT_SIZE;
                  goto error;
                }

              if (vector_length == 0 && length == 4)
                {
                  errcode = ERROR_HL_MODIFIER_USED_WITHOUT_VECTOR_LENGTH;
                  goto error;
                }

              if (vector_length == 0)
                vector_length = 1;

#ifdef DISABLE_VECTOR_PRINTF
              vector_length = 1;
#endif

              DEBUG_PRINTF (("[printf:length=%d]\n", length));

              p->flags = flags;
              p->conv = ch;
              p->width = field_width;
              p->precision = precision;

              switch (ch)
                {

                  /* Output integers */
                case 'd':
                case 'i':
                case 'o':
                case 'u':
                case 'x':
                case 'X':
                  {
                    unsigned base = 10;
                    if (ch == 'x' || ch == 'X')
                      base = 16;
                    if (ch == 'o')
                      base = 8;
                    if (ch == 'X')
                      p->flags.uc = 1;
                    int is_unsigned = (ch == 'u') || (base != 10);
                    /* if a precision is specified, the 0 flag is ignored */
                    if (p->precision > 0)
                      p->flags.zero = 0; /* The default precision is 1. */
                    if (precision < 0)
                      precision = p->precision = 1;
                    p->base = base;

/* TODO: 3-size vector va-arg crashes LLVM when compiled with -O > 0 */
#define CALL_PRINT_INTS(WIDTH, PROMOTED_WIDTH)                                \
  {                                                                           \
    WIDTH##16 val;                                                            \
    switch (vector_length)                                                    \
      {                                                                       \
      default:                                                                \
        __builtin_unreachable ();                                             \
      case 1:                                                                 \
        val.s0 = va_arg (ap, PROMOTED_WIDTH);                                 \
        break;                                                                \
      case 2:                                                                 \
        val.s01 = va_arg (ap, WIDTH##2);                                      \
        break;                                                                \
      case 3:                                                                 \
      case 4:                                                                 \
        val.s0123 = va_arg (ap, WIDTH##4);                                    \
        break;                                                                \
      case 8:                                                                 \
        val.lo = va_arg (ap, WIDTH##8);                                       \
        break;                                                                \
      case 16:                                                                \
        val = va_arg (ap, WIDTH##16);                                         \
        break;                                                                \
      }                                                                       \
    __pocl_print_ints_##WIDTH (p, &val, vector_length, is_unsigned);          \
  }

                    DEBUG_PRINTF (("[printf:int:conversion=%c]\n", ch));
                    switch (length)
                      {
                      case 1:
                        CALL_PRINT_INTS (uchar, uint);
                        break;
                      case 2:
                        CALL_PRINT_INTS (ushort, uint);
                        break;
                      case 0:
                      case 4:
                        CALL_PRINT_INTS (uint, uint);
                        break;
#ifdef cl_khr_int64
                      case 8:
                        CALL_PRINT_INTS (ulong, ulong);
                        break;
#endif
                      default:
                        __builtin_unreachable ();
                      }
                  }

#undef CALL_PRINT_INTS

                  break;

                  /* Output floats */
                case 'f':
                case 'F':
                case 'e':
                case 'E':
                case 'g':
                case 'G':
                case 'a':
                case 'A':
                  {
                    p->base = 10;
                    if (ch < 'X')
                      {
                        p->flags.uc = 1;
                        p->conv += 32;
                      }

/* TODO: 3-size vector va-arg crashes LLVM when compiled with -O > 0 */
#define CALL_PRINT_FLOATS(WIDTH, PROMOTED_WIDTH)                              \
  {                                                                           \
    WIDTH##16 val;                                                            \
    switch (vector_length)                                                    \
      {                                                                       \
      default:                                                                \
        __builtin_unreachable ();                                             \
      case 1:                                                                 \
        val.s0 = va_arg (ap, PROMOTED_WIDTH);                                 \
        break;                                                                \
      case 2:                                                                 \
        val.s01 = va_arg (ap, WIDTH##2);                                      \
        break;                                                                \
      case 3:                                                                 \
      case 4:                                                                 \
        val.s0123 = va_arg (ap, WIDTH##4);                                    \
        break;                                                                \
      case 8:                                                                 \
        val.lo = va_arg (ap, WIDTH##8);                                       \
        break;                                                                \
      case 16:                                                                \
        val = va_arg (ap, WIDTH##16);                                         \
        break;                                                                \
      }                                                                       \
    __pocl_print_floats_##WIDTH (p, &val, vector_length);                     \
  }

                    DEBUG_PRINTF (("[printf:float:conversion=%c]\n", ch));

                    switch (length)
                      {
                      default:
                        __builtin_unreachable ();
#ifdef cl_khr_fp16
                      /* case 2: CALL_PRINT_FLOATS(half, double); break; */
                      case 2:
                        goto error; /* not yet implemented */
#endif
                      case 0:
#ifdef cl_khr_fp64
                      case 4:
                        CALL_PRINT_FLOATS (float, double);
                        break;
                      case 8:
                        CALL_PRINT_FLOATS (double, double);
                        break;
#else
                      case 4:
                        CALL_PRINT_FLOATS (float, float);
                        break;
#endif
                        break;
                      }
                  }
#undef CALL_PRINT_FLOATS
                  break;


                  /* Output a character */
                case 'c':
                  {
                    DEBUG_PRINTF (("[printf:char]\n"));
                    if (flags.always_sign || flags.space || flags.alt
                        || flags.zero)
                      {
                        errcode = ERROR_FLAGS_WITH_C_CONVERSION_SPECIFIER;
                        goto error;
                      }
                    DEBUG_PRINTF (("[printf:char1]\n"));
                    if (precision >= 0)
                      {
                        errcode = ERROR_PRECISION_WITH_C_CONVERSION_SPECIFIER;
                        goto error;
                      }
                    DEBUG_PRINTF (("[printf:char2]\n"));
                    if (vector_length != 1)
                      {
                        errcode
                            = ERROR_VECTOR_LENGTH_WITH_C_CONVERSION_SPECIFIER;
                        goto error;
                      }
                    DEBUG_PRINTF (("[printf:char3]\n"));
                    if (length != 0)
                      {
                        errcode
                            = ERROR_LENGTH_MODIFIER_WITH_C_CONVERSION_SPECIFIER;
                        goto error;
                      }
                    DEBUG_PRINTF (("[printf:char4]\n"));
                    /* The int argument is converted to an unsigned char, and
                     * the resulting character is written */
                    int i = va_arg (ap, int);
                    bf[0] = (char)i;
                    bf[1] = 0;
                    __pocl_printf_putchw (p);
                    break;
                  }

                  /**************************************************************************/

                  /* Output a string */
                case 's':
                  {
                    if (flags.always_sign || flags.space || flags.alt
                        || flags.zero)
                      {
                        errcode = ERROR_FLAGS_WITH_S_CONVERSION_SPECIFIER;
                        goto error;
                      }
                    if (vector_length != 1)
                      {
                        errcode
                            = ERROR_VECTOR_LENGTH_WITH_S_CONVERSION_SPECIFIER;
                        goto error;
                      }
                    if (length != 0)
                      {
                        errcode
                            = ERROR_LENGTH_MODIFIER_WITH_S_CONVERSION_SPECIFIER;
                        goto error;
                      }

                    OCL_C_AS const char *val
                        = va_arg (ap, OCL_C_AS const char *);
                    if (val == 0)
                      __pocl_printf_puts_ljust (p, "(null)", field_width,
                                                precision);
                    else if (flags.align_left)
                      __pocl_printf_puts_ljust (p, val, field_width,
                                                precision);
                    else
                      __pocl_printf_puts_rjust (p, val, field_width,
                                                precision);
                    break;
                  }

                  /**************************************************************************/

                  /* Output a pointer */
                case 'p':
                  {
                    if (flags.always_sign || flags.space || flags.alt
                        || flags.zero)
                      {
                        errcode = ERROR_FLAGS_WITH_P_CONVERSION_SPECIFIER;
                        goto error;
                      }
                    if (precision >= 0)
                      {
                        errcode = ERROR_PRECISION_WITH_P_CONVERSION_SPECIFIER;
                        goto error;
                      }
                    if (vector_length != 1)
                      {
                        errcode
                            = ERROR_VECTOR_LENGTH_WITH_P_CONVERSION_SPECIFIER;
                        goto error;
                      }
                    if (length != 0)
                      {
                        errcode
                            = ERROR_LENGTH_MODIFIER_WITH_P_CONVERSION_SPECIFIER;
                        goto error;
                      }

                    OCL_C_AS const void *val
                        = va_arg (ap, OCL_C_AS const void *);
                    __pocl_printf_ptr (p, val);
                    break;
                  }

                  /**************************************************************************/

                default:
                  {
                    errcode = ERROR_UNKNOWN_CONVERSION_SPECIFIER;
                    goto error;
                  }
                }
            }
        }
      else
        {
          DEBUG_PRINTF (("[printf:literal]\n"));
          __pocl_printf_putcf (p, ch);
        }
    }

  DEBUG_PRINTF (("[printf:done]\n"));
  return 0;

error:;
  DEBUG_PRINTF (("[printf:error]\n"));
  const char *err_str = ERROR_STRING;
  __pocl_printf_puts (p, err_str);
  char c1 = '0' + (char)(errcode >> 4);
  char c2 = '0' + (char)(errcode & 7);
  __pocl_printf_putcf (p, c1);
  __pocl_printf_putcf (p, c2);
  __pocl_printf_putcf (p, '\n');
  return -1;
}

/**************************************************************************/
/**************************************************************************/
/**************************************************************************/

/* This is the actual printf function that will be used,
 * after the external (buffer) variables are handled in a LLVM pass. */

int
__pocl_printf (char *restrict __buffer, uint32_t *__buffer_index,
               uint32_t __buffer_capacity,
               const PRINTF_FMT_STR_AS char *restrict fmt, ...)
{
  param_t p = { 0 };

  p.printf_buffer = (PRINTF_BUFFER_AS char *)__buffer;
  p.printf_buffer_capacity = __buffer_capacity;
  p.printf_buffer_index = *(PRINTF_BUFFER_AS uint32_t *)__buffer_index;

  va_list va;
  va_start (va, fmt);
  int r = __pocl_printf_format_full (fmt, &p, va);
  va_end (va);

  *(PRINTF_BUFFER_AS uint32_t *)__buffer_index = p.printf_buffer_index;

  return r;
}

/**************************************************************************/

extern char *_printf_buffer;
extern uint32_t *_printf_buffer_position;
extern uint32_t _printf_buffer_capacity;

/* This is a placeholder printf function that will be replaced by calls
 * to __pocl_printf(), after an LLVM pass handles the hidden arguments.
 * both __pocl_printf and __pocl_printf_format_simple must be referenced
 * here, so that the kernel library linker pulls them in. */

int
printf (const PRINTF_FMT_STR_AS char *restrict fmt, ...)
{
  param_t p = { 0 };

  p.printf_buffer = (PRINTF_BUFFER_AS char *)_printf_buffer;
  p.printf_buffer_capacity = _printf_buffer_capacity;
  p.printf_buffer_index
      = *(PRINTF_BUFFER_AS uint32_t *)_printf_buffer_position;

  va_list va;
  va_start (va, fmt);
  int r = __pocl_printf_format_full (fmt, &p, va);
  va_end (va);

  __pocl_printf (_printf_buffer, _printf_buffer_position,
                 _printf_buffer_capacity, NULL);

  *(PRINTF_BUFFER_AS uint32_t *)_printf_buffer_position
      = p.printf_buffer_index;
  return r;
}

/**************************************************************************/
