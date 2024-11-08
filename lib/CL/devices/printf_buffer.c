/* OpenCL runtime library: print the content of the printf_buffer to STDOUT

   Copyright (c) 2013 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                      Perimeter Institute for Theoretical Physics
   Copyright (c) 2018 Michal Babej / Tampere University of Technology
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

#include "printf_buffer.h"
#include "common.h"
#include "pocl_debug.h"
#include "printf_base.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif

#if 0

int debugprintf(const char *fmt, ...)
{
  int ret;
  va_list vargs;
  va_start(vargs, fmt);
  ret = vdprintf(2, fmt, vargs);
  va_end(vargs);
  return ret;
}
#define DEBUG_PRINTF(args) (debugprintf args)

#else

#define DEBUG_PRINTF(args) ((void)0)

#endif

/**************************************************************************/

/* flags are saved on each loop iter because
 * __pocl_printf_(u)long can change flags */
#define DEFINE_PRINT_INTS(NAME, INT_TYPE, UINT_TYPE)                          \
  void __pocl_print_ints_##NAME (param_t *p, const char *vals, int n,         \
                                 int is_unsigned)                             \
  {                                                                           \
    DEBUG_PRINTF (("[printf:ints:n=%df]\n", n));                              \
    flags_t saved_user_flags = p->flags;                                      \
    for (int d = 0; d < n; ++d)                                               \
      {                                                                       \
        DEBUG_PRINTF (                                                        \
          ("[printf:ints:d=%d|size=%d]\n", d, sizeof (UINT_TYPE)));           \
        p->flags = saved_user_flags;                                          \
        if (d != 0)                                                           \
          __pocl_printf_putcf (p, ',');                                       \
        if (is_unsigned)                                                      \
          {                                                                   \
            UINT_TYPE tmp;                                                    \
            memcpy (&tmp, vals, sizeof (UINT_TYPE));                          \
            vals += sizeof (UINT_TYPE);                                       \
            DEBUG_PRINTF (("[printf:ints:VAL=%lu]\n", (UINT_T)tmp));          \
            __pocl_printf_ulong (p, (UINT_T)tmp);                             \
          }                                                                   \
        else                                                                  \
          {                                                                   \
            INT_TYPE tmp;                                                     \
            memcpy (&tmp, vals, sizeof (INT_TYPE));                           \
            vals += sizeof (INT_TYPE);                                        \
            DEBUG_PRINTF (("[printf:ints:VAL=%li]\n", (INT_T)tmp));           \
            __pocl_printf_long (p, (INT_T)tmp);                               \
          }                                                                   \
      }                                                                       \
    DEBUG_PRINTF (("[printf:ints:done]\n"));                                  \
  }

DEFINE_PRINT_INTS (uchar, int8_t, uint8_t)
DEFINE_PRINT_INTS (ushort, int16_t, uint16_t)
DEFINE_PRINT_INTS (uint, int32_t, uint32_t)
DEFINE_PRINT_INTS (ulong, int64_t, uint64_t)

#undef DEFINE_PRINT_INTS

/**************************************************************************/

/* Note: NANs are printed always positive.
 * This is required to pass 1.2 conformance test. */
#define DEFINE_PRINT_FLOATS(FLOAT_TYPE)                                       \
  void __pocl_print_floats_##FLOAT_TYPE (param_t *p, const void *vals, int n) \
  {                                                                           \
    const char *NANs[2] = { "nan", "NAN" };                                   \
    const char *INFs[2] = { "inf", "INF" };                                   \
    FLOAT_TYPE val;                                                           \
    DEBUG_PRINTF (("[printf:floats:n=%d]\n", n));                             \
    flags_t saved_user_flags = p->flags;                                      \
    for (int d = 0; d < n; ++d)                                               \
      {                                                                       \
        DEBUG_PRINTF (("[printf:floats:d=%d]\n", d));                         \
        p->flags = saved_user_flags;                                          \
        if (d != 0)                                                           \
          __pocl_printf_putcf (p, ',');                                       \
        memcpy (&val, vals, sizeof (FLOAT_TYPE));                             \
        vals = (char *)vals + sizeof (FLOAT_TYPE);                            \
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
          __pocl_printf_float (p, (FLOAT_T)val);                              \
      }                                                                       \
    DEBUG_PRINTF (("[printf:floats:done]\n"));                                \
  }

DEFINE_PRINT_FLOATS (cl_half)
DEFINE_PRINT_FLOATS (float)
DEFINE_PRINT_FLOATS (double)

#undef DEFINE_PRINT_FLOATS

/**************************************************************************/

#define FORWARD_BUFFER(SIZE)                                                  \
  buffer += (SIZE);                                                           \
  if (buffer_size < (SIZE))                                                   \
    {                                                                         \
      POCL_MSG_ERR (                                                          \
        "printf error: exhausted arguments before format string end\n");      \
      return -1;                                                              \
    }                                                                         \
  buffer_size -= (SIZE);

uint32_t
__pocl_printf_format_full (param_t *p, char *buffer, uint32_t buffer_size)
{
  char ch = 0;

  /* fetch & decode the control dword */
  uint32_t control_dword;
  memcpy (&control_dword, buffer, sizeof (uint32_t));
  FORWARD_BUFFER (sizeof (uint32_t));
  /* if this flag is set, the fmt str is a constant stored as pointer
   * otherwise it's a dynamic string stored directly in the buffer */
  uint32_t skip_fmt_str = control_dword & PRINTF_BUFFER_CTWORD_SKIP_FMT_STR;
  uint32_t char_short_promotion
    = control_dword & PRINTF_BUFFER_CTWORD_CHAR_SHORT_PR;
  uint32_t char2_promotion = control_dword & PRINTF_BUFFER_CTWORD_CHAR2_PR;
  uint32_t float_promotion = control_dword & PRINTF_BUFFER_CTWORD_FLOAT_PR;
  uint32_t big_endian = control_dword & PRINTF_BUFFER_CTWORD_BIG_ENDIAN;
  if (big_endian)
    {
      POCL_MSG_ERR ("printf error: printf for big endian devices not yet"
                    "implemented\n");
      return -1;
    }
  assert ((control_dword >> PRINTF_BUFFER_CTWORD_FLAG_BITS)
          == (buffer_size + sizeof (uint32_t)));

  const char *format;
  if (skip_fmt_str)
    {
      /* pointer to format string
       * 8 bytes are always reserved for the address */
      memcpy (&format, buffer, sizeof (const char *));
      FORWARD_BUFFER (sizeof (uint64_t));
      DEBUG_PRINTF (("[printf:SKIP START:buffer=%p buffer_size=%u]\n", buffer,
                     buffer_size));
      if (format == NULL)
        {
          POCL_MSG_ERR ("printf error: invalid (NULL) format string!\n");
          return -1;
        }
      if (*format == 0)
        return 0;
    }
  else
    {
      /* format string stored directly */
      format = buffer;
      /* strlen */
      while (*buffer && (buffer - format < buffer_size))
        ++buffer;
      if (*buffer)
        {
          POCL_MSG_ERR ("printf error: unterminated format string ?\n");
          return -1;
        }
      else
        {
          /* skip NULL */
          ++buffer;
        }
      uint32_t fmt_str_len = buffer - format;
      buffer_size -= fmt_str_len;
      DEBUG_PRINTF (("[printf:FULL START:buffer=%p buffer_size=%u]\n", buffer,
                     buffer_size));
      if (fmt_str_len == 0)
        return 0;
    }

  DEBUG_PRINTF (("[printf:format=%s]\n", format));

  while ((ch = *format++))
    {
      if (ch == '%')
        {
          ch = *format++;
          if (ch == 0)
            {
              POCL_MSG_ERR ("printf error: NULL after format sign (%%)\n");
              return -1;
            }

          if (ch == '%')
            {
              DEBUG_PRINTF (("[printf:%%]\n"));
              __pocl_printf_putcf (p, '%'); /* literal % */
            }
          else
            {
              DEBUG_PRINTF (("###################\n[printf:arg]\n"));
              if (buffer_size == 0)
                {
                  POCL_MSG_ERR ("printf error: exhausted arguments but format "
                                "string contains more specifiers\n");
                  return -1;
                }

              /* Flags */
              flags_t flags;
              flags.align_left = 0;
              flags.sign = 0;
              flags.space = 0;
              flags.alt = 0;
              flags.zero = 0;
              flags.uc = 0;
              flags.always_sign = 0;
              flags.nonzeroparam = 0;
              for (;;)
                {
                  switch (ch)
                    {
                    case '-':
                      if (flags.align_left)
                        {
                          POCL_MSG_ERR (
                            "printf error: repeated align-left flag (-)\n");
                          return -1;
                        }
                      flags.align_left = 1;
                      break;
                    case '+':
                      if (flags.always_sign)
                        {
                          POCL_MSG_ERR (
                            "printf error: repeated always-sign flag (+)\n");
                          return -1;
                        }
                      flags.always_sign = 1;
                      break;
                    case ' ':
                      if (flags.space)
                        {
                          POCL_MSG_ERR ("printf error: repeated space flag\n");
                          return -1;
                        }
                      flags.space = 1;
                      break;
                    case '#':
                      if (flags.alt)
                        {
                          POCL_MSG_ERR (
                            "printf error: repeated sharp flag (#)\n");
                          return -1;
                        }
                      flags.alt = 1;
                      break;
                    case '0':
                      if (flags.zero)
                        {
                          POCL_MSG_ERR (
                            "printf error: repeated zero flag (0)\n");
                          return -1;
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
              uint32_t field_width = 0;
              while (ch >= '0' && ch <= '9')
                {
                  if (ch == '0' && field_width == 0)
                    {
                      POCL_MSG_ERR ("printf error: field-width is zero\n");
                      return -1;
                    }
                  if (field_width > (UINT32_MAX - 9) / 10)
                    {
                      POCL_MSG_ERR ("printf error: field-width overflow\n");
                      return -1;
                    }
                  field_width = 10 * field_width + (ch - '0');
                  ch = *format++;
                }
              DEBUG_PRINTF (("[printf:width=%d]\n", field_width));

              /* Precision */
              int32_t precision = -1;
              if (ch == '.')
                {
                  precision = 0;
                  ch = *format++;
                  while (ch >= '0' && ch <= '9')
                    {
                      if (precision > (INT32_MAX - 9) / 10)
                        {
                          POCL_MSG_ERR ("printf error: precision overflow\n");
                          return -1;
                        }
                      precision = 10 * precision + (ch - '0');
                      ch = *format++;
                    }
                }
              DEBUG_PRINTF (("[printf:precision=%d]\n", precision));

              /* Vector specifier */
              int32_t vector_length = 0;
              if (ch == 'v')
                {
                  ch = *format++;
                  while (ch >= '0' && ch <= '9')
                    {
                      if (ch == '0' && vector_length == 0)
                        {
                          POCL_MSG_ERR (
                            "printf error: vector-length is zero\n");
                          return -1;
                        }
                      if (vector_length > 16)
                        {
                          POCL_MSG_ERR (
                            "printf error: vector-length overflow\n");
                          return -1;
                        }
                      vector_length = 10 * vector_length + (ch - '0');
                      ch = *format++;
                    }
                  if (!(vector_length == 2 || vector_length == 3
                        || vector_length == 4 || vector_length == 8
                        || vector_length == 16))
                    {
                      POCL_MSG_ERR (
                        "printf error: unrecognized vector length (%d)\n",
                        vector_length);
                      return -1;
                    }
                }

              /* Length modifier */
              uint32_t length = 0;
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
                  POCL_MSG_ERR (
                    "printf error: vector-length used without element size\n");
                  return -1;
                }

              if (vector_length == 0 && length == 4)
                {
                  POCL_MSG_ERR (
                    "printf error: hl modifier used without vector length\n");
                  return -1;
                }

              if (vector_length == 0)
                vector_length = 1;
              uint32_t alloca_length = vector_length;
              if (vector_length == 3)
                alloca_length = 4;

              DEBUG_PRINTF (("[printf:vector_length=%d]\n", vector_length));

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
                    DEBUG_PRINTF (("[printf:int:conversion=%c]\n", ch));
                    if (length == 0)
                      length = 4;
                    alloca_length *= length;
                    switch (length)
                      {
                      default:
                        return -1;
                      case 1:
                        __pocl_print_ints_uchar (p, buffer, vector_length,
                                                 is_unsigned);
                        if (char_short_promotion && vector_length == 1)
                          {
                            alloca_length = 4;
                          }
                        if (char2_promotion && vector_length == 2)
                          {
                            alloca_length = 4;
                          }
                        break;
                      case 2:
                        __pocl_print_ints_ushort (p, buffer, vector_length,
                                                  is_unsigned);
                        if (char_short_promotion && vector_length == 1)
                          {
                            alloca_length = 4;
                          }
                        break;
                      case 4:
                        __pocl_print_ints_uint (p, buffer, vector_length,
                                                is_unsigned);
                        break;
                      case 8:
                        __pocl_print_ints_ulong (p, buffer, vector_length,
                                                 is_unsigned);
                        break;
                      }
                    FORWARD_BUFFER (alloca_length);
                    DEBUG_PRINTF (
                      ("[printf:after int:buffer=%p buffer_size=%u]\n", buffer,
                       buffer_size));
                    break;
                  }

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
                    DEBUG_PRINTF (
                      ("[printf:float:conversion=%c|promotion=%u]\n", ch,
                       (unsigned)float_promotion));
                    if (length == 0)
                      length = 4;
                    alloca_length *= length;
                    switch (length)
                      {
                      default:
                      case 2:
                        {
                          POCL_MSG_ERR ("printf error: printing halfs is not "
                                        "yet implemented\n");
                          return -1; /* half type not yet implemented */
                        }
                      case 4:
                        /* single float can be promoted to double */
                        if (!float_promotion || vector_length > 1)
                          {
                            __pocl_print_floats_float (p, buffer,
                                                       vector_length);
                            break;
                          }
                        else
                          {
                            /* .. else fallthrough to double */
                            alloca_length = 8;
                          }
                      case 8:
                        __pocl_print_floats_double (p, buffer, vector_length);
                        break;
                      }
                    FORWARD_BUFFER (alloca_length);
                    DEBUG_PRINTF (
                      ("[printf:after float:buffer=%p buffer_size=%u]\n",
                       buffer, buffer_size));
                    break;
                  }

                  /* Output a character */
                case 'c':
                  {
                    DEBUG_PRINTF (("[printf:char]\n"));
                    if (flags.always_sign || flags.space || flags.alt
                        || flags.zero)
                      {
                        POCL_MSG_ERR (
                          "printf error: flags used with '%%c' conversion\n");
                        return -1;
                      }
                    DEBUG_PRINTF (("[printf:char1]\n"));
                    if (precision >= 0)
                      {
                        POCL_MSG_ERR ("printf error: precision used with "
                                      "'%%c' conversion\n");
                        return -1;
                      }
                    DEBUG_PRINTF (("[printf:char2]\n"));
                    if (vector_length != 1)
                      {
                        POCL_MSG_ERR ("printf error: vector length used with "
                                      "'%%c' conversion\n");
                        return -1;
                      }
                    DEBUG_PRINTF (("[printf:char3]\n"));
                    if (length != 0)
                      {
                        POCL_MSG_ERR ("printf error: length-modifier used "
                                      "with '%%c' conversion\n");
                        return -1;
                      }
                    DEBUG_PRINTF (("[printf:char4]\n"));
                    p->bf[0] = (char)*buffer;
                    p->bf[1] = 0;
                    /* char is always promoted to int32 */
                    FORWARD_BUFFER (sizeof (int32_t));
                    __pocl_printf_putchw (p);
                    DEBUG_PRINTF (
                      ("[printf:after char:buffer=%p buffer_size=%u]\n",
                       buffer, buffer_size));
                    break;
                  }

                  /**************************************************************************/

                  /* Output a string */
                case 's':
                  {
                    if (flags.always_sign || flags.space || flags.alt
                        || flags.zero)
                      {
                        POCL_MSG_ERR (
                          "printf error: flags used with '%%s' conversion\n");
                        return -1;
                      }
                    if (vector_length != 1)
                      {
                        POCL_MSG_ERR ("printf error: vector-length used with "
                                      "'%%s' conversion\n");
                        return -1;
                      }
                    if (length != 0)
                      {
                        POCL_MSG_ERR ("printf error: length-modifier used "
                                      "with '%%s' conversion\n");
                        return -1;
                      }

                    const char *val = buffer;
                    // strings are stored directly
                    const char *tmp = buffer;
                    // strlen
                    while (*tmp && (tmp - val < buffer_size))
                      ++tmp;
                    if (*tmp)
                      {
                        POCL_MSG_ERR ("printf error: string not terminated by "
                                      "NULL for '%%s' conversion\n");
                        return -1;
                      }
                    else
                      {
                        // skip NULL
                        ++tmp;
                      }
                    uint32_t str_len = tmp - val;
                    DEBUG_PRINTF (("[printf:string:%s:%u]\n", val, str_len));
                    FORWARD_BUFFER (str_len);
                    DEBUG_PRINTF (("[printf:START:buffer=%p buffer_size=%u]\n",
                                   buffer, buffer_size));

                    if (flags.align_left)
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
                    DEBUG_PRINTF (("[printf:pointer]\n"));
                    if (flags.always_sign || flags.space || flags.alt
                        || flags.zero)
                      {
                        POCL_MSG_ERR (
                          "printf error: flags used with '%%p' conversion\n");
                        return -1;
                      }
                    if (precision >= 0)
                      {
                        POCL_MSG_ERR ("printf error: precision used with "
                                      "'%%p' conversion\n");
                        return -1;
                      }
                    if (vector_length != 1)
                      {
                        POCL_MSG_ERR ("printf error: vector-length used with "
                                      "'%%p' conversion\n");
                        return -1;
                      }
                    if (length != 0)
                      {
                        POCL_MSG_ERR ("printf error: length-modifier used "
                                      "with '%%p' conversion\n");
                        return -1;
                      }

                    const void *val;
                    memcpy (&val, buffer, sizeof (val));
                    FORWARD_BUFFER (sizeof (val));
                    DEBUG_PRINTF (("[printf:ptr:%p]\n", val));
                    __pocl_printf_ptr (p, val);
                    DEBUG_PRINTF (
                      ("[printf:after ptr:buffer=%p buffer_size=%u]\n", buffer,
                       buffer_size));
                    break;
                  }

                  /**************************************************************************/

                default:
                  {
                    POCL_MSG_ERR (
                      "printf error: unknown conversion specifier (%c)\n", ch);
                    return -1;
                  }
                }
            }
        }
      else
        {
          DEBUG_PRINTF (("###################\n[printf:literal]\n"));
          __pocl_printf_putcf (p, ch);
        }
    }

  DEBUG_PRINTF (("[printf:done]\n"));
  return 0;
}

/**************************************************************************/

#define IMM_FLUSH_BUFFER_SIZE 65536

void
pocl_flush_printf_buffer (char *buffer, uint32_t buffer_size)
{
  param_t p = { 0 };
  char bf[BUFSIZE];
  memset (bf, 0, BUFSIZE);
  p.bf = bf;

  char result[IMM_FLUSH_BUFFER_SIZE];
  p.printf_buffer = result;
  p.printf_buffer_capacity = IMM_FLUSH_BUFFER_SIZE;
  p.printf_buffer_index = 0;

  __pocl_printf_format_full (&p, buffer, buffer_size);

  if (p.printf_buffer_index > 0)
    {
#ifdef _MSC_VER
      write (_fileno (stdout), p.printf_buffer, p.printf_buffer_index);
#else
      write (STDOUT_FILENO, p.printf_buffer, p.printf_buffer_index);
#endif
    }
}

/**************************************************************************/

void
pocl_write_printf_buffer (char *printf_buffer, uint32_t bytes)
{
  uint32_t control_dword, single_entry_bytes;

  if (bytes == 0)
    return;

  do
    {
      if (bytes < sizeof (uint32_t))
        {
          POCL_MSG_ERR ("printf buffer entry size < sizeof(control word)\n");
          return;
        }

      memcpy (&control_dword, printf_buffer, sizeof (uint32_t));
      single_entry_bytes = control_dword >> PRINTF_BUFFER_CTWORD_FLAG_BITS;

      if (single_entry_bytes > bytes)
        {
          POCL_MSG_ERR ("Error: less bytes stored in printf_buffer "
                        "than control word suggests\n");
          return;
        }
      /* at minimum, an empty format string must exist */
      if (single_entry_bytes < 5)
        {
          POCL_MSG_ERR ("Error: malformed entry in printf_buffer\n");
          return;
        }

      pocl_flush_printf_buffer (printf_buffer, single_entry_bytes);
      printf_buffer += single_entry_bytes;
      bytes -= single_entry_bytes;
    }
  while (bytes > 0);
}
