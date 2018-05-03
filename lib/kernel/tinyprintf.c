/* OpenCL built-in library: tiny_printf

Copyright (C) 2018  Michal Babej / Tampere University of Technology

Code originally from https://github.com/cjlano/tinyprintf

Copyright (c) 2004,2012 Kustaa Nyholm / SpareTimeLabs
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Kustaa Nyholm or SpareTimeLabs nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "printf_base.h"

/* This file implements a very limited subset of OpenCL printf().
 * In particular, no vectors or any floating point formatters
 * are supported. Precision specifier is parsed but ignored.
 * Supported formatters: d,i,o,p,u,x,X,c,s
 * Unsupported: f,e,g,a
 *
 * This printf should work correctly even on devices which do not
 * support 64bit integers (but the 'l' length modifier will be unavailable).
 */

int
__pocl_printf_a2d (char ch)
{
  if (ch >= '0' && ch <= '9')
    return ch - '0';
  else if (ch >= 'a' && ch <= 'f')
    return ch - 'a' + 10;
  else if (ch >= 'A' && ch <= 'F')
    return ch - 'A' + 10;
  else
    return -1;
}

/* scans width specifier, returns 1st char after */
char
__pocl_printf_a2u (char ch, const char **src, unsigned int *width)
{
  const char *p = *src;
  unsigned int num = 0;
  int digit;
  while ((digit = __pocl_printf_a2d (ch)) >= 0)
    {
      if (digit > 10)
        break;
      num = num * 10 + digit;
      ch = *p++;
    }
  *src = p;
  *width = num;
  return ch;
}

void
__pocl_printf_format_simple (const char *fmt, param_t *p, va_list va)
{
  char bf[BF_SIZE * 2];
  char ch, cc;
  p->bf = bf;
  UINT_T ul;
  INT_T l;

  while ((ch = *(fmt++)))
    {
      if (ch != '%')
        {
          __pocl_printf_putcf (p, ch);
          continue;
        }
      ch = *(fmt++);
      if (ch == '%')
        {
          __pocl_printf_putcf (p, '%');
          continue;
        }
      /* Init parameter struct */
      p->flags.zero = 0;
      p->flags.alt = 0;
      p->flags.align_left = 0;
      p->flags.always_sign = 0;
      p->flags.uc = 0;
      p->width = 0;

      /* Flags */
      if (ch < '1')
        {
          switch (ch)
            {
            case '-':
              p->flags.align_left = 1;
              break;
            case '0':
              p->flags.zero = 1;
              break;
            case '#':
              p->flags.alt = 1;
              break;
            default:
              break;
            }
          ch = *(fmt++);
        }

      /* Width */
      if (ch >= '0' && ch <= '9')
        {
          ch = __pocl_printf_a2u (ch, &fmt, &(p->width));
        }

      /* We accept 'x.y' format but don't support it completely:
       * we ignore the 'y' digit => this ignores 0-fill
       * size and makes it == width (ie. 'x') */
      if (ch == '.')
        {
          p->flags.zero = 1; /* zero-padding */
          /* ignore actual 0-fill size: */
          do
            {
              ch = *(fmt++);
            }
          while ((ch >= '0') && (ch <= '9'));
        }

      if (ch == 'c')
        {
          char cc = (char)va_arg (va, int);
          __pocl_printf_putcf (p, cc);
          continue;
        }
      if (ch == 's')
        {
          char *str = (char *)va_arg (va, char *);
          __pocl_printf_puts (p, str);
          continue;
        }

      char lng = 0, shr = 0;

      if (ch == 'l')
        {
#ifdef cl_khr_int64
          ch = *(fmt++);
          lng = 1;
#else
          goto abort;
#endif
        }

      if (ch == 'h')
        {
          ch = *(fmt++);
          shr = 1;
          if (ch == 'h')
            {
              ch = *(fmt++);
              shr = 2;
            }
        }

      if (ch > 'i')
        {
          // unsigned formatters - o, u, x, X
          if (shr == 2)
            ul = (cl_uchar)va_arg (va, unsigned int);
          else if (shr == 1)
            ul = (cl_ushort)va_arg (va, unsigned int);
#ifdef cl_khr_int64
          else if (lng == 1)
            ul = (cl_ulong)va_arg (va, cl_ulong);
#endif
          else
            ul = (cl_uint)va_arg (va, unsigned int);
        }
      else
        {
          // signed formatters - d, i
          if (shr == 2)
            l = (cl_char)va_arg (va, int);
          else if (shr == 1)
            l = (cl_short)va_arg (va, int);
#ifdef cl_khr_int64
          else if (lng == 1)
            l = (cl_long)va_arg (va, cl_long);
#endif
          else
            l = (cl_int)va_arg (va, int);
        }

      switch (ch)
        {
        case 0:
          goto abort;
        case 'o':
          p->base = 8;
          __pocl_printf_ulong (p, ul);
          break;
        case 'u':
          p->base = 10;
          __pocl_printf_ulong (p, ul);
          break;
        case 'd':
        case 'i':
          p->base = 10;
          __pocl_printf_long (p, l);
          break;
        case 'p':
          p->alt = 1;
        case 'x':
        case 'X':
          p->base = 16;
          p->flags.uc = (ch == 'X') ? 1 : 0;
          __pocl_printf_ulong (p, ul);
          break;

        default:
          break;
        }
    }
abort:;
}

/* This is the actual printf function that will be used,
 * after the external (buffer) variables are handled in a LLVM pass. */

int
__pocl_printf (char *restrict __buffer, size_t *__buffer_index,
               size_t __buffer_capacity, const char *restrict fmt, ...)
{
  param_t p = { 0 };

  p.printf_buffer = __buffer;
  p.printf_buffer_capacity = __buffer_capacity;
  p.printf_buffer_index = *__buffer_index;

  va_list va;
  va_start (va, fmt);
  __pocl_printf_format_simple (fmt, &p, va);
  va_end (va);

  *__buffer_index = p.printf_buffer_index;

  return 0;
}

extern char *_printf_buffer;
extern size_t *_printf_buffer_position;
extern size_t _printf_buffer_capacity;

/* This is a placeholder printf function that will be replaced by calls
 * to __pocl_printf(), after a LLVM pass handles the hidden arguments.
 * both __pocl_printf and __pocl_printf_format_simple must be referenced
 * here, so that the kernel library linker pulls them in. */

int
__cl_printf (const char *restrict fmt, ...)
{
  va_list va;
  va_start (va, fmt);
  __pocl_printf_format_simple (fmt, NULL, va);
  va_end (va);

  __pocl_printf (_printf_buffer, _printf_buffer_position,
                 _printf_buffer_capacity, NULL);

  return 0;
}
