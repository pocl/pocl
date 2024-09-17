/* OpenCL runtime: printf helper functions

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

#include "printf_base.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define BUILD_WITH_LIBC

void
__pocl_printf_putcf (param_t *p, char c)
{
  if (p->printf_buffer_index < p->printf_buffer_capacity)
    p->printf_buffer[p->printf_buffer_index++] = c;
}

void
__pocl_printf_puts (param_t *p, const char *string)
{
  char c;
  while ((c = *string++))
    {
      if (p->printf_buffer_index < p->printf_buffer_capacity)
        p->printf_buffer[p->printf_buffer_index++] = c;
    }
}

/* UNSIGNED long, base <= 10*/
void
__pocl_printf_ul_base (param_t *p, UINT_T num)
{
  char temp[SMALL_BUF_SIZE];
  unsigned i = 0, j = 0;
  unsigned digit;
  unsigned base = p->base;
  while (num > 0)
    {
      digit = num % base;
      num = num / base;
      temp[i++] = '0' + digit;
    }

  if (p->precision > 0)
    {
      for (; i < p->precision; i++)
        temp[i] = '0';
    }

  char *out = p->bf;
  for (j = i; j > 0; --j)
    *out++ = temp[j - 1];
  *out = 0;
}

/* UNSIGNED long, base == 16 */
void
__pocl_printf_ul16 (param_t *p, UINT_T num)
{
  char temp[SMALL_BUF_SIZE];
  unsigned i = 0, j = 0;
  unsigned digit;
  const unsigned base = 16;
  char digit_offset = (p->flags.uc ? 'A' : 'a');
  while (num > 0)
    {
      digit = num % base;
      num = num / base;
      temp[i++] = ((digit < 10) ? ('0' + digit) : (digit_offset + digit - 10));
    }

  if (p->precision > 0)
    {
      for (; i < p->precision; i++)
        temp[i] = '0';
    }

  char *out = p->bf;
  for (j = i; j > 0; --j)
    *out++ = temp[j - 1];
  *out = 0;
}

/* SIGNED long, base <= 10*/
void
__pocl_printf_l_base (param_t *p, INT_T num)
{
  UINT_T n;
  if (num < 0)
    {
      if (num == INT_T_MIN)
        n = (UINT_T)num;
      else
        n = -num;
      p->flags.sign = 1;
    }
  else
    {
      n = num;
      p->flags.sign = 0;
    }
  __pocl_printf_ul_base (p, n);
}

/* prints just the exponent */
void
__pocl_printf_exp (char *out, INT_T exp, unsigned min_output_chars)
{
  char temp[SMALL_BUF_SIZE];
  unsigned i = 0, j = 0;
  unsigned digit;

  if (exp < 0)
    {
      *out++ = '-';
      exp = -exp;
    }
  else
    *out++ = '+';

  do
    {
      digit = exp % 10;
      exp = exp / 10;
      temp[i++] = '0' + digit;
    }
  while (exp > 0);

  while (i < min_output_chars)
    {
      temp[i++] = '0';
    }

  for (j = i; j > 0; --j)
    *out++ = temp[j - 1];
  *out = 0;
}

/* print mantissa in nibbles (4bits) for %a */
void
__pocl_printf_nibbles (param_t *p, UINT_T num, INT_T exp,
                       unsigned max_fract_digits, int exact, int print_dec)
{
  char *out = p->bf;
  char temp[SMALL_BUF_SIZE];
  unsigned i = 0, available_digits = 0, written_digits = 0;
  unsigned digit, stop;
  const unsigned base = 16;
  char digit_offset = (p->flags.uc ? 'A' : 'a');
  unsigned trailing_zeroes = 0;
  int encountered_nonzero = 0;

  /* this loop will always produce MAX_NIBBLES+1 digits. */
  for (i = 0; i <= MAX_NIBBLES; ++i)
    {
      digit = num % base;
      num = num / base;
      temp[i] = ((digit < 10) ? ('0' + digit) : (digit_offset + digit - 10));
      /* count the trailing zeroes */
      if (digit)
        encountered_nonzero = 1;
      if (encountered_nonzero == 0)
        ++trailing_zeroes;
    }

  /* buffer now has "i" digits in reverse order */
  available_digits = i;

  /* always print first digit */
  *out++ = temp[--available_digits];

  /* precision == 0 */
  if (max_fract_digits == 0)
    goto SKIP_DECIMAL_PART;

  /* for exact printing, stop on trailing zeroes,
   * otherwise print max_digits digits. */
  if (exact)
    stop = trailing_zeroes;
  else
    stop = 0;

  /* decimal point if needed. */
  if (print_dec || (available_digits > stop))
    *out++ = '.';

  /* digits */
  while ((available_digits > stop) && (written_digits < max_fract_digits))
    {
      char c = temp[--available_digits];
      *out++ = c;
      written_digits++;
    }

SKIP_DECIMAL_PART:
  *out++ = (p->flags.uc ? 'P' : 'p');
  __pocl_printf_exp (out, exp, 0);
}

/*
 * style [−]0xh.hhhh p±d, where there is
 *
 * one hexadecimal digit
 * (which is nonzero if the argument is a normalized floating-point
 * number and is otherwise unspecified) before the decimal-point
 * character)
 *
 * and the number of hexadecimal digits after it is equal
 * to the precision;
 *
 * if the precision is missing, then the precision
 * is sufficient for an exact representation of the value;
 *
 * if the
 * precision is zero and the # flag is not specified, no decimal
 * point character appears.
 *
 * The letters abcdef are used for a conversion
 * and the letters ABCDEF for A conversion.
 *
 * The A conversion specifier
 * produces a number with X and P instead of x and p.
 *
 * The exponent
 * always contains at least one digit,
 * and only as many more digits as
 * necessary to represent the decimal exponent of 2.
 *
 * If the value is
 * zero, the exponent is zero.
 *
 * A double, halfn, floatn or doublen argument
 * representing an infinity or NaN is converted in the style of an f or F
 * conversion specifier.
 *
 * Binary implementations can choose the hexadecimal
 * digit to the left of the decimal-point character so that subsequent
 * digits align to nibble (4-bit) boundaries.
 *
 * For a and A conversions, the value is correctly rounded to a hexadecimal
 * floating number with the given precision.
 */
void
__pocl_printf_float_a (param_t *p, int print_dec, FLOAT_T f)
{
  union
  {
    FLOAT_T ff;
    FLOAT_UINT_T uu;
    FLOAT_INT_T ii;
  } tmp;

  tmp.ff = f;
  FLOAT_INT_T exp = L_EXPONENT (tmp.ii);
  FLOAT_UINT_T mant = L_MANTISSA (tmp.uu);

  /* handle denorms */
  if (exp == (-EXPBIAS) && mant > 0)
    {
      while (L_MSB_NIBBLE (mant) == 0)
        {
          exp -= 4;
          mant <<= 4;
        }
    }

  FLOAT_UINT_T max_fract_digits = 0;
  int exact = 0;

  if (p->precision < 0)
    {
      max_fract_digits = (FLOAT_UINT_T) (-1);
      exact = 1;
    }
  else
    max_fract_digits = (FLOAT_UINT_T)p->precision;

  if (tmp.uu == 0)
    exp = 0;
  else
    {
      mant |= LEADBIT;
      /* perform RTE rounding */
      if (max_fract_digits < MAX_NIBBLES)
        {
          /* No of bits that are beyond last wanted digit */
          FLOAT_UINT_T shift = EXPSHIFTBITS - (max_fract_digits * 4);
          FLOAT_UINT_T mask = (1UL << shift) - 1;
          FLOAT_UINT_T half = (1UL << (shift - 1));
          FLOAT_UINT_T rem = (mant & mask);
          FLOAT_UINT_T shifted_mant = mant >> shift;
          if ((rem == half) && ((shifted_mant & 1) == 0))
            mant = (shifted_mant) << shift;
          else if (rem >= half)
            mant = (shifted_mant + 1) << shift;
        }
    }

  __pocl_printf_nibbles (p, mant, exp, max_fract_digits, exact, print_dec);
  /* force putchw to print "0x" */
  p->flags.alt = 1;
  p->base = 16;
  p->flags.nonzeroparam = 1;
}

#ifdef BUILD_WITH_LIBC
void
__pocl_printf_float_libc (param_t *p, FLOAT_T f)
{
  char outfmt[128];
  char conv = p->conv;
  if (p->flags.uc)
    conv -= 32;
  int prec = p->precision;
  const char str[] = "%%%s%s%s%s%s%.0d%s%.0d" "%c";
  snprintf(outfmt, sizeof outfmt,
           str,
           p->flags.align_left ? "-" : "",
           p->flags.always_sign ? "+" : "",
           p->flags.space ? " " : "",
           p->flags.alt ? "#" : "",
           p->flags.zero ? "0" : "",
           p->width,
           prec != -1 ? "." : "",
           prec != -1 ? prec : 0,
           conv
           );

  snprintf (p->bf, BUFSIZE, outfmt, f);
  p->bf[BUFSIZE - 1] = 0;
  __pocl_printf_puts (p, p->bf);
}

#else

/* Rounds a decimal float to 'prec' decimal places, using RTE.
 * Note that input buf must have '0' as its first char
 * (to make round up easier).
 *
 * in F mode, round to prec decimal places from dec_point;
 * in E mode, round to 2+prec decimal places (one for the extra 0 we added, one
 * for the always present first digit) */
void
__pocl_printf_float_round (FLOAT_T f, char *buf, int dec_point, int prec,
                           int e_mode)
{
  /* invalid args, no rounding */
  if (prec < 0 || prec > 18)
    return;

  int round_point = prec;
  if (e_mode)
    round_point += 2;
  else
    round_point += dec_point;

  /* result rounds to zero */
  if (round_point < 0)
    {
      buf[0] = 0;
      return;
    }

  char *p = buf;
  while (*p)
    ++p;
  int len = p - buf;
  /* if round_point >= len, we're done */
  if (round_point >= len)
    return;

  /* round_point < len; find out if the digits
   * after round_point are exactly halfway between */
  int i = round_point;
  int direction = (buf[i] - '5');
  ++i;
  int is_half = (direction == 0);
  if (is_half && i < len)
    {
      while (i < len && (buf[i] == '0'))
        ++i;
      is_half = (i == len);
      direction = (i != len);
    }

  i = round_point - 1;

  if (is_half) /* RTE */
    {
      char last_dig = buf[i];
      if ((last_dig - '0') % 2)
        direction = 1; /* odd round up */
    }
  if (direction > 0) /* round up */
    {
      int carry = 0;
      do
        {
          char c = buf[i];
          carry = (c == '9');
          if (carry)
            buf[i] = '0';
          else
            buf[i] += 1;
          --i;
        }
      while (carry);
    }

  buf[round_point] = 0;
}

/* if needed, drops the extra zero added for rounding */
char *
__pocl_printf_float_round_buf (FLOAT_T f, char *buf, int *exp, int prec,
                               int e_mode)
{
  __pocl_printf_float_round (f, buf, *exp, prec, e_mode);

  if (buf[0] == '0' && buf[1])
    {
      ++buf;
      *exp -= 1;
    }

  return buf;
}

/* print the decimal float in buf with 'e' format. float must be already
 * rounded */
void
__pocl_printf_float_e (param_t *p, char *buf, int point, int print_dec,
                       int notrailing0)
{
  char *out = p->bf;
  char *in = buf;

  int prec = p->precision;

  int exp = point;
  int printed_dec = 0;

  /* we're always printing one digit for integral part */
  --exp;

  char c;
  /* always print first digit */
  *out++ = *in++;

  if (print_dec || (prec > 0))
    *out++ = '.';

  /* decimal part */
  if (prec > 0)
    {
      c = *in++;
      while (printed_dec < prec)
        {
          /* note: for notrailing0 mode, we cannot simply not print '0' here,
           * since the buffer may contain trailing 0 as result of rounding;
           * we must remove trailing zeroes manually. */
          if (c)
            {
              *out++ = c;
              c = *in++;
            }
          else
            *out++ = '0';

          ++printed_dec;
        }
    }

  /* remove trailing zeroes & dec. point if needed */
  if (notrailing0)
    {
      char *tmp = out - 1;
      while (tmp > p->bf && (*tmp == '0'))
        --tmp;
      if (*tmp == '.')
        --tmp;
      out = tmp + 1;
    }

  *out++ = (p->flags.uc ? 'E' : 'e');
  __pocl_printf_exp (out, exp, 2);
}

/* print the decimal float in buf with 'f' format. float must be already
 * rounded */
void
__pocl_printf_float_f (param_t *p, char *buf, int point, int print_dec,
                       int notrailing0)
{
  int prec = p->precision;
  int i;
  char *out = p->bf; // output
  char *in = buf;    // input
  char c;
  /* pointer to where '.' was printed, for trailing 0 removal */
  char *decpoint_ptr = NULL;

  /* bytes written for mantissa. must be <= prec */
  int written = 0;

  /* if required, print leading zeros */
  if (point <= 0)
    {
      int poi = point;
      *out++ = '0';
      if (print_dec || (prec > 0))
        {
          decpoint_ptr = out;
          *out++ = '.';
        }
      while ((poi < 0) && (written < prec))
        {
          *out++ = '0';
          ++poi;
          ++written;
        }
      c = *in++;
    }
  /* else print integral part */
  else
    {
      c = *in++;
      for (i = 0; i < point; ++i)
        {
          if (c)
            {
              *out++ = c;
              c = *in++;
            }
          else
            *out++ = '0';
        }
      if (print_dec || (prec > 0))
        {
          decpoint_ptr = out;
          *out++ = '.';
        }
    }

  /* print decimal part */
  while (written < prec)
    {
      if (c)
        {
          *out++ = c;
          c = *in++;
        }
      else
        *out++ = '0';

      ++written;
    }

  *out = 0;

  /* remove trailing zeroes if needed*/
  if (notrailing0 && (decpoint_ptr != NULL))
    {
      out -= 1;
      unsigned i = 0;
      while (out > decpoint_ptr && *out == '0')
        {
          *out-- = 0;
        }
      if (*out == '.')
        *out = 0;
    }
}

#endif

/* prints the number (float or integer) in p->bf,
 * taking care of various flags. */
void
__pocl_printf_putchw (param_t *p)
{
  char ch;
  int n = p->width;
  /* For x (or X) conversion, a nonzero result has 0x (or 0X) prefixed to it.
   */
  int althex = (p->flags.nonzeroparam && p->flags.alt && p->base == 16);
  /* For o conversion, it increases the precision, if and only if necessary,
   * to force the first digit of the result to be a zero */
  int altoct = (p->bf[0] != '0' && p->flags.alt && p->base == 8);
  int sign_required = (p->flags.always_sign || p->flags.sign);
  int space_required = (p->flags.space && p->flags.sign == 0);
  char *bf = p->bf;

  /* Number of filling characters */
  while (*bf++ && n > 0)
    n--;
  if (sign_required)
    n--;
  if (space_required)
    n--;
  if (althex)
    n -= 2;
  if (altoct)
    n--;

  /* Fill with space to align to the right, before alternate or sign */
  if (!p->flags.zero && !p->flags.align_left)
    {
      while (n-- > 0)
        __pocl_printf_putcf (p, ' ');
    }

  if (space_required)
    __pocl_printf_putcf (p, ' ');

  /* print sign */
  if (sign_required)
    {
      __pocl_printf_putcf (p, (p->flags.sign ? '-' : '+'));
    }

  /* Alternate */
  if (althex)
    {
      __pocl_printf_putcf (p, '0');
      __pocl_printf_putcf (p, (p->flags.uc ? 'X' : 'x'));
    }
  else if (altoct)
    {
      __pocl_printf_putcf (p, '0');
    }

  /* Fill with zeros, after alternate or sign */
  if (p->flags.zero)
    {
      while (n-- > 0)
        __pocl_printf_putcf (p, '0');
    }

  /* Put actual buffer */
  __pocl_printf_puts (p, p->bf);

  /* Fill with space to align to the left, after string */
  if (!p->flags.zero && p->flags.align_left)
    {
      while (n-- > 0)
        __pocl_printf_putcf (p, ' ');
    }
}

unsigned
__pocl_printf_puts_ljust (param_t *p,
                          const char *string,
                          unsigned width,
                          int max_width)
{
  char c;
  unsigned written = 0;
  if (max_width < 0)
    max_width = INT32_MAX;
  while ((c = *string++))
    {
      if (written < max_width)
        __pocl_printf_putcf (p, c);
      ++written;
    }
  while (written < width)
    {
      if (written < max_width)
        __pocl_printf_putcf (p, ' ');
      ++written;
    }
  return written;
}

unsigned
__pocl_printf_puts_rjust (param_t *p,
                          const char *string,
                          unsigned width,
                          int max_width)
{
  char c;
  unsigned i, strleng = 0, written = 0;
  if (max_width < 0)
    max_width = INT32_MAX;

  const char *tmp = string;
  while ((c = *tmp++))
    ++strleng;

  for (i = strleng; i < width; ++i)
    {
      if (written < max_width)
        __pocl_printf_putcf (p, ' ');
      ++written;
    }

  while ((c = *string++))
    {
      if (written < max_width)
        __pocl_printf_putcf (p, c);
      ++written;
    }
  return written;
}

void
__pocl_printf_ptr (param_t *p, const void *ptr)
{
  p->base = 16;
  p->flags.uc = 0;
  p->flags.alt = 1;
  p->flags.sign = 0;
  p->flags.nonzeroparam = 1;
  __pocl_printf_ul16 (p, (uintptr_t)ptr);
  __pocl_printf_putchw (p);
}

/* prints NANs and INFs */
void
__pocl_printf_nonfinite (param_t *p, const char *ptr)
{
  char c;
  char *dest = p->bf;
  while ((c = *ptr++))
    *dest++ = c;
  *dest = 0;
  /* When applied to infinite and NaN values, the -, +, and space flag
   * characters have their usual meaning; the # and 0 flag characters
   * have no effect */
  p->flags.zero = 0;

  __pocl_printf_putchw (p);
}

void
__pocl_printf_ulong (param_t *p, UINT_T u)
{
  if (p->base == 16)
    {
      p->flags.nonzeroparam = (u > 0 ? 1 : 0);
      __pocl_printf_ul16 (p, u);
    }
  else
    __pocl_printf_ul_base (p, u);

  __pocl_printf_putchw (p);
}

void
__pocl_printf_long (param_t *p, INT_T i)
{
  __pocl_printf_l_base (p, i);
  __pocl_printf_putchw (p);
}

void
__pocl_printf_float (param_t *p, FLOAT_T f)
{
#ifdef BUILD_WITH_LIBC
  __pocl_printf_float_libc (p, f);
#else

  int print_dec = 0;
  if (p->precision == 0)
    {
      if (p->flags.alt)
        {
          /* For a, A, e, E, f, F, g, and G conversions, the result of
           * converting a floating-point number always contains a decimal-point
           * character, even if no digits follow it. */
          print_dec = 1;
        }
    }

  /* f is not flipped in caller, because libc mode */
  p->flags.sign = SIGNBIT(f) ? 1 : 0;

  if (p->flags.sign)
    f = -f;

  if (p->conv == 'a')
    {
      __pocl_printf_float_a (p, print_dec, f);
      __pocl_printf_putchw (p);
      return;
    }

  char float_digits[BUFSIZE];
  char *rounded_digits = float_digits;
  int P, X, notrail0;
  int round_exp = 0, dec_point = 0;

  /* save is required for 'g' mode with float vectors */
  int saved_prec = p->precision;
  if (p->precision < 0)
    p->precision = 6;

  /* prepend a zero digit to the buffer, makes rounding up easier */
  int nonzero = (f != 0.0f);
  float_digits[0] = '0';
  float_digits[1] = 0;
  if (nonzero)
    dec_point = __pocl_generate_float_digits (f, float_digits + 1);
  round_exp = dec_point + 1;

  switch (p->conv)
    {
    case 'e':
      if (nonzero)
        rounded_digits = __pocl_printf_float_round_buf (
            f, float_digits, &round_exp, p->precision, 1);
      __pocl_printf_float_e (p, rounded_digits, round_exp, print_dec, 0);
      break;
    case 'f':
      if (nonzero)
        rounded_digits = __pocl_printf_float_round_buf (
            f, float_digits, &round_exp, p->precision, 0);
      __pocl_printf_float_f (p, rounded_digits, round_exp, print_dec, 0);
      break;

    /* Let P equal the precision if nonzero,
     * 6 if the precision is omitted,
     * or 1 if the precision is zero.
     *
     * Then, if a conversion with style E
     * would have an exponent of X:
     *
     * — if P > X ≥ −4, the conversion is with
     * style f (or F) and precision P − (X + 1).
     *
     * — otherwise, the conversion is with
     * style e (or E) and precision P − 1.
     *
     * Finally, unless the # flag
     * is used, any trailing zeros are removed from the fractional portion of
     * the result and the decimal-point character is removed if there is no
     * fractional portion remaining. */

    /* TODO: dec_point is exponent BEFORE rounding, so decision between 'e' and
     * 'f' happens before rounding. is this correct behaviour ? */
    case 'g':

      notrail0 = (p->flags.alt ? 0 : 1);

      P = p->precision;
      if (P == 0)
        P = 1;

      X = dec_point - 1;
      if ((P > X) && (X >= -4))
        {
          p->precision = P - (X + 1);
          if (nonzero)
            rounded_digits = __pocl_printf_float_round_buf (
                f, float_digits, &round_exp, p->precision, 0);
          __pocl_printf_float_f (p, rounded_digits, round_exp, print_dec,
                                 notrail0);
        }
      else
        {
          p->precision = P - 1;
          if (nonzero)
            rounded_digits = __pocl_printf_float_round_buf (
                f, float_digits, &round_exp, p->precision, 1);
          __pocl_printf_float_e (p, rounded_digits, round_exp, print_dec,
                                 notrail0);
        }
      break;
    }

  p->precision = saved_prec;
  __pocl_printf_putchw (p);
#endif
}
