/* OpenCL built-in library: printf()

   Copyright (c) 2013 Erik Schnetter <eschnetter@perimeterinstitute.ca>
                      Perimeter Institute for Theoretical Physics
   
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

// Make the C99 printf visible again
#undef printf

#include <stdarg.h>

// We implement the OpenCL printf by calling the C99 printf. This is
// not very efficient, but is easy to implement.
int printf(const char* restrict fmt, ...);
int snprintf(char* restrict str, size_t size, const char* restrict fmt, ...);



// For debugging
// Use as: DEBUG_PRINTF((fmt, args...)) -- note double parentheses!
// #define DEBUG_PRINTF(args) (printf args)
#define DEBUG_PRINTF(args) ((void)0)



// Conversion flags
typedef struct {
  bool left;
  bool plus;
  bool space;
  bool alt;
  bool zero;
} flags_t;



// Helper routines to output integers

#define INT_CONV_char  "hh"
#define INT_CONV_short "h"
#define INT_CONV_int   ""
#define INT_CONV_long  "ll"     // C99 printf uses "ll" for int64_t

#define DEFINE_PRINT_INTS(WIDTH)                                        \
  void _cl_print_ints_##WIDTH(flags_t flags, int field_width, int precision, \
                              char conv, const void* vals, int n)       \
  {                                                                     \
    DEBUG_PRINTF(("[printf:ints:n=%df]\n", n));                         \
    char outfmt[1000];                                                  \
    snprintf(outfmt, sizeof outfmt,                                     \
             "%%%s%s%s%s%s%.0d%s%.0d" INT_CONV_##WIDTH "%c",            \
             flags.left ? "-" : "",                                     \
             flags.plus ? "+" : "",                                     \
             flags.space ? " " : "",                                    \
             flags.alt ? "#" : "",                                      \
             flags.zero ? "0" : "",                                     \
             field_width,                                               \
             precision != -1 ? "." : "",                                \
             precision != -1 ? precision : 0,                           \
             conv);                                                     \
    DEBUG_PRINTF(("[printf:ints:outfmt=%s]\n", outfmt));                \
    for (int d=0; d<n; ++d) {                                           \
      DEBUG_PRINTF(("[printf:ints:d=%d]\n", d));                        \
      if (d != 0) printf(",");                                          \
      printf(outfmt, ((const WIDTH*)vals)[d]);                          \
    }                                                                   \
    DEBUG_PRINTF(("[printf:ints:done]\n"));                             \
  }

DEFINE_PRINT_INTS(char)
DEFINE_PRINT_INTS(short)
DEFINE_PRINT_INTS(int)
#ifdef cl_khr_int64
DEFINE_PRINT_INTS(long)
#endif

#undef DEFINE_PRINT_INTS



// Helper routines to output floats

// Note: To simplify implementation, we print double values with %lf,
// although %f would suffice as well
#define FLOAT_CONV_half   "h"
#define FLOAT_CONV_float  ""
#define FLOAT_CONV_double "l"

#define DEFINE_PRINT_FLOATS(WIDTH)                                      \
  void _cl_print_floats_##WIDTH(flags_t flags, int field_width, int precision, \
                                char conv, const void* vals, int n)     \
  {                                                                     \
    DEBUG_PRINTF(("[printf:floats:n=%dd]\n", n));                       \
    char outfmt[1000];                                                  \
    snprintf(outfmt, sizeof outfmt,                                     \
             "%%%s%s%s%s%s%.0d%s%.0d" FLOAT_CONV_##WIDTH "%c",          \
             flags.left ? "-" : "",                                     \
             flags.plus ? "+" : "",                                     \
             flags.space ? " " : "",                                    \
             flags.alt ? "#" : "",                                      \
             flags.zero ? "0" : "",                                     \
             field_width,                                               \
             precision != -1 ? "." : "",                                \
             precision != -1 ? precision : 0,                           \
             conv);                                                     \
    DEBUG_PRINTF(("[printf:floats:outfmt=%s]\n", outfmt));              \
    for (int d=0; d<n; ++d) {                                           \
      DEBUG_PRINTF(("[printf:floats:d=%d]\n", d));                      \
      if (d != 0) printf(",");                                          \
      printf(outfmt, ((const WIDTH*)vals)[d]);                          \
    }                                                                   \
    DEBUG_PRINTF(("[printf:floats:done]\n"));                           \
  }

#ifdef cl_khr_fp16
DEFINE_PRINT_FLOATS(half)
#endif
DEFINE_PRINT_FLOATS(float)
#ifdef cl_khr_fp64
DEFINE_PRINT_FLOATS(double)
#endif

#undef DEFINE_PRINT_FLOATS



// The OpenCL printf routine.

// The implementation is straightforward:
// - walk through the format string
// - when a variable should be output, parse flags, field width,
//   precision, vector specifier, length, and conversion specifier
// - call a helper routine to perform the actual output
// - the helper routine is based on calling C99 printf, and constructs
//   a format string via snprintf
// - if there is an error during parsing, a "goto error" aborts the
//   routine, returning -1

int _cl_printf(const char* restrict format, ...)
{
  DEBUG_PRINTF(("[printf:format=%s]\n", format));
  va_list ap;
  va_start(ap, format);
  
  char ch = *format;
  while (ch) {
    if (ch == '%') {
      ch = *++format;
      
      if (ch == '%') {
        DEBUG_PRINTF(("[printf:%%]\n"));
        printf("%%");           // literal %
        ch = *++format;
      } else {
        DEBUG_PRINTF(("[printf:arg]\n"));
        // Flags
        flags_t flags;
        flags.left = false;
        flags.plus = false;
        flags.space = false;
        flags.alt = false;
        flags.zero = false;
        for (;;) {
          switch (ch) {
          case '-': if (flags.left) goto error; flags.left = true; break;
          case '+': if (flags.plus) goto error; flags.plus = true; break;
          case ' ': if (flags.space) goto error; flags.space = true; break;
          case '#': if (flags.alt) goto error; flags.alt = true; break;
          case '0': if (flags.zero) goto error; flags.zero = true; break;
          default: goto flags_done;
          }
          ch = *++format;
        }
      flags_done:;
        DEBUG_PRINTF(("[printf:flags:left=%d,plus=%d,space=%d,alt=%d,zero=%d]\n",
                      flags.left, flags.plus, flags.space, flags.alt, flags.zero));
        
        // Field width
        int field_width = 0;
        while (ch >= '0' && ch <= '9') {
          if (ch == '0' && field_width == 0) goto error;
          if (field_width > (INT_MAX - 9) / 10) goto error;
          field_width = 10 * field_width + (ch - '0');
          ch = *++format;
        }
        DEBUG_PRINTF(("[printf:width=%d]\n", field_width));
        
        // Precision
        int precision;
        if (ch == '.') {
          ch = *++format;
          precision = 0;
          while (ch >= '0' && ch <= '9') {
            if (precision > (INT_MAX - 9) / 10) goto error;
            precision = 10 * precision + (ch - '0');
            ch = *++format;
          }
        } else {
          precision = -1;
        }
        DEBUG_PRINTF(("[printf:precision=%d]\n", precision));
        
        // Vector specifier
        int vector_length = 0;
        if (ch == 'v') {
          ch = *++format;
          while (ch >= '0' && ch <= '9') {
            if (ch == '0' && vector_length == 0) goto error;
            if (vector_length > (INT_MAX - 9) / 10) goto error;
            vector_length = 10 * vector_length + (ch - '0');
            ch = *++format;
          }
          if (! (vector_length == 2 ||
                 vector_length == 3 ||
                 vector_length == 4 ||
                 vector_length == 8 ||
                 vector_length == 16)) goto error;
          
        }
        DEBUG_PRINTF(("[printf:vector_length=%d]\n", vector_length));
        
        // Length modifier
        int length = 0;           // default
        if (ch == 'h') {
          ch = *++format;
          if (ch == 'h') {
            ch = *++format;
            length = 1;           // "hh" -> char
          } else if (ch == 'l') {
            ch = *++format;
            length = 4;           // "hl" -> int
          } else {
            length = 2;           // "h" -> short
          }
        } else if (ch == 'l') {
          ch = *++format;
          length = 8;             // "l" -> long
        }
        if (vector_length > 0 && length == 0) goto error;
        if (vector_length == 0 && length == 4) goto error;
        if (vector_length == 0) vector_length = 1;
        DEBUG_PRINTF(("[printf:length=%d]\n", length));
        
        // Conversion specifier
        switch (ch) {
          
          // Output integers
        case 'd':
        case 'i':
        case 'o':
        case 'u':
        case 'x':
        case 'X':
          
#define CALL_PRINT_INTS(WIDTH, PROMOTED_WIDTH)                          \
          {                                                             \
            WIDTH##16 val;                                              \
            switch (vector_length) {                                    \
            default: __builtin_unreachable();                           \
            case 1: val.s0 = va_arg(ap, PROMOTED_WIDTH); break;         \
            case 2: val.s01 = va_arg(ap, WIDTH##2); break;              \
            case 3: val.s012 = va_arg(ap, WIDTH##3); break;             \
            case 4: val.s0123 = va_arg(ap, WIDTH##4); break;            \
            case 8: val.lo = va_arg(ap, WIDTH##8); break;               \
            case 16: val = va_arg(ap, WIDTH##16); break;                \
            }                                                           \
            _cl_print_ints_##WIDTH(flags, field_width, precision,       \
                                   ch, &val, vector_length);            \
          }
          
          DEBUG_PRINTF(("[printf:int:conversion=%c]\n", ch));
          switch (length) {
          default: __builtin_unreachable();
          case 1: CALL_PRINT_INTS(char, int); break;
          case 2: CALL_PRINT_INTS(short, int); break;
          case 0:
          case 4: CALL_PRINT_INTS(int, int); break;
#ifdef cl_khr_int64
          case 8: CALL_PRINT_INTS(long, long); break;
#endif
          }

#undef CALL_PRINT_INTS
          
          break;
          
          // Output floats
        case 'f':
        case 'F':
        case 'e':
        case 'E':
        case 'g':
        case 'G':
        case 'a':
        case 'A':
          
#define CALL_PRINT_FLOATS(WIDTH, PROMOTED_WIDTH)                        \
          {                                                             \
            WIDTH##16 val;                                              \
            switch (vector_length) {                                    \
            default: __builtin_unreachable();                           \
            case 1: val.s0 = va_arg(ap, PROMOTED_WIDTH); break;         \
            case 2: val.s01 = va_arg(ap, WIDTH##2); break;              \
            case 3: val.s012 = va_arg(ap, WIDTH##3); break;             \
            case 4: val.s0123 = va_arg(ap, WIDTH##4); break;            \
            case 8: val.lo = va_arg(ap, WIDTH##8); break;               \
            case 16: val = va_arg(ap, WIDTH##16); break;                \
            }                                                           \
            _cl_print_floats_##WIDTH(flags, field_width, precision,     \
                                     ch, &val, vector_length);          \
          }
          
          DEBUG_PRINTF(("[printf:float:conversion=%c]\n", ch));
          switch (length) {
          default: __builtin_unreachable();
#ifdef cl_khr_fp16
            // case 2: CALL_PRINT_FLOATS(half, double); break;
          case 2: goto error;   // not yet implemented
#endif
          case 0:
            // Note: width 0 cleverly falls through to float if double
            // is not supported
#ifdef cl_khr_fp64
          case 8: CALL_PRINT_FLOATS(double, double); break;
#endif
          case 4: CALL_PRINT_FLOATS(float, double); break;
          }
          
#undef CALL_PRINT_FLOATS
          
          break;
          
        case 'c': goto error;   // not yet implemented
        case 's': goto error;   // not yet implemented
        case 'p': goto error;   // not yet implemented
          
        default: goto error;
        }
        ch = *++format;
        
      } // not a literal %

    } else {
      DEBUG_PRINTF(("[printf:char]\n"));
      printf("%c", ch);
      ch = *++format;
    }
  }
  
  va_end(ap);
  DEBUG_PRINTF(("[printf:done]\n"));
  return 0;
  
 error:;
  va_end(ap);
  DEBUG_PRINTF(("[printf:error]\n"));
  return -1;
}
