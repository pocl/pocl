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

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-security"

#ifdef cl_khr_fp64
#define LARGEST_FLOAT_TYPE double
#define NAN __builtin_nan("1")
#else
#define LARGEST_FLOAT_TYPE float
#define NAN __builtin_nanf("1")
#endif

#include <stdarg.h>
#include <limits.h>

// We implement the OpenCL printf by calling the C99 printf. This is
// not very efficient, but is easy to implement.
#if LLVM_OLDER_THAN_5_0
#define OCL_C_AS __attribute__((address_space(0)))
#else
#define OCL_C_AS
#endif
int printf(OCL_C_AS const char* restrict fmt, ...);
int snprintf(OCL_C_AS char* restrict str, size_t size,
             OCL_C_AS const char* restrict fmt, ...);

// For debugging
// Use as: DEBUG_PRINTF((fmt, args...)) -- note double parentheses!
//#define DEBUG_PRINTF(args) (printf args)
#define DEBUG_PRINTF(args) ((void)0)

// Conversion flags
typedef struct {
  int left:1;
  int plus:1;
  int space:1;
  int alt:1;
  int zero:1;
} flags_t;

// Helper routines to output integers

#define INT_CONV_char  "hh"
#define INT_CONV_short "h"
#define INT_CONV_int   ""
#define INT_CONV_long  "ll"     // C99 printf uses "ll" for int64_t

#define DEFINE_PRINT_INTS(WIDTH)                                        \
  void _cl_print_ints_##WIDTH(flags_t flags, int field_width, int precision, \
                              char conv, OCL_C_AS const void* vals, int n) \
  {                                                                     \
    DEBUG_PRINTF(("[printf:ints:n=%df]\n", n));                         \
    char outfmt[1000];                                                  \
    OCL_C_AS char str[] = "%%%s%s%s%s%s%.0d%s%.0d" INT_CONV_##WIDTH "%c"; \
    snprintf(outfmt, sizeof outfmt,                                     \
             str,                                                       \
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
    OCL_C_AS char comma[] = ",";                                        \
    for (int d=0; d<n; ++d) {                                           \
      DEBUG_PRINTF(("[printf:ints:d=%d]\n", d));                        \
      if (d != 0) printf(comma);                                        \
      printf(outfmt, ((OCL_C_AS const WIDTH*)vals)[d]);                 \
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

// Defined in OpenCL
float __attribute__((overloadable)) vload_half(size_t offset,
                                               OCL_C_AS const half *p);

// Note: To simplify implementation, we print double values with %lf,
// although %f would suffice as well
#define FLOAT_GET_half(ptr)   vload_half(0, ptr)
#define FLOAT_GET_float(ptr)  (*(ptr))
#define FLOAT_GET_double(ptr) (*(ptr))

#define DEFINE_PRINT_FLOATS(WIDTH)                                      \
  void _cl_print_floats_##WIDTH(flags_t flags, int field_width, int precision, \
                                char conv, OCL_C_AS const void* vals, int n)     \
  {                                                                     \
    DEBUG_PRINTF(("[printf:floats:n=%dd]\n", n));                       \
    char outfmt[1000];                                                  \
    OCL_C_AS char str[] = "%%%s%s%s%s%s%.0d%s%.0d" "%c";                \
    snprintf(outfmt, sizeof outfmt,                                     \
             str,                                                       \
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
    OCL_C_AS char comma[] = ",";                                        \
    for (int d=0; d<n; ++d) {                                           \
      DEBUG_PRINTF(("[printf:floats:d=%d]\n", d));                      \
      if (d != 0) printf(comma);                                        \
      WIDTH val = (FLOAT_GET_##WIDTH((OCL_C_AS const WIDTH*)vals+d));   \
      if (val != val)                                                   \
        val = NAN;                                                      \
      printf (outfmt, (LARGEST_FLOAT_TYPE)val);                         \
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



// Helper routines to output characters, strings, and pointers

void _cl_print_char(flags_t flags, int field_width, int val)
{
  DEBUG_PRINTF(("[printf:char]\n"));
  char outfmt[1000];
  char string[] = "%%%s%.0dc";
  snprintf(outfmt, sizeof outfmt,
           string,
           flags.left ? "-" : "",
           field_width);
  DEBUG_PRINTF(("[printf:char:outfmt=%s]\n", outfmt));
  printf(outfmt, val);
  DEBUG_PRINTF(("[printf:char:done]\n"));
}

void
_cl_print_string (flags_t flags, int field_width, int precision,
                  OCL_C_AS const char *val)
{
  DEBUG_PRINTF(("[printf:char]\n"));
  char outfmt[1000];
  char string[] = "%%%s%.0d%s%.0ds";
  snprintf(outfmt, sizeof outfmt,
           string,
           flags.left ? "-" : "",
           field_width,
           (precision > 0) ? "." : "",
           (precision > 0) ? precision : 0);
  DEBUG_PRINTF(("[printf:char:outfmt=%s]\n", outfmt));
  printf(outfmt, val);
  DEBUG_PRINTF(("[printf:char:done]\n"));
}

void _cl_print_pointer(flags_t flags, int field_width, OCL_C_AS const void* val)
{
  DEBUG_PRINTF(("[printf:char]\n"));
  char outfmt[1000];
  char string[] = "%%%s%.0dp";
  snprintf(outfmt, sizeof outfmt,
           string,
           flags.left ? "-" : "",
           field_width);
  DEBUG_PRINTF(("[printf:char:outfmt=%s]\n", outfmt));
  printf(outfmt, val);
  DEBUG_PRINTF(("[printf:char:done]\n"));
}



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

// This should be queried from the machine in the non-TAS case.
// For now assume that if we are not using the fake address space
// ids then we have a single address space. This version of printf
// doesn't work with multiple address spaces anyways.
#ifdef POCL_USE_FAKE_ADDR_SPACE_IDS
#define OCL_CONSTANT_AS __attribute__((address_space(3)))
#else
#define OCL_CONSTANT_AS
#endif
int __cl_printf(const OCL_CONSTANT_AS char* restrict format, ...)
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
        char s[] = "%%";
        printf(s);           // literal %
        ch = *++format;
      } else {
        DEBUG_PRINTF(("[printf:arg]\n"));
        // Flags
        flags_t flags;
        flags.left = 0;
        flags.plus = 0;
        flags.space = 0;
        flags.alt = 0;
        flags.zero = 0;
        for (;;) {
          switch (ch) {
          case '-': if (flags.left) goto error; flags.left = 1; break;
          case '+': if (flags.plus) goto error; flags.plus = 1; break;
          case ' ': if (flags.space) goto error; flags.space = 1; break;
          case '#': if (flags.alt) goto error; flags.alt = 1; break;
          case '0': if (flags.zero) goto error; flags.zero = 1; break;
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
        int precision = -1;
        if (ch == '.') {
          ch = *++format;
          precision = 0;
          while (ch >= '0' && ch <= '9') {
            if (precision > (INT_MAX - 9) / 10) goto error;
            precision = 10 * precision + (ch - '0');
            ch = *++format;
          }
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
          case 4: CALL_PRINT_FLOATS(float, double); break;
#else
              break;
#endif
          }
          
#undef CALL_PRINT_FLOATS
          
          break;
          
          // Output a character
        case 'c': {
          DEBUG_PRINTF(("[printf:char]\n"));
          if (flags.plus || flags.space || flags.alt || flags.zero) goto error;
          DEBUG_PRINTF(("[printf:char1]\n"));
          if (precision != -1) goto error;
          DEBUG_PRINTF(("[printf:char2]\n"));
          if (vector_length != 1) goto error;
          DEBUG_PRINTF(("[printf:char3]\n"));
          if (length != 0) goto error;
          DEBUG_PRINTF(("[printf:char4]\n"));
          int val = va_arg(ap, int);
          _cl_print_char(flags, field_width, val);
          break;
        }
          
          // Output a string
        case 's': {
          if (flags.plus || flags.space || flags.alt || flags.zero) goto error;
          if (vector_length != 1) goto error;
          if (length != 0) goto error;
          OCL_C_AS const char* val = va_arg(ap, OCL_C_AS const char*);
          _cl_print_string (flags, field_width, precision, val);
          break;
        }
          
          // Output a pointer
        case 'p': {
          if (flags.plus || flags.space || flags.alt || flags.zero) goto error;
          if (precision != -1) goto error;
          if (vector_length != 1) goto error;
          if (length != 0) goto error;
          OCL_C_AS const void* val = va_arg(ap, OCL_C_AS const void*);
          _cl_print_pointer(flags, field_width, val);
          break;
        }
          
        default: goto error;
        }
        ch = *++format;
        
      } // not a literal %

    } else {
      DEBUG_PRINTF(("[printf:literal]\n"));
      char literal[] = "%c";
      printf(literal, ch);
      ch = *++format;
    }
  }
  
  va_end(ap);
  DEBUG_PRINTF(("[printf:done]\n"));
  return 0;
  
 error:;
  va_end(ap);
  DEBUG_PRINTF(("[printf:error]\n"));
  char string [] = "(printf format string error)";
  printf(string);
  return -1;
}

#pragma clang diagnostic pop
