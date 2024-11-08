/* OpenCL runtime library: printf buffer handling

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

#ifndef POCL_PRINTF_BUFFER_H
#define POCL_PRINTF_BUFFER_H

/* Number of bits reserved to flags in the control word. */
#define PRINTF_BUFFER_CTWORD_FLAG_BITS 7
#define PRINTF_BUFFER_CTWORD_SKIP_FMT_STR (1 << 1)
#define PRINTF_BUFFER_CTWORD_CHAR_SHORT_PR (1 << 2)
#define PRINTF_BUFFER_CTWORD_CHAR2_PR (1 << 3)
#define PRINTF_BUFFER_CTWORD_FLOAT_PR (1 << 4)
#define PRINTF_BUFFER_CTWORD_BIG_ENDIAN (1 << 5)
#define PRINTF_BUFFER_CTWORD_32BIT_POINTERS (1 << 6)

#endif // POCL_PRINTF_BUFFER_H
