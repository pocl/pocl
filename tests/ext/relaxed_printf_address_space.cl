/* relaxed_printf_address_space.cl -
   Test cases for cl_ext_relaxed_printf_address_space

   Copyright (c) 2022 Pekka Jääskeläinen / Parmance

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

#ifndef cl_ext_relaxed_printf_address_space
#error cl_ext_relaxed_printf_address_space not supported by the device!
#endif

constant const char *constant c_fmt = "%s through a constant format str\n";
constant const char *constant c_arg = "constant address space str arg";

constant const char *constant l_fmt_init = "%s through a local format str\n";
constant const char *constant l_arg_init = "local address space str arg";

constant const char *constant p_fmt_init = "%s through a private format str\n";
constant const char *constant p_arg_init = "private address space str arg";

#define strcpy(dst, src) do {			\
    char c;					\
    int i = 0;					\
    do {					\
      c = src[i];				\
      dst[i] = c;				\
      ++i;					\
    } while (c != '\0');			\
  } while (0)

kernel void relaxed_printf_address_space_tests(global int *io,
					       global char *g_fmt,
					       global char *g_arg,
					       local char *l_fmt,
					       local char *l_arg) {
  char p_arg[1000];
  char p_fmt[1000];

  strcpy(l_fmt, l_fmt_init);
  strcpy(l_arg, l_arg_init);

  strcpy(p_arg, p_arg_init);
  strcpy(p_fmt, p_fmt_init);

  io[0] = printf(g_fmt, g_arg);
  io[0] += printf(c_fmt, c_arg);
  io[0] += printf(l_fmt, l_arg);
  io[0] += printf(p_fmt, p_arg);

  printf("\n");

  io[0] += printf(g_fmt, p_arg);
  io[0] += printf(g_fmt, c_arg);
  io[0] += printf(g_fmt, l_arg);
  io[0] += printf(g_fmt, g_arg);

  printf("\n");

  io[0] += printf(c_fmt, p_arg);
  io[0] += printf(c_fmt, g_arg);
  io[0] += printf(c_fmt, l_arg);
  io[0] += printf(c_fmt, c_arg);

  printf("\n");

  io[0] += printf(l_fmt, p_arg);
  io[0] += printf(l_fmt, g_arg);
  io[0] += printf(l_fmt, c_arg);
  io[0] += printf(l_fmt, l_arg);

  printf("\n");

  io[0] += printf(p_fmt, l_arg);
  io[0] += printf(p_fmt, g_arg);
  io[0] += printf(p_fmt, c_arg);
  io[0] += printf(p_fmt, p_arg);
}
