/* printf-kernels.cl - Test cases for SPIR-V printf (the kernels).

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

// Example build steps:
//
// clang -cc1 -triple=spir64-unknown-unknown -cl-std=CL2.0 -include \
// ~/src/llvm/clang/lib/Headers/opencl-c.h printf-kernels.cl \
// -emit-llvm-bc -o printf-kernels.bc
//
// llvm-spirv printf-kernels.bc
//
// spirv-val printf-kernels.spv

global char hello_g[] = "HELLO GLOBAL\n";
global char strings_g[] = "STRiNGS\n";

/// Cases that should adhere to both the SPIR-V and OpenCL 1.2 printf ///

kernel void ocl_std_no_args(global int *io) {
  io[0] = printf("Hello\n");
}

kernel void ocl_std_literal_str_arg(global int *io) {
  io[0] = printf("%s", "Hello");
  io[0] += printf(" %s\n", "strings.");
}

kernel void ocl_std_nop_str_arg(global int *io) {
  printf("Howdy HIPpies!\n");
  io[0] = printf("%s", "");
  io[0] += printf("");
  io[0] += printf("I'm correct since I'm not an empty string!\n");
}

kernel void ocl_std_percent_escape(global int *io) {
  printf("hello\n");
  printf("%%\n");
  printf("hello %% world\n");
  io[0] = printf("%%s\n");
}

// OpenCL 1.2-compliant fallback to print strings that are not in the
// constant memory.
void putstr(global char *str) {
  global char *pos = str;
  while (*pos) {
    printf("%c", *pos);
    ++pos;
  }
}

kernel void ocl_std_global_str_arg_emulated(global int *io) {
  putstr(hello_g);
  putstr(strings_g);
  io[0] = 0;
}

/// Useful cases that adhere to the SPIR-V/OpenCL/printf specs wording,
/// but not strictly to the OpenCL 1.2 printf specs, which the implementations
/// thus _might_ support (like PoCL in some cases does) ///

// This does not adhere to the OpenCL specs as the wording is "string literals"
// when defining %s args, thus the arg itself should be a literal, and cannot
// be given via a variable (even though the variable has a literal assigned to
// it directly and is in constant AS)? Clang 14 seems to digest this case
// without noise.
constant const char *hello_c_p = "CONSTANT AS program scope\n";
kernel void spirv_std_var_str_arg_constant_program_scope(global int *io) {
  io[0] = printf("%s", hello_c_p);
}

// This case is specifically given as an example of a non-valid use
// in the OpenCL 1.2 specs.
kernel void spirv_std_var_str_arg_constant_func_scope(global int *io) {
  constant const char *hello_c_f = "CONSTANT AS function scope\n";
  io[0] = printf("%s", hello_c_f);
}

// This is not allowed by OpenCL 1.2 standard which requires %s to originate
// from string literals and the string literals are supposed to be in constant
// AS.
kernel void spriv_std_str_arg_global(global int *io) {
  io[0] = printf("%s", hello_g);
  io[0] += printf(" %s\n", strings_g);
}

// Dynamic selection of the constant string.
kernel void spirv_std_conditional_string(global int *io) {
  constant const char *dyn_a = "io[0] was 1";
  constant const char *dyn_b = "io[0] was something else than 1, it was";
  constant const char *str = NULL;

  if (io[0] == 1) {
    str = dyn_a;
  } else {
    str = dyn_b;
  }
  io[0] = printf("%s %d\n", str, io[0]);
}

kernel void spirv_std_array_of_strings(global int* io) {
  constant const char *dyn_a = "I am a dynamic str arg A.\n";
  constant const char *dyn_b = "I am a dynamic str arg B.\n";

  constant const char *array_of_strings[] = {dyn_a, dyn_b};

  io[0] = printf("%s\n", array_of_strings[io[0]]);
}

kernel void spirv_host_defined_strings(global int* io, global char *str) {
  io[0] = printf("host defined string: %s", str);
}
