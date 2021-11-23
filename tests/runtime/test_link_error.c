/* Test that link errors are caught at clBuildProgram time

   Copyright (c) 2015 Giuseppe Bilotta
                      Kalray

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

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define MAX_PLATFORMS 32
#define MAX_DEVICES   32
#define MAX_BINARIES  32

/* A program that fails to link */
char program_src[] =
  "void test_func();\n" /* declared but not defined */
  "kernel void test_kernel(global int *a) { test_func(); }\n";

int
main(void){
  cl_int err;

  cl_context context;
  cl_device_id did;
  cl_command_queue queue;

  CHECK_CL_ERROR(poclu_get_any_device(&context, &did, &queue));
  TEST_ASSERT( context );
  TEST_ASSERT( did );
  TEST_ASSERT( queue );

  size_t program_size = strlen(program_src);
  char* program_buffer = program_src;

  cl_program program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer,
                                     &program_size, &err);
  //clCreateProgramWithSource for the program with #include failed
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  err = clBuildProgram(program, 1, &did, NULL, NULL, NULL);
  TEST_ASSERT(err == CL_BUILD_PROGRAM_FAILURE);

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseProgram (program));
  CHECK_CL_ERROR (clReleaseContext (context));

  CHECK_CL_ERROR (clUnloadCompiler ());

  printf ("OK\n");
  return EXIT_SUCCESS;
}
