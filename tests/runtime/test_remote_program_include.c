/* test_remote_program_include.c - test client-side preprocessing of
   programs built with the remote driver

   Copyright (c) 2026 PoCL developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, and/or copy the Software,
   and to permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

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

#include "config.h"
#include "poclu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM 16

/* resolved with #include "..." relative to the CWD (the build dir) */
static char helper1[] = "#define MULVAL 3.0f\n";

/* resolved with #include <...> via the -I. build option */
static char helper2[] = "#define ADDVAL 7.0f\n";

static const char src1[] =
  "#include \"pocl_remote_test_helper1.h\"\n"
  "#ifndef NO_MUL\n"
  "#define MUL(a) ((a) * MULVAL)\n"
  "#else\n"
  "#define MUL(a) (a)\n"
  "#endif\n"
  "kernel void k1(global float *a, global float *b)\n"
  "{\n"
  "  int i = get_global_id (0);\n"
  "  a[i] = MUL (a[i]) + b[i];\n"
  "}\n";

static const char src2[] =
  "#ifdef USE_SYS_INC\n"
  "#include <pocl_remote_test_helper2.h>\n"
  "#else\n"
  "#error USE_SYS_INC not defined\n"
  "#endif\n"
  "kernel void k2(global float *a, global float *b)\n"
  "{\n"
  "  int i = get_global_id (0);\n"
  "  a[i] = a[i] + b[i] * ADDVAL;\n"
  "}\n";

/* the input header never exists in the client's filesystem */
static const char src3[] =
  "#include \"pocl_remote_test_input_header.h\"\n"
  "kernel void k3(global float *a)\n"
  "{\n"
  "  int i = get_global_id (0);\n"
  "  a[i] = a[i] * OFFSET;\n"
  "}\n";

static const char header3_src[] = "#define OFFSET 5.0f\n";

static const char src_missing[] =
  "#include \"pocl_remote_test_does_not_exist.h\"\n"
  "kernel void k4(global float *a) { a[0] = 1.0f; }\n";

#define CHECK_ERR(e)                                                        \
  do                                                                        \
    {                                                                       \
      if ((e) != CL_SUCCESS)                                                \
        {                                                                   \
          printf ("%s failed: %i at %s:%i\n", #e, (int)(e), __FILE__,       \
                  __LINE__);                                                \
          exit (1);                                                         \
        }                                                                   \
    }                                                                       \
  while (0)

#define CHECK_LAST(func)                                                    \
  do                                                                        \
    {                                                                       \
      if (err != CL_SUCCESS)                                                \
        {                                                                   \
          printf ("%s failed: %i at %s:%i\n", #func, (int)err, __FILE__,    \
                  __LINE__);                                                \
          exit (1);                                                         \
        }                                                                   \
    }                                                                       \
  while (0)

static cl_program
create_program (cl_context ctx, const char *src)
{
  cl_int err;
  cl_program p = clCreateProgramWithSource (ctx, 1, &src, NULL, &err);
  CHECK_LAST (clCreateProgramWithSource);
  return p;
}

static void
run_float_kernel (cl_command_queue q, cl_program p, const char *kname,
                  cl_uint num_args, cl_mem *args)
{
  cl_int err;
  cl_kernel k = clCreateKernel (p, kname, &err);
  CHECK_LAST (clCreateKernel);
  for (cl_uint i = 0; i < num_args; ++i)
    {
      err = clSetKernelArg (k, i, sizeof (cl_mem), &args[i]);
      CHECK_ERR (err);
    }
  size_t off[3] = { 0, 0, 0 };
  size_t gs[3] = { NUM, 1, 1 };
  err = clEnqueueNDRangeKernel (q, k, 1, off, gs, gs, 0, NULL, NULL);
  CHECK_LAST (clEnqueueNDRangeKernel);
  err = clFinish (q);
  CHECK_LAST (clFinish);
  CHECK_ERR (clReleaseKernel (k));
}

int
main (void)
{
  cl_int err;
  cl_context ctx = NULL;
  cl_device_id did = NULL;
  cl_command_queue q = NULL;
  cl_platform_id pid = NULL;

  CHECK_ERR (poclu_get_any_device2 (&ctx, &did, &q, &pid));
  if (!ctx || !did || !q)
    {
      printf ("no device\n");
      return 1;
    }

  err = poclu_write_file_in_mode (
      BUILDDIR "/tests/runtime/pocl_remote_test_helper1.h", helper1,
      sizeof (helper1) - 1, "w");
  CHECK_ERR (err);
  err = poclu_write_file_in_mode (
      BUILDDIR "/tests/runtime/pocl_remote_test_helper2.h", helper2,
      sizeof (helper2) - 1, "w");
  CHECK_ERR (err);

  float ain[NUM], bin[NUM], out[NUM];
  for (int i = 0; i < NUM; ++i)
    {
      ain[i] = (float)i;
      bin[i] = (float)(NUM - i);
    }

  /* test 1: #include "..." + macros + conditional compilation */
  {
    cl_program p = create_program (ctx, src1);
    err = clBuildProgram (p, 1, &did,
                          "-I" BUILDDIR "/tests/runtime", NULL, NULL);
    CHECK_LAST (clBuildProgram);
    cl_mem a = clCreateBuffer (ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               sizeof (ain), ain, &err);
    CHECK_LAST (clCreateBuffer);
    cl_mem b = clCreateBuffer (ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof (bin), bin, &err);
    CHECK_LAST (clCreateBuffer);
    cl_mem args[2] = { a, b };
    run_float_kernel (q, p, "k1", 2, args);
    err = clEnqueueReadBuffer (q, a, CL_TRUE, 0, sizeof (out), out, 0, NULL,
                               NULL);
    CHECK_LAST (clEnqueueReadBuffer);
    for (int i = 0; i < NUM; ++i)
      if (out[i] != (float)i * 3.0f + (float)(NUM - i))
        {
          printf ("test 1 mismatch at %i: %f\n", i, out[i]);
          return 1;
        }
    CHECK_ERR (clReleaseMemObject (a));
    CHECK_ERR (clReleaseMemObject (b));
    CHECK_ERR (clReleaseProgram (p));
  }

  /* test 2: #include <...> via -I. + -D from the build options */
  {
    cl_program p = create_program (ctx, src2);
    err = clBuildProgram (p, 1, &did,
                          "-I" BUILDDIR "/tests/runtime -DUSE_SYS_INC", NULL,
                          NULL);
    CHECK_LAST (clBuildProgram);
    for (int i = 0; i < NUM; ++i)
      ain[i] = (float)i;
    cl_mem a = clCreateBuffer (ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               sizeof (ain), ain, &err);
    CHECK_LAST (clCreateBuffer);
    cl_mem b = clCreateBuffer (ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               sizeof (bin), bin, &err);
    CHECK_LAST (clCreateBuffer);
    cl_mem args[2] = { a, b };
    run_float_kernel (q, p, "k2", 2, args);
    err = clEnqueueReadBuffer (q, a, CL_TRUE, 0, sizeof (out), out, 0, NULL,
                               NULL);
    CHECK_LAST (clEnqueueReadBuffer);
    for (int i = 0; i < NUM; ++i)
      if (out[i] != (float)i + (float)(NUM - i) * 7.0f)
        {
          printf ("test 2 mismatch at %i: %f\n", i, out[i]);
          return 1;
        }
    CHECK_ERR (clReleaseMemObject (a));
    CHECK_ERR (clReleaseMemObject (b));
    CHECK_ERR (clReleaseProgram (p));
  }

  /* test 3: clCompileProgram() input header that doesn't exist on disk,
     followed by clLinkProgram() */
  {
    cl_program hdr = create_program (ctx, header3_src);
    cl_program p = create_program (ctx, src3);
    const char *inc_name = "pocl_remote_test_input_header.h";
    err = clCompileProgram (p, 1, &did, NULL, 1, &hdr, &inc_name, NULL, NULL);
    CHECK_LAST (clCompileProgram);

    cl_program linked
        = clLinkProgram (ctx, 1, &did, NULL, 1, &p, NULL, NULL, &err);
    CHECK_LAST (clLinkProgram);

    for (int i = 0; i < NUM; ++i)
      ain[i] = (float)(i + 1);
    cl_mem a = clCreateBuffer (ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               sizeof (ain), ain, &err);
    CHECK_LAST (clCreateBuffer);
    cl_mem args[1] = { a };
    run_float_kernel (q, linked, "k3", 1, args);
    err = clEnqueueReadBuffer (q, a, CL_TRUE, 0, sizeof (out), out, 0, NULL,
                               NULL);
    CHECK_LAST (clEnqueueReadBuffer);
    for (int i = 0; i < NUM; ++i)
      if (out[i] != (float)(i + 1) * 5.0f)
        {
          printf ("test 3 mismatch at %i: %f\n", i, out[i]);
          return 1;
        }
    CHECK_ERR (clReleaseMemObject (a));
    CHECK_ERR (clReleaseProgram (linked));
    CHECK_ERR (clReleaseProgram (p));
    CHECK_ERR (clReleaseProgram (hdr));
  }

  /* test 4: an unresolvable #include must fail the build, not crash */
  {
    cl_program p = create_program (ctx, src_missing);
    err = clBuildProgram (p, 1, &did, NULL, NULL, NULL);
    if (err != CL_BUILD_PROGRAM_FAILURE)
      {
        printf ("expected CL_BUILD_PROGRAM_FAILURE, got %i\n", (int)err);
        return 1;
      }
    CHECK_ERR (clReleaseProgram (p));
  }

  CHECK_ERR (clReleaseCommandQueue (q));
  CHECK_ERR (clReleaseContext (ctx));

  printf ("OK\n");
  return EXIT_SUCCESS;
}
