/* Tests that clCreateProgramWithBinary gives each device its own binary, and
   writes every binary_status entry, when the device list mixes sub-devices of
   one root device with a second root device.

   PoCL builds programs for root devices, so the device list is reduced to its
   distinct roots. The per-device arrays (binaries, lengths, binary_status)
   are the caller's, though, and must be read and written by the caller's
   index. With the list [sub1, sub2, other] the distinct roots are
   [other, root] (in that order), so reading the arrays by the reduced index
   gives `other` the sub-devices' binary and never writes binary_status[2].
   The same devices listed as [other, sub1, sub2] are the control.

   Needs a partitionable device and a second root device; run with
   POCL_DEVICES="pthread basic". Returns 77 (skip) when there aren't two.

   Copyright (c) 2026 PoCL developers

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

#include "poclu.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK(x)                                                              \
  do                                                                          \
    {                                                                         \
      cl_int e_ = (x);                                                        \
      if (e_ != CL_SUCCESS)                                                   \
        {                                                                     \
          printf ("FAIL %s: %d\n", #x, e_);                                   \
          exit (1);                                                           \
        }                                                                     \
    }                                                                         \
  while (0)
#define UNWRITTEN 12345

static const char *SRC7
    = "kernel void k(global int *o) { o[get_global_id(0)] = 7; }";
static const char *SRC9
    = "kernel void k(global int *o) { o[get_global_id(0)] = 9; }";

/* Build src on ctx's devices; return the binary reported for `want`, or for
   `alt` (its root) where the program lists root devices. */
static unsigned char *
binary_for (cl_context ctx, const char *src, cl_device_id want,
            cl_device_id alt, size_t *len)
{
  cl_int e;
  cl_program p = clCreateProgramWithSource (ctx, 1, &src, NULL, &e);
  CHECK (e);
  CHECK (clBuildProgram (p, 0, NULL, NULL, NULL, NULL));
  cl_uint n;
  CHECK (clGetProgramInfo (p, CL_PROGRAM_NUM_DEVICES, sizeof n, &n, NULL));
  cl_device_id *d = malloc (n * sizeof *d);
  size_t *sz = malloc (n * sizeof *sz);
  unsigned char **b = malloc (n * sizeof *b);
  CHECK (clGetProgramInfo (p, CL_PROGRAM_DEVICES, n * sizeof *d, d, NULL));
  CHECK (clGetProgramInfo (p, CL_PROGRAM_BINARY_SIZES, n * sizeof *sz, sz,
                           NULL));
  for (cl_uint i = 0; i < n; i++)
    b[i] = malloc (sz[i] ? sz[i] : 1);
  CHECK (clGetProgramInfo (p, CL_PROGRAM_BINARIES, n * sizeof *b, b, NULL));
  for (cl_uint i = 0; i < n; i++)
    if (d[i] == want || d[i] == alt)
      {
        *len = sz[i];
        return b[i];
      }
  printf ("FAIL: device not in CL_PROGRAM_DEVICES\n");
  exit (1);
}

/* Create a program from bins for devs; every status must be written and
   CL_SUCCESS, and every device must run its own binary (want[i]). */
static int
trial (const char *name, cl_context ctx, cl_device_id *devs,
       unsigned char **bins, size_t *lens, const int *want)
{
  cl_int e, st[3] = { UNWRITTEN, UNWRITTEN, UNWRITTEN };
  cl_program p = clCreateProgramWithBinary (
      ctx, 3, devs, lens, (const unsigned char **)bins, st, &e);
  printf ("%s: error %d, binary_status {%d, %d, %d}", name, e, st[0], st[1],
          st[2]);
  int bad = (e != CL_SUCCESS);
  for (int i = 0; i < 3; i++)
    bad |= (st[i] != CL_SUCCESS);
  if (e == CL_SUCCESS)
    {
      CHECK (clBuildProgram (p, 3, devs, NULL, NULL, NULL));
      cl_kernel k = clCreateKernel (p, "k", &e);
      CHECK (e);
      cl_mem m = clCreateBuffer (ctx, CL_MEM_READ_WRITE, sizeof (int), NULL,
                                 &e);
      CHECK (e);
      CHECK (clSetKernelArg (k, 0, sizeof m, &m));
      printf (", outputs {");
      for (int i = 0; i < 3; i++)
        {
          cl_command_queue q
              = clCreateCommandQueueWithProperties (ctx, devs[i], NULL, &e);
          CHECK (e);
          int z = 0, r = -1;
          size_t g = 1;
          CHECK (clEnqueueWriteBuffer (q, m, CL_TRUE, 0, sizeof z, &z, 0, NULL,
                                       NULL));
          CHECK (clEnqueueNDRangeKernel (q, k, 1, NULL, &g, NULL, 0, NULL,
                                         NULL));
          CHECK (clEnqueueReadBuffer (q, m, CL_TRUE, 0, sizeof r, &r, 0, NULL,
                                      NULL));
          printf ("%s%d", i ? ", " : "", r);
          bad |= (r != want[i]);
          CHECK (clReleaseCommandQueue (q));
        }
      printf ("}");
      CHECK (clReleaseMemObject (m));
      CHECK (clReleaseKernel (k));
      CHECK (clReleaseProgram (p));
    }
  printf (" -> %s\n", bad ? "WRONG" : "ok");
  return bad;
}

int
main (void)
{
  cl_platform_id pl;
  CHECK (clGetPlatformIDs (1, &pl, NULL));
  cl_uint nd;
  CHECK (clGetDeviceIDs (pl, CL_DEVICE_TYPE_ALL, 0, NULL, &nd));
  cl_device_id all[8];
  CHECK (clGetDeviceIDs (pl, CL_DEVICE_TYPE_ALL, nd > 8 ? 8 : nd, all, NULL));
  cl_device_id root = NULL, other = NULL;
  for (cl_uint i = 0; i < nd && i < 8; i++)
    {
      cl_uint maxsub = 0;
      clGetDeviceInfo (all[i], CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
                       sizeof maxsub, &maxsub, NULL);
      if (maxsub >= 2 && !root)
        root = all[i];
      else if (!other)
        other = all[i];
    }
  if (!root || !other)
    {
      printf ("SKIP: needs a partitionable device and a second device\n");
      return 77;
    }

  cl_device_partition_property pp[]
      = { CL_DEVICE_PARTITION_BY_COUNTS, 1, 1,
          CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0 };
  cl_device_id sub[2];
  CHECK (clCreateSubDevices (root, pp, 2, sub, NULL));
  cl_int e;
  cl_device_id cdevs[3] = { sub[0], sub[1], other };
  cl_context ctx = clCreateContext (NULL, 3, cdevs, NULL, NULL, &e);
  CHECK (e);

  size_t lsub, lother;
  unsigned char *bsub = binary_for (ctx, SRC7, sub[0], root, &lsub);
  unsigned char *bother = binary_for (ctx, SRC9, other, other, &lother);

  cl_device_id ctl[3] = { other, sub[0], sub[1] };
  unsigned char *cbins[3] = { bother, bsub, bsub };
  size_t clens[3] = { lother, lsub, lsub };
  const int cwant[3] = { 9, 7, 7 };
  cl_device_id shift[3] = { sub[0], sub[1], other };
  unsigned char *sbins[3] = { bsub, bsub, bother };
  size_t slens[3] = { lsub, lsub, lother };
  const int swant[3] = { 7, 7, 9 };

  int bad = trial ("[other, sub1, sub2]", ctx, ctl, cbins, clens, cwant);
  bad |= trial ("[sub1, sub2, other]", ctx, shift, sbins, slens, swant);

  CHECK (clReleaseContext (ctx));
  CHECK (clReleaseDevice (sub[0]));
  CHECK (clReleaseDevice (sub[1]));
  printf (bad ? "FAIL\n" : "OK\n");
  return bad ? EXIT_FAILURE : EXIT_SUCCESS;
}
