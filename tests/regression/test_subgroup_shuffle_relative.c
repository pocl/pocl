// Copyright (c) 2026 PoCL developers
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.

/*
  cl_khr_subgroup_shuffle_relative: sub_group_shuffle_up/down with uniform
  deltas 0, 1 and sub-group size - 1, and a varying one, for several types.
  The kernel is built from OpenCL C source and from SPIR-V, where DPC++'s
  shift_group_left/right end up as OpGroupNonUniformShuffleDown/Up. Results
  whose source work-item is outside the sub-group are undefined and not
  checked. The expected values follow the sub-group size and local id each
  work-item reports.

  With cl_intel_required_subgroup_size, the OpenCL C kernel is also built
  with a required sub-group size of 8, so each work-group of 12 has a full
  sub-group and a partial one of 4.
*/

#include "pocl_opencl.h"

#include <stdio.h>
#include <stdlib.h>

#define CHECK_ERROR(err)                                                      \
  if (err != CL_SUCCESS)                                                      \
    {                                                                         \
      printf ("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__);         \
      return 1;                                                               \
    }

enum
{
  N = 48,
  LOCAL = 12,
  NTYPES = 7,
  REQD_SG = 8
};

static const char *type_names[NTYPES]
  = { "char", "ushort", "int", "ulong", "float", "double", "half" };

static cl_long
value (int type, cl_uint gid)
{
  if (type == 3)
    return (cl_long)(((cl_ulong)gid << 33) + 4);
  return gid + 1 + type;
}

static unsigned char *
read_file (const char *path, size_t *size)
{
  FILE *f = fopen (path, "rb");
  if (!f)
    return NULL;
  fseek (f, 0, SEEK_END);
  *size = ftell (f);
  fseek (f, 0, SEEK_SET);
  unsigned char *buf = malloc (*size + 1);
  if (fread (buf, 1, *size, f) != *size)
    {
      free (buf);
      fclose (f);
      return NULL;
    }
  buf[*size] = 0;
  fclose (f);
  return buf;
}

static int
run (cl_context context,
     cl_device_id device,
     cl_command_queue queue,
     cl_program program,
     const char *what,
     const char *options,
     int ntypes,
     cl_uint reqd_sg)
{
  cl_int err = clBuildProgram (program, 1, &device, options, NULL, NULL);
  if (err != CL_SUCCESS)
    {
      size_t log_size;
      clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                             &log_size);
      char *log = malloc (log_size);
      clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, log_size,
                             log, NULL);
      printf ("%s: build error:\n%s\n", what, log);
      free (log);
      return 1;
    }
  cl_kernel kernel = clCreateKernel (program, "shuffle_relative", &err);
  CHECK_ERROR (err);

  cl_mem info_buf = clCreateBuffer (context, CL_MEM_WRITE_ONLY,
                                    sizeof (cl_uint) * 2 * N, NULL, &err);
  CHECK_ERROR (err);
  cl_mem out_buf = clCreateBuffer (
    context, CL_MEM_WRITE_ONLY, sizeof (cl_long) * NTYPES * 8 * N, NULL, &err);
  CHECK_ERROR (err);
  CHECK_ERROR (clSetKernelArg (kernel, 0, sizeof (cl_mem), &info_buf));
  CHECK_ERROR (clSetKernelArg (kernel, 1, sizeof (cl_mem), &out_buf));

  size_t global_size = N, local_size = LOCAL;
  CHECK_ERROR (clEnqueueNDRangeKernel (queue, kernel, 1, NULL, &global_size,
                                       &local_size, 0, NULL, NULL));
  static cl_uint info[2 * N];
  static cl_long out[NTYPES * 8 * N];
  CHECK_ERROR (clEnqueueReadBuffer (queue, info_buf, CL_TRUE, 0, sizeof (info),
                                    info, 0, NULL, NULL));
  CHECK_ERROR (clEnqueueReadBuffer (queue, out_buf, CL_TRUE, 0, sizeof (out),
                                    out, 0, NULL, NULL));

  int failed = 0;
  for (cl_uint gid = 0; reqd_sg && gid < N; gid++)
    {
      cl_uint l = gid % LOCAL, first = l / reqd_sg * reqd_sg;
      cl_uint sg_size = first + reqd_sg <= LOCAL ? reqd_sg : LOCAL - first;
      if (info[2 * gid] != sg_size || info[2 * gid + 1] != l - first)
        {
          printf ("%s: work-item %u has sg size %u and lid %u, expected %u "
                  "and %u\n",
                  what, gid, info[2 * gid], info[2 * gid + 1], sg_size,
                  l - first);
          failed = 1;
        }
    }
  for (int t = 0; t < ntypes; t++)
    {
      if (t == 5 && !poclu_supports_extension (device, "cl_khr_fp64"))
        continue;
      if (t == 6 && !poclu_supports_extension (device, "cl_khr_fp16"))
        continue;
      for (cl_uint gid = 0; gid < N; gid++)
        {
          cl_uint sg_size = info[2 * gid], lid = info[2 * gid + 1];
          cl_uint first = gid - lid;
          cl_uint d[4] = { 0, 1, sg_size - 1, lid % 3 };
          for (int i = 0; i < 8; i++)
            {
              cl_uint delta = d[i % 4];
              int up = i < 4;
              if (up ? delta > lid : lid + delta >= sg_size)
                continue;
              cl_long expected
                = value (t, first + (up ? lid - delta : lid + delta));
              cl_long got = out[(t * 8 + i) * N + gid];
              if (got != expected)
                {
                  printf (
                    "%s: %s sub_group_shuffle_%s(delta %u) at %u (sg size %u, "
                    "lid %u) = %lld, expected %lld\n",
                    what, type_names[t], up ? "up" : "down", delta, gid,
                    sg_size, lid, (long long)got, (long long)expected);
                  failed = 1;
                }
            }
        }
    }

  clReleaseMemObject (info_buf);
  clReleaseMemObject (out_buf);
  clReleaseKernel (kernel);
  return failed;
}

int
main (int argc, char **argv)
{
  cl_uint platform_index = argc > 1 ? (cl_uint)atoi (argv[1]) : 0;
  cl_int err;

  cl_uint num_platforms;
  CHECK_ERROR (clGetPlatformIDs (0, NULL, &num_platforms));
  if (platform_index >= num_platforms)
    {
      printf ("Platform index %u out of range\n", platform_index);
      return 1;
    }
  cl_platform_id *platforms = malloc (sizeof (cl_platform_id) * num_platforms);
  CHECK_ERROR (clGetPlatformIDs (num_platforms, platforms, NULL));
  cl_platform_id platform = platforms[platform_index];
  free (platforms);

  cl_device_id device;
  CHECK_ERROR (
    clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

  if (!poclu_supports_extension (device, "cl_khr_subgroup_shuffle_relative"))
    {
      printf ("SKIP: The test requires cl_khr_subgroup_shuffle_relative\n");
      return 77;
    }

  cl_context context = clCreateContext (NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR (err);
  cl_queue_properties props[] = { 0 };
  cl_command_queue queue
    = clCreateCommandQueueWithProperties (context, device, props, &err);
  CHECK_ERROR (err);

  int failed = 0;
  size_t size;
  unsigned char *source
    = read_file (SRCDIR "/test_subgroup_shuffle_relative.cl", &size);
  if (!source)
    {
      printf ("Failed to read test_subgroup_shuffle_relative.cl\n");
      return 1;
    }
  cl_program program = clCreateProgramWithSource (
    context, 1, (const char **)&source, NULL, &err);
  CHECK_ERROR (err);
  failed |= run (context, device, queue, program, "OpenCL C", "-cl-std=CL3.0",
                 NTYPES, 0);
  clReleaseProgram (program);

  size_t sg_sizes[32], sg_sizes_size = 0;
  int has_reqd_sg = 0;
  if (poclu_supports_extension (device, "cl_intel_required_subgroup_size"))
    CHECK_ERROR (clGetDeviceInfo (device, CL_DEVICE_SUB_GROUP_SIZES_INTEL,
                                  sizeof (sg_sizes), sg_sizes,
                                  &sg_sizes_size));
  for (size_t i = 0; i < sg_sizes_size / sizeof (size_t); i++)
    has_reqd_sg |= sg_sizes[i] == REQD_SG;
  if (has_reqd_sg)
    {
      program = clCreateProgramWithSource (context, 1, (const char **)&source,
                                           NULL, &err);
      CHECK_ERROR (err);
      failed |= run (context, device, queue, program, "OpenCL C, partial SG",
                     "-cl-std=CL3.0 -DREQD_SG_SIZE=8", NTYPES, REQD_SG);
      clReleaseProgram (program);
    }
  else
    printf ("No sub-group size of %d, not testing partial sub-groups\n",
            REQD_SG);
  free (source);

  // The group non-uniform instructions need SPIR-V 1.3.
  if (poclu_device_supports_il (device, "SPIR-V_1.3"))
    {
      unsigned char *binary
        = read_file (SRCDIR "/test_subgroup_shuffle_relative.spv", &size);
      if (!binary)
        {
          printf ("Failed to read test_subgroup_shuffle_relative.spv\n");
          return 1;
        }
      program = clCreateProgramWithIL (context, binary, size, &err);
      CHECK_ERROR (err);
      failed |= run (context, device, queue, program, "SPIR-V", "", 5, 0);
      clReleaseProgram (program);
      free (binary);
    }
  else
    {
      printf ("SPIR-V 1.3 not supported, only testing OpenCL C\n");
    }

  clReleaseCommandQueue (queue);
  clReleaseContext (context);

  if (failed)
    return 1;
  printf ("OK\n");
  return 0;
}
