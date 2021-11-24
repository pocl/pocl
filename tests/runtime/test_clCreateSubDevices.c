/* Test clCreateSubDevices

   Copyright (C) 2015 Giuseppe Bilotta <giuseppe.bilotta@gmail.com>

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "poclu.h"
#include "config.h"

/* Two different kernel sources, to ensure that the test for all devices
 * and the test with only sub-devices do not interphere with each other
 * (cache-wise)
 */
static const char *prog_src_all = "kernel void\n"
"setidx(global int *buf, int idx) {\n"
"buf[idx] = idx;\n"
"}\n";

static const char *prog_src_two = "kernel void\n"
"setidx(global int *buf, int idx) {\n"
"buf[idx] = -idx;\n"
"}\n";


int test_context(cl_context ctx, const char *prog_src, int mul,
  int ndevs, cl_device_id *devs) {
  cl_int err;
  cl_command_queue queue[ndevs];
  cl_program prog;
  cl_kernel krn;
  cl_mem buf;
  cl_event evt[ndevs];
  cl_int i;

  prog = clCreateProgramWithSource(ctx, 1, &prog_src, NULL, &err);
  CHECK_OPENCL_ERROR_IN("create program");

  CHECK_CL_ERROR(clBuildProgram(prog, 0, NULL, NULL, NULL, NULL));

  krn = clCreateKernel(prog, "setidx", &err);
  CHECK_OPENCL_ERROR_IN("create kernel");

  buf = clCreateBuffer(ctx, CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_WRITE |
    CL_MEM_HOST_READ_ONLY, ndevs*sizeof(cl_int), NULL, &err);
  CHECK_OPENCL_ERROR_IN("create buffer");

  CHECK_CL_ERROR(clSetKernelArg(krn, 0, sizeof(cl_mem), &buf));

  /* create one queue per device, and submit task, waiting for all
   * previous */
  for (i = 0; i < ndevs; ++i) {
    queue[i] = clCreateCommandQueue(ctx, devs[i], 0, &err);
    CHECK_OPENCL_ERROR_IN("create queue");
    err = clSetKernelArg(krn, 1, sizeof(i), &i);
    CHECK_OPENCL_ERROR_IN("set kernel arg 1");
    // no wait list for first (root) device
    err = clEnqueueTask(queue[i], krn, i, i ? evt : NULL, evt + i);
    CHECK_OPENCL_ERROR_IN("submit task");
  }

  /* enqueue map on last */
  cl_int *buf_host = clEnqueueMapBuffer(queue[ndevs - 1], buf, CL_TRUE,
    CL_MAP_READ, 0, ndevs*sizeof(cl_int), ndevs, evt, NULL, &err);
  CHECK_OPENCL_ERROR_IN("map buffer");

  int mismatch = 0;
  for (i = 0; i < ndevs; ++i) {
    mismatch += !!(buf_host[i] != i*mul);
  }
  TEST_ASSERT(mismatch == 0);

  /* enqueue unmap on first */
  CHECK_CL_ERROR(clEnqueueUnmapMemObject(queue[0], buf, buf_host, 0, NULL, NULL));

  for (i = 0 ; i < ndevs; ++i) {
    err = clFinish(queue[i]);
    err |= clReleaseCommandQueue(queue[i]);
    err |= clReleaseEvent(evt[i]);
  }

  err |= clReleaseKernel(krn);
  err |= clReleaseMemObject(buf);
  err |= clReleaseProgram(prog);
  err |= clReleaseContext(ctx);

  CHECK_OPENCL_ERROR_IN("cleanup");

  return CL_SUCCESS;

}

int main(int argc, char **argv)
{
  cl_context ctx;
  cl_command_queue q;
  // root device, all devices
#define NUMDEVS 6
  cl_device_id rootdev, alldevs[NUMDEVS];
  // pointers to the sub devices of the partitions EQUALLY and BY_COUNTS
  // respectively
  cl_device_id
    *eqdev = alldevs + 1,
    *countdev = alldevs + 4;
  cl_uint max_cus, max_subs, split;
  cl_uint i, j;

  cl_int err = poclu_get_any_device(&ctx, &rootdev, &q);
  CHECK_OPENCL_ERROR_IN("poclu_get_any_device");
  TEST_ASSERT( ctx );
  TEST_ASSERT( rootdev );
  TEST_ASSERT( q );

  alldevs[0] = rootdev;

  err = clGetDeviceInfo(rootdev, CL_DEVICE_MAX_COMPUTE_UNITS,
    sizeof(max_cus), &max_cus, NULL);
  CHECK_OPENCL_ERROR_IN("CL_DEVICE_MAX_COMPUTE_UNITS");
  if (max_cus < 2)
    {
      printf("This test requires a cl device with at least 2 compute units"
             " (a dual-core or better CPU)\n");
      return 77;
    }

  err = clGetDeviceInfo(rootdev, CL_DEVICE_PARTITION_MAX_SUB_DEVICES,
    sizeof(max_subs), &max_subs, NULL);
  CHECK_OPENCL_ERROR_IN("CL_DEVICE_PARTITION_MAX_SUB_DEVICES");

  // test fails without possible sub-devices, e.g. with basic pocl device
  TEST_ASSERT(max_subs > 1);

  cl_device_partition_property *dev_pt;
  size_t dev_pt_size;

  err = clGetDeviceInfo(rootdev, CL_DEVICE_PARTITION_PROPERTIES,
    0, NULL, &dev_pt_size);
  CHECK_OPENCL_ERROR_IN("CL_DEVICE_PARTITION_PROPERTIES size");

  dev_pt = malloc(dev_pt_size);
  TEST_ASSERT(dev_pt);
  err = clGetDeviceInfo(rootdev, CL_DEVICE_PARTITION_PROPERTIES,
    dev_pt_size, dev_pt, NULL);
  CHECK_OPENCL_ERROR_IN("CL_DEVICE_PARTITION_PROPERTIES");

  j = dev_pt_size / sizeof (*dev_pt); // number of partition types

  // check that partition types EQUALLY and BY_COUNTS are supported
  int found = 0;
  for (i = 0; i < j; ++i)
    {
      if (dev_pt[i] == CL_DEVICE_PARTITION_EQUALLY
          || dev_pt[i] == CL_DEVICE_PARTITION_BY_COUNTS)
        ++found;
    }

  TEST_ASSERT(found == 2);

  // here we will store the partition types returned by the subdevices
  cl_device_partition_property *ptype = NULL;
  size_t ptype_size;
  cl_uint numdevs = 0;

  cl_device_id parent;
  cl_uint sub_cus;

  /* CL_DEVICE_PARTITION_EQUALLY */

  printf("Max CUs: %u\n", max_cus);

  /* if the device has 3 CUs, 3 subdevices will be created, otherwise 2. */
  if (max_cus == 3)
    split = 3;
  else
    split = 2;

  const cl_device_partition_property equal_splitter[] = {
    CL_DEVICE_PARTITION_EQUALLY, max_cus/split, 0 };

  err = clCreateSubDevices(rootdev, equal_splitter, 0, NULL, &numdevs);
  CHECK_OPENCL_ERROR_IN("count sub devices");
  TEST_ASSERT(numdevs == split);

  err = clCreateSubDevices(rootdev, equal_splitter, split, eqdev, NULL);
  CHECK_OPENCL_ERROR_IN("partition equally");
  if (split == 2)
     eqdev[2] = NULL;

  cl_uint refc;
  err = clGetDeviceInfo (eqdev[0], CL_DEVICE_REFERENCE_COUNT, sizeof (refc),
                         &refc, NULL);
  CHECK_OPENCL_ERROR_IN ("get refcount");
  TEST_ASSERT (refc == 1);

  /* First, check that the root device is untouched */

  err = clGetDeviceInfo(rootdev, CL_DEVICE_MAX_COMPUTE_UNITS,
    sizeof(sub_cus), &sub_cus, NULL);
  CHECK_OPENCL_ERROR_IN("parenty CU");
  TEST_ASSERT(sub_cus == max_cus);

  err = clGetDeviceInfo(rootdev, CL_DEVICE_PARENT_DEVICE,
    sizeof(parent), &parent, NULL);
  CHECK_OPENCL_ERROR_IN("root parent device");
  TEST_ASSERT(parent == NULL);

  /* partition type may either be NULL or contain a 0 entry */
  err = clGetDeviceInfo(rootdev, CL_DEVICE_PARTITION_TYPE,
    0, NULL, &ptype_size);
  CHECK_OPENCL_ERROR_IN("root partition type");

  if (ptype_size != 0) {
    /* abuse dev_pt which should be large enough */
    TEST_ASSERT(ptype_size == sizeof(cl_device_partition_property));
    TEST_ASSERT(ptype_size <= dev_pt_size);
    err = clGetDeviceInfo(rootdev, CL_DEVICE_PARTITION_TYPE,
      ptype_size, dev_pt, NULL);
    CHECK_OPENCL_ERROR_IN("root partition type #2");
    TEST_ASSERT(dev_pt[0] == 0);
  }

  /* now test the subdevices */
  for (i = 0; i < split; ++i) {
    err = clGetDeviceInfo(eqdev[i], CL_DEVICE_MAX_COMPUTE_UNITS,
      sizeof(sub_cus), &sub_cus, NULL);
    CHECK_OPENCL_ERROR_IN("sub CU");
    TEST_ASSERT(sub_cus == max_cus/split);

    err = clGetDeviceInfo(eqdev[i], CL_DEVICE_PARENT_DEVICE,
      sizeof(parent), &parent, NULL);
    CHECK_OPENCL_ERROR_IN("sub parent device");
    TEST_ASSERT(parent == rootdev);

    err = clGetDeviceInfo(eqdev[i], CL_DEVICE_PARTITION_TYPE,
      0, NULL, &ptype_size);
    CHECK_OPENCL_ERROR_IN("sub partition type");
    TEST_ASSERT(ptype_size == sizeof(equal_splitter));

    ptype = malloc(ptype_size);
    TEST_ASSERT(ptype);
    err = clGetDeviceInfo(eqdev[i], CL_DEVICE_PARTITION_TYPE,
      ptype_size, ptype, NULL);
    CHECK_OPENCL_ERROR_IN("sub partition type #2");

    TEST_ASSERT(memcmp(ptype, equal_splitter, ptype_size) == 0);

    /* free the partition type */
    free(ptype) ; ptype = NULL;
  }

  /* CL_DEVICE_PARTITION_BY_COUNTS */

  /* Note that the platform will only read this to the first 0,
   * which is actually CL_DEVICE_PARTITION_BY_COUNTS_LIST_END;
   * the test is structured with an additional final 0 intentionally,
   * to follow the Khoronos doc example
   */
  const cl_device_partition_property count_splitter[] = {
    CL_DEVICE_PARTITION_BY_COUNTS, 1, max_cus - 1,
    CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0 };

  err = clCreateSubDevices(rootdev, count_splitter, 0, NULL, &numdevs);
  CHECK_OPENCL_ERROR_IN("count sub devices");
  TEST_ASSERT(numdevs == 2);

  err = clCreateSubDevices(rootdev, count_splitter, 2, countdev, NULL);
  CHECK_OPENCL_ERROR_IN("partition by counts");

  /* First, check that the root device is untouched */

  err = clGetDeviceInfo(rootdev, CL_DEVICE_MAX_COMPUTE_UNITS,
    sizeof(sub_cus), &sub_cus, NULL);
  CHECK_OPENCL_ERROR_IN("parenty CU");
  TEST_ASSERT(sub_cus == max_cus);

  err = clGetDeviceInfo(rootdev, CL_DEVICE_PARENT_DEVICE,
    sizeof(parent), &parent, NULL);
  CHECK_OPENCL_ERROR_IN("root parent device");
  TEST_ASSERT(parent == NULL);

  /* partition type may either be NULL or contain a 0 entry */
  err = clGetDeviceInfo(rootdev, CL_DEVICE_PARTITION_TYPE,
    0, NULL, &ptype_size);
  CHECK_OPENCL_ERROR_IN("root partition type");

  if (ptype_size != 0) {
    /* abuse dev_pt which should be large enough */
    TEST_ASSERT(ptype_size == sizeof(cl_device_partition_property));
    TEST_ASSERT(ptype_size <= dev_pt_size);
    err = clGetDeviceInfo(rootdev, CL_DEVICE_PARTITION_TYPE,
      ptype_size, dev_pt, NULL);
    CHECK_OPENCL_ERROR_IN("root partition type #2");
    TEST_ASSERT(dev_pt[0] == 0);
  }

  // devices might be returned in different order than the counts
  // in the count_splitter

  int found_cus[2] = {0, 0};

  /* now test the subdevices */
  for (i = 0; i < 2; ++i) {
    err = clGetDeviceInfo(countdev[i], CL_DEVICE_MAX_COMPUTE_UNITS,
      sizeof(sub_cus), &sub_cus, NULL);
    CHECK_OPENCL_ERROR_IN("sub CU");
    if (sub_cus == count_splitter[1])
        found_cus[0] += 1;
    else if (sub_cus == count_splitter[2])
        found_cus[1] += 1;

    err = clGetDeviceInfo(countdev[i], CL_DEVICE_PARENT_DEVICE,
      sizeof(parent), &parent, NULL);
    CHECK_OPENCL_ERROR_IN("sub parent device");
    TEST_ASSERT(parent == rootdev);

    /* The partition type returned is up to the first 0,
     * which happens to be the CL_DEVICE_PARTITION_BY_COUNTS_LIST_END,
     * not the final terminating 0 in count_splitter, so it has one less
     * element. It should be otherwise equal */
    err = clGetDeviceInfo(countdev[i], CL_DEVICE_PARTITION_TYPE,
      0, NULL, &ptype_size);
    CHECK_OPENCL_ERROR_IN("sub partition type");
    TEST_ASSERT(ptype_size == sizeof(count_splitter) - sizeof(*count_splitter));

    ptype = malloc(ptype_size);
    TEST_ASSERT(ptype);
    err = clGetDeviceInfo(countdev[i], CL_DEVICE_PARTITION_TYPE,
      ptype_size, ptype, NULL);
    CHECK_OPENCL_ERROR_IN("sub partition type #2");

    TEST_ASSERT(memcmp(ptype, count_splitter, ptype_size) == 0);

    /* free the partition type */
    free(ptype) ; ptype = NULL;
  }

  /* the previous loop finds 1+1 subdevices only on >dual core systems;
   * on dual cores, the count_splitter is [1, 1] and the above
   * "(sub_cus == count_splitter[x])" results in 2+0 subdevices found */
  if (max_cus > 2)
    TEST_ASSERT(found_cus[0] == 1 && found_cus[1] == 1);
  else
    TEST_ASSERT((found_cus[0] + found_cus[1]) == 2);

  /* So far, so good. Let's now try and use these devices,
   * by building a program for all of them and launching kernels on them.
   *
   * Note that there's a discrepancy in behavior between implementations:
   * some assume you can treat sub-devices as their parent device, and thus
   * e.g. using them through any context which includes their parent devices,
   * other fail miserably if you try this.
   *
   * For the time being we will test the stricter behavior, where
   * sub-devices should be added manually to a context.
   */

  err = clReleaseCommandQueue(q);
  CHECK_OPENCL_ERROR_IN("clReleaseCommandQueue");
  err = clReleaseContext(ctx);
  CHECK_OPENCL_ERROR_IN("clReleaseContext");

  /* if we split into 2 equal parts, third pointer is NULL. Let's copy the
   * previous device to it */
  if (split == 2)
    eqdev[2] = eqdev[1];

  ctx = clCreateContext(NULL, NUMDEVS, alldevs, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateContext");
  TEST_ASSERT( test_context(ctx, prog_src_all, 1, NUMDEVS, alldevs) == CL_SUCCESS );

  ctx = clCreateContext(NULL, NUMDEVS - 1, alldevs + 1, NULL, NULL, &err);
  CHECK_OPENCL_ERROR_IN("clCreateContext");
  TEST_ASSERT( test_context(ctx, prog_src_two, -1, NUMDEVS - 1, alldevs + 1)
    == CL_SUCCESS );

  /* Don't release the same device twice. clReleaseDevice(NULL) should return
   * an error but not crash. */
  if (split == 2)
    eqdev[2] = NULL;

  for (i = 0; i < NUMDEVS; i++)
    clReleaseDevice (alldevs[i]);

  CHECK_CL_ERROR (clUnloadCompiler ());
  free (dev_pt);

  printf ("OK\n");
  return EXIT_SUCCESS;
}
