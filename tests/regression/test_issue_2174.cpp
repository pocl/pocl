// pocl-vector-align-repro.c
//
// Reproducer for a pocl OpenCL C compiler bug observed on aarch64
// (Apple M3 / Linux): `__alignof__` of OpenCL built-in vector types is
// capped at 16 bytes inside the kernel.
//
// The reference for the expected alignment is taken from the host
// `cl_<type><n>` types in <CL/cl_platform.h>, evaluated at host compile
// time with `__alignof__`. The kernel calls `__alignof__` on the
// matching device-side `<type><n>` and writes the result to a buffer.
// Any divergence between host and device for the same logical type is
// an ABI mismatch.
//
// Build:  cc repro.c -lOpenCL -o repro
// Run:    ./repro
//   (set OCL_ICD_VENDORS to point at the pocl ICD if needed)

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *KERNEL_SRC = R"(
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    __kernel void report_alignments(__global uint *out) {
      out[ 0] = (uint)__alignof__(char2);
      out[ 1] = (uint)__alignof__(char3);
      out[ 2] = (uint)__alignof__(char4);
      out[ 3] = (uint)__alignof__(char8);
      out[ 4] = (uint)__alignof__(char16);
      out[ 5] = (uint)__alignof__(uchar2);
      out[ 6] = (uint)__alignof__(uchar3);
      out[ 7] = (uint)__alignof__(uchar4);
      out[ 8] = (uint)__alignof__(uchar8);
      out[ 9] = (uint)__alignof__(uchar16);
      out[10] = (uint)__alignof__(short2);
      out[11] = (uint)__alignof__(short3);
      out[12] = (uint)__alignof__(short4);
      out[13] = (uint)__alignof__(short8);
      out[14] = (uint)__alignof__(short16);
      out[15] = (uint)__alignof__(ushort2);
      out[16] = (uint)__alignof__(ushort3);
      out[17] = (uint)__alignof__(ushort4);
      out[18] = (uint)__alignof__(ushort8);
      out[19] = (uint)__alignof__(ushort16);
      out[20] = (uint)__alignof__(int2);
      out[21] = (uint)__alignof__(int3);
      out[22] = (uint)__alignof__(int4);
      out[23] = (uint)__alignof__(int8);
      out[24] = (uint)__alignof__(int16);
      out[25] = (uint)__alignof__(uint2);
      out[26] = (uint)__alignof__(uint3);
      out[27] = (uint)__alignof__(uint4);
      out[28] = (uint)__alignof__(uint8);
      out[29] = (uint)__alignof__(uint16);
      out[30] = (uint)__alignof__(long2);
      out[31] = (uint)__alignof__(long3);
      out[32] = (uint)__alignof__(long4);
      out[33] = (uint)__alignof__(long8);
      out[34] = (uint)__alignof__(long16);
      out[35] = (uint)__alignof__(ulong2);
      out[36] = (uint)__alignof__(ulong3);
      out[37] = (uint)__alignof__(ulong4);
      out[38] = (uint)__alignof__(ulong8);
      out[39] = (uint)__alignof__(ulong16);
      out[40] = (uint)__alignof__(float2);
      out[41] = (uint)__alignof__(float3);
      out[42] = (uint)__alignof__(float4);
      out[43] = (uint)__alignof__(float8);
      out[44] = (uint)__alignof__(float16);
      out[45] = (uint)__alignof__(double2);
      out[46] = (uint)__alignof__(double3);
      out[47] = (uint)__alignof__(double4);
      out[48] = (uint)__alignof__(double8);
      out[49] = (uint)__alignof__(double16);
    }
)";

struct check {
  const char *name;
  cl_uint expected_align; // taken from <CL/cl_platform.h> at host compile time
};

// Each row pairs a kernel-side type name with the host-side
// __alignof__(cl_<type>). The host header is the spec authority for
// the OpenCL ABI on this platform; if the kernel disagrees, the
// kernel's compiler is wrong.
#define ENTRY(t) {#t, (cl_uint) __alignof__(cl_##t)}

static const struct check CHECKS[] = {
    ENTRY(char2),   ENTRY(char3),    ENTRY(char4),   ENTRY(char8),
    ENTRY(char16),  ENTRY(uchar2),   ENTRY(uchar3),  ENTRY(uchar4),
    ENTRY(uchar8),  ENTRY(uchar16),  ENTRY(short2),  ENTRY(short3),
    ENTRY(short4),  ENTRY(short8),   ENTRY(short16), ENTRY(ushort2),
    ENTRY(ushort3), ENTRY(ushort4),  ENTRY(ushort8), ENTRY(ushort16),
    ENTRY(int2),    ENTRY(int3),     ENTRY(int4),    ENTRY(int8),
    ENTRY(int16),   ENTRY(uint2),    ENTRY(uint3),   ENTRY(uint4),
    ENTRY(uint8),   ENTRY(uint16),   ENTRY(long2),   ENTRY(long3),
    ENTRY(long4),   ENTRY(long8),    ENTRY(long16),  ENTRY(ulong2),
    ENTRY(ulong3),  ENTRY(ulong4),   ENTRY(ulong8),  ENTRY(ulong16),
    ENTRY(float2),  ENTRY(float3),   ENTRY(float4),  ENTRY(float8),
    ENTRY(float16), ENTRY(double2),  ENTRY(double3), ENTRY(double4),
    ENTRY(double8), ENTRY(double16),
};

#define NTYPES (sizeof(CHECKS) / sizeof(CHECKS[0]))

#define CL_CHECK(call)                                                         \
  do {                                                                         \
    cl_int _err = (call);                                                      \
    if (_err != CL_SUCCESS) {                                                  \
      fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #call,     \
              _err);                                                           \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

int main(void) {
  cl_platform_id platform;
  cl_uint num_platforms;
  CL_CHECK(clGetPlatformIDs(1, &platform, &num_platforms));

  cl_device_id device;
  CL_CHECK(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL));

  char platform_name[256], driver_version[256], device_name[256];
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name),
                    platform_name, NULL);
  clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(driver_version),
                    driver_version, NULL);
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name,
                  NULL);
  printf("Platform: %s\n", platform_name);
  printf("Version:  %s\n", driver_version);
  printf("Device:   %s\n\n", device_name);

  cl_int err;
  cl_context ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CL_CHECK(err);
  cl_command_queue queue =
      clCreateCommandQueueWithProperties(ctx, device, NULL, &err);
  CL_CHECK(err);

  cl_program program =
      clCreateProgramWithSource(ctx, 1, &KERNEL_SRC, NULL, &err);
  CL_CHECK(err);
  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    char *log = new char[log_size];
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log,
                          NULL);
    fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
    delete[] log;
    exit(1);
  }

  cl_kernel kernel = clCreateKernel(program, "report_alignments", &err);
  CL_CHECK(err);

  cl_mem buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, NTYPES * sizeof(cl_uint),
                              NULL, &err);
  CL_CHECK(err);
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf));

  size_t global = 1;
  CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, NULL, 0,
                                  NULL, NULL));
  CL_CHECK(clFinish(queue));

  cl_uint result[NTYPES];
  CL_CHECK(clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, sizeof(result), result,
                               0, NULL, NULL));

  printf("%-10s | host (cl_*) | kernel | result\n", "type");
  printf("-----------+-------------+--------+--------\n");
  int failures = 0;
  for (size_t i = 0; i < NTYPES; ++i) {
    int ok = (result[i] == CHECKS[i].expected_align);
    if (!ok)
      ++failures;
    printf("%-10s |       %4u  |   %4u | %s\n", CHECKS[i].name,
           CHECKS[i].expected_align, result[i], ok ? "PASS" : "FAIL");
  }
  if (failures)
    printf("FAILED: \n%zu/%zu \n", NTYPES - failures, NTYPES);
  else
    printf("PASSED: \n%zu/%zu \n", NTYPES - failures, NTYPES);

  clReleaseMemObject(buf);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(ctx);

  return failures > 0 ? 1 : 0;
}
