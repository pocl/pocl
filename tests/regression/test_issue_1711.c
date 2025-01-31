// Copyright (c) 2024 PoCL developers
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/*
  This test was contributed by maleadt in issue 1711.
  It tests that a kernel generated from Julia OpenCL is supported correctly, where 
  the kernel contains an "error handling exit" which is reduced to "unreachable".
  This triggered an endless loop with CBS.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

typedef struct {
    cl_long* ptr;
    size_t maxsize;
    struct {
        size_t dim1;
        size_t dim2;
    } dims;
    size_t len;
} DeviceArray;

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("OpenCL Error %d at %s:%d\n", err, __FILE__, __LINE__); \
        exit(1); \
    }

int
device_supports_il (cl_device_id device, const char *il)
{
  size_t param_size = 0;
  cl_int err
    = clGetDeviceInfo (device, CL_DEVICE_IL_VERSION, 0, NULL, &param_size);
  CHECK_ERROR (err);
  char *ils = malloc (param_size);
  err = clGetDeviceInfo (device, CL_DEVICE_IL_VERSION, param_size, ils, NULL);
  CHECK_ERROR (err);
  int has_il = strstr (ils, il) != NULL;
  free (ils);
  return has_il;
}

cl_platform_id get_platform(int index) {
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    CHECK_ERROR(err);

    if (index >= num_platforms) {
        printf("Platform index %d is out of range (max %d) at %s:%d\n",
               index, num_platforms - 1, __FILE__, __LINE__);
        exit(1);
    }

    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    CHECK_ERROR(err);

    cl_platform_id platform = platforms[index];
    free(platforms);
    return platform;
}

int main(int argc, char** argv) {
    // Use first platform by default or take from command line
    int platform_index = 0;
    if (argc > 1) {
        platform_index = atoi(argv[1]);
    }

    cl_int err;
    cl_platform_id platform = get_platform(platform_index);
    cl_device_id device;

    // Get platform name
    char platform_name[128];
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    CHECK_ERROR(err);

    // Get device
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    CHECK_ERROR(err);

    // Get device name
    char device_name[128];
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    CHECK_ERROR(err);

    if (!device_supports_il (device, "SPIR-V_1.4"))
      {
        printf ("SKIP: The test requires support for SPIR-V 1.4\n");
        exit (77);
      }

    // Rest of the code remains the same...
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    cl_queue_properties props[] = {0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    CHECK_ERROR(err);

    // Create DeviceArray structure
    DeviceArray A;
    A.maxsize = 64;
    A.dims.dim1 = 8;
    A.dims.dim2 = 1;
    A.len = 8;

    // Allocate SVM memory for the data array
    A.ptr = (cl_long*)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeof(cl_long) * A.len, 0);
    if (A.ptr == NULL) {
        printf("SVM allocation failed at %s:%d\n", __FILE__, __LINE__);
        return 1;
    }

    // Load SPIR-V binary
    FILE* f = fopen(SRCDIR "/test_issue_1711.spv", "rb");
    if (!f) {
        printf("Failed to open test_issue_1711.spv at %s:%d\n", __FILE__, __LINE__);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    unsigned char* binary = (unsigned char*)malloc(size);
    fread(binary, 1, size, f);
    fclose(f);

    cl_program program = clCreateProgramWithIL(context, binary, size, &err);
    CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build error at %s:%d:\n%s\n", __FILE__, __LINE__, log);
        free(log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "_Z7forloop13CLDeviceArrayI5Int64Li2ELi1EE", &err);
    CHECK_ERROR(err);

    // Inform runtime about SVM pointers
    void* svm_ptrs[] = { A.ptr };
    err = clSetKernelExecInfo(kernel, CL_KERNEL_EXEC_INFO_SVM_PTRS, sizeof(void*), svm_ptrs);
    CHECK_ERROR(err);

    // Set kernel argument
    err = clSetKernelArg(kernel, 0, sizeof(DeviceArray), &A);
    CHECK_ERROR(err);

    // Execute kernel
    size_t global_size = 8;
    size_t local_size = 8;

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clFinish(queue);
    CHECK_ERROR(err);
    printf("OK\n");

    // Cleanup
    clSVMFree(context, A.ptr);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(binary);

    return 0;
}
