#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/opencl.h>
#include "poclu.h"

#ifdef _MSC_VER
#  include "vccompat.hpp"
#endif

#ifndef SRCDIR
#  define SRCDIR="."
#endif

int call_test(const char *name)
{
  size_t global_work_size[1] = { 1 }, local_work_size[1]= { 1 };
  size_t srcdir_length, name_length, filename_size;
  size_t source_size, source_read;
  char const *sources[1];
  char *filename = NULL;
  char *source = NULL;
  FILE *source_file = NULL;
  cl_device_id devices[1];
  cl_context context = NULL;
  cl_command_queue queue = NULL;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  cl_int result;
  int retval = -1;
  
  assert(name);

  /* determine file name of kernel source to load */
  srcdir_length = strlen(SRCDIR);
  name_length = strlen(name);
  filename_size = srcdir_length + name_length + 16;
  filename = (char *)malloc(filename_size + 1);
  if (!filename) {
    puts("out of memory");
    goto error;
  }

  snprintf(filename, filename_size, "%s/%s.cl", SRCDIR, name);

  /* read source code */
  source_file = fopen(filename, "r");
  if (!source_file) {
     puts("source file not found\n");
     goto error;
  }

  fseek(source_file, 0, SEEK_END);
  source_size = ftell(source_file);
  fseek(source_file, 0, SEEK_SET);
  
  source = (char *)malloc(source_size + 1);
  if (!source) {
    puts("out of memory\n");
    goto error;
  }

  source_read = fread(source, 1, source_size, source_file);
  if (source_read != source_size) {
    puts("error reading from file\n");
    goto error;
  }

  source[source_size] = '\0';
  fclose(source_file);
  source_file = NULL;

  /* setup an OpenCL context and command queue using default device */
  context = poclu_create_any_context();
  if (!context) {
    puts("clCreateContextFromType call failed\n");
    goto error;
  }

  result = clGetContextInfo(context, CL_CONTEXT_DEVICES,
      sizeof(cl_device_id), devices, NULL);
  if (result != CL_SUCCESS) {
    puts("clGetContextInfo call failed\n");
    goto error;
  }

  queue = clCreateCommandQueue(context, devices[0], 0, NULL); 
  if (!queue) {
    puts("clCreateCommandQueue call failed\n");
    goto error;
  }

  /* create and build program */
  sources[0] = source;
  program = clCreateProgramWithSource(context, 1, sources, NULL, NULL); 
  if (!program) {
    puts("clCreateProgramWithSource call failed\n");
    goto error;
  }

  result = clBuildProgram(program, 0, NULL, NULL, NULL, NULL); 
  if (result != CL_SUCCESS) {
    puts("clBuildProgram call failed\n");
    goto error;
  }

  /* execute the kernel with give name */
  kernel = clCreateKernel(program, name, NULL); 
  if (!kernel) {
    puts("clCreateKernel call failed\n");
    goto error;
  }

  result = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, 
      global_work_size, local_work_size, 0, NULL, NULL); 
  if (result != CL_SUCCESS) {
    puts("clEnqueueNDRangeKernel call failed\n");
    goto error;
  }

  result = clFinish(queue);
  if (result == CL_SUCCESS)
    retval = 0;

error:

  if (kernel) {
    clReleaseKernel(kernel);
  }
  if (program) {
    clReleaseProgram(program);
  }
  if (queue) {
    clReleaseCommandQueue(queue);
  }
  if (context) {
    clReleaseContext(context);
  }
  if (source_file) {
    fclose(source_file);
  }
  if (source) {
    free(source);
  }
  if (filename) {
    free(filename);
  }

  return retval;
}

const char * const all_tests[] = {
  "test_sizeof",
  "test_as_type",
  "test_convert_type",
  "test_bitselect",
  "test_fabs",
  "test_hadd",
  "test_rotate",
  "test_block",
};
   
const int num_all_tests = (int)(sizeof(all_tests) / sizeof(all_tests[0]));

int main(int argc, char **argv)
{
  int i, retval;

  if (argc < 2) {
    /* Run all tests */
    for (i = 0; i < num_all_tests; ++i) {
      printf("Running test #%d %s...\n", i, all_tests[i]);
      retval = call_test(all_tests[i]);
      if (retval) {
        printf("FAIL\n");
        return 1;
      }
    }
  } else {
    /* Run one test */
    printf("Running test %s...\n", argv[1]);
    retval = call_test(argv[1]);
    if (retval) {
      printf("FAIL\n");
      return 1;
    }
  }
  
  printf("OK\n");
  return 0;
}

