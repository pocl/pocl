#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <CL/opencl.h>


#ifndef SRCDIR
#  define SRCDIR "."
#endif



int call_test(char const *const name)
{
  /* read source code */
  char filename[1000];
  snprintf(filename, sizeof filename, "%s/%s.cl", SRCDIR, name);
  FILE *const source_file = fopen(filename, "r");
  assert(source_file != NULL && "source file not found");
  
  fseek(source_file, 0, SEEK_END);
  long const source_size = ftell(source_file);
  fseek(source_file, 0, SEEK_SET);
  
  char source[source_size + 1];
  fread(source, source_size, 1, source_file);
  source[source_size] = '\0';
  
  fclose(source_file);
  
  /* call OpenCL program */
  cl_context const context =
    clCreateContextFromType(NULL, CL_DEVICE_TYPE_ALL, NULL, NULL, NULL);
  if (context == 0) return -1;
  
  size_t ndevices;
  clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &ndevices);
  ndevices /= sizeof(cl_device_id);
  cl_device_id devices[ndevices];
  clGetContextInfo(context, CL_CONTEXT_DEVICES,
                   ndevices*sizeof(cl_device_id), devices, NULL); 
  
  cl_command_queue const cmd_queue =
    clCreateCommandQueue(context, devices[0], 0, NULL); 
  if (cmd_queue == 0) return -1;
  
  char const *sources[] = {source};
  cl_program const program =
    clCreateProgramWithSource(context, 1, sources, NULL, NULL); 
  if (program == 0) return -1;
  
  int ierr;
  ierr = clBuildProgram(program, 0, NULL, NULL, NULL, NULL); 
  if (ierr != CL_SUCCESS) return -1;
  
  cl_kernel const kernel = clCreateKernel(program, name, NULL); 
  if (kernel == 0) return -1;
  
  size_t global_work_size[] = {1};
  size_t local_work_size[]= {1};
  ierr = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, 
                                global_work_size, local_work_size,  
                                0, NULL, NULL); 
  if (ierr != CL_SUCCESS) return -1;
  
  return 0;
}



int
main(int argc, char **argv)
{
  if (argc < 2) {
    
    /* Run all tests */
    char const *const tests[] = {
      "test_as_type",
      "test_bitselect",
      "test_fabs",
      "test_hadd",
      "test_rotate",
    };
    int const ntests = sizeof(tests)/sizeof(*tests);
    for (int i=0; i<ntests; ++i) {
      printf("Running test #%d %s...\n", i, tests[i]);
      int ierr;
      ierr = call_test(tests[i]);
      if (ierr) {
        printf("FAIL\n");
        return 1;
      }
    }
    
  } else {
    
    /* Run one test */
    printf("Running test %s...\n", argv[1]);
    int ierr;
    ierr = call_test(argv[1]);
    if (ierr) {
      printf("FAIL\n");
      return 1;
    }
    
  }
  
  printf("OK\n");
  return 0;
}

