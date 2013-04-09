#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#define MAX_PLATFORMS 32
#define MAX_DEVICES   32
#define MAX_BINARIES  32
char kernel[] = "__kernel void k() {\n    return;\n}";

int
main(void){
  cl_int err;
  cl_platform_id platforms[MAX_PLATFORMS];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES + 1]; // + 1 for duplicate test
  cl_device_id device_id0;
  cl_uint num_devices;
  cl_uint i, j;  
  size_t num_binaries;
  const unsigned char **binaries = NULL;
  size_t *binary_sizes = NULL;
  size_t num_bytes_copied;
  cl_int binary_statuses[MAX_BINARIES];
  cl_int binary_statuses2[MAX_BINARIES];
  cl_program program = NULL;
  cl_program program_with_binary = NULL;
  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &nplatforms);	
  if (err != CL_SUCCESS && !nplatforms)
    return EXIT_FAILURE;
  
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
		       devices, &num_devices);  
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;

  cl_context context = clCreateContext(NULL, num_devices, devices, NULL, NULL, &err);
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;

  size_t kernel_size = strlen(kernel);
  char* kernel_buffer = kernel;

  program = clCreateProgramWithSource(context, 1, (const char**)&kernel_buffer, 
				      &kernel_size, &err);
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;

  err = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;
  
  err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, 0, 0, &num_binaries);
  if (err != CL_SUCCESS)
      goto FREE_AND_EXIT;

  num_binaries = num_binaries/sizeof(size_t);
  binary_sizes = (size_t*)malloc(num_binaries * sizeof(size_t)); 
  binaries = (const unsigned char**)calloc(num_binaries, sizeof(unsigned char*));

  err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, 
			 num_binaries*sizeof(size_t), binary_sizes , 
			 &num_bytes_copied);
  if (err != CL_SUCCESS)
      goto FREE_AND_EXIT;
  
  for (i = 0; i < num_binaries; ++i) 
    binaries[i] = (const unsigned char*) malloc(binary_sizes[i] * 
						sizeof(const unsigned char));

  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, 
			 num_binaries*sizeof(char*), binaries, &num_bytes_copied);
  if (err != CL_SUCCESS)
      goto FREE_AND_EXIT;
  
  cl_int num = num_binaries < num_devices ? num_binaries : num_devices;
  if (num == 0)
    {
      err = !CL_SUCCESS;
      goto FREE_AND_EXIT;
    }
  
  program_with_binary = clCreateProgramWithBinary(context, num, devices, binary_sizes, 
						  binaries, binary_statuses, &err);
  if (err != CL_SUCCESS)
    goto FREE_AND_EXIT;

  clReleaseProgram(program_with_binary);
  for (i = 0; i < num; i++)
    {
      if (binary_statuses[i] != CL_SUCCESS)
	{
	  err = !CL_SUCCESS;
	  goto FREE_AND_EXIT;
	}
    }
    
  // negative test1: invalid device
  device_id0 = devices[0];
  devices[0] = NULL; // invalid device
  program_with_binary = clCreateProgramWithBinary(context, num, devices, binary_sizes, 
						  binaries, binary_statuses, &err);

  if (err != CL_INVALID_DEVICE || program_with_binary != NULL)
    {
      err = !CL_SUCCESS;
      goto FREE_AND_EXIT;
    }
  err = CL_SUCCESS;

  devices[0] = device_id0;
  for (i = 0; i < num_binaries; ++i) free((void*)binaries[i]);
  free(binary_sizes);
  free(binaries);
  
  // negative test2: duplicate device
  num_binaries = 2;
  devices[1] = devices[0]; // duplicate
  
  binary_sizes = (size_t*)malloc(num_binaries * sizeof(size_t)); 
  binaries = (const unsigned char**)calloc(num_binaries, sizeof(unsigned char*));
  
  err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, 1*sizeof(size_t), 
			 binary_sizes , &num_bytes_copied);
  if (err != CL_SUCCESS)
    goto FREE_AND_EXIT;
  
  binary_sizes[1] = binary_sizes[0];
  
  binaries[0] = (const unsigned char*) malloc(binary_sizes[0] * 
					      sizeof(const unsigned char));
  binaries[1] = (const unsigned char*) malloc(binary_sizes[1] * 
					      sizeof(const unsigned char));
  
  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, 1 * sizeof(char*), 
			 binaries, &num_bytes_copied);
  if (err != CL_SUCCESS)
    goto FREE_AND_EXIT;
  
  memcpy((void*)binaries[1], (void*)binaries[0], binary_sizes[0]);      
  program_with_binary = clCreateProgramWithBinary(context, 2, devices, binary_sizes, 
						  binaries, binary_statuses2, &err);
  if (err != CL_INVALID_DEVICE || program_with_binary != NULL)
    {
      err = !CL_SUCCESS;
      goto FREE_AND_EXIT;
    }
  err = CL_SUCCESS;

 FREE_AND_EXIT:  
  // Free resources
  for (i = 0; i < num_binaries; ++i) 
    if (binaries) 
      if(binaries[i]) 
	free((void*)binaries[i]);

  if (binary_sizes) 
    free(binary_sizes);
  if (binaries) 
    free(binaries);
  if (program)
    clReleaseProgram(program);  
  if (program_with_binary)
    clReleaseProgram(program_with_binary);
  return err == CL_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
