#include <stdlib.h>

#include "pocl_opencl.h"

void 
delete_memobjs(cl_mem *memobjs, int n) 
{ 
  int i; 
  for (i=0; i<n; i++) 
    clReleaseMemObject(memobjs[i]); 
} 
 
int 
exec_trig_kernel(const char *program_source, 
                 int n, void *srcA, void *dst) 
{ 
  cl_context  context; 
  cl_command_queue cmd_queue; 
  cl_device_id  *devices; 
  cl_program  program; 
  cl_kernel  kernel; 
  cl_mem       memobjs[2]; 
  size_t       global_work_size[1]; 
  size_t       local_work_size[1]; 
  size_t       cb; 
  cl_int       err; 

  float c = 7.3f; // a scalar number to test non-pointer args
 
  // create the OpenCL context on a GPU device 
  context = poclu_create_any_context();
  if (context == (cl_context)0) 
    return -1; 
 
  // get the list of GPU devices associated with context 
  clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb); 
  devices = (cl_device_id *) malloc(cb);
  clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, devices, NULL); 
 
  // create a command-queue 
  cmd_queue = clCreateCommandQueue(context, devices[0], 0, NULL); 
  if (cmd_queue == (cl_command_queue)0) 
    { 
      clReleaseContext(context); 
      free(devices); 
      return -1; 
    } 
  free(devices); 
 
  // allocate the buffer memory objects 
  memobjs[0] = clCreateBuffer(context, 
                              CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                              sizeof(cl_float4) * n, srcA, NULL); 
  if (memobjs[0] == (cl_mem)0) 
    { 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  memobjs[1] = clCreateBuffer(context, 
			      CL_MEM_READ_WRITE, 
			      sizeof(cl_float4) * n, NULL, NULL); 
  if (memobjs[1] == (cl_mem)0) 
    { 
      delete_memobjs(memobjs, 1); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // create the program 
  program = clCreateProgramWithSource(context, 
				      1, (const char**)&program_source, NULL, NULL); 
  if (program == (cl_program)0) 
    { 
      delete_memobjs(memobjs, 2); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // build the program 
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL); 
  if (err != CL_SUCCESS) 
    { 
      delete_memobjs(memobjs, 2); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // create the kernel 
  kernel = clCreateKernel(program, "trig", NULL); 
  if (kernel == (cl_kernel)0) 
    { 
      delete_memobjs(memobjs, 2); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // set the args values 
  err = clSetKernelArg(kernel,  0,  
		       sizeof(cl_mem), (void *) &memobjs[0]); 
  err |= clSetKernelArg(kernel, 1,
			sizeof(cl_mem), (void *) &memobjs[1]); 
  err |= clSetKernelArg(kernel, 2,
			sizeof(float), (void *) &c); 
 
  if (err != CL_SUCCESS) 
    { 
      delete_memobjs(memobjs, 2); 
      clReleaseKernel(kernel); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // set work-item dimensions 
  global_work_size[0] = n; 
  local_work_size[0]= 2; 
 
  // execute kernel 
  err = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, 
			       global_work_size, local_work_size,  
			       0, NULL, NULL); 
  if (err != CL_SUCCESS) 
    { 
      delete_memobjs(memobjs, 2); 
      clReleaseKernel(kernel); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // read output image 
  err = clEnqueueReadBuffer(cmd_queue, memobjs[1], CL_TRUE, 
			    0, n * sizeof(cl_float4), dst, 
			    0, NULL, NULL); 
  if (err != CL_SUCCESS) 
    { 
      delete_memobjs(memobjs, 2); 
      clReleaseKernel(kernel); 
      clReleaseProgram(program); 
      clReleaseCommandQueue(cmd_queue); 
      clReleaseContext(context); 
      return -1; 
    } 
 
  // release kernel, program, and memory objects 
  delete_memobjs(memobjs, 2); 
  clReleaseKernel(kernel); 
  clReleaseProgram(program); 
  clReleaseCommandQueue(cmd_queue); 
  clReleaseContext(context); 
  return 0; // success... 
}
