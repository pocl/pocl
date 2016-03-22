/* Tests clSetEventCallback() 

   Copyright (c) 2013 Ville Korhonen / Tampere University of Technology
   
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
#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

volatile int submit = 0;
volatile int running = 0;
volatile int complete = 0;

void callback_function(cl_event event, 
                       cl_int   event_command_exec_status, 
                       void     *user_data)
{
  printf("%s ", (const char *)user_data);
  if(event_command_exec_status == CL_SUBMITTED)
    {
      printf("CL_SUBMITTED\n");
      submit = 1;
    }
  if(event_command_exec_status == CL_RUNNING)
    {
      printf("CL_RUNNING\n");
      running = 1;
    }

  if(event_command_exec_status == CL_COMPLETE)
    {
      printf("CL_COMPLETE\n");
      complete = 1;
    }
  return;
}

char kernelASourceCode[] = 
"kernel \n"
"void test_kernel(constant char* input) {\n"
"    printf(\"%s\", input);\n"
"}\n";

int main()
{
  size_t global_work_size[1] = { 1 }, local_work_size[1]= { 1 };
  cl_int err;
  cl_platform_id platforms[1];
  cl_uint nplatforms;
  cl_device_id devices[1]; // + 1 for duplicate test
  cl_uint num_devices;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  char input[] = "kernel in execution\n";
  char *user_data = "Callback function: event status:";
  cl_mem inputBuffer = NULL;
  /* command queues */
  cl_command_queue queue = NULL;
  /* events */
  cl_event an_event = NULL;
  int i;
  
  err = clGetPlatformIDs(1, platforms, &nplatforms);	
  if (err != CL_SUCCESS && !nplatforms)
    return EXIT_FAILURE;
  
  err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1,
                       devices, &num_devices);  
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;

  cl_context context = clCreateContext(NULL, num_devices, devices, NULL, 
                                       NULL, &err);
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;

  err = clGetContextInfo(context, CL_CONTEXT_DEVICES,
                         sizeof(cl_device_id), devices, NULL);
  if (err != CL_SUCCESS) 
    {
      puts("clGetContextInfo call failed\n");
      goto error;
    }

  queue = clCreateCommandQueue(context, devices[0], 0, NULL); 
  if (!queue) 
    {
      puts("clCreateCommandQueue call failed\n");
      goto error;
    }

  inputBuffer = clCreateBuffer(context, 
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                               strlen (input)+1, (void *) input, &err);
  if (inputBuffer == NULL)
    {
      printf("clCreateBuffer call failed err = %d\n", err);
      goto error;
    }
  
  size_t kernel_size = strlen (kernelASourceCode);
  char* kernel_buffer = kernelASourceCode;
  
  program = clCreateProgramWithSource (context, 1, 
                                       (const char**)&kernel_buffer, 
                                       &kernel_size, &err);
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;

  err = clBuildProgram (program, num_devices, devices, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;

  kernel = clCreateKernel (program, "test_kernel", NULL); 
  if (!kernel) 
    {
      puts("clCreateKernel call failed\n");
      goto error;
    }
  
  err = clSetKernelArg (kernel, 0, sizeof (cl_mem), &inputBuffer);
  if (err)
    {
      puts("clSetKernelArg failed\n");
      goto error;
    }
 
  /* launch kernel*/
  err = clEnqueueNDRangeKernel (queue, kernel, 1, NULL, global_work_size, 
                                local_work_size, 0, NULL, &an_event); 
  if (err != CL_SUCCESS) 
    {
      puts("clEnqueueNDRangeKernel call failed\n");
      goto error;
    }
  clSetEventCallback(an_event, CL_SUBMITTED, callback_function, user_data);
  clSetEventCallback(an_event, CL_RUNNING, callback_function, user_data);
  clSetEventCallback(an_event, CL_COMPLETE, callback_function, user_data);

  clFinish(queue);

  i = 0;
  while (!submit || !running || !complete)
    {
      sleep(1);
      ++i;
      if (i >= 10)
        {
          puts("Callback functions were not called in 10s -> assume FAIL\n");
          return EXIT_FAILURE;
        }
    }
  return EXIT_SUCCESS;

 error:
  return EXIT_FAILURE;

}
