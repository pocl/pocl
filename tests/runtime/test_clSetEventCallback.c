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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <pthread.h>

#include "poclu.h"

int submit = 0;
int running = 0;
int complete = 0;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
#define LOCK pthread_mutex_lock(&mutex)
#define UNLOCK pthread_mutex_unlock(&mutex)


void callback_function(cl_event event, 
                       cl_int   event_command_exec_status, 
                       void     *user_data)
{
  printf("%s ", (const char *)user_data);
  LOCK;
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
  UNLOCK;
  fflush(stdout);
  return;
}

/* TODO this test relied on output of printf() from kernel
 * appearing in specific order WRT event state callbacks,
 * which is likely UB. OpenCL states printf()
 * should happen "at clFinish time" but AFAICT
 * does not state, if it should happen before
 * or after the event-completed callback call.
 * -> test disabled for now. */
char kernelASourceCode[] = 
"kernel \n"
"void test_kernel(constant char* input) {\n"
"    if (input[0] == 'X') printf(\"match\");\n"
"}\n";

int main()
{
  size_t global_work_size[1] = { 1 }, local_work_size[1]= { 1 };
  cl_int err;
  cl_program program = NULL;
  cl_kernel kernel = NULL;
  char input[] = "kernel in execution\n";
  char *user_data = "Callback function: event status:";
  cl_mem inputBuffer = NULL;
  /* events */
  cl_event an_event = NULL;
  int i;

  cl_context context;
  cl_command_queue queue;
  cl_device_id device;

  CHECK_CL_ERROR(poclu_get_any_device(&context, &device, &queue));
  TEST_ASSERT( context );
  TEST_ASSERT( device );
  TEST_ASSERT( queue );

  CHECK_CL_ERROR(clGetContextInfo(context, CL_CONTEXT_DEVICES,
                         sizeof(cl_device_id), &device, NULL));

  inputBuffer = clCreateBuffer(context, 
                               CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                               strlen (input)+1, (void *) input, &err);
  CHECK_OPENCL_ERROR_IN("clCreateBuffer");

  size_t kernel_size = strlen (kernelASourceCode);
  char* kernel_buffer = kernelASourceCode;
  
  program = clCreateProgramWithSource (context, 1, 
                                       (const char**)&kernel_buffer, 
                                       &kernel_size, &err);
  CHECK_OPENCL_ERROR_IN("clCreateProgramWithSource");

  CHECK_CL_ERROR(clBuildProgram (program, 1, &device, NULL, NULL, NULL));

  kernel = clCreateKernel (program, "test_kernel", &err);
  CHECK_OPENCL_ERROR_IN("clCreateKernel");
  
  CHECK_CL_ERROR(clSetKernelArg (kernel, 0, sizeof (cl_mem), &inputBuffer));
 
  /* launch kernel*/
  CHECK_CL_ERROR(clEnqueueNDRangeKernel (queue, kernel, 1, NULL, global_work_size,
                                local_work_size, 0, NULL, &an_event));

  CHECK_CL_ERROR(clSetEventCallback(an_event, CL_SUBMITTED, callback_function, user_data));
  CHECK_CL_ERROR(clSetEventCallback(an_event, CL_RUNNING, callback_function, user_data));
  CHECK_CL_ERROR(clSetEventCallback(an_event, CL_COMPLETE, callback_function, user_data));

  CHECK_CL_ERROR(clFinish(queue));

  i = 0;
  LOCK;
  int all_done = submit + running + complete;
  UNLOCK;
  while (all_done != 3)
    {
      sleep(1);
      ++i;
      if (i >= 10)
        {
          puts("Callback functions were not called in 10s -> assume FAIL\n");
          return EXIT_FAILURE;
        }
      LOCK;
      all_done = submit + running + complete;
      UNLOCK;
    }

  CHECK_CL_ERROR (clReleaseEvent (an_event));

  CHECK_CL_ERROR (clReleaseCommandQueue (queue));
  CHECK_CL_ERROR (clReleaseMemObject (inputBuffer));

  CHECK_CL_ERROR (clReleaseKernel (kernel));
  CHECK_CL_ERROR (clReleaseProgram (program));

  CHECK_CL_ERROR (clReleaseContext (context));
  CHECK_CL_ERROR (clUnloadCompiler ());

  printf ("OK\n");
  return EXIT_SUCCESS;
}
