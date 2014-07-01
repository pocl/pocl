/* Tests clFinish 

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
  cl_kernel kernelA = NULL;
  cl_kernel kernelB = NULL;
  cl_kernel kernelC= NULL;
  char inputA[] = "A";
  char inputB[] = "B";
  char inputC[] = "C";
  cl_mem inputBufferA = NULL;
  cl_mem inputBufferB = NULL;
  cl_mem inputBufferC = NULL;
  /* command queues */
  cl_command_queue queueA = NULL;
  cl_command_queue queueB = NULL;
  cl_command_queue queueC = NULL;
  /* events */
  cl_event eventA1 = NULL;
  cl_event eventB2 = NULL;
  cl_event eventA3 = NULL;
  cl_event eventB4 = NULL;
  /* event wait lists */
  cl_event B2_wait_list[1];
  cl_event A3_wait_list[1];
  cl_event B4_wait_list[1];
  cl_event C5_wait_list[2];

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

  queueA = clCreateCommandQueue(context, devices[0], 0, NULL); 
  if (!queueA) 
    {
      puts("clCreateCommandQueue call failed\n");
      goto error;
    }

  queueB = clCreateCommandQueue(context, devices[0], 0, NULL); 
  if (!queueB) 
    {
      puts("clCreateCommandQueue call failed\n");
      goto error;
    }

  queueC = clCreateCommandQueue(context, devices[0], 0, NULL); 
  if (!queueB) 
    {
      puts("clCreateCommandQueue call failed\n");
      goto error;
    }

  inputBufferA = clCreateBuffer(context, 
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                strlen (inputB)+1, (void *) inputA, &err);
  if (inputBufferA == NULL)
    {
      printf("clCreateBuffer call failed err = %d\n", err);
      goto error;
    }

  inputBufferB = clCreateBuffer(context, 
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                strlen (inputA)+1, (void *) inputB, &err);
  if (inputBufferB == NULL)
    {
      printf("clCreateBuffer call failed err = %d\n", err);
      goto error;
    }
  
  inputBufferC = clCreateBuffer(context, 
                                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                strlen (inputA)+1, (void *) inputC, &err);
  if (inputBufferC == NULL)
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

  kernelA = clCreateKernel (program, "test_kernel", NULL); 
  if (!kernelA) 
    {
      puts("clCreateKernel call failed\n");
      goto error;
    }

  kernelB = clCreateKernel (program, "test_kernel", NULL); 
  if (!kernelB) 
    {
      puts("clCreateKernel call failed\n");
      goto error;
    }
  
  kernelC = clCreateKernel (program, "test_kernel", NULL); 
  if (!kernelC) 
    {
      puts("clCreateKernel call failed\n");
      goto error;
    }
  
  err = clSetKernelArg (kernelA, 0, sizeof (cl_mem), &inputBufferA);
  if (err)
    {
      puts("clSetKernelArg failed\n");
      goto error;
    }
 
  err = clSetKernelArg (kernelB, 0, sizeof (cl_mem), &inputBufferB);
  if (err)
    {
      puts("clSetKernelArg failed\n");
      goto error;
    }
  
  err = clSetKernelArg (kernelC, 0, sizeof (cl_mem), &inputBufferC);
  if (err)
    {
      puts("clSetKernelArg failed\n");
      goto error;
    }

    

  /* first enqueue A1*/
  err = clEnqueueNDRangeKernel (queueA, kernelA, 1, NULL, global_work_size, 
                                local_work_size, 0, NULL, &eventA1); 
  if (err != CL_SUCCESS) 
    {
      puts("clEnqueueNDRangeKernel call failed\n");
      goto error;
    }

  /* enqueue B2 */
  B2_wait_list[0] = eventA1;
  err = clEnqueueNDRangeKernel (queueB, kernelB, 1, NULL, global_work_size, 
                                local_work_size, 1, B2_wait_list, &eventB2); 
  if (err != CL_SUCCESS) 
    {
      puts("clEnqueueNDRangeKernel call failed\n");
      goto error;
    }

  /* enqueue A3 */
  A3_wait_list[0] = eventB2;
  err = clEnqueueNDRangeKernel (queueA, kernelA, 1, NULL, global_work_size, 
                                local_work_size, 1, A3_wait_list, &eventA3); 
  if (err != CL_SUCCESS) 
    {
      puts("clEnqueueNDRangeKernel call failed\n");
      goto error;
    }

  /* enqueue B4 */
  B4_wait_list[0] = eventA3;
  err = clEnqueueNDRangeKernel (queueB, kernelB, 1, NULL, global_work_size, 
                                local_work_size, 1, B4_wait_list, &eventB4); 
  if (err != CL_SUCCESS) 
    {
      puts("clEnqueueNDRangeKernel call failed\n");
      goto error;
    }

  /* enqueue C5 */
  C5_wait_list[0] = eventA3;
  C5_wait_list[1] = eventB4;
  err = clEnqueueNDRangeKernel (queueC, kernelC, 1, NULL, global_work_size, 
                                local_work_size, 2, C5_wait_list, NULL); 
  if (err != CL_SUCCESS) 
    {
      puts("clEnqueueNDRangeKernel call failed\n");
      goto error;
    }


  clFinish(queueC);
  printf("\n");
  return EXIT_SUCCESS;

 error:
  return EXIT_FAILURE;

}
