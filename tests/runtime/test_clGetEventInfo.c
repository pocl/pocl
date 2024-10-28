#include <stdio.h>
#include <stdlib.h>

#include "poclu.h"

#define MAX_PLATFORMS 32
#define MAX_DEVICES   32
#define BUF_SIZE 1024
int
main(void)
{
  cl_int err;
  cl_platform_id platforms[MAX_PLATFORMS];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES];
  cl_uint ndevices;
  cl_uint i, j;
  cl_context context;
  cl_command_queue queue;
  cl_mem buf;
  cl_event buf_event;
  cl_command_queue event_command_queue;

  CHECK_CL_ERROR(clGetPlatformIDs(MAX_PLATFORMS, platforms, &nplatforms));

  for (i = 0; i < nplatforms; i++)
    {
      CHECK_CL_ERROR(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
                           devices, &ndevices));
      
      for (j = 0; j < ndevices; j++)
        {
          context = clCreateContext (NULL, 1, &devices[j], NULL, NULL, &err);
          queue = clCreateCommandQueue (context, devices[j], 0, &err);

          cl_int host_buf[BUF_SIZE];

          buf = clCreateBuffer (context, CL_MEM_READ_WRITE,
                                sizeof (cl_int) * BUF_SIZE, NULL, &err);
          CHECK_CL_ERROR (clEnqueueReadBuffer (
              queue, buf, CL_TRUE, 0, sizeof (cl_int) * BUF_SIZE, &host_buf, 0,
              NULL, &buf_event));
          CHECK_CL_ERROR(clFinish(queue));
          size_t param_val_size_ret;
          CHECK_CL_ERROR(clGetEventInfo(buf_event, CL_EVENT_COMMAND_QUEUE, sizeof(cl_command_queue), &event_command_queue, &param_val_size_ret));
          TEST_ASSERT(param_val_size_ret == sizeof(cl_command_queue));
          TEST_ASSERT(event_command_queue == queue);
          
          cl_command_type command_type;
          CHECK_CL_ERROR(clGetEventInfo(buf_event, CL_EVENT_COMMAND_TYPE, sizeof(cl_command_type), &command_type, &param_val_size_ret));
          TEST_ASSERT(param_val_size_ret == sizeof(cl_command_type));
          TEST_ASSERT(command_type == CL_COMMAND_READ_BUFFER);

          cl_int execution_status;
          CHECK_CL_ERROR(clGetEventInfo(buf_event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &execution_status, &param_val_size_ret));
          TEST_ASSERT(param_val_size_ret == sizeof(cl_int));
          TEST_ASSERT(execution_status == CL_COMPLETE);

          CHECK_CL_ERROR(clReleaseEvent(buf_event));

          CHECK_CL_ERROR(clReleaseMemObject(buf));

          CHECK_CL_ERROR(clReleaseCommandQueue(queue));

          CHECK_CL_ERROR (clReleaseContext (context));
          CHECK_CL_ERROR (clUnloadPlatformCompiler (platforms[i]));
        }
    }

  printf ("OK\n");
  return EXIT_SUCCESS;
}
