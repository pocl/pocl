#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define MAX_PLATFORMS 32
#define MAX_DEVICES   32

int
main(void)
{
  cl_int err;
  cl_platform_id platforms[MAX_PLATFORMS];
  cl_uint nplatforms;
  cl_device_id devices[MAX_DEVICES];
  cl_uint ndevices;
  cl_uint i, j;

  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &nplatforms);	
  if (err != CL_SUCCESS)
    return EXIT_FAILURE;

  for (i = 0; i < nplatforms; i++)
    {
      err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES,
			   devices, &ndevices);
      if (err != CL_SUCCESS)
	return EXIT_FAILURE;
      
      for (j = 0; j < ndevices; j++)
	{
	  cl_context context = clCreateContext(NULL, 1, &devices[j], NULL, NULL, &err);
	  if (err != CL_SUCCESS)
	    return EXIT_FAILURE;
	  cl_command_queue queue = clCreateCommandQueue(context, devices[j], 0, &err);
	  if (err != CL_SUCCESS)
	    return EXIT_FAILURE;

	  const int buf_size = 1024;
	  cl_int host_buf[buf_size];

	  cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * buf_size, NULL, &err);
	  if (err != CL_SUCCESS)
	    return EXIT_FAILURE;
	  cl_event buf_event;
	  if (clEnqueueReadBuffer(queue, buf, CL_TRUE, 0, sizeof(cl_int) * buf_size, &host_buf, 0, NULL, &buf_event) != CL_SUCCESS)
	    return EXIT_FAILURE;
	  clFinish(queue);
	  cl_command_queue event_command_queue;
	  size_t param_val_size_ret;
	  if (clGetEventInfo(buf_event, CL_EVENT_COMMAND_QUEUE, sizeof(cl_command_queue), &event_command_queue, &param_val_size_ret) != CL_SUCCESS)
	    return EXIT_FAILURE;
	  if (param_val_size_ret != sizeof(cl_command_queue) || event_command_queue != queue)
	    return EXIT_FAILURE;

	  cl_command_type command_type;
	  if (clGetEventInfo(buf_event, CL_EVENT_COMMAND_TYPE, sizeof(cl_command_type), &command_type, &param_val_size_ret) != CL_SUCCESS)
	    return EXIT_FAILURE;
	  if (param_val_size_ret != sizeof(cl_command_type) || command_type != CL_COMMAND_READ_BUFFER)
	    return EXIT_FAILURE;

	  cl_int execution_status;
	  if (clGetEventInfo(buf_event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &execution_status, &param_val_size_ret) != CL_SUCCESS)
	    return EXIT_FAILURE;
	  if (param_val_size_ret != sizeof(cl_int) || execution_status != CL_COMPLETE)
	    return EXIT_FAILURE;

	  cl_uint ref_count;
	  if (clGetEventInfo(buf_event, CL_EVENT_REFERENCE_COUNT, sizeof(cl_uint), &ref_count, &param_val_size_ret) != CL_SUCCESS)
	    return EXIT_FAILURE;
	  if (param_val_size_ret != sizeof(cl_uint) || ref_count != 1)
	    return EXIT_FAILURE;

	  clReleaseEvent(buf_event);
	  clReleaseMemObject(buf);
	  clReleaseCommandQueue(queue);
	}
    }
  return EXIT_SUCCESS;
}
