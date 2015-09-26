/* poclu_misc - misc generic OpenCL helper functions

   Copyright (c) 2013 Pekka Jääskeläinen / Tampere University of Technology
   Copyright (c) 2014 Kalle Raiskila
   
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

#include "poclu.h"
#include <CL/opencl.h>
#include <stdlib.h>
#include <stdio.h>
#include "config.h"

cl_context
poclu_create_any_context() 
{
  cl_uint i;
  cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id));

  clGetPlatformIDs(1, platforms, &i);
  if (i == 0)
    return (cl_context)0;

  cl_context_properties properties[] = 
    {CL_CONTEXT_PLATFORM, 
     (cl_context_properties)platforms[0], 
     0};

  // create the OpenCL context on any available OCL device 
  cl_context context = clCreateContextFromType(
      properties, 
      CL_DEVICE_TYPE_ALL,
      NULL, NULL, NULL); 

  free (platforms);
  return context;
}

cl_int
poclu_get_any_device( cl_context *context, cl_device_id *device, cl_command_queue *queue)
{
  cl_int err;  
  cl_platform_id platform;

  if (context == NULL ||
      device  == NULL ||
      queue   == NULL)
    return CL_INVALID_VALUE;  

  err = clGetPlatformIDs(1, &platform, NULL);
  if (err != CL_SUCCESS)
    return err;

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, device, NULL);
  if (err != CL_SUCCESS)
    return err;

  *context = clCreateContext(NULL, 1, device, NULL, NULL, &err);
  if (err != CL_SUCCESS)
    return err;
  
  *queue = clCreateCommandQueue(*context, *device, 0, &err); 
  if (err != CL_SUCCESS)
    return err;

  return CL_SUCCESS;
}

char *
poclu_read_file(char *filename)
{
  FILE *file;
  long size;
  char* src;
  
  file = fopen(filename, "r");
  if (file == NULL)
    return NULL;
  
  fseek( file, 0, SEEK_END);
  size = ftell(file);
  src = (char*)malloc(size+1);
  if (src == NULL) 
    {
      fclose(file);
      return NULL;
    }

  fseek(file, 0, SEEK_SET);
  fread(src, size, 1, file);
  fclose(file);
  src[size]=0;

  return src;
}

int
poclu_write_file(char* filemane, char* content, size_t size)
{
  FILE *file;

  file = fopen(filemane, "w");
  if (file == NULL)
    return -1;

  if (fwrite(content, sizeof(char), size, file) < size)
    return -1;

  if (fclose(file))
    return -1;

  return 0;
}

#define OPENCL_ERROR_CASE(ERR) \
  case ERR: \
    { fprintf(stderr, "" #ERR " in %s on line %i\n", func_name, line); return 1; }

int check_cl_error(cl_int cl_err, int line, const char* func_name) {

  switch(cl_err)
  {
    case CL_SUCCESS: return 0;

    OPENCL_ERROR_CASE(CL_DEVICE_NOT_FOUND)
    OPENCL_ERROR_CASE(CL_DEVICE_NOT_AVAILABLE)
    OPENCL_ERROR_CASE(CL_COMPILER_NOT_AVAILABLE)
    OPENCL_ERROR_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE)
    OPENCL_ERROR_CASE(CL_OUT_OF_RESOURCES)
    OPENCL_ERROR_CASE(CL_OUT_OF_HOST_MEMORY)
    OPENCL_ERROR_CASE(CL_PROFILING_INFO_NOT_AVAILABLE)
    OPENCL_ERROR_CASE(CL_MEM_COPY_OVERLAP)
    OPENCL_ERROR_CASE(CL_IMAGE_FORMAT_MISMATCH)
    OPENCL_ERROR_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED)
    OPENCL_ERROR_CASE(CL_BUILD_PROGRAM_FAILURE)
    OPENCL_ERROR_CASE(CL_MAP_FAILURE)
    OPENCL_ERROR_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET)
    OPENCL_ERROR_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST)
    OPENCL_ERROR_CASE(CL_COMPILE_PROGRAM_FAILURE)
    OPENCL_ERROR_CASE(CL_LINKER_NOT_AVAILABLE)
    OPENCL_ERROR_CASE(CL_LINK_PROGRAM_FAILURE)
    OPENCL_ERROR_CASE(CL_DEVICE_PARTITION_FAILED)
    OPENCL_ERROR_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)

    OPENCL_ERROR_CASE(CL_INVALID_VALUE)
    OPENCL_ERROR_CASE(CL_INVALID_DEVICE_TYPE)
    OPENCL_ERROR_CASE(CL_INVALID_PLATFORM)
    OPENCL_ERROR_CASE(CL_INVALID_DEVICE)
    OPENCL_ERROR_CASE(CL_INVALID_CONTEXT)
    OPENCL_ERROR_CASE(CL_INVALID_QUEUE_PROPERTIES)
    OPENCL_ERROR_CASE(CL_INVALID_COMMAND_QUEUE)
    OPENCL_ERROR_CASE(CL_INVALID_HOST_PTR)
    OPENCL_ERROR_CASE(CL_INVALID_MEM_OBJECT)
    OPENCL_ERROR_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
    OPENCL_ERROR_CASE(CL_INVALID_IMAGE_SIZE)
    OPENCL_ERROR_CASE(CL_INVALID_SAMPLER)
    OPENCL_ERROR_CASE(CL_INVALID_BINARY)
    OPENCL_ERROR_CASE(CL_INVALID_BUILD_OPTIONS)
    OPENCL_ERROR_CASE(CL_INVALID_PROGRAM)
    OPENCL_ERROR_CASE(CL_INVALID_PROGRAM_EXECUTABLE)
    OPENCL_ERROR_CASE(CL_INVALID_KERNEL_NAME)
    OPENCL_ERROR_CASE(CL_INVALID_KERNEL_DEFINITION)
    OPENCL_ERROR_CASE(CL_INVALID_KERNEL)
    OPENCL_ERROR_CASE(CL_INVALID_ARG_INDEX)
    OPENCL_ERROR_CASE(CL_INVALID_ARG_VALUE)
    OPENCL_ERROR_CASE(CL_INVALID_ARG_SIZE)
    OPENCL_ERROR_CASE(CL_INVALID_KERNEL_ARGS)
    OPENCL_ERROR_CASE(CL_INVALID_WORK_DIMENSION)
    OPENCL_ERROR_CASE(CL_INVALID_WORK_GROUP_SIZE)
    OPENCL_ERROR_CASE(CL_INVALID_WORK_ITEM_SIZE)
    OPENCL_ERROR_CASE(CL_INVALID_GLOBAL_OFFSET)
    OPENCL_ERROR_CASE(CL_INVALID_EVENT_WAIT_LIST)
    OPENCL_ERROR_CASE(CL_INVALID_EVENT)
    OPENCL_ERROR_CASE(CL_INVALID_OPERATION)
    OPENCL_ERROR_CASE(CL_INVALID_GL_OBJECT)
    OPENCL_ERROR_CASE(CL_INVALID_BUFFER_SIZE)
    OPENCL_ERROR_CASE(CL_INVALID_MIP_LEVEL)
    OPENCL_ERROR_CASE(CL_INVALID_GLOBAL_WORK_SIZE)
    OPENCL_ERROR_CASE(CL_INVALID_PROPERTY)
    OPENCL_ERROR_CASE(CL_INVALID_IMAGE_DESCRIPTOR)
    OPENCL_ERROR_CASE(CL_INVALID_COMPILER_OPTIONS)
    OPENCL_ERROR_CASE(CL_INVALID_LINKER_OPTIONS)
    OPENCL_ERROR_CASE(CL_INVALID_DEVICE_PARTITION_COUNT)

    default:
      printf("Unknown OpenCL error %i in %s on line %i\n", cl_err, func_name, line);
      return 1;
  }
}
