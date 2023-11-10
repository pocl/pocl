/* OpenCL runtime library: clGetFunctionAddressForPlatformKHR()

   Copyright (c) 2017 Michal Babej / Tampere University of Technology

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

#include "pocl_cl.h"

#include <string.h>

#ifdef BUILD_ICD

#define POCL_GET_FUNCTION_ADDRESS_IF_NAME(__name__, __api__)                   \
do {                                                                           \
  if (strcmp (func_name, #__name__) == 0)                                       \
    return (void *)&POname(__api__);                                           \
} while (0)

#define POCL_GET_FUNCTION_ADDRESS(__API__)                                     \
POCL_GET_FUNCTION_ADDRESS_IF_NAME(__API__, __API__)

POCL_EXPORT CL_API_ENTRY void * CL_API_CALL
POname (clGetFunctionAddressForPlatformKHR) (cl_platform_id  platform,
                                             const char *func_name)
{
  cl_platform_id pocl_platform;
  cl_uint actual_num = 0;
  POname (clGetPlatformIDs) (1, &pocl_platform, &actual_num);
  if (actual_num != 1)
    {
      POCL_MSG_WARN ("Couldn't get the platform ID of PoCL platform\n");
      return NULL;
    }

  assert (pocl_platform);
  if (platform != pocl_platform)
    {
      POCL_MSG_WARN ("Requested Function Address not "
                     "for PoCL platform, ignoring\n");
      return NULL;
    }

  POCL_GET_FUNCTION_ADDRESS(clGetPlatformIDs);
  POCL_GET_FUNCTION_ADDRESS(clGetPlatformInfo);
  POCL_GET_FUNCTION_ADDRESS(clGetDeviceIDs);
  POCL_GET_FUNCTION_ADDRESS(clGetDeviceInfo);
  POCL_GET_FUNCTION_ADDRESS(clCreateContext);
  POCL_GET_FUNCTION_ADDRESS(clCreateContextFromType);
  POCL_GET_FUNCTION_ADDRESS(clRetainContext);
  POCL_GET_FUNCTION_ADDRESS(clReleaseContext);
  POCL_GET_FUNCTION_ADDRESS(clGetContextInfo);
  POCL_GET_FUNCTION_ADDRESS(clCreateCommandQueue);
  POCL_GET_FUNCTION_ADDRESS(clRetainCommandQueue);
  POCL_GET_FUNCTION_ADDRESS(clReleaseCommandQueue);
  POCL_GET_FUNCTION_ADDRESS(clGetCommandQueueInfo);
  POCL_GET_FUNCTION_ADDRESS(clCreateBuffer);
  POCL_GET_FUNCTION_ADDRESS(clCreateImage2D);
  POCL_GET_FUNCTION_ADDRESS(clCreateImage3D);
  POCL_GET_FUNCTION_ADDRESS(clRetainMemObject);
  POCL_GET_FUNCTION_ADDRESS(clReleaseMemObject);
  POCL_GET_FUNCTION_ADDRESS(clGetSupportedImageFormats);
  POCL_GET_FUNCTION_ADDRESS(clGetMemObjectInfo);
  POCL_GET_FUNCTION_ADDRESS(clGetImageInfo);
  POCL_GET_FUNCTION_ADDRESS(clCreateSampler);
  POCL_GET_FUNCTION_ADDRESS(clRetainSampler);
  POCL_GET_FUNCTION_ADDRESS(clReleaseSampler);
  POCL_GET_FUNCTION_ADDRESS(clGetSamplerInfo);
  POCL_GET_FUNCTION_ADDRESS(clCreateProgramWithSource);
  POCL_GET_FUNCTION_ADDRESS(clCreateProgramWithBinary);
  POCL_GET_FUNCTION_ADDRESS(clRetainProgram);
  POCL_GET_FUNCTION_ADDRESS(clReleaseProgram);
  POCL_GET_FUNCTION_ADDRESS(clBuildProgram);
  POCL_GET_FUNCTION_ADDRESS(clUnloadCompiler);
  POCL_GET_FUNCTION_ADDRESS(clGetProgramInfo);
  POCL_GET_FUNCTION_ADDRESS(clGetProgramBuildInfo);
  POCL_GET_FUNCTION_ADDRESS(clCreateKernel);
  POCL_GET_FUNCTION_ADDRESS(clCreateKernelsInProgram);
  POCL_GET_FUNCTION_ADDRESS(clRetainKernel);
  POCL_GET_FUNCTION_ADDRESS(clReleaseKernel);
  POCL_GET_FUNCTION_ADDRESS(clSetKernelArg);
  POCL_GET_FUNCTION_ADDRESS(clGetKernelInfo);
  POCL_GET_FUNCTION_ADDRESS(clGetKernelWorkGroupInfo);
  POCL_GET_FUNCTION_ADDRESS(clWaitForEvents);
  POCL_GET_FUNCTION_ADDRESS(clGetEventInfo);
  POCL_GET_FUNCTION_ADDRESS(clRetainEvent);
  POCL_GET_FUNCTION_ADDRESS(clReleaseEvent);
  POCL_GET_FUNCTION_ADDRESS(clGetEventProfilingInfo);
  POCL_GET_FUNCTION_ADDRESS(clFlush);
  POCL_GET_FUNCTION_ADDRESS(clFinish);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueReadBuffer);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueWriteBuffer);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueCopyBuffer);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueReadImage);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueWriteImage);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueCopyImage);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueCopyImageToBuffer);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueCopyBufferToImage);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueMapBuffer);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueMapImage);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueUnmapMemObject);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueNDRangeKernel);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueTask);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueNativeKernel);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueMarker);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueWaitForEvents);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueBarrier);
  POCL_GET_FUNCTION_ADDRESS(clGetExtensionFunctionAddress);
  POCL_GET_FUNCTION_ADDRESS(clCreateFromGLBuffer);
  POCL_GET_FUNCTION_ADDRESS(clCreateFromGLTexture2D);
  POCL_GET_FUNCTION_ADDRESS(clCreateFromGLTexture3D);
  POCL_GET_FUNCTION_ADDRESS(clCreateFromGLRenderbuffer);
  POCL_GET_FUNCTION_ADDRESS(clGetGLObjectInfo);
  POCL_GET_FUNCTION_ADDRESS(clGetGLTextureInfo);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueAcquireGLObjects);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueReleaseGLObjects);
  POCL_GET_FUNCTION_ADDRESS(clGetGLContextInfoKHR);
  POCL_GET_FUNCTION_ADDRESS(clSetEventCallback);
  POCL_GET_FUNCTION_ADDRESS(clCreateSubBuffer);
  POCL_GET_FUNCTION_ADDRESS(clSetMemObjectDestructorCallback);
  POCL_GET_FUNCTION_ADDRESS(clCreateUserEvent);
  POCL_GET_FUNCTION_ADDRESS(clSetUserEventStatus);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueReadBufferRect);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueWriteBufferRect);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueCopyBufferRect);
  POCL_GET_FUNCTION_ADDRESS_IF_NAME(clRetainDeviceEXT, clRetainDevice);
  POCL_GET_FUNCTION_ADDRESS_IF_NAME(clReleaseDeviceEXT, clReleaseDevice);
  POCL_GET_FUNCTION_ADDRESS(clCreateSubDevices);
  POCL_GET_FUNCTION_ADDRESS(clRetainDevice);
  POCL_GET_FUNCTION_ADDRESS(clReleaseDevice);
  POCL_GET_FUNCTION_ADDRESS(clCreateImage);
  POCL_GET_FUNCTION_ADDRESS(clCreateProgramWithBuiltInKernels);
  POCL_GET_FUNCTION_ADDRESS(clCompileProgram);
  POCL_GET_FUNCTION_ADDRESS(clLinkProgram);
  POCL_GET_FUNCTION_ADDRESS(clUnloadPlatformCompiler);
  POCL_GET_FUNCTION_ADDRESS(clGetKernelArgInfo);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueFillBuffer);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueFillImage);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueMigrateMemObjects);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueMarkerWithWaitList);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueBarrierWithWaitList);
  POCL_GET_FUNCTION_ADDRESS(clGetExtensionFunctionAddressForPlatform);
  POCL_GET_FUNCTION_ADDRESS(clCreateFromGLTexture);
  POCL_GET_FUNCTION_ADDRESS(clCreateCommandQueueWithProperties);
  POCL_GET_FUNCTION_ADDRESS(clCreatePipe);
  POCL_GET_FUNCTION_ADDRESS(clGetPipeInfo);
  POCL_GET_FUNCTION_ADDRESS(clSVMAlloc);
  POCL_GET_FUNCTION_ADDRESS(clSVMFree);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueSVMFree);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueSVMMemcpy);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueSVMMemFill);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueSVMMap);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueSVMUnmap);
  POCL_GET_FUNCTION_ADDRESS(clCreateSamplerWithProperties);
  POCL_GET_FUNCTION_ADDRESS(clSetKernelArgSVMPointer);
  POCL_GET_FUNCTION_ADDRESS(clSetKernelExecInfo);
  POCL_GET_FUNCTION_ADDRESS(clGetKernelSubGroupInfo);
  POCL_GET_FUNCTION_ADDRESS(clCloneKernel);
  POCL_GET_FUNCTION_ADDRESS(clCreateProgramWithIL);
  POCL_GET_FUNCTION_ADDRESS(clEnqueueSVMMigrateMem);
  POCL_GET_FUNCTION_ADDRESS(clGetDeviceAndHostTimer);
  POCL_GET_FUNCTION_ADDRESS(clGetHostTimer);
  POCL_GET_FUNCTION_ADDRESS(clGetKernelSubGroupInfo);
  POCL_GET_FUNCTION_ADDRESS(clSetDefaultDeviceCommandQueue);
  POCL_GET_FUNCTION_ADDRESS(clSetProgramReleaseCallback);
  POCL_GET_FUNCTION_ADDRESS(clSetProgramSpecializationConstant);
  POCL_GET_FUNCTION_ADDRESS(clCreateBufferWithProperties);
  POCL_GET_FUNCTION_ADDRESS(clCreateImageWithProperties);
  POCL_GET_FUNCTION_ADDRESS(clSetContextDestructorCallback);

  POCL_MSG_WARN ("Unsupported function %s required\n", func_name);
  return NULL;
}
POsymICD(clGetFunctionAddressForPlatformKHR)
#endif
