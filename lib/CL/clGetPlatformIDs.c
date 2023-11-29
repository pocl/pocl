/* OpenCL runtime library: clGetPlatformIDs()

   Copyright (c) 2011 Kalle Raiskila 
   
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

#include <assert.h>
#include <string.h>
#include "pocl_cl.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/* The "implementation" of the _cl_device_id struct.
* Instantiated in clGetPlatformIDs.c
*
* TODO: the NULL entries are functions that lack implementation
* (or even stubs) in pocl
*/
#ifdef BUILD_ICD
struct _cl_icd_dispatch pocl_dispatch = {
  &POname(clGetPlatformIDs),
  &POname(clGetPlatformInfo),
  &POname(clGetDeviceIDs),
  &POname(clGetDeviceInfo),
  &POname(clCreateContext),
  &POname(clCreateContextFromType),
  &POname(clRetainContext),
  &POname(clReleaseContext),
  &POname(clGetContextInfo),
  &POname(clCreateCommandQueue),
  &POname(clRetainCommandQueue), /* 10 */
  &POname(clReleaseCommandQueue),
  &POname(clGetCommandQueueInfo),
  NULL /*clSetCommandQueueProperty*/,
  &POname(clCreateBuffer),
  &POname(clCreateImage2D),
  &POname(clCreateImage3D),
  &POname(clRetainMemObject),
  &POname(clReleaseMemObject),
  &POname(clGetSupportedImageFormats),
  &POname(clGetMemObjectInfo), /* 20 */
  &POname(clGetImageInfo),
  &POname(clCreateSampler),
  &POname(clRetainSampler),
  &POname(clReleaseSampler),
  &POname(clGetSamplerInfo),
  &POname(clCreateProgramWithSource),
  &POname(clCreateProgramWithBinary),
  &POname(clRetainProgram),
  &POname(clReleaseProgram),
  &POname(clBuildProgram), /* 30 */
  &POname(clUnloadCompiler),
  &POname(clGetProgramInfo),
  &POname(clGetProgramBuildInfo),
  &POname(clCreateKernel),
  &POname(clCreateKernelsInProgram),
  &POname(clRetainKernel),
  &POname(clReleaseKernel),
  &POname(clSetKernelArg),
  &POname(clGetKernelInfo),
  &POname(clGetKernelWorkGroupInfo), /* 40 */
  &POname(clWaitForEvents),
  &POname(clGetEventInfo),
  &POname(clRetainEvent),
  &POname(clReleaseEvent),
  &POname(clGetEventProfilingInfo),
  &POname(clFlush),
  &POname(clFinish),
  &POname(clEnqueueReadBuffer),
  &POname(clEnqueueWriteBuffer),
  &POname(clEnqueueCopyBuffer), /* 50 */
  &POname(clEnqueueReadImage),
  &POname(clEnqueueWriteImage),
  &POname(clEnqueueCopyImage),
  &POname(clEnqueueCopyImageToBuffer),
  &POname(clEnqueueCopyBufferToImage),
  &POname(clEnqueueMapBuffer),
  &POname(clEnqueueMapImage),
  &POname(clEnqueueUnmapMemObject),
  &POname(clEnqueueNDRangeKernel),
  &POname(clEnqueueTask), /* 60 */
  &POname(clEnqueueNativeKernel),
  &POname(clEnqueueMarker),
  &POname(clEnqueueWaitForEvents),
  &POname(clEnqueueBarrier),
  &POname(clGetExtensionFunctionAddress),
  &POname(clCreateFromGLBuffer),
  &POname(clCreateFromGLTexture2D),
  &POname(clCreateFromGLTexture3D),
  &POname(clCreateFromGLRenderbuffer),
  &POname(clGetGLObjectInfo),
  &POname(clGetGLTextureInfo),
  &POname(clEnqueueAcquireGLObjects),
  &POname(clEnqueueReleaseGLObjects),
  &POname(clGetGLContextInfoKHR),
  NULL, /* &clUnknown75 */
  NULL, /* &clUnknown76 */
  NULL, /* &clUnknown77 */
  NULL, /* &clUnknown78 */
  NULL, /* &clUnknown79 */
  NULL, /* &clUnknown80 */
  &POname(clSetEventCallback),
  &POname(clCreateSubBuffer),
  &POname(clSetMemObjectDestructorCallback),
  &POname(clCreateUserEvent),
  &POname(clSetUserEventStatus),
  &POname(clEnqueueReadBufferRect),
  &POname(clEnqueueWriteBufferRect),
  &POname(clEnqueueCopyBufferRect),
  NULL, /* &POname(clCreateSubDevicesEXT),     */
  &POname(clRetainDevice), /* &POname(clRetainDeviceEXT),         */
  &POname(clReleaseDevice), /* &POname(clReleaseDeviceEXT),        */
  NULL, /* &clUnknown92 */
  &POname(clCreateSubDevices),
  &POname(clRetainDevice),
  &POname(clReleaseDevice),
  &POname(clCreateImage),
  &POname(clCreateProgramWithBuiltInKernels),
  &POname(clCompileProgram),
  &POname(clLinkProgram),
  &POname(clUnloadPlatformCompiler),
  &POname(clGetKernelArgInfo),
  &POname(clEnqueueFillBuffer),
  &POname(clEnqueueFillImage),
  &POname(clEnqueueMigrateMemObjects),
  &POname(clEnqueueMarkerWithWaitList),
  &POname(clEnqueueBarrierWithWaitList),
  &POname(clGetExtensionFunctionAddressForPlatform),
  &POname(clCreateFromGLTexture),
  NULL, /* &clUnknown109 */
  NULL, /* &clUnknown110 */
  NULL, /* &clUnknown111 */
  NULL, /* &clUnknown112 */
  NULL, /* &clUnknown113 */
  NULL, /* &clUnknown114 */
  NULL, /* &clUnknown115 */
  NULL, /* &clUnknown116 */
  NULL, /* &clUnknown117 */
  NULL, /* &clUnknown118 */
  NULL, /* &clUnknown119 */
  NULL, /* &clUnknown120 */
  NULL, /* &clUnknown121 */
  NULL, /* &clUnknown122 */
#if (OCL_ICD_IDENTIFIED_FUNCTIONS > 110)
  &POname(clCreateCommandQueueWithProperties),
  &POname(clCreatePipe),
  &POname(clGetPipeInfo),
  &POname(clSVMAlloc),
  &POname(clSVMFree),
  &POname(clEnqueueSVMFree),
  &POname(clEnqueueSVMMemcpy),
  &POname(clEnqueueSVMMemFill),
  &POname(clEnqueueSVMMap),
  &POname(clEnqueueSVMUnmap),
  &POname(clCreateSamplerWithProperties),
  &POname(clSetKernelArgSVMPointer),
  &POname(clSetKernelExecInfo),
  &POname(clGetKernelSubGroupInfo),
  &POname(clCloneKernel),
  &POname(clCreateProgramWithIL),
  &POname(clEnqueueSVMMigrateMem),
  &POname(clGetDeviceAndHostTimer),
  &POname(clGetHostTimer),
  &POname(clGetKernelSubGroupInfo),
  &POname(clSetDefaultDeviceCommandQueue),
  &POname(clSetProgramReleaseCallback),
  &POname(clSetProgramSpecializationConstant),
  &POname(clCreateBufferWithProperties),
  &POname(clCreateImageWithProperties),
  &POname(clSetContextDestructorCallback),
  NULL, /* &clUnknown149 */
  NULL, /* &clUnknown150 */
  NULL, /* &clUnknown151 */
  NULL, /* &clUnknown152 */
  NULL, /* &clUnknown153 */
  NULL, /* &clUnknown154 */
  NULL, /* &clUnknown155 */
  NULL, /* &clUnknown156 */
  NULL, /* &clUnknown157 */
  NULL, /* &clUnknown158 */
  NULL, /* &clUnknown159 */
  NULL, /* &clUnknown160 */
  NULL, /* &clUnknown161 */
  NULL, /* &clUnknown162 */
  NULL, /* &clUnknown163 */
  NULL, /* &clUnknown164 */
  NULL, /* &clUnknown165 */
#endif
#if (OCL_ICD_IDENTIFIED_FUNCTIONS > 127)
  NULL, /* &clUnknown166 */
  NULL, /* &clUnknown167 */
  NULL, /* &clUnknown168 */
  NULL, /* &clUnknown169 */
  NULL, /* &clUnknown170 */
  NULL, /* &clUnknown171 */
  NULL, /* &clUnknown172 */
  NULL, /* &clUnknown173 */
  NULL, /* &clUnknown174 */
  NULL, /* &clUnknown175 */
  NULL, /* &clUnknown176 */
  NULL, /* &clUnknown177 */
#endif
#if (OCL_ICD_IDENTIFIED_FUNCTIONS > 129)
  NULL, /* &clUnknown178 */
  NULL, /* &clUnknown179 */
#endif
};

static struct _cl_platform_id _platforms[1]  = {{&pocl_dispatch}};
#else

static struct _cl_platform_id _platforms[1] = {{ 1 }};
#endif

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

/*
 * Get the number of supported platforms on this system. 
 * On POCL, this trivially reduces to 1 - POCL itself.
 */ 
CL_API_ENTRY cl_int CL_API_CALL
POname(clGetPlatformIDs)(cl_uint           num_entries,
                         cl_platform_id *  platforms,
                         cl_uint *         num_platforms) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((platforms == NULL && num_entries > 0),
                          CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((platforms != NULL && num_entries == 0),
                          CL_INVALID_VALUE);

  POCL_RETURN_ERROR_COND ((num_platforms == NULL && num_entries == 0),
                          CL_SUCCESS);

  if (platforms != NULL) {
      platforms[0] = &_platforms[0];
  }

  if (num_platforms != NULL)
    *num_platforms = 1;

  return CL_SUCCESS;
}
POsym(clGetPlatformIDs)
