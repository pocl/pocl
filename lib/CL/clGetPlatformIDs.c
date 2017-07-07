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
  &POclGetPlatformIDs,
  &POclGetPlatformInfo,
  &POclGetDeviceIDs,
  &POclGetDeviceInfo,
  &POclCreateContext,
  &POclCreateContextFromType,
  &POclRetainContext,
  &POclReleaseContext,
  &POclGetContextInfo,
  &POclCreateCommandQueue,
  &POclRetainCommandQueue, /* 10 */
  &POclReleaseCommandQueue,
  &POclGetCommandQueueInfo,
  NULL /*clSetCommandQueueProperty*/,
  &POclCreateBuffer,
  &POclCreateImage2D,
  &POclCreateImage3D,
  &POclRetainMemObject,
  &POclReleaseMemObject,
  &POclGetSupportedImageFormats,
  &POclGetMemObjectInfo, /* 20 */
  &POclGetImageInfo,
  &POclCreateSampler,
  &POclRetainSampler,
  &POclReleaseSampler,
  &POclGetSamplerInfo,
  &POclCreateProgramWithSource,
  &POclCreateProgramWithBinary,
  &POclRetainProgram,
  &POclReleaseProgram,
  &POclBuildProgram, /* 30 */
  &POclUnloadCompiler,
  &POclGetProgramInfo,
  &POclGetProgramBuildInfo,
  &POclCreateKernel,
  &POclCreateKernelsInProgram,
  &POclRetainKernel,
  &POclReleaseKernel,
  &POclSetKernelArg,
  &POclGetKernelInfo,
  &POclGetKernelWorkGroupInfo, /* 40 */
  &POclWaitForEvents,
  &POclGetEventInfo,
  &POclRetainEvent,
  &POclReleaseEvent,
  &POclGetEventProfilingInfo,
  &POclFlush,
  &POclFinish,
  &POclEnqueueReadBuffer,
  &POclEnqueueWriteBuffer,
  &POclEnqueueCopyBuffer, /* 50 */
  &POclEnqueueReadImage,
  &POclEnqueueWriteImage,
  &POclEnqueueCopyImage,
  &POclEnqueueCopyImageToBuffer,
  &POclEnqueueCopyBufferToImage,
  &POclEnqueueMapBuffer,
  &POclEnqueueMapImage,
  &POclEnqueueUnmapMemObject,
  &POclEnqueueNDRangeKernel,
  &POclEnqueueTask, /* 60 */
  &POclEnqueueNativeKernel,
  &POclEnqueueMarker,
  &POclEnqueueWaitForEvents,
  &POclEnqueueBarrier,
  &POclGetExtensionFunctionAddress,
  NULL, /* &POclCreateFromGLBuffer,      */
  &POclCreateFromGLTexture2D,
  &POclCreateFromGLTexture3D,
  NULL, /* &POclCreateFromGLRenderbuffer, */
  NULL, /* &POclGetGLObjectInfo,  70       */
  NULL, /* &POclGetGLTextureInfo,        */
  NULL, /* &POclEnqueueAcquireGLObjects, */
  NULL, /* &POclEnqueueReleaseGLObjects, */
  NULL, /* &POclGetGLContextInfoKHR,     */
  NULL, /* &clUnknown75 */
  NULL, /* &clUnknown76 */
  NULL, /* &clUnknown77 */
  NULL, /* &clUnknown78 */
  NULL, /* &clUnknown79 */
  NULL, /* &clUnknown80 */
  &POclSetEventCallback,
  &POclCreateSubBuffer,
  &POclSetMemObjectDestructorCallback,
  &POclCreateUserEvent,
  &POclSetUserEventStatus,
  &POclEnqueueReadBufferRect,
  &POclEnqueueWriteBufferRect,
  &POclEnqueueCopyBufferRect,
  NULL, /* &POclCreateSubDevicesEXT,     */
  &POclRetainDevice, /* &POclRetainDeviceEXT,         */
  &POclReleaseDevice, /* &POclReleaseDeviceEXT,        */
  NULL, /* &clUnknown92 */
  &POclCreateSubDevices,
  &POclRetainDevice,
  &POclReleaseDevice,
  &POclCreateImage,
  NULL, /* &POclCreateProgramWithBuiltInKernels, */
  &POclCompileProgram,
  &POclLinkProgram,
  &POclUnloadPlatformCompiler, 
  &POclGetKernelArgInfo,
  &POclEnqueueFillBuffer,
  &POclEnqueueFillImage,
  &POclEnqueueMigrateMemObjects,
  &POclEnqueueMarkerWithWaitList,
  &POclEnqueueBarrierWithWaitList,
  NULL, /* &POclGetExtensionFunctionAddressForPlatform, */
  NULL, /* &POclCreateFromGLTexture,     */
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
  &POclCreateCommandQueueWithProperties,
  NULL, /* &POclCreatePipe,*/
  NULL, /* &POclGetPipeInfo,*/
  &POclSVMAlloc,
  &POclSVMFree,
  &POclEnqueueSVMFree,
  &POclEnqueueSVMMemcpy,
  &POclEnqueueSVMMemFill,
  &POclEnqueueSVMMap,
  &POclEnqueueSVMUnmap,
  NULL, /* clCreateSamplerWithProperties */
  &POclSetKernelArgSVMPointer,
  &POclSetKernelExecInfo,
  NULL, /* &clUnknown136 */
  NULL, /* &clUnknown137 */
  NULL, /* &clUnknown138 */
  NULL, /* &clUnknown139 */
  NULL, /* &clUnknown140 */
  NULL, /* &clUnknown141 */
  NULL, /* &clUnknown142 */
  NULL, /* &clUnknown143 */
  NULL, /* &clUnknown144 */
  NULL, /* &clUnknown145 */
  NULL, /* &clUnknown146 */
  NULL, /* &clUnknown147 */
  NULL, /* &clUnknown148 */
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
};

struct _cl_platform_id _platforms[1]  = {{&pocl_dispatch}};
#else
struct _cl_platform_id _platforms[1]  = {};
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
  const unsigned num = 1;
  unsigned i;
  
  if (platforms != NULL) {
    if (num_entries < num)
      return CL_INVALID_VALUE;
    
    for (i=0; i<num; ++i)
      platforms[i] = &_platforms[i];
  }
  
  if (num_platforms != NULL)
    *num_platforms = num;
  
  return CL_SUCCESS;
}
POsym(clGetPlatformIDs)
