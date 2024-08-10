/* pocl_intfn.h - local (non-exported) OpenCL functions

   Copyright (c) 2012 Vincent Danjean <Vincent.Danjean@ens-lyon.org>
   
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

#ifndef POCL_INTFN_H
#define POCL_INTFN_H

#ifndef POCL_CL_H
#  error this file must be included through pocl_cl.h, not directly
#endif

#ifdef __cplusplus
extern "C" {
#endif

POdeclsym(clBuildProgram)
POdeclsym(clLinkProgram)
POdeclsym(clCompileProgram)
POdeclsymExport(clCreateBuffer)
POdeclsym(clCreateCommandQueue)
POdeclsymExport(clCreateContext)
POdeclsym(clCreateContextFromType)
POdeclsym(clCreateImage2D) 
POdeclsym(clCreateImage3D)
POdeclsym(clCreateImage)
POdeclsym(clCreateKernel)
POdeclsym(clCreateKernelsInProgram)
POdeclsym(clCreatePipe)
POdeclsym(clCreateProgramWithBinary)
POdeclsym(clCreateProgramWithIL)
POdeclsym(clCreateProgramWithBuiltInKernels)
POdeclsym(clCreateProgramWithSource)
POdeclsym(clCreateSampler)
POdeclsym(clCreateSubBuffer)
POdeclsym(clCreateSubDevices)
POdeclsym(clCreateUserEvent)
POdeclsym(clEnqueueBarrier)
POdeclsym(clEnqueueBarrierWithWaitList)
POdeclsym(clEnqueueCopyBuffer)
POdeclsym(clEnqueueCopyBufferRect)
POdeclsym(clEnqueueCopyBufferToImage) 
POdeclsym(clEnqueueCopyImage)
POdeclsym(clEnqueueCopyImageToBuffer)
POdeclsym(clEnqueueMapBuffer)
POdeclsym(clEnqueueMapImage)
POdeclsym(clEnqueueMarker) 
POdeclsym(clEnqueueMigrateMemObjects)
POdeclsym(clEnqueueNativeKernel)
POdeclsym(clEnqueueNDRangeKernel)
POdeclsym(clEnqueueReadBuffer)
POdeclsym(clEnqueueReadBufferRect)
POdeclsym(clEnqueueReadImage)
POdeclsym(clEnqueueTask)
POdeclsym(clEnqueueUnmapMemObject)
POdeclsym(clEnqueueWaitForEvents)
POdeclsym(clEnqueueMarkerWithWaitList)
POdeclsym(clEnqueueWriteBuffer)
POdeclsym(clEnqueueWriteBufferRect)
POdeclsym(clEnqueueWriteImage)
POdeclsym(clEnqueueFillImage)
POdeclsym(clEnqueueFillBuffer)
POdeclsym(clFinish)
POdeclsym(clFlush)
POdeclsym(clGetCommandQueueInfo)
POdeclsym(clGetContextInfo)
POdeclsym(clGetDeviceIDs)
POdeclsym(clGetDeviceInfo)
POdeclsym(clGetEventInfo)
POdeclsym(clGetEventProfilingInfo)
POdeclsym(clGetExtensionFunctionAddress)
POdeclsym(clGetExtensionFunctionAddressForPlatform)
POdeclsym(clGetImageInfo)
POdeclsym(clGetKernelInfo)
POdeclsym(clGetKernelArgInfo)
POdeclsym(clGetKernelWorkGroupInfo)
POdeclsym(clGetKernelSubGroupInfo)
POdeclsymExport(clGetMemObjectInfo)
POdeclsym(clGetPlatformIDs)
POdeclsym(clGetPlatformInfo)
POdeclsym(clGetProgramBuildInfo)
POdeclsym(clGetProgramInfo)
POdeclsym(clGetSamplerInfo)
POdeclsym(clGetSupportedImageFormats)
POdeclsymICD(clIcdGetPlatformIDsKHR)
POdeclsym(clReleaseCommandQueue)
POdeclsymExport(clReleaseContext)
POdeclsymExport(clReleaseDevice)
POdeclsymExport(clReleaseEvent)
POdeclsym(clReleaseKernel)
POdeclsym(clReleaseMemObject)
POdeclsym(clReleaseProgram)
POdeclsym(clReleaseSampler)
POdeclsym(clRetainCommandQueue)
POdeclsym(clRetainContext)
POdeclsym(clRetainDevice)
POdeclsymExport(clRetainEvent)
POdeclsym(clRetainKernel)
POdeclsym(clRetainMemObject)
POdeclsym(clRetainProgram)
POdeclsym(clRetainSampler)
POdeclsym(clSetEventCallback)
POdeclsym(clSetKernelArg)
POdeclsym(clSetMemObjectDestructorCallback)
POdeclsym(clSetUserEventStatus)
POdeclsym(clUnloadCompiler)
POdeclsym(clUnloadPlatformCompiler)
POdeclsym(clWaitForEvents)
POdeclsym(clEnqueueSVMFree)
POdeclsym(clEnqueueSVMMap)
POdeclsym(clEnqueueSVMMemcpy)
POdeclsym(clEnqueueSVMMemFill)
POdeclsym(clEnqueueSVMUnmap)
POdeclsym(clSVMFree)
POdeclsym(clSVMAlloc)
POdeclsym(clSetKernelArgSVMPointer)
POdeclsym(clSetKernelExecInfo)
POdeclsym(clCreateCommandQueueWithProperties)
POdeclsym(clCreateFromGLBuffer)
POdeclsym(clCreateFromGLTexture)
POdeclsym(clCreateFromGLTexture2D)
POdeclsym(clCreateFromGLTexture3D)
POdeclsym(clCreateFromGLRenderbuffer)
POdeclsym(clGetGLObjectInfo)
POdeclsym(clGetGLTextureInfo)
POdeclsym(clEnqueueAcquireGLObjects)
POdeclsym(clEnqueueReleaseGLObjects)
POdeclsym(clGetGLContextInfoKHR)
POdeclsym(clSetContentSizeBufferPoCL)
POdeclsym(clCreatePipe)
POdeclsym(clGetPipeInfo)
POdeclsym(clSetDefaultDeviceCommandQueue)
POdeclsym(clGetDeviceAndHostTimer)
POdeclsym(clGetHostTimer)
POdeclsym(clSetProgramReleaseCallback)
POdeclsym(clSetContextDestructorCallback)
POdeclsym(clSetProgramSpecializationConstant)
POdeclsym(clCreateSamplerWithProperties)
POdeclsym(clCreateBufferWithProperties)
POdeclsym(clCreateImageWithProperties)
POdeclsym(clCloneKernel)
POdeclsym(clEnqueueSVMMigrateMem)

/* cl_khr_command_buffer */
POdeclsym(clCreateCommandBufferKHR)
POdeclsym(clRetainCommandBufferKHR)
POdeclsym(clReleaseCommandBufferKHR)
POdeclsym(clFinalizeCommandBufferKHR)
POdeclsym(clEnqueueCommandBufferKHR)
POdeclsym(clCommandBarrierWithWaitListKHR)
POdeclsym(clCommandNDRangeKernelKHR)
POdeclsym(clCommandCopyBufferKHR)
POdeclsym(clCommandCopyBufferRectKHR)
POdeclsym(clCommandCopyBufferToImageKHR)
POdeclsym(clCommandCopyImageKHR)
POdeclsym(clCommandCopyImageToBufferKHR)
POdeclsym(clCommandFillBufferKHR)
POdeclsym(clCommandFillImageKHR)
POdeclsym(clGetCommandBufferInfoKHR)

/* cl_khr_command_buffer_multi_device */
POdeclsym(clRemapCommandBufferKHR)

/* cl_intel_unified_shared_memory */
POdeclsym(clHostMemAllocINTEL)
POdeclsym(clDeviceMemAllocINTEL)
POdeclsym(clSharedMemAllocINTEL)
POdeclsym(clMemFreeINTEL)
POdeclsym(clMemBlockingFreeINTEL)
POdeclsym(clGetMemAllocInfoINTEL)
POdeclsym(clSetKernelArgMemPointerINTEL)
POdeclsym(clEnqueueMemFillINTEL)
POdeclsym(clEnqueueMemcpyINTEL)
POdeclsym(clEnqueueMemAdviseINTEL)
POdeclsym(clEnqueueMigrateMemINTEL)

/* cl_khr_command_buffer 0.9.4 */
POdeclsym(clCommandSVMMemcpyKHR)
POdeclsym(clCommandSVMMemFillKHR)

/* cl_pocl_command_buffer_svm */
POdeclsym(clCommandSVMMemcpyPOCL)
POdeclsym(clCommandSVMMemcpyRectPOCL)
POdeclsym(clCommandSVMMemfillPOCL)
POdeclsym(clCommandSVMMemfillRectPOCL)

/* cl_pocl_command_buffer_host_buffer */
POdeclsym(clCommandReadBufferPOCL)
POdeclsym(clCommandReadBufferRectPOCL)
POdeclsym(clCommandReadImagePOCL)
POdeclsym(clCommandWriteBufferPOCL)
POdeclsym(clCommandWriteBufferRectPOCL)
POdeclsym(clCommandWriteImagePOCL)

/* cl_pocl_svm_rect */
POdeclsym(clEnqueueSVMMemFillRectPOCL)
POdeclsym(clEnqueueSVMMemcpyRectPOCL)

/* cl_ext_buffer_device_address */
POdeclsym (clSetKernelArgDevicePointerEXT);

/* cl_exp_defined_builtin_kernels */
POdeclsym (clCreateProgramWithDefinedBuiltInKernels)

#ifdef __cplusplus
}
#endif

#endif
