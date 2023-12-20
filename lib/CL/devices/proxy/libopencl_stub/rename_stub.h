/* rename_stub.h - renames ocl functions to the libopencl-stub prefix

   Copyright (c) 2023 PoCL Developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#ifndef _RENAME_STUB_H_
#define _RENAME_STUB_H_

#define clGetPlatformIDs stubclGetPlatformIDs
#define clGetPlatformInfo stubclGetPlatformInfo
#define clGetDeviceIDs stubclGetDeviceIDs
#define clGetDeviceInfo stubclGetDeviceInfo
#define clCreateSubDevices stubclCreateSubDevices
#define clRetainDevice stubclRetainDevice
#define clReleaseDevice stubclReleaseDevice
#define clCreateContext stubclCreateContext
#define clCreateContextFromType stubclCreateContextFromType
#define clRetainContext stubclRetainContext
#define clReleaseContext stubclReleaseContext
#define clGetContextInfo stubclGetContextInfo
#define clCreateCommandQueueWithProperties                                    \
  stubclCreateCommandQueueWithProperties
#define clRetainCommandQueue stubclRetainCommandQueue
#define clReleaseCommandQueue stubclReleaseCommandQueue
#define clGetCommandQueueInfo stubclGetCommandQueueInfo
#define clCreateBuffer stubclCreateBuffer
#define clCreateSubBuffer stubclCreateSubBuffer
#define clCreateImage stubclCreateImage
#define clCreatePipe stubclCreatePipe
#define clRetainMemObject stubclRetainMemObject
#define clReleaseMemObject stubclReleaseMemObject
#define clGetSupportedImageFormats stubclGetSupportedImageFormats
#define clGetMemObjectInfo stubclGetMemObjectInfo
#define clGetImageInfo stubclGetImageInfo
#define clGetPipeInfo stubclGetPipeInfo
#define clSetMemObjectDestructorCallback stubclSetMemObjectDestructorCallback
#define clSVMAlloc stubclSVMAlloc
#define clSVMFree stubclSVMFree
#define clCreateSamplerWithProperties stubclCreateSamplerWithProperties
#define clRetainSampler stubclRetainSampler
#define clReleaseSampler stubclReleaseSampler
#define clGetSamplerInfo stubclGetSamplerInfo
#define clCreateProgramWithSource stubclCreateProgramWithSource
#define clCreateProgramWithBinary stubclCreateProgramWithBinary
#define clCreateProgramWithIL stubclCreateProgramWithIL
#define clCreateProgramWithBuiltInKernels stubclCreateProgramWithBuiltInKernels
#define clRetainProgram stubclRetainProgram
#define clReleaseProgram stubclReleaseProgram
#define clBuildProgram stubclBuildProgram
#define clCompileProgram stubclCompileProgram
#define clLinkProgram stubclLinkProgram
#define clUnloadPlatformCompiler stubclUnloadPlatformCompiler
#define clGetProgramInfo stubclGetProgramInfo
#define clGetProgramBuildInfo stubclGetProgramBuildInfo
#define clCreateKernel stubclCreateKernel
#define clCreateKernelsInProgram stubclCreateKernelsInProgram
#define clRetainKernel stubclRetainKernel
#define clReleaseKernel stubclReleaseKernel
#define clSetKernelArg stubclSetKernelArg
#define clSetKernelArgSVMPointer stubclSetKernelArgSVMPointer
#define clSetKernelExecInfo stubclSetKernelExecInfo
#define clGetKernelInfo stubclGetKernelInfo
#define clGetKernelArgInfo stubclGetKernelArgInfo
#define clGetKernelWorkGroupInfo stubclGetKernelWorkGroupInfo
#define clWaitForEvents stubclWaitForEvents
#define clGetEventInfo stubclGetEventInfo
#define clCreateUserEvent stubclCreateUserEvent
#define clRetainEvent stubclRetainEvent
#define clReleaseEvent stubclReleaseEvent
#define clSetUserEventStatus stubclSetUserEventStatus
#define clSetEventCallback stubclSetEventCallback
#define clGetEventProfilingInfo stubclGetEventProfilingInfo
#define clFlush stubclFlush
#define clFinish stubclFinish
#define clEnqueueReadBuffer stubclEnqueueReadBuffer
#define clEnqueueReadBufferRect stubclEnqueueReadBufferRect
#define clEnqueueWriteBuffer stubclEnqueueWriteBuffer
#define clEnqueueWriteBufferRect stubclEnqueueWriteBufferRect
#define clEnqueueFillBuffer stubclEnqueueFillBuffer
#define clEnqueueCopyBuffer stubclEnqueueCopyBuffer
#define clEnqueueCopyBufferRect stubclEnqueueCopyBufferRect
#define clEnqueueReadImage stubclEnqueueReadImage
#define clEnqueueWriteImage stubclEnqueueWriteImage
#define clEnqueueFillImage stubclEnqueueFillImage
#define clEnqueueCopyImage stubclEnqueueCopyImage
#define clEnqueueCopyImageToBuffer stubclEnqueueCopyImageToBuffer
#define clEnqueueCopyBufferToImage stubclEnqueueCopyBufferToImage
#define clEnqueueMapBuffer stubclEnqueueMapBuffer
#define clEnqueueMapImage stubclEnqueueMapImage
#define clEnqueueUnmapMemObject stubclEnqueueUnmapMemObject
#define clEnqueueMigrateMemObjects stubclEnqueueMigrateMemObjects
#define clEnqueueNDRangeKernel stubclEnqueueNDRangeKernel
#define clEnqueueNativeKernel stubclEnqueueNativeKernel
#define clEnqueueMarkerWithWaitList stubclEnqueueMarkerWithWaitList
#define clEnqueueBarrierWithWaitList stubclEnqueueBarrierWithWaitList
#define clEnqueueSVMFree stubclEnqueueSVMFree
#define clEnqueueSVMMemcpy stubclEnqueueSVMMemcpy
#define clEnqueueSVMMemFill stubclEnqueueSVMMemFill
#define clEnqueueSVMMap stubclEnqueueSVMMap
#define clEnqueueSVMUnmap stubclEnqueueSVMUnmap
#define clGetExtensionFunctionAddressForPlatform                              \
  stubclGetExtensionFunctionAddressForPlatform
#define clCreateImage2D stubclCreateImage2D
#define clCreateImage3D stubclCreateImage3D
#define clEnqueueMarker stubclEnqueueMarker
#define clEnqueueWaitForEvents stubclEnqueueWaitForEvents
#define clEnqueueBarrier stubclEnqueueBarrier
#define clUnloadCompiler stubclUnloadCompiler
#define clGetExtensionFunctionAddress stubclGetExtensionFunctionAddress
#define clCreateCommandQueue stubclCreateCommandQueue
#define clCreateSampler stubclCreateSampler
#define clEnqueueTask stubclEnqueueTask

#define clCreateFromGLTexture stubclCreateFromGLTexture
#define clCreateFromGLTexture2D stubclCreateFromGLTexture2D
#define clCreateFromGLTexture3D stubclCreateFromGLTexture3D
#define clEnqueueAcquireGLObjects stubclEnqueueAcquireGLObjects
#define clEnqueueReleaseGLObjects stubclEnqueueReleaseGLObjects
#define clGetGLContextInfoKHR stubclGetGLContextInfoKHR

/* cl_khr_command_buffer */
#define clCreateCommandBufferKHR stubclCreateCommandBufferKHR
#define clRetainCommandBufferKHR stubclRetainCommandBufferKHR
#define clReleaseCommandBufferKHR stubclReleaseCommandBufferKHR
#define clGetCommandBufferInfoKHR stubclGetCommandBufferInfoKHR
#define clEnqueueCommandBufferKHR stubclEnqueueCommandBufferKHR
#define clCommandBarrierWithWaitListKHR stubclCommandBarrierWithWaitListKHR
#define clCommandNDRangeKernelKHR stubclCommandNDRangeKernelKHR
#define clCommandCopyBufferKHR stubclCommandCopyBufferKHR
#define clCommandCopyBufferRectKHR stubclCommandCopyBufferRectKHR
#define clCommandCopyBufferToImageKHR stubclCommandCopyBufferToImageKHR
#define clCommandCopyImageKHR stubclCommandCopyImageKHR
#define clCommandCopyImageToBufferKHR stubclCommandCopyImageToBufferKHR
#define clCommandFillBufferKHR stubclCommandFillBufferKHR
#define clCommandFillImageKHR stubclCommandFillImageKHR

#endif //_RENAME_STUB_H_
