/* rename_opencl.h - a file that renames OpenCL calls to PoCL-specific calls
   This is required to use the proxy driver. For more info, please look into
   doc/sphinx/source/proxy.rst

   Copyright (c) 2021 Michal Babej / Tampere University

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

#ifndef RENAME_OPENCL_POCL_H
#define RENAME_OPENCL_POCL_H

#define clGetPlatformIDs POclGetPlatformIDs
#define clGetPlatformInfo POclGetPlatformInfo
#define clGetDeviceIDs POclGetDeviceIDs
#define clGetDeviceInfo POclGetDeviceInfo
#define clCreateSubDevices POclCreateSubDevices
#define clRetainDevice POclRetainDevice
#define clReleaseDevice POclReleaseDevice
#define clCreateContext POclCreateContext
#define clCreateContextFromType POclCreateContextFromType
#define clRetainContext POclRetainContext
#define clReleaseContext POclReleaseContext
#define clGetContextInfo POclGetContextInfo
#define clCreateCommandQueueWithProperties POclCreateCommandQueueWithProperties
#define clRetainCommandQueue POclRetainCommandQueue
#define clReleaseCommandQueue POclReleaseCommandQueue
#define clGetCommandQueueInfo POclGetCommandQueueInfo
#define clCreateBuffer POclCreateBuffer
#define clCreateSubBuffer POclCreateSubBuffer
#define clCreateImage POclCreateImage
#define clCreatePipe POclCreatePipe
#define clRetainMemObject POclRetainMemObject
#define clReleaseMemObject POclReleaseMemObject
#define clGetSupportedImageFormats POclGetSupportedImageFormats
#define clGetMemObjectInfo POclGetMemObjectInfo
#define clGetImageInfo POclGetImageInfo
#define clGetPipeInfo POclGetPipeInfo
#define clSetMemObjectDestructorCallback POclSetMemObjectDestructorCallback
#define clSVMAlloc POclSVMAlloc
#define clSVMFree POclSVMFree
#define clCreateSamplerWithProperties POclCreateSamplerWithProperties
#define clRetainSampler POclRetainSampler
#define clReleaseSampler POclReleaseSampler
#define clGetSamplerInfo POclGetSamplerInfo
#define clCreateProgramWithSource POclCreateProgramWithSource
#define clCreateProgramWithBinary POclCreateProgramWithBinary
#define clCreateProgramWithIL POclCreateProgramWithIL
#define clCreateProgramWithBuiltInKernels POclCreateProgramWithBuiltInKernels
#define clRetainProgram POclRetainProgram
#define clReleaseProgram POclReleaseProgram
#define clBuildProgram POclBuildProgram
#define clCompileProgram POclCompileProgram
#define clLinkProgram POclLinkProgram
#define clUnloadPlatformCompiler POclUnloadPlatformCompiler
#define clGetProgramInfo POclGetProgramInfo
#define clGetProgramBuildInfo POclGetProgramBuildInfo
#define clCreateKernel POclCreateKernel
#define clCreateKernelsInProgram POclCreateKernelsInProgram
#define clRetainKernel POclRetainKernel
#define clReleaseKernel POclReleaseKernel
#define clSetKernelArg POclSetKernelArg
#define clSetKernelArgSVMPointer POclSetKernelArgSVMPointer
#define clSetKernelExecInfo POclSetKernelExecInfo
#define clGetKernelInfo POclGetKernelInfo
#define clGetKernelArgInfo POclGetKernelArgInfo
#define clGetKernelWorkGroupInfo POclGetKernelWorkGroupInfo
#define clWaitForEvents POclWaitForEvents
#define clGetEventInfo POclGetEventInfo
#define clCreateUserEvent POclCreateUserEvent
#define clRetainEvent POclRetainEvent
#define clReleaseEvent POclReleaseEvent
#define clSetUserEventStatus POclSetUserEventStatus
#define clSetEventCallback POclSetEventCallback
#define clGetEventProfilingInfo POclGetEventProfilingInfo
#define clFlush POclFlush
#define clFinish POclFinish
#define clEnqueueReadBuffer POclEnqueueReadBuffer
#define clEnqueueReadBufferRect POclEnqueueReadBufferRect
#define clEnqueueWriteBuffer POclEnqueueWriteBuffer
#define clEnqueueWriteBufferRect POclEnqueueWriteBufferRect
#define clEnqueueFillBuffer POclEnqueueFillBuffer
#define clEnqueueCopyBuffer POclEnqueueCopyBuffer
#define clEnqueueCopyBufferRect POclEnqueueCopyBufferRect
#define clEnqueueReadImage POclEnqueueReadImage
#define clEnqueueWriteImage POclEnqueueWriteImage
#define clEnqueueFillImage POclEnqueueFillImage
#define clEnqueueCopyImage POclEnqueueCopyImage
#define clEnqueueCopyImageToBuffer POclEnqueueCopyImageToBuffer
#define clEnqueueCopyBufferToImage POclEnqueueCopyBufferToImage
#define clEnqueueMapBuffer POclEnqueueMapBuffer
#define clEnqueueMapImage POclEnqueueMapImage
#define clEnqueueUnmapMemObject POclEnqueueUnmapMemObject
#define clEnqueueMigrateMemObjects POclEnqueueMigrateMemObjects
#define clEnqueueNDRangeKernel POclEnqueueNDRangeKernel
#define clEnqueueNativeKernel POclEnqueueNativeKernel
#define clEnqueueMarkerWithWaitList POclEnqueueMarkerWithWaitList
#define clEnqueueBarrierWithWaitList POclEnqueueBarrierWithWaitList
#define clEnqueueSVMFree POclEnqueueSVMFree
#define clEnqueueSVMMemcpy POclEnqueueSVMMemcpy
#define clEnqueueSVMMemFill POclEnqueueSVMMemFill
#define clEnqueueSVMMap POclEnqueueSVMMap
#define clEnqueueSVMUnmap POclEnqueueSVMUnmap
#define clGetExtensionFunctionAddressForPlatform POclGetExtensionFunctionAddressForPlatform
#define clCreateImage2D POclCreateImage2D
#define clCreateImage3D POclCreateImage3D
#define clEnqueueMarker POclEnqueueMarker
#define clEnqueueWaitForEvents POclEnqueueWaitForEvents
#define clEnqueueBarrier POclEnqueueBarrier
#define clUnloadCompiler POclUnloadCompiler
#define clGetExtensionFunctionAddress POclGetExtensionFunctionAddress
#define clCreateCommandQueue POclCreateCommandQueue
#define clCreateSampler POclCreateSampler
#define clEnqueueTask POclEnqueueTask

#define clCreateFromGLTexture POclCreateFromGLTexture
#define clCreateFromGLTexture2D POclCreateFromGLTexture2D
#define clCreateFromGLTexture3D POclCreateFromGLTexture3D
#define clEnqueueAcquireGLObjects POclEnqueueAcquireGLObjects
#define clEnqueueReleaseGLObjects POclEnqueueReleaseGLObjects
#define clGetGLContextInfoKHR POclGetGLContextInfoKHR

/* cl_khr_command_buffer */
#define clCreateCommandBufferKHR POclCreateCommandBufferKHR
#define clRetainCommandBufferKHR POclRetainCommandBufferKHR
#define clReleaseCommandBufferKHR POclReleaseCommandBufferKHR
#define clGetCommandBufferInfoKHR POclGetCommandBufferInfoKHR
#define clEnqueueCommandBufferKHR POclEnqueueCommandBufferKHR
#define clCommandBarrierWithWaitListKHR POclCommandBarrierWithWaitListKHR
#define clCommandNDRangeKernelKHR POclCommandNDRangeKernelKHR
#define clCommandCopyBufferKHR POclCommandCopyBufferKHR
#define clCommandCopyBufferRectKHR POclCommandCopyBufferRectKHR
#define clCommandCopyBufferToImageKHR POclCommandCopyBufferToImageKHR
#define clCommandCopyImageKHR POclCommandCopyImageKHR
#define clCommandCopyImageToBufferKHR POclCommandCopyImageToBufferKHR
#define clCommandFillBufferKHR POclCommandFillBufferKHR
#define clCommandFillImageKHR POclCommandFillImageKHR

#endif
