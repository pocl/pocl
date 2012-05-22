/* Installable Client Driver-realated things. */
#ifndef POCL_ICD_H
#define POCL_ICD_H

// this define is a kludge!
// The ICD loaders seem to require OCL 1.1, so we cannot (can we?) leave deprecated 
// functions out
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#ifndef CL_USE_DEPRECATED_OPENCL_1_1_APIS
#error CL_USE_DEPRECATED_OPENCL_1_1_APIS not in use
#endif

#include "pocl_cl.h"

#ifdef BUILD_ICD
extern struct _cl_icd_dispatch pocl_dispatch;  //from clGetPlatformIDs.c
#define POCL_DEVICE_ICD_DISPATCH &pocl_dispatch,
#else
#define POCL_DEVICE_ICD_DISPATCH
#endif

// TODO: Add functions from OCL 1.2
/* Correct order of these functions is specified in the OPEN CL ICD extension example code, 
 * which is available only to Khronos members. TODO: dig this order out from somewhere. 
 * A few of these orders are reversed by trial and error. Those are marked.
 */
struct _cl_icd_dispatch {
  void *clBuildProgram;
  void *clGetPlatformInfo;	// 2nd
  void *clGetDeviceIDs;  	// 3rd
  void *clGetDeviceInfo;	// 4th
  void *clCreateBuffer;
  void *clCreateCommandQueue;
  void *clCreateContext;
  void *clCreateContextFromType;
  void *clCreateImage2D;
  void *clCreateImage3D;
  void *clCreateKernel;
  void *clCreateKernelsInProgram;
  void *clCreateProgramWithBinary;
  void *clCreateProgramWithSource;
  void *clCreateSampler;
  void *clCreateSubBuffer;
  void *clCreateUserEvent;
  void *clEnqueueBarrier;
  void *clEnqueueCopyBuffer;
  void *clEnqueueCopyBufferRect;
  void *clEnqueueCopyBufferToImage;
  void *clEnqueueCopyImage;
  void *clEnqueueCopyImageToBuffer;
  void *clEnqueueMapBuffer;
  void *clEnqueueMapImage;
  void *clEnqueueMarker;
  void *clEnqueueNDRangeKernel;
  void *clEnqueueNativeKernel;
  void *clEnqueueReadBuffer;
  void *clEnqueueReadBufferRect;
  void *clEnqueueReadImage;
  void *clEnqueueTask;
  void *clEnqueueUnmapMemObject;
  void *clEnqueueWaitForEvents;
  void *clEnqueueWriteBuffer;
  void *clEnqueueWriteBufferRect;
  void *clEnqueueWriteImage;
  void *clFinish;
  void *clFlush;
  void *clGetCommandQueueInfo;
  void *clGetContextInfo;
  void *clGetEventInfo;
  void *clGetEventProfilingInfo;
  void *clGetExtensionFunctionAddress;
  void *clGetImageInfo;
  void *clGetKernelInfo;
  void *clGetKernelWorkGroupInfo;
  void *clGetMemObjectInfo;
  void *clGetPlatformIDs;
  void *clGetProgramBuildInfo;
  void *clGetProgramInfo;
  void *clGetSamplerInfo;
  void *clGetSupportedImageFormats;
  void *clReleaseCommandQueue;
  void *clReleaseContext;
  void *clReleaseEvent;
  void *clReleaseKernel;
  void *clReleaseMemObject;
  void *clReleaseProgram;
  void *clReleaseSampler;
  void *clRetainCommandQueue;
  void *clRetainContext;
  void *clRetainEvent;
  void *clRetainKernel;
  void *clRetainMemObject;
  void *clRetainProgram;
  void *clRetainSampler;
  void *clSetEventCallback;
  void *clSetKernelArg;
  void *clSetMemObjectDestructorCallback;
  void *clSetUserEventStatus;
  void *clUnloadCompiler;
  void *clWaitForEvents;
};

/* The "implementation" of the _cl_device_id struct. 
 * Instantiated in clGetPlatformIDs.c
 *
 * TODO: the NULL entries are functions that lack implementation (or even stubs) in pocl
 */
#define POCL_ICD_DISPATCH { \
  (void *)&clBuildProgram,	\
  (void *)&clGetPlatformInfo,	\
  (void *)&clGetDeviceIDs,	\
  (void *)&clGetDeviceInfo,	\
  (void *)&clCreateBuffer,	\
  (void *)&clCreateCommandQueue,	\
  (void *)&clCreateContext,	\
  (void *)&clCreateContextFromType,	\
  NULL /*(void *)&clCreateImage2D*/,	\
  NULL /*(void *)&clCreateImage3D*/,	\
  (void *)&clCreateKernel,	\
  NULL /*(void *)&clCreateKernelsInProgram*/,	\
  (void *)&clCreateProgramWithBinary,	\
  (void *)&clCreateProgramWithSource,	\
  NULL /*(void *)&clCreateSampler*/,	\
  (void *)&clCreateSubBuffer,	\
  NULL /*(void *)&clCreateUserEvent*/,	\
  NULL /*(void *)&clEnqueueBarrier*/,	\
  (void *)&clEnqueueCopyBuffer,	\
  (void *)&clEnqueueCopyBufferRect,	\
  NULL /*(void *)&clEnqueueCopyBufferToImage*/,	\
  NULL /*(void *)&clEnqueueCopyImage*/,	\
  NULL /*(void *)&clEnqueueCopyImageToBuffer*/,	\
  (void *)&clEnqueueMapBuffer,	\
  NULL /*(void *)&clEnqueueMapImage*/,	\
  NULL /*(void *)&clEnqueueMarker*/,	\
  (void *)&clEnqueueNDRangeKernel,	\
  NULL /*(void *)&clEnqueueNativeKernel*/,	\
  (void *)&clEnqueueReadBuffer,	\
  (void *)&clEnqueueReadBufferRect,	\
  NULL /*(void *)&clEnqueueReadImage*/,	\
  (void *)&clEnqueueTask,	\
  (void *)&clEnqueueUnmapMemObject,	\
  NULL /*(void *)&clEnqueueWaitForEvents*/,	\
  (void *)&clEnqueueWriteBuffer,	\
  (void *)&clEnqueueWriteBufferRect,	\
  NULL /*(void *)&clEnqueueWriteImage*/,	\
  (void *)&clFinish,	\
  NULL/*(void *)&clFlush*/,	\
  NULL /*(void *)&clGetCommandQueueInfo*/,	\
  (void *)&clGetContextInfo,	\
  NULL /*(void *)&clGetEventInfo*/,	\
  (void *)&clGetEventProfilingInfo,	\
  (void *)&clGetExtensionFunctionAddress,	\
  NULL /*(void *)&clGetImageInfo*/,	\
  NULL /*(void *)&clGetKernelInfo*/,	\
  (void *)&clGetKernelWorkGroupInfo,	\
  NULL /*(void *)&clGetMemObjectInfo*/,	\
  (void *)&clGetPlatformIDs,	\
  (void *)&clGetProgramBuildInfo,	\
  (void *)&clGetProgramInfo,	\
  NULL /*(void *)&clGetSamplerInfo*/,	\
  NULL /*(void *)&clGetSupportedImageFormats*/,	\
  (void *)&clReleaseCommandQueue,	\
  (void *)&clReleaseContext,	\
  (void *)&clReleaseEvent,	\
  (void *)&clReleaseKernel,	\
  (void *)&clReleaseMemObject,	\
  (void *)&clReleaseProgram,	\
  NULL /*(void *)&clReleaseSampler*/,	\
  (void *)&clRetainCommandQueue,	\
  (void *)&clRetainContext,	\
  NULL /*(void *)&clRetainEvent*/,	\
  (void *)&clRetainKernel,	\
  (void *)&clRetainMemObject,	\
  (void *)&clRetainProgram,	\
  NULL /*(void *)&clRetainSampler*/,	\
  NULL /*(void *)&clSetEventCallback*/,	\
  (void *)&clSetKernelArg,	\
  NULL /*(void *)&clSetMemObjectDestructorCallback*/,	\
  NULL /*(void *)&clSetUserEventStatus*/,	\
  NULL /*(void *)&clUnloadCompiler*/,	\
  NULL /*(void *)&clWaitForEvents*/	\
}

#endif

