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
  void *clCreateContext;  // 5th
  void *clCreateCommandQueue;
  void *clCreateBuffer;
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
  (void *)&clCreateContext,	\
  (void *)&clCreateCommandQueue,	\
  (void *)&clCreateBuffer,	\
  (void *)&clCreateContextFromType,	\
  (void *)&clCreateImage2D,	\
  (void *)&clCreateImage3D,	\
  (void *)&clCreateKernel,	\
  (void *)&clCreateKernelsInProgram,	\
  (void *)&clCreateProgramWithBinary,	\
  (void *)&clCreateProgramWithSource,	\
  (void *)&clCreateSampler,	\
  (void *)&clCreateSubBuffer,	\
  (void *)&clCreateUserEvent,	\
  (void *)&clEnqueueBarrier,	\
  (void *)&clEnqueueCopyBuffer,	\
  (void *)&clEnqueueCopyBufferRect,	\
  (void *)&clEnqueueCopyBufferToImage,	\
  (void *)&clEnqueueCopyImage,	\
  (void *)&clEnqueueCopyImageToBuffer,	\
  (void *)&clEnqueueMapBuffer,	\
  (void *)&clEnqueueMapImage,	\
  (void *)&clEnqueueMarker,	\
  (void *)&clEnqueueNDRangeKernel,	\
  (void *)&clEnqueueNativeKernel,	\
  (void *)&clEnqueueReadBuffer,	\
  (void *)&clEnqueueReadBufferRect,	\
  (void *)&clEnqueueReadImage,	\
  (void *)&clEnqueueTask,	\
  (void *)&clEnqueueUnmapMemObject,	\
  (void *)&clEnqueueWaitForEvents,	\
  (void *)&clEnqueueWriteBuffer,	\
  (void *)&clEnqueueWriteBufferRect,	\
  (void *)&clEnqueueWriteImage,	\
  (void *)&clFinish,	\
  (void *)&clFlush,	\
  (void *)&clGetCommandQueueInfo,	\
  (void *)&clGetContextInfo,	\
  (void *)&clGetEventInfo,	\
  (void *)&clGetEventProfilingInfo,	\
  (void *)&clGetExtensionFunctionAddress,	\
  (void *)&clGetImageInfo,	\
  (void *)&clGetKernelInfo,	\
  (void *)&clGetKernelWorkGroupInfo,	\
  (void *)&clGetMemObjectInfo,	\
  (void *)&clGetPlatformIDs,	\
  (void *)&clGetProgramBuildInfo,	\
  (void *)&clGetProgramInfo,	\
  (void *)&clGetSamplerInfo,	\
  (void *)&clGetSupportedImageFormats,	\
  (void *)&clReleaseCommandQueue,	\
  (void *)&clReleaseContext,	\
  (void *)&clReleaseEvent,	\
  (void *)&clReleaseKernel,	\
  (void *)&clReleaseMemObject,	\
  (void *)&clReleaseProgram,	\
  (void *)&clReleaseSampler,	\
  (void *)&clRetainCommandQueue,	\
  (void *)&clRetainContext,	\
  (void *)&clRetainEvent,	\
  (void *)&clRetainKernel,	\
  (void *)&clRetainMemObject,	\
  (void *)&clRetainProgram,	\
  (void *)&clRetainSampler,	\
  (void *)&clSetEventCallback,	\
  (void *)&clSetKernelArg,	\
  (void *)&clSetMemObjectDestructorCallback,	\
  (void *)&clSetUserEventStatus,	\
  (void *)&clUnloadCompiler,	\
  (void *)&clWaitForEvents	\
}

#endif

