#include "config.h"

/* Installable Client Driver-realated things. */
#ifndef POCL_ICD_H
#define POCL_ICD_H

// stub out ICD related stuff 
#ifndef BUILD_ICD
#define POCL_DEVICE_ICD_DISPATCH
#define POCL_INIT_ICD_OBJECT(__obj__)

// rest of the file: ICD is enabled 
#else

// this define is a kludge!
// The ICD loaders seem to require OCL 1.1, so we cannot (can we?) leave deprecated 
// functions out
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#ifndef CL_USE_DEPRECATED_OPENCL_1_1_APIS
#error CL_USE_DEPRECATED_OPENCL_1_1_APIS not in use
#endif

#include "pocl_cl.h"

extern struct _cl_icd_dispatch pocl_dispatch;  //from clGetPlatformIDs.c
#define POCL_DEVICE_ICD_DISPATCH &pocl_dispatch,
#define POCL_INIT_ICD_OBJECT(__obj__) (__obj__)->dispatch=&pocl_dispatch

// TODO: Add functions from OCL 1.2
/* Correct order of these functions is specified in the OPEN CL ICD extension example code, 
 * which is available only to Khronos members. TODO: dig this order out from somewhere. 
 * A few of these orders are reversed by trial and error. Those are marked.
 */
struct _cl_icd_dispatch {
  void *clGetPlatformIDs;
  void *clGetPlatformInfo;
  void *clGetDeviceIDs;
  void *clGetDeviceInfo;
  void *clCreateContext;
  void *clCreateContextFromType;
  void *clRetainContext;
  void *clReleaseContext;
  void *clGetContextInfo;
  void *clCreateCommandQueue;
  void *clRetainCommandQueue;
  void *clReleaseCommandQueue;
  void *clGetCommandQueueInfo;
  void *clSetCommandQueueProperty;
  void *clCreateBuffer;
  void *clCreateImage2D;
  void *clCreateImage3D;
  void *clRetainMemObject;
  void *clReleaseMemObject;
  void *clGetSupportedImageFormats;
  void *clGetMemObjectInfo;
  void *clGetImageInfo;
  void *clCreateSampler;
  void *clRetainSampler;
  void *clReleaseSampler;
  void *clGetSamplerInfo;
  void *clCreateProgramWithSource;
  void *clCreateProgramWithBinary;
  void *clRetainProgram;
  void *clReleaseProgram;
  void *clBuildProgram;
  void *clUnloadCompiler;
  void *clGetProgramInfo;
  void *clGetProgramBuildInfo;
  void *clCreateKernel;
  void *clCreateKernelsInProgram;
  void *clRetainKernel;
  void *clReleaseKernel;
  void *clSetKernelArg;
  void *clGetKernelInfo;
  void *clGetKernelWorkGroupInfo;
  void *clWaitForEvents;
  void *clGetEventInfo;
  void *clRetainEvent;
  void *clReleaseEvent;
  void *clGetEventProfilingInfo;
  void *clFlush;
  void *clFinish;
  void *clEnqueueReadBuffer;
  void *clEnqueueWriteBuffer;
  void *clEnqueueCopyBuffer;
  void *clEnqueueReadImage;
  void *clEnqueueWriteImage;
  void *clEnqueueCopyImage;
  void *clEnqueueCopyImageToBuffer;
  void *clEnqueueCopyBufferToImage;
  void *clEnqueueMapBuffer;
  void *clEnqueueMapImage;
  void *clEnqueueUnmapMemObject;
  void *clEnqueueNDRangeKernel;
  void *clEnqueueTask;
  void *clEnqueueNativeKernel;
  void *clEnqueueMarker;
  void *clEnqueueWaitForEvents;
  void *clEnqueueBarrier;
  void *clSetEventCallback;
  void *clCreateSubBuffer;
  void *clSetMemObjectDestructorCallback;
  void *clCreateUserEvent;
  void *clSetUserEventStatus;
  void *clEnqueueReadBufferRect;
  void *clEnqueueWriteBufferRect;
  void *clEnqueueCopyBufferRect;
};

/* The "implementation" of the _cl_device_id struct. 
 * Instantiated in clGetPlatformIDs.c
 *
 * TODO: the NULL entries are functions that lack implementation (or even stubs) in pocl
 */
#define POCL_ICD_DISPATCH {           \
  (void *)&clGetPlatformIDs,          \
  (void *)&clGetPlatformInfo,         \
  (void *)&clGetDeviceIDs,            \
  (void *)&clGetDeviceInfo,           \
  (void *)&clCreateContext,           \
  (void *)&clCreateContextFromType,   \
  (void *)&clRetainContext,           \
  (void *)&clReleaseContext,          \
  (void *)&clGetContextInfo,          \
  (void *)&clCreateCommandQueue,      \
  (void *)&clRetainCommandQueue,      \
  (void *)&clReleaseCommandQueue,     \
  (void *)&clGetCommandQueueInfo,     \
  NULL /*clSetCommandQueueProperty*/, \
  (void *)&clCreateBuffer,            \
  (void *)&clCreateImage2D,           \
  (void *)&clCreateImage3D,           \
  (void *)&clRetainMemObject,         \
  (void *)&clReleaseMemObject,        \
  (void *)&clGetSupportedImageFormats,\
  (void *)&clGetMemObjectInfo,        \
  (void *)&clGetImageInfo,            \
  (void *)&clCreateSampler,           \
  (void *)&clRetainSampler,           \
  (void *)&clReleaseSampler,          \
  (void *)&clGetSamplerInfo,          \
  (void *)&clCreateProgramWithSource, \
  (void *)&clCreateProgramWithBinary, \
  (void *)&clRetainProgram,           \
  (void *)&clReleaseProgram,          \
  (void *)&clBuildProgram,            \
  (void *)&clUnloadCompiler,          \
  (void *)&clGetProgramInfo,          \
  (void *)&clGetProgramBuildInfo,     \
  (void *)&clCreateKernel,            \
  (void *)&clCreateKernelsInProgram,  \
  (void *)&clRetainKernel,            \
  (void *)&clReleaseKernel,           \
  (void *)&clSetKernelArg,            \
  (void *)&clGetKernelInfo,           \
  (void *)&clGetKernelWorkGroupInfo,  \
  (void *)&clWaitForEvents,           \
  (void *)&clGetEventInfo,            \
  (void *)&clRetainEvent,             \
  (void *)&clReleaseEvent,            \
  (void *)&clGetEventProfilingInfo,   \
  (void *)&clFlush,                   \
  (void *)&clFinish,                  \
  (void *)&clEnqueueReadBuffer,       \
  (void *)&clEnqueueWriteBuffer,      \
  (void *)&clEnqueueCopyBuffer,       \
  (void *)&clEnqueueReadImage,        \
  (void *)&clEnqueueWriteImage,       \
  (void *)&clEnqueueCopyImage,        \
  (void *)&clEnqueueCopyImageToBuffer,\
  (void *)&clEnqueueCopyBufferToImage,\
  (void *)&clEnqueueMapBuffer,        \
  (void *)&clEnqueueMapImage,         \
  (void *)&clEnqueueUnmapMemObject,   \
  (void *)&clEnqueueNDRangeKernel,    \
  (void *)&clEnqueueTask,             \
  (void *)&clEnqueueNativeKernel,     \
  (void *)&clEnqueueMarker,           \
  (void *)&clEnqueueWaitForEvents,    \
  (void *)&clEnqueueBarrier,          \
  (void *)&clSetEventCallback,        \
  (void *)&clCreateSubBuffer,         \
  (void *)&clSetMemObjectDestructorCallback, \
  (void *)&clCreateUserEvent,         \
  (void *)&clSetUserEventStatus,      \
  (void *)&clEnqueueReadBufferRect,   \
  (void *)&clEnqueueWriteBufferRect,  \
  (void *)&clEnqueueCopyBufferRect    \
}

#endif
#endif

