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
  void *clEnqueueReadImage;
  void *clGetPlatformInfo;	// 2nd
  void *clGetDeviceIDs;  	// 3rd
  void *clGetDeviceInfo;	// 4th
  void *clCreateContext;  // 5th
  void *clCreateImage3D;
  void *clCreateSubBuffer;
  void *clCreateContextFromType;
  void *clGetContextInfo; // correct
  void *clCreateCommandQueue; // correct
  void *clGetProgramBuildInfo;
  void *clReleaseCommandQueue; //correct
  void *clCreateProgramWithBinary;
  void *clCreateSampler;
  void *clCreateBuffer;  // correct
  void *clCreateUserEvent;
  void *clEnqueueBarrier;
  void *clEnqueueCopyBuffer;
  void *clReleaseMemObject; //correct
  void *clEnqueueCopyBufferToImage;
  void *clEnqueueCopyImage;
  void *clEnqueueCopyImageToBuffer;
  void *clEnqueueMapBuffer;
  void *clEnqueueMapImage;
  void *clEnqueueMarker;
  void *clReleaseSampler;
  void *clCreateProgramWithSource; //correct
  void *clEnqueueNativeKernel;
  void *clGetPlatformIDs;
  void *clReleaseProgram; // correct
  void *clBuildProgram;   // correct
  void *clEnqueueTask;
  void *clEnqueueUnmapMemObject;
  void *clEnqueueWaitForEvents;
  void *clCreateKernel;   // correct
  void *clEnqueueWriteBufferRect;
  void *clEnqueueWriteImage;
  void *clReleaseKernel; // correct
  void *clSetKernelArg;  // correct
  void *clGetCommandQueueInfo;
  void *clCreateImage2D;
  void *clWaitForEvents; // correct
  void *clGetEventProfilingInfo;
  void *clGetExtensionFunctionAddress;
  void *clGetImageInfo;
  void *clGetKernelInfo;
  void *clGetKernelWorkGroupInfo;
  void *clGetMemObjectInfo;
  void *clEnqueueReadBuffer; //correct
  void *clEnqueueWriteBuffer; // correct
  void *clGetProgramInfo;
  void *clGetSamplerInfo;
  void *clGetSupportedImageFormats;
  void *clCreateKernelsInProgram;
  void *clReleaseContext;
  void *clReleaseEvent;
  void *clFinish;
  void *clEnqueueCopyBufferRect;
  void *clEnqueueReadBufferRect;
  void *clEnqueueNDRangeKernel; //correct
  void *clRetainCommandQueue;
  void *clRetainContext;
  void *clRetainEvent;
  void *clRetainKernel;
  void *clRetainMemObject;
  void *clRetainProgram;
  void *clRetainSampler;
  void *clSetEventCallback;
  void *clFlush;
  void *clSetMemObjectDestructorCallback;
  void *clSetUserEventStatus;
  void *clUnloadCompiler;
  void *clGetEventInfo;
};

/* The "implementation" of the _cl_device_id struct. 
 * Instantiated in clGetPlatformIDs.c
 *
 * TODO: the NULL entries are functions that lack implementation (or even stubs) in pocl
 */
#define POCL_ICD_DISPATCH { \
  (void *)&clEnqueueReadImage,	\
  (void *)&clGetPlatformInfo,	\
  (void *)&clGetDeviceIDs,	\
  (void *)&clGetDeviceInfo,	\
  (void *)&clCreateContext,	\
  (void *)&clCreateImage3D,	\
  (void *)&clCreateSubBuffer,	\
  (void *)&clCreateContextFromType,	\
  (void *)&clGetContextInfo,	\
  (void *)&clCreateCommandQueue,	\
  (void *)&clGetProgramBuildInfo,	\
  (void *)&clReleaseCommandQueue,	\
  (void *)&clCreateProgramWithBinary,	\
  (void *)&clCreateSampler,	\
  (void *)&clCreateBuffer,	\
  (void *)&clCreateUserEvent,	\
  (void *)&clEnqueueBarrier,	\
  (void *)&clEnqueueCopyBuffer,	\
  (void *)&clReleaseMemObject,	\
  (void *)&clEnqueueCopyBufferToImage,	\
  (void *)&clEnqueueCopyImage,	\
  (void *)&clEnqueueCopyImageToBuffer,	\
  (void *)&clEnqueueMapBuffer,	\
  (void *)&clEnqueueMapImage,	\
  (void *)&clEnqueueMarker,	\
  (void *)&clReleaseSampler,	\
  (void *)&clCreateProgramWithSource,	\
  (void *)&clEnqueueNativeKernel,	\
  (void *)&clGetPlatformIDs,	\
  (void *)&clReleaseProgram,	\
  (void *)&clBuildProgram,	\
  (void *)&clEnqueueTask,	\
  (void *)&clEnqueueUnmapMemObject,	\
  (void *)&clEnqueueWaitForEvents,	\
  (void *)&clCreateKernel,	\
  (void *)&clEnqueueWriteBufferRect,	\
  (void *)&clEnqueueWriteImage,	\
  (void *)&clReleaseKernel,	\
  (void *)&clSetKernelArg,	\
  (void *)&clGetCommandQueueInfo,	\
  (void *)&clCreateImage2D,	\
  (void *)&clWaitForEvents,	\
  (void *)&clGetEventProfilingInfo,	\
  (void *)&clGetExtensionFunctionAddress,	\
  (void *)&clGetImageInfo,	\
  (void *)&clGetKernelInfo,	\
  (void *)&clGetKernelWorkGroupInfo,	\
  (void *)&clGetMemObjectInfo,	\
  (void *)&clEnqueueReadBuffer,	\
  (void *)&clEnqueueWriteBuffer,	\
  (void *)&clGetProgramInfo,	\
  (void *)&clGetSamplerInfo,	\
  (void *)&clGetSupportedImageFormats,	\
  (void *)&clCreateKernelsInProgram,	\
  (void *)&clReleaseContext,	\
  (void *)&clReleaseEvent,	\
  (void *)&clFinish,	\
  (void *)&clEnqueueCopyBufferRect,	\
  (void *)&clEnqueueReadBufferRect,	\
  (void *)&clEnqueueNDRangeKernel,	\
  (void *)&clRetainCommandQueue,	\
  (void *)&clRetainContext,	\
  (void *)&clRetainEvent,	\
  (void *)&clRetainKernel,	\
  (void *)&clRetainMemObject,	\
  (void *)&clRetainProgram,	\
  (void *)&clRetainSampler,	\
  (void *)&clSetEventCallback,	\
  (void *)&clFlush,	\
  (void *)&clSetMemObjectDestructorCallback,	\
  (void *)&clSetUserEventStatus,	\
  (void *)&clUnloadCompiler,	\
  (void *)&clGetEventInfo	\
}

#endif
#endif

