#include "config.h"

/* Installable Client Driver-realated things. */
#ifndef POCL_ICD_H
#define POCL_ICD_H

// stub out ICD related stuff 
#ifndef BUILD_ICD

#  define POCL_DEVICE_ICD_DISPATCH
#  define POCL_INIT_ICD_OBJECT(__obj__)

// rest of the file: ICD is enabled 
#else

// this define is a kludge!
// The ICD loaders seem to require OCL 1.1, so we cannot (can we?) leave deprecated 
// functions out
// Answer: not really. ICD loader will call OCL 1.1 function throught the
// function table, but the registered function can be then only stubs
// (perhaps with a warning) or even NULL (in this case, a program using
// OCL 1.1 function will crash: ICD Loaders does not do any check)
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#ifndef CL_USE_DEPRECATED_OPENCL_1_1_APIS
#error CL_USE_DEPRECATED_OPENCL_1_1_APIS not in use
#endif

#pragma GCC visibility push(hidden)
extern struct _cl_icd_dispatch pocl_dispatch;  //from clGetPlatformIDs.c
#pragma GCC visibility pop

#define POCL_DEVICE_ICD_DISPATCH &pocl_dispatch,
#define POCL_INIT_ICD_OBJECT(__obj__) (__obj__)->dispatch=&pocl_dispatch

// Get the structure struct _cl_icd_dispatch from the OCL-ICD project
// (that provides a free OCL ICD Loader)
#include <ocl_icd.h>

/* The "implementation" of the _cl_device_id struct. 
 * Instantiated in clGetPlatformIDs.c
 *
 * TODO: the NULL entries are functions that lack implementation (or even stubs) in pocl
 */
#define POCL_ICD_DISPATCH {           \
  &POclGetPlatformIDs,          \
  &POclGetPlatformInfo,         \
  &POclGetDeviceIDs,            \
  &POclGetDeviceInfo,           \
  &POclCreateContext,           \
  &POclCreateContextFromType,   \
  &POclRetainContext,           \
  &POclReleaseContext,          \
  &POclGetContextInfo,          \
  &POclCreateCommandQueue,      \
  &POclRetainCommandQueue,      \
  &POclReleaseCommandQueue,     \
  &POclGetCommandQueueInfo,     \
  NULL /*clSetCommandQueueProperty*/, \
  &POclCreateBuffer,            \
  &POclCreateImage2D,           \
  &POclCreateImage3D,           \
  &POclRetainMemObject,         \
  &POclReleaseMemObject,        \
  &POclGetSupportedImageFormats,\
  &POclGetMemObjectInfo,        \
  &POclGetImageInfo,            \
  &POclCreateSampler,           \
  &POclRetainSampler,           \
  &POclReleaseSampler,          \
  &POclGetSamplerInfo,          \
  &POclCreateProgramWithSource, \
  &POclCreateProgramWithBinary, \
  &POclRetainProgram,           \
  &POclReleaseProgram,          \
  &POclBuildProgram,            \
  &POclUnloadCompiler,          \
  &POclGetProgramInfo,          \
  &POclGetProgramBuildInfo,     \
  &POclCreateKernel,            \
  &POclCreateKernelsInProgram,  \
  &POclRetainKernel,            \
  &POclReleaseKernel,           \
  &POclSetKernelArg,            \
  &POclGetKernelInfo,           \
  &POclGetKernelWorkGroupInfo,  \
  &POclWaitForEvents,           \
  &POclGetEventInfo,            \
  &POclRetainEvent,             \
  &POclReleaseEvent,            \
  &POclGetEventProfilingInfo,   \
  &POclFlush,                   \
  &POclFinish,                  \
  &POclEnqueueReadBuffer,       \
  &POclEnqueueWriteBuffer,      \
  &POclEnqueueCopyBuffer,       \
  &POclEnqueueReadImage,        \
  &POclEnqueueWriteImage,       \
  &POclEnqueueCopyImage,        \
  &POclEnqueueCopyImageToBuffer,\
  &POclEnqueueCopyBufferToImage,\
  &POclEnqueueMapBuffer,        \
  &POclEnqueueMapImage,         \
  &POclEnqueueUnmapMemObject,   \
  &POclEnqueueNDRangeKernel,    \
  &POclEnqueueTask,             \
  &POclEnqueueNativeKernel,     \
  &POclEnqueueMarker,           \
  &POclEnqueueWaitForEvents,    \
  &POclEnqueueBarrier,          \
  &POclGetExtensionFunctionAddress, \
  NULL, /* &POclCreateFromGLBuffer,      */ \
  &POclCreateFromGLTexture2D,   \
  &POclCreateFromGLTexture3D,   \
  NULL, /* &POclCreateFromGLRenderbuffer, */ \
  NULL, /* &POclGetGLObjectInfo,         */ \
  NULL, /* &POclGetGLTextureInfo,        */ \
  NULL, /* &POclEnqueueAcquireGLObjects, */ \
  NULL, /* &POclEnqueueReleaseGLObjects, */ \
  NULL, /* &POclGetGLContextInfoKHR,     */ \
  NULL, /* &clUnknown75 */      \
  NULL, /* &clUnknown76 */      \
  NULL, /* &clUnknown77 */      \
  NULL, /* &clUnknown78 */      \
  NULL, /* &clUnknown79 */      \
  NULL, /* &clUnknown80 */      \
  &POclSetEventCallback,        \
  &POclCreateSubBuffer,         \
  &POclSetMemObjectDestructorCallback, \
  &POclCreateUserEvent,         \
  &POclSetUserEventStatus,      \
  &POclEnqueueReadBufferRect,   \
  &POclEnqueueWriteBufferRect,  \
  &POclEnqueueCopyBufferRect,   \
  NULL, /* &POclCreateSubDevicesEXT,     */ \
  NULL, /* &POclRetainDeviceEXT,         */ \
  NULL, /* &POclReleaseDeviceEXT,        */ \
  NULL, /* &clUnknown92 */      \
  NULL, /* &POclCreateSubDevices,        */ \
  NULL, /* &POclRetainDevice,            */ \
  NULL, /* &POclReleaseDevice,           */ \
  &POclCreateImage,             \
  NULL, /* &POclCreateProgramWithBuiltInKernels, */ \
  NULL, /* &POclCompileProgram,          */ \
  NULL, /* &POclLinkProgram,             */ \
  NULL, /* &POclUnloadPlatformCompiler,  */ \
  NULL, /* &POclGetKernelArgInfo,        */ \
  NULL, /* &POclEnqueueFillBuffer,       */ \
  NULL, /* &POclEnqueueFillImage,        */ \
  NULL, /* &POclEnqueueMigrateMemObjects, */ \
  NULL, /* &POclEnqueueMarkerWithWaitList, */ \
  NULL, /* &POclEnqueueBarrierWithWaitList, */ \
  NULL, /* &POclGetExtensionFunctionAddressForPlatform, */ \
  NULL, /* &POclCreateFromGLTexture,     */ \
}

#endif
#endif

