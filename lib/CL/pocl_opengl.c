
#include "pocl_cl.h"

CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateFromGLTexture)(cl_context      context,
                        cl_mem_flags    flags,
                        cl_GLenum       texture_target,
                        cl_GLint        miplevel,
                        cl_GLuint       texture,
                        cl_int *        errcode_ret)
CL_API_SUFFIX__VERSION_1_2
{
  POCL_ABORT_UNIMPLEMENTED("The entire clCreateFromGLTexture call");
  return NULL;
}
POsym(clCreateFromGLTexture)



CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateFromGLBuffer)(cl_context      context,
                        cl_mem_flags    flags,
                        cl_GLuint       bufobj,
                        cl_int *        errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED("The entire clCreateFromGLBuffer call");
  return NULL;
}
POsym(clCreateFromGLBuffer)



CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateFromGLRenderbuffer)(cl_context      context,
                        cl_mem_flags    flags,
                        cl_GLuint       renderbuffer,
                        cl_int *        errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED("The entire clCreateFromGLRenderbuffer call");
  return NULL;
}
POsym(clCreateFromGLRenderbuffer)



CL_API_ENTRY cl_int CL_API_CALL
POname(clGetGLObjectInfo)(cl_mem        memobj,
                          cl_gl_object_type *gl_object_type,
                          cl_GLuint       *gl_object_name)
CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED("The entire clGetGLObjectInfo call");
  return CL_OUT_OF_RESOURCES;
}
POsym(clGetGLObjectInfo)




CL_API_ENTRY cl_int CL_API_CALL
POname(clGetGLTextureInfo) (cl_mem        memobj,
                            cl_gl_texture_info param_name,
                            size_t  param_value_size,
                            void  *param_value,
                            size_t  *param_value_size_ret)
CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED("The entire clGetGLTextureInfo call");
  return CL_OUT_OF_RESOURCES;
}
POsym(clGetGLTextureInfo)



CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueAcquireGLObjects) ( cl_command_queue command_queue,
                                    cl_uint num_objects,
                                    const cl_mem *mem_objects,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event *event_wait_list,
                                    cl_event *event)
CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED("The entire clEnqueueAcquireGLObjects call");
  return CL_OUT_OF_RESOURCES;
}
POsym(clEnqueueAcquireGLObjects)



CL_API_ENTRY cl_int CL_API_CALL
POname(clEnqueueReleaseGLObjects) ( cl_command_queue command_queue,
                                    cl_uint num_objects,
                                    const cl_mem *mem_objects,
                                    cl_uint num_events_in_wait_list,
                                    const cl_event *event_wait_list,
                                    cl_event *event)
CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED("The entire clEnqueueReleaseGLObjects call");
  return CL_OUT_OF_RESOURCES;
}
POsym(clEnqueueReleaseGLObjects)



CL_API_ENTRY cl_int CL_API_CALL
POname(clGetGLContextInfoKHR) ( const cl_context_properties  *properties ,
  cl_gl_context_info  param_name ,
  size_t  param_value_size ,
  void  *param_value ,
  size_t  *param_value_size_ret )

CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED("The entire clGetGLContextInfoKHR call");
  return CL_OUT_OF_RESOURCES;
}
POsym(clGetGLContextInfoKHR)
