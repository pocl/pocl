#include "pocl_cl.h"
CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateFromGLTexture2D)(cl_context      context,
                        cl_mem_flags    flags,
                        cl_GLenum       target,
                        cl_GLint        miplevel,
                        cl_GLuint       texture,
                        cl_int *        errcode_ret) 
CL_API_SUFFIX__VERSION_1_0
{
  POCL_ABORT_UNIMPLEMENTED("The entire clCreateFromGLTexture2D call");
  return NULL;
}
POsym(clCreateFromGLTexture2D)
