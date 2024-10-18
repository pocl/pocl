
#include "pocl_cl.h"

CL_API_ENTRY cl_mem CL_API_CALL
POname(clCreateFromGLBuffer)(cl_context      context,
                        cl_mem_flags    flags,
                        cl_GLuint       bufobj,
                        cl_int *        errcode_ret)
CL_API_SUFFIX__VERSION_1_0
{
  POCL_MSG_ERR ("The clCreateFromGLBuffer API is not implemented\n");
  if (errcode_ret)
    *errcode_ret = CL_INVALID_OPERATION;
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
  POCL_MSG_ERR ("The clCreateFromGLRenderbuffer API is not implemented\n");
  if (errcode_ret)
    *errcode_ret = CL_INVALID_OPERATION;
  return NULL;
}
POsym(clCreateFromGLRenderbuffer)



CL_API_ENTRY cl_int CL_API_CALL
POname(clGetGLObjectInfo)(cl_mem        memobj,
                          cl_gl_object_type *gl_object_type,
                          cl_GLuint       *gl_object_name)
CL_API_SUFFIX__VERSION_1_0
{
  POCL_MSG_ERR ("The clGetGLObjectInfo API is not implemented\n");
  return CL_INVALID_OPERATION;
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
  POCL_MSG_ERR ("The clGetGLTextureInfo API is not implemented\n");
  return CL_INVALID_OPERATION;
}
POsym(clGetGLTextureInfo)
