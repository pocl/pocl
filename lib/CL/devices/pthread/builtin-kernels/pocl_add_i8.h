#ifndef POCL_PTHREAD_ADD_I8_H
#define POCL_PTHREAD_ADD_I8_H

#include <CL/cl.h>
#include <pocl_cl.h>
#include <pocl_types.h>

#ifdef __cplusplus
extern "C"
{
#endif

  POCL_EXPORT
  void init_pocl_add_i8 ();

  POCL_EXPORT
  void free_pocl_add_i8 ();

  POCL_EXPORT
  void _pocl_kernel_pocl_add_i8_workgroup (cl_uchar *args,
                                           cl_uchar *context,
                                           ulong group_x,
                                           ulong group_y,
                                           ulong group_z);

#ifdef __cplusplus
}
#endif

#endif // POCL_PTHREAD_ADD_I8_H
