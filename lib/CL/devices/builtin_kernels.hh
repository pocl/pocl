#include "pocl_cl.h"
#include "pocl_export.h"

/*
  steps to add a new builtin kernel:

  1) add it to the end of BuiltinKernelId enum in this file

  2) open builtin_kernels.cc and edit BIDescriptors, add a new struct
     for the new kernel, with argument metadata

  3) make sure that devices where you want to support this builtin kernel,
     report it. Every driver does this a bit differently, but at pocl_XYZ_init
     it must properly fill dev->builtin_kernel_list, dev->num_builtin_kernels
     Note: the kernel name reported to user should use dots as separators
     (example: pocl.add.apples.to.oranges)

  4) add the code for the builtin kernel for each device that will support it.
     Note: if the builtin kernel is in source format, its name in the source
     MUST have the dots replaced with underscore
     (example: pocl_add_apples_to_oranges)

     How to do this, depends on device:
       * CUDA has OpenCL-source builtins in lib/CL/devices/cuda/builtins.cl,
         it also has CUDA-source builtins in lib/CL/devices/cuda/builtins.cu
       * accel driver with TTASIM backend has opencl-source builtins in
         lib/CL/devices/accel/tce_builtins.cl
       * accel driver with other backends has builtin kernels in binary format
  (bitstream)


*/

#ifndef POCL_BUILTIN_KERNELS_H
#define POCL_BUILTIN_KERNELS_H

#ifdef __cplusplus

#include <vector>

enum BuiltinKernelId : uint16_t
{
  // CD = custom device, BI = built-in
  // 1D array byte copy, get_global_size(0) defines the size of data to copy
  // kernel prototype: pocl.copy(char *input, char *output)
  POCL_CDBI_COPY_I8 = 0,
  POCL_CDBI_ADD_I32 = 1,
  POCL_CDBI_MUL_I32 = 2,
  POCL_CDBI_LEDBLINK = 3,
  POCL_CDBI_COUNTRED = 4,
  POCL_CDBI_DNN_CONV2D_RELU_I8 = 5,
  POCL_CDBI_SGEMM_LOCAL_F32 = 6,
  POCL_CDBI_SGEMM_TENSOR_F16F16F32_SCALE = 7,
  POCL_CDBI_SGEMM_TENSOR_F16F16F32 = 8,
  POCL_CDBI_ABS_F32 = 9,
  POCL_CDBI_DNN_DENSE_RELU_I8 = 10,
  POCL_CDBI_MAXPOOL_I8 = 11,
  POCL_CDBI_ADD_I8 = 12,
  POCL_CDBI_MUL_I8 = 13,
  POCL_CDBI_ADD_I16 = 14,
  POCL_CDBI_MUL_I16 = 15,
  POCL_CDBI_STREAMOUT_I32 = 16,
  POCL_CDBI_STREAMIN_I32 = 17,
  POCL_CDBI_VOTE_U32 = 18,
  POCL_CDBI_VOTE_U8 = 19,
  POCL_CDBI_DNN_CONV2D_NCHW_F32 = 20,
  POCL_CDBI_OPENVX_SCALEIMAGE_NN_U8 = 21,
  POCL_CDBI_OPENVX_SCALEIMAGE_BL_U8 = 22,
  POCL_CDBI_OPENVX_TENSORCONVERTDEPTH_WRAP_U8_F32 = 23,
  POCL_CDBI_LAST = 24,
  POCL_CDBI_JIT_COMPILER = 0xFFFF
};

// An initialization wrapper for kernel argument metadatas.
struct BIArg : public pocl_argument_info
{
  BIArg (const char *TypeName, const char *Name, pocl_argument_type Type,
         cl_kernel_arg_address_qualifier ADQ = CL_KERNEL_ARG_ADDRESS_GLOBAL,
         cl_kernel_arg_access_qualifier ACQ = CL_KERNEL_ARG_ACCESS_NONE,
         cl_kernel_arg_type_qualifier TQ = CL_KERNEL_ARG_TYPE_NONE,
         size_t size = 0)
  {
    name = strdup (Name);
    address_qualifier = ADQ;
    access_qualifier = ACQ;
    type_qualifier = TQ;
    type_name = strdup (TypeName);
    type_size = size;
    type = Type;
  }

  ~BIArg ()
  {
    free (name);
    free (type_name);
  }
};

// An initialization wrapper for kernel metadatas.
// BIKD = Built-in Kernel Descriptor
struct BIKD : public pocl_kernel_metadata_t
{
  BIKD (BuiltinKernelId KernelId, const char *KernelName,
        const std::vector<pocl_argument_info> &ArgInfos,
        unsigned local_mem_size = 0);

  ~BIKD ()
  {
    delete[] arg_info;
    free (name);
  }

  BuiltinKernelId KernelId;
};


#define BIKERNELS POCL_CDBI_LAST
POCL_EXPORT extern BIKD BIDescriptors[BIKERNELS];

#endif // #ifdef __cplusplus

#ifdef __cplusplus
extern "C"
{
#endif

POCL_EXPORT
int pocl_setup_builtin_metadata(cl_device_id device, cl_program program,
                                unsigned program_device_i);

POCL_EXPORT
int sanitize_builtin_kernel_name(cl_kernel kernel, char **saved_name);

POCL_EXPORT
int restore_builtin_kernel_name(cl_kernel kernel, char* saved_name);

#ifdef __cplusplus
}
#endif


#endif
