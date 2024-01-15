
#ifndef OPENCL_EXP_DEFINED_BUILTIN_KERNELS
#define OPENCL_EXP_DEFINED_BUILTIN_KERNELS

#include <CL/cl_exp_tensor.h>

#define CL_DBK_UNAVAILABLE 0x8101
#define CL_DBK_INVALID_ATTRIBUTE 0x8102

typedef cl_properties cl_dbk_properties;

enum cl_dbk_property
{
  // Maximum relative error in ULPs allowed for the results respect to
  // infinitely precise result.
  CL_DBK_PROPERTY_MAX_RELATIVE_ERROR = 1, // <float>

  // Built-in kernel attributes are immutable values (this allows
  // drivers to specialize their kernels). CL_DBK_MUTABLE_ATTR
  // followed by attribute index (cl_uint) enables the attribute to be
  // mutable via clSetKernelArg(attribute_index, ...).
  CL_DBK_PROPERTY_MUTABLE_ATTR, // <cl_uint>

  // Allows the results of the DBK to fluctuate* with the exactly same
  // inputs across kernel launches.
  //
  // *: CL_DBK_PROPERTY_MAX_RELATIVE_ERROR must still be respected if present.
  //
  // Drivers may ignore this property.
  CL_DBK_PROPERTY_NON_DETERMINISTIC,

  // Allow driver to trade off accuracy for speed by allowing it to flush
  // denormals to zero.
  //
  // Drivers may ignore this property, meaning the behavior is not guaranteed.
  CL_DBK_PROPERTY_ALLOW_FTZ
};

typedef cl_kernel (CL_API_CALL *clCreateBuiltinKernelWithAttributesEXP_fn) (
    cl_program prog, const char *kernel_name, const void *kernel_attributes,
    cl_int *errcode_ret);

extern CL_API_ENTRY cl_kernel CL_API_CALL
clCreateBuiltinKernelWithAttributesEXP (cl_program prog,
                                        const char *kernel_name,
                                        const void *kernel_attributes,
                                        cl_int *errcode_ret);

// Name: "khr_gemm"
// General multiply operation for matrices.
//
// Note that this also performs matrix-vector operations by setting
// tensor shapes accordingly.
typedef struct _cl_dbk_attributes_khr_gemm
{
  const cl_tensor_desc *a;
  const cl_tensor_desc *b;
  const cl_tensor_desc *c_in;
  const cl_tensor_desc *c_out;
  cl_int trans_a;
  cl_int trans_b;
  // Pointers to scaler values. Type depends on the tensor operands. E.g.
  // CL_TENSOR_FLOAT --> cl_float, CL_TENSOR_DOUBLE --> cl_double.
  const void *alpha;
  const void *beta;
  const cl_dbk_properties *kernel_props;
} cl_dbk_attributes_khr_gemm;

// Name: "khr_matmul" Matrix multiplication. Alias for khr_gemm with
// alpha and beta set to 1 and 0, respectively
//
// Note that this also performs matrix-vector operations by setting
// tensor shapes accordingly.
typedef struct _cl_dbk_attributes_khr_matmul
{
  const cl_tensor_desc *a;
  const cl_tensor_desc *b;
  const cl_tensor_desc *c;
  cl_int trans_a;
  cl_int trans_b;
  const cl_dbk_properties *kernel_props;
} cl_dbk_attributes_khr_matmul;

#endif // OPENCL_EXP_DEFINED_BUILTIN_KERNELS
