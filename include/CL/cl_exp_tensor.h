
#ifndef OPENCL_EXP_TENSOR_H
#define OPENCL_EXP_TENSOR_H
#include <CL/opencl.h>

typedef cl_ulong cl_tensor_shape;
typedef cl_uint cl_tensor_dim;
typedef cl_uint cl_tensor_desc_type;
typedef cl_uint cl_tensor_datatype;
typedef cl_uint cl_tensor_layout_type;

// cl_tensor_desc_type
#define CL_TENSOR_DESC_BASE 1

// cl_tensor_datatype
#define CL_TENSOR_FLOAT 1
#define CL_TENSOR_DOUBLE 2
#define CL_TENSOR_INT 3
// TODO: To be completed later.

// cl_tensor_layout_type
#define CL_TENSOR_LAYOUT_OPAQUE 1
#define CL_TENSOR_LAYOUT_BLAS 2

// Additions to cl_mem_object_type

// A clCreateBufferWithProperties() property used for creating a tensor.
#define CL_MEM_TENSOR 0x8000

// TBC: A clCreateSubBuffer() cl_buffer_create_type used for creating a
// subtensor for the purpose of:
//
// * coercing an existing buffer (non-tensor) to a tensor (or part of it).
// * splitting large tensors to smaller ones.
// * reshaping existing tensor to another.
// * coercing data type of an existing tensor to other type of same size.
#define CL_MEM_TENSOR_VIEW 0x8001

typedef struct _cl_tensor_desc
{
  cl_tensor_desc_type stype; // Must be CL_TENSOR_DESC_BASE
  void *next;

  // The rank of the tensor.
  cl_uint rank;

  // The shape of the tensor described by an array. Describes number
  // of elements in the tensor dimensions starting with "outermost"
  // dimension first. E.g. {..., NumOf2DBlocks, NumOf1DBlocks,
  // NumEltsIn1D}.  (This convention is tentatively chosen for
  // matching python, numpy and popular ML frameworks).
  //
  // Conditions:
  //
  // * Must be non-NULL.
  //
  // * Lenght of the array must be at least <rank> elements.
  //
  // * TBC: A dimension can be zero meaning the size is unspeficied. However,
  //   commands involing tensors must have fully specified shape.
  const cl_tensor_shape *shape;

  // The element type of the tensor.
  cl_tensor_datatype dtype;

  // Optional data layout description. Must be NULL or one of
  // cl_tensor_layout_* structures in the below.
  //
  // If NULL, cl{Enqueue,Command}{ImportFrom,ExportTo}Tensor must be
  // used for transferring data from or to tensor. If a pointer to the
  // tensor data is aquired (somehow), dereferencing that pointer is
  // undefined behavior.
  const void *layout;
} cl_tensor_desc;

// All tensor layout descriptions start with the following common
// initial sequence.
typedef struct _cl_tensor_layout_base
{
  cl_tensor_layout_type stype;
  // Vulkan style extension mechanism. Must be NULL if not used.
  void *next;
} cl_tensor_layout_base;

// Describes data layout similar to one used in BLAS APIs.
typedef struct _cl_tensor_layout_blas
{
  cl_tensor_layout_type stype; // Must be set to CL_TENSOR_LAYOUT_BLAS.
  void *next;

  // Leading tensor dimensions. This describes which elements along
  // tensor dimensions are laid out first in the memory. Tensor
  // coodrinates (tensor_coords = {x0, x1, ..., x2}) map to buffer
  // (buffer_offset) as followed:
  //
  //   size_t index = 0;
  //   for (unsigned i = 0; i < tensor_rank; i++) {
  //      index += tensor_coords[leading_dims[i]] * leading_strides[i];
  //   size_t buffer_offset = index;
  //
  // Conditions:
  //
  // * Array length must be at least 'tensor_rank - 1' (last dimension
  //   is implied)
  //
  // * Each tensor dimension 0..<tensor_rank - 1> must appear ones in
  //   the array.
  const cl_tensor_dim *leading_dims;

  // Strides of the leading dimensions. Array length must be at least
  // (tensor_rank - 1) and following assertion must hold:
  //
  //   for (unsigned i = 0; i < tensor_rank - 1; i++) {
  //     size_t tensor_slice_size = 1;
  //     for (unsigned j = 0; j <= i; j++)
  //       tensor_slice_size *= tensor_shape[j];
  //     assert(leading_dims[i] >= tensor_slize_size);
  //   }
  //
  // TBC: Allow leading_strides == NULL in which case the tensor data
  //      is non-strided (e.g. for matrices there is no gaps between
  //      columns/rows) for convenience?
  const size_t *leading_strides;

  // TBC: This field specifies an optional alignment guarantee for the
  // first element (an element at coordinate = (0, 0, ..., 0)). The
  // value must be 0 or power-of-two. If zero, the alignment inferred from
  // the dtype. This could also be a layout extension.
  //size_t base_alignment;

} cl_tensor_layout_blas;

#endif // OPENCL_EXP_TENSOR_H
