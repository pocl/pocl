/* pocl_tensor_util.c - Tensor related utilities

   Copyright (c) 2024 Michal Babej / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "pocl_tensor_util.h"
#include "pocl_util.h"

/* Check the tensor layout is well defined.
 * Return CL_INVALID_TENSOR_LAYOUT if there is an error. */
int
pocl_check_tensor_layout (cl_uint rank,
                          const cl_tensor_shape *shape,
                          cl_tensor_layout_type layout_type,
                          const void *layout)
{
  /* Checked already at check_tensor_desc(). */
  assert (rank > 0 && rank <= CL_MEM_MAX_TENSOR_RANK);

  if (layout == NULL && layout_type == CL_TENSOR_LAYOUT_NONE)
    {
      /* TODO: we should allow this at some point,
        but this needs more support. For now, return error.
        Additional checks needed: memory flags.
      * CL_MEM_{COPY,HOST}_host_ptr -> Error due to unspecified
        mapping of the host data to tensor coordinates.
      * CL_MEM_ALLOC_HOST_PTR -> Error for the same reason as for
        CL_MEM_{COPY,HOST}_host_ptr. Could be valid but not
        sensible as users may not know how the tensor elements are
        mapped to the allocation. Perhaps, we could support this
        case, if we extend the clGetMemObjectInfo() to return the
        datalayout the driver picked (and wants to expose)? */
      POCL_RETURN_ERROR_ON (1, CL_INVALID_TENSOR_LAYOUT,
                            "NULL layout currently unsupported\n");
    }

  POCL_RETURN_ERROR_ON (
    (layout_type == CL_TENSOR_LAYOUT_NONE || layout == NULL),
    CL_INVALID_TENSOR_LAYOUT,
    "layout == NULL must be used with CL_TENSOR_LAYOUT_NONE\n");

  switch (layout_type)
    {
    case CL_TENSOR_LAYOUT_NONE:
    default:
      assert (0 && "should have been handled");
      return CL_INVALID_TENSOR_LAYOUT;
    case CL_TENSOR_LAYOUT_ML:
      {
        cl_tensor_layout_ml *ml_layout = (cl_tensor_layout_ml *)layout;
        POCL_RETURN_ERROR_ON (
          (ml_layout->ml_type == CL_TENSOR_LAYOUT_ML_UNKNOWN
           || ml_layout->ml_type >= CL_TENSOR_LAYOUT_ML_LAST),
          CL_INVALID_TENSOR_LAYOUT, "ML layout: unknown type %u",
          ml_layout->ml_type);
        return CL_SUCCESS;
      }
    case CL_TENSOR_LAYOUT_BLAS:
      {
        cl_tensor_layout_blas *blas_layout = (cl_tensor_layout_blas *)layout;

        /* Check leading_dims array does not point out-of-rank dimensions
         * nor the same dimension index does not appear twice.
         *
         * tensor_rank == 4: leading_dims = {0, 2, 1} --> Ok.
         * tensor_rank == 4: leading_dims = {0, 4, 1} --> error.
         * tensor_rank == 4: leading_dims = {1, 1, 0} --> error. */
        unsigned defined_dims = 0;
        const cl_tensor_dim *ld = blas_layout->leading_dims;
        for (unsigned i = 0; i < rank - 1; i++)
          {
            POCL_RETURN_ERROR_ON (
              ld[i] >= rank, CL_INVALID_TENSOR_LAYOUT,
              "BLAS layout: out-of-bounds tensor dimension! %u >= %u\n", ld[i],
              rank);
            POCL_RETURN_ERROR_ON ((defined_dims & (1u << ld[i])),
                                  CL_INVALID_TENSOR_LAYOUT,
                                  "BLAS layout: Dimension defined "
                                  "twice!\n");
            defined_dims |= (1u << ld[i]);
          }

        const cl_tensor_stride *ls = blas_layout->leading_strides;
        cl_tensor_stride prev_stride = 0;
        for (unsigned i = 0; i < rank - 1; i++)
          {
            /* Check the stride configuration does not cause aliasing. */
            POCL_RETURN_ERROR_ON (ls[i] <= shape[ld[i]] * prev_stride,
                                  CL_INVALID_TENSOR_LAYOUT,
                                  "BLAS layout: Invalid stride\n");
            prev_stride = ls[i];
          }

        return CL_SUCCESS;
      }
    }

  assert (!"Unreachable!");
  return CL_INVALID_TENSOR_LAYOUT;
}

/* Checks validity of the tensor shape. Returns CL_INVALID_XYZ on error. */
int
pocl_check_tensor_desc (const cl_tensor_desc *tdesc)
{
  /* Invalid to pass NULL tensor description in clCreateBufferWithProperties.
   */
  POCL_RETURN_ERROR_COND ((tdesc == NULL), CL_INVALID_ARG_VALUE);

  POCL_RETURN_ERROR_ON ((tdesc->rank > CL_MEM_MAX_TENSOR_RANK),
                        CL_INVALID_TENSOR_RANK, "Unsupported tensor rank.");

  for (unsigned i = 0; i < tdesc->rank; i++)
    POCL_RETURN_ERROR_ON ((tdesc->shape[i] == 0), CL_INVALID_TENSOR_SHAPE,
                          "Tensor shape must be fully specified!");

  const cl_tensor_properties *P = tdesc->properties;
  while (*P)
    {
      switch (*P)
        {
        case CL_TENSOR_PROPERTY_MUTABLE_SHAPE:
        case CL_TENSOR_PROPERTY_MUTABLE_DTYPE:
        case CL_TENSOR_PROPERTY_MUTABLE_LAYOUT:
          break;
        default:
          POCL_RETURN_ERROR_ON (1, CL_INVALID_TENSOR_PROPERTY,
                                "Unknown property %" PRIu64 "\n", *P);
        }
      ++P;
    }
  return pocl_check_tensor_layout (tdesc->rank, tdesc->shape,
                                   tdesc->layout_type, tdesc->layout);
}

static void *
duplicate_array (const void *src, size_t num_objects, size_t object_size)
{
  void *new_objects = calloc (num_objects, object_size);
  if (!new_objects)
    return NULL;
  memcpy (new_objects, src, object_size * num_objects);
  return new_objects;
}

#define DUPLICATE_ARRAY(source_ptr, num_objects, object_type)                 \
  duplicate_array ((source_ptr), (num_objects), sizeof (object_type));

/* Copies the tensor description (deep copy) into the cl_mem.
 * The 'tdesc' must be valid (checked by check_tensor_desc). */
int
pocl_copy_tensor_desc2mem (cl_mem mem, const cl_tensor_desc *tdesc)
{
  if (!tdesc)
    return CL_SUCCESS;

  mem->is_tensor = CL_TRUE;
  mem->tensor_rank = tdesc->rank;
  memcpy (mem->tensor_shape, tdesc->shape, sizeof (mem->tensor_shape));
  mem->tensor_dtype = tdesc->dtype;
  mem->tensor_layout_type = 0;
  mem->tensor_layout = NULL;

  if (!tdesc->layout)
    {
      assert (tdesc->layout_type == CL_TENSOR_LAYOUT_NONE);
      return CL_SUCCESS;
    }

  cl_tensor_layout_ml *new_layout1 = NULL;
  cl_tensor_layout_blas *new_layout2 = NULL;

  switch (tdesc->layout_type)
    {
    default:
    case CL_TENSOR_LAYOUT_NONE:
      assert (0 && "this should have been caught earlier");
      return CL_FAILED;

    case CL_TENSOR_LAYOUT_ML:
      {
        cl_tensor_layout_ml *ml_layout = (cl_tensor_layout_ml *)tdesc->layout;
        new_layout1 = DUPLICATE_ARRAY (ml_layout, 1, cl_tensor_layout_blas);
        if (!new_layout1)
          goto error;
        mem->tensor_layout = new_layout1;
        mem->tensor_layout_type = CL_TENSOR_LAYOUT_ML;
        return CL_SUCCESS;
      }

    case CL_TENSOR_LAYOUT_BLAS:
      {
        cl_tensor_layout_blas *blas_layout
          = (cl_tensor_layout_blas *)tdesc->layout;
        new_layout2 = DUPLICATE_ARRAY (blas_layout, 1, cl_tensor_layout_blas);
        if (!new_layout2)
          goto error;

        memcpy (new_layout2->leading_dims, blas_layout->leading_dims,
                sizeof (blas_layout->leading_dims));
        memcpy (new_layout2->leading_strides, blas_layout->leading_strides,
                sizeof (blas_layout->leading_strides));
        mem->tensor_layout = new_layout2;
        mem->tensor_layout_type = CL_TENSOR_LAYOUT_BLAS;
        return CL_SUCCESS;
      }
    }

error:
  free (new_layout1);
  free (new_layout2);
  return CL_OUT_OF_HOST_MEMORY;
}

/* Copies the tensor layout only
 * The 'src' must be valid (checked by check_tensor_desc). */
int
pocl_copy_tensor_desc_layout (cl_tensor_desc *dest, const cl_tensor_desc *src)
{
  if (src == NULL)
    return CL_SUCCESS;
  switch (src->layout_type)
    {
    case CL_TENSOR_LAYOUT_ML:
      dest->layout = malloc (sizeof (cl_tensor_layout_ml));
      memcpy (dest->layout, src->layout, sizeof (cl_tensor_layout_ml));
      return CL_SUCCESS;
    case CL_TENSOR_LAYOUT_BLAS:
      dest->layout = malloc (sizeof (cl_tensor_layout_blas));
      memcpy (dest->layout, src->layout, sizeof (cl_tensor_layout_blas));
      return CL_SUCCESS;
    default:
      return CL_SUCCESS;
    }
}

int
pocl_tensor_type_is_int (cl_tensor_datatype T)
{
  switch (T)
    {
    case CL_TENSOR_DTYPE_FP64:
    case CL_TENSOR_DTYPE_FP32:
    case CL_TENSOR_DTYPE_FP16:
    case CL_TENSOR_DTYPE_FP8:
      return 0;

    case CL_TENSOR_DTYPE_INT64:
    case CL_TENSOR_DTYPE_UINT64:
    case CL_TENSOR_DTYPE_INT32:
    case CL_TENSOR_DTYPE_UINT32:
    case CL_TENSOR_DTYPE_INT16:
    case CL_TENSOR_DTYPE_UINT16:
    case CL_TENSOR_DTYPE_INT8:
    case CL_TENSOR_DTYPE_UINT8:
    case CL_TENSOR_DTYPE_INT4:
    case CL_TENSOR_DTYPE_UINT4:
      return 1;

    case CL_TENSOR_DTYPE_UNKNOWN:
    default:
      return -1;
    }
}

int
pocl_tensor_type_size (cl_tensor_datatype T)
{
  switch (T)
    {
    case CL_TENSOR_DTYPE_FP64:
    case CL_TENSOR_DTYPE_INT64:
    case CL_TENSOR_DTYPE_UINT64:
      return 8;
    case CL_TENSOR_DTYPE_FP32:
    case CL_TENSOR_DTYPE_INT32:
    case CL_TENSOR_DTYPE_UINT32:
      return 4;
    case CL_TENSOR_DTYPE_FP16:
    case CL_TENSOR_DTYPE_INT16:
    case CL_TENSOR_DTYPE_UINT16:
      return 2;
    case CL_TENSOR_DTYPE_FP8:
    case CL_TENSOR_DTYPE_INT8:
    case CL_TENSOR_DTYPE_UINT8:
    case CL_TENSOR_DTYPE_INT4:
    case CL_TENSOR_DTYPE_UINT4:
      return 1;
    case CL_TENSOR_DTYPE_UNKNOWN:
    default:
      return -1;
    }
}

cl_bool
pocl_tensor_dtype_value_equals (const cl_tensor_datatype DType,
                                const cl_tensor_datatype_value *Value,
                                cl_double doubleConst,
                                cl_long longConst,
                                cl_ulong ulongConst,
                                char fp8Const,
                                char int4Const)
{
  cl_float floatConst = (cl_float)doubleConst;
  cl_half halfConst = float_to_half (floatConst);
  switch (DType)
    {
    case CL_TENSOR_DTYPE_FP64:
      return (Value->d == doubleConst);
    case CL_TENSOR_DTYPE_FP32:
      return (Value->f == floatConst);
    case CL_TENSOR_DTYPE_FP16:
      return (Value->h == halfConst);
    /* TODO fp8 comparison */
    case CL_TENSOR_DTYPE_FP8:
      return (Value->c == fp8Const);
    case CL_TENSOR_DTYPE_INT64:
      return (Value->l == longConst);
    case CL_TENSOR_DTYPE_UINT64:
      return (Value->l == ulongConst);
    case CL_TENSOR_DTYPE_INT32:
      return ((cl_long)Value->l == longConst);
    case CL_TENSOR_DTYPE_UINT32:
      return ((cl_ulong)Value->i == ulongConst);
    case CL_TENSOR_DTYPE_INT16:
      return ((cl_long)Value->s == longConst);
    case CL_TENSOR_DTYPE_UINT16:
      return ((cl_ulong)Value->s == ulongConst);
    case CL_TENSOR_DTYPE_INT8:
      return ((cl_long)Value->c == longConst);
    case CL_TENSOR_DTYPE_UINT8:
      return ((cl_ulong)Value->c == ulongConst);
    /* TODO int4 comparison */
    case CL_TENSOR_DTYPE_INT4:
    case CL_TENSOR_DTYPE_UINT4:
      return (Value->c == int4Const);
    default:
      return CL_FALSE;
    }
}
