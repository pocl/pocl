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
#include "CL/cl_exp_tensor.h"
#include "pocl_cl_half_util.h"

/* Check the tensor layout is well defined.
 * Return CL_INVALID_TENSOR_LAYOUT_EXP if there is an error. */
int
pocl_check_tensor_layout (cl_uint rank,
                          const cl_tensor_shape_exp *shape,
                          cl_tensor_layout_type_exp layout_type,
                          const void *layout)
{
  /* Checked already at check_tensor_desc(). */
  assert (rank > 0 && rank <= CL_MEM_MAX_TENSOR_RANK_EXP);

  if (layout == NULL && layout_type == CL_TENSOR_LAYOUT_OPAQUE_EXP)
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
      POCL_RETURN_ERROR (CL_INVALID_TENSOR_LAYOUT_EXP,
                         "NULL layout currently unsupported\n");
    }

  POCL_RETURN_ERROR_ON (
    (layout_type == CL_TENSOR_LAYOUT_OPAQUE_EXP || layout == NULL),
    CL_INVALID_TENSOR_LAYOUT_EXP,
    "layout == NULL must be used with CL_TENSOR_LAYOUT_OPAQUE_EXP\n");

  switch (layout_type)
    {
    case CL_TENSOR_LAYOUT_OPAQUE_EXP:
    default:
      assert (0 && "should have been handled");
      return CL_INVALID_TENSOR_LAYOUT_EXP;
    case CL_TENSOR_LAYOUT_ML_EXP:
      {
        cl_tensor_layout_ml_exp *ml_layout = (cl_tensor_layout_ml_exp *)layout;
        POCL_RETURN_ERROR_ON (
          (ml_layout->ml_type == CL_TENSOR_LAYOUT_ML_UNKNOWN
           || ml_layout->ml_type >= CL_TENSOR_LAYOUT_ML_LAST),
          CL_INVALID_TENSOR_LAYOUT_EXP, "ML layout: unknown type %u",
          ml_layout->ml_type);
        return CL_SUCCESS;
      }
    case CL_TENSOR_LAYOUT_BLAS_EXP:
      {
        cl_tensor_layout_blas_exp *blas_layout
          = (cl_tensor_layout_blas_exp *)layout;

        /* Check leading_dims array does not point out-of-rank dimensions
         * nor the same dimension index does not appear twice.
         *
         * tensor_rank == 4: leading_dims = {0, 2, 1} --> Ok.
         * tensor_rank == 4: leading_dims = {0, 4, 1} --> error.
         * tensor_rank == 4: leading_dims = {1, 1, 0} --> error. */
        unsigned defined_dims = 0;
        const cl_tensor_dim_exp *ld = blas_layout->leading_dims;
        for (unsigned i = 0; i < rank - 1; i++)
          {
            POCL_RETURN_ERROR_ON (
              ld[i] >= rank, CL_INVALID_TENSOR_LAYOUT_EXP,
              "BLAS layout: out-of-bounds tensor dimension! %u >= %u\n", ld[i],
              rank);
            POCL_RETURN_ERROR_ON ((defined_dims & (1u << ld[i])),
                                  CL_INVALID_TENSOR_LAYOUT_EXP,
                                  "BLAS layout: Dimension defined "
                                  "twice!\n");
            defined_dims |= (1u << ld[i]);
          }
      }
    case CL_TENSOR_LAYOUT_BLAS_PITCHED_EXP:
      {
        cl_tensor_layout_blas_pitched_exp *blas_layout
          = (cl_tensor_layout_blas_pitched_exp *)layout;

        /* Check leading_dims array does not point out-of-rank dimensions
         * nor the same dimension index does not appear twice.
         *
         * tensor_rank == 4: leading_dims = {0, 2, 1} --> Ok.
         * tensor_rank == 4: leading_dims = {0, 4, 1} --> error.
         * tensor_rank == 4: leading_dims = {1, 1, 0} --> error. */
        unsigned defined_dims = 0;
        const cl_tensor_dim_exp *ld = blas_layout->leading_dims;
        for (unsigned i = 0; i < rank - 1; i++)
          {
            POCL_RETURN_ERROR_ON (
              ld[i] >= rank, CL_INVALID_TENSOR_LAYOUT_EXP,
              "BLAS layout: out-of-bounds tensor dimension! %u >= %u\n", ld[i],
              rank);
            POCL_RETURN_ERROR_ON ((defined_dims & (1u << ld[i])),
                                  CL_INVALID_TENSOR_LAYOUT_EXP,
                                  "BLAS layout: Dimension defined "
                                  "twice!\n");
            defined_dims |= (1u << ld[i]);
          }

        cl_tensor_layout_blas_pitched_exp *blas_pitched_layout
          = (cl_tensor_layout_blas_pitched_exp *)layout;

        const cl_tensor_stride_exp *ls = blas_pitched_layout->leading_strides;
        cl_tensor_stride_exp prev_stride = 0;
        for (unsigned i = 0; i < rank - 1; i++)
          {
            /* Check the stride configuration does not cause aliasing. */
            POCL_RETURN_ERROR_ON (ls[i] <= shape[ld[i]] * prev_stride,
                                  CL_INVALID_TENSOR_LAYOUT_EXP,
                                  "BLAS layout: Invalid stride\n");
            prev_stride = ls[i];
          }

        return CL_SUCCESS;
      }
    }

  assert (!"Unreachable!");
  return CL_INVALID_TENSOR_LAYOUT_EXP;
}

/* Checks validity of the tensor shape. Returns CL_INVALID_XYZ on error. */
int
pocl_check_tensor_desc (const cl_tensor_desc_exp *tdesc)
{
  /* Invalid to pass NULL tensor description in clCreateBufferWithProperties.
   */
  POCL_RETURN_ERROR_COND ((tdesc == NULL), CL_INVALID_ARG_VALUE);

  POCL_RETURN_ERROR_ON ((tdesc->rank > CL_MEM_MAX_TENSOR_RANK_EXP),
                        CL_INVALID_TENSOR_RANK_EXP,
                        "Unsupported tensor rank.");

  for (unsigned i = 0; i < tdesc->rank; i++)
    POCL_RETURN_ERROR_ON ((tdesc->shape[i] == 0), CL_INVALID_TENSOR_SHAPE_EXP,
                          "Tensor shape must be fully specified!");

  const cl_tensor_properties_exp *P = tdesc->properties;
  while (*P)
    {
      switch (*P)
        {
        case CL_TENSOR_PROPERTY_MUTABLE_SHAPE_EXP:
        case CL_TENSOR_PROPERTY_MUTABLE_DTYPE_EXP:
        case CL_TENSOR_PROPERTY_MUTABLE_LAYOUT_EXP:
          break;
        default:
          POCL_RETURN_ERROR (CL_INVALID_TENSOR_PROPERTY_EXP,
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
pocl_copy_tensor_desc2mem (cl_mem mem, const cl_tensor_desc_exp *tdesc)
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
      assert (tdesc->layout_type == CL_TENSOR_LAYOUT_OPAQUE_EXP);
      return CL_SUCCESS;
    }

  cl_tensor_layout_ml_exp *new_layout1 = NULL;
  cl_tensor_layout_blas_exp *new_layout2 = NULL;
  cl_tensor_layout_blas_pitched_exp *new_layout3 = NULL;

  switch (tdesc->layout_type)
    {
    default:
    case CL_TENSOR_LAYOUT_OPAQUE_EXP:
      assert (0 && "this should have been caught earlier");
      return CL_FAILED;

    case CL_TENSOR_LAYOUT_ML_EXP:
      {
        cl_tensor_layout_ml_exp *ml_layout
          = (cl_tensor_layout_ml_exp *)tdesc->layout;
        new_layout1 = DUPLICATE_ARRAY (ml_layout, 1, cl_tensor_layout_ml_exp);
        if (!new_layout1)
          goto error;
        mem->tensor_layout = new_layout1;
        mem->tensor_layout_type = CL_TENSOR_LAYOUT_ML_EXP;
        return CL_SUCCESS;
      }

    case CL_TENSOR_LAYOUT_BLAS_EXP:
      {
        cl_tensor_layout_blas_exp *blas_layout
          = (cl_tensor_layout_blas_exp *)tdesc->layout;
        new_layout2
          = DUPLICATE_ARRAY (blas_layout, 1, cl_tensor_layout_blas_exp);
        if (!new_layout2)
          goto error;

        memcpy (new_layout2->leading_dims, blas_layout->leading_dims,
                sizeof (blas_layout->leading_dims));
        mem->tensor_layout = new_layout2;
        mem->tensor_layout_type = CL_TENSOR_LAYOUT_BLAS_EXP;
        return CL_SUCCESS;
      }
    case CL_TENSOR_LAYOUT_BLAS_PITCHED_EXP:
      {
        cl_tensor_layout_blas_pitched_exp *blas_layout
          = (cl_tensor_layout_blas_pitched_exp *)tdesc->layout;
        new_layout3 = DUPLICATE_ARRAY (blas_layout, 1,
                                       cl_tensor_layout_blas_pitched_exp);
        if (!new_layout3)
          goto error;

        memcpy (new_layout3->leading_dims, blas_layout->leading_dims,
                sizeof (blas_layout->leading_dims));
        memcpy (new_layout3->leading_strides, blas_layout->leading_strides,
                sizeof (blas_layout->leading_strides));
        mem->tensor_layout = new_layout3;
        mem->tensor_layout_type = CL_TENSOR_LAYOUT_BLAS_PITCHED_EXP;
        return CL_SUCCESS;
      }
    }

error:
  free (new_layout1);
  free (new_layout2);
  free (new_layout3);
  return CL_OUT_OF_HOST_MEMORY;
}

/* Copies the tensor layout only
 * The 'src' must be valid (checked by check_tensor_desc). */
int
pocl_copy_tensor_desc_layout (cl_tensor_desc_exp *dest,
                              const cl_tensor_desc_exp *src)
{
  if (src == NULL)
    return CL_SUCCESS;
  switch (src->layout_type)
    {
    case CL_TENSOR_LAYOUT_ML_EXP:
      dest->layout = malloc (sizeof (cl_tensor_layout_ml_exp));
      memcpy ((void *)dest->layout, src->layout,
              sizeof (cl_tensor_layout_ml_exp));
      return CL_SUCCESS;
    case CL_TENSOR_LAYOUT_BLAS_EXP:
      dest->layout = malloc (sizeof (cl_tensor_layout_blas_exp));
      memcpy ((void *)dest->layout, src->layout,
              sizeof (cl_tensor_layout_blas_exp));
      return CL_SUCCESS;
    case CL_TENSOR_LAYOUT_BLAS_PITCHED_EXP:
      dest->layout = malloc (sizeof (cl_tensor_layout_blas_pitched_exp));
      memcpy ((void *)dest->layout, src->layout,
              sizeof (cl_tensor_layout_blas_pitched_exp));
      return CL_SUCCESS;
    default:
      return CL_SUCCESS;
    }
}

int
pocl_tensor_type_is_int (cl_tensor_datatype_exp dtype)
{
  switch (dtype)
    {
    case CL_TENSOR_DTYPE_FP64_EXP:
    case CL_TENSOR_DTYPE_FP32_EXP:
    case CL_TENSOR_DTYPE_FP16_EXP:
    case CL_TENSOR_DTYPE_FP8E4M3_EXP:
    case CL_TENSOR_DTYPE_FP8E5M2_EXP:
      return 0;

    case CL_TENSOR_DTYPE_INT64_EXP:
    case CL_TENSOR_DTYPE_UINT64_EXP:
    case CL_TENSOR_DTYPE_INT32_EXP:
    case CL_TENSOR_DTYPE_UINT32_EXP:
    case CL_TENSOR_DTYPE_INT16_EXP:
    case CL_TENSOR_DTYPE_UINT16_EXP:
    case CL_TENSOR_DTYPE_INT8_EXP:
    case CL_TENSOR_DTYPE_UINT8_EXP:
    case CL_TENSOR_DTYPE_INT4_EXP:
    case CL_TENSOR_DTYPE_UINT4_EXP:
      return 1;

    case CL_TENSOR_DTYPE_UNKNOWN:
    default:
      return -1;
    }
}

int
pocl_tensor_type_size (cl_tensor_datatype_exp dtype)
{
  switch (dtype)
    {
    case CL_TENSOR_DTYPE_FP64_EXP:
    case CL_TENSOR_DTYPE_INT64_EXP:
    case CL_TENSOR_DTYPE_UINT64_EXP:
      return 8;
    case CL_TENSOR_DTYPE_FP32_EXP:
    case CL_TENSOR_DTYPE_INT32_EXP:
    case CL_TENSOR_DTYPE_UINT32_EXP:
      return 4;
    case CL_TENSOR_DTYPE_FP16_EXP:
    case CL_TENSOR_DTYPE_INT16_EXP:
    case CL_TENSOR_DTYPE_UINT16_EXP:
      return 2;
    case CL_TENSOR_DTYPE_FP8E4M3_EXP:
    case CL_TENSOR_DTYPE_FP8E5M2_EXP:
    case CL_TENSOR_DTYPE_INT8_EXP:
    case CL_TENSOR_DTYPE_UINT8_EXP:
    case CL_TENSOR_DTYPE_INT4_EXP:
    case CL_TENSOR_DTYPE_UINT4_EXP:
      return 1;
    case CL_TENSOR_DTYPE_UNKNOWN:
    default:
      return -1;
    }
}

size_t
pocl_tensor_data_size (const cl_tensor_desc_exp *t)
{
  size_t data_len = pocl_tensor_type_size (t->dtype);
  for (size_t dim = 0; dim < t->rank; ++dim)
    {
      data_len *= t->shape[dim];
    }
  return data_len;
}

cl_bool
pocl_tensor_dtype_value_equals (const cl_tensor_datatype_exp dtype,
                                const cl_tensor_datatype_value_exp *value,
                                cl_double double_const,
                                cl_long long_const,
                                cl_ulong ulong_const,
                                char fp8_const,
                                char int4_const)
{
  cl_float float_const = (cl_float)double_const;
  cl_half half_const = pocl_float_to_half (float_const);
  switch (dtype)
    {
    case CL_TENSOR_DTYPE_FP64_EXP:
      return (value->fd == double_const);
    case CL_TENSOR_DTYPE_FP32_EXP:
      return (value->ff == float_const);
    case CL_TENSOR_DTYPE_FP16_EXP:
      return (value->fh == half_const);
    /* TODO fp8 comparison */
    case CL_TENSOR_DTYPE_FP8E4M3_EXP:
    case CL_TENSOR_DTYPE_FP8E5M2_EXP:
      /* FIXME: Need to add new union members or pack them into value->raw. */
      return (value->uc == fp8_const);
    case CL_TENSOR_DTYPE_INT64_EXP:
      return (value->sl == long_const);
    case CL_TENSOR_DTYPE_UINT64_EXP:
      return ((cl_ulong)value->ul == ulong_const);
    case CL_TENSOR_DTYPE_INT32_EXP:
      return (value->sl == long_const);
    case CL_TENSOR_DTYPE_UINT32_EXP:
      return ((cl_ulong)value->ui == ulong_const);
    case CL_TENSOR_DTYPE_INT16_EXP:
      return ((cl_long)value->ss == long_const);
    case CL_TENSOR_DTYPE_UINT16_EXP:
      return ((cl_ulong)value->us == ulong_const);
    case CL_TENSOR_DTYPE_INT8_EXP:
      return ((cl_long)value->sc == long_const);
    case CL_TENSOR_DTYPE_UINT8_EXP:
      return ((cl_ulong)value->uc == ulong_const);
    /* TODO int4 comparison */
    case CL_TENSOR_DTYPE_INT4_EXP:
    case CL_TENSOR_DTYPE_UINT4_EXP:
      /* FIXME: Need to specify how objects less than 8-bit are represented.
       *        One object per byte? Multiple objects per byte? */
      return (value->uc == int4_const);
    default:
      return CL_FALSE;
    }
}
