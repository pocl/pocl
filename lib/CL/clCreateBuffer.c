/* OpenCL runtime library: clCreateBuffer()

   Copyright (c) 2011-2013 Universidad Rey Juan Carlos
                           Pekka Jääskeläinen / Tampere University of Tech.
                 2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "common.h"
#include "devices.h"
#include "pocl_cl.h"
#include "pocl_shared.h"
#include "pocl_util.h"

extern unsigned long buffer_c;

cl_mem
pocl_create_memobject (cl_context context, cl_mem_flags flags, size_t size,
                       cl_mem_object_type type, int* device_image_support,
                       void *host_ptr, int host_ptr_is_svm, cl_int *errcode_ret)
{
  cl_mem mem = NULL;
  int errcode = CL_SUCCESS;
  unsigned i;
  cl_mem_flags stdflags = flags;

  POCL_GOTO_ERROR_COND((size == 0), CL_INVALID_BUFFER_SIZE);

  POCL_GOTO_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

  if (flags == 0)
    flags = CL_MEM_READ_WRITE;

  /* validate flags */
  if (flags & CL_MEM_DEVICE_ADDRESS_EXT)
    {
      for (i = 0; i < context->num_devices; ++i)
        {
          cl_device_id dev = context->devices[i];
          POCL_GOTO_ERROR_ON (
              strstr ("cl_ext_buffer_device_address", dev->extensions) != 0,
              CL_INVALID_VALUE,
              "Requested CL_MEM_DEVICE_ADDRESS allocation, but a device in "
              "context doesn't support the 'cl_ext_buffer_device_address' "
              "extension.");
        }
      stdflags = flags ^ CL_MEM_DEVICE_ADDRESS_EXT;
    }

  if (stdflags & CL_MEM_DEVICE_PRIVATE_EXT)
    stdflags = stdflags ^ CL_MEM_DEVICE_PRIVATE_EXT;

  POCL_GOTO_ERROR_ON ((stdflags > (1 << 10) - 1), CL_INVALID_VALUE,
                      "There are only 10 non-SVM flags)\n");

  POCL_GOTO_ERROR_ON (
      ((flags & CL_MEM_READ_WRITE)
       && (flags & CL_MEM_WRITE_ONLY || flags & CL_MEM_READ_ONLY)),
      CL_INVALID_VALUE,
      "Invalid flags: CL_MEM_READ_WRITE cannot be used "
      "together with CL_MEM_WRITE_ONLY or CL_MEM_READ_ONLY\n");

  POCL_GOTO_ERROR_ON (
      ((flags & CL_MEM_READ_ONLY) && (flags & CL_MEM_WRITE_ONLY)),
      CL_INVALID_VALUE,
      "Invalid flags: "
      "can't have both CL_MEM_WRITE_ONLY and CL_MEM_READ_ONLY\n");

  POCL_GOTO_ERROR_ON (
      ((flags & CL_MEM_USE_HOST_PTR)
       && (flags & CL_MEM_ALLOC_HOST_PTR || flags & CL_MEM_COPY_HOST_PTR)),
      CL_INVALID_VALUE,
      "Invalid flags: CL_MEM_USE_HOST_PTR cannot be used "
      "together with CL_MEM_ALLOC_HOST_PTR or CL_MEM_COPY_HOST_PTR\n");

  POCL_GOTO_ERROR_ON (
      ((flags & CL_MEM_HOST_WRITE_ONLY) && (flags & CL_MEM_HOST_READ_ONLY)),
      CL_INVALID_VALUE,
      "Invalid flags: "
      "can't have both CL_MEM_HOST_READ_ONLY and CL_MEM_HOST_WRITE_ONLY\n");

  POCL_GOTO_ERROR_ON (
      ((flags & CL_MEM_HOST_NO_ACCESS)
       && ((flags & CL_MEM_HOST_READ_ONLY)
           || (flags & CL_MEM_HOST_WRITE_ONLY))),
      CL_INVALID_VALUE,
      "Invalid flags: CL_MEM_HOST_NO_ACCESS cannot be used "
      "together with CL_MEM_HOST_READ_ONLY or CL_MEM_HOST_WRITE_ONLY\n");

  if (host_ptr == NULL)
    {
      POCL_GOTO_ERROR_ON (
          ((flags & CL_MEM_USE_HOST_PTR) || (flags & CL_MEM_COPY_HOST_PTR)),
          CL_INVALID_HOST_PTR,
          "host_ptr is NULL, but flags specify {COPY|USE}_HOST_PTR\n");
    }
  else
    {
      POCL_GOTO_ERROR_ON (
          ((~flags & CL_MEM_USE_HOST_PTR) && (~flags & CL_MEM_COPY_HOST_PTR)),
          CL_INVALID_HOST_PTR,
          "host_ptr is not NULL, but flags don't specify "
          "{COPY|USE}_HOST_PTR\n");
    }

  POCL_GOTO_ERROR_ON ((size > context->max_mem_alloc_size),
                      CL_INVALID_BUFFER_SIZE,
                      "Size (%zu) is bigger than max mem alloc size (%zu) "
                      "of all devices in context\n",
                      size, (size_t)context->max_mem_alloc_size);

  mem = (cl_mem)calloc (1, sizeof (struct _cl_mem));
  POCL_GOTO_ERROR_COND ((mem == NULL), CL_OUT_OF_HOST_MEMORY);

  POCL_INIT_OBJECT (mem);
  mem->type = type;
  mem->flags = flags;
  mem->device_supports_this_image = device_image_support;

  mem->device_ptrs = (pocl_mem_identifier *)calloc (
      POCL_ATOMIC_LOAD (pocl_num_devices), sizeof (pocl_mem_identifier));
  POCL_GOTO_ERROR_COND ((mem->device_ptrs == NULL), CL_OUT_OF_HOST_MEMORY);

  mem->size = size;
  mem->context = context;
  mem->is_image = (type != CL_MEM_OBJECT_PIPE && type != CL_MEM_OBJECT_BUFFER);
  mem->is_pipe = (type == CL_MEM_OBJECT_PIPE);
  mem->mem_host_ptr_version = 0;
  mem->latest_version = 0;

  if (flags & CL_MEM_DEVICE_ADDRESS_EXT)
    {
      mem->has_device_address = 1;
    }
  /* https://www.khronos.org/registry/OpenCL/sdk/2.0/docs/man/xhtml/dataTypes.html
   *
   * The user is responsible for ensuring that data passed into and out of
   * OpenCL buffers are natively aligned relative to the start of the buffer as
   * described above. This implies that OpenCL buffers created with
   * CL_MEM_USE_HOST_PTR need to provide an appropriately aligned host memory
   * pointer that is aligned to the data types used to access these buffers in
   * a kernel(s).
   */
  if (flags & CL_MEM_USE_HOST_PTR)
    {
      POCL_MSG_PRINT_MEMORY ("CL_MEM_USE_HOST_PTR %p \n", host_ptr);
      assert (host_ptr);
      mem->mem_host_ptr = host_ptr;
      if (((uintptr_t)host_ptr % context->min_buffer_alignment) != 0)
        {
          POCL_MSG_WARN ("host_ptr (%p) given to "
                         "clCreateBuffer(CL_MEM_USE_HOST_PTR, ..)\n"
                         "isn't aligned for any device in context;\n"
                         "The minimum required alignment is: %zu;\n"
                         "This can cause various problems later.\n",
                         host_ptr, context->min_buffer_alignment);
        }
      mem->mem_host_ptr_version = 1;
      mem->mem_host_ptr_refcount = 1;
      mem->mem_host_ptr_is_svm = host_ptr_is_svm;
      mem->latest_version = 1;
    }

  /* If ALLOC flag is present, try to pre-allocate host-visible
   * backing store memory from a driver.
   * First driver to allocate for a physical memory wins; if none of
   * the drivers do it, we allocate the backing store via malloc */
  if (flags & CL_MEM_ALLOC_HOST_PTR)
    {
      POCL_MSG_PRINT_MEMORY (
          "Trying driver allocation for CL_MEM_ALLOC_HOST_PTR\n");
      unsigned i;
      for (i = 0; i < context->num_devices; ++i)
        {
          cl_device_id dev = context->devices[i];
          assert (dev->ops->alloc_mem_obj != NULL);
          // skip already allocated
          if (mem->device_ptrs[dev->global_mem_id].mem_ptr != NULL)
            continue;
          int err = dev->ops->alloc_mem_obj (dev, mem, host_ptr);

          if ((err == CL_SUCCESS) && (mem->mem_host_ptr))
            break;
        }

      POCL_GOTO_ERROR_ON ((pocl_alloc_or_retain_mem_host_ptr (mem) != 0),
                          CL_OUT_OF_HOST_MEMORY,
                          "Cannot allocate backing memory!\n");
      mem->mem_host_ptr_version = 0;
      mem->latest_version = 0;
    }

  /* With CL_MEM_DEVICE_ADDRESS_EXT we must proactively allocate the device
     memory so it gets the fixed address range assigned, even if the buffer was
     never used. The address can be queried via clGetMemobjInfo() and used
     inside data structures. */
  if (flags & CL_MEM_DEVICE_ADDRESS_EXT)
    {
      POCL_MSG_PRINT_MEMORY (
          "Trying driver allocation for CL_MEM_DEVICE_ADDRESS_EXT\n");
      unsigned i;
      void *ptr = NULL;
      for (i = 0; i < context->num_devices; ++i)
        {
          cl_device_id dev = context->devices[i];
          assert (dev->ops->alloc_mem_obj != NULL);
          int err = 0;

          if (ptr != NULL && !(flags & CL_MEM_DEVICE_PRIVATE_EXT))
            {
              /* In the default case, we have the same ptr for all devices.
                 TODO: check that the devices have access to each
                 other's memories/address spaces. */
              assert (mem->device_ptrs[dev->global_mem_id].mem_ptr == NULL);
              mem->device_ptrs[dev->global_mem_id].mem_ptr = ptr;
            }
          else if (mem->device_ptrs[dev->global_mem_id].mem_ptr == NULL)
            {
              err = dev->ops->alloc_mem_obj (dev, mem, host_ptr);
              ptr = mem->device_ptrs[dev->global_mem_id].mem_ptr;
              POCL_GOTO_ERROR_ON (err != CL_SUCCESS, CL_OUT_OF_RESOURCES,
                                  "Out of device memory?");

              pocl_raw_ptr *item = calloc (1, sizeof (pocl_raw_ptr));
              POCL_RETURN_ERROR_ON ((item == NULL), NULL,
                                    "out of host memory\n");

              POCL_LOCK_OBJ (context);
              item->vm_ptr = NULL;
              item->dev_ptr = ptr;
              item->size = size;
              item->shadow_cl_mem = mem;
              DL_APPEND (context->raw_ptrs, item);
              POCL_UNLOCK_OBJ (context);

              POCL_MSG_PRINT_MEMORY ("Registered a CL_MEM_DEVICE_ADDRESS_EXT "
                                     "allocation with address '%p'.\n",
                                     ptr);
              if (!(flags & CL_MEM_DEVICE_PRIVATE_EXT))
                mem->device_ptrs[dev->global_mem_id].device_addr = ptr;
              else
                mem->device_ptrs[dev->global_mem_id].device_addr = NULL;
            }
        }
    }

  /* if COPY_HOST_PTR is present but no copying happened,
   * do the copy here */
  if ((flags & CL_MEM_COPY_HOST_PTR) && (mem->mem_host_ptr_version == 0))
    {
      POCL_GOTO_ERROR_ON ((pocl_alloc_or_retain_mem_host_ptr (mem) != 0),
                          CL_OUT_OF_HOST_MEMORY,
                          "Cannot allocate backing memory!\n");
      memcpy (mem->mem_host_ptr, host_ptr, size);
      mem->mem_host_ptr_version = 1;
      mem->latest_version = 1;
    }

  goto SUCCESS;

ERROR:
  if (mem)
  {
    if (mem->device_ptrs)
    {
      for (i = 0; i < context->num_devices; ++i)
        {
          cl_device_id dev = context->devices[i];
          pocl_mem_identifier *p = &mem->device_ptrs[dev->global_mem_id];
          if (p->mem_ptr)
            dev->ops->free (dev, mem);
        }
      POCL_MEM_FREE (mem->device_ptrs);
    }

    if (((flags & CL_MEM_USE_HOST_PTR) == 0) && mem->mem_host_ptr)
      POCL_MEM_FREE (mem->mem_host_ptr);

    POCL_MEM_FREE (mem);
  }

SUCCESS:
  if (errcode_ret)
    {
      *errcode_ret = errcode;
    }

  return mem;
}



CL_API_ENTRY cl_mem CL_API_CALL POname (clCreateBuffer) (
    cl_context context, cl_mem_flags flags, size_t size, void *host_ptr,
    cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
  cl_mem mem = NULL;
  int errcode = CL_SUCCESS;
  int host_ptr_is_svm = CL_FALSE;

  if ((flags & CL_MEM_USE_HOST_PTR) && host_ptr != NULL)
    {
      pocl_raw_ptr *item = pocl_find_raw_ptr_with_vm_ptr (context, host_ptr);
      if (item)
        {
          POCL_GOTO_ERROR_ON ((item->size < size), CL_INVALID_BUFFER_SIZE,
                              "The provided host_ptr is SVM pointer, "
                              "but the allocated SVM size (%zu) is smaller "
                              "then requested size (%zu)",
                              item->size, size);
          host_ptr_is_svm = CL_TRUE;
        }
    }

  mem = pocl_create_memobject (context, flags, size, CL_MEM_OBJECT_BUFFER,
                               NULL, host_ptr, host_ptr_is_svm, &errcode);
  if (mem == NULL)
    goto ERROR;

  TP_CREATE_BUFFER (context->id, mem->id);

  POname(clRetainContext)(context);

  POCL_MSG_PRINT_MEMORY ("Created Buffer %" PRIu64 " (%p), MEM_HOST_PTR: %p, "
                         "device_ptrs[0]: %p, SIZE %zu, FLAGS %" PRIu64 " \n",
                         mem->id, mem, mem->mem_host_ptr,
                         mem->device_ptrs[0].mem_ptr, size, flags);

  POCL_ATOMIC_INC (buffer_c);

ERROR:
  if (errcode_ret)
    *errcode_ret = errcode;

  return mem;
}
POsym (clCreateBuffer);

// TBC: We probably want to limit tensor rank - should it be
//      documented in the spec or queried at runtime?
#define MAX_TENSOR_RANK (8u)

// Check the tensor layout is well defined. Return non-zero if there
// is an error.
static int
check_tensor_layout (cl_uint rank, const cl_tensor_shape *shape,
                     const cl_tensor_layout_base *tlayout)
{
  // Checked already at check_tensor_desc().
  assert (rank > 0 && rank <= MAX_TENSOR_RANK);

  // '!tlayout' is same as tlayout->stype == CL_TENSOR_LAYOUT_OPAQUE.
  if (!tlayout || tlayout->stype == CL_TENSOR_LAYOUT_OPAQUE)
    {
      // TODO: check memory flags.
      //
      // * CL_MEM_{COPY,HOST}_host_ptr -> Error due to unspecified
      //   mapping of the host data to tensor coordinates.
      //
      // * CL_MEM_ALLOC_HOST_PTR -> Error for the same reason as for
      //   CL_MEM_{COPY,HOST}_host_ptr. Could be valid but not
      //   sensible as users may not know how the tensor elements are
      //   mapped to the allocation. Perhaps, we could support this
      //   case, if we extend the clGetMemObjectInfo() to return the
      //   datalayout the driver picked (and wants to expose)?
      return 0;
    }

  // Not currently supporting any tensor layout extensions.
  POCL_RETURN_ERROR_ON (tlayout->next, 1,
                        "Unsupported tensor layout extension.");

  switch (tlayout->stype)
    {
    case CL_TENSOR_LAYOUT_OPAQUE:
    default:
      return 0;
    case CL_TENSOR_LAYOUT_BLAS:
      {
        cl_tensor_layout_blas *blas_layout = (cl_tensor_layout_blas *)tlayout;

        POCL_RETURN_ERROR_ON (!blas_layout->leading_dims, 1,
                              "NULL leading_dims array!");
        POCL_RETURN_ERROR_ON (!blas_layout->leading_strides, 1,
                              "NULL leading_strides array!");

        // Check leading_dims array does not point out-of-rank dimensions
        // nor the same dimension index does not appear twice.
        //
        // tensor_rank == 4: leading_dims = {0, 2, 1} --> Ok.
        // tensor_rank == 4: leading_dims = {0, 4, 1} --> error.
        // tensor_rank == 4: leading_dims = {1, 1, 0} --> error.
        unsigned defined_dims = 0;
        const cl_tensor_dim *ld = blas_layout->leading_dims;
        for (unsigned i = 0; i < rank - 1; i++)
          {
            POCL_RETURN_ERROR_ON (ld[i] >= rank, 1,
                                  "out-of-bounds tensor dimension!");
            POCL_RETURN_ERROR_ON ((defined_dims & (1u << ld[i])), 1,
                                  "Dimension defined twice!");
            defined_dims |= (1u << ld[i]);
          }

        const size_t *ls = blas_layout->leading_strides;
        size_t prev_stride = 0;
        for (unsigned i = 0; i < rank - 1; i++)
          {
            // Check the stride configuration does not cause aliasing.
            POCL_RETURN_ERROR_ON (ls[i] <= shape[ld[i]] * prev_stride, 1,
                                  "Invalid stride value.");
            prev_stride = ls[i];
          }

        return 0;
      }
    }
  assert (!"Unreachable!");
}

// Checks validity the the tensor shape. Returns non-zero on error.
static int
check_tensor_desc (const cl_tensor_desc *tdesc)
{
  // Invalid to pass NULL tensor description in clCreateBufferWithProperties.
  POCL_RETURN_ERROR_ON ((!tdesc), 1, "tensor desc is NULL.");

  // Currently no tensor extensions.
  POCL_RETURN_ERROR_ON ((tdesc->stype != CL_TENSOR_DESC_BASE || tdesc->next),
                        1, "Unsupported tensor extension.");

  // TBC: Should there be upper limit for tensor rank?
  POCL_RETURN_ERROR_ON ((tdesc->rank > MAX_TENSOR_RANK), 1,
                        "Unsupported tensor rank.");

  POCL_RETURN_ERROR_ON ((!tdesc->shape), 1,
                        "Tensor shape array must not be NULL!");

  for (unsigned i = 0; i < tdesc->rank; i++)
    POCL_RETURN_ERROR_ON ((tdesc->shape[i] == 0), 1,
                          "Tensor shape must be fully specified!");

  POCL_RETURN_ERROR_ON (
      (check_tensor_layout (tdesc->rank, tdesc->shape,
                            (const cl_tensor_layout_base *)tdesc->layout)),
      1, "invalid tensor layout.");

  return 0;
}

static void *
duplicate (const void *src, size_t num_objects, size_t object_size)
{
  void *new_objects = calloc (num_objects, object_size);
  if (!new_objects)
    return NULL;
  memcpy (new_objects, src, object_size * num_objects);
  return new_objects;
}

#define DUPLICATE(source_ptr, num_objects, object_type)                       \
  duplicate ((source_ptr), (num_objects), sizeof (object_type));

// Duplicates the tensor description (deep copy). The 'tdesc' must be valid.
static cl_tensor_desc *
duplicate_tensor_desc (const cl_tensor_desc *tdesc)
{
  if (!tdesc)
    return NULL;

  assert (!tdesc->next && "UNIMPLEMENTED: deep copy of tensor extensions.");

  cl_tensor_desc *new_tdesc = DUPLICATE (tdesc, 1, cl_tensor_desc);
  cl_tensor_shape *new_shape
      = DUPLICATE (tdesc->shape, tdesc->rank, cl_tensor_dim);
  cl_tensor_layout_blas *new_layout = NULL;
  cl_tensor_dim *new_ld_dims = NULL;
  size_t *new_ld_strides = NULL;

  if (!new_tdesc || !new_shape)
    goto error;

  new_tdesc->shape = new_shape;
  if (!tdesc->layout)
    return new_tdesc;

  switch (((const cl_tensor_layout_base *)tdesc->layout)->stype)
    {
    default:
    case CL_TENSOR_LAYOUT_OPAQUE:
      return NULL;

    case CL_TENSOR_LAYOUT_BLAS:
      {
        cl_tensor_layout_blas *blas_layout
            = (cl_tensor_layout_blas *)tdesc->layout;
        new_layout = DUPLICATE (blas_layout, 1, cl_tensor_layout_blas);
        new_ld_dims = DUPLICATE (blas_layout->leading_dims, tdesc->rank - 1,
                                 cl_tensor_dim);
        new_ld_strides = DUPLICATE (blas_layout->leading_strides,
                                    tdesc->rank - 1, size_t);

        if (!new_layout || !new_ld_dims || !new_ld_strides)
          goto error;

        new_layout->leading_dims = new_ld_dims;
        new_layout->leading_strides = new_ld_strides;
        new_tdesc->layout = new_layout;
        return new_tdesc;
      }
    }
  assert (!"Unreachable!");
  return NULL;

error:
  free (new_tdesc);
  free (new_shape);
  free (new_layout);
  free (new_ld_dims);
  free (new_ld_strides);
  return NULL;
}

static cl_int
parse_properties (const cl_mem_properties *prop_ptr, cl_mem target)
{
  // Assuming cl_mem::num_properties and cl_mem::properties are zero prior
  // parsing.
  assert (target->num_properties == 0 && "Already parsed cl_mem properties?");

  if (!prop_ptr)
    {
      return CL_SUCCESS;
    }

  if (*prop_ptr == 0)
    {
      target->num_properties = 1;
      target->properties[0] = 0;
      return CL_SUCCESS;
    }

  while (*prop_ptr)
    {
      switch (*prop_ptr)
        {
        default:
          return CL_INVALID_PROPERTY;
        case CL_MEM_TENSOR:
          {
            const cl_tensor_desc *tdesc = (const cl_tensor_desc *)prop_ptr[1];
            prop_ptr += 2; // = CL_MEM_TENSOR and its value.

            POCL_RETURN_ERROR_ON ((check_tensor_desc (tdesc)),
                                  CL_INVALID_PROPERTY,
                                  "invalid tensor description.");

            target->tensor_desc = duplicate_tensor_desc (tdesc);
            POCL_RETURN_ERROR_ON (
                (!target->tensor_desc), CL_OUT_OF_HOST_MEMORY,
                "Couldn't allocate space for tensor description.");

            target->is_tensor = 1;
            return CL_SUCCESS;
          }
        }
    }
  assert (!"Unreachable!");
  return CL_OUT_OF_HOST_MEMORY;
}

CL_API_ENTRY cl_mem CL_API_CALL POname (clCreateBufferWithProperties)(
                               cl_context                context,
                               const cl_mem_properties * properties,
                               cl_mem_flags              flags,
                               size_t                    size,
                               void *                    host_ptr,
                               cl_int *                  errcode_ret)
CL_API_SUFFIX__VERSION_3_0
{
  int errcode;

  cl_mem mem_ret = POname(clCreateBuffer) (context, flags, size,
                                           host_ptr, errcode_ret);
  if (mem_ret == NULL)
    return NULL;

  if ((errcode = parse_properties (properties, mem_ret)) != CL_SUCCESS)
    {
      POname (clReleaseMemObject) (mem_ret);
      goto ERROR;
    }

  return mem_ret;

ERROR:
  if (errcode_ret)
    {
      *errcode_ret = errcode;
    }

  return NULL;
}
POsym (clCreateBufferWithProperties)
