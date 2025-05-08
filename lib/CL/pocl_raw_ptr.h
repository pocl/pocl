/* Copyright (c) 2025 Henry Linjamäki / Intel Finland Oy

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
#include "pocl.h"

#ifndef POCL_RAW_PTR_H
#define POCL_RAW_PTR_H

/**
 * Enumeration for raw buffer/pointer types managed by PoCL.
 */
typedef enum
{
  /* SVM from OpenCL 2.0. */
  POCL_RAW_PTR_SVM = 0,
  /* Intel USM extension. */
  POCL_RAW_PTR_INTEL_USM,
  /* cl_ext_buffer_device_address. */
  POCL_RAW_PTR_DEVICE_BUFFER
} pocl_raw_pointer_kind;

typedef struct _pocl_raw_ptr pocl_raw_ptr;
struct _pocl_raw_ptr
{
  /* The virtual address, if any.  NULL if there's none. */
  void *vm_ptr;
  /* The device address, if known. NULL if not. */
  void *dev_ptr;
  /* The owner device of the allocation, if any. Should be set to non-null for
     USM Device and for non-null 'dev_ptr' member. */
  cl_device_id device;

  size_t size;
  /* A cl_mem for internal bookkeeping and implicit buffer migration. */
  cl_mem shadow_cl_mem;

  /* The raw pointer/buffer API used for the allocation. */
  pocl_raw_pointer_kind kind;

  struct
  {
    cl_mem_alloc_flags_intel flags;
    unsigned alloc_type;
  } usm_properties;

  struct _pocl_raw_ptr *prev, *next;
};

#endif /* POCL_RAW_PTR_H */
