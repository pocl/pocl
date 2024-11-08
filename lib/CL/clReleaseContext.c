/* OpenCL runtime library: clReleaseContext()

   Copyright (c) 2011 Universidad Rey Juan Carlos
                 2011-2019 pocl developers
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

#include "devices/devices.h"
#include "pocl_runtime_config.h"
#include "pocl_util.h"

#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

#ifdef ENABLE_RDMA
#include "pocl_rdma.h"
#endif

#ifdef _MSC_VER
#include "vccompat.hpp"
#else
#include <unistd.h>
#endif

extern unsigned cl_context_count;
extern pocl_lock_t pocl_context_handling_lock;

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseContext)(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
  POCL_RETURN_ERROR_COND ((!IS_CL_OBJECT_VALID (context)), CL_INVALID_CONTEXT);

  int new_refcount;
  POCL_LOCK (pocl_context_handling_lock);
  POCL_LOCK_OBJ (context);
  POCL_RELEASE_OBJECT_UNLOCKED (context, new_refcount);
  POCL_MSG_PRINT_REFCOUNTS ("Release Context %" PRId64 " (%p), Refcount: %d\n",
                            context->id, context, new_refcount);

  if (new_refcount == 0)
    {
      if (context->destructor_callbacks)
        {
          pocl_context_cb_push (context);
          POCL_UNLOCK_OBJ (context);
          POCL_UNLOCK (pocl_context_handling_lock);
          return CL_SUCCESS;
        }
      POCL_UNLOCK_OBJ (context);
      VG_REFC_ZERO (context);

      POCL_ATOMIC_DEC (context_c);

      POCL_MSG_PRINT_REFCOUNTS ("Free Context %" PRId64 " (%p)\n", context->id,
                                context);

      /* The context holds references to all its devices,
         memory objects, command-queues etc. Release the
         references and let the objects to get freed. */
      unsigned i;
      for (i = 0; i < context->num_devices; ++i)
        {
          cl_device_id dev = context->devices[i];
          if (context->default_queues && context->default_queues[i])
            POname (clReleaseCommandQueue) (context->default_queues[i]);
          if (dev->ops->free_context)
            dev->ops->free_context (dev, context);
        }


      for (i = 0; i < context->num_create_devices; ++i)
      {
        POname (clReleaseDevice) (context->create_devices[i]);
      }

      POCL_MEM_FREE (context->create_devices);
      POCL_MEM_FREE (context->default_queues);
      POCL_MEM_FREE (context->devices);
      POCL_MEM_FREE (context->properties);

      for (i = 0; i < NUM_OPENCL_IMAGE_TYPES; ++i)
        POCL_MEM_FREE (context->image_formats[i]);

#ifdef ENABLE_LLVM
      pocl_llvm_release_context (context);
#endif

      POCL_DESTROY_OBJECT (context);
      POCL_MEM_FREE(context);

      /* see below on why we don't call uninit_devices here anymore */
      --cl_context_count;
    }
  else
    {
      VG_REFC_NONZERO (context);
      POCL_UNLOCK_OBJ (context);
    }

  POCL_UNLOCK (pocl_context_handling_lock);

  return CL_SUCCESS;
}
POsym(clReleaseContext)

void
pocl_check_uninit_devices ()
{
  int do_uninit = pocl_get_bool_option ("POCL_ENABLE_UNINIT", 0);
  if (!do_uninit)
    return;

  POCL_LOCK (pocl_context_handling_lock);
  if (cl_context_count == 0)
    {
      POCL_MSG_PRINT_REFCOUNTS (
          "Zero contexts left, calling pocl_uninit_devices\n");
      pocl_uninit_devices ();
#ifdef ENABLE_LLVM
      UnInitializeLLVM ();
#endif
    }
  else
    {
      POCL_MSG_ERR ("Alive contexts remaining, cannot uninit. \n");
      POCL_MSG_ERR ("Contexts alive: %zu\n", POCL_ATOMIC_LOAD (context_c));
      if (POCL_ATOMIC_LOAD (queue_c) > 0)
        POCL_MSG_ERR ("Queues alive: %zu\n", POCL_ATOMIC_LOAD (queue_c));
      if (POCL_ATOMIC_LOAD (buffer_c) > 0)
        POCL_MSG_ERR ("Buffers alive: %zu\n", POCL_ATOMIC_LOAD (buffer_c));
      if (POCL_ATOMIC_LOAD (svm_buffer_c) > 0)
        POCL_MSG_ERR ("SVM buffers alive: %zu\n",
                      POCL_ATOMIC_LOAD (svm_buffer_c));
      if (POCL_ATOMIC_LOAD (usm_buffer_c) > 0)
        POCL_MSG_ERR ("USM buffers alive: %zu\n",
                      POCL_ATOMIC_LOAD (usm_buffer_c));
      if (POCL_ATOMIC_LOAD (image_c) > 0)
        POCL_MSG_ERR ("Images alive: %zu\n", POCL_ATOMIC_LOAD (image_c));
      if (POCL_ATOMIC_LOAD (program_c) > 0)
        POCL_MSG_ERR ("Programs alive: %zu\n", POCL_ATOMIC_LOAD (program_c));
      if (POCL_ATOMIC_LOAD (kernel_c) > 0)
        POCL_MSG_ERR ("Kernels alive: %zu\n", POCL_ATOMIC_LOAD (kernel_c));
      if (POCL_ATOMIC_LOAD (sampler_c) > 0)
        POCL_MSG_ERR ("Samplers alive: %zu\n", POCL_ATOMIC_LOAD (sampler_c));
      if (POCL_ATOMIC_LOAD (event_c) > 0)
        POCL_MSG_ERR ("Command events alive: %zu\n",
                      POCL_ATOMIC_LOAD (event_c));
      if (POCL_ATOMIC_LOAD (uevent_c) > 0)
        POCL_MSG_ERR ("User events alive: %zu\n", POCL_ATOMIC_LOAD (uevent_c));
    }

  POCL_UNLOCK (pocl_context_handling_lock);
}
