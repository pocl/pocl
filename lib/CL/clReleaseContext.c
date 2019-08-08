/* OpenCL runtime library: clReleaseContext()

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "pocl_cl.h"
#include "devices/devices.h"
#include "pocl_runtime_config.h"

#include <unistd.h>

extern unsigned cl_context_count;
extern pocl_lock_t pocl_context_handling_lock;

CL_API_ENTRY cl_int CL_API_CALL
POname(clReleaseContext)(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
  int new_refcount;
  if (!context->valid)
    {
      POCL_MEM_FREE (context);
      return CL_SUCCESS;
    }

  POCL_LOCK (pocl_context_handling_lock);

  POCL_MSG_PRINT_REFCOUNTS ("Release Context \n");
  POCL_RELEASE_OBJECT(context, new_refcount);
  if (new_refcount == 0)
    {
      POCL_MSG_PRINT_REFCOUNTS ("Free Context %p\n", context);
      /* The context holds references to all its devices,
         memory objects, command-queues etc. Release the
         references and let the objects to get freed. */
      unsigned i;
      for (i = 0; i < context->num_devices; ++i)
        {
          POname(clReleaseDevice) (context->devices[i]);
        }
      POCL_MEM_FREE(context->devices);
      POCL_MEM_FREE(context->properties);

      for (i = 0; i < NUM_OPENCL_IMAGE_TYPES; ++i)
        POCL_MEM_FREE (context->image_formats[i]);

      POCL_DESTROY_OBJECT (context);
      POCL_MEM_FREE(context);

      /* see below on why we don't call uninit_devices here anymore */
      --cl_context_count;
    }

  POCL_UNLOCK (pocl_context_handling_lock);

  return CL_SUCCESS;
}
POsym(clReleaseContext)

/*
  This is a workaround for a problem explained below.

  Note this is function *only* used by pocl tests, is subject to change without notice,
  and only available when compiling in no-ICD mode (= test is directly linked to libpocl).
*/

void
pocl_check_uninit_devices ()
{
  /* This is a sleep with rather arbitrary amount, but it's required.
   * See the full explanation below, but basically,
   * this gives the pocl driver a bit of time to cleanup
   * and go idle, so we can safely call pocl_uninit_devices() from
   * a user thread. */

  int do_uninit = pocl_get_bool_option ("POCL_ENABLE_UNINIT", 0);
  if (!do_uninit)
    return;

  usleep (100000);

  POCL_LOCK (pocl_context_handling_lock);
  int do_cleanup = (cl_context_count == 0);
  POCL_UNLOCK (pocl_context_handling_lock);

  if (do_cleanup)
    {
      POCL_MSG_PRINT_REFCOUNTS (
          "Zero contexts left, calling pocl_uninit_devices\n");
      pocl_uninit_devices ();
    }
  else
    POCL_MSG_ERR ("Contexts remaining!! \n");
}


/*

There used to be code that called pocl_uninit_devices() from clReleaseContext()
to un-initialize all devices when cl_context_count reached 0.
Unfortunately it ignores some realities of OpenCL.

What happens (usually) is:

  Application     Pocl driver thread(s)
-----------------------------------------------------------------------
  clEnqueueSomething()
                            process command

  clFinish()
   .. in libpocl: wait for notify from driver

                            process any commands
                            notify user app thread (via pthread_cond_signal) that command queue is empty
                            release any cl_objects used by last command (decrease refcounts)

  clReleaseBuffer() etc
  clReleaseCommandQueue()
  clReleaseContext()
   .. in libpocl: decrease context's refcount
     .. context refcount is 0 -> call pocl_uninit_devices()

                            uninit_device() called, unitialize device, stop driver threads etc

What can happen instead is:

  Application               Pocl driver thread(s)
-----------------------------------------------------------------------
  clEnqueueSomething()
                            process command

  clFinish()
   .. in libpocl: wait for notify from driver

                            process any commands
                            notify user app thread (via pthread_cond_signal) that command queue is empty
                            driver thread gets unscheduled by OS

  clReleaseBuffer() etc
  clReleaseCommandQueue()
  clReleaseContext()
   .. in libpocl: decrease context's refcount
     .. context refcount is NOT 0 (because driver hasn't clReleased cl_objects used by last command yet)

** POINT X ** (see below)

                            release any cl_objects used by commands:
                            clReleaseEvent()
                            clReleaseCommandQueue()
                            clReleaseContext(): now refcount is 0
                                -> calls pocl_uninit_devices() ** from the driver thread **
The symptoms are
  1) pocl_uninit_devices() is called from a driver thread - it would have to release itself
  2) at "POINT X" the application usually calls exit() which also causes the driver threads to terminate,
     leaving unreleased objects, triggering sanitizer checks

The root of the problem is that the driver notifies the user app when the command finished executing,
but before it has done the clRelease* calls on commands/buffers etc of the command's event. This *may*
be possible to fix but possibly requires significant changes to "event complete" handling code.

The bigger problem is that if we can't guarantee clReleaseContext() reaches refcount 0 only in user
app's thread, we either have to refactor drivers to be able to handle this,
or solve uninit() in some entirely different way (or ignore it completely).

*/
