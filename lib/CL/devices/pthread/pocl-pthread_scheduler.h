/* pocl-pthread_scheduler.h - kernel/workgroup scheduler for native 
   pthreaded device.

   Copyright (c) 2015 Ville Korhonen, Tampere University of Technology
   
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

#ifndef POCL_PTHREAD_SCHEDULER_H
#define POCL_PTHREAD_SCHEDULER_H

#include "pocl-pthread_utils.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

typedef struct pool_thread_data thread_data;

/* Initializes scheduler. Must be called before any kernel enqueue */
void pthread_scheduler_init (size_t num_worker_threads);

void pthread_scheduler_uinit ();

/* Gives ready-to-execute command for scheduler */
void pthread_scheduler_push_command (_cl_command_node *cmd);

void pthread_scheduler_push_kernel (kernel_run_command *run_cmd);

/* blocks until given command queue is empty == finished */
void pthread_scheduler_wait_cq (cl_command_queue cq);

void pthread_scheduler_release_host ();

void pthread_scheduler_get_work (thread_data *td, _cl_command_node **cmd_ptr);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
