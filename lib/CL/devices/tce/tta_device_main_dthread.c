/* tta_device_main.c - the main program for the tta devices executing ocl kernels

   Copyright (c) 2012 Pekka Jääskeläinen / Tampere University of Technology
   
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

/* Note: Most of the debug code is broken because lwpr_print_str() only works
 *       with char* to local address space */
#define DEBUG_TTA_DEVICE

#include <malloc.h>
#include <stdlib.h>
#include <dthread.h>
#include <lwpr.h>

#ifdef DEBUG_TTA_DEVICE
#include <stdio.h>
#endif

#include "pocl_device.h"

#define __local__ __attribute__((address_space(3)))
#define __global__ __attribute__((address_space(1)))
#define __constant__ __attribute__((address_space(2)))

typedef volatile __global__ __kernel_exec_cmd kernel_exec_cmd;
typedef __global__ __kernel_metadata kernel_metadata;

struct wg_thread_arg {
  kernel_exec_cmd* cmd;
  int first_gid_x; 
  int last_gid_x;
};

int min(int a, int b) {
    if (a < b) return a;
    else return b;
}

/**
 * Executes the work groups of the kernel command.
 */
static void *wg_thread(void *targ) {
    struct wg_thread_arg *targs = (struct wg_thread_arg*)targ;
    kernel_exec_cmd *cmd = targs->cmd;
    int first_gidx = targs->first_gid_x;
    int last_gidx = targs->last_gid_x;
    kernel_metadata *kernel = (kernel_metadata*)cmd->kernel;

    void* args[MAX_KERNEL_ARGS];

    /* Copy the kernel function arguments from the global memory 
       to the stack in the local memory. */
    for (int i = 0; i < kernel->num_args + kernel->num_locals; ++i) {
        args[i] = (void*)cmd->args[i];
    }

    const int num_groups_x = cmd->num_groups[0];
    const int num_groups_y = (cmd->work_dim >= 2) ? (cmd->num_groups[1]) : 1;
    const int num_groups_z = (cmd->work_dim == 3) ? (cmd->num_groups[2]) : 1;

    struct pocl_context32 context;
    context.work_dim = cmd->work_dim;
    context.num_groups[0] = cmd->num_groups[0];
    context.num_groups[1] = cmd->num_groups[1];
    context.num_groups[2] = cmd->num_groups[2];
    context.local_size[0] = cmd->local_size[0];
    context.local_size[1] = cmd->local_size[1];
    context.local_size[2] = cmd->local_size[2];
    context.global_offset[0] = cmd->global_offset[0];
    context.global_offset[1] = cmd->global_offset[1];
    context.global_offset[2] = cmd->global_offset[2];

    for (unsigned gid_x = first_gidx; gid_x <= last_gidx; gid_x++) { 
        for (unsigned gid_y = 0; gid_y < num_groups_y; gid_y++) { 
            for (unsigned gid_z = 0; gid_z < num_groups_z; gid_z++) {
                context.group_id[0] = gid_x;
                context.group_id[1] = gid_y;
                context.group_id[2] = gid_z;
#ifdef DEBUG_TTA_DEVICE
                lwpr_print_str("tta: ------------------- launching WG ");
                lwpr_print_int(gid_x); lwpr_print_str("-");
                lwpr_print_int(gid_y); lwpr_print_str("-");
                lwpr_print_int(gid_z); lwpr_print_str(" @ ");
                lwpr_print_int((unsigned)kernel->work_group_func);
                lwpr_newline();
#endif
                kernel->work_group_func (args, &context);
            } 
        }
    }
    return NULL;
}

#define MAX_WG_THREADS 128
dthread_t wg_threads[MAX_WG_THREADS];

/**
 * Prepares a work group for execution and launches it.
 */
static void tta_opencl_wg_launch(kernel_exec_cmd* cmd) {

    int num_groups_x = cmd->num_groups[0];
    int i, first_gid_x, last_gid_x;
    int thread_count = min(min(num_groups_x, dthread_get_core_count()), MAX_WG_THREADS);
    int wgs_per_thread = num_groups_x / thread_count;
    int leftover_wgs = num_groups_x - wgs_per_thread * thread_count;
    wgs_per_thread += leftover_wgs / thread_count;
    leftover_wgs = num_groups_x - wgs_per_thread * thread_count;

#ifdef DEBUG_TTA_DEVICE
    lwpr_print_str("tta: ------------------- starting kernel ");
    puts(kernel->name);
#endif
    first_gid_x = 0;
    last_gid_x = wgs_per_thread - 1;
    for (i = 0; i < thread_count; 
         ++i, first_gid_x += wgs_per_thread, last_gid_x += wgs_per_thread) {
        int status;
        struct wg_thread_arg arg;
        dthread_attr_t attr;

        if (i + 1 == thread_count) last_gid_x += leftover_wgs;

        arg.cmd = cmd;
        arg.first_gid_x = first_gid_x;
        arg.last_gid_x = last_gid_x;

        dthread_attr_init(&attr);
        dthread_attr_setargs(&attr, &arg, sizeof(arg));
        status = dthread_create(&wg_threads[i], &attr, wg_thread);
        /* Assume there's always enough space in the STT. */
        if (status) {
            exit(-1);
        } 
    }

   for (int i = 0; i < thread_count; i++){ 
       dthread_join(wg_threads[i], NULL);
   }

#ifdef DEBUG_TTA_DEVICE
    lwpr_print_str("\ntta: ------------------- kernel finished\n");
#endif
}

extern kernel_metadata _test_kernel_md;

/* The shared kernel_command object using which the device is controlled. */
kernel_exec_cmd kernel_command;

static kernel_exec_cmd* wait_for_command() {
    while (kernel_command.status != POCL_KST_READY) 
        ;
    kernel_command.status = POCL_KST_RUNNING;
    return &kernel_command;
}

int main() {
    kernel_exec_cmd *next_command;
    kernel_metadata *next_kernel;
    size_t dynamic_local_arg_sizes[MAX_KERNEL_ARGS];
    int work_dim = 1;
    size_t local_work_sizes[3] = {1, 0, 0};
    size_t global_work_sizes[3] = {2, 0, 0};

#ifdef DEBUG_TTA_DEVICE
    lwpr_print_str("tta: Hello from a TTA device\n");
    lwpr_print_str("tta: initializing the command objects\n");
#endif

    do {

#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("tta: waiting for commands\n");
#endif

        next_command = wait_for_command();

        next_kernel = (kernel_metadata*)next_command->kernel;

#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("tta: got a command to execute: ");
        lwpr_print_str(next_kernel->name);
        lwpr_print_str(" with ");
        lwpr_print_int(next_command->work_dim);
        lwpr_print_str(" dimensions. num_groups ");
        lwpr_print_int(next_command->num_groups[0]);
        lwpr_print_str("-"),
        lwpr_print_int(next_command->num_groups[1]);
        lwpr_print_str("-"),
        lwpr_print_int(next_command->num_groups[2]);
        lwpr_print_str(" dimensions. global offset ");
        lwpr_print_int(next_command->global_offset[0]);
        lwpr_print_str("-"),
        lwpr_print_int(next_command->global_offset[1]);
        lwpr_print_str("-"),
        lwpr_print_int(next_command->global_offset[2]);
        lwpr_newline();
#endif
        tta_opencl_wg_launch(next_command);
        kernel_command.status = POCL_KST_FINISHED;   

    } while (1);

    return 0;
}
