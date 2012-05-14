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

#define DEBUG_TTA_DEVICE

#include <malloc.h>
#include <stdlib.h>

#include <lwpr.h>

#ifdef DEBUG_TTA_DEVICE
#include <stdio.h>
#endif

#include "pocl_device.h"

#define __local__ __attribute__((address_space(0)))
#define __global__ __attribute__((address_space(3)))
#define __constant__ __attribute__((address_space(3)))

typedef volatile __global__ __kernel_exec_cmd kernel_exec_cmd;

/**
 * Executes the work groups of the kernel command.
 */
static void tta_opencl_wg_execute(
    kernel_exec_cmd* cmd,
    void** args,
    int first_gidx, int last_gidx) {

    __kernel_metadata *kernel = (__kernel_metadata*)cmd->kernel;

    const int num_groups_x = cmd->num_groups[0];
    const int num_groups_y = (cmd->work_dim >= 2) ? (cmd->num_groups[1]) : 1;
    const int num_groups_z = (cmd->work_dim == 3) ? (cmd->num_groups[2]) : 1;

    struct pocl_context context;
    context.work_dim = cmd->work_dim;
    context.num_groups[0] = cmd->num_groups[0];
    context.num_groups[1] = cmd->num_groups[1];
    context.num_groups[2] = cmd->num_groups[2];
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
                lwpr_print_int(kernel->work_group_func);
                lwpr_newline();
#endif
                kernel->work_group_func (args, &context);
            } 
        }
    }
}

/**
 * Prepares a work group for execution and launches it.
 *
 * Allocates the local memory buffers. 
 * TODO: replace the malloc with a more light weight bufalloc.
 */
static void tta_opencl_wg_launch(kernel_exec_cmd* cmd) {

    void* args[MAX_KERNEL_ARGS];
    __kernel_metadata *kernel = (__kernel_metadata*)cmd->kernel;

    int num_groups_x = cmd->num_groups[0];
    int i, first_gid_x, last_gid_x;

    /* single thread version: execute all work groups in
       a single trampoline call as fast as possible. 

       Do not create any threads. */
    /* allocate data for the local buffers. TODO: this is not
       thread safe due to the shared 'args' struct. In MT we need
       to copy the local pointers for each invocation. */
    for (int i = 0; i < kernel->num_args; ++i) {
#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("tta: processing arg ");
        lwpr_print_int(i); lwpr_print_str(" of ");
        lwpr_print_int(kernel->num_args); 
        lwpr_print_str(" value: ");
        lwpr_print_int(cmd->args[i]);
        lwpr_newline();
#endif
        if (!kernel->arg_is_local[i]) {
            args[i] = (unsigned)cmd->args[i];
            continue;
        }
        /* TODO: this is broken. It should store a pointer to the buffer pointer
           instead of the buffer pointer directly. */
        args[i] = (uint32_t)malloc(cmd->dynamic_local_arg_sizes[i]);
        if (args[i] == 0) {
#ifdef DEBUG_TTA_DEVICE
            lwpr_print_str("tta: out of memory while allocating the local buffers\\n");
#endif
            exit(1);
        }
    }
    /* Allocate data for the automatic local buffers which have
       been converted to pointer arguments in the kernel launcher
       function. They are the last arguments always. */
        for (int i = kernel->num_args; 
             0 && i < kernel->num_args + kernel->num_locals; 
             ++i) {
            /* TODO: this is broken. It should store a pointer to the buffer pointer
               instead of the buffer pointer directly. */
            args[i] = (uint32_t)malloc(kernel->alocal_sizes[i - kernel->num_args]);
#if 0 && defined(DEBUG_TTA_DEVICE)
            lwpr_print_str("tta: allocated ");
            iprintf("tta: allocated %%d bytes for the automatic local arg %%d at %%x\\n", 
                    kernel->alocal_sizes[i - kernel->num_args], i, *(unsigned int*)args[i]);
#endif
            if (args[i] == 0) {
#ifdef DEBUG_TTA_DEVICE
                lwpr_print_str("tta: out of memory while allocating the local buffers\\n");
#endif
                exit(1);
            }          
        }

#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("tta: ------------------- starting kernel\n");
#endif
        tta_opencl_wg_execute(cmd, args, 0, num_groups_x - 1);
#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("\ntta: ------------------- kernel finished\n");
#endif
        /* free the local buffers */
        for (int i = 0; i < kernel->num_args; ++i) {
            if (!kernel->arg_is_local[i]) continue;

            /* TODO: this is broken. It should store a pointer to the buffer pointer
               instead of the buffer pointer directly. */
            free(*((void**)args[i]));
            //free(args[1]);
        }
        /* free the automatic local buffers */
        for (int i = kernel->num_args;
             i < kernel->num_args + kernel->num_locals;
             ++i) {
#if 0
            iprintf("tta: freed automatic local arg %%d\\n", i);
#endif
            /* TODO: this is broken. It should store a pointer to the buffer pointer
               instead of the buffer pointer directly. */
            free(*((void**)args[i]));
            //free(args[1]);
        }
}

extern __kernel_metadata _test_kernel_md;


/* The shared kernel_command object using which the device is controlled. */
kernel_exec_cmd kernel_command;

static void init_command_objects() {
    kernel_command.status = POCL_KST_FREE;
}

static kernel_exec_cmd* wait_for_command() {
    while (kernel_command.status != POCL_KST_READY) 
        ;
    kernel_command.status = POCL_KST_RUNNING;
    return &kernel_command;
}

int main() {
    kernel_exec_cmd *next_command;
    __kernel_metadata *next_kernel;
    size_t dynamic_local_arg_sizes[MAX_KERNEL_ARGS];
    int work_dim = 1;
    size_t local_work_sizes[3] = {1, 0, 0};
    size_t global_work_sizes[3] = {2, 0, 0};

#ifdef DEBUG_TTA_DEVICE
    lwpr_print_str("tta: Hello from a TTA device\n");
    lwpr_print_str("tta: initializing the command objects\n");
#endif

    init_command_objects();

    do {

#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("tta: waiting for commands\n");
#endif

        next_command = wait_for_command();

        next_kernel = (__kernel_metadata*)next_command->kernel;

#ifdef DEBUG_TTA_DEVICE
        iprintf("tta: got a command to execute '%s' with dim %lu num_groups:  %lu-%lu-%lu global_offset: %lu-%lu-%lu\n",
                next_kernel->name, 
                next_command->work_dim,
                next_command->num_groups[0],
                next_command->num_groups[1],
                next_command->num_groups[2],
                next_command->global_offset[0],
                next_command->global_offset[1],
                next_command->global_offset[2]);
#endif

        tta_opencl_wg_launch(next_command);
        kernel_command.status = POCL_KST_FINISHED;   

    } while (1);

    return 0;
}
