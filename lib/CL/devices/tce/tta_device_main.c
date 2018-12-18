/* tta_device_main.c - the main program for the tta devices executing ocl kernels

   Copyright (c) 2012-2018 Pekka Jääskeläinen
   
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
//#define DEBUG_TTA_DEVICE

#include <malloc.h>
#include <stdlib.h>

#include <lwpr.h>

#ifdef DEBUG_TTA_DEVICE
#include <stdio.h>
#endif

#ifndef __CBUILD__
#define __CBUILD__
#endif
#include "pocl_device.h"
#include "pocl_context.h"
#include "pocl_workgroup_func.h"

#define __local__ __attribute__((address_space(3)))
#define __global__ __attribute__((address_space(1)))
#define __constant__ __attribute__((address_space(2)))

typedef volatile __global__ __kernel_exec_cmd kernel_exec_cmd;
typedef __global__ __kernel_metadata kernel_metadata;

/**
 * Executes the work groups of the kernel command.
 */
static void tta_opencl_wg_execute(
    kernel_exec_cmd* cmd,
    void** args,
    int first_gidx, int last_gidx) {

    kernel_metadata *kernel = (kernel_metadata*)cmd->kernel;

    const int num_groups_x = cmd->num_groups[0];
    const int num_groups_y = (cmd->work_dim >= 2) ? (cmd->num_groups[1]) : 1;
    const int num_groups_z = (cmd->work_dim == 3) ? (cmd->num_groups[2]) : 1;

    struct pocl_context32 context;
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
#ifdef DEBUG_TTA_DEVICE
                lwpr_print_str("tta: ------------------- launching WG ");
                lwpr_print_int(gid_x); lwpr_print_str("-");
                lwpr_print_int(gid_y); lwpr_print_str("-");
                lwpr_print_int(gid_z); lwpr_print_str(" @ ");
                lwpr_print_int((unsigned)kernel->work_group_func);
                lwpr_newline();
#endif
                ((pocl_workgroup_func32)kernel->work_group_func)(
		     (uint8_t*)args, (uint8_t*)&context,
		     gid_x, gid_y, gid_z);
            } 
        }
    }
}

/**
 * Prepares a work group for execution and launches it.
 */
static void tta_opencl_wg_launch(kernel_exec_cmd* cmd) {

    void* args[MAX_KERNEL_ARGS];
    kernel_metadata *kernel = (kernel_metadata*)cmd->kernel;

    int num_groups_x = cmd->num_groups[0];
    int i, first_gid_x, last_gid_x;

    cmd->status = POCL_KST_RUNNING;

    /* single thread version: execute all work groups in
       a single trampoline call as fast as possible. 

       Do not create any threads. */
    for (int i = 0; i < kernel->num_args + kernel->num_locals; ++i) {
#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("tta: processing arg ");
        lwpr_print_int(i); 
        lwpr_print_str(" value: ");
        lwpr_print_int(cmd->args[i]);
        lwpr_newline();
#endif
        args[i] = (void*)cmd->args[i];
    }

#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("tta: ------------------- starting kernel\n");
#endif
        tta_opencl_wg_execute(cmd, args, 0, num_groups_x - 1);
#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("\ntta: ------------------- kernel finished\n");
#endif
        cmd->status = POCL_KST_FINISHED;
}

extern kernel_metadata _test_kernel_md;

/* The shared kernel_command object using which the device is controlled. */
#if !defined(_STANDALONE_MODE) || _STANDALONE_MODE == 0

kernel_exec_cmd *kernel_command_ptr = (kernel_exec_cmd*)KERNEL_EXE_CMD_OFFSET;

#ifndef _STANDALONE_MODE
#define _STANDALONE_MODE 0
#endif

#else

/* The kernel command is pregenerated in the standalone mode to reproduce
   an execution command. The command along with the input buffers and arguments
   is initialized in a separate .c file. */

extern kernel_exec_cmd kernel_command;

#endif

#if _STANDALONE_MODE == 1
void initialize_kernel_launch();
#endif

int main() {
    kernel_metadata *next_kernel;
    int work_dim = 1;
    size_t local_work_sizes[3] = {1, 0, 0};
    size_t global_work_sizes[3] = {2, 0, 0};

#ifdef DEBUG_TTA_DEVICE
    lwpr_print_str("tta: Hello from a TTA device\n");
    lwpr_print_str("tta: initializing the command objects\n");
#endif

#if _STANDALONE_MODE == 1
    initialize_kernel_launch();
    kernel_exec_cmd *next_command = &kernel_command;
#else
    kernel_exec_cmd *next_command = kernel_command_ptr;
#endif

    do {

#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("tta: waiting for commands\n");
#endif

        while (next_command->status != POCL_KST_READY)
            ;

        next_kernel = (kernel_metadata*)(next_command->kernel);

#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("tta: got a command to execute: ");
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

        /* In case this is the host-device setup (not the standalone mode),
           wait forever for commands from the host. Otherwise, execute the
           only command from the standalone binary and quit. */
    } while (1 && !_STANDALONE_MODE);

    return 0;
}
