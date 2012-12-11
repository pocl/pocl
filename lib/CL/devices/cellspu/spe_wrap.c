/* spe_wrap.c - wrapper for executingOCL kernels on SPUs. Derived from:
   tta_device_main.c

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

#define DEBUG_SPU_DEVICE

#include "include/pocl_device.h"
#include <stdlib.h>

#ifdef DEBUG_SPU_DEVICE
#include <stdio.h>
#endif

//#define __local__ __attribute__((address_space(0)))
//#define __global__ __attribute__((address_space(3)))
//#define __constant__ __attribute__((address_space(3)))

typedef volatile __kernel_exec_cmd kernel_exec_cmd;
typedef __kernel_metadata kernel_metadata;

// Placeholder for the "global" memory.
// this is managed by the host cellpsu driver.
// Pointers to the buffers are given in the 
// __kernel_exec_cmd structure
char _ocl_buffer[128*1024];

/* The shared kernel_command object using which the device is controlled. */
kernel_exec_cmd kernel_command;

// This is a pointer to *the* kernel in this binary.
// It gets #define'd at compile/link time by spu-gcc in 
// pocl_cellspu_run.
void _KERNEL(void**, struct pocl_context*);

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

    struct pocl_context context;
    context.work_dim = cmd->work_dim;
    context.num_groups[0] = cmd->num_groups[0];
    context.num_groups[1] = cmd->num_groups[1];
    context.num_groups[2] = cmd->num_groups[2];
    context.global_offset[0] = cmd->global_offset[0];
    context.global_offset[1] = cmd->global_offset[1];
    context.global_offset[2] = cmd->global_offset[2];
             
#error TODO: these values seem to be garbled. Check the alignment of the kernel_cmd_structure on PPU vs. SPU 

    for (unsigned gid_x = first_gidx; gid_x <= last_gidx; gid_x++) { 
        for (unsigned gid_y = 0; gid_y < num_groups_y; gid_y++) { 
            for (unsigned gid_z = 0; gid_z < num_groups_z; gid_z++) {
                context.group_id[0] = gid_x;
                context.group_id[1] = gid_y;
                context.group_id[2] = gid_z;
#ifdef DEBUG_SPU_DEVICE
                printf("SPU: ------------------- launching WG \n");
		//TODO: print some interesting info about workgroup
#endif
		//This is how it should be done (and was with the TTA)
                //kernel->work_group_func (args, &context);
		//While waiting for the implementation, run a kludge:
		_KERNEL(args, &context);
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
    kernel_metadata *kernel = (kernel_metadata*)cmd->kernel;

    int num_groups_x = cmd->num_groups[0];
    int i, first_gid_x, last_gid_x;

    /* single thread version: execute all work groups in
       a single trampoline call as fast as possible. 

       Do not create any threads. */
    for (int i = 0; i < kernel->num_args + kernel->num_locals; ++i) {
#ifdef DEBUG_SPU_DEVICE
        printf("SPU: processing arg %d", i);
        printf(" value: %x\n", cmd->args[i]);
#endif
        args[i] = (void*)cmd->args[i];
    }

#ifdef DEBUG_SPU_DEVICE
        printf("SPU: ------------------- starting kernel %s", kernel->name);
#endif
        tta_opencl_wg_execute(cmd, args, 0, num_groups_x - 1);
#ifdef DEBUG_SPU_DEVICE
        printf("\nSPU: ------------------- kernel finished\n");
#endif
}

extern kernel_metadata _test_kernel_md;


static kernel_exec_cmd* wait_for_command() {
    printf("addy of status is %x. offset is %d\n", (int)&(kernel_command.status), ((int)&(kernel_command.status) - (int)&kernel_command));
    printf("status is %x\n", kernel_command.status);
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

#ifdef DEBUG_SPU_DEVICE
    printf("SPU: main() starts\n");
#endif

    do {

#ifdef DEBUG_SPU_DEVICE
        printf("SPU: waiting for commands\n");
#endif

        next_command = wait_for_command();

        next_kernel = (kernel_metadata*)next_command->kernel;

#ifdef DEBUG_SPU_DEVICE
	printf("SPU: got a kernel to execute\n");
	//TODO: print info about kernel
#endif
        tta_opencl_wg_launch(next_command);
        kernel_command.status = POCL_KST_FINISHED;   

    } while (1);

    return 0;
}

