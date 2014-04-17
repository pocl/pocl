/* pocl_device.h - global pocl declarations to be used in the device binaries in
                   case applicable by the target

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

#ifndef POCL_DEVICE_H
#define POCL_DEVICE_H

#include <stddef.h>
#include <stdint.h>

/* This struct is device-specific because of the varying size_t
   width. In case we fix it to something, unnecessary casts are
   generated. */
struct pocl_context {
  uint32_t work_dim;
  size_t num_groups[3];
  size_t group_id[3];
  size_t global_offset[3];
  
};

typedef void (*pocl_workgroup) (void **, struct pocl_context *);

#define MAX_KERNEL_ARGS 64
#define MAX_KERNEL_NAME_LENGTH 64

/* Metadata of a single kernel stored in the device.*/
typedef struct {
    const char name[MAX_KERNEL_NAME_LENGTH];
    unsigned short num_args;
    unsigned short num_locals;
#if 0
    /* TODO: remove these if no (private) branch needs them
       anymore. Also from pocl_llvm_api.cc */
    unsigned short alocal_sizes[MAX_KERNEL_ARGS];
#endif
    pocl_workgroup work_group_func;
} __kernel_metadata;

#define ALIGN4 __attribute__ ((aligned (4)))
#define ALIGN8 __attribute__ ((aligned (8)))

/* A kernel invocation command. */
typedef struct {
    /* The execution status of this queue slot. */
    uint32_t status ALIGN8;
    /* The kernel to execute. Points to the metadata in the device global
       memory. It will be casted to a __kernel_metadata* */
    uint32_t kernel ALIGN8;
    /* Pointers to the kernel arguments in the global memory. Will be
       casted to 32 bit void* */
    uint32_t args[MAX_KERNEL_ARGS] ALIGN8;
    /* Sizes of the dynamically allocated local buffers. */
/*    uint32_t dynamic_local_arg_sizes[MAX_KERNEL_ARGS] ALIGN4; */
    /* Number of dimensions in the work space. */
    uint32_t work_dim ALIGN4;
    uint32_t num_groups[3] ALIGN4;
    uint32_t global_offset[3] ALIGN4;
} __kernel_exec_cmd;

/* Kernel execution statuses. */

/* The invocation entry is free to use. */
#define POCL_KST_FREE     1
/* The kernel structure has been populated and is waiting to be
   executed. */
#define POCL_KST_READY    2
/* The kernel is currently running in the device. */
#define POCL_KST_RUNNING  3
/* The kernel has finished execution. The results can be collected and the
   execution entry be freed (by writing POCL_KST_FREE to the status). */
#define POCL_KST_FINISHED 4

#endif
