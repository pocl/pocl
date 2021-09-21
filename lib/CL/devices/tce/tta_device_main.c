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

#include <stdlib.h>

#ifdef DEBUG_TTA_DEVICE
#include <lwpr.h>
#include <stdio.h>
#endif

#ifndef __CBUILD__
#define __CBUILD__
#endif
#include "pocl_device.h"
#include "pocl_context.h"
#include "pocl_workgroup_func.h"

#define AQL_PACKET_INVALID (1)
#define AQL_PACKET_KERNEL_DISPATCH (2)
#define AQL_PACKET_BARRIER_AND (3)
#define AQL_PACKET_AGENT_DISPATCH (4)
#define AQL_PACKET_BARRIER_OR (5)
#define AQL_PACKET_BARRIER (1 << 8)
#define AQL_PACKET_LENGTH (64)

#define __cq__ __attribute__ ((address_space (3)))
#define __buffer__ __attribute__ ((address_space (1)))

#ifndef _STANDALONE_MODE
#define _STANDALONE_MODE 0
#endif

#define MEM_SIZE 2048
#define QUEUE_LENGTH 2

#define AQL_OFFSET (MEM_SIZE - QUEUE_LENGTH * AQL_PACKET_LENGTH)

typedef void (*pocl_workgroup_func32_argb) (
    void __buffer__ * /* args */, void __buffer__ * /* pocl_context */,
    uint /* group_x */, uint /* group_y */, uint /* group_z */);

struct AQLDispatchPacket
{
  uint16_t header;
  uint16_t dimensions;

  uint16_t workgroup_size_x;
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;

  uint16_t reserved0;

  uint32_t grid_size_x;
  uint32_t grid_size_y;
  uint32_t grid_size_z;

  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint32_t kernel_object_low;
  uint32_t kernel_object_high;
  uint32_t kernarg_address_low;
  uint32_t kernarg_address_high;

  uint32_t reserved1;
  uint32_t reserved2;

  uint32_t completion_signal_low;
  uint32_t completion_signal_high;
};

typedef void (*pocl_workgroup_func32_argb) (
    void __global__ * /* args */, void __global__ * /* pocl_context */,
    uint /* group_x */, uint /* group_y */, uint /* group_z */);

/**
 * Prepares a work group for execution and launches it.
 */
<<<<<<< f7ea90b56ad8298de9cedb54c7fcbcc999cca591
static void
tta_opencl_wg_execute (kernel_exec_cmd *cmd)
{

  kernel_metadata *kernel = (kernel_metadata *)cmd->kernel_meta;

#ifdef DEBUG_TTA_DEVICE
  lwpr_print_str ("tta: ------------------- CMD: ");
  lwpr_print_int ((unsigned)cmd);
  lwpr_print_str ("\ntta: ------------------- KERNEL META: ");
  lwpr_print_int ((unsigned)kernel);
  lwpr_print_str ("\ntta: ------------------- NUM ARGS: ");
  lwpr_print_int ((unsigned)kernel->num_args);
  lwpr_print_str ("\ntta: ------------------- NUM LOCS: ");
  lwpr_print_int ((unsigned)kernel->num_locals);

  lwpr_print_str ("\ntta: ------------------- CTX ");
#endif
  struct pocl_context32 __global__ *context
      = (struct pocl_context32 __global__ *)cmd->ctx;

#ifdef DEBUG_TTA_DEVICE
  lwpr_print_int ((unsigned)context);
  lwpr_print_str (" SIZE: ");
  lwpr_print_int ((unsigned)cmd->ctx_size);
#endif
  unsigned num_groups_x = context->num_groups[0];
  unsigned num_groups_y
      = (context->work_dim >= 2) ? (context->num_groups[1]) : 1;
  unsigned num_groups_z
      = (context->work_dim == 3) ? (context->num_groups[2]) : 1;

#ifdef DEBUG_TTA_DEVICE
  lwpr_print_str ("\ntta: ------------------- GROUPS:  ");
  lwpr_print_int (num_groups_x);
  lwpr_print_str ("-");
  lwpr_print_int (num_groups_y);
  lwpr_print_str ("-");
  lwpr_print_int (num_groups_z);
  unsigned printf_buffer_capacity = context->printf_buffer_capacity;
  lwpr_print_str ("\ntta: ------------------- PBUF CAP:  ");
  lwpr_print_hex (printf_buffer_capacity);
#endif

  uint32_t __global__ *args = (uint32_t __global__ *)cmd->args;

#ifdef DEBUG_TTA_DEVICE
  lwpr_print_str ("\ntta: ------------------- ARGS ");
  lwpr_print_int ((unsigned)args);
  lwpr_print_str (" SIZE: ");
  lwpr_print_int ((unsigned)cmd->args_size);
  unsigned i;
  unsigned lim = (unsigned)cmd->args_size >> 2;
  for (i = 0; i < lim; ++i)
    {
      lwpr_print_str ("\n ARG: ");
      lwpr_print_int ((unsigned)args[i]);
      lwpr_print_str (" / ");
      lwpr_print_hex ((unsigned)args[i]);
    }
  lwpr_print_str ("\n");
#endif

  for (unsigned gid_x = 0; gid_x < num_groups_x; gid_x++)
    {
      for (unsigned gid_y = 0; gid_y < num_groups_y; gid_y++)
        {
          for (unsigned gid_z = 0; gid_z < num_groups_z; gid_z++)
            {
=======
static void
tta_opencl_wg_launch (__cq__ volatile struct AQLDispatchPacket *packet)
{

#ifdef DEBUG_TTA_DEVICE
  lwpr_print_str ("tta: ------------------- starting kernel\n");
#endif

  const int work_dim = packet->dimensions;
  struct pocl_context32 __buffer__ *context
      = (struct pocl_context32 __buffer__ *)(packet->reserved1);

  const int num_groups_x = context->num_groups[0];
  const int num_groups_y = (work_dim >= 2) ? (context->num_groups[1]) : 1;
  const int num_groups_z = (work_dim == 3) ? (context->num_groups[2]) : 1;

  //    int num_args = 3;
  //    int num_locals = 0;
  /*
      void* args[MAX_KERNEL_ARGS];
      for (int i=0; i < num_args + num_locals; i++) {
          args[i] = (void*)((__buffer__ void**)packet->kernarg_address_low)[i];
      }
  */

  for (unsigned gid_x = 0; gid_x <= num_groups_x; gid_x++)
    {
      for (unsigned gid_y = 0; gid_y < num_groups_y; gid_y++)
        {
          for (unsigned gid_z = 0; gid_z < num_groups_z; gid_z++)
            {
>>>>>>> add online compilation to accel
#ifdef DEBUG_TTA_DEVICE lwpr_print_str(
                  "tta: ------------------- launching WG ");
              lwpr_print_int (gid_x);
              lwpr_print_str ("-");
              lwpr_print_int (gid_y);
              lwpr_print_str ("-");
              lwpr_print_int (gid_z);
              lwpr_print_str (" @ ");
              lwpr_print_int ((unsigned)packet->kernel_object_low);
              lwpr_newline ();
#endif
<<<<<<< f7ea90b56ad8298de9cedb54c7fcbcc999cca591
                ((pocl_workgroup_func32_argb)kernel->work_group_func) (
                    args, context, gid_x, gid_y, gid_z);
            }
        }
    }
}

/**
 * Prepares a work group for execution and launches it.
 */
static void tta_opencl_wg_launch(kernel_exec_cmd* cmd) {

    cmd->status = POCL_KST_RUNNING;

    /* single thread version: execute all work groups in
       a single trampoline call as fast as possible. 

       Do not create any threads. */
#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("tta: ------------------- starting kernel\n");
#endif
        tta_opencl_wg_execute (cmd);
#ifdef DEBUG_TTA_DEVICE
=======
        ((pocl_workgroup_func32_argb)packet->kernel_object_low) (
            (__buffer__ void *)(packet->kernarg_address_low), context, gid_x,
            gid_y, gid_z);
}
}
}

#ifdef DEBUG_TTA_DEVICE
>>>>>>> add online compilation to accel
lwpr_print_str ("\ntta: ------------------- kernel finished\n");
#endif

        *(__buffer__ uint32_t *)packet->completion_signal_low = 1;
        packet->header = AQL_PACKET_INVALID;
}

#if _STANDALONE_MODE == 1
void initialize_kernel_launch();
#endif

int main() {

#ifdef DEBUG_TTA_DEVICE
    lwpr_print_str("tta: Hello from a TTA device\n");
    lwpr_print_str("tta: initializing the command objects\n");
#endif

#if _STANDALONE_MODE == 1
    initialize_kernel_launch();
#endif

    int read_iter = 0;
    _TCE_GET_READ_IDX_LOW (0, read_iter);
    do {
        uint32_t packet_loc
            = AQL_OFFSET
              + ((read_iter & (QUEUE_LENGTH - 1)) * AQL_PACKET_LENGTH);
        __cq__ volatile struct AQLDispatchPacket *packet
            = (__cq__ volatile struct AQLDispatchPacket *)packet_loc;

#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("tta: waiting for commands\n");
#endif

        while (packet->header == AQL_PACKET_INVALID)
          ;

<<<<<<< f7ea90b56ad8298de9cedb54c7fcbcc999cca591
#ifdef DEBUG_TTA_DEVICE lwpr_print_str(
            "tta: got a command to execute: ");
        lwpr_newline();
#endif
        tta_opencl_wg_launch(next_command);
=======
              // next_kernel = (kernel_metadata*)(next_command->kernel);

              tta_opencl_wg_launch (packet);

              // read_iter++; // move on to the next AQL packet
              _TCE_INC_READ_IDX (1);
              _TCE_GET_READ_IDX_LOW (0, read_iter);
              //	break; 
>>>>>>> add online compilation to accel

        /* In case this is the host-device setup (not the standalone mode),
           wait forever for commands from the host. Otherwise, execute the
           only command from the standalone binary and quit. */
    } while (1 && !_STANDALONE_MODE);

    return 0;
}
