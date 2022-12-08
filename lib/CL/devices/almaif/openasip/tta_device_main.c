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

#include "almaif-tce-device-defs.h"

/**
 * Prepares a work group for execution and launches it.
 */
static void
tta_opencl_wg_launch (__cq__ volatile struct AQLDispatchPacket *packet)
{

#ifdef DEBUG_TTA_DEVICE
  lwpr_print_str ("tta: ------------------- starting kernel with packet\n");
  for (int k = 0; k < AQL_PACKET_LENGTH/4; k++){
      lwpr_print_int(((__cq__ uint32_t*)(packet))[k]);
      lwpr_newline ();
  }

#endif

  const int work_dim = packet->dimensions;
  struct pocl_context32 __global__ *context
      = (struct pocl_context32 __global__ *)(packet->reserved1);

  const int num_groups_x = context->num_groups[0];
  const int num_groups_y = (work_dim >= 2) ? (context->num_groups[1]) : 1;
  const int num_groups_z = (work_dim == 3) ? (context->num_groups[2]) : 1;

  for (unsigned gid_x = 0; gid_x < num_groups_x; gid_x++)
  {
    for (unsigned gid_y = 0; gid_y < num_groups_y; gid_y++)
    {
      for (unsigned gid_z = 0; gid_z < num_groups_z; gid_z++)
      {
#ifdef DEBUG_TTA_DEVICE
        lwpr_print_str("tta: ------------------- launching WG ");
        lwpr_print_int (gid_x);
        lwpr_print_str ("-");
        lwpr_print_int (gid_y);
        lwpr_print_str ("-");
        lwpr_print_int (gid_z);
        lwpr_print_str (" @ ");
        lwpr_print_int ((unsigned)packet->kernel_object_low);
        lwpr_newline ();
#endif
        ((pocl_workgroup_func32_argb)packet->kernel_object_low) (
            (__global__ void *)(packet->kernarg_address_low), context, gid_x,
            gid_y, gid_z);
      }
    }
  }

#ifdef DEBUG_TTA_DEVICE
  lwpr_print_str ("\ntta: ------------------- kernel finished\n");
#endif

  ((__global__ struct CommandMetadata *)packet->cmd_metadata_low)
      ->completion_signal
      = 1;
}

#if _STANDALONE_MODE == 1
void initialize_kernel_launch();
extern __cq__ struct AQLDispatchPacket standalone_packet;
#endif

int main() {

#ifdef DEBUG_TTA_DEVICE
  lwpr_print_str("tta: Hello from a TTA device\n");
  lwpr_print_str("tta: initializing the command objects\n");
#endif

  __cq__ struct AQLQueueInfo *queue_info =
      (__cq__ struct AQLQueueInfo *)QUEUE_START;
  queue_info->size = QUEUE_LENGTH;


#if _STANDALONE_MODE == 1
  initialize_kernel_launch();
#endif

  int read_iter = queue_info->read_index_low;
 #ifdef DEBUG_TTA_DEVICE
    lwpr_print_str("tta: starting with read iter:");
    lwpr_print_int(read_iter);
    lwpr_newline();
#endif

 
  do {
#if _STANDALONE_MODE == 1
    // Standalone mode will only process the single packet generated
    // by the external standalone.c-file
    uint32_t packet_loc = (uint32_t)&standalone_packet;
#else
    uint32_t packet_loc
      = QUEUE_START + AQL_PACKET_LENGTH
      + ((read_iter % QUEUE_LENGTH) * AQL_PACKET_LENGTH);
#endif
    __cq__ volatile struct AQLDispatchPacket *packet
      = (__cq__ volatile struct AQLDispatchPacket *)packet_loc;

#ifdef DEBUG_TTA_DEVICE
    lwpr_print_str("tta: waiting for commands\n");
#endif

    while (packet->header == AQL_PACKET_INVALID);

    uint16_t header = packet->header;
    if (header & (1 << AQL_PACKET_BARRIER_AND))
    {
      __cq__ volatile struct AQLAndPacket *andPacket = (__cq__ volatile struct AQLAndPacket *)packet_loc;

      for (int i = 0; i < AQL_MAX_SIGNAL_COUNT; i++)
      {
        volatile __global__ uint32_t *signal
            = (volatile __global__ uint32_t *)(andPacket->dep_signals[2 * i]);
        if (signal != NULL)
        {
          while (*signal == 0)
            ;
        }
      }
      packet->header = AQL_PACKET_INVALID;
      read_iter++; // move on to the next AQL packet
      queue_info->read_index_low = read_iter;
    }
    else if (header & (1 << AQL_PACKET_KERNEL_DISPATCH))
    {
      tta_opencl_wg_launch (packet);
      packet->header = AQL_PACKET_INVALID;
      read_iter++; // move on to the next AQL packet
      queue_info->read_index_low = read_iter;
    }


    //_TCE_INC_READ_IDX (1);
    //_TCE_GET_READ_IDX_LOW (0, read_iter);

    /* In case this is the host-device setup (not the standalone mode),
       wait forever for commands from the host. Otherwise, execute the
       only command from the standalone binary and quit. */
  } while (1 && !_STANDALONE_MODE);

  return 0;
}
