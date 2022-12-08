/* almaif-tce-device-defs.h - datatype definitions for Almaif TCE device

   Copyright (c) 2022 Topi Leppänen

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

#define AQL_PACKET_INVALID (1)
#define AQL_PACKET_KERNEL_DISPATCH (2)
#define AQL_PACKET_BARRIER_AND (3)
#define AQL_PACKET_AGENT_DISPATCH (4)
#define AQL_PACKET_BARRIER_OR (5)
#define AQL_PACKET_BARRIER (1 << 8)
#define AQL_PACKET_LENGTH (64)

#define AQL_MAX_SIGNAL_COUNT (5)

#define __global__ __attribute__ ((address_space (1)))
#define __constant__ __attribute__ ((address_space (2)))
#define __local__ __attribute__ ((address_space (3)))
#define __cq__ __attribute__ ((address_space (5)))

#ifndef QUEUE_START
#define QUEUE_START 0
#endif

#ifndef _STANDALONE_MODE
#define _STANDALONE_MODE 0
#endif

typedef void (*pocl_workgroup_func32_argb) (
    void __global__ * /* args */, void __global__ * /* pocl_context */,
    uint /* group_x */, uint /* group_y */, uint /* group_z */);

struct CommandMetadata
{
  uint32_t completion_signal;
  uint32_t reserved0;
  uint32_t start_timestamp_low;
  uint32_t start_timestamp_high;
  uint32_t finish_timestamp_low;
  uint32_t finish_timestamp_high;
  uint32_t reserved1;
  uint32_t reserved2;
};

struct AQLQueueInfo
{
  uint32_t type;
  uint32_t features;

  uint32_t base_address_low;
  uint32_t base_address_high;
  uint32_t doorbell_signal_low;
  uint32_t doorbell_signal_high;

  uint32_t size;
  uint32_t reserved0;

  uint32_t id_low;
  uint32_t id_high;

  volatile uint32_t write_index_low;
  volatile uint32_t write_index_high;

  uint32_t read_index_low;
  uint32_t read_index_high;

  uint32_t reserved1;
  uint32_t reserved2;
};

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

  uint32_t cmd_metadata_low;
  uint32_t cmd_metadata_high;
};

struct AQLAndPacket
{
  uint16_t header;
  uint16_t reserved0;
  uint32_t reserved1;

  uint32_t dep_signals[10];

  uint32_t signal_count_low;
  uint32_t signal_count_high;

  uint32_t completion_signal_low;
  uint32_t completion_signal_high;
};
