/* poclAccel.h - Example HLS synthesizable accelerator implementing
                   the AlmaIF interface

   Copyright (c) 2022 Topi Leppänen / Tampere University

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

#ifndef POCLACCEL_H
#define POCLACCEL_H

#include <stdint.h>

// #define MEM_MAX_SIZE_WORD 8192
// #define MEM_MAX_SIZE_BYTES (4*MEM_MAX_SIZE_WORD)
#define MEM_MAX_SIZE_WORD (MEM_MAX_SIZE_BYTES / 4)

#define PTR_SIZE sizeof (uint32_t *)

//#define BASE_ADDRESS 0x40000000

//#define BASE_ADDRESS 0x0

#define POCL_CDBI_COPY_I8 0
#define POCL_CDBI_ADD_I32 1
#define POCL_CDBI_MUL_I32 2
#define POCL_CDBI_STREAMIN_I32 17

#define ACCEL_DEFAULT_CTRL_SIZE (1024)

#define ACCEL_STATUS_REG (0x00)
#define ACCEL_STATUS_REG_PC (0x04)
#define ACCEL_STATUS_REG_CC_LOW (0x08)
#define ACCEL_STATUS_REG_CC_HIGH (0x0C)
#define ACCEL_STATUS_REG_SC_LOW (0x10)
#define ACCEL_STATUS_REG_SC_HIGH (0x14)

#define ACCEL_CONTROL_REG_COMMAND (0x200)

#define ACCEL_RESET_CMD (1)
#define ACCEL_CONTINUE_CMD (2)
#define ACCEL_BREAK_CMD (4)

#define ACCEL_INFO_DEV_CLASS (0x300)
#define ACCEL_INFO_DEV_ID (0x304)
#define ACCEL_INFO_IF_TYPE (0x308)
#define ACCEL_INFO_CORE_COUNT (0x30C)
#define ACCEL_INFO_CTRL_SIZE (0x310)

#define ACCEL_INFO_IMEM_SIZE (0x314)
#define ACCEL_INFO_IMEM_START_LOW (0x318)
#define ACCEL_INFO_IMEM_START_HIGH (0x31C)

#define ACCEL_INFO_CQMEM_SIZE_LOW (0x320)
#define ACCEL_INFO_CQMEM_SIZE_HIGH (0x324)
#define ACCEL_INFO_CQMEM_START_LOW (0x328)
#define ACCEL_INFO_CQMEM_START_HIGH (0x32C)

#define ACCEL_INFO_DMEM_SIZE_LOW (0x330)
#define ACCEL_INFO_DMEM_SIZE_HIGH (0x334)
#define ACCEL_INFO_DMEM_START_LOW (0x338)
#define ACCEL_INFO_DMEM_START_HIGH (0x33C)

#define ACCEL_INFO_FEATURE_FLAGS_LOW (0x340)
#define ACCEL_INFO_FEATURE_FLAGS_HIGH (0x344)

#define ACCEL_INFO_PTR_SIZE (0x348)

#define ACCEL_FF_BIT_AXI_MASTER (1 << 0)

#define AQL_PACKET_INVALID (1)
#define AQL_PACKET_KERNEL_DISPATCH (2)
#define AQL_PACKET_BARRIER_AND (3)
#define AQL_PACKET_AGENT_DISPATCH (4)
#define AQL_PACKET_BARRIER_OR (5)
#define AQL_PACKET_BARRIER (1 << 8)
#define AQL_PACKET_LENGTH (64)
#define AQL_MAX_SIGNAL_COUNT (5)

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

  uint32_t completion_signal_low;
  uint32_t completion_signal_high;
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

void poclAccel (uint32_t base_address[MEM_MAX_SIZE_WORD]);

#endif /* POCLACCEL_H */
