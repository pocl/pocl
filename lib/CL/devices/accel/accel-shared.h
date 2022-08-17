
#ifndef POCL_ACCELSHARED_H
#define POCL_ACCELSHARED_H

#include <set>
#include <vector>

//#include "pocl_util.h"

#include "EmulationDevice.h"
#include "MMAPDevice.h"
#include "bufalloc.h"
#include "builtin_kernels.hh"

#define ACCEL_DEFAULT_PRIVATE_MEM_SIZE (2048)

#define ACCEL_DEFAULT_CTRL_SIZE (1024)

#define ALMAIF_VERSION_2 (2)
#define ALMAIF_VERSION_3 (3)

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
#define ACCEL_FF_BIT_W_IMEM_START (1 << 1)
#define ACCEL_FF_BIT_W_DMEM_START (1 << 2)
#define ACCEL_FF_BIT_PAUSE (1 << 3)

#define ACCEL_INFO_DMEM_SIZE_LEGACY (0x314)
#define ACCEL_INFO_IMEM_SIZE_LEGACY (0x318)
#define ACCEL_INFO_PMEM_SIZE_LEGACY (0x31C)

#define AQL_PACKET_INVALID (1)
#define AQL_PACKET_KERNEL_DISPATCH (2)
#define AQL_PACKET_BARRIER_AND (3)
#define AQL_PACKET_AGENT_DISPATCH (4)
#define AQL_PACKET_BARRIER_OR (5)

#define AQL_PACKET_BARRIER (1 << 8)

#define AQL_PACKET_LENGTH (64)
#define AQL_MAX_SIGNAL_COUNT (5)

#define DEFAULT_BUILD_HASH "ALMAIF_unset_hash"

struct AQLQueueInfo
{
  uint32_t type;
  uint32_t features;

  uint64_t base_address;
  uint64_t doorbell_signal;

  uint32_t size;
  uint32_t reserved0;

  uint64_t id;

  uint64_t write_index;
  uint64_t read_index;

  uint64_t reserved1;
};

#define ACCEL_CQ_READ (offsetof (AQLQueueInfo, read_index))
#define ACCEL_CQ_WRITE (offsetof (AQLQueueInfo, write_index))
#define ACCEL_CQ_LENGTH (offsetof (AQLQueueInfo, size))

#define ACCEL_DRIVER_SLEEP 200

struct CommandMetadata
{
  uint32_t completion_signal;
  uint32_t reserved0;
  uint64_t start_timestamp;
  uint64_t finish_timestamp;
  uint64_t reserved1;
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
  uint64_t kernel_object;
  uint64_t kernarg_address;

  uint64_t reserved;

  uint64_t command_meta_address;
};

struct AQLAndPacket
{
  uint16_t header;
  uint16_t reserved0;
  uint32_t reserved1;

  uint64_t dep_signals[5];

  uint64_t signal_count;

  uint64_t completion_signal;
};

// declaration to resolve circular dependency
typedef struct compilation_data_s compilation_data_t;

struct AccelData
{
  size_t BaseAddress;

  Device *Dev;

  std::set<BIKD *> SupportedKernels;
  // List of commands ready to be executed.
  _cl_command_node *ReadyList;
  // List of commands not yet ready to be executed.
  _cl_command_node *CommandList;
  // Lock for command list related operations.
  pocl_lock_t CommandListLock;

  // Lock for device-side command queue manipulation
  pocl_lock_t AQLQueueLock;

  emulation_data_t EmulationData;

  // Backend-agnostic compilation data
  compilation_data_t *compilationData;
};

bool isEventDone (AccelData *data, cl_event event);
void submit_kernel_packet (AccelData *D, _cl_command_node *cmd);
void submit_and_barrier (AccelData *D, _cl_command_node *cmd);

/* The address space ids in the ADFs. */
#define TTA_ASID_PRIVATE 0
#define TTA_ASID_GLOBAL 1
#define TTA_ASID_LOCAL 3
#define TTA_ASID_CONSTANT 2

#endif
