
#ifndef POCL_ACCELSHARED_H
#define POCL_ACCELSHARED_H

#include <set>
#include <vector>

//#include "pocl_util.h"

#include "MMAPRegion.h"
#include "accel-emulate.h"
#include "bufalloc.h"
//#include "almaif-compile.h"
//#include "almaif-compile-tce.h"

#define ACCEL_DEFAULT_CTRL_SIZE (1024)

#define ACCEL_STATUS_REG (0x00)
#define ACCEL_STATUS_REG_PC (0x04)
#define ACCEL_STATUS_REG_CC_LOW (0x08)
#define ACCEL_STATUS_REG_CC_HIGH (0x0C)
#define ACCEL_STATUS_REG_SC_LOW (0x10)
#define ACCEL_STATUS_REG_SC_HIGH (0x14)

#define ACCEL_AQL_READ_LOW (0x100)
#define ACCEL_AQL_READ_HIGH (0x104)
#define ACCEL_AQL_WRITE_LOW (0x108)
#define ACCEL_AQL_WRITE_HIGH (0x10C)
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

#define ACCEL_FF_BIT_AXI_MASTER (1 << 0)
#define ACCEL_FF_BIT_W_IMEM_START (1 << 1)
#define ACCEL_FF_BIT_W_DMEM_START (1 << 2)
#define ACCEL_FF_BIT_PAUSE (1 << 3)

#define AQL_PACKET_INVALID (1)
#define AQL_PACKET_KERNEL_DISPATCH (2)
#define AQL_PACKET_BARRIER_AND (3)
#define AQL_PACKET_AGENT_DISPATCH (4)
#define AQL_PACKET_BARRIER_OR (5)

#define AQL_PACKET_BARRIER (1 << 8)

#define AQL_PACKET_LENGTH (64)

#define DEFAULT_BUILD_HASH "ALMAIF_unset_hash"

enum BuiltinKernelId : uint16_t
{
  // CD = custom device, BI = built-in
  // 1D array byte copy, get_global_size(0) defines the size of data to copy
  // kernel prototype: pocl.copy(char *input, char *output)
  POCL_CDBI_COPY = 0,
  POCL_CDBI_ADD32 = 1,
  POCL_CDBI_MUL32 = 2,
  POCL_CDBI_LEDBLINK = 3,
  POCL_CDBI_COUNTRED = 4,
  POCL_CDBI_JIT_COMPILER = 0xFFFF
};

// An initialization wrapper for kernel argument metadatas.
struct BIArg : public pocl_argument_info
{
  BIArg (const char *TypeName, const char *Name, pocl_argument_type Type,
         cl_kernel_arg_address_qualifier AQ)
  {
    type = Type;
    address_qualifier = AQ;
    type_name = strdup (TypeName);
    name = strdup (Name);
  }

  ~BIArg ()
  {
    free (name);
    free (type_name);
  }
};

// An initialization wrapper for kernel metadatas.
// BIKD = Built-in Kernel Descriptor
struct BIKD : public pocl_kernel_metadata_t
{
  BIKD (BuiltinKernelId KernelId, const char *KernelName,
        const std::vector<pocl_argument_info> &ArgInfos);

  ~BIKD ()
  {
    delete[] arg_info;
    free (name);
  }

  BuiltinKernelId KernelId;
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

  uint64_t completion_signal;
};

// declaration to resolve circular dependency
typedef struct compilation_data_s compilation_data_t;

struct AccelData
{
  size_t BaseAddress;

  MMAPRegion *ControlMemory;
  MMAPRegion *InstructionMemory;
  MMAPRegion *DataMemory;
  MMAPRegion *CQMemory;
  MMAPRegion *ScratchpadMemory;
  memory_region_t AllocRegion;

  std::set<BIKD *> SupportedKernels;
  // List of commands ready to be executed.
  _cl_command_node *ReadyList;
  // List of commands not yet ready to be executed.
  _cl_command_node *CommandList;
  // Lock for command list related operations.
  pocl_lock_t CommandListLock;

  // Lock for device-side command queue manipulation
  pocl_lock_t AQLQueueLock;

  int RelativeAddressing;
  emulation_data_t EmulationData;

  // Backend-agnostic compilation data
  compilation_data_t *compilationData;
};

#endif
