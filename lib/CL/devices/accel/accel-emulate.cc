
#include <iostream>

#include "accel-emulate.h"
#include "accel-shared.h"
#include "accel.h"

/*
 * AlmaIF v1 based accel emulator
 * Base_address is a preallocated emulation array that corresponds to the
 * memory map of the accelerator
 * Does operations 0,1,2,3,4
 * */
void *emulate_accel(void *E_void) {

  emulation_data_t *E = (emulation_data_t *)E_void;
  void *base_address = E->emulating_address;

  uint32_t ctrl_size = 1024;
  uint32_t imem_size = 0;
  uint32_t dmem_size = 2097152;
  // The accelerator can choose the size of the queue (must be a power-of-two)
  // Can be even 1, to make the packet handling easiest with static offsets
  uint32_t queue_length = 2;
  uint32_t cqmem_size = queue_length * AQL_PACKET_LENGTH;

  // The accelerator can set the starting addresses
  // Even the order can be changed if the accelerator wants to
  // Here packing the memory regions tighly as an example.
  uintptr_t imem_start = (uintptr_t)base_address + ctrl_size;
  uintptr_t cqmem_start = imem_start + imem_size;
  uintptr_t dmem_start = cqmem_start + cqmem_size;

  volatile uint32_t *Control = (uint32_t *)base_address;
  volatile uint8_t *Instruction = (uint8_t *)imem_start;
  volatile uint8_t *CQ = (uint8_t *)cqmem_start;
  volatile uint8_t *Data = (uint8_t *)dmem_start;

  // Set initial values for info registers:
  Control[ACCEL_INFO_DEV_CLASS / 4] = 0xE; // Unused
  Control[ACCEL_INFO_DEV_ID / 4] = 0;      // Unused
  Control[ACCEL_INFO_IF_TYPE / 4] = 1;
  Control[ACCEL_INFO_CORE_COUNT / 4] = 1;
  Control[ACCEL_INFO_CTRL_SIZE / 4] = 1024;

  // The emulation doesn't use Instruction/Configuration memory. This memory
  // space is a place to write accelerator specific configuration values
  // that are written BEFORE hw reset is deasserted.
  // E.g. program binaries of a processor-based accelerator
  Control[ACCEL_INFO_IMEM_SIZE / 4] = 0;
  Control[ACCEL_INFO_IMEM_START_LOW / 4] = (uint32_t)imem_start;
  Control[ACCEL_INFO_IMEM_START_HIGH / 4] = (uint32_t)(imem_start >> 32);

  Control[ACCEL_INFO_CQMEM_SIZE_LOW / 4] = cqmem_size;
  Control[ACCEL_INFO_CQMEM_START_LOW / 4] = (uint32_t)cqmem_start;
  Control[ACCEL_INFO_CQMEM_START_HIGH / 4] = (uint32_t)(cqmem_start >> 32);

  Control[ACCEL_INFO_DMEM_SIZE_LOW / 4] = dmem_size;
  Control[ACCEL_INFO_DMEM_START_LOW / 4] = (uint32_t)dmem_start;
  Control[ACCEL_INFO_DMEM_START_HIGH / 4] = (uint32_t)(dmem_start >> 32);

  uint32_t feature_flags_low = ACCEL_FF_BIT_AXI_MASTER;
  Control[ACCEL_INFO_FEATURE_FLAGS_LOW / 4] = feature_flags_low;

  // Signal the driver that the initial values are set
  // (in hardware this signal is probably not needed, since the values are
  // initialized in hw reset)
  E->emulate_init_done = 1;
  POCL_MSG_PRINT_INFO("accel emulate: Emulator initialized");

  int read_iter = 0;
  Control[ACCEL_AQL_READ_LOW / 4] = read_iter;
  // Accelerator is in infinite loop to process the commands
  // For emulating purposes we include the exit signal that the driver can
  // use to terminate the emulating thread. In hw this could be
  // a while(1) loop.
  while (!E->emulate_exit_called) {

    // Don't start computing anything before soft reset is lifted.
    // (This could probably be outside of the loop)
    int reset = Control[ACCEL_CONTROL_REG_COMMAND / 4];
    if (reset != ACCEL_CONTINUE_CMD) {
      continue;
    }

    // Compute packet location
    uint32_t packet_loc = (read_iter & (queue_length - 1)) * AQL_PACKET_LENGTH;
    struct AQLDispatchPacket *packet =
        (struct AQLDispatchPacket *)(CQ + packet_loc);

    // The driver will mark the packet as not INVALID when it wants us to
    // compute it
    if (packet->header == AQL_PACKET_INVALID) {
      continue;
    }

    POCL_MSG_PRINT_INFO("accel emulate: Found valid AQL_packet from location "
                        "%u, starting parsing:",
                        packet_loc);
    POCL_MSG_PRINT_INFO("accel emulate: kernargs are at %lu\n",
                        packet->kernarg_address);
    // Find the 3 pointers
    // Pointer size can be different on different systems
    // Also the endianness might need some attention in the real case.
#define PTR_SIZE sizeof(uint32_t *)
    union args_u {
      uint32_t *ptrs[3];
      uint8_t values[3 * PTR_SIZE];
    } args;
    for (int i = 0; i < 3; i++) {
      for (int k = 0; k < PTR_SIZE; k++) {
        args.values[PTR_SIZE * i + k] =
            *(uint8_t *)(packet->kernarg_address + PTR_SIZE * i + k);
      }
    }
    uint32_t *arg0 = args.ptrs[0];
    uint32_t *arg1 = args.ptrs[1];
    uint32_t *arg2 = args.ptrs[2];

    POCL_MSG_PRINT_INFO("accel emulate: FOUND args arg0=%p, arg1=%p, arg2=%p\n",
                        arg0, arg1, arg2);

    // Check how many dimensions are in use, and set the unused ones to 1.
    int dim_x = packet->grid_size_x;
    int dim_y = (packet->dimensions >= 2) ? (packet->grid_size_y) : 1;
    int dim_z = (packet->dimensions == 3) ? (packet->grid_size_z) : 1;

    int red_count = 0;
    POCL_MSG_PRINT_INFO(
        "accel emulate: Parsing done: starting loops with dims (%i,%i,%i)",
        dim_x, dim_y, dim_z);
    for (int x = 0; x < dim_x; x++) {
      for (int y = 0; y < dim_y; y++) {
        for (int z = 0; z < dim_z; z++) {
          // Linearize grid
          int idx = x * dim_y * dim_z + y * dim_z + z;
          // Do the operation based on the kernel_object (integer id)
          switch (packet->kernel_object) {
          case (POCL_CDBI_COPY):
            arg1[idx] = arg0[idx];
            break;
          case (POCL_CDBI_ADD32):
            arg2[idx] = arg0[idx] + arg1[idx];
            break;
          case (POCL_CDBI_MUL32):
            arg2[idx] = arg0[idx] * arg1[idx];
            break;
          case (POCL_CDBI_COUNTRED):
            uint32_t pixel = arg0[idx];
            uint8_t pixel_r = pixel & 0xFF;
            if (pixel_r > 100) {
              red_count++;
            }
          }
        }
      }
    }
    if (packet->kernel_object == POCL_CDBI_LEDBLINK) {
      std::cout << "Emulation blinking " << dim_x << " led(s) at interval "
                << arg0[0] << " us " << arg1[0] << " times" << std::endl;
    }
    if (packet->kernel_object == POCL_CDBI_COUNTRED) {
      arg1[0] = red_count;
    }

    POCL_MSG_PRINT_INFO("accel emulate: Kernel done");

    // Completion signal is given as absolute address
    *(uint32_t *)packet->completion_signal = 1;
    packet->header = AQL_PACKET_INVALID;

    read_iter++; // move on to the next AQL packet
    Control[ACCEL_AQL_READ_LOW / 4] = read_iter;
  }
}
