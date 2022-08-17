/* EmulationDevice.cc - accessing accelerator memory as memory mapped region.

   Copyright (c) 2021 Pekka Jääskeläinen / Tampere University

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

#include "EmulationDevice.h"

#include "EmulationRegion.h"
#include "accel-shared.h"
#include "pocl_timing.h"

#include <iostream>
#include <unistd.h>

EmulationDevice::EmulationDevice() {

  // The principle of the emulator is that instead of mmapping a real
  // accelerator, we just allocate a regular array, which corresponds
  // to the mmap. The accel_emulate function is working in another thread
  // and will fill that array and respond to the driver asynchronously.
  // The driver doesn't really need to know about emulating except
  // in the initial mapping of the accelerator
  E.emulating_address = calloc(1, EMULATING_MAX_SIZE);
  assert(E.emulating_address != NULL && "Emulating calloc failed\n");

  // Create emulator thread
  E.emulate_exit_called = 0;
  E.emulate_init_done = 0;
  pthread_create(&emulate_thread, NULL, emulate_accel, &E);
  while (!E.emulate_init_done)
    ; // Wait for the thread to initialize
  POCL_MSG_PRINT_ACCEL("accel: started emulating\n");

  ControlMemory =
      new EmulationRegion((size_t)E.emulating_address, ACCEL_DEFAULT_CTRL_SIZE);

  discoverDeviceParameters();

  InstructionMemory = new EmulationRegion(imem_start, imem_size);
  CQMemory = new EmulationRegion(cq_start, cq_size);
  DataMemory = new EmulationRegion(dmem_start, dmem_size);
}

EmulationDevice::~EmulationDevice() {
  POCL_MSG_PRINT_ACCEL("accel: freeing emulated accel");
  E.emulate_exit_called = 1; // signal for the emulator to stop
  pthread_join(emulate_thread, NULL);
  free((void *)E.emulating_address); // from
                                     // calloc(emulating_address)
}

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
  uint32_t dmem_size = EMULATING_MAX_SIZE * 3 / 4;
  // The accelerator can choose the size of the queue (must be a power-of-two)
  // Can be even 1, to make the packet handling easiest with static offsets
  uint32_t queue_length = 3;
  uint32_t cqmem_size = (queue_length + 1) * AQL_PACKET_LENGTH;

  // The accelerator can set the starting addresses
  // Even the order can be changed if the accelerator wants to
  // Here packing the memory regions tighly as an example.
  uintptr_t imem_start = (uintptr_t)base_address + ctrl_size;
  uintptr_t cqmem_start = imem_start + imem_size;
  uintptr_t dmem_start = cqmem_start + cqmem_size;

  volatile uint32_t *Control = (uint32_t *)base_address;
  volatile uint8_t *Instruction = (uint8_t *)imem_start;
  volatile uint32_t *CQ = (uint32_t *)cqmem_start;
  volatile uint8_t *Data = (uint8_t *)dmem_start;

  // Set initial values for info registers:
  Control[ACCEL_INFO_DEV_CLASS / 4] = 0xE; // Unused
  Control[ACCEL_INFO_DEV_ID / 4] = 0;      // Unused
  Control[ACCEL_INFO_IF_TYPE / 4] = 3;
  Control[ACCEL_INFO_CORE_COUNT / 4] = 1;
  Control[ACCEL_INFO_CTRL_SIZE / 4] = 1024;

#define PTR_SIZE sizeof(uint32_t *)
  Control[ACCEL_INFO_PTR_SIZE / 4] = PTR_SIZE;

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
  POCL_MSG_PRINT_ACCEL("accel emulate: Emulator initialized");

  int read_iter = 0;
  CQ[ACCEL_CQ_READ / 4] = read_iter;
  CQ[ACCEL_CQ_LENGTH / 4] = queue_length;

  // Accelerator is in infinite loop to process the commands
  // For emulating purposes we include the exit signal that the driver can
  // use to terminate the emulating thread. In hw this could be
  // a while(1) loop.
  while (!E->emulate_exit_called) {

    // Don't start computing anything before soft reset is lifted.
    // (This could probably be outside of the loop)
    int reset = Control[ACCEL_CONTROL_REG_COMMAND / 4];
    if (reset != ACCEL_CONTINUE_CMD) {
      usleep(ACCEL_DRIVER_SLEEP);
      continue;
    }

    // Compute packet location
    uint32_t packet_loc = ((read_iter % queue_length) + 1) * AQL_PACKET_LENGTH;
    struct AQLDispatchPacket *packet =
        (struct AQLDispatchPacket *)(CQ + packet_loc / 4);

    // The driver will mark the packet as not INVALID when it wants us to
    // compute it
    if (packet->header == AQL_PACKET_INVALID) {
      usleep(ACCEL_DRIVER_SLEEP);
      continue;
    }

    CommandMetadata *cmd = (CommandMetadata *)packet->command_meta_address;
    assert(cmd);
    cmd->start_timestamp = pocl_gettimemono_ns();

    uint16_t header = packet->header;
    if (header & (1 << AQL_PACKET_BARRIER_AND)) {
      struct AQLAndPacket *andPacket = (struct AQLAndPacket *)packet;
      POCL_MSG_PRINT_ACCEL(
          "accel emulate: Found valid AND packet from location "
          "%u, starting parsing:",
          packet_loc);
      for (int i = 0; i < AQL_MAX_SIGNAL_COUNT; i++) {
        uint32_t *signal = (uint32_t *)(andPacket->dep_signals[i]);
        if (signal != NULL) {
          while (*signal == 0) {
            usleep(ACCEL_DRIVER_SLEEP);
          }
        }
      }
      POCL_MSG_PRINT_ACCEL("accel emulate: And packet done\n");
    } else if (header & (1 << AQL_PACKET_KERNEL_DISPATCH)) {
      POCL_MSG_PRINT_ACCEL(
          "accel emulate: Found valid kernel dispatch packet from location "
          "%u, starting parsing:\n",
          packet_loc);
      POCL_MSG_PRINT_ACCEL("accel emulate: kernargs are at 0x%zx\n",
                           packet->kernarg_address);
      // Find the 3 pointers
      // Pointer size can be different on different systems
      // Also the endianness might need some attention in the real case.

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

      POCL_MSG_PRINT_ACCEL(
          "accel emulate: FOUND args arg0=%p, arg1=%p, arg2=%p\n", arg0, arg1,
          arg2);

      // Check how many dimensions are in use, and set the unused ones to 1.
      int dim_x = packet->grid_size_x;
      int dim_y = (packet->dimensions >= 2) ? (packet->grid_size_y) : 1;
      int dim_z = (packet->dimensions == 3) ? (packet->grid_size_z) : 1;

      int red_count = 0;
      POCL_MSG_PRINT_ACCEL(
          "accel emulate: Parsing done: starting loops with dims (%i,%i,%i)\n",
          dim_x, dim_y, dim_z);
      for (int x = 0; x < dim_x; x++) {
        for (int y = 0; y < dim_y; y++) {
          for (int z = 0; z < dim_z; z++) {
            // Linearize grid
            int idx = x * dim_y * dim_z + y * dim_z + z;
            // Do the operation based on the kernel_object (integer id)
            switch (packet->kernel_object) {
            case (POCL_CDBI_COPY_I8):
              arg1[idx] = arg0[idx];
              break;
            case (POCL_CDBI_ADD_I32):
              arg2[idx] = arg0[idx] + arg1[idx];
              break;
            case (POCL_CDBI_MUL_I32):
              arg2[idx] = arg0[idx] * arg1[idx];
              break;
            case (POCL_CDBI_COUNTRED): {
              uint32_t pixel = arg0[idx];
              uint8_t pixel_r = pixel & 0xFF;
              if (pixel_r > 100) {
                red_count++;
              }
            } break;
            case (POCL_CDBI_ABS_F32): {
              float *arg0f = (float *)arg0;
              float *arg1f = (float *)arg1;
              arg1f[idx] = std::abs(arg0f[idx]);
            } break;
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

      POCL_MSG_PRINT_ACCEL("accel emulate: Kernel done\n");
    }

    cmd->finish_timestamp = pocl_gettimemono_ns();
    cmd->completion_signal = 1;
    packet->header = AQL_PACKET_INVALID;

    read_iter++; // move on to the next AQL packet
    CQ[ACCEL_CQ_READ / 4] = read_iter;
  }

  return NULL;
}
