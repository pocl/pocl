/* poclAccel.c - generic/example driver for hardware accelerators with memory
   mapped control.
*/

#include "poclAccel.h"

#define SLEEP_LOOP 50000

volatile int dummy = 0;


void poclAccel(volatile uint32_t Control[MEM_MAX_SIZE_WORD], volatile uint32_t* output, volatile uint64_t cycle_counter) {
#pragma HLS INTERFACE bram port=Control storage_type=RAM_1P
#pragma HLS INTERFACE mode=m_axi port=output offset=off

#pragma HLS INTERFACE mode=ap_none port=cycle_counter
#pragma HLS INTERFACE ap_ctrl_none port=return



  // Set initial values for info registers:
  Control[ACCEL_INFO_DEV_CLASS / 4] = 0xE; // Unused
  Control[ACCEL_INFO_DEV_ID / 4] = 0;      // Unused
  Control[ACCEL_INFO_IF_TYPE / 4] = 3;
  Control[ACCEL_INFO_CORE_COUNT / 4] = 1;
  Control[ACCEL_INFO_CTRL_SIZE / 4] = 1024;

  // The accelerator can choose the size of the queue (must be a power-of-two)
  // Can be even 1, to make the packet handling easiest with static offsets
  // The maximum size for this emulation to work is
  // segment_size/AQL_PACKET_LENGTH
  const uint32_t queue_length = 4;
  // const uint32_t queue_length = segment_size / AQL_PACKET_LENGTH;

  // Here we set the actual hardware memory region size. Even though the
  // address spaces are equally sized, the actual memory region sizes
  // don't have to be that big.The driver will adjust to these values.

  Control[ACCEL_INFO_CQMEM_SIZE_LOW / 4] = AQL_PACKET_LENGTH * (queue_length + 1);
  Control[ACCEL_INFO_CQMEM_SIZE_HIGH / 4] = 0;

  Control[ACCEL_INFO_IMEM_SIZE / 4] = 0;

  Control[ACCEL_INFO_DMEM_SIZE_LOW / 4] = MEM_MAX_SIZE_BYTES - 1024 - AQL_PACKET_LENGTH * (queue_length+1);
  Control[ACCEL_INFO_DMEM_SIZE_HIGH / 4] = 0;

  Control[ACCEL_INFO_IMEM_START_LOW / 4] = 0;
  Control[ACCEL_INFO_IMEM_START_HIGH / 4] = 0;

  Control[ACCEL_INFO_CQMEM_START_LOW / 4] = BASE_ADDRESS +  1024;
  Control[ACCEL_INFO_CQMEM_START_HIGH / 4] = 0;

  Control[ACCEL_INFO_DMEM_START_LOW / 4] =  BASE_ADDRESS + 1024 + AQL_PACKET_LENGTH * (queue_length+1);
  Control[ACCEL_INFO_DMEM_START_HIGH / 4] = 0;

  Control[ACCEL_INFO_FEATURE_FLAGS_LOW / 4] = 1;
  Control[ACCEL_INFO_PTR_SIZE / 4] = 4;
  Control[ACCEL_CONTROL_REG_COMMAND / 4] = ACCEL_RESET_CMD;

  const uint32_t CQInfoOffset = 1024/4;
  const uint32_t CQOffset = 1024/4 + AQL_PACKET_LENGTH/4;

  while ( 1 ) {
	// Don't start computing anything before hw reset is lifted.
	int reset = Control[ACCEL_CONTROL_REG_COMMAND / 4];
	if (reset != ACCEL_CONTINUE_CMD) {
	  continue;
	}
	int read_iter = Control[CQInfoOffset + 12];

	// Compute packet location
	uint32_t packet_loc = (read_iter & (queue_length - 1)) * (AQL_PACKET_LENGTH/4);
    uint32_t packet_offset = CQOffset + packet_loc;
	uint16_t packet_header              = (uint16_t)(Control[packet_offset + 0]);


	// The driver will mark the packet as not INVALID when it wants us to
	// compute it
	while (packet_header == AQL_PACKET_INVALID) {
      //Control[75+2]=1;
      packet_header = (uint16_t)Control[packet_offset + 0];
	}
	Control[75+11] = packet_header;
	//uint16_t packet_dimensions          = (uint16_t)(Control[packet_offset + 0] >> 16);
	uint32_t packet_grid_size_x         = Control[packet_offset + 3];
	//uint32_t packet_grid_size_y         = Control[packet_offset + 4];
	//uint32_t packet_grid_size_z         = Control[packet_offset + 5];
	uint32_t packet_kernel_object       = Control[packet_offset + 8];
	//packet_kernel_object                |= ((uint64_t)AQLstruct[9] << 32);
	uint32_t packet_kernarg_address     = Control[packet_offset + 10];
	//packet_kernarg_address              |= ((uint64_t)AQLstruct[11] << 32);
	uint32_t packet_completion_sig_addr = Control[packet_offset + 14];
	//packet_completion_sig_addr          |= ((uint64_t)AQLstruct[15] << 32);
    if (packet_header & (1 << AQL_PACKET_BARRIER_AND))
    {
      for (int i = 0; i < AQL_MAX_SIGNAL_COUNT; i++)
      {
        uint32_t signal = Control[packet_offset + 2 + 2*i];
        if (signal != 0)
        {
          while (output[signal/4] == 0) {
        	  for (int kk = 0;kk < SLEEP_LOOP; kk++) {
        		  dummy++;
        	  }
          }
        }
      }
    }
    else if (packet_header & (1 << AQL_PACKET_KERNEL_DISPATCH)) {
     	uint32_t index = packet_kernarg_address - BASE_ADDRESS;
    	uint32_t arg0 = Control[index/4];
    	uint32_t arg1 = Control[index/4 + 1];
    	uint32_t arg2 = Control[index/4 + 2];

    	Control[packet_completion_sig_addr/4 + 2] = cycle_counter;
    	Control[packet_completion_sig_addr/4 + 3] = cycle_counter >> 32;

        for (int i = 0; i < packet_grid_size_x; i++) {
          // Do the operation based on the kernel_object (integer id)
          switch (packet_kernel_object) {
          case (POCL_CDBI_COPY_I8):
            ((uint8_t*)output)[arg1 + i] = ((uint8_t*)output)[arg0 + i];
            break;
          case (POCL_CDBI_ADD_I32):
            output[arg2/4 + i] =  output[arg0/4 + i] + output[arg1/4 + i];
            break;
          case (POCL_CDBI_MUL_I32):
            output[arg2/4 + i] =  output[arg0/4 + i] * output[arg1/4 + i];
            break;
          }
        }

    	Control[packet_completion_sig_addr/4 + 4] = cycle_counter;
    	Control[packet_completion_sig_addr/4 + 5] = cycle_counter >> 32;
    } else {
    	Control[75+17] = 99;
     	uint32_t index = packet_kernarg_address - BASE_ADDRESS;
    	uint32_t arg_addr = Control[index/4];
    	continue;
    }
	//Completion signal is given as absolute address
    if (packet_completion_sig_addr){
	    packet_completion_sig_addr = packet_completion_sig_addr - BASE_ADDRESS;
	    Control[packet_completion_sig_addr/4] = 1;
    }
	Control[packet_offset] = AQL_PACKET_INVALID;

	read_iter++; // move on to the next AQL packet
	Control[CQInfoOffset + 12] = read_iter;
  }
}
