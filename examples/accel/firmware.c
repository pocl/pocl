/* firmware.c - Example firmware for tta device implementing AlmaIF.

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


#include <stdint.h>

#ifndef QUEUE_LENGTH
#define QUEUE_LENGTH 3
#endif

#define AQL_PACKET_INVALID (1)
#define AQL_PACKET_KERNEL_DISPATCH (2)
#define AQL_PACKET_BARRIER_AND (3)
#define AQL_PACKET_AGENT_DISPATCH (4)
#define AQL_PACKET_BARRIER_OR (5)
#define AQL_PACKET_BARRIER (1 << 8)
#define AQL_PACKET_LENGTH (64)

#define AQL_MAX_SIGNAL_COUNT (5)

#define ALMAIF_STATUS_REG (0x00)
#define ALMAIF_STATUS_REG_PC (0x04)
#define ALMAIF_STATUS_REG_CC_LOW (0x08)
#define ALMAIF_STATUS_REG_CC_HIGH (0x0C)
#define ALMAIF_STATUS_REG_SC_LOW (0x10)
#define ALMAIF_STATUS_REG_SC_HIGH (0x14)

#define SLEEP_CYCLES 400

#ifndef QUEUE_START
#define QUEUE_START 0
#endif

#define __cq__ __attribute__ ((address_space (5)))
#define __buffer__ __attribute__ ((address_space (1)))

// NOTE: This enum contains highly experimental built-in kernel IDs, that are
// subject to change in future PoCL releases without any deprecation period.
enum BuiltinKernelId : uint16_t
{
  // CD = custom device, BI = built-in
  // 1D array byte copy, get_global_size(0) defines the size of data to copy
  // kernel prototype: pocl.copy(char *input, char *output)
  POCL_CDBI_COPY_I8 = 0,
  POCL_CDBI_ADD_I32 = 1,
  POCL_CDBI_MUL_I32 = 2,
  POCL_CDBI_LEDBLINK = 3,
  POCL_CDBI_COUNTRED = 4,
  POCL_CDBI_DNN_CONV2D_RELU_I8 = 5,
  POCL_CDBI_SGEMM_LOCAL_F32 = 6,
  POCL_CDBI_SGEMM_TENSOR_F16F16F32_SCALE = 7,
  POCL_CDBI_SGEMM_TENSOR_F16F16F32 = 8,
  POCL_CDBI_ABS_F32 = 9,
  POCL_CDBI_DNN_DENSE_RELU_I8 = 10,
  POCL_CDBI_MAXPOOL_I8 = 11,
  POCL_CDBI_ADD_I8 = 12,
  POCL_CDBI_MUL_I8 = 13,
  POCL_CDBI_ADD_I16 = 14,
  POCL_CDBI_MUL_I16 = 15,
  POCL_CDBI_STREAMOUT_I32 = 16,
  POCL_CDBI_STREAMIN_I32 = 17,
  POCL_CDBI_VOTE_U32 = 18,
  POCL_CDBI_VOTE_U8 = 19,
  POCL_CDBI_DNN_CONV2D_NCHW_F32 = 20,
  POCL_CDBI_OPENVX_SCALEIMAGE_NN_U8 = 21,
  POCL_CDBI_OPENVX_SCALEIMAGE_BL_U8 = 22,
  POCL_CDBI_OPENVX_TENSORCONVERTDEPTH_WRAP_U8_F32 = 23,
  POCL_CDBI_OPENVX_MINMAXLOC_R1_U8 = 24,
  POCL_CDBI_LAST = 25,
  POCL_CDBI_JIT_COMPILER = 0xFFFF
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

struct CommandMetadata
{
  uint32_t completion_signal;
  uint32_t reserved0;
  uint32_t start_timestamp_l;
  uint32_t start_timestamp_h;
  uint32_t finish_timestamp_l;
  uint32_t finish_timestamp_h;
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

int
main ()
{
  __cq__ struct AQLQueueInfo *queue_info
      = (__cq__ struct AQLQueueInfo *)QUEUE_START;
  int read_iter = queue_info->read_index_low;

  while (1)
    {
      // Compute packet location
      uint32_t packet_loc = QUEUE_START + AQL_PACKET_LENGTH
                            + ((read_iter % QUEUE_LENGTH) * AQL_PACKET_LENGTH);
      __cq__ volatile struct AQLDispatchPacket *packet
          = (__cq__ volatile struct AQLDispatchPacket *)packet_loc;
      // The driver will mark the packet as not INVALID when it wants us to
      // compute it
      while (packet->header == AQL_PACKET_INVALID)
        {
/*        *((__cq__ int*)0 ) = packet_loc;
        *((__cq__ int*)4 ) = read_iter + 5;
        *((__cq__ uint16_t*)8 ) = packet->header;
  */  };
      uint16_t header = packet->header;
      if (header & (1 << AQL_PACKET_BARRIER_AND))
        {
          __cq__ volatile struct AQLAndPacket *andPacket
              = (__cq__ volatile struct AQLAndPacket *)packet_loc;

          for (int i = 0; i < AQL_MAX_SIGNAL_COUNT; i++)
            {
              volatile __buffer__ uint32_t *signal
                  = (volatile __buffer__ uint32_t *)(andPacket
                                                         ->dep_signals[2 * i]);
              if (signal != 0)
                {
                  while (*signal == 0)
                    {
                      for (int kk = 0; kk < SLEEP_CYCLES; kk++)
                        {
                          asm volatile("...;");
                        }
                    }
                }
            }
        }
      else if (header & (1 << AQL_PACKET_KERNEL_DISPATCH))
        {
#ifdef BASE_ADDRESS
          __buffer__ uint32_t *control_region
              = (__buffer__ uint32_t *)BASE_ADDRESS;
          uint32_t cc_l = control_region[ALMAIF_STATUS_REG_CC_LOW / 4];
          // uint32_t cc_h = control_region[ALMAIF_STATUS_REG_CC_HIGH/4];
          uint32_t sc_l = control_region[ALMAIF_STATUS_REG_SC_LOW / 4];
          // uint32_t sc_h = control_region[ALMAIF_STATUS_REG_SC_HIGH/4];
          __buffer__ struct CommandMetadata *cmd_meta
              = (__buffer__ struct CommandMetadata*)packet->completion_signal_low;
          cmd_meta->start_timestamp_l = cc_l + sc_l;
          // cmd_meta->start_timestamp_h = cc_h + sc_h;
#endif

          uint32_t kernel_id = packet->kernel_object_low;

          __buffer__ uint32_t *kernarg_ptr
              = (__buffer__ uint32_t *)(packet->kernarg_address_low);

          __buffer__ uint32_t *arg0 = (__buffer__ uint32_t *)kernarg_ptr[0];
          __buffer__ uint32_t *arg1 = (__buffer__ uint32_t *)kernarg_ptr[1];
          __buffer__ uint32_t *arg2 = (__buffer__ uint32_t *)kernarg_ptr[2];
          __buffer__ uint32_t *arg3 = (__buffer__ uint32_t *)kernarg_ptr[3];
          __buffer__ uint32_t *arg4 = (__buffer__ uint32_t *)kernarg_ptr[4];

          // Check how many dimensions are in use, and set the unused ones
          // to 1.
          int dim_x = packet->grid_size_x;
          int dim_y = (packet->dimensions >= 2) ? (packet->grid_size_y) : 1;
          int dim_z = (packet->dimensions == 3) ? (packet->grid_size_z) : 1;

          uint8_t min = 255;
          uint8_t max = 0;
          uint32_t minlocx, minlocy, maxlocx, maxlocy;
          for (int z = 0; z < dim_z; z++)
            {
              for (int y = 0; y < dim_y; y++)
                {
                  for (int x = 0; x < dim_x; x++)
                    {
                      // Linearize grid
                      int idx = z * dim_y * dim_x + dim_x * y + x;
                      // Do the operation based on the kernel_object (integer
                      // id)
                      switch (kernel_id)
                        {
                        case (POCL_CDBI_COPY_I8):
                          arg1[idx] = arg0[idx];
                          break;
                        case (POCL_CDBI_ADD_I32):
                          arg2[idx] = arg0[idx] + arg1[idx];
                          break;
                        case (POCL_CDBI_MUL_I32):
                          arg2[idx] = arg0[idx] * arg1[idx];
                          break;
                        case (POCL_CDBI_OPENVX_MINMAXLOC_R1_U8):
                          {
                            uint8_t pixel = ((__buffer__ uint8_t *)arg0)[idx];
                            if (pixel < min)
                              {
                                min = pixel;
                                minlocx = x;
                                minlocy = y;
                              }
                            if (pixel > max)
                              {
                                max = pixel;
                                maxlocx = x;
                                maxlocy = y;
                              }
                          }
                          break;
                        }
                    }
                }
            }

          if (kernel_id == POCL_CDBI_OPENVX_MINMAXLOC_R1_U8)
            {
              arg1[0] = min;
              arg2[0] = max;
              arg3[0] = minlocx;
              arg3[1] = minlocy;
              arg4[0] = maxlocx;
              arg4[1] = maxlocy;
            }
#ifdef BASE_ADDRESS
          cc_l = control_region[ALMAIF_STATUS_REG_CC_LOW / 4];
          // cc_h = control_region[ALMAIF_STATUS_REG_CC_HIGH/4];
          sc_l = control_region[ALMAIF_STATUS_REG_SC_LOW / 4];
          // sc_h = control_region[ALMAIF_STATUS_REG_SC_HIGH/4];
          cmd_meta->finish_timestamp_l = cc_l + sc_l;
          // cmd_meta->finish_timestamp_h = cc_h + sc_h;
#endif
        }
      // Completion signal is given as absolute address
      if (packet->completion_signal_low)
        {
          *(__buffer__ uint32_t *)packet->completion_signal_low = 1;
        }
      packet->header = AQL_PACKET_INVALID;

      read_iter++; // move on to the next AQL packet
      queue_info->read_index_low = read_iter;
    }
}
