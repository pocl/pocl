/* messages.h - message types for PoCL-Remote communication

   Copyright (c) 2018 Michal Babej / Tampere University of Technology
   Copyright (c) 2019-2023 Jan Solanti / Tampere University

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

#include "pocl_remote.h"

#ifndef POCL_REMOTE_MESSAGES_H
#define POCL_REMOTE_MESSAGES_H

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

  /* ########################## */

#define DEFAULT_QUE_ID (1UL << 24)

#define ENUM_TYPE uint8_t

#define MAX_PACKED_STRING_LEN 1024

#define SESSION_ID_LENGTH 16

#define STRING_TYPE(x) char x[MAX_PACKED_STRING_LEN]

#define WRITEV_REQ(num, SIZE) writev_req (data, vecs, num, SIZE)

#define CHECK_REPLY(type)                                                     \
  if (!netcmd->reply.failed)                                                  \
    assert (netcmd->reply.message_type == MessageType_##type##Reply);         \
  assert (netcmd->reply.did == ddata->remote_device_index);                   \
  assert (netcmd->reply.pid == ddata->remote_platform_index);                 \
  if (netcmd->reply.failed)                                                   \
    {                                                                         \
      POCL_MSG_ERR ("Reply is FAIL: %i\n", netcmd->reply.fail_details);       \
      return netcmd->reply.fail_details;                                      \
    }

#define READ_DATA(ptr, size) read_data (data, ptr, size)

#include <stddef.h>
#include <stdint.h>

  typedef struct vec3_s
  {
    uint64_t x;
    uint64_t y;
    uint64_t z;
  } vec3_t;

  typedef enum DevType
  {
    CPU = 1,
    GPU,
    ACCELERATOR,
    CUSTOM
  } DevType;

#define MAX_IMAGE_FORMAT_TYPES 1024

  enum RequestMessageType
  {
    MessageType_ServerInfo,
    MessageType_DeviceInfo,
    MessageType_ConnectPeer,
    MessageType_PeerHandshake,

    MessageType_CreateBuffer,
    MessageType_FreeBuffer,

    MessageType_CreateCommandQueue,
    MessageType_FreeCommandQueue,

    MessageType_CreateSampler,
    MessageType_FreeSampler,

    MessageType_CreateImage,
    MessageType_FreeImage,

    MessageType_CreateKernel,
    MessageType_FreeKernel,

    MessageType_BuildProgramFromSource,
    MessageType_BuildProgramFromBinary,
    MessageType_BuildProgramWithBuiltins,
    MessageType_FreeProgram,

    // ***********************************************

    MessageType_MigrateD2D,

    MessageType_ReadBuffer,
    MessageType_WriteBuffer,
    MessageType_CopyBuffer,
    MessageType_FillBuffer,

    MessageType_ReadBufferRect,
    MessageType_WriteBufferRect,
    MessageType_CopyBufferRect,

    MessageType_CopyImage2Buffer,
    MessageType_CopyBuffer2Image,
    MessageType_CopyImage2Image,
    MessageType_ReadImageRect,
    MessageType_WriteImageRect,
    MessageType_FillImageRect,

    MessageType_RunKernel,

    MessageType_NotifyEvent,
    MessageType_RdmaBufferRegistration,

    // TODO finish
    MessageType_Finish,

    MessageType_Shutdown,
  };

  enum ReplyMessageType
  {
    MessageType_ServerInfoReply,
    MessageType_DeviceInfoReply,
    MessageType_ConnectPeerReply,
    MessageType_PeerHandshakeReply,

    MessageType_CreateBufferReply,
    MessageType_FreeBufferReply,

    MessageType_CreateCommandQueueReply,
    MessageType_FreeCommandQueueReply,

    MessageType_CreateSamplerReply,
    MessageType_FreeSamplerReply,

    MessageType_CreateImageReply,
    MessageType_FreeImageReply,

    MessageType_CreateKernelReply,
    MessageType_FreeKernelReply,

    MessageType_BuildProgramReply,
    MessageType_FreeProgramReply,

    // ***********************************************

    MessageType_MigrateD2DReply,

    MessageType_ReadBufferReply,
    MessageType_WriteBufferReply,
    MessageType_CopyBufferReply,
    MessageType_FillBufferReply,

    MessageType_CopyImage2BufferReply,
    MessageType_CopyBuffer2ImageReply,
    MessageType_CopyImage2ImageReply,
    MessageType_ReadImageRectReply,
    MessageType_WriteImageRectReply,
    MessageType_FillImageRectReply,

    MessageType_RunKernelReply,

    MessageType_Failure
  };

  typedef struct __attribute__ ((packed, aligned (8))) ImgFormatType_s
  {
    uint32_t channel_order;
    uint32_t channel_data_type;
  } ImgFormatType_t;

  typedef struct __attribute__ ((packed, aligned (8))) ImgFormatInfo_s
  {
    uint32_t memobj_type;
    uint32_t num_formats;
    ImgFormatType_t formats[MAX_IMAGE_FORMAT_TYPES];
  } ImgFormatInfo_t;

  typedef struct __attribute__ ((packed, aligned (8))) ServerInfoMsg_s
  {
    uint32_t peer_id;
  } ServerInfoMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) DeviceInfo_s
  {

    /* ######## device properties ############## */

    STRING_TYPE (name);
    STRING_TYPE (opencl_c_version);
    STRING_TYPE (device_version);
    STRING_TYPE (driver_version);
    STRING_TYPE (vendor);
    STRING_TYPE (extensions);
    STRING_TYPE (builtin_kernels);

    uint32_t vendor_id;
    //  uint32_t device_id;
    uint32_t address_bits;
    uint32_t mem_base_addr_align;

    uint32_t global_mem_cache_size;
    uint32_t global_mem_cache_type;
    uint64_t global_mem_size;
    uint32_t global_mem_cacheline_size;

    uint64_t double_fp_config;
    uint64_t single_fp_config;
    uint64_t half_fp_config;

    uint32_t local_mem_size;
    uint32_t local_mem_type;
    uint32_t max_clock_frequency;
    uint32_t max_compute_units;

    uint32_t max_constant_args;
    uint64_t max_constant_buffer_size;
    uint64_t max_mem_alloc_size;
    uint32_t max_parameter_size;

    uint32_t max_read_image_args;
    uint32_t max_write_image_args;
    uint32_t max_samplers;

    uint32_t max_work_item_dimensions;
    uint64_t max_work_group_size;
    uint64_t max_work_item_size_x;
    uint64_t max_work_item_size_y;
    uint64_t max_work_item_size_z;

    /* ############  CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE DEPRECATED */

    uint32_t native_vector_width_char;
    uint32_t native_vector_width_short;
    uint32_t native_vector_width_int;
    uint32_t native_vector_width_long;
    uint32_t native_vector_width_float;
    uint32_t native_vector_width_double;
    uint32_t native_vector_width_half;

    uint32_t preferred_vector_width_char;
    uint32_t preferred_vector_width_short;
    uint32_t preferred_vector_width_int;
    uint32_t preferred_vector_width_long;
    uint32_t preferred_vector_width_float;
    uint32_t preferred_vector_width_double;
    uint32_t preferred_vector_width_half;

    /* ############# SUBDEVICES - later */

    uint32_t printf_buffer_size;
    uint32_t profiling_timer_resolution;

    /* ########### images */

    uint32_t image2d_max_height;
    uint32_t image2d_max_width;
    uint32_t image3d_max_height;
    uint32_t image3d_max_width;
    uint32_t image3d_max_depth;
    uint64_t image_max_buffer_size;
    uint64_t image_max_array_size;

    ENUM_TYPE type;
    uint8_t available;
    uint8_t compiler_available;
    uint8_t endian_little;
    uint8_t error_correction_support;
    uint8_t image_support;
    uint8_t full_profile;

    ImgFormatInfo_t supported_image_formats[6];
  } DeviceInfo_t;

  typedef struct __attribute__ ((packed, aligned (8))) ConnectPeerMsg_s
  {
    uint16_t port;
    uint8_t session[SESSION_ID_LENGTH];
    char address[MAX_REMOTE_PARAM_LENGTH + 1];
  } ConnectPeerMsg_t;

  /* ########################## */
  /* ########################## */
  /* ########################## */
  /* ########################## */
  /* ########################## */
  /* ########################## */

  typedef struct __attribute__ ((packed, aligned (8))) MigrateD2DMsg_s
  {
    uint64_t size;
    uint32_t source_pid;
    uint32_t source_did;
    uint32_t dest_peer_id;
    uint32_t source_peer_id;
    uint32_t is_image;
    uint32_t is_external;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
  } MigrateD2DMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) CreateBufferMsg_s
  {
    uint64_t size;
    uint32_t flags;
  } CreateBufferMsg_t;

#ifdef ENABLE_RDMA
  typedef struct __attribute__ ((packed, aligned (8))) CreateRdmaBufferReply_s
  {
    uint64_t server_vaddr;
    uint32_t server_rkey;
  } CreateRdmaBufferReply_t;
#endif

  typedef struct __attribute__ ((packed, aligned (8))) FreeBufferMsg_s
  {
    uint64_t padding;
  } FreeBufferMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) ReadBufferMsg_s
  {
    uint64_t src_offset;
    uint64_t size;
    uint64_t content_size;
#ifdef ENABLE_RDMA
    uint64_t client_vaddr;
    uint32_t client_rkey;
#endif
  } ReadBufferMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) WriteBufferMsg_s
  {
    uint64_t dst_offset;
    uint64_t size;
    uint64_t content_size;
  } WriteBufferMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) CopyBufferMsg_s
  {
    uint32_t src_buffer_id;
    uint32_t dst_buffer_id;
    uint64_t src_offset;
    uint64_t dst_offset;
    uint64_t size;
  } CopyBufferMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) FillBufferMsg_s
  {
    uint64_t dst_offset;
    uint64_t size;
    uint64_t pattern_size;
  } FillBufferMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) ReadBufferRectMsg_s
  {
    vec3_t buffer_origin;
    vec3_t region;

    uint64_t buffer_row_pitch;
    uint64_t buffer_slice_pitch;

    uint64_t host_bytes;
#ifdef ENABLE_RDMA
    uint64_t client_vaddr;
    uint32_t client_rkey;
#endif
  } ReadBufferRectMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) WriteBufferRectMsg_s
  {
    vec3_t buffer_origin;
    vec3_t region;

    uint64_t buffer_row_pitch;
    uint64_t buffer_slice_pitch;

    uint64_t host_bytes;
  } WriteBufferRectMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) CopyBufferRectMsg_s
  {
    uint32_t src_buffer_id;
    uint32_t dst_buffer_id;

    vec3_t dst_origin;
    vec3_t src_origin;
    vec3_t region;

    uint64_t dst_row_pitch;
    uint64_t dst_slice_pitch;
    uint64_t src_row_pitch;
    uint64_t src_slice_pitch;

  } CopyBufferRectMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) CreateImageMsg_s
  {
    uint32_t flags;
    // format
    uint32_t channel_order;
    uint32_t channel_data_type;
    // desc
    uint32_t type;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
    uint32_t array_size;
    uint32_t row_pitch;
    uint32_t slice_pitch;
    //    cl_uint                 num_mip_levels;
    //    cl_uint                 num_samples;
  } CreateImageMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) CreateSamplerMsg_s
  {
    uint32_t normalized;
    uint32_t address_mode;
    uint32_t filter_mode;
  } CreateSamplerMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) CopyImage2ImageMsg_s
  {
    uint32_t src_image_id;
    uint32_t dst_image_id;

    vec3_t dst_origin;
    vec3_t src_origin;
    vec3_t region;

  } CopyImg2ImgMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) CopyBuf2ImgMsg_s
  {
    vec3_t origin;
    vec3_t region;

    uint32_t src_buf_id;
    uint64_t src_offset;
  } CopyBuf2ImgMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) CopyImg2BufMsg_s
  {
    vec3_t origin;
    vec3_t region;

    uint32_t dst_buf_id;
    uint64_t dst_offset;
  } CopyImg2BufMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) ReadImageRectMsg_s
  {
    vec3_t origin;
    vec3_t region;

    uint64_t host_bytes;
#ifdef ENABLE_RDMA
    uint64_t client_vaddr;
    uint32_t client_rkey;
#endif
  } ReadImageRectMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) WriteImageRectMsg_s
  {
    vec3_t origin;
    vec3_t region;

    uint64_t host_bytes;
  } WriteImageRectMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) FillImageRectMsg_s
  {
    vec3_t origin;
    vec3_t region;

  } FillImageRectMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) BuildProgramMsg_s
  {
    uint64_t payload_size;
    uint64_t options_len;
    uint32_t num_devices;
    uint32_t devices[MAX_REMOTE_DEVICES];
    uint32_t platforms[MAX_REMOTE_DEVICES];
    // program: char*
    // options: char*
  } BuildProgramMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) CreateKernelMsg_s
  {
    uint64_t name_len;
    uint32_t prog_id;
  } CreateKernelMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) FreeKernelMsg_s
  {
    uint32_t prog_id;
  } FreeKernelMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) RunKernelMsg_s
  {
    vec3_t global;
    vec3_t local;
    vec3_t offset;
    uint8_t has_local;
    uint8_t dim;
    // if new args are present
    uint16_t has_new_args;
    uint32_t args_num;
    uint64_t pod_arg_size;
  } RunKernelMsg_t;

  typedef struct __attribute__ ((packed, aligned (8))) RequestMsg_s
  {
    uint64_t msg_id;
    uint64_t event_id;
    uint32_t pid;
    uint32_t did;
    uint32_t client_did;
    uint32_t waitlist_size;

    uint32_t message_type;
    uint32_t obj_id;
    uint32_t cq_id;

    union
    {
      ServerInfoMsg_t server_info;
      ConnectPeerMsg_t connect_peer;

      CreateBufferMsg_t create_buffer;
      CreateSamplerMsg_t create_sampler;
      CreateImageMsg_t create_image;

      FreeBufferMsg_t free_buffer;

      MigrateD2DMsg_t migrate;
      ReadBufferMsg_t read;
      WriteBufferMsg_t write;
      CopyBufferMsg_t copy;
      FillBufferMsg_t fill_buffer;

      ReadBufferRectMsg_t read_rect;
      WriteBufferRectMsg_t write_rect;
      CopyBufferRectMsg_t copy_rect;

      CopyImg2ImgMsg_t copy_img2img;
      CopyBuf2ImgMsg_t copy_buf2img;
      CopyImg2BufMsg_t copy_img2buf;

      FillImageRectMsg_t fill_image;
      ReadImageRectMsg_t read_image_rect;
      WriteImageRectMsg_t write_image_rect;

      BuildProgramMsg_t build_program;
      CreateKernelMsg_t create_kernel;
      FreeKernelMsg_t free_kernel;
      RunKernelMsg_t run_kernel;
    } m;
  } RequestMsg_t;

  /* #################################################################### */
  /* #################################################################### */
  /* #################################################################### */
  /* #################################################################### */
  /* #################################################################### */
  /* #################################################################### */

  typedef enum PoclRemoteArgType
  {
    POD = 0,
    Pointer,
    Image,
    Sampler,
    Local
  } PoclRemoteArgType;

  typedef struct __attribute__ ((packed, aligned (8))) ArgumentInfo_s
  {
    STRING_TYPE (name);
    STRING_TYPE (type_name);
    uint32_t address_qualifier;
    uint32_t access_qualifier;
    uint32_t type_qualifier;
    PoclRemoteArgType type;
    // uint32_t type_size;
  } ArgumentInfo_t;

  /*
  typedef struct pocl_kernel_metadata_s
  {
    cl_uint num_args;
    cl_uint num_locals;
    size_t *local_sizes;
    char *name;
    char *attributes;
    struct pocl_argument_info *arg_info;
    cl_bitfield has_arg_metadata;
    size_t reqd_wg_size[OPENCL_MAX_DIMENSION];

    void **data;
  } pocl_kernel_metadata_t;
  */

  typedef struct EventTiming_s
  {
    uint64_t queued;
    uint64_t submitted;
    uint64_t started;
    uint64_t completed;
  } EventTiming_t;

  typedef struct __attribute__ ((packed, aligned (8))) KernelMetaInfo_s
  {
    STRING_TYPE (name);
    STRING_TYPE (attributes);
    vec3_t reqd_wg_size;
    uint64_t total_local_size;
    uint32_t num_args;
  } KernelMetaInfo_t;

  typedef struct __attribute__ ((packed, aligned (8))) ReplyMsg_s
  {
    uint64_t msg_id;
    // this is the Platform ID on remote side
    uint32_t pid;
    // this is the Device ID on remote side
    uint32_t did;
    // this is the Device ID on local (client) side
    uint32_t client_did;

    uint32_t message_type;
    uint32_t failed;
    int32_t fail_details;

    uint64_t data_size;
    uint32_t obj_id;

    // remote server timing data from libOpenCL
    EventTiming_t timing;
    // set by remote server
    uint64_t server_read_start_timestamp_ns;
    uint64_t server_read_end_timestamp_ns;
    uint64_t server_write_start_timestamp_ns;
  } ReplyMsg_t;

  /* ########################## */

  typedef struct __attribute__ ((packed, aligned (8))) PeerHandshake_s
  {
    uint64_t msg_id;
    uint32_t message_type;
    uint32_t peer_id;
    uint8_t session[SESSION_ID_LENGTH];
  } PeerHandshake_t;

  /* ########################## */

  typedef struct __attribute__ ((packed, aligned (8))) ClientHandshake_s
  {
    uint8_t session_id[SESSION_ID_LENGTH];
    uint16_t peer_port;
    uint8_t rdma_supported;
  } ClientHandshake_t;

  /* ########################## */

  static inline size_t
  request_size (uint32_t message_type)
  {
    size_t body;
    switch (message_type)
      {
      case MessageType_ConnectPeer:
        body = sizeof (ConnectPeerMsg_t);
        break;

      case MessageType_CreateBuffer:
        body = sizeof (CreateBufferMsg_t);
        break;

      case MessageType_FreeBuffer:
        body = sizeof (FreeBufferMsg_t);
        break;

      case MessageType_CreateSampler:
        body = sizeof (CreateSamplerMsg_t);
        break;

      case MessageType_CreateImage:
        body = sizeof (CreateImageMsg_t);
        break;
      case MessageType_CreateKernel:
        body = sizeof (CreateKernelMsg_t);
        break;
      case MessageType_FreeKernel:
        body = sizeof (FreeKernelMsg_t);
        break;

      case MessageType_BuildProgramFromSource:
      case MessageType_BuildProgramFromBinary:
      case MessageType_BuildProgramWithBuiltins:
        body = sizeof (BuildProgramMsg_t);
        break;

      case MessageType_MigrateD2D:
        body = sizeof (MigrateD2DMsg_t);
        break;

      case MessageType_ReadBuffer:
        body = sizeof (ReadBufferMsg_t);
        break;
      case MessageType_WriteBuffer:
        body = sizeof (WriteBufferMsg_t);
        break;
      case MessageType_CopyBuffer:
        body = sizeof (CopyBufferMsg_t);
        break;
      case MessageType_FillBuffer:
        body = sizeof (FillBufferMsg_t);
        break;

      case MessageType_ReadBufferRect:
        body = sizeof (ReadBufferRectMsg_t);
        break;
      case MessageType_WriteBufferRect:
        body = sizeof (WriteBufferRectMsg_t);
        break;
      case MessageType_CopyBufferRect:
        body = sizeof (CopyBufferRectMsg_t);
        break;

      case MessageType_CopyImage2Buffer:
        body = sizeof (CopyImg2BufMsg_t);
        break;
      case MessageType_CopyBuffer2Image:
        body = sizeof (CopyBuf2ImgMsg_t);
        break;
      case MessageType_CopyImage2Image:
        body = sizeof (CopyImg2ImgMsg_t);
        break;
      case MessageType_ReadImageRect:
        body = sizeof (ReadImageRectMsg_t);
        break;
      case MessageType_WriteImageRect:
        body = sizeof (WriteImageRectMsg_t);
        break;
      case MessageType_FillImageRect:
        body = sizeof (FillImageRectMsg_t);
        break;

      case MessageType_RunKernel:
        body = sizeof (RunKernelMsg_t);
        break;

      default:
        body = 0;
        break;
      }

    return offsetof (RequestMsg_t, m) + body;
  }

#ifdef ENABLE_RDMA
  static inline int
  pocl_request_is_rdma (RequestMsg_t *req, int is_p2p)
  {
    switch ((enum RequestMessageType) (req->message_type))
      {
      case MessageType_WriteBuffer:
      case MessageType_WriteBufferRect:
        return 1;
        break;
      case MessageType_MigrateD2D:
        // Migration from a device on the client is done using WriteBuffer
        // so MigrateD2D only ever has RDMA data in the p2p context
        return is_p2p ? 1 : 0;
        break;
      default:
        return 0;
      }
  }

  static inline int
  pocl_request_has_rdma_reply (RequestMsg_t *rep)
  {
    switch ((enum RequestMessageType) (rep->message_type))
      {
      case MessageType_ReadBuffer:
        return 1;
        break;
      default:
        return 0;
      }
  }
#endif

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif
