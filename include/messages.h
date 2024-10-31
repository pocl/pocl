/* messages.h - message types for PoCL-Remote communication

   Copyright (c) 2018 Michal Babej / Tampere University of Technology
   Copyright (c) 2019-2023 Jan Solanti / Tampere University
   Copyright (c) 2023-2024 Pekka Jääskeläinen / Intel Finland Oy

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

#include "pocl.h"
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

#define AUTHKEY_LENGTH 16

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
    MessageType_InvalidRequest,
    MessageType_CreateOrAttachSession,
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
    MessageType_BuildProgramWithDefinedBuiltins,
    // Special message type for SPIR-V IL for now. No support for
    // vendor-specific ILs.
    MessageType_BuildProgramFromSPIRV,
    MessageType_CompileProgramFromSPIRV,
    MessageType_CompileProgramFromSource,
    MessageType_LinkProgram,
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
    MessageType_InvalidReply,
    MessageType_CreateOrAttachSessionReply,
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

  typedef struct __attribute__ ((packed)) ImgFormatType_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint32_t channel_order;
    uint32_t channel_data_type;
  } ImgFormatType_t;

  typedef struct __attribute__ ((packed)) ImgFormatInfo_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint32_t memobj_type;
    uint32_t num_formats;
    ImgFormatType_t formats[MAX_IMAGE_FORMAT_TYPES];
  } ImgFormatInfo_t;

  typedef struct __attribute__ ((packed))
  CreateOrAttachSessionMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t peer_id;
    uint16_t peer_port;
    uint8_t use_rdma;
    uint8_t fast_socket;
  } CreateOrAttachSessionMsg_t;

  typedef struct __attribute__ ((packed))
  CreateOrAttachSessionReply_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t session;
    uint8_t authkey[AUTHKEY_LENGTH];
    uint16_t peer_port;
    uint8_t use_rdma;
  } CreateOrAttachSessionReply_t;

  typedef struct __attribute__ ((packed)) DeviceInfo_s
  {

    /* ######## device properties ############## */

    /* Offsets to the strings-section. */
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t name;
    uint64_t opencl_c_version;
    uint64_t device_version;
    uint64_t driver_version;
    uint64_t vendor;
    uint64_t extensions;
    uint64_t builtin_kernels;
    uint64_t supported_spir_v_versions;

    uint32_t vendor_id;
    //  uint32_t device_id;
    uint32_t address_bits;
    uint32_t mem_base_addr_align;

    uint32_t global_mem_cache_size;
    uint32_t global_mem_cache_type;
    uint64_t global_mem_size;
    uint32_t global_mem_cacheline_size;

    /* The starting address of a region from which coarse grain SVM
       allocations should be made. */
    uint64_t svm_pool_start_address;
    /* And the size of it. Set to 0 in case CG SVM is not supported
       by the remote device. */
    uint64_t svm_pool_size;

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

    /* ########### subgroups */
    uint32_t max_num_sub_groups;

    ENUM_TYPE type;
    uint8_t available;
    uint8_t compiler_available;
    uint8_t endian_little;
    uint8_t error_correction_support;
    uint8_t image_support;
    uint8_t full_profile;

    cl_bool host_unified_memory;

    ImgFormatInfo_t supported_image_formats[6];

    /* If a single (vendor-specific) device info is queried (instead
       of all the basic device info data all at once), its data
       will be stored in this union verbatim. */
    union
    {
      /* E.g. for CL_​DEVICE_​SUB_​GROUP_​SIZES_​INTEL. */
      size_t size_t_array[16];
      cl_uint uint_val;
      cl_bool bool_val;
      cl_device_feature_capabilities_intel intel_capab_val;
    } specific_info_data;

    /* The size of the value is returned in specificInfoSize. Zero
       is used to signal an unknown/unsupported key type. */
    uint32_t specific_info_size;
  } DeviceInfo_t;

  typedef struct __attribute__ ((packed)) ConnectPeerMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint16_t port;
    uint64_t session;
    uint8_t authkey[AUTHKEY_LENGTH];
    char address[MAX_REMOTE_PARAM_LENGTH + 1];
  } ConnectPeerMsg_t;

  /* ########################## */
  /* ########################## */
  /* ########################## */
  /* ########################## */
  /* ########################## */
  /* ########################## */

  typedef struct __attribute__ ((packed)) MigrateD2DMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t size;
    uint32_t source_pid;
    uint32_t source_did;
    uint32_t size_id;
    uint32_t dest_peer_id;
    uint32_t source_peer_id;
    uint32_t is_image;
    uint32_t is_external;
    uint32_t width;
    uint32_t height;
    uint32_t depth;
  } MigrateD2DMsg_t;

  typedef struct __attribute__ ((packed)) CreateBufferMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t size;
    uint32_t flags;
    /* If non-zero, a previously allocated SVM pointer to be wrapped as
       the backing store for the buffer OR a pointer to a host-side
       backing store. Should set to CL_MEM_USES_SVM_POINTER to flags,
       if the former. */
    uint64_t host_ptr;
    /* Parent buffer id, if this is a sub-buffer allocation request. */
    pocl_obj_id_t parent_id;
    /* The offset inside the parent buffer, if this is a sub-buffer. */
    uint64_t origin;
  } CreateBufferMsg_t;

  typedef struct __attribute__ ((packed)) CreateBufferReply_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t device_addr;
  } CreateBufferReply_t;

#ifdef ENABLE_RDMA
  typedef struct __attribute__ ((packed)) CreateRdmaBufferReply_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t server_vaddr;
    uint32_t server_rkey;
  } CreateRdmaBufferReply_t;
#endif

  typedef struct __attribute__ ((packed)) FreeBufferMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t padding;
    /* If set to 1, the id of the buffer is the device side SVM allocation
       address to free, otherwise a cl_mem id.*/
    unsigned char is_svm;
  } FreeBufferMsg_t;

  typedef struct __attribute__ ((packed)) ReadBufferMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t src_offset;
    uint64_t size;
    uint64_t content_size;
    pocl_obj_id_t content_size_id;
#ifdef ENABLE_RDMA
    uint64_t client_vaddr;
    uint32_t client_rkey;
#endif
    /* If set to 1, the buffer to be written is an SVM buffer, not a cl_mem
       one. In that case, the obj_id of the request is set to the raw svm pool
       offset adjusted (remote VM) pointer instead of a cl_mem object id. */
    unsigned char is_svm;
  } ReadBufferMsg_t;

  typedef struct __attribute__ ((packed)) WriteBufferMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t dst_offset;
    uint64_t size;
    uint64_t content_size;
    /* If set to 1, the buffer to be written is an SVM buffer, not a cl_mem
       one. In that case, the obj_id of the request is set to the raw svm pool
       offset adjusted (remote VM) pointer instead of a cl_mem object id. */
    unsigned char is_svm;
  } WriteBufferMsg_t;

  typedef struct __attribute__ ((packed)) CopyBufferMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint32_t src_buffer_id;
    uint32_t dst_buffer_id;
    uint32_t size_buffer_id;
    uint64_t src_offset;
    uint64_t dst_offset;
    uint64_t size;
  } CopyBufferMsg_t;

  typedef struct __attribute__ ((packed)) FillBufferMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t dst_offset;
    uint64_t size;
    uint64_t pattern_size;
  } FillBufferMsg_t;

  typedef struct __attribute__ ((packed)) ReadBufferRectMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
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

  typedef struct __attribute__ ((packed)) WriteBufferRectMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    vec3_t buffer_origin;
    vec3_t region;

    uint64_t buffer_row_pitch;
    uint64_t buffer_slice_pitch;

    uint64_t host_bytes;
  } WriteBufferRectMsg_t;

  typedef struct __attribute__ ((packed)) CopyBufferRectMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
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

  typedef struct __attribute__ ((packed)) CreateImageMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
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

  typedef struct __attribute__ ((packed)) CreateSamplerMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint32_t normalized;
    uint32_t address_mode;
    uint32_t filter_mode;
  } CreateSamplerMsg_t;

  typedef struct __attribute__ ((packed)) CopyImage2ImageMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint32_t src_image_id;
    uint32_t dst_image_id;

    vec3_t dst_origin;
    vec3_t src_origin;
    vec3_t region;

  } CopyImg2ImgMsg_t;

  typedef struct __attribute__ ((packed)) CopyBuf2ImgMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    vec3_t origin;
    vec3_t region;

    uint32_t src_buf_id;
    uint64_t src_offset;
  } CopyBuf2ImgMsg_t;

  typedef struct __attribute__ ((packed)) CopyImg2BufMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    vec3_t origin;
    vec3_t region;

    uint32_t dst_buf_id;
    uint64_t dst_offset;
  } CopyImg2BufMsg_t;

  typedef struct __attribute__ ((packed)) ReadImageRectMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    vec3_t origin;
    vec3_t region;

    uint64_t host_bytes;
#ifdef ENABLE_RDMA
    uint64_t client_vaddr;
    uint32_t client_rkey;
#endif
  } ReadImageRectMsg_t;

  typedef struct __attribute__ ((packed)) WriteImageRectMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    vec3_t origin;
    vec3_t region;

    uint64_t host_bytes;
  } WriteImageRectMsg_t;

  typedef struct __attribute__ ((packed)) FillImageRectMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    vec3_t origin;
    vec3_t region;

  } FillImageRectMsg_t;

  typedef struct __attribute__ ((packed)) BuildProgramMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t payload_size;
    uint64_t options_len;
    // nonzero, if the program's memory accesses should be offset-adjusted
    // to match the SVM region starts in the remote device and the host
    uint64_t svm_region_offset;
    uint32_t num_devices;
    uint32_t devices[MAX_REMOTE_DEVICES];
    uint32_t platforms[MAX_REMOTE_DEVICES];
    // program: char*
    // options: char*
  } BuildProgramMsg_t;

  typedef struct __attribute__ ((packed)) CreateKernelMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t name_len;
    uint32_t prog_id;
  } CreateKernelMsg_t;

  typedef struct __attribute__ ((packed)) FreeKernelMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint32_t prog_id;
  } FreeKernelMsg_t;

  typedef struct __attribute__ ((packed)) RunKernelMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
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

  typedef struct __attribute__ ((packed)) DeviceInfoMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    /* If non-zero, the client has requested a specific device info
       instead of all standard ones at once. */
    cl_device_info id;
  } DeviceInfoMsg_t;

  /* ########################## */

  typedef struct __attribute__ ((packed)) PeerHandshake_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t peer_id;
  } PeerHandshake_t;

  /* ########################## */

  typedef struct __attribute__ ((packed)) RequestMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    uint64_t session;
    uint8_t authkey[AUTHKEY_LENGTH];
    uint64_t msg_id;
    uint64_t event_id;
    uint32_t pid;
    uint32_t did;
    uint32_t client_did;
    uint32_t waitlist_size;

    uint32_t message_type;
    uint64_t obj_id;
    uint32_t cq_id;

    union
    {
      CreateOrAttachSessionMsg_t get_session;
      ConnectPeerMsg_t connect_peer;
      PeerHandshake_t peer_handshake;

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

      DeviceInfoMsg_t device_info;
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

  typedef struct __attribute__ ((packed)) ArgumentInfo_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
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

  typedef struct __attribute__ ((packed)) KernelMetaInfo_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
    STRING_TYPE (name);
    STRING_TYPE (attributes);
    vec3_t reqd_wg_size;
    uint64_t total_local_size;
    uint32_t num_args;
  } KernelMetaInfo_t;

  typedef struct __attribute__ ((packed)) ReplyMsg_s
  {
    POCL_ALIGNAS(8) // Meant for aligning the structure, not the members.
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
    /* This has to be 64b since freeBuffer() uses it for the SVM pointer. */
    uint64_t obj_id;

    /* If the reply has a dynamic pool of c-strings after the end of
       the structure's fields, this is set to its size. */
    /* The actual strings will be appended after the object as a sequence
       of 0-terminated strings.*/
    uint64_t strings_size;

    // remote server timing data from libOpenCL
    EventTiming_t timing;
    // set by remote server
    uint64_t server_read_start_timestamp_ns;
    uint64_t server_read_end_timestamp_ns;
    uint64_t server_write_start_timestamp_ns;
    union
    {
      CreateOrAttachSessionReply_t get_session;
      PeerHandshake_t peer_handshake;
      CreateBufferReply_t create_buffer;
    } m;
  } ReplyMsg_t;

  /* ########################## */

  static inline size_t
  request_size (uint32_t message_type)
  {
    size_t body;
    switch (message_type)
      {
      case MessageType_CreateOrAttachSession:
        body = sizeof (CreateOrAttachSessionMsg_t);
        break;
      case MessageType_ConnectPeer:
        body = sizeof (ConnectPeerMsg_t);
        break;
      case MessageType_PeerHandshake:
        body = sizeof (PeerHandshake_t);
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
      case MessageType_BuildProgramFromSPIRV:
      case MessageType_CompileProgramFromSource:
      case MessageType_CompileProgramFromSPIRV:
      case MessageType_BuildProgramWithBuiltins:
      case MessageType_BuildProgramWithDefinedBuiltins:
      case MessageType_LinkProgram:
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
      case MessageType_DeviceInfo:
        body = sizeof (DeviceInfoMsg_t);
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
