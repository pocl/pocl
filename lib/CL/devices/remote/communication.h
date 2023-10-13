/* communication.h - part of pocl-remote driver that talks to pocl-remote
   server

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

#ifndef POCL_REMOTE_COMMUNICATION_H
#define POCL_REMOTE_COMMUNICATION_H

#include "messages.h"

#ifdef ENABLE_RDMA
#include "pocl_rdma.h"
#include "uthash.h"
#endif

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#define MAX_ADDRESS_SIZE 512
#define MAX_ADDRESS_PORT_SIZE (MAX_ADDRESS_SIZE + 11) // 11 bytes for the port

#define REMOTE_DEV_DATA                                                       \
  remote_device_data_t *data = (remote_device_data_t *)device->data

#define REMOTE_SERV_DATA                                                      \
  remote_server_data_t *data = ((remote_device_data_t *)device->data)->server

#define REMOTE_SERV_DATA2 remote_server_data_t *data = (ddata->server)

#define MAX_STATUSES 984

typedef enum
{
  NETCMD_STARTED,
  NETCMD_WRITTEN,
  NETCMD_READ,
  NETCMD_FINISHED,
  NETCMD_FAILED
} network_command_status_t;

typedef void (*network_command_callback) (void *arg, _cl_command_node *node,
                                          size_t extra_rep_size);

typedef struct network_command network_command;

/** Lock used for synchronous commands */
typedef struct sync_t
{
  pocl_lock_t mutex;
  pocl_cond_t cond;
} sync_t;

/** Callback data used for asynchronous commands */
typedef struct async_t
{
  network_command_callback cb;
  void *arg;
  _cl_command_node *node;
} async_t;

#define CREATE_SYNC_NETCMD                                                    \
  network_command nc;                                                         \
  network_command *netcmd = &nc;                                              \
  memset (netcmd, 0, sizeof (network_command));                               \
  nc.receiver = data->inflight_queue;                                         \
  nc.status = NETCMD_STARTED;                                                 \
  nc.synchronous = 1;

#define CREATE_ASYNC_NETCMD                                                   \
  network_command *netcmd = calloc (1, sizeof (network_command));             \
  netcmd->status = NETCMD_STARTED;                                            \
  netcmd->event_id = node->sync.event.event->id;                              \
  struct event_node *n;                                                       \
  LL_COMPUTE_LENGTH (node->sync.event.event->wait_list, n,                    \
                     netcmd->req_waitlist_size);                              \
  if (netcmd->req_waitlist_size > 0)                                          \
    netcmd->req_wait_list                                                     \
        = calloc (netcmd->req_waitlist_size, sizeof (uint64_t));              \
  {                                                                           \
    uint64_t *dst = netcmd->req_wait_list;                                    \
    LL_FOREACH (node->sync.event.event->wait_list, n)                         \
    {                                                                         \
      *(dst++) = n->event->id;                                                \
    }                                                                         \
  }                                                                           \
  netcmd->receiver = data->inflight_queue;                                    \
  netcmd->synchronous = 0;                                                    \
  netcmd->data.async.cb = cb;                                                 \
  netcmd->data.async.arg = arg;                                               \
  netcmd->data.async.node = node;

typedef struct network_queue network_queue;

#ifdef ENABLE_RDMA
typedef struct rdma_buffer_info_s
{
  uint32_t mem_id;
  uint32_t remote_rkey;
  uint64_t remote_vaddr;
  UT_hash_handle hh; // To make this struct usable with uthash
} rdma_buffer_info_t;
#endif

struct network_command
{
  RequestMsg_t request;
  ReplyMsg_t reply;

  network_command *next;
  network_command *prev;
  uint64_t event_id;
  uint64_t *req_wait_list;
  const char *req_extra_data;
  const char *req_extra_data2;
  char *rep_extra_data;
  uint64_t req_waitlist_size;
  uint64_t req_extra_size;
  uint64_t req_extra_size2;
  uint64_t rep_extra_size;
  /* Points to an (optional) dynamic strings section appened after the message.
   */
  char *strings;
  network_command_status_t status;
  uint64_t client_write_start_timestamp_ns;
  uint64_t client_write_end_timestamp_ns;
  uint64_t client_read_start_timestamp_ns;
  uint64_t client_read_end_timestamp_ns;
  int synchronous;
  network_queue *receiver;
#ifdef ENABLE_RDMA
  struct ibv_mr *rdma_region;
#endif

  union
  {
    sync_t sync;
    async_t async;
  } data;
};

#define INITIAL_ARRAY_CAP 1024

// in nanoseconds
#define POCL_REMOTE_RECONNECT_TIMEOUT_NS 60 * 1000000000L

typedef struct remote_server_data_s
{
  char address[MAX_ADDRESS_SIZE];
  char address_with_port[MAX_ADDRESS_PORT_SIZE];
  char peer_address[MAX_ADDRESS_SIZE];
  uint64_t peer_id;
  unsigned slow_port;
  unsigned fast_port;
  unsigned peer_port;

  unsigned refcount;

  struct remote_server_data_s *next;
  struct remote_server_data_s *prev;

  uint8_t session[SESSION_ID_LENGTH];
  uint32_t available;
  sync_t setup_lock;
  int threads_awaiting_reconnect;
  int slow_socket_fd;
  int fast_socket_fd;

  uint32_t num_platforms;
  uint32_t num_devices;
  uint32_t *platform_devices;

  // network handling threads / ids
  network_queue *slow_read_queue;
  network_queue *fast_read_queue;
  network_queue *inflight_queue;
  network_queue *slow_write_queue;
  network_queue *fast_write_queue;
#ifdef ENABLE_RDMA
  network_queue *rdma_read_queue;
  network_queue *rdma_write_queue;
  rdma_data_t rdma_data;
  rdma_buffer_info_t *rdma_keys; // needs to be initialized to NULL, but we
                                 // memset(0) the whole struct anyway
  uint8_t use_rdma;
#endif
#ifdef ENABLE_TRAFFIC_MONITOR
  network_queue *traffic_monitor;
  uint64_t rx_bytes_requested;
  uint64_t rx_bytes_confirmed;
  uint64_t tx_bytes_submitted;
  uint64_t tx_bytes_confirmed;
#endif

  // ID maps.
  // TODO locking required ??? prolly not, because all create/release are
  // called sequentially

  SMALL_VECTOR_DEFINE (uint32_t, buffer_ids, INITIAL_ARRAY_CAP);

  SMALL_VECTOR_DEFINE (uint32_t, program_ids, INITIAL_ARRAY_CAP);

  SMALL_VECTOR_DEFINE (uint32_t, kernel_ids, INITIAL_ARRAY_CAP);

  SMALL_VECTOR_DEFINE (uint32_t, image_ids, INITIAL_ARRAY_CAP);

  SMALL_VECTOR_DEFINE (uint32_t, sampler_ids, INITIAL_ARRAY_CAP);

} remote_server_data_t;

#define RETURN_IF_REMOTE_ID(map, id)                                          \
  do                                                                          \
    {                                                                         \
      if (small_vector_find_##map##_ids (data, id) >= 0)                      \
        return CL_SUCCESS;                                                    \
    }                                                                         \
  while (0)

#define RETURN_IF_NOT_REMOTE_ID(map, id)                                      \
  do                                                                          \
    {                                                                         \
      if (small_vector_find_##map##_ids (data, id) == -1)                     \
        return CL_SUCCESS;                                                    \
    }                                                                         \
  while (0)

#define SET_REMOTE_ID(map, id)                                                \
  do                                                                          \
    {                                                                         \
      small_vector_append_##map##_ids (data, id);                             \
    }                                                                         \
  while (0)

#define UNSET_REMOTE_ID(map, id)                                              \
  do                                                                          \
    {                                                                         \
      small_vector_remove_##map##_ids (data, id);                             \
    }                                                                         \
  while (0)

typedef struct pocl_remote_event_data_s
{
  pocl_cond_t event_cond;
} pocl_remote_event_data_t;

typedef struct remote_queue_data_s
{
  pocl_cond_t cq_cond;
  char *printf_buffer;
} remote_queue_data_t;

typedef struct remote_device_data_s
{
  remote_server_data_t *server;
  cl_image_format *supported_image_formats;
  char *build_hash;
  unsigned remote_device_index;
  unsigned remote_platform_index;
  unsigned local_did;

  // migrated -> ready to launch queue
  _cl_command_node *work_queue;
  // finished queue
  _cl_command_node *finished_list;

  // driver wake + lock
  ALIGN_CACHE (pocl_lock_t wq_lock);
  ALIGN_CACHE (pocl_cond_t wakeup_cond);
  ALIGN_CACHE (pocl_lock_t mem_lock);

  /* device pthread */
  pocl_thread_t driver_thread_id;
  size_t driver_thread_exit_requested;

} remote_device_data_t;

typedef struct kernel_data_s
{
  char *pod_arg_storage;
  uint64_t pod_total_size;
  uint64_t *arg_array;
} kernel_data_t;

typedef struct program_data_s
{
  char *kernel_meta_bytes;
  size_t kernel_meta_size;
  size_t refcount;
} program_data_t;

// ##################################################################################
// ##################################################################################
// ##################################################################################

cl_int pocl_network_init_device (cl_device_id device,
                                 remote_device_data_t *ddata, int dev_idx,
                                 const char *const parameters);

cl_int pocl_network_free_device (cl_device_id device);

cl_int pocl_network_setup_peer_mesh ();

cl_int pocl_network_setup_devinfo (cl_device_id device,
                                   remote_device_data_t *ddata,
                                   remote_server_data_t *data, uint32_t pid,
                                   uint32_t did);

cl_int pocl_network_create_buffer (remote_device_data_t *d, uint32_t mem_id,
                                   uint32_t mem_flags, uint64_t mem_size,
                                   void **device_addr);

cl_int pocl_network_free_buffer (remote_device_data_t *d, uint32_t mem_id);

cl_int pocl_network_create_kernel (remote_device_data_t *ddata,
                                   const char *name, uint32_t prog_id,
                                   uint32_t kernel_id, kernel_data_t *kd);

cl_int pocl_network_free_kernel (remote_device_data_t *ddata,
                                 kernel_data_t *kernel, uint32_t kernel_id,
                                 uint32_t program_id);

cl_int pocl_network_setup_metadata (char *buffer, size_t total_size,
                                    cl_program program, size_t *num_kernels,
                                    pocl_kernel_metadata_t **kernel_meta);

cl_int pocl_network_build_program (
    remote_device_data_t *ddata, const void *payload, size_t payload_size,
    int is_binary, int is_builtin, int is_spirv, uint32_t prog_id,
    const char *options, char **kernel_meta_bytes, size_t *kernel_meta_size,
    uint32_t *devices, uint32_t *platforms, size_t num_devices,
    char **build_log, char **binaries, size_t *binary_sizes);

cl_int pocl_network_free_program (remote_device_data_t *ddata,
                                  uint32_t prog_id);

cl_int pocl_network_create_queue (remote_device_data_t *ddata,
                                  uint32_t queue_id);

cl_int pocl_network_free_queue (remote_device_data_t *ddata,
                                uint32_t queue_id);

cl_int pocl_network_create_sampler (remote_device_data_t *ddata,
                                    cl_bool normalized_coords,
                                    cl_addressing_mode addressing_mode,
                                    cl_filter_mode filter_mode,
                                    uint32_t samp_id);

cl_int pocl_network_free_sampler (remote_device_data_t *ddata,
                                  uint32_t samp_id);

cl_int pocl_network_create_image (remote_device_data_t *ddata, cl_mem image);

cl_int pocl_network_free_image (remote_device_data_t *ddata,
                                uint32_t image_id);

// ##################################################################################
// ##################################################################################
// ##################################################################################

cl_int pocl_network_migrate_d2d (
    uint32_t cq_id, uint32_t mem_id, uint32_t size_id, unsigned mem_is_image,
    uint32_t height, uint32_t width, uint32_t depth, size_t size,
    remote_device_data_t *dest, remote_device_data_t *source,
    network_command_callback cb, void *arg, _cl_command_node *node);

cl_int pocl_network_read (uint32_t cq_id, remote_device_data_t *ddata,
                          uint32_t mem, uint32_t size_id, void *host_ptr,
                          size_t offset, size_t size,
                          network_command_callback cb, void *arg,
                          _cl_command_node *node);

cl_int pocl_network_write (uint32_t cq_id, remote_device_data_t *ddata,
                           uint32_t mem, const void *host_ptr, size_t offset,
                           size_t size, network_command_callback cb, void *arg,
                           _cl_command_node *node);

cl_int pocl_network_copy (uint32_t cq_id, remote_device_data_t *ddata,
                          uint32_t src, uint32_t dst, uint32_t size_buf,
                          size_t src_offset, size_t dst_offset, size_t size,
                          network_command_callback cb, void *arg,
                          _cl_command_node *node);

cl_int pocl_network_read_rect (
    uint32_t cq_id, remote_device_data_t *ddata, uint32_t src_mem,
    const size_t *__restrict__ const buffer_origin,
    const size_t *__restrict__ const region, size_t const buffer_row_pitch,
    size_t const buffer_slice_pitch,
    void *host_ptr, size_t size, network_command_callback cb, void *arg,
    _cl_command_node *node);

cl_int pocl_network_write_rect (
    uint32_t cq_id, remote_device_data_t *ddata, uint32_t dst_mem,
    const size_t *__restrict__ const buffer_origin,
    const size_t *__restrict__ const region, size_t const buffer_row_pitch,
    size_t const buffer_slice_pitch,
    const void *host_ptr, size_t size, network_command_callback cb, void *arg,
    _cl_command_node *node);

cl_int pocl_network_copy_rect (
    uint32_t cq_id, remote_device_data_t *ddata, uint32_t src, uint32_t dst,
    const size_t *__restrict__ const dst_origin,
    const size_t *__restrict__ const src_origin,
    const size_t *__restrict__ const region, size_t const dst_row_pitch,
    size_t const dst_slice_pitch, size_t const src_row_pitch,
    size_t const src_slice_pitch, network_command_callback cb, void *arg,
    _cl_command_node *node);

cl_int pocl_network_fill_buffer (uint32_t cq_id, remote_device_data_t *ddata,
                                 uint32_t mem, size_t size, size_t offset,
                                 const void *__restrict__ pattern,
                                 size_t pattern_size,
                                 network_command_callback cb, void *arg,
                                 _cl_command_node *node);

cl_int pocl_network_run_kernel (uint32_t cq_id, remote_device_data_t *ddata,
                                cl_kernel kernel, kernel_data_t *kd,
                                int requires_kernarg_update, unsigned dim,
                                vec3_t local, vec3_t global, vec3_t offset,
                                network_command_callback cb, void *arg,
                                _cl_command_node *node);

/****************************************************************************/

cl_int pocl_network_copy_image_rect (
    uint32_t cq_id, remote_device_data_t *ddata, uint32_t src_remote_id,
    uint32_t dst_remote_id, const size_t *__restrict__ const src_origin,
    const size_t *__restrict__ const dst_origin,
    const size_t *__restrict__ const region, network_command_callback cb,
    void *arg, _cl_command_node *node);

cl_int pocl_network_copy_buf2img (uint32_t cq_id, remote_device_data_t *ddata,
                                  uint32_t src_remote_id, size_t src_offset,
                                  uint32_t dst_remote_id,
                                  const size_t *__restrict__ const origin,
                                  const size_t *__restrict__ const region,
                                  network_command_callback cb, void *arg,
                                  _cl_command_node *node);

cl_int pocl_network_write_image_rect (
    uint32_t cq_id, remote_device_data_t *ddata, uint32_t dst_remote_id,
    const size_t *__restrict__ const origin,
    const size_t *__restrict__ const region, const void *__restrict__ p,
    size_t alloc_size, network_command_callback cb, void *arg,
    _cl_command_node *node);

cl_int pocl_network_copy_img2buf (uint32_t cq_id, remote_device_data_t *ddata,
                                  uint32_t dst_remote_id, size_t dst_offset,
                                  uint32_t src_remote_id,
                                  const size_t *__restrict__ const origin,
                                  const size_t *__restrict__ const region,
                                  network_command_callback cb, void *arg,
                                  _cl_command_node *node);

cl_int pocl_network_read_image_rect (
    uint32_t cq_id, remote_device_data_t *ddata, uint32_t src_remote_id,
    const size_t *__restrict__ const origin,
    const size_t *__restrict__ const region, void *p, size_t alloc_size,
    network_command_callback cb, void *arg, _cl_command_node *node);

cl_int pocl_network_fill_image (uint32_t cq_id, remote_device_data_t *ddata,
                                uint32_t image_id,
                                const size_t *__restrict__ const origin,
                                const size_t *__restrict__ const region,
                                cl_uint4 *fill_pixel,
                                network_command_callback cb, void *arg,
                                _cl_command_node *node);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif /* POCL_SERVER_H */
