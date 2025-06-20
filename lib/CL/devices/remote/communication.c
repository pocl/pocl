/* communication.c - part of pocl-remote driver that talks to pocl-remote
   server

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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <unistd.h>

#include "CL/cl_ext.h"
#include "CL/cl_platform.h"
#include "common.h"
#include "pocl.h"
#include "pocl_cl.h"
#include "pocl_debug.h"
#include "pocl_image_util.h"
#include "pocl_networking.h"
#include "pocl_threads.h"
#include "pocl_timing.h"
#include "remote.h"
#include "spirv_queries.h"
#include "utlist.h"
#include "utlist_addon.h"

#ifdef ENABLE_RDMA
#include "pocl_rdma.h"
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
#endif

#ifdef HAVE_LINUX_VSOCK_H
#include <linux/vm_sockets.h>
#endif

/* TODO mess */
#include "communication.h"

/* https://access.redhat.com/documentation/en-US/Red_Hat_Enterprise_MRG/1.2/html/Realtime_Tuning_Guide/sect-Realtime_Tuning_Guide-Application_Tuning_and_Deployment-TCP_NODELAY_and_Small_Buffer_Writes.html
 */
/* https://eklitzke.org/the-caveats-of-tcp-nodelay */
/* to be used with TCP_NODELAY:
 * ssize_t writev(int fildes, const struct iovec *iov, int iovcnt); */
#include <poll.h>
#include <sys/uio.h>

#ifdef HAVE_LTTNG_UST

#include "pocl_lttng.h"

#define TP_MSG_RECEIVED(msg_id, event_id, local_did, remote_did, type,        \
                        status)                                               \
  tracepoint (pocl_trace, msg_received, msg_id, event_id, local_did,          \
              remote_did, type, status);
#define TP_MSG_SENT(msg_id, event_id, local_did, remote_did, type, status)    \
  tracepoint (pocl_trace, msg_sent, msg_id, event_id, local_did, remote_did,  \
              type, status);

#else

#define TP_MSG_RECEIVED(msg_id, event_id, local_did, remote_did, type, status)
#define TP_MSG_SENT(msg_id, event_id, local_did, remote_did, type, status)

#endif

#define TP_READ_BUFFER(msg_id, dev_id, queue_id, evt_id)
#define TP_WRITE_BUFFER(msg_id, dev_id, queue_id, evt_id)
#define TP_COPY_BUFFER(msg_id, dev_id, queue_id, evt_id)

#define TP_READ_BUFFER_RECT(msg_id, dev_id, queue_id, evt_id)
#define TP_WRITE_BUFFER_RECT(msg_id, dev_id, queue_id, evt_id)
#define TP_COPY_BUFFER_RECT(msg_id, dev_id, queue_id, evt_id)

#define TP_READ_IMAGE_RECT(msg_id, dev_id, queue_id, evt_id)
#define TP_WRITE_IMAGE_RECT(msg_id, dev_id, queue_id, evt_id)
#define TP_COPY_IMAGE_RECT(msg_id, dev_id, queue_id, evt_id)
#define TP_COPY_IMAGE_2_BUF(msg_id, dev_id, queue_id, evt_id)
#define TP_COPY_BUF_2_IMAGE(msg_id, dev_id, queue_id, evt_id)

#define TP_FILL_BUFFER(msg_id, dev_id, queue_id, evt_id)
#define TP_FILL_IMAGE(msg_id, dev_id, queue_id, evt_id)

#define TP_NDRANGE_KERNEL(msg_id, dev_id, queue_id, evt_id, kernel_id, name)
#define TP_COMMAND_BUFFER(msg_id, evt_id, cmdbuf_id, name)

static uint64_t last_message_id = 1992;
static uint64_t last_peer_id = 42;

#ifndef POLLRDHUP
#define POLLRDHUP 0
#endif
#define POLLFD_ERROR_BITS (POLLHUP | POLLERR | POLLNVAL | POLLRDHUP)

#define CHECK_READ_INNER(readb, timeout_flag)                                 \
  do                                                                          \
    {                                                                         \
      if (readb < 0)                                                          \
        {                                                                     \
          int e = errno;                                                      \
          if (timeout_flag && ((e == EAGAIN) || (e == EWOULDBLOCK)))          \
            continue;                                                         \
          else                                                                \
            {                                                                 \
              POCL_MSG_PRINT_REMOTE ("error %i on read() call at " __FILE__   \
                                     ":%i, trying to reconnect\n",            \
                                     e, __LINE__);                            \
              goto TRY_RECONNECT;                                             \
            }                                                                 \
        }                                                                     \
      if (readb == 0)                                                         \
        {                                                                     \
          POCL_MSG_PRINT_REMOTE ("Filedescriptor closed (server "             \
                                 "disconnect). Trying to reconnect\n");       \
          goto TRY_RECONNECT;                                                 \
        }                                                                     \
    }                                                                         \
  while (0)

#define CHECK_READ(readb) CHECK_READ_INNER (readb, 0)
#define CHECK_READ_TIMEOUT(readb) CHECK_READ_INNER (readb, 1)

#define CHECK_WRITE(call)                                                     \
  do                                                                          \
    {                                                                         \
      int res = call;                                                         \
      if (res < 0)                                                            \
        {                                                                     \
          int e = errno;                                                      \
          POCL_MSG_PRINT_REMOTE ("error %i on write() call at " __FILE__      \
                                 ":%i, trying to reconnect\n",                \
                                 e, __LINE__);                                \
          goto TRY_RECONNECT;                                                 \
        }                                                                     \
    }                                                                         \
  while (0)

#define REQUEST(type)                                                         \
  RequestMsg_t *req = &netcmd->request;                                       \
  memset (req, 0, sizeof (RequestMsg_t));                                     \
  req->session = data->session;                                               \
  memcpy (req->authkey, data->authkey, AUTHKEY_LENGTH);                       \
  req->message_type = MessageType_##type;                                     \
  req->msg_id = POCL_ATOMIC_INC (last_message_id);                            \
  req->event_id = netcmd->event_id;                                           \
  req->did = ddata->remote_device_index;                                      \
  req->client_did = ddata->local_did;                                         \
  req->pid = ddata->remote_platform_index;                                    \
  req->obj_id = (uint32_t)(-1);

#define ID_REQUEST(type, req_id)                                              \
  RequestMsg_t *req = &netcmd->request;                                       \
  memset (req, 0, sizeof (RequestMsg_t));                                     \
  req->session = data->session;                                               \
  memcpy (req->authkey, data->authkey, AUTHKEY_LENGTH);                       \
  req->message_type = MessageType_##type;                                     \
  req->msg_id = POCL_ATOMIC_INC (last_message_id);                            \
  req->event_id = netcmd->event_id;                                           \
  req->did = ddata->remote_device_index;                                      \
  req->client_did = ddata->local_did;                                         \
  req->pid = ddata->remote_platform_index;                                    \
  req->obj_id = (uint64_t)req_id;

#define REQUEST_PEERCONN                                                      \
  RequestMsg_t *req = &netcmd->request;                                       \
  memset (req, 0, sizeof (RequestMsg_t));                                     \
  req->session = data->session;                                               \
  memcpy (req->authkey, data->authkey, AUTHKEY_LENGTH);                       \
  req->message_type = MessageType_ConnectPeer;                                \
  req->msg_id = POCL_ATOMIC_INC (last_message_id);                            \
  req->event_id = netcmd->event_id;                                           \
  req->obj_id = (uint64_t)(-1);

#define SEND_REQ_SLOW                                                         \
  POCL_LOCK (data->slow_write_queue->mutex);                                  \
  DL_APPEND (data->slow_write_queue->queue, netcmd);                          \
  POCL_SIGNAL_COND (data->slow_write_queue->cond);                            \
  POCL_UNLOCK (data->slow_write_queue->mutex);

#define SEND_REQ_FAST                                                         \
  POCL_LOCK (data->fast_write_queue->mutex);                                  \
  DL_APPEND (data->fast_write_queue->queue, netcmd);                          \
  POCL_SIGNAL_COND (data->fast_write_queue->cond);                            \
  POCL_UNLOCK (data->fast_write_queue->mutex);

#define SEND_REQ_RDMA                                                         \
  POCL_LOCK (data->rdma_write_queue->mutex);                                  \
  DL_APPEND (data->rdma_write_queue->queue, netcmd);                          \
  POCL_SIGNAL_COND (data->rdma_write_queue->cond);                            \
  POCL_UNLOCK (data->rdma_write_queue->mutex);

#define NETWORK_BUF_SIZE_FAST (4 * 1024)
#define NETWORK_BUF_SIZE_SLOW (4 * 1024 * 1024)

static remote_server_data_t *servers = NULL;

struct network_queue
{
  network_command *queue;
  pocl_lock_t mutex;
  pocl_cond_t cond;
  pocl_thread_t thread_id;
  int exit_requested;
};

typedef struct network_queue_arg
{
  remote_server_data_t *remote;
  network_queue *in_flight;
  network_queue *ours;
  remote_connection_t *connection;
} network_queue_arg;

#define SETUP_NETW_Q(n)                                                       \
  POCL_INIT_LOCK (n->mutex);                                                  \
  POCL_INIT_COND (n->cond);

/* n->dev = device; n->devd = devd; n->d = d; s = server data; c = connection
 */
#define SETUP_NETW_Q_ARG(n, s, o, c)                                          \
  n = calloc (1, sizeof (network_queue_arg));                                 \
  n->in_flight = d->inflight_queue;                                           \
  n->ours = o;                                                                \
  n->remote = s;                                                              \
  n->connection = c;

/*****************************************************************************/
/*****************************************************************************/

SMALL_VECTOR_HELPERS (buffer_ids, remote_server_data_t, uint32_t, buffer_ids)

SMALL_VECTOR_HELPERS (kernel_ids, remote_server_data_t, uint32_t, kernel_ids)

SMALL_VECTOR_HELPERS (program_ids, remote_server_data_t, uint32_t, program_ids)

SMALL_VECTOR_HELPERS (image_ids, remote_server_data_t, uint32_t, image_ids)

SMALL_VECTOR_HELPERS (sampler_ids, remote_server_data_t, uint32_t, sampler_ids)

/*****************************************************************************/
/*****************************************************************************/

#define PKT_THRESHOLD 1200

#define NUM_SERVER_SOCKET_THREADS 4

/*
 * # A note on socket error handling
 *
 * The socket reconnection code may look intimidatingly all over the place, but
 * its logic is fairly straightforward:
 * - If the current fd is not a valid handle (i.e. < 0), jump to reconnect
 * - If the writer thread has encountered an I/O error[1], jump to reconnect
 * - If poll() reports that the socket is closed or unusable, jump to reconnect
 * - If a read() call has an error other than EAGAIN, jump to reconnect
 *
 * All possible jumps to the reconnection code happen before a netcmd is
 * removed from its work queue, so commands are not lost on reconnect.
 * Interrupted write and read results are discarded and the command is
 * retransmitted in its entirety after reconnecting.
 *
 * If the writer notices an error it will notify the reader via a pipe and
 * proceed to wait on the socket's setup_cond to avoid busy looping and
 * interfering with the handshake performed in the reconnection process. Once
 * reconnecting has succeeded and handshakes have been exchanged, the reader
 * signals the writer to resume normal operation (sleep on the work queue's
 * condition variable until a new command arrives from the application).
 *
 * [1]: Linux *aggressively* reuses fd numbers, so in order to avoid race
 * conditions, only the reader thread actually reconnects. The writer holds the
 * socket's setup_mutex during writes to ensure it does not get changed between
 * the check whether it has changed fd and the actual write.
 */

/* Functions for manipulating a remote connection */

static ssize_t
connection_read_full (remote_connection_t *connection,
                      void *p,
                      size_t total,
                      remote_server_data_t *sinfo)
{

  size_t readb = 0;
  ssize_t res;
  char *ptr = (char *)(p);
  POCL_ATOMIC_ADD (sinfo->rx_bytes_requested, total);
  while (readb < total)
    {
      size_t remain = total - readb;
      res = read (connection->fd, ptr + readb, remain);
      if (res < 0)
        { /* ERROR */

          /* In the case of these errors, try again. */
          int e = errno;
          if (e == EAGAIN || e == EWOULDBLOCK || e == EINTR)
            continue;
          POCL_MSG_ERR ("error reading remote data: %d (%s).\n", errno,
                        strerror (errno));
          return -1;
        }
      if (res == 0)
        { /* EOF */
          return 0;
        }
      POCL_ATOMIC_ADD (sinfo->rx_bytes_confirmed, (uint64_t)res);
      readb += (size_t)res;
    }

  return (ssize_t)(total);
}

static int
connection_write_full (remote_connection_t *connection,
                       void *p,
                       size_t total,
                       remote_server_data_t *sinfo)
{

  size_t written = 0;
  ssize_t res;
  char *ptr = (char *)(p);
  POCL_ATOMIC_ADD (sinfo->tx_bytes_submitted, total);
  while (written < total)
    {
      size_t remain = total - written;
      res = write (connection->fd, ptr + written, remain);
      if (res < 0)
        {
          int e = errno;
          if (e == EAGAIN || e == EWOULDBLOCK || e == EINTR)
            continue;
          else
            return -1;
        }
      POCL_ATOMIC_ADD (sinfo->tx_bytes_confirmed, (uint64_t)res);
      written += (size_t)res;
    }
  return 0;
}

#define THRESHOLD 1200

static int
connection_writev_full (remote_connection_t *connection,
                        size_t num,
                        void **arys,
                        size_t *sizes,
                        remote_server_data_t *sinfo)
{

  struct iovec *ary = alloca (sizeof (struct iovec) * num);
  size_t total = 0;
  int res = 0;
  unsigned i;

  for (i = 0; i < num; ++i)
    {
      ary[i].iov_base = arys[i];
      ary[i].iov_len = sizes[i];
      total += sizes[i];
    }

  /* TODO there has to be a better way to handle this */
  if (total >= THRESHOLD)
    {

      for (i = 0; i < num; ++i)
        {
          res = connection_write_full (connection, ary[i].iov_base,
                                       ary[i].iov_len, sinfo);
          if (res < 0)
            break;
        }
    }
  else
    {
      POCL_ATOMIC_ADD (sinfo->tx_bytes_submitted, total);
      ssize_t written = writev (connection->fd, ary, num);
      if (written < 0)
        res = -1;
      else
        {
          POCL_ATOMIC_ADD (sinfo->tx_bytes_confirmed, total);
          assert ((size_t)written == total);
        }
    }

  return res;
}

static cl_int
connection_init (remote_connection_t *connection,
                 transport_domain_t domain,
                 int is_fast)
{
  connection->domain = domain;
  connection->fd = -1;
  POCL_INIT_LOCK (connection->setup_guard.mutex);
  POCL_INIT_COND (connection->setup_guard.cond);
  connection->reconnect_count = 0;
  int pipe_pair[2];
  int pipe_res = pipe (pipe_pair);
  connection->is_fast = is_fast;
  connection->reconnect_attempts = 0;

  if (pipe_res != 0)
    POCL_MSG_ERR ("Failed to open socket notification pipe: %s\n",
                  strerror (errno));

  return pipe_res;
}

static cl_int
connection_connect (remote_server_data_t *data,
                    remote_connection_t *connection,
                    unsigned port,
                    int bufsize,
                    ReplyMsg_t *reply_out)
{
  const int32_t one = 1;
  const int32_t zero = 0;
  unsigned addrlen = 0;
  int err = 0;
  remote_connection_t new_connection = *connection;

  struct sockaddr_storage server;
  memset (&server, 0, sizeof (server));
  struct addrinfo *ai = NULL;

  assert (connection->fd == -1);
  assert (connection->domain != TransportDomain_Unset);

  if (connection->domain != TransportDomain_Unix)
    {
      ai = pocl_resolve_address (data->address, port, &err);
      if (err)
        {
          POCL_MSG_ERR ("Failed to resolve address: %s\n", gai_strerror (err));
          return err;
        }
      memcpy (&server, ai->ai_addr, ai->ai_addrlen);
      addrlen = ai->ai_addrlen;
    }

  switch (connection->domain)
    {
    case TransportDomain_Unset:
      return CL_INVALID_DEVICE;

    case TransportDomain_Unix:
      {

        POCL_RETURN_ERROR_ON (
          ((new_connection.fd = socket (AF_UNIX, SOCK_STREAM, 0)) == -1),
          CL_INVALID_DEVICE, "socket() returned errno: %i\n", errno);

        struct sockaddr_un server;
        memset (&server, 0, sizeof (server));
        server.sun_family = AF_UNIX;
        strncpy (server.sun_path, "/tmp/pocl.socket",
                 sizeof (server.sun_path) - 1);
        addrlen = sizeof (server);
        break;
      }

    case TransportDomain_Inet:
      {
        POCL_RETURN_ERROR_ON (
          ((new_connection.fd
            = socket (ai->ai_family, ai->ai_socktype, IPPROTO_TCP))
           == -1),
          CL_INVALID_DEVICE, "socket() returned errno: %i\n", errno);

        break;
      }

    case TransportDomain_Vsock:
      {
#ifdef HAVE_LINUX_VSOCK_H
        POCL_RETURN_ERROR_ON (
          ((new_connection.fd
            = socket (ai->ai_family, ai->ai_socktype,
                      ai->ai_family == AF_VSOCK ? 0 : IPPROTO_TCP))
           == -1),
          CL_INVALID_DEVICE, "socket() returned errno: %i\n", errno);
#else
        return CL_INVALID_DEVICE;
#endif
      }
    }

  pocl_freeaddrinfo (ai);

  err = pocl_remote_client_set_socket_options (
    new_connection.fd, bufsize, connection->is_fast, server.ss_family);
  if (err)
    return err;

  POCL_RETURN_ERROR_ON (
    (connect (new_connection.fd, (struct sockaddr *)&server, addrlen) == -1),
    CL_INVALID_DEVICE, "connect() returned errno: %i\n", errno);

  RequestMsg_t hs;
  ReplyMsg_t hsr;
  memset (&hs, 0, sizeof (RequestMsg_t));
  hs.message_type = MessageType_CreateOrAttachSession;
  hs.m.get_session.peer_id = data->peer_id;
  hs.session = data->session;
  hs.m.get_session.fast_socket = connection->is_fast;
  memcpy (hs.authkey, data->authkey, AUTHKEY_LENGTH);
  ssize_t readb, writeb;
  uint32_t req_len = request_size (hs.message_type);
  writeb = connection_write_full (&new_connection, &req_len, sizeof (req_len),
                                  data);
  assert ((size_t)(writeb) == 0);
  writeb = connection_write_full (&new_connection, &hs, req_len, data);
  assert ((size_t)(writeb) == 0);
  readb = connection_read_full (&new_connection, &hsr, sizeof (hsr), data);
  assert ((size_t)(readb) == sizeof (hsr));
  if (reply_out)
    memcpy (reply_out, &hsr, sizeof (ReplyMsg_t));

  *connection = new_connection;

  return 0;
}

static int
connection_disconnect (remote_connection_t *connection)
{
  int res = 0;

  /* <0 is invalid and 0,1,2 are stdio/stderr */
  if (connection->fd > 2)
    {
      shutdown (connection->fd, SHUT_RDWR);
      close (connection->fd);
      connection->fd = -1;
    }

  return res;
}

static void
connection_release (remote_connection_t *connection)
{
  close (connection->notify_pipe_w);
  close (connection->notify_pipe_w);
  POCL_DESTROY_COND (connection->setup_guard.cond);
  POCL_DESTROY_LOCK (connection->setup_guard.mutex);
}

/** Helper function for reconnecting a socket that has become unusable
 *
 * Shuts down and closes existing socket, if the handle has not been set to -1,
 * then attempts to open a new socket and perform the PoCL client-server
 * handshake.
 *
 * socket_data->setup_mutex is expected to be locked when this function is
 * called. */
static int
pocl_remote_reconnect_socket (remote_server_data_t *remote,
                              remote_connection_t *connection)
{
  if (connection->fd != -1)
    {
      POCL_ATOMIC_INC (remote->threads_awaiting_reconnect);
      POCL_ATOMIC_CAS (&remote->available, CL_TRUE, CL_FALSE);
      connection_disconnect (connection);
    }

  POCL_MSG_PRINT_REMOTE (
    "Attempting to connect to session %" PRIu64 " on %s:%d (%s)\n",
    remote->session, remote->address,
    connection->is_fast ? remote->fast_port : remote->slow_port,
    connection->is_fast ? "fast" : "slow");

  connection->reconnect_attempts += 1;

  int res = connection_connect (
    remote, connection,
    connection->is_fast ? remote->fast_port : remote->slow_port,
    connection->is_fast ? NETWORK_BUF_SIZE_FAST : NETWORK_BUF_SIZE_SLOW, NULL);
  if (res == CL_SUCCESS)
    {
      connection->reconnect_count += 1;
      int waiting = POCL_ATOMIC_DEC (remote->threads_awaiting_reconnect);
      if (waiting == 0)
        POCL_ATOMIC_CAS (&remote->available, CL_FALSE, CL_TRUE);
    }

  return res;
}

/**
 * A server with the same session may get re-discovered after it was previously
 * disconnected. Instead of re-adding the server anew, we re-connect to the
 * previous instance of the server.
 */
cl_int
pocl_remote_reconnect_rediscover (const char *address_with_port)
{
  remote_server_data_t *d = NULL;
  char *temp = strchr (address_with_port, '/');
  char *addr;

  if (temp)
    addr = strndup (address_with_port, (temp - address_with_port));
  else
    addr = strdup (address_with_port);

  size_t len = strlen (addr);

  DL_FOREACH (servers, d)
    {
      if ((strncmp (d->address_with_port, address_with_port, len) == 0)
          && (strlen (d->address_with_port) == len))
        break;
    }

  if (d == NULL)
    {
      free (addr);
      POCL_MSG_ERR ("Could not attempt reconnect. Server corresponding to "
                    "given paramenters not found.\n");
      return -1;
    }

  free (addr);

  POCL_LOCK (d->slow_connection.discovery_reconnect_guard.mutex);
  POCL_BROADCAST_COND (d->slow_connection.discovery_reconnect_guard.cond);
  POCL_UNLOCK (d->slow_connection.discovery_reconnect_guard.mutex);

  POCL_LOCK (d->fast_connection.discovery_reconnect_guard.mutex);
  POCL_BROADCAST_COND (d->fast_connection.discovery_reconnect_guard.cond);
  POCL_UNLOCK (d->fast_connection.discovery_reconnect_guard.mutex);

  return CL_SUCCESS;
}

/*****************************************************************************/

/* Communication thread loops */

/**
 * Helper function to keep reader thread functions more focused. This does some
 * final tweaks to the timestamps. For async commands it also calls the command
 * finished callback and cleans up the netcmd when done.
 */
static void
finish_running_cmd (network_command *running_cmd,
                    network_command_status_t status)
{

  /* When a server is lost while some application is running then the running
   * commands certain fields have to be accordingly modified.
   *
   * TODO:
   * IMPORTANT: To handle losing server in case of migration commands. For
   * migration command from server-1 to server-2, the command is written in
   * the inflight queue of server-2. If, server-1 fails after the client
   * successfully sends the migration command to server-1, we have an
   * undefined state. The client can't know if server-1 failed before or after
   * the migration completed. Even though all incomplete commands in server-1's
   * inflight queue are marked as failed, the migration command is written in
   * server-2's inflight queue. Command in server-2's inflight queue will be
   * expected to be completed at some point and it may or may not get
   * completed. This can cause deadlock and needs to handled. */

  running_cmd->status = status;
  if (status == NETCMD_FAILED)
    {
      running_cmd->reply.message_type = MessageType_Failure;
      running_cmd->reply.failed = 1;
      running_cmd->reply.fail_details = CL_DEVICE_NOT_AVAILABLE;
      running_cmd->reply.data_size = 0;
      running_cmd->reply.obj_id = 0;
      running_cmd->reply.did = running_cmd->request.did;
      running_cmd->reply.pid = running_cmd->request.pid;
    }
  running_cmd->client_read_end_timestamp_ns = pocl_gettimemono_ns ();

  TP_MSG_RECEIVED (running_cmd->reply.msg_id, running_cmd->event_id,
                   running_cmd->reply.client_did, running_cmd->reply.did,
                   running_cmd->reply.message_type, 1);

  if (running_cmd->synchronous)
    {
      POCL_LOCK (running_cmd->data.sync.mutex);
      POCL_SIGNAL_COND (running_cmd->data.sync.cond);
      TP_MSG_RECEIVED (running_cmd->reply.msg_id, running_cmd->event_id,
                       running_cmd->reply.client_did, running_cmd->reply.did,
                       running_cmd->reply.message_type, 2);
      POCL_UNLOCK (running_cmd->data.sync.mutex);
      /* RACE CONDITION ALERT!
       *
       * Every synchronous network_command is allocated on the caller's stack.
       * Signaling the condition and releasing the mutex allows the caller to
       * proceed with its control flow, soon ending the scope that owns the
       * network_command. DO NOT ATTEMPT any access to runnign_cmd from here
       * on.
       */
    }
  else
    {
      /* setup event timestamps */
      cl_event e = running_cmd->data.async.node->sync.event.event;
      cl_command_type type = running_cmd->data.async.node->type;

      uint64_t ocl_in_host_queue = 0, ocl_in_dev_queue = 0, ocl_on_dev = 0;
      if (running_cmd->reply.timing.submitted
          >= running_cmd->reply.timing.queued)
        ocl_in_host_queue = running_cmd->reply.timing.submitted
                            - running_cmd->reply.timing.queued;
      if (running_cmd->reply.timing.started
          >= running_cmd->reply.timing.submitted)
        ocl_in_dev_queue = running_cmd->reply.timing.started
                           - running_cmd->reply.timing.submitted;
      if (running_cmd->reply.timing.completed
          >= running_cmd->reply.timing.started)
        ocl_on_dev = running_cmd->reply.timing.completed
                     - running_cmd->reply.timing.started;

      /* No-op until writer has finished writing timestamps
       * If we get stuck here something has gone wrong */
      uint64_t start, end;
      do
        {
          end = POCL_ATOMIC_LOAD (running_cmd->client_write_end_timestamp_ns);
          start
            = POCL_ATOMIC_LOAD (running_cmd->client_write_start_timestamp_ns);
        }
      while (end <= start);
      /* TODO this compares times of write() syscalls, but that may not be
       * equal to transfer times */
      uint64_t local_writing_ns = end - start;

      assert (running_cmd->reply.server_read_end_timestamp_ns
              >= running_cmd->reply.server_read_start_timestamp_ns);
      uint64_t remote_reading_ns
        = running_cmd->reply.server_read_end_timestamp_ns
          - running_cmd->reply.server_read_start_timestamp_ns;

      /* in theory, local_writing should be +- equal remote reading, select
       * larger */
      uint64_t client_to_remote = remote_reading_ns > local_writing_ns
                                    ? remote_reading_ns
                                    : local_writing_ns;

      /* TODO we don't have the timings for remote writing */
      uint64_t remote_writing_ns = 0;

      /* No-op until reader has finished writing timestamps (should never be
       * necessary) If we get stuck here something has gone wrong */
      end = POCL_ATOMIC_LOAD (running_cmd->client_read_end_timestamp_ns);
      do
        {
          start
            = POCL_ATOMIC_LOAD (running_cmd->client_read_start_timestamp_ns);
        }
      while (end <= start);
      uint64_t local_reading_ns = end - start;

      /* should be +- equal, select larger */
      uint64_t remote_to_client = local_reading_ns > remote_writing_ns
                                    ? local_reading_ns
                                    : remote_writing_ns;

      switch (type)
        {

        case CL_COMMAND_NDRANGE_KERNEL:
        case CL_COMMAND_TASK:
        case CL_COMMAND_NATIVE_KERNEL:
          e->time_queue = running_cmd->reply.server_read_end_timestamp_ns;
          e->time_submit = e->time_queue + ocl_in_host_queue;
          e->time_start = e->time_submit + ocl_in_dev_queue;
          e->time_end = e->time_start + ocl_on_dev;
          break;

        case CL_COMMAND_READ_BUFFER:
        case CL_COMMAND_READ_IMAGE:
        case CL_COMMAND_MAP_BUFFER:
        case CL_COMMAND_MAP_IMAGE:
        case CL_COMMAND_READ_BUFFER_RECT:
          e->time_queue = running_cmd->reply.server_read_end_timestamp_ns;
          e->time_submit = e->time_queue + ocl_in_host_queue;
          e->time_start = e->time_submit + ocl_in_dev_queue;
          e->time_end = e->time_start + ocl_on_dev + remote_to_client;
          break;

        case CL_COMMAND_WRITE_BUFFER:
        case CL_COMMAND_WRITE_IMAGE:
        case CL_COMMAND_UNMAP_MEM_OBJECT:
        case CL_COMMAND_WRITE_BUFFER_RECT:
          e->time_queue = running_cmd->reply.server_read_start_timestamp_ns;
          e->time_submit = e->time_queue + ocl_in_host_queue;
          e->time_start = e->time_submit + ocl_in_dev_queue;
          e->time_end = e->time_start + ocl_on_dev + client_to_remote;
          break;

        default:
          break;
        }

#ifdef ENABLE_RDMA
      if (running_cmd->rdma_region)
        {
          rdma_unregister_mem_region (running_cmd->rdma_region);
        }
#endif
      running_cmd->data.async.cb (running_cmd->data.async.arg,
                                  running_cmd->data.async.node,
                                  running_cmd->reply.data_size);
      TP_MSG_RECEIVED (running_cmd->reply.msg_id, running_cmd->event_id,
                       running_cmd->reply.client_did, running_cmd->reply.did,
                       running_cmd->reply.message_type, 2);

      void *p = NULL;
      switch (type)
        {
        case CL_COMMAND_NDRANGE_KERNEL:
        case CL_COMMAND_TASK:
        case CL_COMMAND_NATIVE_KERNEL:
          p = (void *)running_cmd->req_extra_data;
          POCL_MEM_FREE (p);
          p = (void *)running_cmd->req_extra_data2;
          POCL_MEM_FREE (p);
          break;
        default:
          break;
        }

      POCL_MEM_FREE (running_cmd->req_wait_list);
      POCL_MEM_FREE (running_cmd);
    }
}

/** Main function for the socket reader thread */
static void *
pocl_remote_reader_pthread (void *aa)
{
  network_queue_arg *a = aa;
  remote_server_data_t *remote = a->remote;
  network_queue *inflight = a->in_flight;
  network_queue *this = a->ours;
  remote_connection_t *connection = a->connection;
  POCL_MEM_FREE (a);

  ssize_t readb;
  struct pollfd pfd[2];
  pfd[0].events = POLLIN;
  pfd[1].events = POLLIN;
  pfd[1].fd = connection->notify_pipe_r;
  int nevs;
  unsigned writer_reconnects;
  unsigned reader_reconnects;
#ifdef SIGPIPE
  /* Don't kill the thread on I/O errors */
  POCL_IGNORE_SIGNAL_IN_THREAD (SIGPIPE);
#endif

  while (!this->exit_requested)
    {
      POCL_LOCK (connection->setup_guard.mutex);
      int fd = connection->fd;
      reader_reconnects = connection->reconnect_count;
      POCL_UNLOCK (connection->setup_guard.mutex);
      if (fd < 0)
        {
          POCL_LOCK (connection->setup_guard.mutex);
          int reconnected;
        TRY_RECONNECT:

          reconnected = pocl_remote_reconnect_socket (remote, connection);
          if (reconnected != CL_SUCCESS)
            {
              if (connection->reconnect_attempts
                  >= POCL_REMOTE_RECONNECT_MAX_ATTEMPTS)
                {
                  network_command *cmd = NULL, *tmp = NULL;
                  POCL_LOCK (inflight->mutex);
                  /* Each command in the inflight queue of the failed server
                   * has to be handled and marked as failed to prevent
                   * deadlock. */
                  DL_FOREACH_SAFE (inflight->queue, cmd, tmp)
                    {
                      DL_DELETE (inflight->queue, cmd);
                      finish_running_cmd (cmd, NETCMD_FAILED);
                    }
                  POCL_UNLOCK (inflight->mutex);

#if defined(ENABLE_REMOTE_DISCOVERY_AVAHI)                                    \
  || defined(ENABLE_REMOTE_DISCOVERY_DHT)                                     \
  || defined(ENABLE_REMOTE_DISCOVERY_ANDROID)
                  POCL_LOCK (connection->discovery_reconnect_guard.mutex);
                  POCL_WAIT_COND (connection->discovery_reconnect_guard.cond,
                                  connection->discovery_reconnect_guard.mutex);
                  POCL_UNLOCK (connection->discovery_reconnect_guard.mutex);
#endif
                }
              goto TRY_RECONNECT;
            }
          else
            {
              fd = connection->fd;
              reader_reconnects = connection->reconnect_count;
              connection->reconnect_attempts = 0;
              POCL_BROADCAST_COND (connection->setup_guard.cond);
              POCL_UNLOCK (connection->setup_guard.mutex);
            }
        }

      /* Block until there is something to read. This is especially needed to
       * get accurate profiling timestamps for read commands */
      pfd[0].fd = fd;
      nevs = poll (pfd, 2, -1);
      if (nevs < 1)
          continue;
      else
        {
          if (pfd[1].revents & POLLIN)
            {
              /* If woken up by the writer pipe, reconnect iff the writer's
               * reconnect count is still up to date (writer noticed an I/O
               * error before poll did). If the writer's reconnect count is
               * behind, this was a stale notification and no further action is
               * needed. In that case proceed to checking events of the socket
               * fd. */
              read (pfd[1].fd, &writer_reconnects, sizeof (writer_reconnects));
              if (writer_reconnects == reader_reconnects)
                goto TRY_RECONNECT;
            }
          if (pfd[0].revents & POLLFD_ERROR_BITS)
            goto TRY_RECONNECT;
          if (!(pfd[0].revents & POLLIN))
            continue;
        }

      /* READ MSG */
      ReplyMsg_t rep;
      uint64_t start_ts = pocl_gettimemono_ns ();
      readb
        = connection_read_full (connection, &rep, sizeof (ReplyMsg_t), remote);
      CHECK_READ_TIMEOUT (readb); /* TODO: continue instead of abort */

      /* we have a message */
      assert ((size_t)readb == sizeof (ReplyMsg_t));
      POCL_MSG_PRINT_REMOTE (
        "READER THR: MESSAGE READ, TYPE:  %u  ID: %zu  SIZE: %zu\n",
        rep.message_type, rep.msg_id, readb);

      /* find it */
      network_command *running_cmd = NULL;
      POCL_LOCK (inflight->mutex);
      DL_FOREACH (inflight->queue, running_cmd)
        {
          if (running_cmd->request.msg_id == rep.msg_id)
            {
              break;
            }
        }
      if (!running_cmd)
        {
          /* Not found in queue. This can happen when the remote resends old
           * replies after reconnecting to make sure none got lost on the way
           */
          POCL_UNLOCK (inflight->mutex);
          continue;
        }
      POCL_ATOMIC_STORE (running_cmd->client_read_start_timestamp_ns,
                         start_ts);
      assert (running_cmd->request.msg_id == rep.msg_id);
      POCL_UNLOCK (inflight->mutex);
      memcpy (&running_cmd->reply, &rep, sizeof (ReplyMsg_t));

      TP_MSG_RECEIVED (running_cmd->reply.msg_id, running_cmd->event_id,
                       running_cmd->reply.client_did, running_cmd->reply.did,
                       running_cmd->reply.message_type, 0);

      assert (running_cmd->status == NETCMD_WRITTEN);
      running_cmd->status = NETCMD_READ;

      /* READ EXTRA DATA */
      if (running_cmd->reply.data_size > 0)
        {
          if (running_cmd->reply.strings_size > 0)
            {
              /* Allocate the memory for the reply here since only now we
                 know the string pool's size. */
              assert (running_cmd->rep_extra_data == NULL);
              running_cmd->rep_extra_data
                = (char *)malloc (running_cmd->reply.data_size);
              running_cmd->strings
                = running_cmd->rep_extra_data + running_cmd->rep_extra_size;
            }
          running_cmd->rep_extra_size = running_cmd->reply.data_size;
          readb
            = connection_read_full (connection, running_cmd->rep_extra_data,
                                    running_cmd->reply.data_size, remote);
          CHECK_READ (readb);
        }
      POCL_LOCK (inflight->mutex);
      DL_DELETE (inflight->queue, running_cmd);
      POCL_UNLOCK (inflight->mutex);
      finish_running_cmd (running_cmd, NETCMD_FINISHED);
    }
  return NULL;
}

#ifdef ENABLE_RDMA
static void *
pocl_remote_rdma_reader_pthread (void *aa)
{
  network_queue_arg *a = aa;
  remote_server_data_t *remote = a->remote;
  network_queue *this = a->ours;
  rdma_data_t *rdma_data = &remote->rdma_data;
  POCL_MEM_FREE (a);

  POCL_LOCK (this->mutex);
  while (!this->exit_requested)
    {
      POCL_UNLOCK (this->mutex);

      /* Block until an RDMA (receive) completion occurs */
      struct ibv_cq *comp_queue;
      void *context;
      if (ibv_get_cq_event (rdma_data->comp_channel_in, &comp_queue, &context))
        {
          POCL_MSG_ERR ("Get RDMA channel event failed");
        }

      /* Receive operations should go into the in-queue */
      assert (comp_queue == rdma_data->cq_in);

      struct ibv_wc wc = {};
      if (ibv_poll_cq (comp_queue, 1, &wc) != 1)
        {
          POCL_MSG_ERR ("Poll RDMA event queue failed");
        }

      uint64_t start_ts = pocl_gettimemono_ns ();
      ibv_ack_cq_events (comp_queue, 1);

      if (wc.status != IBV_WC_SUCCESS)
        {
          const char *status_msg = ibv_wc_status_str (wc.status);
          POCL_MSG_ERR ("RDMA receive request failed - unrecoverable error");
          /* TODO: Some failures here could be recoverable */
        }

      POCL_MSG_PRINT_REMOTE ("RDMA RECEIVE: ID: %lu \n", wc.wr_id);

      /* FIND THE RELEVANT COMMAND FROM QUEUE */

      network_command *cmd = NULL;

      /* Loop until a command matching the work completion id is found */
      POCL_LOCK (this->mutex);
      while (1)
        {

          /* TODO: Inefficiently checking the entire queue for every loop */
          DL_FOREACH (this->queue, cmd)
          {
            if (cmd->request.msg_id == wc.wr_id)
              {
                break;
              }
          }
          POCL_UNLOCK (this->mutex);

          if (cmd)
            {
              break;
            }

          /* WAIT FOR NEW COMMANDS */

          POCL_LOCK (this->mutex);
          /* wait for main reader to pass a new command */
          POCL_WAIT_COND (this->cond, this->mutex);
        }

      /* UPDATE COMMAND EXTRA DATA */

      uint32_t write_size = wc.byte_len;

      assert (cmd->rep_extra_size == write_size);

      /* REMOVE COMMAND FROM QUEUE */
      POCL_ATOMIC_STORE (cmd->client_read_start_timestamp_ns, start_ts);

      POCL_LOCK (this->mutex);
      DL_DELETE (this->queue, cmd);
      POCL_UNLOCK (this->mutex);

      finish_running_cmd (cmd, NETCMD_FINISHED);

      POCL_LOCK (this->mutex);
    }

  return NULL;
}

static void *
pocl_remote_rdma_writer_pthread (void *aa)
{
  network_queue_arg *a = aa;
  remote_server_data_t *remote = a->remote;
  network_queue *this = a->ours;
  rdma_data_t *rdma_data = &remote->rdma_data;
  POCL_MEM_FREE (a);

  network_command *cmd;
  RequestMsg_t request;
  struct ibv_mr *request_mr = rdma_register_mem_region (
      rdma_data, (void *)&request, sizeof (request));
  POCL_LOCK (this->mutex);
  while (!this->exit_requested)
    {
      cmd = this->queue;
      if (cmd)
        {
          POCL_UNLOCK (this->mutex);
          memcpy (&request, &cmd->request, sizeof (RequestMsg_t));

          struct ibv_sge req_sge;
          memset (&req_sge, 0, sizeof (req_sge));
          req_sge.addr = (uintptr_t)request_mr->addr;
          req_sge.length = request_size (cmd->request.message_type);
          req_sge.lkey = request_mr->lkey;

          struct ibv_send_wr cmd_wr;
          memset (&cmd_wr, 0, sizeof (cmd_wr));
          cmd_wr.wr_id = cmd->request.msg_id;
          cmd_wr.sg_list = &req_sge;
          cmd_wr.num_sge = 1;
          cmd_wr.opcode = IBV_WR_SEND;
          cmd_wr.send_flags = IBV_SEND_SIGNALED;

          uint32_t mem_id = cmd->request.obj_id;
          rdma_buffer_info_t *s;
          HASH_FIND (hh, remote->rdma_keys, &mem_id, sizeof (uint32_t), s);

          /* XXX: registering and deregistering memory regions in a tight loop
           * should be avoided */
          struct ibv_mr *mem_region = rdma_register_mem_region (
              rdma_data, (void *)cmd->req_extra_data, cmd->req_extra_size);

          struct ibv_sge sge;
          memset (&sge, 0, sizeof (sge));
          sge.addr = (uint64_t)mem_region->addr;
          sge.length = cmd->req_extra_size;
          sge.lkey = mem_region->lkey;

          struct ibv_send_wr write_wr = {};
          memset (&write_wr, 0, sizeof (write_wr));
          write_wr.next = &cmd_wr;
          write_wr.sg_list = &sge;
          write_wr.num_sge = 1;
          write_wr.opcode = IBV_WR_RDMA_WRITE;

          /* TODO: we probably don't actually need this */
          uint64_t offset = 0;

          write_wr.wr.rdma.rkey = s->remote_rkey;
          write_wr.wr.rdma.remote_addr = s->remote_vaddr + offset;

          struct ibv_wc wc = {};

          POCL_MSG_PRINT_REMOTE ("RDMA WRITE: ID: %lu, SIZE: %lu\n",
                                 cmd->request.msg_id, cmd->req_extra_size);

          int attempts = 10;
          while (attempts > 0)
            {
              POCL_LOCK (cmd->receiver->mutex);
              DL_APPEND (cmd->receiver->queue, cmd);
              POCL_UNLOCK (cmd->receiver->mutex);

              POCL_ATOMIC_STORE (cmd->client_write_start_timestamp_ns,
                                 pocl_gettimemono_ns ());
              struct ibv_send_wr *bad_send_wr;
              if (ibv_post_send (rdma_data->cm_id->qp, &write_wr,
                                 &bad_send_wr))
                {
                  POCL_MSG_ERR ("Post RDMA send request failed\n");
                }

              struct ibv_cq *comp_queue;
              void *context;
              if (ibv_get_cq_event (rdma_data->comp_channel_out, &comp_queue,
                                    &context))
                {
                  POCL_MSG_ERR ("Get RDMA channel event failed\n");
                }

              /* Send operations should go into the out-queue */
              assert (comp_queue == rdma_data->cq_out);

              if (ibv_poll_cq (comp_queue, 1, &wc) != 1)
                {
                  POCL_MSG_ERR ("Poll RDMA event queue failed\n");
                }

              ibv_ack_cq_events (comp_queue, 1);

              switch (wc.status)
                {
                case IBV_WC_SUCCESS:
                  attempts = -1;
                  break;
                case IBV_WC_RNR_RETRY_EXC_ERR:
                  POCL_MSG_WARN (
                      "RDMA send request failed - receiver not ready\n");
                  attempts--;
                  break;
                default:
                  POCL_MSG_ERR (
                      "RDMA send request failed - unrecoverable error %i\n",
                      wc.status);
                  break;
                  /* TODO: Some other failures here could be recoverable */
                }

              POCL_ATOMIC_STORE (cmd->client_write_end_timestamp_ns,
                                 pocl_gettimemono_ns ());
            }

          if (attempts == 0)
            {
              POCL_MSG_ERR ("RDMA send request failed - max. number of "
                            "attempts exceeded\n");
            }

          rdma_unregister_mem_region (mem_region);

          /* Hand command over to reply receiver thread */

          POCL_LOCK (this->mutex);
          DL_DELETE (this->queue, cmd);
        }
      else
        {
          /* no cmds, wait for one to arrive */
          POCL_WAIT_COND (this->cond, this->mutex);
        }
    }
  POCL_UNLOCK (this->mutex);
  rdma_unregister_mem_region (request_mr);

  return NULL;
}
#endif

static void *
pocl_remote_writer_pthread (void *aa)
{
  network_queue_arg *a = aa;
  network_queue *this = a->ours;
  network_queue *inflight = a->in_flight;
  remote_server_data_t *remote = a->remote;
  remote_connection_t *connection = a->connection;
  POCL_MEM_FREE (a);
  unsigned reconnect_count = 0;
#ifdef SIGPIPE
  /* Don't kill the thread on I/O errors */
  POCL_IGNORE_SIGNAL_IN_THREAD (SIGPIPE);
#endif
  POCL_LOCK (connection->setup_guard.mutex);
  reconnect_count = connection->reconnect_count;
  POCL_UNLOCK (connection->setup_guard.mutex);

  network_command *cmd;
  POCL_LOCK (this->mutex);
  while (!this->exit_requested)
    {
      cmd = this->queue;
      if (cmd)
        {
          DL_DELETE (this->queue, cmd);
          POCL_UNLOCK (this->mutex);

          if (POCL_ATOMIC_LOAD (cmd->client_write_start_timestamp_ns) == 0)
            POCL_ATOMIC_STORE (cmd->client_write_start_timestamp_ns,
                               pocl_gettimemono_ns ());

          assert (cmd->status == NETCMD_STARTED);

          uint32_t msg_size = request_size (cmd->request.message_type);

          POCL_MSG_PRINT_REMOTE ("WRITER THR: WRITING MSG, TYPE: %u  ID: %zu  "
                                 "EVENT: %zu  SIZE: msg_size: %u + waitlist: "
                                 "%zu + extra: %zu + extra2: %zu\n",
                                 cmd->request.message_type,
                                 cmd->request.msg_id, cmd->event_id, msg_size,
                                 cmd->req_waitlist_size * sizeof (uint64_t),
                                 cmd->req_extra_size, cmd->req_extra_size2);

          cmd->request.waitlist_size = cmd->req_waitlist_size;
          if (cmd->synchronous)
            {
              POCL_LOCK (cmd->data.sync.mutex);
              cmd->status = NETCMD_WRITTEN;
              POCL_UNLOCK (cmd->data.sync.mutex);
            }
          else
            cmd->status = NETCMD_WRITTEN;

          TP_MSG_SENT (cmd->request.msg_id, cmd->event_id,
                       cmd->request.client_did, cmd->request.did,
                       cmd->request.message_type, 0);

          POCL_LOCK (cmd->receiver->mutex);
          DL_APPEND (cmd->receiver->queue, cmd);
          POCL_UNLOCK (cmd->receiver->mutex);

          POCL_LOCK (connection->setup_guard.mutex);

          if (0)
            {
            /* This is only hit if there is an error from CHECK_WRITE */
            TRY_RECONNECT:
              /* Only sleep if the reader thread has *not* reconnected yet */
              if (reconnect_count == connection->reconnect_count)
                {
                  POCL_MSG_PRINT_REMOTE (
                    "(%s) writer waiting for reader to reconnect\n",
                    connection->is_fast ? "fast" : "slow");
                  /* In synthetic benchmarks poll() would sometimes not notice
                   * the socket getting closed from under it, leading to
                   * deadlocks (poll has no time limit). To avoid this, there
                   * is now a pipe that is part of the poll so the writer can
                   * force the reader to wake up. */
                  write (connection->notify_pipe_w, &reconnect_count,
                         sizeof (reconnect_count));
                  POCL_WAIT_COND (connection->setup_guard.cond,
                                  connection->setup_guard.mutex);
                }
            }
          reconnect_count = connection->reconnect_count;

          /* WRITE DATA */
          if (cmd->req_extra_data2)
            {
              void *ptrs[5]
                  = { &msg_size, &cmd->request, (void *)cmd->req_wait_list,
                      (void *)cmd->req_extra_data,
                      (void *)cmd->req_extra_data2 };
              size_t sizes[5] = { sizeof (uint32_t), msg_size,
                                  cmd->req_waitlist_size * sizeof (uint64_t),
                                  cmd->req_extra_size, cmd->req_extra_size2 };
              CHECK_WRITE (
                connection_writev_full (connection, 5, ptrs, sizes, remote));
            }
          else if (cmd->req_extra_data)
            {

              void *ed = (void *)cmd->req_extra_data;
              size_t eds = cmd->req_extra_size;

              void *ptrs[4] = { &msg_size, &cmd->request,
                                (void *)cmd->req_wait_list, ed };
              size_t sizes[4]
                  = { sizeof (uint32_t), msg_size,
                      cmd->req_waitlist_size * sizeof (uint64_t), eds };
              CHECK_WRITE (
                connection_writev_full (connection, 4, ptrs, sizes, remote));
            }
          else if (cmd->req_waitlist_size > 0)
            {
              void *ptrs[3]
                  = { &msg_size, &cmd->request, (void *)cmd->req_wait_list };
              size_t sizes[3] = { sizeof (uint32_t), msg_size,
                                  cmd->req_waitlist_size * sizeof (uint64_t) };
              CHECK_WRITE (
                connection_writev_full (connection, 3, ptrs, sizes, remote));
            }
          else
            {
              assert (cmd->req_waitlist_size == 0);
              assert (cmd->req_wait_list == NULL);
              CHECK_WRITE (connection_write_full (connection, &msg_size,
                                                  sizeof (uint32_t), remote));
              CHECK_WRITE (connection_write_full (connection, &cmd->request,
                                                  msg_size, remote));
            }

          POCL_UNLOCK (connection->setup_guard.mutex);

          POCL_ATOMIC_STORE (cmd->client_write_end_timestamp_ns,
                             pocl_gettimemono_ns ());

          TP_MSG_SENT (cmd->request.msg_id, cmd->event_id,
                       cmd->request.client_did, cmd->request.did,
                       cmd->request.message_type, 1);

          POCL_LOCK (this->mutex);
        }
      else
        {
          POCL_WAIT_COND (this->cond, this->mutex);
        }
    }

  POCL_UNLOCK (this->mutex);

  return NULL;
}

static void
wait_on_netcmd (network_command *n)
{
  POCL_LOCK (n->data.sync.mutex);
  while (n->status < NETCMD_FINISHED)
    POCL_WAIT_COND (n->data.sync.cond, n->data.sync.mutex);
  POCL_UNLOCK (n->data.sync.mutex);
}

/*****************************************************************************/
/*****************************************************************************/

static void *
traffic_monitor_pthread (void *arg)
{
  network_queue_arg *a = (network_queue_arg *)arg;
  remote_server_data_t *server = a->remote;
  network_queue *q = a->ours;

  const char *output_dir = pocl_get_string_option ("POCL_TRAFFIC_LOG_DIR", 0);
  if (!output_dir)
    goto EXIT;

  char *servname = strdup (server->address_with_port);
  int servname_len = strlen (servname);
  for (int i = 0; i < servname_len; ++i)
    {
      if (servname[i] == ':')
        servname[i] = '_';
    }
  const char *ext = ".csv";
  size_t path_len = strlen (output_dir) + servname_len + strlen (ext) + 1;
  char *file_path = calloc (path_len, sizeof (char));
  snprintf (file_path, path_len, "%s/%s%s", output_dir, servname, ext);

  FILE *f = fopen (file_path, "w");
  struct timespec now;
  uint64_t rx_bytes_requested;
  uint64_t rx_bytes_confirmed;
  uint64_t tx_bytes_submitted;
  uint64_t tx_bytes_confirmed;

  POCL_LOCK (q->mutex);
  while (1)
    {
      if (q->exit_requested)
        break;
      POCL_UNLOCK (q->mutex);

      /*TODO*/
      clock_gettime (CLOCK_REALTIME, &now);
      rx_bytes_requested = POCL_ATOMIC_LOAD (server->rx_bytes_requested);
      rx_bytes_confirmed = POCL_ATOMIC_LOAD (server->rx_bytes_confirmed);
      tx_bytes_submitted = POCL_ATOMIC_LOAD (server->tx_bytes_submitted);
      tx_bytes_confirmed = POCL_ATOMIC_LOAD (server->tx_bytes_confirmed);
      fprintf (f, "%jd,%ld,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
               now.tv_sec, now.tv_nsec, rx_bytes_requested, rx_bytes_confirmed,
               tx_bytes_submitted, tx_bytes_confirmed);
      fflush (f);

      POCL_LOCK (q->mutex);
      if (!q->exit_requested)
        POCL_TIMEDWAIT_COND (q->cond, q->mutex, 10000); /* 10ms */
    }
  POCL_UNLOCK (q->mutex);

  fclose (f);
  free (servname);
  free (file_path);

EXIT:
  free (a);
  return NULL;
}

void
pocl_remote_get_traffic_stats (uint64_t *out_buf, cl_device_id device)
{
  remote_device_data_t *device_data = (remote_device_data_t *)device->data;
  remote_server_data_t *server = device_data->server;
  struct timespec now;
  clock_gettime (CLOCK_REALTIME, &now);
  out_buf[0] = now.tv_sec;
  out_buf[1] = now.tv_nsec;
  out_buf[2] = POCL_ATOMIC_LOAD (server->rx_bytes_requested);
  out_buf[3] = POCL_ATOMIC_LOAD (server->rx_bytes_confirmed);
  out_buf[4] = POCL_ATOMIC_LOAD (server->tx_bytes_submitted);
  out_buf[5] = POCL_ATOMIC_LOAD (server->tx_bytes_confirmed);
}

/**
 * Start all threads needed for the given server connection
 */
static void
start_engines (remote_server_data_t *d, remote_device_data_t *devd,
               cl_device_id device)
{
  /* start alll background threads for IO */
  network_queue_arg *a;

  d->inflight_queue = calloc (1, sizeof (network_queue));
  SETUP_NETW_Q (d->inflight_queue);

  d->traffic_monitor = calloc (1, sizeof (network_queue));
  SETUP_NETW_Q (d->traffic_monitor);
  SETUP_NETW_Q_ARG (a, d, d->traffic_monitor, NULL);
  POCL_CREATE_THREAD (d->traffic_monitor->thread_id, traffic_monitor_pthread,
                      a);

#ifdef ENABLE_RDMA
  d->rdma_read_queue = calloc (1, sizeof (network_queue));
  d->rdma_write_queue = calloc (1, sizeof (network_queue));
  SETUP_NETW_Q (d->rdma_read_queue);
  SETUP_NETW_Q (d->rdma_write_queue);
#endif

  d->slow_read_queue = calloc (1, sizeof (network_queue));
  SETUP_NETW_Q (d->slow_read_queue);
  SETUP_NETW_Q_ARG (a, d, d->slow_read_queue, &d->slow_connection);
  POCL_CREATE_THREAD (d->slow_read_queue->thread_id,
                      pocl_remote_reader_pthread, a);

  d->fast_read_queue = calloc (1, sizeof (network_queue));
  SETUP_NETW_Q (d->fast_read_queue);
  SETUP_NETW_Q_ARG (a, d, d->fast_read_queue, &d->fast_connection);
  POCL_CREATE_THREAD (d->fast_read_queue->thread_id,
                      pocl_remote_reader_pthread, a);

  d->slow_write_queue = calloc (1, sizeof (network_queue));
  SETUP_NETW_Q (d->slow_write_queue);
  SETUP_NETW_Q_ARG (a, d, d->slow_write_queue, &d->slow_connection);
  POCL_CREATE_THREAD (d->slow_write_queue->thread_id,
                      pocl_remote_writer_pthread, a);

  d->fast_write_queue = calloc (1, sizeof (network_queue));
  SETUP_NETW_Q (d->fast_write_queue);
  SETUP_NETW_Q_ARG (a, d, d->fast_write_queue, &d->fast_connection);
  POCL_CREATE_THREAD (d->fast_write_queue->thread_id,
                      pocl_remote_writer_pthread, a);

#ifdef ENABLE_RDMA
  if (d->use_rdma)
    {
      /* rdma thread for reader */
      SETUP_NETW_Q_ARG (a, d, d->rdma_read_queue, NULL);
      POCL_CREATE_THREAD (d->rdma_read_queue->thread_id,
                          pocl_remote_rdma_reader_pthread, a);

      /* rdma thread for writer */
      SETUP_NETW_Q_ARG (a, d, d->rdma_write_queue, NULL);
      POCL_CREATE_THREAD (d->rdma_write_queue->thread_id,
                          pocl_remote_rdma_writer_pthread, a);
    }
#endif
}

static void
stop_engines (remote_server_data_t *d)
{
  /* Inform the server that it's time to go */
  remote_server_data_t *data = d;
  remote_server_data_t *ddata = d;
  CREATE_SYNC_NETCMD;
  RequestMsg_t *req = &netcmd->request;
  memset (req, 0, sizeof (RequestMsg_t));
  req->message_type = MessageType_Shutdown;
  req->msg_id = POCL_ATOMIC_INC (last_message_id);
  req->event_id = (uint64_t)(-1);
  req->obj_id = (uint64_t)(-1);
  SEND_REQ_FAST;
  wait_on_netcmd (netcmd);

  /* stop threads and wait for them */
#define NOTIFY_SHUTDOWN(queue)                                                \
  POCL_LOCK ((queue)->mutex);                                                 \
  (queue)->exit_requested = 1;                                                \
  POCL_SIGNAL_COND ((queue)->cond);                                           \
  POCL_UNLOCK ((queue)->mutex);

  NOTIFY_SHUTDOWN (d->fast_read_queue);
  NOTIFY_SHUTDOWN (d->slow_read_queue);
  NOTIFY_SHUTDOWN (d->fast_write_queue);
  NOTIFY_SHUTDOWN (d->slow_write_queue);

#ifdef ENABLE_RDMA
  NOTIFY_SHUTDOWN (d->rdma_read_queue);
  NOTIFY_SHUTDOWN (d->rdma_write_queue);
#endif

  POCL_JOIN_THREAD (d->fast_write_queue->thread_id);
  POCL_JOIN_THREAD (d->slow_write_queue->thread_id);
  POCL_JOIN_THREAD (d->fast_read_queue->thread_id);
  POCL_JOIN_THREAD (d->slow_read_queue->thread_id);

#ifdef ENABLE_RDMA
  POCL_JOIN_THREAD (d->rdma_read_queue->thread_id);
  POCL_JOIN_THREAD (d->rdma_write_queue->thread_id);
#endif

  NOTIFY_SHUTDOWN (d->traffic_monitor);
  POCL_JOIN_THREAD (d->traffic_monitor->thread_id);

#undef NOTIFY_SHUTDOWN
}

static remote_server_data_t *
find_or_create_server (const char *address_with_port, unsigned port,
                       remote_device_data_t *ddata, cl_device_id device,
                       const char *const parameters)
{
  size_t l = strlen (address_with_port);

  remote_server_data_t *rsd = NULL;
  DL_FOREACH (servers, rsd)
  {
    if ((strncmp (rsd->address_with_port, address_with_port, l) == 0)
        && (strlen (rsd->address_with_port) == l))
      {
        /* A server is identified by its IP and port. However, the same IP and
         * port may represent different sessions over time. The variable
         * 'available' is set to 'CL_FALSE' when a connection to a server is
         * lost. If a server is found with an IP and port that matches an entry
         * in the list, it indicates a previously connected server that was
         * disconnected. This check ensures that servers which were marked
         * unavailable are correctly handled. When a server joins with a new
         * session via discovery, it should be treated as a new server, as the
         * removal of 'old' servers is not supported.
         */
        if (!rsd->available)
          continue;

        ++rsd->refcount;
        return rsd;
      }
  }

  /* new server */
  remote_server_data_t *d = calloc (1, sizeof (remote_server_data_t));
  d->refcount = 1;
  d->peer_id = POCL_ATOMIC_INC (last_peer_id);
  d->available = CL_TRUE;
  d->threads_awaiting_reconnect = 0;

  connection_init (&d->fast_connection, TransportDomain_Unset, 1);
  connection_init (&d->slow_connection, TransportDomain_Unset, 0);

  /* pocl_network_init_device ensures address_with_port actually contains
   * port */
  if (address_with_port[0] == '/')
    {
      d->fast_connection.domain = TransportDomain_Unix;
      d->slow_connection.domain = TransportDomain_Unix;
    }
  else if (strncmp (address_with_port, "vsock:", strlen ("vsock:")) == 0)
    {
      strncpy (d->address, address_with_port,
               strrchr (address_with_port, ':') - address_with_port);
      d->fast_connection.domain = TransportDomain_Vsock;
      d->slow_connection.domain = TransportDomain_Vsock;
    }
  else
    {
      strncpy (d->address, address_with_port,
               strchr (address_with_port, ':') - address_with_port);
      d->fast_connection.domain = TransportDomain_Inet;
      d->slow_connection.domain = TransportDomain_Inet;
    }

  char *tmp2 = strdup (parameters);
  char *peer_address = strtok (tmp2, "#");
  if (peer_address == NULL)
    {
      strcpy (d->peer_address, d->address);
    }
  else
    {
      if (strlen (peer_address) < strlen (parameters))
        strcpy (d->peer_address, tmp2 + strlen (peer_address) + 1);
      else
        strcpy (d->peer_address, d->address);
    }
  POCL_MEM_FREE (tmp2);

  strncpy (d->address_with_port, address_with_port, MAX_ADDRESS_PORT_SIZE);
  d->address_with_port[MAX_ADDRESS_PORT_SIZE - 1] = 0;
  POCL_MSG_PRINT_REMOTE ("using host %s with port %u\n", d->address, port);

  d->fast_port = port;
  d->slow_port = port + 1;

#ifdef ENABLE_RDMA
  /* TODO: re-enable once client RDMA has been reworked to match server
   * communication */
  if (CL_TRUE || rdma_init_id (&d->rdma_data) == 0)
    {
      /* hs.m.get_session.use_rdma = 0; */
    }
  else
    {
      POCL_MSG_ERR ("Could not create RDMAcm event channel and id, continuing"
                    "without RDMA\n");
    }
#endif

  ReplyMsg_t hsr;
  if (connection_connect (d, &d->fast_connection, d->fast_port,
                          NETWORK_BUF_SIZE_FAST, &hsr))
    {
      POCL_MSG_ERR ("Could not connect to server\n");
      POCL_MEM_FREE (d);
      return NULL;
    }

  memcpy (d->authkey, hsr.m.get_session.authkey, AUTHKEY_LENGTH);
  d->session = hsr.m.get_session.session;

  {
    char tmp[2 * AUTHKEY_LENGTH + 1] = "................................";
    for (uint8_t *b = d->authkey, i = 0; i < AUTHKEY_LENGTH; ++i)
      snprintf (tmp + (2 * i), 3, "%02x", *b++);
    POCL_MSG_PRINT_REMOTE ("Received session id %" PRIu64 ", key %s\n",
                           hsr.m.get_session.session, tmp);
  }

  d->peer_port = hsr.m.get_session.peer_port;

  if (connection_connect (d, &d->slow_connection, d->slow_port,
                          NETWORK_BUF_SIZE_SLOW, NULL))
    {
      POCL_MSG_ERR ("Could not connect to server\n");
      POCL_MEM_FREE (d);
      return NULL;
    }

  DL_APPEND (servers, d);

#ifdef ENABLE_RDMA
  d->use_rdma = 0; /* hsr.m.create_session.rdma_supported; */
  if (d->use_rdma)
    {
      /* TODO: RDMA connect could be moved to its own function */

      /* RDMA ADDRESS RESOLVE */

      const int timeout_ms = 5000;

      int error;

      int n;
      struct addrinfo *resolv_addr;
      struct addrinfo hints = {};
      hints.ai_family = AF_INET;
      hints.ai_socktype = SOCK_STREAM;

      POCL_MSG_PRINT_REMOTE ("Using RDMAcm port %u\n", d->peer_port + 1);

      char rdma_port_str[6];
      sprintf (rdma_port_str, "%u", d->peer_port + 2);

      n = getaddrinfo (d->address, rdma_port_str, &hints, &resolv_addr);
      if (n < 0)
        {
          /* TODO: Return an error */
          assert (n >= 0);
          return NULL;
        }
      error = rdma_resolve_addr (d->rdma_data.cm_id, NULL,
                                 resolv_addr->ai_addr, timeout_ms);
      if (error)
        {
          /* TODO: Return an error */
          assert (!error);
          return NULL;
        }

      struct rdma_cm_event *event;
      error = rdma_get_cm_event (d->rdma_data.cm_channel, &event);
      if (error)
        {
          /* TODO: Return an error */
          assert (!error);
          return NULL;
        }
      if (event->event != RDMA_CM_EVENT_ADDR_RESOLVED)
        {
          /* TODO: Return an error */
          assert (event->event == RDMA_CM_EVENT_ADDR_RESOLVED);
          return NULL;
        }
      rdma_ack_cm_event (event);

      /* RDMA ROUTE RESOLVE */
      /* This fills out our ibv context i.e. cm_id->verbs */
      error = rdma_resolve_route (d->rdma_data.cm_id, timeout_ms);
      if (error)
        {
          /* TODO: Return an error */
          assert (!error);
          return NULL;
        }

      error = rdma_get_cm_event (d->rdma_data.cm_channel, &event);
      if (error)
        {
          /* TODO: Return an error */
          assert (!error);
          return NULL;
        }
      if (event->event != RDMA_CM_EVENT_ROUTE_RESOLVED)
        {
          /* TODO: Return an error */
          assert (event->event == RDMA_CM_EVENT_ROUTE_RESOLVED);
          return NULL;
        }
      rdma_ack_cm_event (event);

      rdma_init_cq (&d->rdma_data);

      /* RDMA CONNECT */

      struct rdma_conn_param conn_param = {};
      conn_param.initiator_depth = 1;
      conn_param.retry_count = 5;
      conn_param.private_data = d->authkey;
      conn_param.private_data_len = AUTHKEY_LENGTH;

      error = rdma_connect (d->rdma_data.cm_id, &conn_param);
      if (error)
        {
          /* TODO: Return an error */
          assert (!error);
          return NULL;
        }

      error = rdma_get_cm_event (d->rdma_data.cm_channel, &event);
      if (error)
        {
          /* TODO: Return an error */
          assert (!error);
          return NULL;
        }
      if (event->event != RDMA_CM_EVENT_ESTABLISHED)
        {
          /* TODO: Return an error */
          assert (event->event == RDMA_CM_EVENT_ESTABLISHED);
          return NULL;
        }

      rdma_ack_cm_event (event);
    }
#endif

  /* initial ServerInfo reply/response */
  RequestMsg_t req;
  memset (&req, 0, sizeof (RequestMsg_t));
  req.message_type = MessageType_ServerInfo;
  req.session = d->session;
  memcpy (req.authkey, d->authkey, AUTHKEY_LENGTH);
  uint32_t msg_size = request_size (req.message_type);

  ssize_t writeb, readb;
  writeb = write (d->fast_connection.fd, &msg_size, sizeof (msg_size));
  assert ((size_t)(writeb) == sizeof (msg_size));
  writeb = write (d->fast_connection.fd, &req, msg_size);
  assert ((size_t)(writeb) == msg_size);

  ReplyMsg_t rep;
  readb = read (d->fast_connection.fd, &rep, sizeof (ReplyMsg_t));
  assert ((size_t)(readb) == sizeof (ReplyMsg_t));
  assert (rep.message_type == MessageType_ServerInfoReply);

  d->num_platforms = rep.obj_id;
  d->platform_devices = malloc (rep.data_size);
  readb = read (d->fast_connection.fd, d->platform_devices, rep.data_size);
  assert ((size_t)(readb) == rep.data_size);
  /****************************************/
  size_t num_plat_devs
    = rep.data_size
      / sizeof (uint32_t); /* = device_counts.size() * sizeof(uint32_t); */
  d->num_devices = 0;
  for (size_t i = 0; i < num_plat_devs; ++i)
    d->num_devices += d->platform_devices[i];

  POCL_MSG_PRINT_REMOTE ("Connected to %s:%d which has %d devices\n",
                         d->address, d->fast_port, d->num_devices);

  /****************************************/

  SMALL_VECTOR_INIT (d, uint32_t, buffer_ids, INITIAL_ARRAY_CAP);

  SMALL_VECTOR_INIT (d, uint32_t, program_ids, INITIAL_ARRAY_CAP);

  SMALL_VECTOR_INIT (d, uint32_t, kernel_ids, INITIAL_ARRAY_CAP);

  SMALL_VECTOR_INIT (d, uint32_t, image_ids, INITIAL_ARRAY_CAP);

  SMALL_VECTOR_INIT (d, uint32_t, sampler_ids, INITIAL_ARRAY_CAP);

  start_engines (d, ddata, device);

  return d;
}

static void
release_server (remote_server_data_t *d)
{
  if (d->refcount > 1)
    {
      d->refcount -= 1;
      return;
    }

  DL_DELETE (servers, d);

  /* shutdown all threads */
  stop_engines (d);

  SMALL_VECTOR_DESTROY (d, buffer_ids, INITIAL_ARRAY_CAP);

  SMALL_VECTOR_DESTROY (d, program_ids, INITIAL_ARRAY_CAP);

  SMALL_VECTOR_DESTROY (d, kernel_ids, INITIAL_ARRAY_CAP);

  SMALL_VECTOR_DESTROY (d, image_ids, INITIAL_ARRAY_CAP);

  SMALL_VECTOR_DESTROY (d, sampler_ids, INITIAL_ARRAY_CAP);

  /* disconnect sockets */
  connection_disconnect (&d->fast_connection);
  connection_release (&d->fast_connection);
  connection_disconnect (&d->slow_connection);
  connection_release (&d->slow_connection);

#ifdef ENABLE_RDMA
  rdma_uninitialize (&d->rdma_data);
#endif
}

cl_int
pocl_network_init_device (cl_device_id device, remote_device_data_t *ddata,
                          int dev_idx, const char *const parameters)
{

  char *tmp = strdup (parameters);

  uint32_t did = 0;
  if (strchr (tmp, '/') != NULL)
    {
      /* determine device ID from parameters */
      char *address_with_port = strtok (tmp, "/");
      char *did_str = tmp + strlen (address_with_port) + 1;
      did = (uint32_t)atoi (did_str);
    }

  char address_with_guaranteed_port[MAX_ADDRESS_PORT_SIZE] = {};

  char address[MAX_ADDRESS_SIZE];
  unsigned port = DEFAULT_POCL_REMOTE_PORT;

  if (strncmp (tmp, "vsock:", strlen ("vsock:")) == 0)
    {
      char *colon = strchr (tmp, ':');
      char *second_colon = strchr (colon + 1, ':');
      if (second_colon)
        {
          /* vsock:vm:port */
          strncpy (address, tmp, second_colon - tmp);
          port = (unsigned)atoi (strchr (colon + 1, ':') + 1);
        }
      else
        {
          /* vsock:vm */
          strcpy (address, tmp);
        }
    }
  else
    {
      char *last_colon = strrchr (tmp, ':');
      if (last_colon)
        {
          strncpy (address, tmp, last_colon - tmp);
          address[last_colon - tmp] = '\0';
          port = (unsigned)atoi (last_colon + 1);
        }
      else
        {
          strcpy (address, tmp);
        }
    }
  if ((port == 0) || (port > UINT16_MAX))
    {
      port = DEFAULT_POCL_REMOTE_PORT;
      POCL_MSG_ERR ("Could not parse port, using default %u\n", port);
    }
  snprintf (address_with_guaranteed_port, MAX_ADDRESS_PORT_SIZE, "%s:%d",
            address, port);

  remote_server_data_t *data = find_or_create_server (
      address_with_guaranteed_port, port, ddata, device, parameters);
  POCL_MEM_FREE (tmp);

  POCL_RETURN_ERROR_ON ((data == NULL), CL_INVALID_DEVICE,
                        "Could not connect to server \n");
  POCL_RETURN_ERROR_ON ((did >= data->num_devices), CL_INVALID_DEVICE,
                        "Device ID (%u) >= number of server devices (%u) \n",
                        did, data->num_devices);
  ddata->server = data;

  /* TODO let user specify platform */
  uint32_t pid = 0;
  while (did >= data->platform_devices[pid])
    {
      assert (data->platform_devices[pid] > 0);
      did -= data->platform_devices[pid];
      ++pid;
      assert (pid < data->num_platforms);
    }
  POCL_MSG_PRINT_REMOTE (
      "Setting up remote device with PLATFORM %u / DEVICE %u\n", pid, did);

  ddata->remote_device_index = did;
  ddata->remote_platform_index = pid;
  ddata->local_did = device->dev_id;
  device->available = &data->available;
  return 0;
}

#define D(x) device->x = devinfo->x

/**
 * Fetches the device info data.
 *
 * @param specific_info Can be set to a specific (vendor-specific) device info
 * id, then that info will be returned only in @param param_value and
 * @param param_value_size_ret. Otherwise, the standard info will be
 * populated and cached for returning later from clGetDeviceInfo().
 */
cl_int
pocl_network_fetch_devinfo (cl_device_id device,
                            cl_device_info specific_info,
                            size_t param_value_size,
                            void *param_value,
                            size_t *param_value_size_ret)
{
  remote_device_data_t *ddata = (remote_device_data_t *)device->data;
  remote_server_data_t *data = ddata->server;
  uint32_t pid = ddata->remote_platform_index;
  uint32_t did = ddata->remote_device_index;

  CREATE_SYNC_NETCMD;

  ID_REQUEST (DeviceInfo, did);

  DeviceInfo_t *devinfo;
  /* Allocate the data in pocl_remote_reader_pthread() when we know the
     string section length. */
  netcmd->rep_extra_data = NULL;
  netcmd->rep_extra_size = sizeof (DeviceInfo_t);
  netcmd->request.m.device_info.id = specific_info;

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  CHECK_REPLY (DeviceInfo);

  assert (netcmd->rep_extra_data != NULL);
  assert (netcmd->reply.data_size >= sizeof (DeviceInfo_t));

  devinfo = (DeviceInfo_t *)netcmd->rep_extra_data;

  if (specific_info != 0)
    {
      if (devinfo->specific_info_size == 0)
        return CL_INVALID_VALUE;

      if (param_value != NULL)
        {
          if (param_value_size != devinfo->specific_info_size)
            return CL_INVALID_VALUE;
          memcpy (param_value, &devinfo->specific_info_data,
                  devinfo->specific_info_size);
        }

      if (param_value_size_ret != NULL)
        *param_value_size_ret = devinfo->specific_info_size;
      return CL_SUCCESS;
    }

  device->host_unified_memory = 0;
  device->execution_capabilities = CL_EXEC_KERNEL;

#define GET_STRING(ATTR) strdup ((char *)netcmd->strings + ATTR)

  device->long_name = device->short_name = GET_STRING (devinfo->name);

  char *remote_dev_version = GET_STRING (devinfo->device_version);
  unsigned dev_ver_major, dev_ver_minor;
  if (remote_dev_version == NULL
      || sscanf (remote_dev_version, "OpenCL %u.%u", &dev_ver_major,
                 &dev_ver_minor)
             != 2)
    {
      /* Illegal version string from the remote device, we should not add it
         to the platform. */
      POCL_MSG_PRINT_REMOTE (
          "Illegal version string '%s' from a remote device,"
          "skipping the device.",
          remote_dev_version);
      free (remote_dev_version);
      return -1;
    }
  SETUP_DEVICE_CL_VERSION (device, dev_ver_major, dev_ver_minor);
  /* Use the remote's device version for the first part of the version string.
   */
  if (dev_ver_major >= 3)
    {
      /* The minimums for OpenCL 3.0 compliant devices. TODO: fetch
         the actually supported capabilities. */
      device->atomic_memory_capabilities
          = CL_DEVICE_ATOMIC_ORDER_RELAXED | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP;
      device->atomic_fence_capabilities = CL_DEVICE_ATOMIC_ORDER_RELAXED
                                          | CL_DEVICE_ATOMIC_ORDER_ACQ_REL
                                          | CL_DEVICE_ATOMIC_SCOPE_WORK_GROUP;
    }
  device->version = remote_dev_version;
  device->driver_version = GET_STRING (devinfo->driver_version);
  device->vendor = GET_STRING (devinfo->vendor);
  device->extensions = GET_STRING (devinfo->extensions);
  pocl_setup_extensions_with_version (device);
  device->supported_spir_v_versions
      = GET_STRING (devinfo->supported_spir_v_versions);
  pocl_setup_ils_with_version (device);
  pocl_setup_spirv_queries (device);

  if (devinfo->builtin_kernels)
    device->builtin_kernel_list = GET_STRING (devinfo->builtin_kernels);

  /* This one is deprecated (and seems to be always 128) */
  device->min_data_type_align_size = 128;

  ddata->device_svm_region_start_addr = devinfo->svm_pool_start_address;
  ddata->device_svm_region_size = devinfo->svm_pool_size;

  D (vendor_id);
  /* TODO:
   * D(device_id); */
  D (address_bits);
  D (mem_base_addr_align);

  D (global_mem_size);
  D (max_mem_alloc_size);
  D (global_mem_cacheline_size);
  D (global_mem_cache_size);
  D (global_mem_cache_type);

  D (double_fp_config);
  D (single_fp_config);
  D (half_fp_config);

  D (image_support);
  D (endian_little);
  D (error_correction_support);

  cl_device_type t = CL_DEVICE_TYPE_CPU;
  switch (devinfo->type)
    {
    case CPU:
      t = CL_DEVICE_TYPE_CPU;
      break;
    case GPU:
      t = CL_DEVICE_TYPE_GPU;
      break;
    case CUSTOM:
      t = CL_DEVICE_TYPE_CUSTOM;
      break;
    case ACCELERATOR:
      t = CL_DEVICE_TYPE_ACCELERATOR;
      break;
    }

  device->type = t;

  device->profile
      = (devinfo->full_profile ? "FULL_PROFILE" : "EMBEDDED_PROFILE");
  D (on_host_queue_props);
  device->compiler_available = 1;
  device->linker_available = 1;
  /* may actually be emulated by the remote pocld */
  device->native_command_buffers = 1;
  D (cmdbuf_capabilities);
  D (cmdbuf_supported_properties);
  D (cmdbuf_required_properties);

  D (local_mem_size);
  D (local_mem_type);
  D (max_clock_frequency);
  D (max_compute_units);

  D (max_constant_args);
  D (max_constant_buffer_size);
  D (max_parameter_size);

  D (max_work_item_dimensions);
  D (max_work_group_size);

  device->max_work_item_sizes[0] = devinfo->max_work_item_size_x;
  device->max_work_item_sizes[1] = devinfo->max_work_item_size_y;
  device->max_work_item_sizes[2] = devinfo->max_work_item_size_z;

  D (native_vector_width_char);
  D (native_vector_width_short);
  D (native_vector_width_int);
  D (native_vector_width_long);
  D (native_vector_width_float);
  D (native_vector_width_double);

  D (preferred_vector_width_char);
  D (preferred_vector_width_short);
  D (preferred_vector_width_int);
  D (preferred_vector_width_long);
  D (preferred_vector_width_float);
  D (preferred_vector_width_double);

  D (printf_buffer_size);
  D (profiling_timer_resolution);

  /****************************************/
  /****************************************/

  if (devinfo->image_support == CL_FALSE)
    {
      free (devinfo);
      return 0;
    }

  D (max_read_image_args);
  D (max_write_image_args);
  D (max_samplers);
  D (image2d_max_height);
  D (image2d_max_width);
  D (image3d_max_height);
  D (image3d_max_width);
  D (image3d_max_depth);
  D (image_max_buffer_size);
  D (image_max_array_size);

  D (max_num_sub_groups);
  D (host_unified_memory);

  size_t i, j;
  for (i = 0; i < NUM_OPENCL_IMAGE_TYPES; ++i)
    {
      ImgFormatInfo_t p = devinfo->supported_image_formats[i];
      if (p.num_formats == 0)
        continue;

      assert (p.memobj_type != 0);
      int k = pocl_opencl_image_type_to_index (
          (cl_mem_object_type)p.memobj_type);

      cl_image_format *ary = calloc (p.num_formats, sizeof (cl_image_format));
      device->image_formats[k] = ary;
      device->num_image_formats[k] = p.num_formats;

      for (j = 0; j < p.num_formats; ++j)
        {
          ary[j].image_channel_data_type = p.formats[j].channel_data_type;
          ary[j].image_channel_order = p.formats[j].channel_order;
        }
    }

  /*  LLVM triple + cpu type => for compilation; build hash => for binaries */
  free (devinfo);
  return 0;
}

#undef D

cl_int
pocl_network_free_device (cl_device_id device)
{
  REMOTE_SERV_DATA;

  release_server (data);

  return 0;
}

/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

/* SYNCHRONOUS COMMANDS
 *
 * These functions block until the server has sent a reply.
 *
 * NOTE: netcmd structs for synchronous commands are allocated on the stack and
 * thus NOT usable after these functions return.
 */

/** Allocate a buffer on the server */
cl_int
pocl_network_create_buffer (remote_device_data_t *ddata, cl_mem mem,
                            void **device_addr)
{
  /* = (remote_device_data_t *)device->data; */
  REMOTE_SERV_DATA2;

  RETURN_IF_REMOTE_ID (buffer, mem->id);

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_ALLOC);

  ID_REQUEST (CreateBuffer, mem->id);

  assert (mem->size > 0);
  assert (mem->flags != 0);

  nc.request.m.create_buffer.flags = mem->flags;
  nc.request.m.create_buffer.size = mem->size;
  /* See
     https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Pointer_002dInteger-Conversion.html
     for the reason behind the double cast. */
  nc.request.m.create_buffer.host_ptr = (uint64_t)(uintptr_t)mem->mem_host_ptr;

  nc.request.m.create_buffer.origin = mem->origin;
  nc.request.m.create_buffer.parent_id
    = mem->parent != NULL ? mem->parent->id : 0;

#ifdef ENABLE_RDMA
  CreateRdmaBufferReply_t info;
  nc.rep_extra_data = (char *)&info;
  nc.rep_extra_size = sizeof (info);
#endif

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_ALLOC);

  CHECK_REPLY (CreateBuffer);

  SET_REMOTE_ID (buffer, mem->id);

  if (mem->flags & CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT)
    {
      assert (device_addr != NULL);
      *device_addr = (void *)netcmd->reply.m.create_buffer.device_addr;
    }

#ifdef ENABLE_RDMA
  rdma_buffer_info_t *s = malloc (sizeof (rdma_buffer_info_t));
  s->mem_id = mem->id;
  s->remote_vaddr = info.server_vaddr;
  s->remote_rkey = info.server_rkey;
  /* NOTE: mem_id here is the name of the struct field holding the hashmap key,
   * not the local variable */
  HASH_ADD (hh, data->rdma_keys, mem_id, sizeof (uint32_t), s);
#endif

  return CL_SUCCESS;
}

/**
 * Frees a remote buffer.
 *
 * @param mem_id Is either the cl_mem identifier or an SVM device-side
 * address (in that case @param is_svm is set to 1).
 */
cl_int
pocl_network_free_buffer (remote_device_data_t *ddata, uint64_t mem_id,
                          int is_svm)
{
  REMOTE_SERV_DATA2;

  if (!is_svm)
    RETURN_IF_NOT_REMOTE_ID (buffer, mem_id);

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_FREE);

  ID_REQUEST (FreeBuffer, mem_id);

  req->m.free_buffer.is_svm = is_svm;

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_FREE);

  CHECK_REPLY (FreeBuffer);

  if (!is_svm)
    {
      UNSET_REMOTE_ID (buffer, mem_id);

#ifdef ENABLE_RDMA
      rdma_buffer_info_t *s;
      HASH_FIND (hh, data->rdma_keys, &mem_id, sizeof (uint32_t), s);
      HASH_DEL (data->rdma_keys, s);
      free (s);
#endif
    }

  return 0;
}

#define READ_BYTES(var)                                                       \
  memcpy (&var, buf, sizeof (var));                                           \
  buf += sizeof (var);                                                        \
  assert ((size_t)(buf - buffer) <= (size_t)nc.reply.data_size)

#define READ_BYTES_SIZE(var, size)                                            \
  memcpy (&var, buf, sizeof (var));                                           \
  buf += sizeof (var);                                                        \
  assert ((size_t)(buf - buffer) <= (size_t)size)

#define READ_STRING(str, len)                                                 \
  (str) = (char *)malloc (len + 1);                                           \
  memcpy ((str), buf, len);                                                   \
  (str)[len] = 0;                                                             \
  buf += len;                                                                 \
  assert ((size_t)(buf - buffer) <= (size_t)nc.reply.data_size)

#define MAX_BUILD_SIZE (16 << 20)

cl_int
pocl_network_setup_metadata (char *buffer, size_t total_size,
                             cl_program program, size_t *num_kernels,
                             pocl_kernel_metadata_t **kernel_meta)
{
  POCL_MSG_PRINT_REMOTE ("Setting up Kernel metadata\n");

  char *buf = buffer;

  uint32_t nk;
  READ_BYTES_SIZE (nk, total_size);
  assert (nk < 1000000);
  pocl_kernel_metadata_t *p = NULL;

  *num_kernels = nk;
  p = calloc (*num_kernels, sizeof (pocl_kernel_metadata_t));
  assert (p);
  *kernel_meta = p;

  POCL_MSG_PRINT_REMOTE ("Num kernels: %zu\n", *num_kernels);

  size_t i;
  uint32_t j;
  for (i = 0; i < *num_kernels; ++i)
    {
      KernelMetaInfo_t temp_kernel;
      READ_BYTES_SIZE (temp_kernel, total_size);
      {
        p[i].attributes = strdup (temp_kernel.attributes);
        p[i].name = strdup (temp_kernel.name);
        p[i].num_args = temp_kernel.num_args;

        /* because have to return total local size */
        p[i].num_locals = 1;
        p[i].local_sizes = calloc (1, sizeof (size_t));
        p[i].local_sizes[0] = temp_kernel.total_local_size;
        p[i].data = calloc (program->num_devices, sizeof (void *));
        p[i].has_arg_metadata = (-1);
        p[i].reqd_wg_size[0] = temp_kernel.reqd_wg_size.x;
        p[i].reqd_wg_size[1] = temp_kernel.reqd_wg_size.y;
        p[i].reqd_wg_size[2] = temp_kernel.reqd_wg_size.z;

        p[i].arg_info = calloc (p[i].num_args, sizeof (pocl_argument_info));
      }

      uint32_t num_a;
      READ_BYTES_SIZE (num_a, total_size);
      assert (num_a == temp_kernel.num_args);
      for (j = 0; j < temp_kernel.num_args; ++j)
        {
          ArgumentInfo_t temp_arg;
          READ_BYTES_SIZE (temp_arg, total_size);

          {
            p[i].arg_info[j].access_qualifier = temp_arg.access_qualifier;
            p[i].arg_info[j].address_qualifier = temp_arg.address_qualifier;
            p[i].arg_info[j].name = strdup (temp_arg.name);

            pocl_argument_type t = POCL_ARG_TYPE_NONE;
            switch (temp_arg.type)
              {
              case POD:
                t = POCL_ARG_TYPE_NONE;
                break;
              case Local:
              case Pointer:
                t = POCL_ARG_TYPE_POINTER;
                break;
              case Sampler:
                t = POCL_ARG_TYPE_SAMPLER;
                break;
              case Image:
                t = POCL_ARG_TYPE_IMAGE;
                break;
              default:
                POCL_MSG_ERR ("Couldn't detect argument type?");
                return CL_BUILD_ERROR;
              }
            p[i].arg_info[j].type = t;

            p[i].arg_info[j].type_name = strdup (temp_arg.type_name);
            p[i].arg_info[j].type_qualifier = temp_arg.type_qualifier;
            /* TODO: there's no way to get this from OpenCL API currently. */
            p[i].arg_info[j].type_size = 0;
          }
        }
    }

  return CL_SUCCESS;
}

cl_int
pocl_network_setup_peer_mesh ()
{
  remote_server_data_t *src;
  remote_server_data_t *tgt;
  remote_server_data_t *tgt_base;

  if (!servers)
    return CL_SUCCESS;

  DL_FOREACH (servers, src)
  {
    tgt_base = src->next;
    if (tgt_base)
      {
        DL_FOREACH (tgt_base, tgt)
        {
          remote_server_data_t *data = src;
          CREATE_SYNC_NETCMD;
          REQUEST_PEERCONN;
          req->m.connect_peer.port = tgt->peer_port;
          req->m.connect_peer.session = tgt->session;
          strncpy (req->m.connect_peer.address, tgt->peer_address,
                   MAX_REMOTE_PARAM_LENGTH);
          SEND_REQ_FAST;
          wait_on_netcmd (netcmd);

          assert (!netcmd->reply.failed);
        }
      }
  }

  return CL_SUCCESS;
}

/**
 * Build, compile or link a program remotely.
 *
 * \param [i] payload The sources or binaries, if compiling/building, or a list
 * of program ids, if linking only. \param [i] is_binary, is_builtin, is_spirv
 * Define the input type. If we are only linking previously compiled programs,
 * setting these have no difference. \param [i] svm_region_offset Nonzero
 * offset if the build process should adjust the memory accessess of the
 * program to account for the offset between the SVM regions. \param [i]
 * compile_only Set to 1 if compiling without linking. Otherwise 0. \param [i]
 * link_only
 *
 */
cl_int
pocl_network_build_or_link_program (remote_device_data_t *ddata,
                                    const void *payload,
                                    size_t payload_size,
                                    int is_binary,
                                    int is_builtin,
                                    int is_dbk,
                                    int is_spirv,
                                    uint32_t prog_id,
                                    const char *options,
                                    char **kernel_meta_bytes,
                                    size_t *kernel_meta_size,
                                    uint32_t *devices,
                                    uint32_t *platforms,
                                    size_t num_devices,
                                    char **build_logs,
                                    char **binaries,
                                    size_t *binary_sizes,
                                    size_t svm_region_offset,
                                    int compile_only,
                                    int link_only)
{
  size_t i, j;
  REMOTE_SERV_DATA2;

  /* TODO */
  RETURN_IF_REMOTE_ID (program, prog_id);

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_BUILD_PROGRAM);

  ID_REQUEST (ReadBuffer, prog_id);
  if (link_only)
    nc.request.message_type = MessageType_LinkProgram;
  else if (is_spirv)
    nc.request.message_type = compile_only ?
      MessageType_CompileProgramFromSPIRV : MessageType_BuildProgramFromSPIRV;
  else if (is_dbk)
    nc.request.message_type = MessageType_BuildProgramWithDefinedBuiltins;
  else if (is_builtin)
    nc.request.message_type = MessageType_BuildProgramWithBuiltins;
  else if (is_binary)
    nc.request.message_type = MessageType_BuildProgramFromBinary;
  else
    nc.request.message_type = compile_only
                                  ? MessageType_CompileProgramFromSource
                                  : MessageType_BuildProgramFromSource;

  nc.request.m.build_program.payload_size = payload_size;
  nc.request.m.build_program.options_len = options ? strlen (options) : 0;
  nc.request.m.build_program.svm_region_offset = svm_region_offset;

  nc.request.m.build_program.num_devices = num_devices;
  assert (num_devices < MAX_REMOTE_DEVICES);
  memcpy (nc.request.m.build_program.devices, devices,
          num_devices * sizeof (uint32_t));
  memcpy (nc.request.m.build_program.platforms, platforms,
          num_devices * sizeof (uint32_t));

  nc.req_extra_data = payload;
  nc.req_extra_size = payload_size;
  nc.req_extra_data2 = options;
  nc.req_extra_size2 = nc.request.m.build_program.options_len;

  nc.rep_extra_data
      = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, MAX_BUILD_SIZE);
  nc.rep_extra_size = MAX_BUILD_SIZE;

  POCL_MSG_PRINT_REMOTE ("Compile/Build/LinkProgram %p\n", netcmd);

  /* The build command contains the source payload which can be orders
   * of magnitudes bigger than other commands, so send this command on the slow
   * queue which is better equipped for large payloads.*/
  SEND_REQ_SLOW;

  wait_on_netcmd (netcmd);

  POCL_MSG_PRINT_REMOTE ("Compile/Build/LinkProgram reply DATA: %zu\n",
                         nc.reply.data_size);
  POCL_MEASURE_FINISH (REMOTE_BUILD_PROGRAM);

  char *buffer = nc.rep_extra_data;
  char *buf = buffer;

  /* read build log also for failed builds */
  if (netcmd->status != NETCMD_FAILED)
    {
      uint32_t build_log_len;
      READ_BYTES (build_log_len);
      assert (build_log_len == num_devices);
      assert (build_logs);
      for (i = 0; i < num_devices; ++i)
        {
          READ_BYTES (build_log_len);
          if (build_log_len > 0)
            {
              READ_STRING (build_logs[i], build_log_len);
            }
          else
            build_logs[i] = NULL;
        }
    }

  if (netcmd->reply.failed)
    pocl_aligned_free (nc.rep_extra_data);
  CHECK_REPLY (BuildProgram);

  /*****************************************************************/
  /*****************************************************************/
  /*****************************************************************/

  /* copy kernel metadata */
  uint64_t metadata_size = 0;
  READ_BYTES (metadata_size);
  assert (metadata_size > 0);

  assert ((size_t)((buf + metadata_size) - buffer)
          <= (size_t)nc.reply.data_size);
  READ_STRING (*kernel_meta_bytes, metadata_size);
  *kernel_meta_size = (size_t)metadata_size;
  POCL_MSG_PRINT_REMOTE ("METADATA SIZE: %zu\n", (size_t)metadata_size);

  /* read program binaries */
  if (!is_binary && !is_builtin)
    {
      assert ((size_t)((buf + sizeof (uint32_t)) - buffer)
              <= (size_t)nc.reply.data_size);
      uint32_t numd;
      assert (binaries);
      assert (binary_sizes);
      READ_BYTES (numd);
      assert (numd == num_devices);

      POCL_MSG_PRINT_REMOTE ("READING binaries for %u devices\n", numd);

      for (i = 0; i < num_devices; ++i)
        {
          binaries[i] = NULL;
          binary_sizes[i] = 0;
          uint32_t binary_len;
          READ_BYTES (binary_len);
          binary_sizes[i] = binary_len;
          if (binary_len > 0)
            {
              READ_STRING (binaries[i], binary_len);
            }
          POCL_MSG_PRINT_REMOTE ("Dev %zu bin size: %u\n", i, binary_len);
        }
    }

  pocl_aligned_free (nc.rep_extra_data);
  POCL_MSG_PRINT_REMOTE ("NEW program ID: %u\n", prog_id);

  SET_REMOTE_ID (program, prog_id);
  return CL_SUCCESS;
}

cl_int
pocl_network_free_program (remote_device_data_t *ddata, uint32_t prog_id)
{
  REMOTE_SERV_DATA2;

  RETURN_IF_NOT_REMOTE_ID (program, prog_id);

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_FREE_PROGRAM);

  ID_REQUEST (FreeProgram, prog_id);

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_FREE_PROGRAM);

  CHECK_REPLY (FreeProgram);

  UNSET_REMOTE_ID (program, prog_id);
  return 0;
}

cl_int
pocl_network_create_kernel (remote_device_data_t *ddata, const char *name,
                            uint32_t prog_id, uint32_t kernel_id,
                            kernel_data_t *kd)
{
  REMOTE_SERV_DATA2;

  RETURN_IF_REMOTE_ID (kernel, kernel_id);

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_CREATE_KERNEL);

  ID_REQUEST (CreateKernel, kernel_id);
  size_t len = strlen (name);
  nc.request.m.create_kernel.name_len = len;
  nc.request.m.create_kernel.prog_id = prog_id;
  nc.req_extra_data = name;
  nc.req_extra_size = len;

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_CREATE_KERNEL);

  CHECK_REPLY (CreateKernel);

  SET_REMOTE_ID (kernel, kernel_id);
  return 0;
}

cl_int
pocl_network_free_kernel (remote_device_data_t *ddata, kernel_data_t *kd,
                          uint32_t kernel_id, uint32_t program_id)
{
  REMOTE_SERV_DATA2;

  RETURN_IF_NOT_REMOTE_ID (kernel, kernel_id);

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_FREE_KERNEL);

  ID_REQUEST (FreeKernel, kernel_id);
  nc.request.m.free_kernel.prog_id = program_id;

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_FREE_KERNEL);

  CHECK_REPLY (FreeKernel);

  UNSET_REMOTE_ID (kernel, kernel_id);
  return 0;
}

cl_int
pocl_network_create_queue (remote_device_data_t *ddata, uint32_t queue_id)
{
  REMOTE_SERV_DATA2;

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_CREATE_QUEUE);

  ID_REQUEST (CreateCommandQueue, queue_id);

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_CREATE_QUEUE);

  CHECK_REPLY (CreateCommandQueue);
  return 0;
}

cl_int
pocl_network_free_queue (remote_device_data_t *ddata, uint32_t queue_id)
{
  REMOTE_SERV_DATA2;

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_FREE_QUEUE);

  ID_REQUEST (FreeCommandQueue, queue_id);

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_FREE_QUEUE);

  CHECK_REPLY (FreeCommandQueue);
  return 0;
}

cl_int
pocl_network_create_command_buffer (remote_device_data_t *ddata,
                                    uint64_t cmdbuf_id,
                                    uint64_t num_commands,
                                    uint64_t commands_offset,
                                    uint64_t commands_size,
                                    uint64_t num_queues,
                                    uint64_t queues_offset,
                                    const char *payload)
{
  REMOTE_SERV_DATA2;

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_CREATE_COMMAND_BUFFER);

  ID_REQUEST (CreateCommandBuffer, cmdbuf_id);

  uint64_t queues_size = sizeof (uint32_t) * num_queues;
  req->m.create_cmdbuf.num_commands = num_commands;
  req->m.create_cmdbuf.commands_size = commands_size;
  req->m.create_cmdbuf.commands_offset = commands_offset;
  req->m.create_cmdbuf.num_queues = num_queues;
  req->m.create_cmdbuf.queues_offset = queues_offset;
  netcmd->req_extra_data = payload;
  netcmd->req_extra_size = commands_size + queues_size;

  SEND_REQ_SLOW;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_CREATE_COMMAND_BUFFER);

  CHECK_REPLY (CreateCommandBuffer);
  return 0;
}

cl_int
pocl_network_free_command_buffer (remote_device_data_t *ddata,
                                  uint64_t cmdbuf_id)
{
  REMOTE_SERV_DATA2;

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_FREE_COMMAND_BUFFER);

  ID_REQUEST (FreeCommandBuffer, cmdbuf_id);

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_FREE_COMMAND_BUFFER);

  CHECK_REPLY (FreeCommandBuffer);
  return 0;
}

cl_int
pocl_network_create_sampler (remote_device_data_t *ddata,
                             cl_bool normalized_coords,
                             cl_addressing_mode addressing_mode,
                             cl_filter_mode filter_mode, uint32_t samp_id)
{
  REMOTE_SERV_DATA2;

  RETURN_IF_REMOTE_ID (sampler, samp_id);

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_CREATE_SAMPLER);

  ID_REQUEST (CreateSampler, samp_id);
  nc.request.m.create_sampler.address_mode = addressing_mode;
  nc.request.m.create_sampler.filter_mode = filter_mode;
  nc.request.m.create_sampler.normalized = normalized_coords ? 1 : 0;

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_CREATE_SAMPLER);

  CHECK_REPLY (CreateSampler);

  SET_REMOTE_ID (sampler, samp_id);
  return 0;
}

cl_int
pocl_network_free_sampler (remote_device_data_t *ddata, uint32_t samp_id)
{
  REMOTE_SERV_DATA2;

  RETURN_IF_NOT_REMOTE_ID (sampler, samp_id);

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_FREE_SAMPLER);

  ID_REQUEST (FreeSampler, samp_id);

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_FREE_SAMPLER);

  CHECK_REPLY (FreeSampler);

  UNSET_REMOTE_ID (sampler, samp_id);
  return 0;
}

cl_int
pocl_network_create_image (remote_device_data_t *ddata, cl_mem image)
{
  REMOTE_SERV_DATA2;

  uint32_t id = (uint32_t)image->id;

  RETURN_IF_REMOTE_ID (image, id);

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_IMAGE_ALLOC);

  ID_REQUEST (CreateImage, id);
  nc.request.m.create_image.flags = (uint32_t)image->flags;
  nc.request.m.create_image.channel_data_type = image->image_channel_data_type;
  nc.request.m.create_image.channel_order = image->image_channel_order;

  nc.request.m.create_image.width = image->image_width;
  nc.request.m.create_image.height = image->image_height;
  nc.request.m.create_image.depth = image->image_depth;
  nc.request.m.create_image.array_size = image->image_array_size;
  nc.request.m.create_image.type = image->type;
  nc.request.m.create_image.row_pitch = image->image_row_pitch;
  nc.request.m.create_image.slice_pitch = image->image_slice_pitch;

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_IMAGE_ALLOC);

  CHECK_REPLY (CreateImage);

  SET_REMOTE_ID (image, id);

  return 0;
}

cl_int
pocl_network_free_image (remote_device_data_t *ddata, uint32_t image_id)
{
  REMOTE_SERV_DATA2;

  RETURN_IF_NOT_REMOTE_ID (image, image_id);

  CREATE_SYNC_NETCMD;

  POCL_MEASURE_START (REMOTE_FREE_IMAGE);

  ID_REQUEST (FreeImage, image_id);

  SEND_REQ_FAST;

  wait_on_netcmd (netcmd);

  POCL_MEASURE_FINISH (REMOTE_FREE_IMAGE);

  CHECK_REPLY (FreeImage);

  UNSET_REMOTE_ID (image, image_id);

  return 0;
}

/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/

/*
 * ASYNCHRONOUS COMMANDS
 *
 * These functions return immediately and give users the option to store an
 * OpenCL event corresponding to the command in order to specify it as a
 * dependency to another command, query the command's status or wait for this
 * specific command to complete.
 */

/** Network command corresponding to migrations directly between devices
 * without a need for a roundtrip to the client. These can be implicitly added
 * by PoCL or explicitly requested with clEnqueueMigrateMemObjects. */
cl_int
pocl_network_migrate_d2d (uint32_t cq_id, uint32_t mem_id, uint32_t size_id,
                          unsigned mem_is_image, uint32_t height,
                          uint32_t width, uint32_t depth, size_t size,
                          remote_device_data_t *dest,
                          remote_device_data_t *source,
                          network_command_callback cb, void *arg,
                          _cl_command_node *node)
{
  remote_device_data_t *ddata = dest;
  remote_server_data_t *data = dest->server;

  /* request */
  CREATE_ASYNC_NETCMD;

  ID_REQUEST (MigrateD2D, mem_id);
  req->cq_id = cq_id;

  req->m.migrate.source_pid = source->remote_platform_index;
  req->m.migrate.source_did = source->remote_device_index;
  req->m.migrate.dest_peer_id = dest->server->peer_id;
  req->m.migrate.source_peer_id = source->server->peer_id;
  req->m.migrate.is_image = mem_is_image;
  req->m.migrate.is_external = 0;
  req->m.migrate.size = size;
  req->m.migrate.depth = depth;
  req->m.migrate.width = width;
  req->m.migrate.height = height;
  req->m.migrate.size_id = size_id;

  data = source->server;
  SEND_REQ_FAST;

  return 0;
}

cl_int
pocl_network_read (uint32_t cq_id, remote_device_data_t *ddata,
                   uint32_t mem_id, int is_svm, uint32_t size_id,
                   void *host_ptr, size_t offset, size_t size,
                   network_command_callback cb, void *arg,
                   _cl_command_node *node)
{
  REMOTE_SERV_DATA2;
  assert (size > 0);

  /* request */
  CREATE_ASYNC_NETCMD;

  ID_REQUEST (ReadBuffer, mem_id);
  req->cq_id = cq_id;
  req->m.read.src_offset = offset;
  req->m.read.size = size;
  req->m.read.content_size_id = size_id;

  req->m.read.is_svm = is_svm;
  if (is_svm)
    req->obj_id = (uint64_t)host_ptr + ddata->svm_region_offset;
  /* REPLY */
  netcmd->rep_extra_data = host_ptr;
  netcmd->rep_extra_size = size;

  TP_READ_BUFFER (req->msg_id, ddata->local_did, cq_id,
                  node->sync.event.event->id);

  SEND_REQ_FAST;

  return 0;
}

cl_int
pocl_network_write (uint32_t cq_id, remote_device_data_t *ddata,
                    uint32_t mem_id, int is_svm, const void *host_ptr,
                    size_t offset, size_t size, network_command_callback cb,
                    void *arg, _cl_command_node *node)
{
  REMOTE_SERV_DATA2;

  CREATE_ASYNC_NETCMD;

  ID_REQUEST (WriteBuffer, mem_id);
  req->cq_id = cq_id;
  req->m.write.dst_offset = offset;
  req->m.write.size = size;

  req->m.write.is_svm = is_svm;
  if (is_svm)
    req->obj_id = (uint64_t)host_ptr + ddata->svm_region_offset;

  /* REQUEST */
  netcmd->req_extra_data = host_ptr;
  netcmd->req_extra_size = size;

  TP_WRITE_BUFFER (req->msg_id, ddata->local_did, cq_id,
                   node->sync.event.event->id);

#ifdef ENABLE_RDMA
  if (data->use_rdma)
    {
      SEND_REQ_RDMA;
    }
  else
    {
      SEND_REQ_SLOW;
    }
#else
  SEND_REQ_SLOW;
#endif

  return 0;
}

cl_int
pocl_network_copy (uint32_t cq_id, remote_device_data_t *ddata,
                   uint32_t src_id, uint32_t dst_id, uint32_t content_size_id,
                   size_t src_offset, size_t dst_offset, size_t size,
                   network_command_callback cb, void *arg,
                   _cl_command_node *node)
{
  REMOTE_SERV_DATA2;

  CREATE_ASYNC_NETCMD;

  REQUEST (CopyBuffer);
  req->cq_id = cq_id;
  req->m.copy.src_buffer_id = src_id;
  req->m.copy.dst_buffer_id = dst_id;
  req->m.copy.size_buffer_id = content_size_id;
  req->m.copy.src_offset = src_offset;
  req->m.copy.dst_offset = dst_offset;
  req->m.copy.size = size;

  TP_COPY_BUFFER (req->msg_id, ddata->local_did, cq_id,
                  node->sync.event.event->id);

  SEND_REQ_FAST;

  return 0;
}

cl_int
pocl_network_read_rect (uint32_t cq_id, remote_device_data_t *ddata,
                        uint32_t src_id,
                        const size_t *__restrict__ const buffer_origin,
                        const size_t *__restrict__ const region,
                        size_t const buffer_row_pitch,
                        size_t const buffer_slice_pitch, void *host_ptr,
                        size_t size, network_command_callback cb, void *arg,
                        _cl_command_node *node)
{
  REMOTE_SERV_DATA2;
  assert (size > 0);

  CREATE_ASYNC_NETCMD;

  ID_REQUEST (ReadBufferRect, src_id);
  req->cq_id = cq_id;

  req->m.read_rect.buffer_origin.x = buffer_origin[0];
  req->m.read_rect.buffer_origin.y = buffer_origin[1];
  req->m.read_rect.buffer_origin.z = buffer_origin[2];
  req->m.read_rect.region.x = region[0];
  req->m.read_rect.region.y = region[1];
  req->m.read_rect.region.z = region[2];
  req->m.read_rect.buffer_row_pitch = buffer_row_pitch;
  req->m.read_rect.buffer_slice_pitch = buffer_slice_pitch;
  req->m.read_rect.host_bytes = size;

  /* REPLY data */
  netcmd->rep_extra_data = host_ptr;
  netcmd->rep_extra_size = size;

  TP_READ_BUFFER_RECT (req->msg_id, ddata->local_did, cq_id,
                       node->sync.event.event->id);

  SEND_REQ_FAST;

  return 0;
}

cl_int
pocl_network_write_rect (uint32_t cq_id, remote_device_data_t *ddata,
                         uint32_t dst_id,
                         const size_t *__restrict__ const buffer_origin,
                         const size_t *__restrict__ const region,
                         size_t const buffer_row_pitch,
                         size_t const buffer_slice_pitch, const void *host_ptr,
                         size_t size, network_command_callback cb, void *arg,
                         _cl_command_node *node)
{
  REMOTE_SERV_DATA2;
  assert (size > 0);

  CREATE_ASYNC_NETCMD;

  ID_REQUEST (WriteBufferRect, dst_id);
  req->cq_id = cq_id;

  req->m.write_rect.buffer_origin.x = buffer_origin[0];
  req->m.write_rect.buffer_origin.y = buffer_origin[1];
  req->m.write_rect.buffer_origin.z = buffer_origin[2];
  req->m.write_rect.region.x = region[0];
  req->m.write_rect.region.y = region[1];
  req->m.write_rect.region.z = region[2];
  req->m.write_rect.buffer_row_pitch = buffer_row_pitch;
  req->m.write_rect.buffer_slice_pitch = buffer_slice_pitch;
  req->m.write_rect.host_bytes = size;

  /* REQUEST */
  netcmd->req_extra_data = host_ptr;
  netcmd->req_extra_size = size;

  TP_WRITE_BUFFER_RECT (req->msg_id, ddata->local_did, cq_id,
                        node->sync.event.event->id);

#ifdef ENABLE_RDMA
  if (data->use_rdma)
    {
      SEND_REQ_RDMA;
    }
  else
    {
      SEND_REQ_SLOW;
    }
#else
  SEND_REQ_SLOW;
#endif

  return 0;
}

cl_int
pocl_network_copy_rect (
    uint32_t cq_id, remote_device_data_t *ddata, uint32_t src_id,
    uint32_t dst_id, const size_t *__restrict__ const dst_origin,
    const size_t *__restrict__ const src_origin,
    const size_t *__restrict__ const region, size_t const dst_row_pitch,
    size_t const dst_slice_pitch, size_t const src_row_pitch,
    size_t const src_slice_pitch, network_command_callback cb, void *arg,
    _cl_command_node *node)
{
  REMOTE_SERV_DATA2;

  CREATE_ASYNC_NETCMD;

  REQUEST (CopyBufferRect);
  req->cq_id = cq_id;

  req->m.copy_rect.src_buffer_id = src_id;
  req->m.copy_rect.dst_buffer_id = dst_id;

  req->m.copy_rect.dst_origin.x = dst_origin[0];
  req->m.copy_rect.dst_origin.y = dst_origin[1];
  req->m.copy_rect.dst_origin.z = dst_origin[2];
  req->m.copy_rect.src_origin.x = src_origin[0];
  req->m.copy_rect.src_origin.y = src_origin[1];
  req->m.copy_rect.src_origin.z = src_origin[2];
  req->m.copy_rect.region.x = region[0];
  req->m.copy_rect.region.y = region[1];
  req->m.copy_rect.region.z = region[2];
  req->m.copy_rect.dst_row_pitch = dst_row_pitch;
  req->m.copy_rect.dst_slice_pitch = dst_slice_pitch;
  req->m.copy_rect.src_row_pitch = src_row_pitch;
  req->m.copy_rect.src_slice_pitch = src_slice_pitch;

  TP_COPY_BUFFER_RECT (req->msg_id, ddata->local_did, cq_id,
                       node->sync.event.event->id);

  SEND_REQ_FAST;

  return 0;
}

cl_int
pocl_network_fill_buffer (uint32_t cq_id, remote_device_data_t *ddata,
                          uint32_t mem_id, size_t size, size_t offset,
                          const void *pattern, size_t pattern_size,
                          network_command_callback cb, void *arg,
                          _cl_command_node *node)
{
  REMOTE_SERV_DATA2;
  assert (size > 0);
  assert (pattern_size > 0);

  CREATE_ASYNC_NETCMD;

  ID_REQUEST (FillBuffer, mem_id);
  req->cq_id = cq_id;
  req->m.fill_buffer.dst_offset = offset;
  req->m.fill_buffer.size = size;
  req->m.fill_buffer.pattern_size = pattern_size;

  netcmd->req_extra_data = pattern;
  netcmd->req_extra_size = pattern_size;

  TP_FILL_BUFFER (req->msg_id, ddata->local_did, cq_id,
                  node->sync.event.event->id);

  SEND_REQ_FAST;

  return 0;
}

#define ARGS_ARRAY_SIZE (kernel->num_args * sizeof (uint64_t))

cl_int
pocl_network_run_kernel (uint32_t cq_id, remote_device_data_t *ddata,
                         cl_kernel kernel, kernel_data_t *kd,
                         int requires_kernarg_update, unsigned dim,
                         vec3_t local, vec3_t global, vec3_t offset,
                         network_command_callback cb, void *arg,
                         _cl_command_node *node)
{
  REMOTE_SERV_DATA2;
  assert (kd != NULL);

  pocl_kernel_metadata_t *kernel_md = kernel->meta;
  uint32_t kernel_id = (uint32_t)kernel->id;

  CREATE_ASYNC_NETCMD;

  ID_REQUEST (RunKernel, kernel_id);
  req->cq_id = cq_id;
  req->m.run_kernel.global = global;
  req->m.run_kernel.local = local;
  req->m.run_kernel.offset = offset;
  req->m.run_kernel.has_local = 1;
  req->m.run_kernel.dim = dim;
  req->m.run_kernel.has_new_args = (uint8_t)requires_kernarg_update;

  if (requires_kernarg_update)
    {
      req->m.run_kernel.args_num = kernel_md->num_args;
      req->m.run_kernel.pod_arg_size = kd->pod_total_size;

      /* Push the arguments as extra data, as well as an array of
         flags which inform whether an argument (buffer) is an
         SVM pointer or not. */
      netcmd->req_extra_size
          = (kernel_md->num_args * sizeof (uint64_t))
            + (kernel_md->num_args * sizeof (unsigned char));
      if (netcmd->req_extra_size != 0)
        {
          netcmd->req_extra_data = malloc (netcmd->req_extra_size);
          unsigned char *ptr_is_svm_pos
              = (void *)netcmd->req_extra_data
                + kernel_md->num_args * sizeof (uint64_t);
          memcpy ((void *)netcmd->req_extra_data, kd->arg_array,
                  kernel_md->num_args * sizeof (uint64_t));
          memcpy (ptr_is_svm_pos, kd->ptr_is_svm,
                  kernel_md->num_args * sizeof (unsigned char));
        }
      netcmd->req_extra_size2 = kd->pod_total_size;
      if (netcmd->req_extra_size2 != 0)
        {
          netcmd->req_extra_data2 = malloc (netcmd->req_extra_size2);
          memcpy ((void *)netcmd->req_extra_data2, kd->pod_arg_storage,
                  netcmd->req_extra_size2);
        }
    }

  TP_NDRANGE_KERNEL (req->msg_id, ddata->local_did, cq_id,
                     node->sync.event.event->id, kernel_id, kernel->name);

  SEND_REQ_FAST;

  return 0;
}

cl_int
pocl_network_run_command_buffer (remote_device_data_t *ddata,
                                 network_command_callback cb,
                                 void *arg,
                                 _cl_command_node *node)
{
  REMOTE_SERV_DATA2;

  CREATE_ASYNC_NETCMD;

  ID_REQUEST (RunCommandBuffer, node->command.replay.buffer->id);
  req->cq_id = node->sync.event.event->queue->id;

  TP_COMMAND_BUFFER (req->msg_id, node->sync.event.event->id,
                     node->command.run_cmdbuf.id, node->sync.event.event->id);

  SEND_REQ_FAST;

  return 0;
}

cl_int
pocl_network_copy_image_rect (uint32_t cq_id, remote_device_data_t *ddata,
                              uint32_t src_remote_id, uint32_t dst_remote_id,
                              const size_t *__restrict__ const src_origin,
                              const size_t *__restrict__ const dst_origin,
                              const size_t *__restrict__ const region,
                              network_command_callback cb, void *arg,
                              _cl_command_node *node)
{
  REMOTE_SERV_DATA2;

  CREATE_ASYNC_NETCMD;

  REQUEST (CopyImage2Image);
  req->cq_id = cq_id;
  req->m.copy_img2img.src_image_id = src_remote_id;
  req->m.copy_img2img.dst_image_id = dst_remote_id;

  req->m.copy_img2img.dst_origin.x = dst_origin[0];
  req->m.copy_img2img.dst_origin.y = dst_origin[1];
  req->m.copy_img2img.dst_origin.z = dst_origin[2];
  req->m.copy_img2img.src_origin.x = src_origin[0];
  req->m.copy_img2img.src_origin.y = src_origin[1];
  req->m.copy_img2img.src_origin.z = src_origin[2];
  req->m.copy_img2img.region.x = region[0];
  req->m.copy_img2img.region.y = region[1];
  req->m.copy_img2img.region.z = region[2];

  TP_COPY_IMAGE_RECT (req->msg_id, ddata->local_did, cq_id,
                      node->sync.event.event->id);

  SEND_REQ_FAST;

  return 0;
}

cl_int
pocl_network_copy_buf2img (uint32_t cq_id, remote_device_data_t *ddata,
                           uint32_t src_remote_id, size_t src_offset,
                           uint32_t dst_remote_id,
                           const size_t *__restrict__ const origin,
                           const size_t *__restrict__ const region,
                           network_command_callback cb, void *arg,
                           _cl_command_node *node)
{
  REMOTE_SERV_DATA2;

  CREATE_ASYNC_NETCMD;

  REQUEST (CopyBuffer2Image);
  req->cq_id = cq_id;
  req->obj_id = dst_remote_id;

  req->m.copy_buf2img.origin.x = origin[0];
  req->m.copy_buf2img.origin.y = origin[1];
  req->m.copy_buf2img.origin.z = origin[2];
  req->m.copy_buf2img.region.x = region[0];
  req->m.copy_buf2img.region.y = region[1];
  req->m.copy_buf2img.region.z = region[2];
  req->m.copy_buf2img.src_buf_id = src_remote_id;
  req->m.copy_buf2img.src_offset = src_offset;

  TP_COPY_BUF_2_IMAGE (req->msg_id, ddata->local_did, cq_id,
                       node->sync.event.event->id);

  SEND_REQ_FAST;

  return 0;
}

cl_int
pocl_network_write_image_rect (uint32_t cq_id, remote_device_data_t *ddata,
                               uint32_t dst_remote_id,
                               const size_t *__restrict__ const origin,
                               const size_t *__restrict__ const region,
                               const void *__restrict__ host_ptr, size_t size,
                               network_command_callback cb, void *arg,
                               _cl_command_node *node)
{
  REMOTE_SERV_DATA2;
  assert (size > 0);

  CREATE_ASYNC_NETCMD;

  REQUEST (WriteImageRect);
  req->cq_id = cq_id;
  req->obj_id = dst_remote_id;

  req->m.write_image_rect.origin.x = origin[0];
  req->m.write_image_rect.origin.y = origin[1];
  req->m.write_image_rect.origin.z = origin[2];
  req->m.write_image_rect.region.x = region[0];
  req->m.write_image_rect.region.y = region[1];
  req->m.write_image_rect.region.z = region[2];
  req->m.write_image_rect.host_bytes = size;

  /* REQUEST */
  netcmd->req_extra_data = host_ptr;
  netcmd->req_extra_size = size;

  TP_WRITE_IMAGE_RECT (req->msg_id, ddata->local_did, cq_id,
                       node->sync.event.event->id);

#ifdef ENABLE_RDMA
  if (data->use_rdma)
    {
      SEND_REQ_RDMA;
    }
  else
    {
      SEND_REQ_SLOW;
    }
#else
  SEND_REQ_SLOW;
#endif

  return 0;
}

cl_int
pocl_network_copy_img2buf (uint32_t cq_id, remote_device_data_t *ddata,
                           uint32_t dst_remote_id, size_t dst_offset,
                           uint32_t src_remote_id,
                           const size_t *__restrict__ const origin,
                           const size_t *__restrict__ const region,
                           network_command_callback cb, void *arg,
                           _cl_command_node *node)
{
  REMOTE_SERV_DATA2;

  CREATE_ASYNC_NETCMD;

  REQUEST (CopyImage2Buffer);
  req->cq_id = cq_id;
  req->obj_id = src_remote_id;

  req->m.copy_img2buf.origin.x = origin[0];
  req->m.copy_img2buf.origin.y = origin[1];
  req->m.copy_img2buf.origin.z = origin[2];
  req->m.copy_img2buf.region.x = region[0];
  req->m.copy_img2buf.region.y = region[1];
  req->m.copy_img2buf.region.z = region[2];
  req->m.copy_img2buf.dst_offset = dst_offset;
  req->m.copy_img2buf.dst_buf_id = dst_remote_id;

  TP_COPY_IMAGE_2_BUF (req->msg_id, ddata->local_did, cq_id,
                       node->sync.event.event->id);

  SEND_REQ_FAST;

  return 0;
}

cl_int
pocl_network_read_image_rect (uint32_t cq_id, remote_device_data_t *ddata,
                              uint32_t src_remote_id,
                              const size_t *__restrict__ const origin,
                              const size_t *__restrict__ const region, void *p,
                              size_t size, network_command_callback cb,
                              void *arg, _cl_command_node *node)
{
  REMOTE_SERV_DATA2;
  assert (size > 0);

  CREATE_ASYNC_NETCMD;

  REQUEST (ReadImageRect);
  req->cq_id = cq_id;
  req->obj_id = src_remote_id;

  /* REPLY */
  netcmd->rep_extra_data = p;
  netcmd->rep_extra_size = size;

  req->m.read_image_rect.origin.x = origin[0];
  req->m.read_image_rect.origin.y = origin[1];
  req->m.read_image_rect.origin.z = origin[2];
  req->m.read_image_rect.region.x = region[0];
  req->m.read_image_rect.region.y = region[1];
  req->m.read_image_rect.region.z = region[2];
  req->m.read_image_rect.host_bytes = size;

  TP_READ_IMAGE_RECT (req->msg_id, ddata->local_did, cq_id,
                      node->sync.event.event->id);

  SEND_REQ_FAST;

  return 0;
}

cl_int
pocl_network_fill_image (uint32_t cq_id, remote_device_data_t *ddata,
                         uint32_t image_id,
                         const size_t *__restrict__ const origin,
                         const size_t *__restrict__ const region,
                         cl_uint4 *fill_pixel, network_command_callback cb,
                         void *arg, _cl_command_node *node)
{
  REMOTE_SERV_DATA2;
  assert (fill_pixel);

  CREATE_ASYNC_NETCMD;

  ID_REQUEST (FillImageRect, image_id);
  req->cq_id = cq_id;

  req->m.fill_image.origin.x = origin[0];
  req->m.fill_image.origin.y = origin[1];
  req->m.fill_image.origin.z = origin[2];
  req->m.fill_image.region.x = region[0];
  req->m.fill_image.region.y = region[1];
  req->m.fill_image.region.z = region[2];

  netcmd->req_extra_data = (void *)fill_pixel;
  netcmd->req_extra_size = 16;

  TP_FILL_IMAGE (req->msg_id, ddata->local_did, cq_id,
                 node->sync.event.event->id);

  SEND_REQ_FAST;

  return 0;
}
