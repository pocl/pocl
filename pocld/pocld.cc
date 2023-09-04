/* pocld.cc - starting point and "main" loop of pocld

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

#include <algorithm>
#include <cassert>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <errno.h>
#include <sstream>
#include <string>
#include <unistd.h>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>

#ifdef __linux__
#include <dlfcn.h>
#include <ifaddrs.h>
#include <libgen.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/time.h>
#endif

#include "pocl_debug.h"
#include "pocl_remote.h"
#include "pocld_config.h"

#include "cmdline.h"
#include "common.hh"
#include "virtual_cl_context.hh"

#ifdef ENABLE_RDMA
#include "guarded_queue.hh"
#include "rdma.hh"
#endif

VirtualContextBase *createVirtualContext(client_connections_t conns,
                                         ClientHandshake_t handshake);
void startVirtualContextMainloop(VirtualContextBase *ctx);

#define PERROR_CHECK(cond, str)                                                \
  do {                                                                         \
    if (cond) {                                                                \
      POCL_MSG_ERR("%s: %s\n", str, strerror(errno));                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

/***************************************************************************/
/***************************************************************************/
/***************************************************************************/

in_addr_t find_default_ip_address() {
  struct in_addr listen_addr;
  listen_addr.s_addr = inet_addr("127.0.0.1");

#ifdef __linux__
  struct ifaddrs *ifa = NULL;
  int err = getifaddrs(&ifa);

  if (err == 0 && ifa) {
    struct ifaddrs *p;

    for (p = ifa; p != NULL; p = p->ifa_next) {
      if ((p->ifa_flags & IFF_UP) == 0)
        continue;
      if (p->ifa_flags & IFF_LOOPBACK)
        continue;
      if ((p->ifa_flags & IFF_RUNNING) == 0)
        continue;

      struct sockaddr *saddr = p->ifa_addr;
      if (saddr->sa_family != AF_INET)
        continue;

      struct sockaddr_in *saddr_in = (struct sockaddr_in *)saddr;
      if (saddr_in->sin_addr.s_addr == 0)
        continue;
      else {
        listen_addr.s_addr = saddr_in->sin_addr.s_addr;
        break;
      }
    }
    freeifaddrs(ifa);
  } else
    POCL_MSG_ERR("getifaddrs() failed or returned no data.\n");

#endif
  return listen_addr.s_addr;
}

int listen_peers(void *data) {
  peer_listener_data_t *d = (peer_listener_data_t *)data;

  int listen_sock = socket(AF_INET, SOCK_STREAM, 0);
  PERROR_CHECK((listen_sock < 0), "peer listener socket");

  struct sockaddr_in listen_addr = {};
  listen_addr.sin_family = AF_INET;
  listen_addr.sin_port = htons(d->port);
  // TODO: make configurable
  listen_addr.sin_addr.s_addr = inet_addr("0.0.0.0");

  int one = 1;
#ifdef SO_REUSEADDR
  if (setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)))
    POCL_MSG_ERR("peer listener: failed to set REUSEADDR on socket\n");
#endif
  if (setsockopt(listen_sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)))
    POCL_MSG_ERR("peer listener: failed to set NODELAY on socket\n");
#ifdef TCP_QUICKACK
  if (setsockopt(listen_sock, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one)))
    POCL_MSG_ERR("peer listener: failed to set QUICKACK on socket\n");
#endif
  int bufsize = 9 * 1024 * 1024;
  if (setsockopt(listen_sock, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize)))
    POCL_MSG_ERR("peer listener: failed to set RCVBUF on socket\n");
  if (setsockopt(listen_sock, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize)))
    POCL_MSG_ERR("peer listener: failed to set SNDBUF on socket\n");

  unsigned len = sizeof(listen_addr);
  PERROR_CHECK((bind(listen_sock, (struct sockaddr *)&listen_addr, len) < 0),
               "peer listener bind");
  PERROR_CHECK((listen(listen_sock, MAX_REMOTE_DEVICES) < 0),
               "peer listener listen");

#ifdef ENABLE_RDMA
  d->rdma_listener->listen(d->peer_rdma_port);
#endif

  do {
    struct sockaddr_in peer_addr;
    unsigned addr_size = sizeof(peer_addr);
    int peer_fd =
        accept(listen_sock, (struct sockaddr *)&peer_addr, &addr_size);
    assert(peer_fd != -1);
    assert(peer_addr.sin_family == AF_INET);
    char peer_ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &peer_addr.sin_addr, peer_ip_str, sizeof(peer_ip_str));

    std::string session(SESSION_ID_LENGTH, '\0');
    PERROR_CHECK(
        (read_full(peer_fd, session.data(), SESSION_ID_LENGTH, nullptr) < 0),
        "read incoming peer session id");
    std::string session_hex = hexstr(session);
    POCL_MSG_PRINT_GENERAL("PL: Incoming peer connection for session %s\n",
                           session_hex.c_str());
#ifdef ENABLE_RDMA
    // Accept RDMA connection
    POCL_MSG_PRINT_GENERAL("PL: Awaiting peer RDMAcm connection\n");
    std::shared_ptr<RdmaConnection> rdma_connection(
        new RdmaConnection(d->rdma_listener->accept()));
    // TODO: ensure that socket and rdma connections actually belong to the same
    // session
#endif

    std::unique_lock<std::mutex> l(d->mutex);
    if (d->incoming_peers.find(session_hex) == d->incoming_peers.end()) {
      POCL_MSG_WARN(
          "PL: Attempted peer connection to invalid session %s from %s\n",
          session_hex.c_str(), peer_ip_str);
      close(peer_fd);
    } else {
      POCL_MSG_PRINT_INFO("PL: Peer connection from %s to session %s\n",
                          peer_ip_str, session_hex.c_str());
      d->incoming_peers.at(session_hex)
          ->second.push_back({peer_fd
#ifdef ENABLE_RDMA
                              ,
                              rdma_connection
#endif
          });
      d->incoming_peers.at(session_hex)->first.notify_one();
#ifdef ENABLE_RDMA
      std::unique_lock<std::mutex> l2(d->vctx_map_mutex);
      d->peer_cm_id_to_vctx.insert(
          {*rdma_connection->id(), d->vctx_map.at(session_hex)});
#endif
    }
  } while (true);
}

#ifdef ENABLE_RDMA
template <typename T>
int listen_rdmacm_events(rdmacm::EventChannelPtr cm_channel,
                         std::unordered_map<rdma_cm_id *, T> &instances,
                         std::mutex &instances_mutex) {
  POCL_MSG_PRINT_GENERAL("RDMAcm event listener started\n");

  rdma_cm_event *event;
  do {
    event = cm_channel->getNext();

    if (event && (event->event == RDMA_CM_EVENT_DISCONNECTED ||
                  event->event == RDMA_CM_EVENT_DEVICE_REMOVAL)) {
      // get corresponding T from instances
      std::unique_lock<std::mutex> guard(instances_mutex);
      auto it = instances.find(event->id);
      if (it != instances.end()) {
        it->second->requestExit(0, "RDMAcm disconnect event");
      }
      rdma_ack_cm_event(event);
    }
  } while (event);

  return 0;
}
#endif

int main(int argc, char *argv[]) {

  struct gengetopt_args_info ai;
  memset(&ai, 0, sizeof(struct gengetopt_args_info));

  if (cmdline_parser(argc, argv, &ai) != 0) {
    exit(1);
  }

  const char *logfilter = NULL;
  if (ai.log_filter_arg)
    logfilter = ai.log_filter_arg;
  else
    logfilter = getenv("POCLD_LOG_FILTER");
  if (!logfilter)
    logfilter = "warn,err";
  pocl_stderr_is_a_tty = isatty(fileno(stderr));
  pocl_debug_messages_setup(logfilter);

  unsigned short command_port = 0;
  unsigned short stream_port = 0;
  unsigned short peer_port = 0;
#ifdef ENABLE_RDMA
  unsigned short rdma_port = 0;
  unsigned short peer_rdma_port = 0;
  std::unordered_map<rdma_cm_id *, VirtualContextBase *> cm_id_to_vctx;
  std::mutex cm_id_to_vctx_mutex;
#endif

  // to avoid cores at abort
  struct rlimit core_limit;
  core_limit.rlim_cur = 0;
  core_limit.rlim_max = 0;
  if (setrlimit(RLIMIT_CORE, &core_limit) != 0)
    POCL_MSG_ERR("setting rlimit_core failed!\n");

  // ignore sigpipe, because we want to properly shutdown on closed sockets
  signal(SIGPIPE, SIG_IGN);

  struct in_addr listen_addr;
  if (ai.address_arg)
    listen_addr.s_addr = inet_addr(ai.address_arg);
  else
    listen_addr.s_addr = find_default_ip_address();

  char *addr = inet_ntoa(listen_addr);

  if (ai.port_arg)
    command_port = (unsigned short)ai.port_arg;
  else
    command_port = DEFAULT_POCL_REMOTE_PORT;

  assert(command_port > 0);
  stream_port = command_port + 1;
  peer_port = command_port + 2;
#ifdef ENABLE_RDMA
  peer_rdma_port = command_port + 3;
  rdma_port = command_port + 4;
#endif

  struct sockaddr_in server_addr_command, server_addr_stream,
      client_addr_command, client_addr_stream;
  int server_sock_stream;
  int server_sock_command;
  server_sock_command = socket(AF_INET, SOCK_STREAM, 0);
  PERROR_CHECK((server_sock_command < 0), "command socket");
  server_sock_stream = socket(AF_INET, SOCK_STREAM, 0);
  PERROR_CHECK((server_sock_stream < 0), "stream socket");

  memset(&server_addr_command, 0, sizeof(server_addr_command));
  server_addr_command.sin_family = AF_INET;
  server_addr_command.sin_port = htons(command_port);
  server_addr_command.sin_addr.s_addr = listen_addr.s_addr;

  memset(&server_addr_stream, 0, sizeof(server_addr_stream));
  server_addr_stream.sin_family = AF_INET;
  server_addr_stream.sin_port = htons(stream_port);
  server_addr_stream.sin_addr.s_addr = listen_addr.s_addr;

  int one = 1;
#ifdef SO_REUSEADDR
  if (setsockopt(server_sock_command, SOL_SOCKET, SO_REUSEADDR, &one,
                 sizeof(one)))
    POCL_MSG_ERR("SERVER: failed to set REUSEADDR on command socket\n");
  if (setsockopt(server_sock_stream, SOL_SOCKET, SO_REUSEADDR, &one,
                 sizeof(one)))
    POCL_MSG_ERR("SERVER: failed to set REUSEADDR on stream socket\n");
#endif

  unsigned len = sizeof(server_addr_command);
  PERROR_CHECK((bind(server_sock_command,
                     (struct sockaddr *)&server_addr_command, len) < 0),
               "command bind");
  PERROR_CHECK((listen(server_sock_command, 10) < 0), "command listen");

  len = sizeof(server_addr_stream);
  PERROR_CHECK((bind(server_sock_stream, (struct sockaddr *)&server_addr_stream,
                     len) < 0),
               "stream bind");
  PERROR_CHECK((listen(server_sock_stream, 10) < 0), "stream listen");

  if (setsockopt(server_sock_command, IPPROTO_TCP, TCP_NODELAY, &one,
                 sizeof(one)))
    POCL_MSG_ERR("SERVER: failed to set NODELAY on command socket\n");
#ifdef TCP_QUICKACK
  if (setsockopt(server_sock_command, IPPROTO_TCP, TCP_QUICKACK, &one,
                 sizeof(one)))
    POCL_MSG_ERR("SERVER: failed to set QUICKACK on command socket\n");
#endif

  unsigned int keepalive = 1;
  int user_timeout = 10000;
  int bufsize = 4 * 1024;
  if (setsockopt(server_sock_command, SOL_SOCKET, SO_RCVBUF, &bufsize,
                 sizeof(bufsize)))
    POCL_MSG_ERR("SERVER: failed to set BUFSIZE on command socket\n");
  if (setsockopt(server_sock_command, SOL_SOCKET, SO_SNDBUF, &bufsize,
                 sizeof(bufsize)))
    POCL_MSG_ERR("SERVER: failed to set BUFSIZE on command socket\n");
  if (setsockopt(server_sock_command, SOL_SOCKET, SO_KEEPALIVE, &keepalive,
                 sizeof(keepalive)))
    POCL_MSG_ERR("SERVER: failed to set KEEPALIVE on command socket\n");
#if defined(TCP_SYNCNT)
  if (setsockopt(server_sock_command, IPPROTO_TCP, TCP_SYNCNT, &keepalive,
                 sizeof(keepalive)))
    POCL_MSG_ERR("SERVER: failed to set TCP_SYNCNT on command socket\n");
#endif
  if (setsockopt(server_sock_command, IPPROTO_TCP, TCP_KEEPCNT, &keepalive,
                 sizeof(keepalive)))
    POCL_MSG_ERR("SERVER: failed to set TCP_KEEPCNT on command socket\n");
  if (setsockopt(server_sock_command, IPPROTO_TCP, TCP_KEEPINTVL, &one,
                 sizeof(one)))
    POCL_MSG_ERR("SERVER: failed to set TCP_KEEPINTVL on command socket\n");
#if defined(TCP_KEEPIDLE)
  if (setsockopt(server_sock_command, IPPROTO_TCP, TCP_KEEPIDLE, &one,
                 sizeof(one)))
    POCL_MSG_ERR("SERVER: failed to set TCP_KEEPIDLE on command socket\n");
#endif
#if defined(TCP_USER_TIMEOUT)
  if (setsockopt(server_sock_command, IPPROTO_TCP, TCP_USER_TIMEOUT,
                 &user_timeout, sizeof(user_timeout)))
    POCL_MSG_ERR("SERVER: failed to set TCP_USER_TIMEOUT on command socket\n");
#elif defined(TCP_CONNECTIONTIMEOUT)
  if (setsockopt(server_sock_command, IPPROTO_TCP, TCP_CONNECTIONTIMEOUT,
                 &user_timeout, sizeof(user_timeout)))
    POCL_MSG_ERR(
        "SERVER: failed to set TCP_CONNECTIONTIMEOUT on command socket\n");
#endif

  bufsize = 4 * 1024 * 1024;
  if (setsockopt(server_sock_stream, SOL_SOCKET, SO_RCVBUF, &bufsize,
                 sizeof(bufsize)))
    POCL_MSG_ERR("SERVER: failed to set BUFSIZE on stream socket\n");
  if (setsockopt(server_sock_stream, SOL_SOCKET, SO_SNDBUF, &bufsize,
                 sizeof(bufsize)))
    POCL_MSG_ERR("SERVER: failed to set BUFSIZE on stream socket\n");
  if (setsockopt(server_sock_stream, SOL_SOCKET, SO_KEEPALIVE, &keepalive,
                 sizeof(keepalive)))
    POCL_MSG_ERR("SERVER: failed to set KEEPALIVE on stream socket\n");
#if defined(TCP_SYNCNT)
  if (setsockopt(server_sock_stream, IPPROTO_TCP, TCP_SYNCNT, &keepalive,
                 sizeof(keepalive)))
    POCL_MSG_ERR("SERVER: failed to set TCP_SYNCNT on stream socket\n");
#endif
  if (setsockopt(server_sock_stream, IPPROTO_TCP, TCP_KEEPCNT, &keepalive,
                 sizeof(keepalive)))
    POCL_MSG_ERR("SERVER: failed to set TCP_KEEPCNT on stream socket\n");
  if (setsockopt(server_sock_stream, IPPROTO_TCP, TCP_KEEPINTVL, &one,
                 sizeof(one)))
    POCL_MSG_ERR("SERVER: failed to set TCP_KEEPINTVL on stream socket\n");
#if defined(TCP_KEEPIDLE)
  if (setsockopt(server_sock_stream, IPPROTO_TCP, TCP_KEEPIDLE, &one,
                 sizeof(one)))
    POCL_MSG_ERR("SERVER: failed to set TCP_KEEPIDLE on stream socket\n");
#endif
#if defined(TCP_USER_TIMEOUT)
  if (setsockopt(server_sock_stream, IPPROTO_TCP, TCP_USER_TIMEOUT,
                 &user_timeout, sizeof(user_timeout)))
    POCL_MSG_ERR("SERVER: failed to set TCP_USER_TIMEOUT on stream socket\n");
#elif defined(TCP_CONNECTIONTIMEOUT)
  if (setsockopt(server_sock_stream, IPPROTO_TCP, TCP_CONNECTIONTIMEOUT,
                 &user_timeout, sizeof(user_timeout)))
    POCL_MSG_ERR(
        "SERVER: failed to set TCP_CONNECTIONTIMEOUT on stream socket\n");
#endif

  /***************************************************************************/

  peer_listener_data_t peer_listener_data = {peer_port
#ifdef ENABLE_RDMA
                                             ,
                                             peer_rdma_port
#endif
  };
#ifdef ENABLE_RDMA
  peer_listener_data.rdma_listener.reset(new RdmaListener());
  std::thread pl_rdma_event_th(
      listen_rdmacm_events<VirtualContextBase *>,
      peer_listener_data.rdma_listener->eventChannel(),
      std::ref(peer_listener_data.peer_cm_id_to_vctx),
      std::ref(peer_listener_data.peer_cm_id_to_vctx_mutex));
#endif
  std::thread peer_listener_thread(listen_peers, (void *)&peer_listener_data);

  /***************************************************************************/

#ifdef ENABLE_RDMA
  RdmaListener rdma_listener;
  POCL_MSG_PRINT_INFO("Listening for RDMAcm connections on port %d\n",
                      rdma_port);
  rdma_listener.listen(rdma_port);
  GuardedQueue<rdma_cm_event *> cm_event_queue;
  std::thread client_rdma_event_th(
      listen_rdmacm_events<VirtualContextBase *>, rdma_listener.eventChannel(),
      std::ref(cm_id_to_vctx), std::ref(cm_id_to_vctx_mutex));
#endif

  /***************************************************************************/

  int fd_stream, fd_command;
  pid_t server_pid;
  server_pid = getpid();

  POCL_MSG_PRINT_INFO(
      "SERVER: PID %d Listening on command=%s:%d, stream=%s:%d, peers=%s:%d"
#ifdef ENABLE_RDMA
      ", peer_rdma=%s:%d"
      ", rdma=%s:%d"
#endif
      "\n",
      (int)server_pid, addr, command_port, addr, stream_port, "0.0.0.0",
      peer_port
#ifdef ENABLE_RDMA
      ,
      "0.0.0.0", peer_rdma_port, "0.0.0.0", rdma_port
#endif
  );

  while (true) {
  RESTART1:
    fd_command = accept(server_sock_command,
                        (struct sockaddr *)&client_addr_command, &len);
    if (fd_command == -1) {
      int e = errno;

      // TODO what here ? otther error
      if (e == EAGAIN)
        goto RESTART1;
      else {
        POCL_MSG_ERR("accept(fd_command): %s\n", strerror(e));
        goto EXIT;
      }
    }

  RESTART2:
    fd_stream = accept(server_sock_stream,
                       (struct sockaddr *)&client_addr_stream, &len);
    if (fd_stream == -1) {
      int e = errno;

      if (e == EAGAIN)
        goto RESTART2;
      else {
        POCL_MSG_ERR("accept(fd_stream): %s\n", strerror(e));
        goto EXIT;
      }
    }

    POCL_MSG_PRINT_GENERAL("FD STREAM: %d  COMMAND: %d\n", fd_stream,
                           fd_command);

    ClientHandshake_t handshake;
    if (read_full(fd_command, &handshake, sizeof(ClientHandshake_t), nullptr) <
        sizeof(ClientHandshake_t)) {
      POCL_MSG_ERR("read client handshake: %s\n", strerror(errno));
      goto EXIT;
    }
    ClientHandshake_t hs_reply;
    memset(&hs_reply, 0, sizeof(ClientHandshake_t));
    hs_reply.peer_port = peer_port;
#ifdef ENABLE_RDMA
    hs_reply.rdma_supported = handshake.rdma_supported;
#endif
    uint8_t initial_session_id[SESSION_ID_LENGTH] = {0};
    client_connections_t connections = {};
    bool is_reconnecting = false;
    std::string id;
    std::string session_hex;

    if (memcmp(handshake.session_id, initial_session_id, SESSION_ID_LENGTH) ==
        0) {
      connections = register_new_session(fd_command, fd_stream, id);
      memcpy(hs_reply.session_id, id.data(), SESSION_ID_LENGTH);
      session_hex = hexstr(id);

      std::unique_lock<std::mutex> l(peer_listener_data.mutex);
      auto p = new std::pair<std::condition_variable,
                             std::vector<peer_connection_t>>();
      connections.incoming_peer_mutex = &peer_listener_data.mutex;
      connections.incoming_peer_queue = p;
      peer_listener_data.incoming_peers.insert({session_hex, p});
      POCL_MSG_PRINT_INFO("Registered new client session %s\n",
                          session_hex.c_str());
    } else {
      session_hex = hexstr(id);
      is_reconnecting = true;
      POCL_MSG_PRINT_INFO(
          "Attempting to attach client to existing session %s\n",
          session_hex.c_str());

      pass_new_client_fds(fd_command, fd_stream, session_hex);
      memcpy(hs_reply.session_id, handshake.session_id, SESSION_ID_LENGTH);
    }
    PERROR_CHECK(write_full(fd_command, &hs_reply, sizeof(ClientHandshake_t),
                            nullptr) != 0,
                 "client handshake reply");

    struct timeval tv;
    tv.tv_sec = 3;
    tv.tv_usec = 0;

    if (setsockopt(fd_command, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv,
                   sizeof tv))
      POCL_MSG_ERR("failed to set RCVTIMEO on command socket\n");
    if (setsockopt(fd_stream, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv,
                   sizeof tv))
      POCL_MSG_ERR("failed to set RCVTIMEO on stream socket\n");

#ifdef ENABLE_RDMA
    if (hs_reply.rdma_supported) {
      // Accept RDMA connection
      POCL_MSG_PRINT_GENERAL("Accepting client RDMAcm connection\n");

      connections.rdma.reset(new RdmaConnection(rdma_listener.accept()));
    }
#endif

    if (!is_reconnecting) {
      // Start virtual_cl_context thread
      VirtualContextBase *ctx = createVirtualContext(connections, hs_reply);
      std::thread t(startVirtualContextMainloop, ctx);
#ifdef ENABLE_RDMA
      if (hs_reply.rdma_supported) {
        {
          std::unique_lock<std::mutex> l(cm_id_to_vctx_mutex);
          cm_id_to_vctx.insert({*connections.rdma->id(), ctx});
        }
        {
          std::unique_lock<std::mutex> l(
              peer_listener_data.peer_cm_id_to_vctx_mutex);
          peer_listener_data.peer_cm_id_to_vctx.insert(
              {*connections.rdma->id(), ctx});
        }
      }
      {
        std::unique_lock<std::mutex> l(peer_listener_data.vctx_map_mutex);
        peer_listener_data.vctx_map.insert({session_hex, ctx});
      }
#endif
      // destroying the thread handle terminates the thread unless it is
      // detached
      t.detach();
    }
  }
EXIT:
  POCL_MSG_PRINT_INFO("SERVER exiting\n");
  close(server_sock_command);
  close(server_sock_stream);
  return 0;
}
