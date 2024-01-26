/* daemon.cc - class representing an instance of the daemon process

   Copyright (c) 2024 Jan Solanti / Tampere University

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

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <set>
#include <sys/poll.h>
#include <unistd.h>

#include "pocl_networking.h"
#include "pocl_runtime_config.h"

#ifdef ENABLE_RDMA
#include "rdma.hh"
#endif

#include "daemon.hh"

#ifndef POLLRDHUP
#define POLLRDHUP 0
#endif
#define POLLFD_ERROR_BITS (POLLHUP | POLLERR | POLLNVAL | POLLRDHUP)

#define PERROR_CHECK(cond, str)                                                \
  do {                                                                         \
    if (cond) {                                                                \
      POCL_MSG_ERR("%s: %s\n", str, strerror(errno));                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

VirtualContextBase *createVirtualContext(client_connections_t conns,
                                         ClientHandshake_t handshake);
void startVirtualContextMainloop(VirtualContextBase *ctx);

int listen_peers(void *data) {
  peer_listener_data_t *d = (peer_listener_data_t *)data;

  int listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
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
  unsigned len = sizeof(listen_addr);
  PERROR_CHECK((bind(listen_sock, (struct sockaddr *)&listen_addr, len) < 0),
               "peer listener bind");
  PERROR_CHECK((listen(listen_sock, MAX_REMOTE_DEVICES) < 0),
               "peer listener listen");

#ifdef ENABLE_RDMA
  d->rdma_listener->listen(d->peer_rdma_port);
#endif

  do {
    struct sockaddr peer_addr;
    socklen_t addr_size = sizeof(peer_addr);
    /* NOTE: size argument must be initialized to length of actual size of the
     * addr argument */
    int peer_fd = accept(listen_sock, &peer_addr, &addr_size);
    assert(peer_fd != -1);
    std::string addr_string =
        describe_sockaddr((struct sockaddr *)&peer_addr, addr_size);
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

    if (setsockopt(peer_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)))
      POCL_MSG_ERR("peer listener: failed to set NODELAY on socket\n");
#ifdef TCP_QUICKACK
    if (setsockopt(peer_fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one)))
      POCL_MSG_ERR("peer listener: failed to set QUICKACK on socket\n");
#endif
    std::unique_lock<std::mutex> l(d->mutex);
    if (d->incoming_peers.find(session_hex) == d->incoming_peers.end()) {
      POCL_MSG_WARN(
          "PL: Attempted peer connection to invalid session %s from %s\n",
          session_hex.c_str(), addr_string.c_str());
      close(peer_fd);
    } else {
      POCL_MSG_PRINT_INFO(
          "PL: Peer connection from %s to session %s, fd_peer=%d\n",
          addr_string.c_str(), session_hex.c_str(), peer_fd);
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

PoclDaemon::~PoclDaemon() {
  if (client_sockets_th.joinable())
    client_sockets_th.join();
  if (peer_listener_th.joinable())
    peer_listener_th.join();
#ifdef ENABLE_RDMA
  if (client_rdma_event_th.joinable())
    client_rdma_event_th.join();
  if (pl_rdma_event_th.joinable())
    pl_rdma_event_th.join();
#endif
  for (auto &t : client_session_threads) {
    if (t.second.joinable())
      t.second.join();
  }
}

int PoclDaemon::launch(struct sockaddr_storage &base_addr,
                       socklen_t base_addrlen, struct ServerPorts &ports) {
  listen_ports = {ports};
  struct sockaddr_storage server_addr_command, server_addr_stream;
  memcpy(&server_addr_command, &base_addr, base_addrlen);
  if (server_addr_command.ss_family == AF_INET)
    ((struct sockaddr_in *)&server_addr_command)->sin_port =
        htons(ports.command);
  else if (server_addr_command.ss_family == AF_INET6)
    ((struct sockaddr_in6 *)&server_addr_command)->sin6_port =
        htons(ports.command);
  else {
    POCL_MSG_ERR("SERVER: unsupported socket address family\n");
    return -1;
  }
  clients_listen_command_fd =
      socket(server_addr_command.ss_family, SOCK_STREAM, IPPROTO_TCP);
  PERROR_CHECK((clients_listen_command_fd < 0), "command socket");

  memcpy(&server_addr_stream, &base_addr, base_addrlen);
  if (server_addr_stream.ss_family == AF_INET)
    ((struct sockaddr_in *)&server_addr_stream)->sin_port = htons(ports.stream);
  else if (server_addr_stream.ss_family == AF_INET6)
    ((struct sockaddr_in6 *)&server_addr_stream)->sin6_port =
        htons(ports.stream);
  else {
    POCL_MSG_ERR("SERVER: unsupported socket address family\n");
    return -1;
  }
  clients_listen_stream_fd =
      socket(server_addr_stream.ss_family, SOCK_STREAM, IPPROTO_TCP);
  PERROR_CHECK((clients_listen_stream_fd < 0), "stream socket");

  int one = 1;
#ifdef SO_REUSEADDR
  if (setsockopt(clients_listen_command_fd, SOL_SOCKET, SO_REUSEADDR, &one,
                 sizeof(one)))
    POCL_MSG_ERR("SERVER: failed to set REUSEADDR on command socket\n");
  if (setsockopt(clients_listen_stream_fd, SOL_SOCKET, SO_REUSEADDR, &one,
                 sizeof(one)))
    POCL_MSG_ERR("SERVER: failed to set REUSEADDR on stream socket\n");
#endif

  PERROR_CHECK(
      (bind(clients_listen_command_fd, (struct sockaddr *)&server_addr_command,
            base_addrlen) < 0),
      "command bind");
  pocl_remote_client_set_socket_options(clients_listen_command_fd, 4 * 1024, 1);
  PERROR_CHECK((listen(clients_listen_command_fd, 10) < 0), "command listen");

  PERROR_CHECK((bind(clients_listen_stream_fd,
                     (struct sockaddr *)&server_addr_stream, base_addrlen) < 0),
               "stream bind");
  pocl_remote_client_set_socket_options(clients_listen_stream_fd,
                                        4 * 1024 * 1024, 0);
  PERROR_CHECK((listen(clients_listen_stream_fd, 10) < 0), "stream listen");

  std::string addr_string =
      describe_sockaddr((struct sockaddr *)&base_addr, base_addrlen);

  peer_listener_data.port = listen_ports.peer;
#ifdef ENABLE_RDMA
  peer_listener_data.peer_rdma_port = listen_ports.peer_rdma;
  peer_listener_data.rdma_listener.reset(new RdmaListener);
  client_rdma_event_th = std::move(std::thread(
      listen_rdmacm_events<VirtualContextBase *>, rdma_listener.eventChannel(),
      std::ref(cm_id_to_vctx), std::ref(cm_id_to_vctx_mutex)));
  rdma_listener.listen(listen_ports.rdma);
  pl_rdma_event_th = std::move(
      std::thread(listen_rdmacm_events<VirtualContextBase *>,
                  peer_listener_data.rdma_listener->eventChannel(),
                  std::ref(peer_listener_data.peer_cm_id_to_vctx),
                  std::ref(peer_listener_data.peer_cm_id_to_vctx_mutex)));
#endif
  peer_listener_th =
      std::move(std::thread(listen_peers, (void *)&peer_listener_data));

  pid_t server_pid;
  server_pid = getpid();

  POCL_MSG_PRINT_INFO(
      "SERVER: PID %d Listening on command=%s:%d, stream=%s:%d, peers=%s:%d"
#ifdef ENABLE_RDMA
      ", peer_rdma=%s:%d, rdma=%s:%d"
#endif
      "\n",
      (int)server_pid, addr_string.c_str(), listen_ports.command,
      addr_string.c_str(), listen_ports.stream, "0.0.0.0", listen_ports.peer
#ifdef ENABLE_RDMA
      ,
      "0.0.0.0", listen_ports.peer_rdma, "0.0.0.0", listen_ports.rdma
#endif
  );

  client_sockets_th =
      std::move(std::thread(&PoclDaemon::readAllClientSocketsThread, this));

  return 0;
}

VirtualContextBase *PoclDaemon::performClientSetup(int command_fd,
                                                   int stream_fd) {
  ClientHandshake_t handshake;
  ClientHandshake_t hs_reply = {};
  hs_reply.peer_port = listen_ports.peer;
  uint8_t initial_session_id[SESSION_ID_LENGTH] = {0};
  client_connections_t connections = {};
  bool is_reconnecting = false;
  std::string id;
  std::string session_hex;
  VirtualContextBase *ctx = nullptr;

  if (read_full(command_fd, &handshake, sizeof(ClientHandshake_t), nullptr) <
      sizeof(ClientHandshake_t)) {
    goto HANDSHAKE_ERROR;
  }

  POCL_MSG_PRINT_GENERAL("Handshake on fd %d\n", command_fd);

#ifdef ENABLE_RDMA
  hs_reply.rdma_supported = handshake.rdma_supported;
#endif

  if (memcmp(handshake.session_id, initial_session_id, SESSION_ID_LENGTH) ==
      0) {
    connections = register_new_session(command_fd, stream_fd, id);
    memcpy(hs_reply.session_id, id.data(), SESSION_ID_LENGTH);
    session_hex = hexstr(id);

    std::unique_lock<std::mutex> l(peer_listener_data.mutex);
    auto p = new std::pair<std::condition_variable,
                           std::vector<peer_connection_t>>();
    connections.incoming_peer_mutex = &peer_listener_data.mutex;
    connections.incoming_peer_queue = p;
    peer_listener_data.incoming_peers.insert({session_hex, p});
    POCL_MSG_PRINT_INFO(
        "Registered new client session %s, fd_command=%d, fd_stream=%d\n",
        session_hex.c_str(), command_fd, stream_fd);
  } else {
    session_hex = hexstr(id);
    is_reconnecting = true;
    POCL_MSG_PRINT_INFO("Attempting to attach client to existing session %s\n, "
                        "fd_command=%d, fd_stream=%d",
                        session_hex.c_str(), command_fd, stream_fd);

    if (!pass_new_client_fds(command_fd, stream_fd, session_hex))
      goto HANDSHAKE_ERROR;
    memcpy(hs_reply.session_id, handshake.session_id, SESSION_ID_LENGTH);
  }

  if (write_full(command_fd, &hs_reply, sizeof(ClientHandshake_t), nullptr) <
      0) {
    goto HANDSHAKE_ERROR;
  }

#ifdef ENABLE_RDMA
  if (hs_reply.rdma_supported) {
    // Accept RDMA connection
    POCL_MSG_PRINT_GENERAL("Accepting client RDMAcm connection\n");

    connections.rdma.reset(new RdmaConnection(rdma_listener.accept()));
  }
#endif

  if (!is_reconnecting) {
    // Start virtual_cl_context thread
    ctx = createVirtualContext(connections, hs_reply);
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
    client_contexts.insert({session_hex, ctx});
    client_session_threads.insert({session_hex, std::move(t)});
  } else {
    ctx = client_contexts.at(session_hex);
  }
  return ctx;

HANDSHAKE_ERROR:
  POCL_MSG_ERR("Client handshake error, dropping client: %s\n",
               strerror(errno));
  return nullptr;
}

void PoclDaemon::readAllClientSocketsThread() {
  /* keep our listening sockets in the list to streamline the polling code */
  std::vector<int> open_client_fds = {clients_listen_command_fd,
                                      clients_listen_stream_fd};
  std::vector<VirtualContextBase *> socket_contexts = {nullptr, nullptr};
  std::vector<Request *> incomplete_requests = {nullptr, nullptr};
  bool fds_changed = true;
  std::vector<struct pollfd> pfds;

  int pending_client_command = 0;
  int pending_client_stream = 0;

  while (!exit_helper.exit_requested()) {
    /* Changes to the list of sockets should be relatively rare so let's
     * just rewrite the whole thing when it happens; it's a trivial
     * operation anyway. */
    if (fds_changed) {
      pfds.clear();
      pfds.reserve(open_client_fds.size());
      for (const int &fd : open_client_fds) {
        /* Unlike the other error flags POLLRDHUP is only returned if explicitly
         * polled for */
        pfds.push_back({fd, POLLIN | POLLRDHUP, 0});
      }
      fds_changed = false;
    }

    /* Just block forever. If/when a socket is closed - including the client
     * listeners - it triggers a POLLERR/POLLHUP/POLLRDHUP/POLLNVAL. */
    int num_event_fds = poll(pfds.data(), pfds.size(), -1);
    POCL_MSG_PRINT_GENERAL("Client socket poll returned %d fds with events\n",
                           num_event_fds);

    if (num_event_fds < 0) {
      int e = errno;
      exit_helper.requestExit(strerror(e), e);
      continue;
    } else if (num_event_fds == 0) {
      continue;
    }

    auto accept_new_connection = [&](struct pollfd &pfd, int bufsize,
                                     int is_fast, int &hack) {
      int ev = pfd.revents;
      pfd.revents = 0; // reset revents to 0 for the next polling round
      if (ev) {
        --num_event_fds;
        if (ev & POLLFD_ERROR_BITS) {
          exit_helper.requestExit("Client listener socket closed", 0);
          return false;
        } else if (ev & POLLIN) {
          /* Listening sockets return a POLLIN result when there are pending
           * connections to accept() */
          struct sockaddr client_address;
          socklen_t client_address_length = sizeof(client_address);
          /* NOTE: address length MUST be initialized to the size of the storage
           * given as the addr argument */
          int newfd = accept(pfd.fd, &client_address, &client_address_length);
          if (newfd > 0) {
            // XXX: rework (eliminate?) handshakes and just store new fds
            // return newly obtained fd to the caller via out parameter
            // See callsite below for detailed explanation.
            hack = newfd;
            // open_client_fds.push_back(newfd);
            // fds_changed = true;
            pocl_remote_client_set_socket_options(newfd, bufsize, 1);
            std::string client_address_string =
                describe_sockaddr(&client_address, client_address_length);
            POCL_MSG_PRINT_INFO("Accepted client %s connection from %s\n",
                                is_fast ? "command" : "stream",
                                client_address_string.c_str());
          }
        }
      }
      return true;
    };

    /* We always put our client listener sockets in the list first so let's
     * handle accepting incoming connections separately here. */
    /*
     * In order to associate the correct "command" and "stream" socket fds with
     * the same context, we hold onto the last opened fd of each listener socket
     * (via the out parameter in accept_new_connection out) and once we have
     * both, create a new context with them (or update and existing one,
     * depending on handshake). This is obviously a race condition and should be
     * eliminated, but that will require redesigning the handshake process so
     * until that happens please try to avoid several clients connecting at the
     * exact same moment.
     */
    /* clients_listen_command_fd */
    if (!accept_new_connection(pfds[0], 4 * 1024, 1, pending_client_command))
      continue;

    /* clients_listen_stream_fd */
    if (!accept_new_connection(pfds[1], 9 * 1024 * 1024, 0,
                               pending_client_stream))
      continue;

    /* XXX: would be nice to also handle new client RDMA connections here but
     * for now they are in performClientSetup */
    std::vector<int> dead_fds;
    if (pending_client_command && pending_client_stream) {
      /* XXX: client handshake needs to be reworked, maybe dropped entirely,
       * doing it here for now */
      VirtualContextBase *vctx = nullptr;
      if ((vctx = performClientSetup(pending_client_command,
                                     pending_client_stream))) {
        open_client_fds.push_back(pending_client_command);
        incomplete_requests.push_back(new Request);
        socket_contexts.push_back(vctx);
        open_client_fds.push_back(pending_client_stream);
        incomplete_requests.push_back(new Request);
        socket_contexts.push_back(vctx);
        fds_changed = true;
      } else {
        dead_fds.push_back(pending_client_command);
        dead_fds.push_back(pending_client_stream);
      }
      pending_client_command = 0;
      pending_client_stream = 0;
    }

    /* Incoming connections have been handled, take care of pending reads on
     * connected client sockets. */
    for (size_t i = 2; i < pfds.size() && num_event_fds > 0; ++i) {
      int ev = pfds[i].revents;
      pfds[i].revents = 0; /* reset to 0 for the next polling round */
      if (ev) {
        --num_event_fds;
        /* Collect dead fds but don't remove them from the list of open fds yet
         * lest the indices of pfds no logner match */
        if (ev & POLLFD_ERROR_BITS) {
          POCL_MSG_PRINT_GENERAL(
              "Poll says fd=%d is dead (0x%X), removing it.\n", pfds[i].fd, ev);
          dead_fds.push_back(pfds[i].fd);
          continue;
        }

        if (ev & POLLIN) {
          Request *r = incomplete_requests.at(i);
          if (r->read(pfds[i].fd)) {
            if (!r->fully_read)
              continue;

            // TODO: add session ID field to all messages
            // auto it = client_contexts.find(hexstring(r->req.session_id));
            VirtualContextBase *ctx = socket_contexts.at(i);
            if (ctx) {
              switch (r->req.message_type) {
              case MessageType_ConnectPeer:
              case MessageType_DeviceInfo:
              case MessageType_CreateBuffer:
              case MessageType_FreeBuffer:
              case MessageType_CreateCommandQueue:
              case MessageType_FreeCommandQueue:
              case MessageType_CreateSampler:
              case MessageType_FreeSampler:
              case MessageType_CreateImage:
              case MessageType_FreeImage:
              case MessageType_CreateKernel:
              case MessageType_FreeKernel:
              case MessageType_BuildProgramFromSource:
              case MessageType_BuildProgramFromBinary:
              case MessageType_BuildProgramFromSPIRV:
              case MessageType_BuildProgramWithBuiltins:
              case MessageType_FreeProgram:
              case MessageType_MigrateD2D:
              case MessageType_RdmaBufferRegistration:
              case MessageType_Shutdown: {
                ctx->nonQueuedPush(r);
                break;
              }
              case MessageType_ReadBuffer:
              case MessageType_WriteBuffer:
              case MessageType_CopyBuffer:
              case MessageType_FillBuffer:
              case MessageType_ReadBufferRect:
              case MessageType_WriteBufferRect:
              case MessageType_CopyBufferRect:
              case MessageType_CopyImage2Buffer:
              case MessageType_CopyBuffer2Image:
              case MessageType_CopyImage2Image:
              case MessageType_ReadImageRect:
              case MessageType_WriteImageRect:
              case MessageType_FillImageRect:
              case MessageType_RunKernel: {
                ctx->queuedPush(r);
                break;
              }
              case MessageType_NotifyEvent: {
                // TODO: this message should probably contain an actual
                // status... (see also rdma thread)
                ctx->notifyEvent(r->req.event_id, CL_COMPLETE);
                delete r;
                break;
              }

              default: {
                ctx->unknownRequest(r);
                break;
              }
              }

              /* r is now someone else's responsibility, simply "leak" it */
              incomplete_requests[i] = new Request;
            } else {
              POCL_MSG_ERR("Client sent request for nonexistent context, "
                           "closing connection\n");
              dead_fds.push_back(pfds[i].fd);
            }
          } else {
            POCL_MSG_ERR("Something went wrong while reading request, closing "
                         "connection\n");
            dead_fds.push_back(pfds[i].fd);
          }
        }
      }
    }

    // Collect vtxs that were used by connections to free those that are
    // not used by any connection when reconnect is not supported.
    std::set<VirtualContextBase *> DroppedVCtxs;
    /* reap dead fds */
    fds_changed |= !dead_fds.empty();
    size_t left_to_reap = dead_fds.size();
    for (size_t i = 0; i < open_client_fds.size() && left_to_reap; ++i) {
      int fd = open_client_fds[i];
      for (int d : dead_fds) {
        if (d == fd) {
          close(fd);
          std::swap(open_client_fds[i], open_client_fds.back());

          // Contexts can outlive their client connection (client may reconnect
          // later) so don't destroy them here, only remove them from the socket
          // bookkeeping list.

          open_client_fds.pop_back();
          std::swap(socket_contexts[i], socket_contexts.back());
          VirtualContextBase *vctx = socket_contexts.back();
          DroppedVCtxs.insert(vctx);
          socket_contexts.pop_back();

          std::swap(incomplete_requests[i], incomplete_requests.back());
          delete incomplete_requests.back();
          incomplete_requests.pop_back();
          --i;
          --left_to_reap;
        }
      }
    }

    if (!pocl_get_bool_option("POCLD_ALLOW_CLIENT_RECONNECT", 1)) {
      // Free unusued vctxs if reconnect is not enabled.
      for (auto VContext : DroppedVCtxs) {
        if (std::find(socket_contexts.begin(), socket_contexts.end(),
                      VContext) != socket_contexts.end())
          continue;
        VContext->requestExit(0,
                              "Client disconnected and reconnect not enabled.");
        delete VContext;
      }
    }
  }

  /* Close all remaining sockets, including the client listeners */
  std::for_each(open_client_fds.cbegin(), open_client_fds.cend(), &close);
}
