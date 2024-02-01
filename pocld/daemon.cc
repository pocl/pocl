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
#include <ifaddrs.h>
#include <net/if.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sys/poll.h>
#include <unistd.h>

#include "pocl_debug.h"
#include "pocl_networking.h"
#include "pocl_runtime_config.h"

#ifdef ENABLE_RDMA
#include "rdma.hh"
#endif

#include "daemon.hh"

#ifndef POLLRDHUP
#define PULLRDHUP 0
#endif
#define POLLFD_ERROR_BITS (POLLHUP | POLLERR | POLLNVAL | POLLRDHUP)

#define COMMAND_SOCKET_BUFSIZE (4 * 1024)
#define STREAM_SOCKET_BUFSIZE (4 * 1024 * 1024)

#define PERROR_CHECK(cond, str)                                                \
  do {                                                                         \
    if (cond) {                                                                \
      POCL_MSG_ERR("%s: %s\n", str, strerror(errno));                          \
      return 1;                                                                \
    }                                                                          \
  } while (0)

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

  POCL_MSG_PRINT_GENERAL("PL: listening for peers on port %d (tcp)"
#ifdef ENABLE_RDMA
                         " and %d (rdma)"
#endif
                         "\n",
                         d->port
#ifdef ENABLE_RDMA
                         ,
                         d->peer_rdma_port
#endif
  );

  do {
    struct sockaddr PeerAddress;
    socklen_t AddressSize = sizeof(PeerAddress);
    /* NOTE: size argument must be initialized to length of actual size of the
     * addr argument */
    int PeerFd = accept(listen_sock, &PeerAddress, &AddressSize);
    assert(PeerFd != -1);
    if (setsockopt(PeerFd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)))
      POCL_MSG_ERR("peer listener: failed to set NODELAY on socket\n");
#ifdef TCP_QUICKACK
    if (setsockopt(PeerFd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one)))
      POCL_MSG_ERR("peer listener: failed to set QUICKACK on socket\n");
#endif
    std::string addr_string =
        describe_sockaddr((struct sockaddr *)&PeerAddress, AddressSize);
    POCL_MSG_PRINT_GENERAL("PL: New peer connection from %s\n",
                           addr_string.c_str());

    ReplyMsg_t Rep;

#ifdef ENABLE_RDMA
    // Accept RDMA connection
    POCL_MSG_PRINT_GENERAL("PL: Awaiting peer RDMAcm connection\n");
    std::shared_ptr<RdmaConnection> rdma_connection(
        new RdmaConnection(d->rdma_listener->accept()));
    // TODO: ensure that socket and rdma connections actually belong to the same
    // session
#endif

    Request R;
    bool ReadError;
    do {
      ReadError = !R.read(PeerFd);
    } while (!R.IsFullyRead && !ReadError);
    if (ReadError) {
      close(PeerFd);
      continue;
    }

    std::string auth_hex =
        std::accumulate(R.req.authkey, R.req.authkey + AUTHKEY_LENGTH,
                        std::string(), hexdigits);
    POCL_MSG_PRINT_GENERAL("PL: Incoming peer connection for session %" PRIu64
                           "\n",
                           R.req.session);

    std::unique_lock<std::mutex> l(d->mutex);
    if (d->incoming_peers.find(R.req.session) == d->incoming_peers.end()) {
      POCL_MSG_WARN("PL: Attempted peer connection to invalid session %" PRIu64
                    " from %s\n",
                    R.req.session, addr_string.c_str());
      close(PeerFd);
    } else {
      POCL_MSG_PRINT_INFO("PL: Peer connection from %s to session %" PRIu64
                          ", fd_peer=%d\n",
                          addr_string.c_str(), R.req.session, PeerFd);
      ReplyMsg_t Rep = {};
      Rep.message_type = MessageType_PeerHandshakeReply;
      Rep.msg_id = R.req.msg_id;
      Rep.m.peer_handshake.peer_id = d->SessionPeerId.at(R.req.session);
      d->incoming_peers.at(R.req.session)
          ->second.push_back({PeerFd, R.req.m.peer_handshake.peer_id
#ifdef ENABLE_RDMA
                              ,
                              rdma_connection
#endif
          });
      d->incoming_peers.at(R.req.session)->first.notify_one();
#ifdef ENABLE_RDMA
      std::unique_lock<std::mutex> l2(d->vctx_map_mutex);
      d->peer_cm_id_to_vctx.insert(
          {*rdma_connection->id(), d->vctx_map.at(R.req.session)});
#endif
      l.unlock();
      write_full(PeerFd, &Rep, sizeof(Rep), nullptr);
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
  if (ClientPoller.joinable())
    ClientPoller.join();
  if (peer_listener_th.joinable())
    peer_listener_th.join();
#ifdef ENABLE_RDMA
  if (client_rdma_event_th.joinable())
    client_rdma_event_th.join();
  if (pl_rdma_event_th.joinable())
    pl_rdma_event_th.join();
#endif
  for (auto &t : ClientSessionThreads) {
    if (t.second.joinable())
      t.second.join();
  }
}

VirtualContextBase *createVirtualContext(PoclDaemon *d,
                                         ClientConnections_t conns,
                                         uint64_t session,
                                         CreateOrAttachSessionMsg_t &params);
void startVirtualContextMainloop(VirtualContextBase *ctx);

static std::string find_default_ip_address() {
  char *listen_addr = NULL;
  char buf[INET_ADDRSTRLEN];

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
        inet_ntop(AF_INET, &saddr_in->sin_addr, buf, INET_ADDRSTRLEN);
        listen_addr = strdup(buf);
        break;
      }
    }
    freeifaddrs(ifa);
  } else
    POCL_MSG_ERR("getifaddrs() failed or returned no data.\n");
#endif

  return std::string(listen_addr ? listen_addr : "127.0.0.1");
}

int PoclDaemon::launch(std::string ListenAddress, struct ServerPorts &Ports) {
#define PERROR_SKIP(cond, str)                                                 \
  do {                                                                         \
    if (cond) {                                                                \
      POCL_MSG_WARN("%s: %s\n", str, strerror(errno));                         \
      goto SOCKET_ERROR;                                                       \
    }                                                                          \
  } while (0)

  ListenPorts = {Ports};
  LastSessionId = 0;
  pid_t server_pid = getpid();
  int one = 1;
  int error = 0;
  std::string Address = ListenAddress;
  if (Address.empty())
    Address = find_default_ip_address();
  addrinfo *ResolvedAddress =
      pocl_resolve_address(Address.c_str(), ListenPorts.command, &error);
  addrinfo *ai = ResolvedAddress;
  NumListenFds = 0;
  for (addrinfo *ai = ResolvedAddress; ai; ai = ai->ai_next) {
    if (ai->ai_family != AF_INET && ai->ai_family != AF_INET6)
      continue;
    struct sockaddr *base_addr = ai->ai_addr;
    int base_addrlen = ai->ai_addrlen;
    std::string addr_string = describe_sockaddr(base_addr, base_addrlen);
    int listen_command_fd = 0;
    int listen_stream_fd = 0;
    struct sockaddr_storage server_addr_command, server_addr_stream;
    std::memcpy(&server_addr_command, base_addr, base_addrlen);
    if (server_addr_command.ss_family == AF_INET)
      ((struct sockaddr_in *)&server_addr_command)->sin_port =
          htons(ListenPorts.command);
    else if (server_addr_command.ss_family == AF_INET6)
      ((struct sockaddr_in6 *)&server_addr_command)->sin6_port =
          htons(ListenPorts.command);
    else {
      POCL_MSG_ERR("SERVER: unsupported socket address family %d\n",
                   (int)server_addr_command.ss_family);
      goto SOCKET_ERROR;
    }

    listen_command_fd =
        socket(server_addr_command.ss_family, SOCK_STREAM, IPPROTO_TCP);
    PERROR_SKIP((listen_command_fd < 0), "command socket");

    std::memcpy(&server_addr_stream, base_addr, base_addrlen);
    if (server_addr_stream.ss_family == AF_INET)
      ((struct sockaddr_in *)&server_addr_stream)->sin_port =
          htons(ListenPorts.stream);
    else if (server_addr_stream.ss_family == AF_INET6)
      ((struct sockaddr_in6 *)&server_addr_stream)->sin6_port =
          htons(ListenPorts.stream);
    else {
      POCL_MSG_ERR("SERVER: unsupported socket address family\n");
      goto SOCKET_ERROR;
    }
    listen_stream_fd =
        socket(server_addr_stream.ss_family, SOCK_STREAM, IPPROTO_TCP);
    PERROR_SKIP((listen_stream_fd < 0), "stream socket");

#ifdef SO_REUSEADDR
    if (setsockopt(listen_command_fd, SOL_SOCKET, SO_REUSEADDR, &one,
                   sizeof(one)))
      POCL_MSG_ERR("SERVER: failed to set REUSEADDR on command socket\n");
    if (setsockopt(listen_stream_fd, SOL_SOCKET, SO_REUSEADDR, &one,
                   sizeof(one)))
      POCL_MSG_ERR("SERVER: failed to set REUSEADDR on stream socket\n");
#endif

    PERROR_SKIP(
        (bind(listen_command_fd, (struct sockaddr *)&server_addr_command,
              base_addrlen) < 0),
        "command bind");
    pocl_remote_client_set_socket_options(listen_command_fd,
                                          COMMAND_SOCKET_BUFSIZE, 1);
    PERROR_SKIP((listen(listen_command_fd, 10) < 0), "command listen");

    PERROR_SKIP((bind(listen_stream_fd, (struct sockaddr *)&server_addr_stream,
                      base_addrlen) < 0),
                "stream bind");
    pocl_remote_client_set_socket_options(listen_stream_fd,
                                          STREAM_SOCKET_BUFSIZE, 0);
    PERROR_SKIP((listen(listen_stream_fd, 10) < 0), "stream listen");

#ifdef ENABLE_RDMA
    rdma_listener.listen(ListenPorts.rdma);
    client_rdma_event_th = std::move(
        std::thread(listen_rdmacm_events<VirtualContextBase *>,
                    rdma_listener.eventChannel(), std::ref(cm_id_to_vctx),
                    std::ref(cm_id_to_vctx_mutex)));
#endif

    POCL_MSG_PRINT_GENERAL("Server PID=%d listening for client connections on "
                           "%s, ports %d (command), %d (stream)"
#ifdef ENABLE_RDMA
                           " %d (rdma)"
#endif
                           "\n",
                           server_pid, addr_string.c_str(), Ports.command,
                           Ports.stream
#ifdef ENABLE_RDMA
                           ,
                           Ports.rdma
#endif
    );
    ++NumListenFds;
    OpenClientFds.push_back(listen_command_fd);
    ListenFdParams.push_back({COMMAND_SOCKET_BUFSIZE, 1});
    ++NumListenFds;
    OpenClientFds.push_back(listen_stream_fd);
    ListenFdParams.push_back({STREAM_SOCKET_BUFSIZE, 0});
    continue;
#undef PERROR_SKIP
  SOCKET_ERROR:
    if (listen_command_fd)
      close(listen_command_fd);
    if (listen_stream_fd)
      close(listen_stream_fd);
  }
  freeaddrinfo(ResolvedAddress);

  if (NumListenFds == 0) {
    POCL_MSG_ERR("Could not bind any socket address for '%s'\n",
                 Address.c_str());
    return -1;
  }

  peer_listener_data.port = ListenPorts.peer;
#ifdef ENABLE_RDMA
  peer_listener_data.peer_rdma_port = ListenPorts.peer_rdma;
  peer_listener_data.rdma_listener.reset(new RdmaListener);
  pl_rdma_event_th = std::move(
      std::thread(listen_rdmacm_events<VirtualContextBase *>,
                  peer_listener_data.rdma_listener->eventChannel(),
                  std::ref(peer_listener_data.peer_cm_id_to_vctx),
                  std::ref(peer_listener_data.peer_cm_id_to_vctx_mutex)));
#endif
  peer_listener_th =
      std::move(std::thread(listen_peers, (void *)&peer_listener_data));

  ClientPoller =
      std::move(std::thread(&PoclDaemon::readAllClientSocketsThread, this));

  return 0;
}

VirtualContextBase *PoclDaemon::performSessionSetup(int fd, Request *R) {
  std::array<uint8_t, AUTHKEY_LENGTH> authkey;
  VirtualContextBase *ctx = nullptr;
  ClientConnections_t connections = {};
  uint64_t session;
  std::string authkey_hex;

  std::random_device rd;
  std::default_random_engine dice(rd());
  std::uniform_int_distribution<uint8_t> dist(0, UINT8_MAX);
  for (int i = 0; i < authkey.size(); i++) {
    authkey[i] = dist(dice);
  }
  session = ++LastSessionId;
  SessionKeys.insert(std::make_pair(session, authkey));
  if (R->req.m.get_session.fast_socket) {
    connections.fd_command = fd;
    connections.fd_stream = -1;
  } else {
    connections.fd_command = -1;
    connections.fd_stream = fd;
  }

  ReplyMsg_t Reply = {};
  Reply.message_type = MessageType_CreateOrAttachSessionReply;
  Reply.m.get_session.session = session;
  Reply.m.get_session.peer_port = ListenPorts.peer;
  Reply.m.get_session.use_rdma = 0;
  memcpy(Reply.m.get_session.authkey, authkey.data(), AUTHKEY_LENGTH);
  authkey_hex =
      std::accumulate(authkey.begin(), authkey.end(), std::string(), hexdigits);

  std::unique_lock<std::mutex> l(peer_listener_data.mutex);
  auto p =
      new std::pair<std::condition_variable, std::vector<PeerConnection>>();
  connections.incoming_peer_mutex = &peer_listener_data.mutex;
  connections.incoming_peer_queue = p;
  peer_listener_data.incoming_peers.insert({session, p});
  peer_listener_data.SessionPeerId.insert(
      {session, uint64_t(R->req.m.get_session.peer_id)});
  POCL_MSG_PRINT_INFO("Registered new client session %" PRIu64 " %s\n", session,
                      authkey_hex.c_str());

  if (write_full(fd, &Reply, sizeof(Reply), nullptr) < 0) {
    POCL_MSG_ERR("Error sending session creation reply, destroying session\n");
    auto it = SessionKeys.find(session);
    if (it != SessionKeys.end())
      SessionKeys.erase(it);
    return nullptr;
  }

#ifdef ENABLE_RDMA
  if (Reply.m.get_session.use_rdma) {
    // Accept RDMA connection
    POCL_MSG_PRINT_GENERAL("Accepting client RDMAcm connection\n");

    connections.rdma.reset(new RdmaConnection(rdma_listener.accept()));
  }
#endif

  // Start virtual_cl_context thread
  ctx = createVirtualContext(this, connections, session, R->req.m.get_session);
#ifdef ENABLE_RDMA
  if (Reply.m.get_session.use_rdma) {
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
    peer_listener_data.vctx_map.insert({session, ctx});
  }
#endif
  ClientSessions.insert({session, ctx});
  ClientSessionThreads.insert(
      {session, std::move(std::thread(startVirtualContextMainloop, ctx))});
  return ctx;
}

void PoclDaemon::readAllClientSocketsThread() {
  std::vector<Request *> IncompleteRequests(NumListenFds, nullptr);
  // Collect vctxs that were used by connections to free those that are
  // not used by any connection when reconnect is not supported.
  std::set<VirtualContextBase *> DroppedVCtxs;
  // Collect fds of closed sockets and close them in bulk at the end of the
  // loop iteration in order to keep indices in sync
  std::vector<int> DroppedFds;
  bool FdsChanged = true;
  std::vector<struct pollfd> pfds;

  SocketContexts.clear();
  SocketContexts.resize(NumListenFds, nullptr);

  while (!exit_helper.exit_requested()) {
    /* Changes to the list of sockets should be relatively rare so let's
     * just rewrite the whole thing when it happens; it's a trivial
     * operation anyway. */
    if (FdsChanged) {
      pfds.clear();
      pfds.reserve(OpenClientFds.size());
      for (const int &fd : OpenClientFds) {
        /* Unlike the other error flags POLLRDHUP is only returned if explicitly
         * polled for */
        pfds.push_back({fd, POLLIN | POLLRDHUP, 0});
      }
      FdsChanged = false;
    }

    /* These *really* ought to stay consistent */
    assert(pfds.size() == OpenClientFds.size() &&
           SocketContexts.size() == OpenClientFds.size() &&
           IncompleteRequests.size() == OpenClientFds.size());

    /* Just block forever. If/when a socket is closed - including the client
     * listeners - it triggers a POLLERR/POLLHUP/POLLRDHUP/POLLNVAL. */
    int NumEventFds = poll(pfds.data(), pfds.size(), -1);
    POCL_MSG_PRINT_GENERAL("Client socket poll returned %d fds with events\n",
                           NumEventFds);

    if (NumEventFds < 0) {
      int e = errno;
      exit_helper.requestExit(strerror(e), e);
      continue;
    } else if (NumEventFds == 0) {
      continue;
    }

    auto accept_new_connection = [&](struct pollfd &pfd,
                                     struct SocketParams &Params) {
      int ev = pfd.revents;
      pfd.revents = 0; // reset revents to 0 for the next polling round
      if (ev) {
        --NumEventFds;
        if (ev & POLLFD_ERROR_BITS) {
          POCL_MSG_ERR("ev = 0x%X\n", ev);
          exit_helper.requestExit("Client listener socket closed", 0);
          return false;
        } else if (ev & POLLIN) {
          /* Listening sockets return a POLLIN result when there are pending
           * connections to accept() */
          struct sockaddr_storage client_address;
          socklen_t client_address_length = sizeof(client_address);
          /* NOTE: address length MUST be initialized to the size of the storage
           * given as the addr argument */
          int newfd = accept(pfd.fd, (struct sockaddr *)&client_address,
                             &client_address_length);
          if (newfd > 0) {
            OpenClientFds.push_back(newfd);
            SocketContexts.push_back(nullptr);
            IncompleteRequests.push_back(new Request());
            FdsChanged = true;
            /* XXX: Set these based on CreateOrAttachSession request instead? */
            pocl_remote_client_set_socket_options(newfd, Params.BufSize,
                                                  Params.IsFast);
            std::string client_address_string = describe_sockaddr(
                (struct sockaddr *)&client_address, client_address_length);
            POCL_MSG_PRINT_INFO("Accepted client %s connection from %s\n",
                                Params.IsFast ? "command" : "stream",
                                client_address_string.c_str());
          }
        }
      }
      return true;
    };

    /* We always put our client listener sockets in the list first so let's
     * handle accepting incoming connections separately here. */
    bool CriticalError = false;
    for (size_t i = 0; i < NumListenFds && !CriticalError && NumEventFds > 0;
         ++i) {
      CriticalError = !accept_new_connection(pfds[i], ListenFdParams[i]);
    }
    if (CriticalError)
      break;

    /* XXX: would be nice to also handle new client RDMA connections here but
     * for now they are very broken */

    /* Incoming connections have been handled, take care of pending reads on
     * connected client sockets. */
    for (size_t i = NumListenFds; i < pfds.size() && NumEventFds > 0; ++i) {
      int ev = pfds[i].revents;
      pfds[i].revents = 0; /* reset to 0 for the next polling round */
      if (ev) {
        --NumEventFds;
        /* Collect dead fds but don't remove them from the list of open fds yet
         * lest the indices of pfds no logner match */
        if (ev & POLLFD_ERROR_BITS) {
          POCL_MSG_PRINT_GENERAL(
              "Poll says fd=%d is dead (0x%X), removing it.\n", pfds[i].fd, ev);
          DroppedFds.push_back(pfds[i].fd);
          continue;
        }

        if (ev & POLLIN) {
          Request *R = IncompleteRequests.at(i);
          if (R->read(pfds[i].fd)) {
            if (R->IsFullyRead) {
              if (R->req.message_type == MessageType_CreateOrAttachSession) {
                int Fast = R->req.m.get_session.fast_socket;
                uint64_t Session = R->req.session;
                if (Session == 0) {
                  VirtualContextBase *ctx = performSessionSetup(pfds[i].fd, R);
                  if (ctx == nullptr) {
                    DroppedFds.push_back(pfds[i].fd);
                  } else {
                    SocketContexts[i] = ctx;
                  }
                } else {
                  std::unique_lock<std::mutex> L(SessionListMtx);
                  auto it = SessionKeys.find(Session);
                  if (it != SessionKeys.end()) {
                    if (std::memcmp(it->second.data(), R->req.authkey,
                                    AUTHKEY_LENGTH) == 0) {
                      auto cit = ClientSessions.find(Session);
                      std::optional<int> command_fd;
                      std::optional<int> stream_fd;
                      if (Fast)
                        command_fd = pfds[i].fd;
                      else
                        stream_fd = pfds[i].fd;
                      assert(cit != ClientSessions.end());
                      cit->second->updateSockets(command_fd, stream_fd);
                      SocketContexts[i] = cit->second;
                    }
                  }
                  L.unlock();
                  ReplyMsg_t Reply = {};
                  Reply.message_type = MessageType_CreateOrAttachSessionReply;
                  Reply.m.get_session.session = Session;
                  memcpy(Reply.m.get_session.authkey, R->req.authkey,
                         AUTHKEY_LENGTH);
                  write_full(pfds[i].fd, &Reply, sizeof(Reply), nullptr);
                }
                delete R;
              } else {
                std::unique_lock<std::mutex> LSessions(SessionListMtx);
                auto it = ClientSessions.find(R->req.session);
                VirtualContextBase *Ctx =
                    it == ClientSessions.end() ? nullptr : it->second;
                LSessions.unlock();
                if (Ctx) {
                  switch (R->req.message_type) {
                  case MessageType_ServerInfo:
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
                  case MessageType_CompileProgramFromSource:
                  case MessageType_CompileProgramFromSPIRV:
                  case MessageType_BuildProgramWithBuiltins:
                  case MessageType_LinkProgram:
                  case MessageType_FreeProgram:
                  case MessageType_MigrateD2D:
                  case MessageType_RdmaBufferRegistration:
                  case MessageType_Shutdown: {
                    Ctx->nonQueuedPush(R);
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
                    Ctx->queuedPush(R);
                    break;
                  }
                  case MessageType_NotifyEvent: {
                    // TODO: this message should probably contain an actual
                    // status... (see also rdma thread)
                    Ctx->notifyEvent(R->req.event_id, CL_COMPLETE);
                    delete R;
                    break;
                  }

                  default: {
                    Ctx->unknownRequest(R);
                    break;
                  }
                  }

                } else {
                  POCL_MSG_ERR(
                      "Client sent request for nonexistent context %" PRIu64
                      ", ignoring \n",
                      R->req.session);
                  delete R;
                }
              }

              /* R is now someone else's responsibility, simply "leak" it */
              IncompleteRequests[i] = new Request();
            }
          } else {
            POCL_MSG_ERR("Something went wrong while reading request, closing "
                         "connection\n");
            DroppedFds.push_back(pfds[i].fd);
          }
        }
      }
    }

    /* reap dead fds */
    FdsChanged |= !DroppedFds.empty();
    size_t left_to_reap = DroppedFds.size();
    for (size_t i = 0; left_to_reap; ++i) {
      int fd = OpenClientFds[i];
      for (int d : DroppedFds) {
        if (d == fd) {
          close(fd);

          // Contexts can outlive their client connection (client may reconnect
          // later) so don't destroy them here, only remove them from the socket
          // bookkeeping list. Swap to the end of the vector & pop instead of
          // directly removing to avoid unnecessarily copying the entire rest
          // of the vector around.

          std::swap(OpenClientFds[i], OpenClientFds.back());
          OpenClientFds.pop_back();

          std::swap(SocketContexts[i], SocketContexts.back());
          VirtualContextBase *vctx = SocketContexts.back();
          DroppedVCtxs.insert(vctx);
          SocketContexts.pop_back();

          std::swap(IncompleteRequests[i], IncompleteRequests.back());
          delete IncompleteRequests.back();
          IncompleteRequests.pop_back();
          --i;
          --left_to_reap;
        }
      }
    }
    DroppedFds.clear();

    if (!pocl_get_bool_option("POCLD_ALLOW_CLIENT_RECONNECT", 1)) {
      // Free unusued vctxs if reconnect is not enabled.
      for (auto VContext : DroppedVCtxs) {
        if (std::find(SocketContexts.begin(), SocketContexts.end(), VContext) !=
            SocketContexts.end())
          continue;
        VContext->requestExit(0,
                              "Client disconnected and reconnect not enabled.");
        delete VContext;
      }
    }
    DroppedVCtxs.clear();
  }

  /* Close all remaining sockets, including the client listeners */
  std::for_each(OpenClientFds.cbegin(), OpenClientFds.cend(), &close);
}
