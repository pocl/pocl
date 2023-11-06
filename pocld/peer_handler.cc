/* peer_handler.cc -- class that supervises server-server connections

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

#include <cassert>
#include <chrono>
#include <sstream>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/types.h>

#ifdef __linux__
#include <net/if.h>
#endif

#include "peer_handler.hh"
#include "pocl_networking.h"

#ifndef PERROR_CHECK
#define PERROR_CHECK(cond, str)                                                \
  do {                                                                         \
    if (cond) {                                                                \
      POCL_MSG_ERR("%s: %s\n", str, strerror(errno));                          \
      eh->requestExit("PHL Error", 1);                                         \
      return;                                                                  \
    }                                                                          \
  } while (0)
#endif

#ifndef PERROR_CHECK404
#define PERROR_CHECK404(cond, str)                                             \
  do {                                                                         \
    if (cond) {                                                                \
      POCL_MSG_ERR("%s: %s\n", str, strerror(errno));                          \
      eh->requestExit("PH Error", 1);                                          \
      return -404;                                                             \
    }                                                                          \
  } while (0)
#endif

const char *request_to_str(RequestMessageType type);

PeerHandler::PeerHandler(uint32_t id, std::mutex *m,
                         std::pair<std::condition_variable,
                                   std::vector<peer_connection_t>> *incoming,
                         VirtualContextBase *c, ExitHelper *e,
                         TrafficMonitor *tm)
    : id(id), incoming_mutex(m), incoming_fds(incoming), ctx(c), eh(e),
      netstat(tm) {
  incoming_peer_handler =
      std::thread(&PeerHandler::handle_incoming_peers, this);
}

const void set_socket_options(int fd, const char *label) {
  int one = 1;
#ifdef SO_REUSEADDR
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)))
    POCL_MSG_ERR("%s: failed to set REUSEADDR on socket\n", label);
#endif
  if (setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)))
    POCL_MSG_ERR("%s: failed to set NODELAY on socket\n", label);
#ifdef TCP_QUICKACK
  if (setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof(one)))
    POCL_MSG_ERR("%s: failed to set QUICKACK on socket\n", label);
#endif
  int bufsize = 9 * 1024 * 1024;
  if (setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize)))
    POCL_MSG_ERR("%s: failed to set RCVBUF on socket\n", label);
  if (setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize)))
    POCL_MSG_ERR("%s: failed to set SNDBUF on socket\n", label);
}

cl_int PeerHandler::connectPeer(uint64_t msg_id, const char *const address,
                                std::string &session, uint16_t port) {
  POCL_MSG_PRINT_INFO("PH: connect to peer at %s:%d with session %s\n", address,
                      int(port), hexstr(session).c_str());

  int err = 0;
  addrinfo *ai = pocl_resolve_address(address, port, &err);
  if (err) {
    POCL_MSG_ERR("Failed to resolve peer address: %s\n", gai_strerror(err));
    return -404;
  }
  PERROR_CHECK404((err != 0), "getaddrinfo()");

  int peer_fd = socket(ai->ai_family, ai->ai_socktype, IPPROTO_TCP);
  PERROR_CHECK404((peer_fd < 0), "peer socket");

  set_socket_options(peer_fd, "PH_outgoing");

  PERROR_CHECK404(connect(peer_fd, ai->ai_addr, ai->ai_addrlen) < 0,
                  "unable to connect peer");
  freeaddrinfo(ai);
  CHECK_WRITE_RET(
      write_full(peer_fd, session.data(), SESSION_ID_LENGTH, netstat), "PH",
      -404);

#ifdef ENABLE_RDMA
  POCL_MSG_PRINT_GENERAL("PH: Attempting RDMAcm connection to %s:%d\n", address,
                         port + 1);
  std::shared_ptr<RdmaConnection> rdma;
  try {
    rdma.reset(new RdmaConnection(RdmaConnection::connect(address, port + 1)));
  } catch (const std::runtime_error &e) {
    POCL_MSG_ERR("PH: %s\n", e.what());
    return -404;
  }
#endif

  POCL_MSG_PRINT_GENERAL("PH: Peer handshake SEND\n");
  PeerHandshake_t request = {
      msg_id, static_cast<uint32_t>(MessageType_PeerHandshake), id};
  CHECK_WRITE_RET(
      write_full(peer_fd, &request, sizeof(PeerHandshake_t), netstat), "PH",
      -404);

  POCL_MSG_PRINT_GENERAL("PH: Peer handshake reply RECV\n");
  PeerHandshake_t reply = {};
  ssize_t readb;
  CHECK_READ_RET(readb,
                 read_full(peer_fd, &reply, sizeof(PeerHandshake_t), netstat),
                 "PH", -404);
  assert(static_cast<size_t>(readb) == sizeof(PeerHandshake_t));
  assert(reply.message_type ==
         static_cast<uint32_t>(MessageType_PeerHandshake));
  assert(reply.msg_id == request.msg_id);

  std::unique_lock<std::mutex> map_lock(peermap_mutex);
#ifdef ENABLE_RDMA
  peers[reply.peer_id].reset(
      new Peer(reply.peer_id, id, ctx, eh, peer_fd, netstat, rdma));
#else
  peers[reply.peer_id].reset(
      new Peer(reply.peer_id, id, ctx, eh, peer_fd, netstat));
#endif

  POCL_MSG_PRINT_INFO("PH: peer %" PRIu32 " is now connected at %s\n",
                      uint32_t(reply.peer_id), address);

  return CL_SUCCESS;
}

void PeerHandler::handle_incoming_peers() {
  while (!eh->exit_requested()) {
    std::unique_lock<std::mutex> l(*incoming_mutex);
    if (incoming_fds->second.empty()) {
      incoming_fds->first.wait_for(l, std::chrono::seconds(1));
      continue;
    }
    peer_connection_t conn = incoming_fds->second.back();
    incoming_fds->second.pop_back();
    l.unlock();

    POCL_MSG_PRINT_GENERAL("PHL: Peer handshake RECV\n");
    ssize_t readb;
    PeerHandshake_t request = {};
    CHECK_READ_RET(
        readb, read_full(conn.fd, &request, sizeof(PeerHandshake_t), netstat),
        "PHL", void());
    assert(static_cast<size_t>(readb) == sizeof(PeerHandshake_t));
    assert(request.message_type ==
           static_cast<uint32_t>(MessageType_PeerHandshake));

    POCL_MSG_PRINT_GENERAL("PHL: Peer handshake reply SEND\n");
    PeerHandshake_t reply = {
        request.msg_id, static_cast<uint32_t>(MessageType_PeerHandshakeReply),
        id};
    CHECK_WRITE_RET(
        write_full(conn.fd, &reply, sizeof(PeerHandshake_t), netstat), "PHL",
        void());

#ifdef ENABLE_RDMA
    Peer *p =
        new Peer(request.peer_id, id, ctx, eh, conn.fd, netstat, conn.rdma);
#else
    Peer *p = new Peer(request.peer_id, id, ctx, eh, conn.fd, netstat);
#endif
    std::unique_lock<std::mutex> map_lock(peermap_mutex);
    peers[request.peer_id].reset(p);
  }
}

void PeerHandler::pushRequest(Request *r, uint32_t peer_id) {
  std::unique_lock<std::mutex> lock(peermap_mutex);
  auto it = peers.find(peer_id);
  if (it != peers.end()) {
    lock.unlock();
    it->second->pushRequest(r);
  } else {
    POCL_MSG_ERR("Tried sending msg (ID: %" PRIu64
                 ") to nonexistent peer %" PRIu32 "\n",
                 uint64_t(r->req.msg_id), peer_id);
  }
}

void PeerHandler::broadcast(const Request &r) {
  std::unique_lock<std::mutex> lock(peermap_mutex);
  for (auto &it : peers) {
    // The writer thread deletes all requets so create a clone for every
    // peer out there... This could be rewritten to not require cloning
    // but that would require some additional signaling which would likely
    // make things even more confusing than they already are.
    Request *clone = new Request(r);
    it.second->pushRequest(clone);
  }
}

#ifdef ENABLE_RDMA
void PeerHandler::notifyRdmaBufferRegistration(uint32_t peer_id,
                                               uint32_t buf_id, uint32_t rkey,
                                               uint64_t vaddr) {
  std::unique_lock<std::mutex> lock(peermap_mutex);
  auto it = peers.find(peer_id);
  if (it != peers.end()) {
    lock.unlock();
    POCL_MSG_PRINT_GENERAL(
        "PH: Received RDMA registration info mem_obj=%" PRIu32 ", "
        "rkey=0x%08" PRIx32 ", vaddr=0x%p\n",
        buf_id, rkey, (void *)vaddr);
    it->second->notifyBufferRegistration(buf_id, rkey, vaddr);
  } else {
    POCL_MSG_ERR("PH: Tried sending RDMA registration notification to "
                 "nonexistent peer %" PRIu32 "\n",
                 peer_id);
  }
}

bool PeerHandler::rdmaRegisterBuffer(uint32_t id, char *ptr, size_t size) {
  std::unique_lock<std::mutex> lock(peermap_mutex);
  for (auto &it : peers) {
    if (!it.second->rdmaRegisterBuffer(id, ptr, size))
      return false;
  }
  return true;
}

void PeerHandler::rdmaUnregisterBuffer(uint32_t id) {
  std::unique_lock<std::mutex> lock(peermap_mutex);
  for (auto &it : peers) {
    it.second->rdmaUnregisterBuffer(id);
  }
}
#endif

PeerHandler::~PeerHandler() {
  eh->requestExit("PH Shutdown", 0);
  listen_thread.join();
  for (auto &t : peers) {
    t.second.reset();
  }
}
