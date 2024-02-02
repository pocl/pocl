/* peer_handler.hh -- class that supervises server-server connections

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

#ifndef POCL_REMOTE_PEER_HANDLER_HH
#define POCL_REMOTE_PEER_HANDLER_HH

#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common.hh"
#include "peer.hh"
#include "traffic_monitor.hh"
#include "virtual_cl_context.hh"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

class PeerHandler {
  uint32_t id;
  VirtualContextBase *ctx;
  std::thread listen_thread;
  ExitHelper *eh;

  std::mutex *NewConnectionsMutex;
  std::pair<std::condition_variable, std::vector<PeerConnection>>
      *NewConnections;

  std::mutex PeermapMutex;
  std::unordered_map<uint32_t, std::unique_ptr<Peer>> Peers;

  std::thread IncomingPeerHandler;
  void handleIncomingPeers();

  TrafficMonitor *netstat;

public:
  PeerHandler(
      uint32_t id, std::mutex *m,
      std::pair<std::condition_variable, std::vector<PeerConnection>> *incoming,
      VirtualContextBase *c, ExitHelper *eh, TrafficMonitor *tm);
  ~PeerHandler();

  cl_int connectPeer(uint64_t msg_id, const char *const address, uint16_t port,
                     uint64_t session,
                     const std::array<uint8_t, AUTHKEY_LENGTH> &authkey);
  void pushRequest(Request *r, uint32_t peer_id);
  void broadcast(const Request &r);
#ifdef ENABLE_RDMA
  GuardedQueue<rdma_cm_event *> cm_event_queue;
  bool rdmaRegisterBuffer(uint32_t id, char *buf, size_t size);
  void rdmaUnregisterBuffer(uint32_t id);
  void notifyRdmaBufferRegistration(uint32_t peer_id, uint32_t buf_id,
                                    uint32_t rkey, uint64_t vaddr);
#endif
};

#if 0
// TOFIX: peers cannot be freed without crashing, let them leak for now.
typedef std::unique_ptr<PeerHandler> PeerHandlerUPtr;
#else
typedef PeerHandler *PeerHandlerUPtr;
#endif

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
