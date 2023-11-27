/* peer.hh - class representing a server-server connection

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

#ifndef PEER_HH
#define PEER_HH

#include <thread>

#include "common.hh"
#include "guarded_queue.hh"

#ifdef ENABLE_RDMA
#include "rdma.hh"
#include "rdma_request_th.hh"
#endif

#include "request_th.hh"
#include "traffic_monitor.hh"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

class Peer {
  uint64_t id;
  uint32_t handler_id;
  std::atomic_int fd;
  VirtualContextBase *ctx;
  ExitHelper *eh;

  GuardedQueue<Request *> out_queue;

  RequestQueueThreadUPtr reader;
  void writerThread();
  std::thread writer;
  TrafficMonitor *netstat;
#ifdef ENABLE_RDMA
  std::shared_ptr<RdmaConnection> rdma;
  RdmaRequestThreadUPtr rdma_reader;
  std::mutex local_regions_mutex;
  std::unordered_map<uint32_t, RdmaBufferData> local_memory_regions;
  std::mutex remote_regions_mutex;
  std::unordered_map<uint32_t, RdmaRemoteBufferData> remote_memory_regions;
  GuardedQueue<Request *> rdma_out_queue;
  void rdmaWriterThread();
  std::thread rdma_writer;
#endif
public:
#ifdef ENABLE_RDMA
  Peer(uint64_t id, uint32_t handler_id, VirtualContextBase *ctx,
       ExitHelper *eh, int fd, TrafficMonitor *tm,
       std::shared_ptr<RdmaConnection> conn);
#else
  Peer(uint64_t id, uint32_t handler_id, VirtualContextBase *ctx,
       ExitHelper *eh, int fd, TrafficMonitor *tm);
#endif
  ~Peer();

  void pushRequest(Request *r);

#ifdef ENABLE_RDMA
  bool rdmaRegisterBuffer(uint32_t id, char *buf, size_t size);
  void rdmaUnregisterBuffer(uint32_t id);
  void notifyBufferRegistration(uint32_t buf_id, uint32_t rkey, uint64_t vaddr);
#endif
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
