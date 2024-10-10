/* rdma_reply_th.hh - alternative ReplyThread that uses RDMA for all traffic

   Copyright (c) 2020-2023 Jan Solanti / Tampere University

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

#ifndef POCL_RDMA_WRITE_QUEUE_HH
#define POCL_RDMA_WRITE_QUEUE_HH

#include <memory>
#include <queue>

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#include "common.hh"
#include "rdma.hh"
#include "traffic_monitor.hh"
#include "virtual_cl_context.hh"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

class RdmaReplyThread {
  VirtualContextBase *virtualContext;
  std::thread io_thread;
  std::queue<Reply *> io_inflight;
  std::mutex io_mutex;
  std::condition_variable io_cond;
  ExitHelper *eh;
  std::string id_str;
  std::shared_ptr<TrafficMonitor> Netstat;

  std::shared_ptr<RdmaConnection> rdma;

  std::mutex *mem_regions_mutex;
  std::unordered_map<uint32_t, RdmaBufferData> *mem_regions;

  void rdmaWriterThread();

public:
  RdmaReplyThread(VirtualContextBase *c, ExitHelper *eh,
                  std::shared_ptr<TrafficMonitor> tm, const char *id_str,
                  std::shared_ptr<RdmaConnection> conn,
                  std::unordered_map<uint32_t, RdmaBufferData> *mem_regions,
                  std::mutex *mem_region_mutex);

  ~RdmaReplyThread();

  void pushReply(Reply *reply);
};

typedef std::unique_ptr<RdmaReplyThread> RdmaReplyThreadUPtr;

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
