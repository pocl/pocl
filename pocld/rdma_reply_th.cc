/* rdma_reply_th.cc - alternative ReplyThread implementation that uses RDMA for
   all communication

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

#include <cassert>

#include "rdma_reply_th.hh"

RdmaReplyThread::RdmaReplyThread(
    VirtualContextBase *c, ExitHelper *e, std::shared_ptr<TrafficMonitor> tm,
    const char *id_str, std::shared_ptr<RdmaConnection> conn,
    std::unordered_map<uint32_t, RdmaBufferData> *mem_regions,
    std::mutex *mem_regions_mutex)
    : virtualContext(c), eh(e), Netstat(tm), id_str(id_str), rdma(conn),
      mem_regions(mem_regions), mem_regions_mutex(mem_regions_mutex) {
  io_thread = std::thread{&RdmaReplyThread::rdmaWriterThread, this};
}

RdmaReplyThread::~RdmaReplyThread() {
  eh->requestExit(id_str.c_str(), 0);
  io_thread.join();
}

void RdmaReplyThread::pushReply(Reply *reply) {
  if (eh->exit_requested())
    return;
  {
    std::unique_lock<std::mutex> lock(io_mutex);
    io_inflight.push(reply);
  }

  io_cond.notify_one();
}

void RdmaReplyThread::rdmaWriterThread() {
  // TODO: have a bunch of these instead of waiting for all sends to complete
  // before handling the next reply
  //    - XXX: how to handle notifications? with a separate thread?
  RdmaBuffer<ReplyMsg_t> cmd_buf(rdma->protectionDomain(), 1);

  while (!eh->exit_requested()) {
    std::unique_lock<std::mutex> lock(io_mutex);
    if (!io_inflight.empty()) {
      Reply *reply = io_inflight.front();
      io_inflight.pop();
      lock.unlock();

      /************* set up command metadata transfer *************/
      cmd_buf.at(0) = reply->rep;
      WorkRequest cmd_wr =
          WorkRequest::Send(0, {{*cmd_buf}}, WorkRequest::Flags::Signaled);

      /************** set up buffer contents transfer *************/
      ptrdiff_t src_offset = transfer_src_offset(reply->req->Body);
      uint64_t data_size = transfer_size(reply->req->Body);

      POCL_MSG_PRINT_GENERAL(
          "%s: RDMA WRITE FOR MESSAGE ID: %" PRIu64 ", SIZE: %" PRIu64 "\n",
          id_str.c_str(), uint64_t(reply->rep.msg_id), data_size);
      RdmaBufferData meta;
      {
        std::unique_lock<std::mutex> l(*mem_regions_mutex);
        auto it = mem_regions->find(reply->req->Body.obj_id);
        if (it == mem_regions->end()) {
          POCL_MSG_ERR("%s: ERROR: no RDMA memory region for buffer %" PRIu32
                       "\n",
                       id_str.c_str(), uint32_t(reply->req->Body.obj_id));
          eh->requestExit("RDMA transfer requested on unregistered buffer",
                          112);
          return;
        }
        meta = it->second;
      }

      RdmaRemoteBufferData remote{0, 0};
      assert(!"not supported yet");

      /* 0 is a special value that means 2^31 in the sg_list. Messages of >2^31
       * bytes are not allowed with RDMA_WRITE. */
      uint32_t write_size = data_size >= 0x80000000 ? 0 : uint32_t(data_size);
      std::vector<ScatterGatherEntry> sg_list;
      if (data_size > 0)
        sg_list.push_back({meta.shadow_region, 0, write_size});
      WorkRequest data_wr = WorkRequest::RdmaWrite(
          0, std::move(sg_list), remote.address + src_offset, remote.rkey);
      for (uint64_t i = 1; i * 0x80000000 < data_size; ++i) {
        uint64_t offset = i * 0x80000000;
        uint64_t remain = data_size - offset;
        write_size = remain >= 0x80000000 ? 0 : uint32_t(remain);
        data_wr.chain(WorkRequest::RdmaWrite(
            0, {{meta.shadow_region, (ptrdiff_t)offset, write_size}},
            remote.address + src_offset + offset, remote.rkey));
      }
      data_wr.chain(std::move(cmd_wr));

      /******************* submit work requests *******************/
      if (Netstat.get())
        Netstat->txSubmitted(sizeof(ReplyMsg_t) + data_size);
      try {
        rdma->post(data_wr);
      } catch (const std::runtime_error &e) {
        POCL_MSG_ERR("%s: RDMA ERROR: %s\n", id_str.c_str(), e.what());
        eh->requestExit("Posting RDMA send request failed", -1);
        delete reply;
        break;
      }

      /************ await work completion notification ************/

      try {
        ibverbs::WorkCompletion wc = rdma->awaitSendCompletion();
      } catch (const std::runtime_error &e) {
        POCL_MSG_ERR("%s: RDMA event polling error: %s\n", id_str.c_str(),
                     e.what());
        eh->requestExit("RDMA event polling failure", -1);
        delete reply;
        break;
      }

      if (Netstat.get())
        Netstat->txConfirmed(sizeof(ReplyMsg_t) + data_size);

      virtualContext->notifyEvent(reply->req->Body.event_id, CL_COMPLETE);
      Request peer_notice{};
      peer_notice.Body.msg_id = reply->rep.msg_id;
      peer_notice.Body.event_id = reply->req->Body.event_id;
      peer_notice.Body.message_type = MessageType_NotifyEvent;
      virtualContext->broadcastToPeers(peer_notice);
      delete reply;
    } else {
      auto now = std::chrono::system_clock::now();
      std::chrono::duration<unsigned long> d(3);
      now += d;
      io_cond.wait_until(lock, now);
    }
  }
}
