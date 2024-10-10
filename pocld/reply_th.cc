/* reply_th.cc - pocld thread that sends command results back to the client

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
#include <mutex>
#include <queue>

#include "common.hh"
#include "messages.h"
#include "pocl_debug.h"
#include "reply_th.hh"
#include "tracing.h"

static const char *reply_to_str(ReplyMessageType type) {
  switch (type) {
  case MessageType_ServerInfoReply:
    return "ServerInfoReply";
  case MessageType_DeviceInfoReply:
    return "DeviceInfoReply";
  case MessageType_ConnectPeerReply:
    return "ConnectPeerReply";

  case MessageType_CreateBufferReply:
    return "CreateBufferReply";
  case MessageType_FreeBufferReply:
    return "FreeBufferReply";

  case MessageType_CreateCommandQueueReply:
    return "CreateCommandQueueReply";
  case MessageType_FreeCommandQueueReply:
    return "FreeCommandQueueReply";

  case MessageType_CreateSamplerReply:
    return "CreateSamplerReply";
  case MessageType_FreeSamplerReply:
    return "FreeSamplerReply";

  case MessageType_CreateImageReply:
    return "CreateImageReply";
  case MessageType_FreeImageReply:
    return "FreeImageReply";

  case MessageType_CreateKernelReply:
    return "CreateKernelReply";
  case MessageType_FreeKernelReply:
    return "FreeKernelReply";

  case MessageType_BuildProgramReply:
    return "BuildProgramReply";
  case MessageType_FreeProgramReply:
    return "FreeProgramReply";

  case MessageType_MigrateD2DReply:
    return "MigrateD2DReply";

  case MessageType_ReadBufferReply:
    return "ReadBufferReply";
  case MessageType_WriteBufferReply:
    return "WriteBufferReply";
  case MessageType_CopyBufferReply:
    return "CopyBufferReply";
  case MessageType_FillBufferReply:
    return "FillBufferReply";

  case MessageType_CopyImage2BufferReply:
    return "CopyImage2BufferReply";
  case MessageType_CopyBuffer2ImageReply:
    return "CopyBuffer2ImageReply";
  case MessageType_CopyImage2ImageReply:
    return "CopyImage2ImageReply";
  case MessageType_ReadImageRectReply:
    return "ReadImageRectReply";
  case MessageType_WriteImageRectReply:
    return "WriteImageRectReply";
  case MessageType_FillImageRectReply:
    return "FillImageRectReply";

  case MessageType_RunKernelReply:
    return "RunKernelReply";

  case MessageType_Failure:
    return "Failure";

  default:
    return "UNKNOWN";
  }
}

ReplyQueueThread::ReplyQueueThread(std::shared_ptr<Connection> Conn,
                                   VirtualContextBase *c, ExitHelper *e,
                                   const char *id_str)
    : Conn(Conn), virtualContext(c), eh(e), id_str(id_str) {
  io_thread = std::thread{&ReplyQueueThread::writeThread, this};
}

ReplyQueueThread::~ReplyQueueThread() {
  eh->requestExit(id_str.c_str(), 0);
  io_thread.join();
}

void ReplyQueueThread::pushReply(Reply *reply) {
  if (eh->exit_requested())
    return;

  {
    std::unique_lock<std::mutex> lock(io_mutex);
    io_inflight.push_back(reply);
  }

  io_cond.notify_one();
}

void ReplyQueueThread::setConnection(
    std::shared_ptr<Connection> NewConnection) {
  std::unique_lock<std::mutex> l(ConnectionGuard);
  Conn = NewConnection;
  ConnectionNotifier.notify_one();
}

void ReplyQueueThread::writeThread() {
  // XXX: Change into a ring buffer?
  std::queue<Reply *> backup;
  bool resending = false;
  size_t i = 0;
  while (1) {
  RETRY:
    if (eh->exit_requested())
      return;

    if (backup.empty())
      resending = false;
    std::unique_lock<std::mutex> lock(io_mutex);
    if ((io_inflight.size() > 0 || resending)) {
      Reply *reply = io_inflight[i];
      lock.unlock();

      // If we need to resend old messages, disregard the inflight queue
      if (resending) {
        reply = backup.front();
        POCL_MSG_PRINT_GENERAL("%s: Resending old replies, %" PRIuS
                               " remaining\n",
                               id_str.c_str(), backup.size());
      }

      cl_int status =
          (reply->event.get() == nullptr)
              ? CL_COMPLETE
              : reply->event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
      if (status <= CL_COMPLETE) {
        // clGetEventInfo is NOT a synchronization mechanism and gives no
        // guarantees that everything related to the event is done, so
        // wait explicitly (should be instant since the event is already
        // signaled as complete)

        EventTiming_t timing;

        if (reply->event()) {
          timing.queued = 0;
          timing.submitted = 0;
          timing.started = 0;
          timing.completed = 0;
          cl_int stat = reply->event.wait();
          // This should never actually happen but can't hurt to check
          if (status == CL_COMPLETE && stat != CL_SUCCESS)
            status = stat;
#ifdef QUEUE_PROFILING
          int err = CL_SUCCESS;
          uint64_t tmp;
          tmp =
              reply->event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>(&err);
          if (err == CL_SUCCESS)
            timing.queued = tmp;
          tmp =
              reply->event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>(&err);
          if (err == CL_SUCCESS)
            timing.submitted = tmp;
          tmp = reply->event.getProfilingInfo<CL_PROFILING_COMMAND_START>(&err);
          if (err == CL_SUCCESS)
            timing.started = tmp;
          tmp = reply->event.getProfilingInfo<CL_PROFILING_COMMAND_END>(&err);
          if (err == CL_SUCCESS)
            timing.completed = tmp;
#endif
        }

        // Change reply to FAILURE if the command has failed after submitting
        if (status < CL_COMPLETE) {
          reply->rep.failed = 1;
          reply->rep.fail_details = status;
          reply->rep.message_type = MessageType_Failure;
        }

        ReplyMessageType t =
            static_cast<ReplyMessageType>(reply->rep.message_type);

        auto now1 = std::chrono::system_clock::now();
        reply->write_start_timestamp_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                now1.time_since_epoch())
                .count();

        reply->rep.timing = timing;
        reply->rep.server_write_start_timestamp_ns =
            reply->write_start_timestamp_ns;

        std::unique_lock<std::mutex> l(ConnectionGuard);
        if (Conn.get() == nullptr) {
          POCL_MSG_PRINT_REMOTE(
              "%s: Got messages to send but no connection, sleeping.\n",
              id_str.c_str());
          ConnectionNotifier.wait(l);
          continue;
        }

        POCL_MSG_PRINT_GENERAL(
            "%s: SENDING MESSAGE, ID: %" PRIu64 " TYPE: %s SIZE: %" PRIuS
            " EXTRA: %" PRIuS " FAILED: %" PRIu32 "\n",
            id_str.c_str(), uint64_t(reply->rep.msg_id), reply_to_str(t),
            sizeof(ReplyMsg_t), reply->extra_size, uint32_t(reply->rep.failed));

        // WRITE REPLY
        CHECK_WRITE_RETRY(Conn->writeFull(&reply->rep, sizeof(ReplyMsg_t)),
                          id_str.c_str());

        // TODO: handle reconnecting & resending when RDMA is used
        if (reply->extra_size > 0 && !reply->extra_data.empty()) {
          POCL_MSG_PRINT_INFO("%s: WRITING EXTRA: %" PRIuS " \n",
                              id_str.c_str(), reply->extra_size);
          CHECK_WRITE_RETRY(
              Conn->writeFull(reply->extra_data.data(), reply->extra_size),
              id_str.c_str());
        }

        l.unlock();

        POCL_MSG_PRINT_GENERAL("%s: MESSAGE FULLY WRITTEN, ID: %" PRIu64 "\n",
                               id_str.c_str(), uint64_t(reply->rep.msg_id));

        TP_MSG_SENT(reply->rep.msg_id, reply->rep.did, reply->rep.failed,
                    reply->rep.message_type);

        if (resending) {
          delete reply;
          backup.pop();
        } else {
          if (reply->event.get() != nullptr) {
            virtualContext->notifyEvent(reply->req->req.event_id, status);
            Request peer_notice{};
            peer_notice.req.msg_id = reply->rep.msg_id;
            peer_notice.req.event_id = reply->req->req.event_id;
            peer_notice.req.message_type = MessageType_NotifyEvent;
            virtualContext->broadcastToPeers(peer_notice);
          }

          // swap the current element into last place and pop it off the vector
          lock.lock();
          if (i != io_inflight.size() - 1) {
            std::swap(io_inflight[i], io_inflight[io_inflight.size() - 1]);
          }
          io_inflight.pop_back();

          // move to next item (now in the old place of the current item)
          i = i % std::max(io_inflight.size(), (size_t)1);
          lock.unlock();

          backup.push(reply);
          if (backup.size() > 5) {
            delete backup.front();
            backup.pop();
          }
        }
      } else {
        lock.lock();
        i = (i + 1) % io_inflight.size();
        // lock is dropped after this
      }
    } else {
      auto now = std::chrono::system_clock::now();
      std::chrono::duration<unsigned long> d(3);
      now += d;
      i = 0;
      io_cond.wait_until(lock, now);
    }
  }
}
