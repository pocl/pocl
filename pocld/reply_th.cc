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

#include "CL/cl.h"
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

  case MessageType_CreateCommandBufferReply:
    return "CreateCommandBufferReply";
  case MessageType_FreeCommandBufferReply:
    return "FreeCommandBufferReply";

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
  case MessageType_BarrierReply:
    return "BarrierReply";
  case MessageType_MarkerReply:
    return "MarkerReply";
  case MessageType_RunCommandBufferReply:
    return "RunCommandBufferReply";

  case MessageType_Failure:
    return "Failure";

  default:
    return "UNKNOWN";
  }
}

ReplyQueueThread::ReplyQueueThread(
    std::shared_ptr<Connection> OutboundConnection, VirtualContextBase *c,
    ExitHelper *e, const char *id_str)
    : Conn(OutboundConnection), ThreadIdentifier(id_str), virtualContext(c),
      eh(e) {
  IOThread = std::thread{&ReplyQueueThread::writeThread, this};
}

ReplyQueueThread::~ReplyQueueThread() {
  eh->requestExit(ThreadIdentifier.c_str(), 0);

  {
    std::unique_lock<std::mutex> Lock(IOMutex);
    IONotifier.notify_one();
  }

  {
    std::unique_lock<std::mutex> Lock(ConnectionGuard);
    ConnectionNotifier.notify_one();
  }

  IOThread.join();
}

void ReplyQueueThread::pushReply(Reply *Completed) {
  if (eh->exit_requested())
    return;

  std::unique_lock<std::mutex> Lock(IOMutex);
  IOInflight.push(Completed);
  IONotifier.notify_one();
}

void ReplyQueueThread::setConnection(
    std::shared_ptr<Connection> NewConnection) {
  std::unique_lock<std::mutex> Lock(ConnectionGuard);
  Conn = NewConnection;
  ConnectionNotifier.notify_one();
}

void ReplyQueueThread::writeThread() {
  size_t i = 0;
  while (1) {
    if (eh->exit_requested())
      break;

    {
      std::unique_lock<std::mutex> ConnectionLock(ConnectionGuard);
      if (Conn.get() == nullptr) {
        ConnectionNotifier.wait(ConnectionLock);
        continue;
      }
    }

    std::unique_lock<std::mutex> InflightLock(IOMutex);
    if (IOInflight.empty()) {
      IONotifier.wait(InflightLock);
    } else {
      Reply *Completed = IOInflight.front();
      InflightLock.unlock();

      EventTiming_t Timing{0, 0, 0, 0};

      cl_int Status =
          Completed->rep.failed ? Completed->rep.fail_details : CL_SUCCESS;
      if (Completed->event()) {
        Timing.queued = 0;
        Timing.submitted = 0;
        Timing.started = 0;
        Timing.completed = 0;

        // clGetEventInfo is NOT a synchronization mechanism and gives no
        // guarantees that everything related to the event is done, so
        // wait explicitly (should be instant since the event is already
        // signaled as complete)
        cl_int err = Completed->event.wait();
        assert(err == CL_SUCCESS ||
               err == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        Status = Completed->event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
#ifdef QUEUE_PROFILING
        uint64_t tmp;
        tmp = Completed->event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>(
            &err);
        if (err == CL_SUCCESS)
          Timing.queued = tmp;
        tmp = Completed->event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>(
            &err);
        if (err == CL_SUCCESS)
          Timing.submitted = tmp;
        tmp =
            Completed->event.getProfilingInfo<CL_PROFILING_COMMAND_START>(&err);
        if (err == CL_SUCCESS)
          Timing.started = tmp;
        tmp = Completed->event.getProfilingInfo<CL_PROFILING_COMMAND_END>(&err);
        if (err == CL_SUCCESS)
          Timing.completed = tmp;
#endif
      }

      // Change reply to FAILURE if the command has failed after submitting
      if (Status < CL_COMPLETE) {
        Completed->rep.failed = 1;
        Completed->rep.fail_details = Status;
        Completed->rep.message_type = MessageType_Failure;
      }

      ReplyMessageType t =
          static_cast<ReplyMessageType>(Completed->rep.message_type);

      auto now1 = std::chrono::steady_clock::now();
      Completed->write_start_timestamp_ns =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              now1.time_since_epoch())
              .count();

      Completed->rep.timing = Timing;
      Completed->rep.server_write_start_timestamp_ns =
          Completed->write_start_timestamp_ns;

      std::unique_lock<std::mutex> ConnectionLock(ConnectionGuard);
      if (Conn.get() == nullptr) {
        POCL_MSG_PRINT_REMOTE(
            "%s: Got messages to send but no connection, sleeping.\n",
            ThreadIdentifier.c_str());
        ConnectionNotifier.wait(ConnectionLock);
        continue;
      }

      POCL_MSG_PRINT_GENERAL(
          "%s: SENDING MESSAGE, ID: %" PRIu64 " TYPE: %s SIZE: %" PRIuS
          " EXTRA: %" PRIuS " FAILED: %" PRIu32 "\n",
          ThreadIdentifier.c_str(), uint64_t(Completed->rep.msg_id),
          reply_to_str(t), sizeof(ReplyMsg_t), Completed->extra_size,
          uint32_t(Completed->rep.failed));

      // WRITE REPLY
      if (Conn->writeFull(&Completed->rep, sizeof(ReplyMsg_t)) < 0) {
        Conn.reset();
        virtualContext->connectionLost();
        continue;
      }

      // TODO: handle reconnecting & resending when RDMA is used
      if (Completed->extra_size > 0 && !Completed->extra_data.empty()) {
        POCL_MSG_PRINT_INFO("%s: WRITING EXTRA: %" PRIuS " \n",
                            ThreadIdentifier.c_str(), Completed->extra_size);
        if (Conn->writeFull(Completed->extra_data.data(),
                            Completed->extra_size) < 0) {
          Conn.reset();
          virtualContext->connectionLost();
          continue;
        }
      }
      ConnectionLock.unlock();

      POCL_MSG_PRINT_GENERAL("%s: MESSAGE FULLY WRITTEN, ID: %" PRIu64 "\n",
                             ThreadIdentifier.c_str(),
                             uint64_t(Completed->rep.msg_id));

      TP_MSG_SENT(reply->rep.msg_id, reply->rep.did, reply->rep.failed,
                  reply->rep.message_type);

      if (Completed->event()) {
        virtualContext->notifyEvent(Completed->req->Body.event_id, Status);

        if (!Completed->req->Body.skip_peer_notify) {
          Request PeerNotice{};
          PeerNotice.Body.msg_id = Completed->rep.msg_id;
          PeerNotice.Body.event_id = Completed->req->Body.event_id;
          PeerNotice.Body.message_type = MessageType_NotifyEvent;
          POCL_MSG_WARN("Notify for %s %lu (%lu)\n", reply_to_str(t),
                        Completed->req->Body.event_id,
                        Completed->req->Body.msg_id);
          virtualContext->broadcastToPeers(PeerNotice);
        } else
          POCL_MSG_WARN("Skipping notify for %s %lu (%lu)\n", reply_to_str(t),
                        Completed->req->Body.event_id,
                        Completed->req->Body.msg_id);
      }

      // pop the successfully written reply from the queue
      InflightLock.lock();
      IOInflight.pop();
      InflightLock.unlock();

      delete Completed;
    }
  }

  POCL_MSG_PRINT_GENERAL("%s: Terminating\n", ThreadIdentifier.c_str());
}
