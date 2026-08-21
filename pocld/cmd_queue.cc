/* cnd_queue.cc - a high level command queue wrapper

   Copyright (c) 2018 Michal Babej / Tampere University of Technology
   Copyright (c) 2019-2023 Jan Solanti / Tampere University
   Copyright (c) 2023 Pekka Jääskeläinen / Intel Finland Oy

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
#include <cstdint>
#include <mutex>
#include <vector>

#include "CL/cl.h"

#include "CL/opencl.hpp"
#include "cmd_queue.hh"
#include "common.hh"
#include "common_cl.hh"
#include "pocl_debug.h"
#include "reply_th.hh"
#include "shared_cl_context.hh"

#include "tracing.h"

CommandQueue::CommandQueue(SharedContextBase *b, uint32_t queue_id,
                           uint32_t did, ReplyQueueThread *s,
                           ReplyQueueThread *f)
    : backend(b), queue_id(queue_id), dev_id(did), write_slow(s),
      write_fast(f) {
  POCL_MSG_PRINT_GENERAL("CQ %" PRIu32 " DID: %" PRIu32 " CONST \n", queue_id,
                         did);
}

CommandQueue::~CommandQueue() {
  POCL_MSG_PRINT_GENERAL("CQ %" PRIu32 " DESTR \n", queue_id);
}

void CommandQueue::push(Request *request) {
  std::unique_lock<std::mutex> Lock(PendingMutex);
  request->LocalWaitlist = backend->remapWaitlist(
      request->ClientWaitlist.size(), request->ClientWaitlist.data(),
      request->Body.event_id);
  if (ReadyToRun(request)) {
    Lock.unlock();
    RunCommand(request);
  } else {
    Pending.push_back(request);
  }
}

void CommandQueue::notify(EventWithId Event) {
  std::vector<Request *> Runnable;
  std::unique_lock<std::mutex> Lock(PendingMutex);
  for (size_t i = 0; i < Pending.size();) {

    // Add real cl::Event to LocalWaitlist if needed and missing
    Request *Req = Pending.at(i);
    for (uint64_t Id : Req->ClientWaitlist) {
      if (Id == Event.first) {
        bool Added = false;
        for (cl::Event &Mapped : Req->LocalWaitlist) {
          if (Mapped() == Event.second()) {
            Added = true;
            break;
          }
        }
        if (!Added) {
          Req->LocalWaitlist.push_back(Event.second);
        }
      }
    }

    if (ReadyToRun(Req)) {
      Runnable.push_back(Req);
      Pending.erase(Pending.begin() + i);
    } else {
      ++i;
    }
  }

  for (Request *Req : Runnable) {
    RunCommand(Req);
  }
}

bool CommandQueue::ReadyToRun(Request *Req) {
  std::string DepString = "";
  for (uint64_t &ID : Req->ClientWaitlist) {
    if (!DepString.empty())
      DepString.push_back(',');
    DepString.append(std::to_string(ID));
  }

  bool Ready = (Req->LocalWaitlist.size() == Req->Body.waitlist_size);
  POCL_MSG_PRINT_EVENTS(
      "Event %lu %s ready to run with %lu/%lu dependencies [%s]\n",
      (unsigned long)Req->Body.event_id, Ready ? "is" : "not",
      (unsigned long)Req->LocalWaitlist.size(),
      (unsigned long)Req->ClientWaitlist.size(), DepString.c_str());
  return Ready;
}

namespace {
class ReplyHelper {
public:
  ReplyHelper() = delete;
  ReplyQueueThread *Queue;
  Reply *Cmd;
  static void Submit(cl_event, cl_int, void *user_data) {
    ReplyHelper *tmp = (ReplyHelper *)user_data;
    tmp->Queue->pushReply(tmp->Cmd);
    delete tmp;
  }
};

class QueuedPushHelper {
public:
  QueuedPushHelper() = delete;
  VirtualContextBase *TopLevel;
  Request *Cmd;
  static void Push(cl_event, cl_int, void *user_data) {
    QueuedPushHelper *tmp = (QueuedPushHelper *)user_data;
    tmp->TopLevel->queuedPush(tmp->Cmd);
    delete tmp;
  }
};
} // anonymous namespace

void CommandQueue::RunCommand(Request *request) {
  if (backend->alreadyProcessed(request->Body.event_id)) {
    delete request;
    return;
  }

  auto Now = std::chrono::steady_clock::now();
  uint64_t ProcessingStart =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Now.time_since_epoch())
          .count();
  Reply *reply = new Reply(request, ProcessingStart);
  int slow = 0;

  POCL_MSG_PRINT_GENERAL("CQ %" PRIu32 " DID %" PRIu32
                         " |||||||||| REQ QID %" PRIu32 " DID %" PRIu32 " \n",
                         queue_id, dev_id, uint32_t(request->Body.cq_id),
                         uint32_t(request->Body.did));
  if (request->Body.message_type == MessageType_MigrateD2D) {
    assert(dev_id == request->Body.did ||
           dev_id == request->Body.m.migrate.source_did);
  } else {
    assert(queue_id == request->Body.cq_id);
    assert(dev_id == request->Body.did);
  }

  // PROCESSS REQUEST, then PUSH REPLY to WRITE Q
  switch (request->Body.message_type) {

  case MessageType_MigrateD2D:
    MigrateMemObj(queue_id, request, reply);
    break;

  case MessageType_ReadBuffer:
    ReadBuffer(queue_id, request, reply);
#ifdef ENABLE_RDMA
    slow = !(backend->clientUsesRdma());
#else
    slow = 1;
#endif
    break;

  case MessageType_WriteBuffer:
    WriteBuffer(queue_id, request, reply);
    break;

  case MessageType_CopyBuffer:
    CopyBuffer(queue_id, request, reply);
    break;

  case MessageType_ReadBufferRect:
    ReadBufferRect(queue_id, request, reply);
#ifdef ENABLE_RDMA
    slow = !(backend->clientUsesRdma());
#else
    slow = 1;
#endif
    break;

  case MessageType_WriteBufferRect:
    WriteBufferRect(queue_id, request, reply);
    break;

  case MessageType_CopyBufferRect:
    CopyBufferRect(queue_id, request, reply);
    break;

  case MessageType_FillBuffer:
    FillBuffer(queue_id, request, reply);
    break;

  case MessageType_RunKernel:
    RunKernel(queue_id, request, reply);
    break;

  case MessageType_Barrier:
    Barrier(queue_id, request, reply);
    break;

  case MessageType_Marker:
    Marker(queue_id, request, reply);
    break;

  case MessageType_RunCommandBuffer:
    RunCommandBuffer(queue_id, request, reply);
    break;

    /*************************************************************************/

  case MessageType_FillImageRect:
    FillImage(queue_id, request, reply);
    break;

  case MessageType_ReadImageRect:
    ReadImageRect(queue_id, request, reply);
    break;

  case MessageType_WriteImageRect:
    WriteImageRect(queue_id, request, reply);
    break;

  case MessageType_CopyBuffer2Image:
    CopyBuffer2Image(queue_id, request, reply);
    break;

  case MessageType_CopyImage2Buffer:
    CopyImage2Buffer(queue_id, request, reply);
    break;

  case MessageType_CopyImage2Image:
    CopyImage2Image(queue_id, request, reply);
    break;

  default:
    assert(false && "unknown message type");
  }

  // IsMigrationExportDone is only set by the export phase and unset again by
  // the import phase to indicate whether the Request needs to be reissued for
  // the next phase or a reply should be sent to the client.
  if (request->Body.message_type == MessageType_MigrateD2D &&
      request->IsMigrationExportDone) {
    // "Leak" the request so we can scrap the Reply and resubmit the Request
    void *_ = reply->req.release();
    QueuedPushHelper *tmp =
        new QueuedPushHelper{backend->parentVCtx(), request};
    reply->event.setCallback(CL_COMPLETE, QueuedPushHelper::Push, tmp);
    delete reply;
    // TODO: handle failed exports
    return;
  }

  ReplyQueueThread *rqt = (slow ? write_slow : write_fast);
  if (reply->event()) {
    ReplyHelper *tmp = new ReplyHelper{rqt, reply};
    reply->event.setCallback(CL_COMPLETE, ReplyHelper::Submit, tmp);
  } else {
    rqt->pushReply(reply);
  }
}

/***********    CMD QUEUE    *******************/

/**********************************************************************/
/**********************************************************************/
/**********************************************************************/

void CommandQueue::MigrateMemObj(uint32_t queue_id, Request *req, Reply *rep) {
  MigrateD2DMsg_t &m = req->Body.m.migrate;
  EventTiming_t EvtTiming{};

  if (m.source_pid == req->Body.pid && m.source_peer_id == m.dest_peer_id) {
    // direct migration within single platform
    // TP_WRITE_BUFFER(req->Body.event_id, req->Body.client_did, queue_id,
    // req->Body.obj_id, m.size, CL_RUNNING);
    // direct mig within 1 platform
    RETURN_IF_ERR_CODE(backend->migrateMemObject(
        req->Body.event_id, queue_id, req->Body.obj_id, m.is_image, EvtTiming,
        req->LocalWaitlist, rep->event));
    // TP_WRITE_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
    // req->Body.obj_id, m.size, CL_FINISHED);
  } else {
#ifndef RDMA_USE_SVM
    void *host_ptr;
#ifdef ENABLE_RDMA
    host_ptr = backend->getRdmaShadowPtr(req->Body.obj_id);
    req->ExtraDataSize = m.size;
#else
    if (req->IsMigrationExportRequired)
      req->ExtraData.resize(m.size);

    assert(req->ExtraData.size() >= req->ExtraDataSize);
    host_ptr = req->ExtraData.data();
#endif

    if (req->IsMigrationExportRequired) {
      // begin export buffer data

      m.is_external = 1;
      req->IsMigrationExportDone = true;
      uint32_t DefaultQueueID = DEFAULT_QUE_ID + m.source_did;
      // The export event is registered locally as the canonical "migration"
      // event so additional migrations from this platform can be enqueued
      // immediately without waiting for the completion notification from the
      // destination.
      uint64_t ExportEventID = req->Body.event_id;
      if (m.is_image) {
        sizet_vec3 origin = {0, 0, 0};
        sizet_vec3 region = {m.width, m.height, m.depth};

        TP_READ_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, def_queue_id,
                           req->Body.obj_id, m.width, m.height, m.depth,
                           CL_RUNNING);
        RETURN_IF_ERR_CODE(backend->readImageRect(
            ExportEventID, DefaultQueueID, req->Body.obj_id, origin, region,
            host_ptr, m.size, EvtTiming, req->LocalWaitlist, rep->event));
        TP_READ_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, def_queue_id,
                           req->Body.obj_id, m.width, m.height, m.depth,
                           CL_FINISHED);
      } else {
        uint64_t content_size;

        TP_READ_BUFFER(req->Body.msg_id, req->Body.client_did, def_queue_id,
                       req->Body.obj_id, m.size, CL_RUNNING);
        RETURN_IF_ERR_CODE(backend->readBuffer(
            ExportEventID, DefaultQueueID, req->Body.obj_id, 0, m.size_id,
            m.size, 0, host_ptr, &content_size, EvtTiming, req->LocalWaitlist,
            rep->event));
        TP_READ_BUFFER(req->Body.msg_id, req->Body.client_did, def_queue_id,
                       req->Body.obj_id, m.size, CL_FINISHED);

        assert(content_size <= m.size);
        m.size = content_size;
#ifdef ENABLE_RDMA
        // RDMA does not use the Request's ExtraData
        req->ExtraDataSize = 0;
        req->ExtraData.clear();
#else
        req->ExtraDataSize = content_size;
        req->ExtraData.resize(content_size);
#endif
      }

      // end export buffer data
    } else {
      // begin import buffer data

      req->IsMigrationExportDone = false;
      if (m.is_image) {
        sizet_vec3 origin = {0, 0, 0};
        sizet_vec3 region = {m.width, m.height, m.depth};

        TP_WRITE_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                            req->Body.obj_id, m.width, m.height, m.depth,
                            CL_RUNNING);
        RETURN_IF_ERR_CODE(backend->writeImageRect(
            req->Body.event_id, queue_id, req->Body.obj_id, origin, region,
            host_ptr, req->ExtraDataSize, EvtTiming, req->LocalWaitlist,
            rep->event));
        TP_WRITE_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                            req->Body.obj_id, m.width, m.height, m.depth,
                            CL_FINISHED);
      } else {
        TP_WRITE_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
                        req->Body.obj_id, m.size, CL_RUNNING);
        RETURN_IF_ERR_CODE(backend->writeBuffer(
            req->Body.event_id, queue_id, req->Body.obj_id, 0, m.size, 0,
            host_ptr, EvtTiming, req->LocalWaitlist, rep->event));
        TP_WRITE_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
                        req->Body.obj_id, m.size, CL_FINISHED);
      }

      // end import buffer data
    }
#endif
  }

  replyOK(rep, EvtTiming, MessageType_MigrateD2DReply);
}

void CommandQueue::ReadBuffer(uint32_t queue_id, Request *req, Reply *rep) {
  ReadBufferMsg_t &m = req->Body.m.read;
  EventTiming_t evt_timing{};

  /*
      // TODO: this should be done AFTER readBuffer() has finished, because
      // here we're acting at enqueue time, but the actual buffer content size
     is known
      // at kernel execution time, which migth be much later than enqueue
      size_t content_size = 0;
      if (backend->hasBufferSize() &&
          backend->getBufferContentSize(req->Body.obj_id, content_size) ==
     CL_SUCCESS) { if (content_size < m.size) POCL_MSG_PRINT_INFO("clReadBuffer:
     using Content Size %" PRIuS " instead of Read Size %" PRIu32 " \n",
     content_size, m.size); else content_size = m.size; } else { content_size =
     m.size;
        }
  */
  rep->extra_size = m.size;
  char *host_ptr = nullptr;
#ifdef ENABLE_RDMA
  if (!backend->clientUsesRdma()) {
    rep->extra_data.resize(rep->extra_size);
    host_ptr = (char*)rep->extra_data.data();
  }
#else
  rep->extra_data.resize(rep->extra_size);
  host_ptr = (char*)rep->extra_data.data();
#endif

  TP_READ_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
                 req->Body.obj_id, m.size, CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->readBuffer(
      req->Body.event_id, queue_id, req->Body.obj_id, m.is_svm,
      m.content_size_id, m.size, m.src_offset, host_ptr, &m.size, evt_timing,
      req->LocalWaitlist, rep->event));
  TP_READ_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
                 req->Body.obj_id, m.size, CL_FINISHED);

  replyData(rep, evt_timing, MessageType_ReadBufferReply, m.size);
}

void CommandQueue::WriteBuffer(uint32_t queue_id, Request *req, Reply *rep) {
  WriteBufferMsg_t &m = req->Body.m.write;
  EventTiming_t evt_timing{};

#ifdef ENABLE_RDMA
  void *data = backend->clientUsesRdma() ? nullptr : req->ExtraData.data();
#else
  void *data = req->ExtraData.data();
#endif

  TP_WRITE_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
                  req->Body.obj_id, m.size, CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->writeBuffer(
      req->Body.event_id, queue_id, req->Body.obj_id, req->Body.m.write.is_svm,
      m.size, m.dst_offset, data, evt_timing, req->LocalWaitlist, rep->event));
  TP_WRITE_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
                  req->Body.obj_id, m.size, CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_WriteBufferReply);
}

void CommandQueue::CopyBuffer(uint32_t queue_id, Request *req, Reply *rep) {
  CopyBufferMsg_t &m = req->Body.m.copy;
  EventTiming_t evt_timing{};

  TP_COPY_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
                 m.src_buffer_id, m.dst_buffer_id, m.size, CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->copyBuffer(
      req->Body.event_id, queue_id, m.src_buffer_id, m.dst_buffer_id,
      m.size_buffer_id, m.size, m.src_offset, m.dst_offset, evt_timing,
      req->LocalWaitlist, rep->event));
  TP_COPY_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
                 m.src_buffer_id, m.dst_buffer_id, m.size, CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_CopyBufferReply);
}

void CommandQueue::ReadBufferRect(uint32_t queue_id, Request *req, Reply *rep) {
  ReadBufferRectMsg_t &m = req->Body.m.read_rect;
  EventTiming_t evt_timing{};

  COPY_VEC3(buffer_origin, m.buffer_origin);
  COPY_VEC3(region, m.region);

  rep->extra_size = m.host_bytes;
  char *host_ptr = nullptr;
#ifdef ENABLE_RDMA
  if (!backend->clientUsesRdma()) {
    rep->extra_data.resize(rep->extra_size);
    host_ptr = (char*)rep->extra_data.data();
  }
#else
  rep->extra_data.resize(rep->extra_size);
  host_ptr = (char*)rep->extra_data.data();
#endif

  TP_READ_BUFFER_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                      req->Body.obj_id, m.region.x, m.region.y, m.region.z,
                      CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->readBufferRect(
      req->Body.event_id, queue_id, req->Body.obj_id, buffer_origin, region,
      m.buffer_row_pitch, m.buffer_slice_pitch, host_ptr, m.host_bytes,
      evt_timing, req->LocalWaitlist, rep->event));
  TP_READ_BUFFER_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                      req->Body.obj_id, m.region.x, m.region.y, m.region.z,
                      CL_FINISHED);

  replyData(rep, evt_timing, MessageType_ReadBufferReply, m.host_bytes);
}

void CommandQueue::WriteBufferRect(uint32_t queue_id, Request *req,
                                   Reply *rep) {
  WriteBufferRectMsg_t &m = req->Body.m.write_rect;
  EventTiming_t evt_timing{};

  COPY_VEC3(buffer_origin, m.buffer_origin);
  COPY_VEC3(region, m.region);

#ifdef ENABLE_RDMA
  void *data = backend->clientUsesRdma() ? nullptr : req->ExtraData.data();
#else
  void *data = req->ExtraData.data();
#endif

  TP_WRITE_BUFFER_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                       req->Body.obj_id, m.region.x, m.region.y, m.region.z,
                       CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->writeBufferRect(
      req->Body.event_id, queue_id, req->Body.obj_id, buffer_origin, region,
      m.buffer_row_pitch, m.buffer_slice_pitch, data, req->ExtraDataSize,
      evt_timing, req->LocalWaitlist, rep->event));
  TP_WRITE_BUFFER_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                       req->Body.obj_id, m.region.x, m.region.y, m.region.z,
                       CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_WriteBufferReply);
}

void CommandQueue::CopyBufferRect(uint32_t queue_id, Request *req, Reply *rep) {
  CopyBufferRectMsg_t &m = req->Body.m.copy_rect;
  EventTiming_t evt_timing{};

  COPY_VEC3(dst_origin, m.dst_origin);
  COPY_VEC3(src_origin, m.src_origin);
  COPY_VEC3(region, m.region);

  TP_COPY_BUFFER_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                      m.src_buffer_id, m.dst_buffer_id, m.region.x, m.region.y,
                      m.region.z, CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->copyBufferRect(
      req->Body.event_id, queue_id, m.dst_buffer_id, m.src_buffer_id,
      dst_origin, src_origin, region, m.dst_row_pitch, m.dst_slice_pitch,
      m.src_row_pitch, m.src_slice_pitch, evt_timing, req->LocalWaitlist,
      rep->event));
  TP_COPY_BUFFER_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                      m.src_buffer_id, m.dst_buffer_id, m.region.x, m.region.y,
                      m.region.z, CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_CopyBufferReply);
}

void CommandQueue::FillBuffer(uint32_t queue_id, Request *req, Reply *rep) {
  FillBufferMsg_t &m = req->Body.m.fill_buffer;
  EventTiming_t evt_timing{};

  TP_FILL_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
                 req->Body.obj_id, m.size, CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->fillBuffer(
      req->Body.event_id, queue_id, req->Body.obj_id, m.dst_offset, m.size,
      req->ExtraData.data(), m.pattern_size, evt_timing, req->LocalWaitlist,
      rep->event));
  TP_FILL_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
                 req->Body.obj_id, m.size, CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_FillBufferReply);
}

void CommandQueue::RunKernel(uint32_t queue_id, Request *req, Reply *rep) {
  RunKernelMsg_t &m = req->Body.m.run_kernel;
  EventTiming_t evt_timing{};

  uint32_t ker_id = req->Body.obj_id;

  sizet_vec3 global = {m.global.x, m.global.y, m.global.z};
  sizet_vec3 local = {m.local.x, m.local.y, m.local.z};
  sizet_vec3 offset = {m.offset.x, m.offset.y, m.offset.z};
  unsigned dim = m.dim;

  TP_NDRANGE_KERNEL(req->Body.msg_id, req->Body.client_did, queue_id, ker_id,
                    CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->runKernel(
      req->Body.event_id, queue_id, dev_id, m.has_new_args, m.args_num,
      (uint64_t *)req->ExtraData.data(),
      (unsigned char *)req->ExtraData.data() + m.args_num * sizeof(uint64_t),
      m.pod_arg_size, (char *)req->ExtraData2.data(), evt_timing,
      req->Body.obj_id, req->LocalWaitlist, rep->event, dim, offset, global,
      (m.has_local ? &local : nullptr)));
  TP_NDRANGE_KERNEL(req->Body.msg_id, req->Body.client_did, queue_id, ker_id,
                    CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_RunKernelReply);
}

void CommandQueue::Barrier(uint32_t queue_id, Request *req, Reply *rep) {
  RunKernelMsg_t &m = req->Body.m.run_kernel;
  EventTiming_t evt_timing{};

  TP_BARRIER(req->Body.msg_id, req->Body.client_did, queue_id, CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->barrier(req->Body.event_id, queue_id, evt_timing,
                                      req->LocalWaitlist, rep->event));
  TP_BARRIER(req->Body.msg_id, req->Body.client_did, queue_id, CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_BarrierReply);
}

void CommandQueue::Marker(uint32_t queue_id, Request *req, Reply *rep) {
  RunKernelMsg_t &m = req->Body.m.run_kernel;
  EventTiming_t evt_timing{};

  TP_MARKER(req->Body.msg_id, req->Body.client_did, queue_id, CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->marker(req->Body.event_id, queue_id, evt_timing,
                                     req->LocalWaitlist, rep->event));
  TP_MARKER(req->Body.msg_id, req->Body.client_did, queue_id, CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_MarkerReply);
}

void CommandQueue::RunCommandBuffer(uint32_t queue_id, Request *req,
                                    Reply *rep) {
  EventTiming_t evt_timing{};
  uint32_t CmdbufId = req->Body.obj_id;

  TP_NDRANGE_KERNEL(req->Body.msg_id, req->Body.client_did, queue_id, cmdbuf_id,
                    CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->runCommandBuffer(req->Body.event_id, evt_timing,
                                               CmdbufId, 1, &req->Body.cq_id,
                                               req->LocalWaitlist, rep->event));
  TP_NDRANGE_KERNEL(req->Body.msg_id, req->Body.client_did, queue_id, cmdbuf_id,
                    CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_RunCommandBufferReply);
}

/******************/

void CommandQueue::FillImage(uint32_t queue_id, Request *req, Reply *rep) {
  FillImageRectMsg_t &m = req->Body.m.fill_image;
  EventTiming_t evt_timing{};

  COPY_VEC3(img_origin, m.origin);
  COPY_VEC3(img_region, m.region);

  TP_FILL_IMAGE(req->Body.msg_id, req->Body.client_did, queue_id,
                req->Body.obj_id, CL_RUNNING);
  assert(req->ExtraDataSize == 16);
  RETURN_IF_ERR_CODE(backend->fillImage(
      req->Body.event_id, queue_id, req->Body.obj_id, img_origin, img_region,
      req->ExtraData.data(), evt_timing, req->LocalWaitlist, rep->event));
  TP_FILL_IMAGE(req->Body.msg_id, req->Body.client_did, queue_id,
                req->Body.obj_id, CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_FillImageRectReply);
}

void CommandQueue::ReadImageRect(uint32_t queue_id, Request *req, Reply *rep) {
  ReadImageRectMsg_t &m = req->Body.m.read_image_rect;
  EventTiming_t evt_timing{};

  COPY_VEC3(img_origin, m.origin);
  COPY_VEC3(img_region, m.region);

  rep->extra_size = m.host_bytes;
  rep->extra_data.resize(rep->extra_size);

  TP_READ_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                     req->Body.obj_id, m.region.x, m.region.y, m.region.z,
                     CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->readImageRect(
      req->Body.event_id, queue_id, req->Body.obj_id, img_origin, img_region,
      rep->extra_data.data(), m.host_bytes, evt_timing, req->LocalWaitlist,
      rep->event));
  TP_READ_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                     req->Body.obj_id, m.region.x, m.region.y, m.region.z,
                     CL_FINISHED);

  replyData(rep, evt_timing, MessageType_ReadImageRectReply, m.host_bytes);
}

void CommandQueue::WriteImageRect(uint32_t queue_id, Request *req, Reply *rep) {
  WriteImageRectMsg_t &m = req->Body.m.write_image_rect;
  EventTiming_t evt_timing{};

  COPY_VEC3(img_origin, m.origin);
  COPY_VEC3(img_region, m.region);

  TP_WRITE_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                      req->Body.obj_id, m.region.x, m.region.y, m.region.z,
                      CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->writeImageRect(
      req->Body.event_id, queue_id, req->Body.obj_id, img_origin, img_region,
      // m.IMAGE_row_pitch, m.IMAGE_slice_pitch,
      req->ExtraData.data(), req->ExtraDataSize, evt_timing, req->LocalWaitlist,
      rep->event));
  TP_WRITE_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                      req->Body.obj_id, m.region.x, m.region.y, m.region.z,
                      CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_WriteImageRectReply);
}

void CommandQueue::CopyBuffer2Image(uint32_t queue_id, Request *req,
                                    Reply *rep) {
  CopyBuf2ImgMsg_t &m = req->Body.m.copy_buf2img;
  EventTiming_t evt_timing{};

  COPY_VEC3(img_origin, m.origin);
  COPY_VEC3(img_region, m.region);

  TP_COPY_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                     m.src_buf_id, req->Body.obj_id, m.region.x, m.region.y,
                     m.region.z, CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->copyBuffer2Image(
      req->Body.event_id, queue_id, req->Body.obj_id, m.src_buf_id, img_origin,
      img_region, m.src_offset, evt_timing, req->LocalWaitlist, rep->event));
  TP_COPY_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                     m.src_buf_id, req->Body.obj_id, m.region.x, m.region.y,
                     m.region.z, CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_CopyBuffer2ImageReply);
}

void CommandQueue::CopyImage2Buffer(uint32_t queue_id, Request *req,
                                    Reply *rep) {
  CopyImg2BufMsg_t &m = req->Body.m.copy_img2buf;
  EventTiming_t evt_timing{};

  COPY_VEC3(img_origin, m.origin);
  COPY_VEC3(img_region, m.region);

  TP_COPY_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                     req->Body.obj_id, m.dst_buf_id, m.region.x, m.region.y,
                     m.region.z, CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->copyImage2Buffer(
      req->Body.event_id, queue_id, req->Body.obj_id, m.dst_buf_id, img_origin,
      img_region, m.dst_offset, evt_timing, req->LocalWaitlist, rep->event));
  TP_COPY_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                     req->Body.obj_id, m.dst_buf_id, m.region.x, m.region.y,
                     m.region.z, CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_CopyImage2BufferReply);
}

void CommandQueue::CopyImage2Image(uint32_t queue_id, Request *req,
                                   Reply *rep) {
  CopyImg2ImgMsg_t &m = req->Body.m.copy_img2img;
  EventTiming_t evt_timing{};

  COPY_VEC3(src_origin, m.src_origin);
  COPY_VEC3(dst_origin, m.dst_origin);
  COPY_VEC3(img_region, m.region);

  TP_COPY_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                     m.src_image_id, m.dst_image_id, m.region.x, m.region.y,
                     m.region.z, CL_RUNNING);
  RETURN_IF_ERR_CODE(backend->copyImage2Image(
      req->Body.event_id, queue_id, m.dst_image_id, m.src_image_id, dst_origin,
      src_origin, img_region, evt_timing, req->LocalWaitlist, rep->event));
  TP_COPY_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                     m.src_image_id, m.dst_image_id, m.region.x, m.region.y,
                     m.region.z, CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_CopyImage2ImageReply);
}
