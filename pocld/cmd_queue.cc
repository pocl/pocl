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

#include "cmd_queue.hh"
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
  if (!TryRun(request))
    pending.push_back(request);
}

void CommandQueue::notify() {
  for (size_t i = 0; i < pending.size();) {
    if (TryRun(pending[i]))
      pending.erase(pending.begin() + i);
    else
      i += 1;
  }
}

bool CommandQueue::TryRun(Request *request) {
  size_t unknown_events = request->Body.waitlist_size;
  for (size_t i = 0; i < request->Body.waitlist_size; ++i) {
    if (backend->isCommandReceived(request->Waitlist[i]))
      unknown_events -= 1;
  }

  if (!unknown_events)
    RunCommand(request);

  return !unknown_events;
}

void CommandQueue::RunCommand(Request *request) {
  Reply *reply = new Reply(request);
  int slow = 0;

  POCL_MSG_PRINT_GENERAL("CQ %" PRIu32 " DID %" PRIu32
                         " |||||||||| REQ QID %" PRIu32 " DID %" PRIu32 " \n",
                         queue_id, dev_id, uint32_t(request->Body.cq_id),
                         uint32_t(request->Body.did));
  assert(queue_id == request->Body.cq_id);
  if (request->Body.message_type == MessageType_MigrateD2D)
    assert(dev_id == request->Body.did ||
           dev_id == request->Body.m.migrate.source_did);
  else
    assert(dev_id == request->Body.did);

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

  // TODO: move this to reply thread?
  // Probably not necessary since we can only have the real event by this
  // point...
  EventPair p = backend->getEventPairForId(request->Body.event_id);
  // If the command failed or was a migration to this server, there won't be a
  // native event.
  // XXX: does killing the server in debug builds help more with debugging than
  // just ignoring the missing event?
  if (request->Body.message_type != MessageType_MigrateD2D)
    assert(p.native.get());
  reply->event = p.native;

  ReplyQueueThread *rqt = (slow ? write_slow : write_fast);
  rqt->pushReply(reply);
}

/***********    CMD QUEUE    *******************/

/**********************************************************************/
/**********************************************************************/
/**********************************************************************/

void CommandQueue::MigrateMemObj(uint32_t queue_id, Request *req, Reply *rep) {
  MigrateD2DMsg_t &m = req->Body.m.migrate;
  EventTiming_t evt_timing{};

  if (m.source_pid == req->Body.pid && m.source_peer_id == m.dest_peer_id) {
    // direct migration within single platform
    // TP_WRITE_BUFFER(req->Body.event_id, req->Body.client_did, queue_id,
    // req->Body.obj_id, m.size, CL_RUNNING);
    // direct mig within 1 platform
    RETURN_IF_ERR_CODE(backend->migrateMemObject(
        req->Body.event_id, queue_id, req->Body.obj_id, m.is_image, evt_timing,
        req->Body.waitlist_size, req->Waitlist.data()));
    // TP_WRITE_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
    // req->Body.obj_id, m.size, CL_FINISHED);
  }
#ifndef RDMA_USE_SVM
  // with RDMA the P2P write is already done by now
  else {
    // after data is read
    assert(m.is_external);
    void *host_ptr;
#ifdef ENABLE_RDMA
    host_ptr = backend->getRdmaShadowPtr(req->Body.obj_id);
    req->ExtraDataSize = m.size;
#else
    assert(req->ExtraData.size() >= req->ExtraDataSize);
    host_ptr = req->ExtraData.data();
#endif
    // finish the migration by import
    if (m.is_image) {
      sizet_vec3 origin = {0, 0, 0};
      sizet_vec3 region = {m.width, m.height, m.depth};

      TP_WRITE_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                          req->Body.obj_id, m.width, m.height, m.depth,
                          CL_RUNNING);
      RETURN_IF_ERR_CODE(backend->writeImageRect(
          req->Body.event_id, queue_id, req->Body.obj_id, origin, region,
          host_ptr, req->ExtraDataSize, evt_timing, req->Body.waitlist_size,
          req->Waitlist.data()));
      TP_WRITE_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                          req->Body.obj_id, m.width, m.height, m.depth,
                          CL_FINISHED);
    } else {
      TP_WRITE_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
                      req->Body.obj_id, m.size, CL_RUNNING);
      RETURN_IF_ERR_CODE(backend->writeBuffer(
          req->Body.event_id, queue_id, req->Body.obj_id, 0, m.size, 0,
          host_ptr, evt_timing, req->Body.waitlist_size, req->Waitlist.data()));
      TP_WRITE_BUFFER(req->Body.msg_id, req->Body.client_did, queue_id,
                      req->Body.obj_id, m.size, CL_FINISHED);
    }
  }
#ifdef ENABLE_RDMA
  // unset size from the request, since the request's extra_data is not used
  req->ExtraDataSize = 0;
#endif
#endif

  replyOK(rep, evt_timing, MessageType_MigrateD2DReply);
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
      req->Body.waitlist_size, req->Waitlist.data()));
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
      m.size, m.dst_offset, data, evt_timing, req->Body.waitlist_size,
      req->Waitlist.data()));
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
      req->Body.waitlist_size, req->Waitlist.data()));
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
      evt_timing, req->Body.waitlist_size, req->Waitlist.data()));
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
      evt_timing, req->Body.waitlist_size, req->Waitlist.data()));
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
      m.src_row_pitch, m.src_slice_pitch, evt_timing, req->Body.waitlist_size,
      req->Waitlist.data()));
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
      req->ExtraData.data(), m.pattern_size, evt_timing,
      req->Body.waitlist_size, req->Waitlist.data()));
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
      req->Body.obj_id, req->Body.waitlist_size, req->Waitlist.data(), dim,
      offset, global, (m.has_local ? &local : nullptr)));
  TP_NDRANGE_KERNEL(req->Body.msg_id, req->Body.client_did, queue_id, ker_id,
                    CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_RunKernelReply);
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
      req->ExtraData.data(), evt_timing, req->Body.waitlist_size,
      req->Waitlist.data()));
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
      rep->extra_data.data(), m.host_bytes, evt_timing, req->Body.waitlist_size,
      req->Waitlist.data()));
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
      req->ExtraData.data(), req->ExtraDataSize, evt_timing,
      req->Body.waitlist_size, req->Waitlist.data()));
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
      img_region, m.src_offset, evt_timing, req->Body.waitlist_size,
      req->Waitlist.data()));
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
      img_region, m.dst_offset, evt_timing, req->Body.waitlist_size,
      req->Waitlist.data()));
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
      src_origin, img_region, evt_timing, req->Body.waitlist_size,
      req->Waitlist.data()));
  TP_COPY_IMAGE_RECT(req->Body.msg_id, req->Body.client_did, queue_id,
                     m.src_image_id, m.dst_image_id, m.region.x, m.region.y,
                     m.region.z, CL_FINISHED);

  replyOK(rep, evt_timing, MessageType_CopyImage2ImageReply);
}
