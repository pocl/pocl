/* request_th.cc - pocld thread that listens for incoming commands from the
   client

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

#include <cassert>
#include <poll.h>
#include <sys/socket.h>

#include "common.hh"
#include "messages.h"
#include "request_th.hh"
#include "tracing.h"

#define CL_INVALID_OPERATION -59

const char *request_to_str(RequestMessageType type) {
  switch (type) {
  case MessageType_ServerInfo:
    return "ServerInfo";
  case MessageType_DeviceInfo:
    return "DeviceInfo";
  case MessageType_ConnectPeer:
    return "ConnectPeer";
  case MessageType_PeerHandshake:
    return "PeerHandshake";

  case MessageType_CreateBuffer:
    return "CreateBuffer";
  case MessageType_FreeBuffer:
    return "FreeBuffer";

  case MessageType_CreateCommandQueue:
    return "CreateCommandQueue";
  case MessageType_FreeCommandQueue:
    return "FreeCommandQueue";

  case MessageType_CreateSampler:
    return "CreateSampler";
  case MessageType_FreeSampler:
    return "FreeSampler";

  case MessageType_CreateImage:
    return "CreateImage";
  case MessageType_FreeImage:
    return "FreeImage";

  case MessageType_CreateKernel:
    return "CreateKernel";
  case MessageType_FreeKernel:
    return "FreeKernel";

  case MessageType_BuildProgramFromSource:
    return "BuildProgramFromSource";
  case MessageType_BuildProgramFromBinary:
    return "BuildProgramFromBinary";
  case MessageType_BuildProgramFromSPIRV:
    return "BuildProgramFromSPIRV";
  case MessageType_BuildProgramWithBuiltins:
    return "BuildProgramWithBuiltins";
  case MessageType_FreeProgram:
    return "FreeProgram";

  case MessageType_MigrateD2D:
    return "MigrateD2D";

  case MessageType_ReadBuffer:
    return "ReadBuffer";
  case MessageType_WriteBuffer:
    return "WriteBuffer";
  case MessageType_CopyBuffer:
    return "CopyBuffer";
  case MessageType_FillBuffer:
    return "FillBuffer";

  case MessageType_ReadBufferRect:
    return "ReadBufferRect";
  case MessageType_WriteBufferRect:
    return "WriteBufferRect";
  case MessageType_CopyBufferRect:
    return "CopyBufferRect";

  case MessageType_CopyImage2Buffer:
    return "CopyImage2Buffer";
  case MessageType_CopyBuffer2Image:
    return "CopyBuffer2Image";
  case MessageType_CopyImage2Image:
    return "CopyImage2Image";
  case MessageType_ReadImageRect:
    return "ReadImageRect";
  case MessageType_WriteImageRect:
    return "WriteImageRect";
  case MessageType_FillImageRect:
    return "FillImageRect";

  case MessageType_RunKernel:
    return "RunKernel";

  case MessageType_NotifyEvent:
    return "NotifyEvent";

  case MessageType_RdmaBufferRegistration:
    return "RdmaBufferRegistration";

  case MessageType_Finish:
    return "Finish";

  case MessageType_Shutdown:
    return "Shutdown";

  default:
    return "UNKNOWN";
  }
}

RequestQueueThread::RequestQueueThread(std::atomic_int *f,
                                       VirtualContextBase *c, ExitHelper *e,
                                       TrafficMonitor *tm, const char *id_str)
    : fd(f), virtualContext(c), eh(e), id_str(id_str), netstat(tm) {
  io_thread = std::thread{&RequestQueueThread::readThread, this};
}

RequestQueueThread::~RequestQueueThread() {
  eh->requestExit(id_str.c_str(), 0);
  io_thread.join();
  shutdown(*fd, SHUT_RD);
}

void RequestQueueThread::readThread() {
  ssize_t readb;

  struct pollfd pfd;
  pfd.events = POLLIN;
  int nevs;

  int fd = *this->fd;
  int oldfd = fd;
  while (1) {
  RETRY:
    fd = *this->fd;
    if (fd != oldfd) {
      POCL_MSG_PRINT_GENERAL("%s: FD change detected: %d -> %d\n",
                             id_str.c_str(), oldfd, fd);
      close(oldfd);
    }
    oldfd = fd;
    if (eh->exit_requested())
      return;

    pfd.fd = fd;
    nevs = poll(&pfd, 1, 3 * MS_PER_S);
    if (nevs < 1)
      continue;
    else if (!(pfd.revents & POLLIN))
      continue;

    uint32_t msg_size;
    Request *request = new Request;
    RequestMsg_t *req = &request->req;
    CHECK_READ_RETRY(readb, read_full(fd, &msg_size, sizeof(uint32_t), netstat),
                     id_str.c_str());
    CHECK_READ_RETRY(readb, read_full(fd, req, msg_size, netstat),
                     id_str.c_str());

    auto now1 = std::chrono::system_clock::now();
    request->read_start_timestamp_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now1.time_since_epoch())
            .count();

    TP_MSG_RECEIVED(req->msg_id, req->did, req->cq_id, req->message_type);

    RequestMessageType t = static_cast<RequestMessageType>(req->message_type);
    POCL_MSG_PRINT_GENERAL("%s: "
                           "---------------------------------------------------"
                           "----------------------------\n",
                           id_str.c_str());
    POCL_MSG_PRINT_GENERAL("%s: MESSAGE RECEIVED, ID: %" PRIu64
                           " TYPE: %s SIZE: %" PRId64 "/%" PRIu32 " \n",
                           id_str.c_str(), uint64_t(req->msg_id),
                           request_to_str(t), int64_t(readb), msg_size);

    assert(static_cast<uint32_t>(readb) == msg_size);

    // read extra parts of requests
    if (req->message_type == MessageType_WriteBuffer) {
      request->extra_size = req->m.write.size;
    }
    if (req->message_type == MessageType_WriteBufferRect) {
      request->extra_size = req->m.write_rect.host_bytes;
    }
    if (req->message_type == MessageType_WriteImageRect) {
      request->extra_size = req->m.write_image_rect.host_bytes;
    }
    if (req->message_type == MessageType_MigrateD2D &&
        req->m.migrate.is_external) {
      request->extra_size = req->m.migrate.size;
    }
    // END of commands with RDMA transferable data

    if (req->message_type == MessageType_FillBuffer) {
      request->extra_size = req->m.fill_buffer.pattern_size;
      assert(request->extra_size <= (16 * sizeof(uint64_t)));
    }
    if (req->message_type == MessageType_FillImageRect) {
      request->extra_size = 16;
    }

    if (req->message_type == MessageType_RunKernel &&
        req->m.run_kernel.has_new_args) {
      // read args
      //      uint64_t *args = new uint64_t [arg_num];
      //      read(args, arg_num * sizeof(uint64_t));
      request->extra_size = req->m.run_kernel.args_num * sizeof(uint64_t);
      request->extra_size2 = req->m.run_kernel.pod_arg_size;
    }
    if (req->message_type == MessageType_BuildProgramFromBinary ||
        req->message_type == MessageType_BuildProgramFromSource ||
        req->message_type == MessageType_BuildProgramFromSPIRV ||
        req->message_type == MessageType_BuildProgramWithBuiltins) {
      request->extra_size = req->m.build_program.payload_size;
    }

    /*****************************/
    request->waitlist_size = req->waitlist_size;
    if (req->waitlist_size > 0) {
      POCL_MSG_PRINT_GENERAL("%s: READING WAIT LIST: %" PRIu32 " \n",
                             id_str.c_str(), request->waitlist_size);
      request->waitlist = new uint64_t[request->waitlist_size];
      CHECK_READ_RETRY(readb,
                       read_full(fd, request->waitlist,
                                 request->waitlist_size * sizeof(uint64_t),
                                 netstat),
                       id_str.c_str());
    }
    /*****************************/

    /*****************************/
    if (request->extra_size > 0) {
      request->extra_data = new char[request->extra_size];

      POCL_MSG_PRINT_GENERAL("%s: READING EXTRA: %" PRIuS " \n",
                             id_str.c_str(), request->extra_size);
      CHECK_READ_RETRY(
          readb,
          read_full(fd, request->extra_data, request->extra_size, netstat),
          id_str.c_str());
    }
    /*****************************/

    // name string
    if (req->message_type == MessageType_CreateKernel) {
      request->extra_size = req->m.create_kernel.name_len;
      request->extra_data = new char[request->extra_size + 1];
      POCL_MSG_PRINT_GENERAL("%s: READING EXTRA: %" PRIuS " \n", id_str.c_str(),
                             request->extra_size);
      CHECK_READ_RETRY(
          readb,
          read_full(fd, request->extra_data, request->extra_size, netstat),
          id_str.c_str());
      request->extra_data[request->extra_size] = 0;
    }

    // options
    if (req->message_type == MessageType_BuildProgramFromBinary ||
        req->message_type == MessageType_BuildProgramFromSource ||
        req->message_type == MessageType_BuildProgramFromSPIRV) {
      request->extra_size2 = req->m.build_program.options_len;
    }
    /*****************************/
    if (request->extra_size2 > 0) {
      request->extra_data2 = new char[request->extra_size2 + 1];
      POCL_MSG_PRINT_GENERAL("%s: READING EXTRA2: %" PRIuS "\n", id_str.c_str(),
                             request->extra_size2);
      CHECK_READ_RETRY(
          readb,
          read_full(fd, request->extra_data2, request->extra_size2, netstat),
          id_str.c_str());
      request->extra_data2[request->extra_size2] = 0;
    }
    /*****************************/
    POCL_MSG_PRINT_GENERAL("%s: ALL READS COMPLETE FOR ID: %" PRIu64 "\n",
                           id_str.c_str(), uint64_t(req->msg_id));

    auto now2 = std::chrono::system_clock::now();
    request->read_end_timestamp_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now2.time_since_epoch())
            .count();

    switch (req->message_type) {
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
    case MessageType_BuildProgramWithBuiltins:
    case MessageType_FreeProgram:
    case MessageType_MigrateD2D:
    case MessageType_RdmaBufferRegistration:
    case MessageType_Shutdown: {
      virtualContext->nonQueuedPush(request);
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
      virtualContext->queuedPush(request);
      break;
    }
    case MessageType_NotifyEvent: {
      // TODO: this message should probably contain an actual status... (see
      // also rdma thread)
      virtualContext->notifyEvent(request->req.event_id, CL_COMPLETE);
      delete request;
      break;
    }

    default: {
      virtualContext->unknownRequest(request);
      break;
    }
    }

    // TODO reply fail
  }
}
