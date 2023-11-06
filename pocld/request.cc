/* request.cc - class representing a command received from a client or peer

   Copyright (c) 2023 Jan Solanti / Tampere University

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

#include "common.hh"
#include "messages.h"
#include "request.hh"
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

/* Returns 0 on success and no-op, otherwise errno */
static int reentrant_read(int fd, void *dest, size_t size, size_t *tracker) {
  if (*tracker == size)
    return 0;

  ssize_t readb;
  readb = ::read(fd, (char *)dest + *tracker, size - *tracker);
  if (readb < 0)
    return errno;

  if (readb == 0)
    return EPIPE;

  *tracker += readb;
  if (*tracker != size)
    return EAGAIN;
  return 0;
}

#define RETURN_UNLESS_DONE(call)                                               \
  do {                                                                         \
    int ret = (call);                                                          \
    if (ret) {                                                                 \
      if (ret == EAGAIN || ret == EWOULDBLOCK)                                 \
        return true;                                                           \
      POCL_MSG_ERR("Read error on " #call ", fd=%d, reason: %s\n", fd,         \
                   strerror(ret));                                             \
      return false;                                                            \
    }                                                                          \
  } while (0);

bool Request::read(int fd) {
  ssize_t readb;
  Request *request = this;
  RequestMsg_t *req = &request->req;

  if (!request->read_start_timestamp_ns) {
    auto now1 = std::chrono::system_clock::now();
    request->read_start_timestamp_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now1.time_since_epoch())
            .count();
  }

  RETURN_UNLESS_DONE(reentrant_read(fd, &request->req_size,
                                    sizeof(request->req_size),
                                    &request->req_size_read));

  RETURN_UNLESS_DONE(
      reentrant_read(fd, req, request->req_size, &request->req_read));

  TP_MSG_RECEIVED(req->msg_id, req->did, req->cq_id, req->message_type);

  RequestMessageType t = static_cast<RequestMessageType>(req->message_type);
  POCL_MSG_PRINT_GENERAL("---------------------------------------------------"
                         "----------------------------\n");
  POCL_MSG_PRINT_GENERAL("MESSAGE RECEIVED, ID: %" PRIu64
                         " TYPE: %s SIZE: %" PRIu64 "/%" PRIu32 " \n",
                         uint64_t(req->msg_id), request_to_str(t),
                         request->req_read, request->req_size);

  request->waitlist_size = req->waitlist_size;
  switch (req->message_type) {
  case MessageType_WriteBuffer:
    request->extra_size = req->m.write.size;
    break;
  case MessageType_WriteBufferRect:
    request->extra_size = req->m.write_rect.host_bytes;
    break;
  case MessageType_WriteImageRect:
    request->extra_size = req->m.write_image_rect.host_bytes;
    break;
  case MessageType_MigrateD2D:
    if (req->m.migrate.is_external) {
      request->extra_size = req->m.migrate.size;
    }
    break;
  case MessageType_FillBuffer:
    request->extra_size = req->m.fill_buffer.pattern_size;
    assert(request->extra_size <= (16 * sizeof(uint64_t)));
    break;
  case MessageType_FillImageRect:
    request->extra_size = 16;
    break;
  case MessageType_RunKernel:
    if (req->m.run_kernel.has_new_args) {
      request->extra_size = req->m.run_kernel.args_num * sizeof(uint64_t);
      request->extra_size2 = req->m.run_kernel.pod_arg_size;
    }
    break;
  /*****************************/
  case MessageType_BuildProgramFromBinary:
  case MessageType_BuildProgramFromSource:
  case MessageType_BuildProgramFromSPIRV:
    request->extra_size2 = req->m.build_program.options_len;
  case MessageType_BuildProgramWithBuiltins:
    request->extra_size = req->m.build_program.payload_size;
    break;
  /*****************************/
  case MessageType_CreateKernel:
    request->extra_size = req->m.create_kernel.name_len;
    break;
  default:
    break;
  }

  /*****************************/
  if (req->waitlist_size > 0) {
    if (!request->waitlist)
      request->waitlist = new uint64_t[request->waitlist_size];
    POCL_MSG_PRINT_GENERAL(
        "READING WAIT LIST FOR ID: %" PRIu64 " = %" PRIuS "/%" PRIu64 "\n",
        uint64_t(req->msg_id), request->waitlist_read, request->waitlist_size);
    RETURN_UNLESS_DONE(reentrant_read(fd, request->waitlist,
                                      request->waitlist_size * sizeof(uint64_t),
                                      &request->waitlist_read));
  }
  /*****************************/

  /*****************************/
  if (request->extra_size > 0) {
    if (!request->extra_data)
      request->extra_data = new char[request->extra_size + 1];

    POCL_MSG_PRINT_GENERAL(
        "READING EXTRA FOR ID: %" PRIu64 " = %" PRIuS "/%" PRIu64 "\n",
        uint64_t(req->msg_id), request->extra_read, request->extra_size);
    RETURN_UNLESS_DONE(reentrant_read(
        fd, request->extra_data, request->extra_size, &request->extra_read));
    /* Always add a null byte at the end - it is needed for strings and it does
     * not harm other things */
    request->extra_data[request->extra_size] = 0;
  }
  /*****************************/

  /*****************************/
  if (request->extra_size2 > 0) {
    if (!request->extra_data2)
      request->extra_data2 = new char[request->extra_size2 + 1];

    POCL_MSG_PRINT_GENERAL(
        "READING EXTRA2 FOR ID:%" PRIu64 " = %" PRIuS "/%" PRIu64 "\n",
        uint64_t(req->msg_id), request->extra_read2, request->extra_size2);
    RETURN_UNLESS_DONE(reentrant_read(
        fd, request->extra_data2, request->extra_size2, &request->extra_read2));
    /* Always add null byte here too, just in case extra2 is a string */
    request->extra_data2[request->extra_size2] = 0;
  }
  /*****************************/

  if (!request->read_end_timestamp_ns) {
    auto now2 = std::chrono::system_clock::now();
    request->read_end_timestamp_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now2.time_since_epoch())
            .count();
  }

  POCL_MSG_PRINT_GENERAL("ALL READS COMPLETE FOR ID: %" PRIu64 ", fd=%d\n",
                         uint64_t(req->msg_id), fd);

  fully_read = true;
  return true;
}

#undef CHECK_READ_RETURN
