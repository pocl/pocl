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
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <unistd.h>

#include "common_cl.hh"
#include "pocl_compiler_macros.h"
#include "pocl_debug.h"
#include "request.hh"
#include "tracing.h"

#define CL_INVALID_OPERATION -59

const char *request_to_str(RequestMessageType type) {
  switch (type) {
  case MessageType_InvalidRequest:
    return "INVALID REQUEST";
  case MessageType_CreateOrAttachSession:
    return "CreateOrAttachSession";
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
  case MessageType_CompileProgramFromSource:
    return "CompileProgramFromSource";
  case MessageType_BuildProgramFromBinary:
    return "BuildProgramFromBinary";
  case MessageType_BuildProgramFromSPIRV:
    return "BuildProgramFromSPIRV";
  case MessageType_CompileProgramFromSPIRV:
    return "CompileProgramFromSPIRV";
  case MessageType_LinkProgram:
    return "LinkProgram";
  case MessageType_BuildProgramWithBuiltins:
    return "BuildProgramWithBuiltins";
  case MessageType_BuildProgramWithDefinedBuiltins:
      return "BuildProgramWithDefinedBuiltins";
  case MessageType_FreeProgram:
    return "FreeProgram";

  case MessageType_MigrateD2D:
    return "MigrateD2D";
  case MessageType_Barrier:
    return "Barrier";

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

  case MessageType_CreateCommandBuffer:
    return "CreateCommandBuffer";
  case MessageType_FreeCommandBuffer:
    return "FreeCommandBuffer";
  case MessageType_RunCommandBuffer:
    return "RunCommandBuffer";

  default:
    return "UNKNOWN";
  }
}

int ByteReader::readReentrant(void *Destination, size_t Bytes,
                              size_t *Tracker) {
  if (*Tracker == Bytes)
    return 0;
  size_t Copied = readFull(Destination, Bytes);
  *Tracker += Copied;
  return 0;
}

int ByteReader::readFull(void *Destination, size_t Bytes) {
  size_t CopyableBytes = std::min(Bytes, Length - Offset);
  std::memcpy(Destination, StartPtr + Offset, CopyableBytes);
  Offset += CopyableBytes;
  return CopyableBytes;
}

std::string ByteReader::describe() {
  const size_t MaxLength = 32;
  std::string Tmp;
  Tmp.reserve(MaxLength);
  std::snprintf(Tmp.data(), MaxLength, "mem:%p", StartPtr);
  return Tmp;
}

#define RETURN_UNLESS_DONE(call)                                               \
  do {                                                                         \
    int ret = (call);                                                          \
    if (ret) {                                                                 \
      if (ret == EAGAIN || ret == EWOULDBLOCK)                                 \
        return true;                                                           \
      POCL_MSG_ERR("Read error on " #call ", %s, reason: %s\n",                \
                   Source->describe().c_str(), strerror(ret));                 \
      return false;                                                            \
    }                                                                          \
  } while (0);

template <class T> bool RequestReadImpl(Request *Req, T *Source) {
  ssize_t readb;

  RequestMsg_t *Body = &Req->Body;

  if (!Req->ReadStartTimestampNS) {
    auto now1 = std::chrono::system_clock::now();
    Req->ReadStartTimestampNS =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now1.time_since_epoch())
            .count();
  }

  RETURN_UNLESS_DONE(Source->readReentrant(
      &Req->BodySize, sizeof(Req->BodySize), &Req->BodySizeBytesRead));

  RETURN_UNLESS_DONE(
      Source->readReentrant(Body, Req->BodySize, &Req->BodyBytesRead));

  TP_MSG_RECEIVED(Body->msg_id, Body->did, Body->cq_id, Body->message_type);

  RequestMessageType t = static_cast<RequestMessageType>(Body->message_type);
  POCL_MSG_PRINT_GENERAL("---------------------------------------------------"
                         "----------------------------\n");
  POCL_MSG_PRINT_GENERAL("MESSAGE RECEIVED, ID: %" PRIu64
                         " TYPE: %s (%d) SIZE: %" PRIu64 "/%" PRIu32 " \n",
                         uint64_t(Body->msg_id), request_to_str(t), t,
                         Req->BodyBytesRead, Req->BodySize);

  switch (Body->message_type) {
  case MessageType_WriteBuffer:
    Req->ExtraDataSize = Body->m.write.size;
    break;
  case MessageType_WriteBufferRect:
    Req->ExtraDataSize = Body->m.write_rect.host_bytes;
    break;
  case MessageType_WriteImageRect:
    Req->ExtraDataSize = Body->m.write_image_rect.host_bytes;
    break;
  case MessageType_MigrateD2D:
    if (Body->m.migrate.is_external) {
      Req->ExtraDataSize = Body->m.migrate.size;
    }
    break;
  case MessageType_FillBuffer:
    Req->ExtraDataSize = Body->m.fill_buffer.pattern_size;
    assert(Req->ExtraDataSize <= (16 * sizeof(uint64_t)));
    break;
  case MessageType_FillImageRect:
    Req->ExtraDataSize = 16;
    break;
  case MessageType_RunKernel:
    if (Body->m.run_kernel.has_new_args) {
      /* The arguments itthis come in through extra data, as well as an array of
         flags which inform whether an argument (buffer) is an
         SVM pointer or not. */
      Req->ExtraDataSize = Body->m.run_kernel.args_num * sizeof(uint64_t) +
                           Body->m.run_kernel.args_num * sizeof(unsigned char);
      Req->ExtraData2Size = Body->m.run_kernel.pod_arg_size;
    }
    break;
  /*****************************/
  case MessageType_BuildProgramFromBinary:
  case MessageType_BuildProgramFromSource:
  case MessageType_BuildProgramFromSPIRV:
  case MessageType_CompileProgramFromSPIRV:
  case MessageType_CompileProgramFromSource:
  case MessageType_LinkProgram:
    Req->ExtraData2Size = Body->m.build_program.options_len;
    /* intentional fall through to setting payload (i.e. binary) size */
    POCL_FALLTHROUGH;
  case MessageType_BuildProgramWithBuiltins:
  case MessageType_BuildProgramWithDefinedBuiltins:
    Req->ExtraDataSize = Body->m.build_program.payload_size;
    break;
  /*****************************/
  case MessageType_CreateKernel:
    Req->ExtraDataSize = Body->m.create_kernel.name_len;
    break;
  /*****************************/
  case MessageType_CreateCommandBuffer:
    Req->ExtraDataSize = Body->m.create_cmdbuf.num_queues * sizeof(uint32_t) +
                         Body->m.create_cmdbuf.commands_size;
  default:
    break;
  }

  /*****************************/
  if (Body->waitlist_size > 0) {
    Req->Waitlist.resize(Body->waitlist_size);
    POCL_MSG_PRINT_GENERAL("READING WAIT LIST FOR ID: %" PRIu64 " = %" PRIuS
                           "/%" PRIu32 "\n",
                           uint64_t(Body->msg_id), Req->WaitlistBytesRead,
                           Req->Body.waitlist_size);
    RETURN_UNLESS_DONE(Source->readReentrant(
        Req->Waitlist.data(), Req->Body.waitlist_size * sizeof(uint64_t),
        &Req->WaitlistBytesRead));
  }
  /*****************************/

  /*****************************/
  if (Req->ExtraDataSize > 0) {
    Req->ExtraData.resize(Req->ExtraDataSize + 1);
    POCL_MSG_PRINT_GENERAL(
        "READING EXTRA FOR ID: %" PRIu64 " = %" PRIuS "/%" PRIu64 "\n",
        uint64_t(Body->msg_id), Req->ExtraDataBytesRead, Req->ExtraDataSize);
    RETURN_UNLESS_DONE(Source->readReentrant(
        Req->ExtraData.data(), Req->ExtraDataSize, &Req->ExtraDataBytesRead));
    /* Always add a null byte at the end - it is needed for strings and it does
     * not harm other things */
    Req->ExtraData[Req->ExtraDataSize] = 0;
  }
  /*****************************/

  /*****************************/
  if (Req->ExtraData2Size > 0) {
    Req->ExtraData2.resize(Req->ExtraData2Size + 1);
    POCL_MSG_PRINT_GENERAL(
        "READING EXTRA2 FOR ID:%" PRIu64 " = %" PRIuS "/%" PRIu64 "\n",
        uint64_t(Body->msg_id), Req->ExtraData2BytesRead, Req->ExtraData2Size);
    RETURN_UNLESS_DONE(Source->readReentrant(Req->ExtraData2.data(),
                                             Req->ExtraData2Size,
                                             &Req->ExtraData2BytesRead));
    /* Always add null byte here too, just in case extra2 is a string */
    Req->ExtraData2[Req->ExtraData2Size] = 0;
  }
  /*****************************/

  if (!Req->ReadEndTimestampNS) {
    auto now2 = std::chrono::system_clock::now();
    Req->ReadEndTimestampNS =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now2.time_since_epoch())
            .count();
  }

  POCL_MSG_PRINT_GENERAL("ALL READS COMPLETE FOR ID: %" PRIu64 ", %s\n",
                         uint64_t(Body->msg_id), Source->describe().c_str());

  Req->IsFullyRead = true;
  return true;
}

bool Request::read(Connection *Conn) { return RequestReadImpl(this, Conn); }

bool Request::readFull(ByteReader *Source) {
  bool Error = false;
  while (!this->IsFullyRead && !Error) {
    bool Error = RequestReadImpl(this, Source);
  }
  return this->IsFullyRead;
}

#undef CHECK_READ_RETURN
