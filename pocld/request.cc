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
#include <unistd.h>

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

#define RETURN_UNLESS_DONE(call)                                               \
  do {                                                                         \
    int ret = (call);                                                          \
    if (ret) {                                                                 \
      if (ret == EAGAIN || ret == EWOULDBLOCK)                                 \
        return true;                                                           \
      POCL_MSG_ERR("Read error on " #call ", %s, reason: %s\n",                \
                   Conn->describe().c_str(), strerror(ret));                   \
      return false;                                                            \
    }                                                                          \
  } while (0);

bool Request::read(Connection *Conn) {
  ssize_t readb;

  RequestMsg_t *Body = &this->Body;

  if (!this->ReadStartTimestampNS) {
    auto now1 = std::chrono::system_clock::now();
    this->ReadStartTimestampNS =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now1.time_since_epoch())
            .count();
  }

  RETURN_UNLESS_DONE(Conn->readReentrant(
      &this->BodySize, sizeof(this->BodySize), &this->BodySizeBytesRead));

  RETURN_UNLESS_DONE(
      Conn->readReentrant(Body, this->BodySize, &this->BodyBytesRead));

  TP_MSG_RECEIVED(Body->msg_id, Body->did, Body->cq_id, Body->message_type);

  RequestMessageType t = static_cast<RequestMessageType>(Body->message_type);
  POCL_MSG_PRINT_GENERAL("---------------------------------------------------"
                         "----------------------------\n");
  POCL_MSG_PRINT_GENERAL("MESSAGE RECEIVED, ID: %" PRIu64
                         " TYPE: %s SIZE: %" PRIu64 "/%" PRIu32 " \n",
                         uint64_t(Body->msg_id), request_to_str(t),
                         this->BodyBytesRead, this->BodySize);

  switch (Body->message_type) {
  case MessageType_WriteBuffer:
    this->ExtraDataSize = Body->m.write.size;
    break;
  case MessageType_WriteBufferRect:
    this->ExtraDataSize = Body->m.write_rect.host_bytes;
    break;
  case MessageType_WriteImageRect:
    this->ExtraDataSize = Body->m.write_image_rect.host_bytes;
    break;
  case MessageType_MigrateD2D:
    if (Body->m.migrate.is_external) {
      this->ExtraDataSize = Body->m.migrate.size;
    }
    break;
  case MessageType_FillBuffer:
    this->ExtraDataSize = Body->m.fill_buffer.pattern_size;
    assert(this->ExtraDataSize <= (16 * sizeof(uint64_t)));
    break;
  case MessageType_FillImageRect:
    this->ExtraDataSize = 16;
    break;
  case MessageType_RunKernel:
    if (Body->m.run_kernel.has_new_args) {
      /* The arguments itthis come in through extra data, as well as an array of
         flags which inform whether an argument (buffer) is an
         SVM pointer or not. */
      this->ExtraDataSize = Body->m.run_kernel.args_num * sizeof(uint64_t) +
                            Body->m.run_kernel.args_num * sizeof(unsigned char);
      this->ExtraData2Size = Body->m.run_kernel.pod_arg_size;
    }
    break;
  /*****************************/
  case MessageType_BuildProgramFromBinary:
  case MessageType_BuildProgramFromSource:
  case MessageType_BuildProgramFromSPIRV:
  case MessageType_CompileProgramFromSPIRV:
  case MessageType_CompileProgramFromSource:
  case MessageType_LinkProgram:
    this->ExtraData2Size = Body->m.build_program.options_len;
    /* intentional fall through to setting payload (i.e. binary) size */
  case MessageType_BuildProgramWithBuiltins:
    this->ExtraDataSize = Body->m.build_program.payload_size;
    break;
  /*****************************/
  case MessageType_CreateKernel:
    this->ExtraDataSize = Body->m.create_kernel.name_len;
    break;
  default:
    break;
  }

  /*****************************/
  if (Body->waitlist_size > 0) {
    this->Waitlist.resize(Body->waitlist_size);
    POCL_MSG_PRINT_GENERAL("READING WAIT LIST FOR ID: %" PRIu64 " = %" PRIuS
                           "/%" PRIu32 "\n",
                           uint64_t(Body->msg_id), this->WaitlistBytesRead,
                           this->Body.waitlist_size);
    RETURN_UNLESS_DONE(Conn->readReentrant(
        this->Waitlist.data(), this->Body.waitlist_size * sizeof(uint64_t),
        &this->WaitlistBytesRead));
  }
  /*****************************/

  /*****************************/
  if (this->ExtraDataSize > 0) {
    this->ExtraData.resize(this->ExtraDataSize + 1);
    POCL_MSG_PRINT_GENERAL(
        "READING EXTRA FOR ID: %" PRIu64 " = %" PRIuS "/%" PRIu64 "\n",
        uint64_t(Body->msg_id), this->ExtraDataBytesRead, this->ExtraDataSize);
    RETURN_UNLESS_DONE(Conn->readReentrant(this->ExtraData.data(),
                                           this->ExtraDataSize,
                                           &this->ExtraDataBytesRead));
    /* Always add a null byte at the end - it is needed for strings and it does
     * not harm other things */
    this->ExtraData[this->ExtraDataSize] = 0;
  }
  /*****************************/

  /*****************************/
  if (this->ExtraData2Size > 0) {
    this->ExtraData2.resize(this->ExtraData2Size + 1);
    POCL_MSG_PRINT_GENERAL("READING EXTRA2 FOR ID:%" PRIu64 " = %" PRIuS
                           "/%" PRIu64 "\n",
                           uint64_t(Body->msg_id), this->ExtraData2BytesRead,
                           this->ExtraData2Size);
    RETURN_UNLESS_DONE(Conn->readReentrant(this->ExtraData2.data(),
                                           this->ExtraData2Size,
                                           &this->ExtraData2BytesRead));
    /* Always add null byte here too, just in case extra2 is a string */
    this->ExtraData2[this->ExtraData2Size] = 0;
  }
  /*****************************/

  if (!this->ReadEndTimestampNS) {
    auto now2 = std::chrono::system_clock::now();
    this->ReadEndTimestampNS =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now2.time_since_epoch())
            .count();
  }

  POCL_MSG_PRINT_GENERAL("ALL READS COMPLETE FOR ID: %" PRIu64 ", %s\n",
                         uint64_t(Body->msg_id), Conn->describe().c_str());

  IsFullyRead = true;
  return true;
}

#undef CHECK_READ_RETURN
