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

#include <memory>
#include <mutex>
#include <poll.h>
#include <sys/socket.h>

#include "common.hh"
#include "messages.h"
#include "request_th.hh"
#include "tracing.h"

#define CL_INVALID_OPERATION -59
#ifndef POLLRDHUP
#define POLLRDHUP 0
#endif

RequestQueueThread::RequestQueueThread(std::shared_ptr<Connection> Conn,
                                       VirtualContextBase *c, ExitHelper *e,
                                       const char *id_str)
    : InboundConnection(Conn), virtualContext(c), eh(e),
      ThreadIdentifier(id_str) {
  IOThread = std::thread{&RequestQueueThread::readThread, this};
}

RequestQueueThread::~RequestQueueThread() {
  eh->requestExit(ThreadIdentifier.c_str(), 0);
  IOThread.join();
}

void RequestQueueThread::setConnection(
    std::shared_ptr<Connection> NewConnection) {
  std::unique_lock<std::mutex> l(ConnectionGuard);
  InboundConnection = NewConnection;
  ConnectionNotifier.notify_one();
}

void RequestQueueThread::readThread() {
  ssize_t readb;

  struct pollfd pfd;
  pfd.events = POLLIN | POLLRDHUP;
  int NumEvents;

  while (1) {
    if (eh->exit_requested())
      return;

    std::unique_lock<std::mutex> l(ConnectionGuard);

    if (InboundConnection.get() == nullptr)
      ConnectionNotifier.wait(l);

    pfd.fd = InboundConnection->pollableFd();
    /* HACK: Timeout after 1s so peer request threads don't get stuck when the
     * session is being shut down. */
    NumEvents = poll(&pfd, 1, 1000);
    if (NumEvents == 0)
      continue;

    if (NumEvents < 0) {
      int e = errno;
      if (e == EINTR)
        continue;
      else {
        // Either a SERIOUS bug in the poll code above (EFAULT, EINVAL) or the
        // system is out of memory. Can't really recover from either case at
        // runtime so let's just bail.
        eh->requestExit("Fatal error during poll(2) in RequestQueueThread", e);
        return;
      }
    }
    if (pfd.revents & (POLLERR | POLLNVAL | POLLHUP | POLLRDHUP)) {
      InboundConnection.reset();
      continue;
    }
    if (!(pfd.revents & POLLIN))
      continue;

    Request *IncomingRequest = new Request();
    while (!IncomingRequest->IsFullyRead) {
      if (!IncomingRequest->read(InboundConnection.get())) {
        delete IncomingRequest;
        InboundConnection.reset();
        continue;
      }
    }

    l.unlock();

    switch (IncomingRequest->Body.message_type) {
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
    case MessageType_CompileProgramFromSource:
    case MessageType_CompileProgramFromSPIRV:
    case MessageType_LinkProgram:
    case MessageType_BuildProgramWithBuiltins:
    case MessageType_FreeProgram:
    case MessageType_MigrateD2D:
    case MessageType_RdmaBufferRegistration:
    case MessageType_Shutdown: {
      virtualContext->nonQueuedPush(IncomingRequest);
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
      virtualContext->queuedPush(IncomingRequest);
      break;
    }
    case MessageType_NotifyEvent: {
      // TODO: this message should probably contain an actual status... (see
      // also rdma thread)
      virtualContext->notifyEvent(IncomingRequest->Body.event_id, CL_COMPLETE);
      delete IncomingRequest;
      break;
    }

    default: {
      virtualContext->unknownRequest(IncomingRequest);
      break;
    }
    }

    // TODO reply fail
  }
}
