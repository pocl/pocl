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
  pfd.events = POLLIN | POLLRDHUP;
  int nevs;

  int fd = *this->fd;
  int oldfd = fd;
  while (1) {
    fd = *this->fd;
    if (fd != oldfd) {
      POCL_MSG_PRINT_GENERAL("%s: FD change detected: %d -> %d\n",
                             id_str.c_str(), oldfd, fd);
    }
    oldfd = fd;
    if (eh->exit_requested())
      return;

    pfd.fd = fd;
    nevs = poll(&pfd, 1, 3 * MS_PER_S);
    if (nevs < 1)
      continue;
    if (pfd.revents & (POLLERR | POLLNVAL | POLLHUP | POLLRDHUP))
      continue;
    if (!(pfd.revents & POLLIN))
      continue;

    Request *request = new Request();
    while (!request->IsFullyRead) {
      if (!request->read(fd)) {
        delete request;
        continue;
      }
    }

    switch (request->req.message_type) {
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
