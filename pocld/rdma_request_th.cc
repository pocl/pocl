/* rdma_request_th.cc - pocld thread that listens to incoming requests via RDMA

   Copyright (c) 2020-2023 Jan Solanti / Tampere University

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

#include "rdma_request_th.hh"

RdmaRequestThread::RdmaRequestThread(
    VirtualContextBase *c, ExitHelper *e, std::shared_ptr<TrafficMonitor> tm,
    const char *id_str, std::shared_ptr<RdmaConnection> conn,
    std::unordered_map<uint32_t, RdmaBufferData> *mem_regions,
    std::mutex *mem_regions_mutex)
    : virtualContext(c), eh(e), netstat(tm), id_str(id_str), connection(conn),
      mem_regions(mem_regions), mem_regions_mutex(mem_regions_mutex) {
  io_thread = std::thread{&RdmaRequestThread::rdmaReaderThread, this};
}

RdmaRequestThread::~RdmaRequestThread() {
  eh->requestExit(id_str.c_str(), 0);
  io_thread.join();
}

const char *request_to_str(RequestMessageType type);

void RdmaRequestThread::rdmaReaderThread() {
  int error;

  /************ Prepare & enqueue work requests for commands ****************/
  // must be <= qp_init_attr.cap.max_recv_wr
  const size_t NUM_OUTSTANDING_REQUESTS = 5;
  RdmaBuffer<RequestMsg_t> requests_buf(
      connection->protectionDomain(), NUM_OUTSTANDING_REQUESTS,
      ibverbs::MemoryRegion::Access::LocalRead |
          ibverbs::MemoryRegion::Access::LocalWrite);

  // When the other end attempts to make an IBV_SEND request, there must be an
  // outstanding work request on the receiving end, else the send request
  // immediately fails. Enqueue a few recv requests so there should always be at
  // least one available. These are only used for the command metadata structs.
  // Buffer contents are transmitted with RDMA_WRITE and do not generate events.
  for (unsigned int i = 0; i < NUM_OUTSTANDING_REQUESTS; ++i) {
    if (netstat)
      netstat->rxRequested(sizeof(RequestMsg_t));
    try {
      connection->post(
          ReceiveRequest{i,
                         {{*requests_buf, i * (ptrdiff_t)sizeof(RequestMsg_t),
                           sizeof(RequestMsg_t)}}});
    } catch (const std::runtime_error &e) {
      POCL_MSG_ERR("%s: ERROR: Post recv request failed: %s\n", id_str.c_str(),
                   e.what());
      eh->requestExit("Posting RDMA recv request failed", -1);
      return;
    }
  }

  while (!eh->exit_requested()) {
    /**************** Wait for a work completion event ********************/

    ibverbs::WorkCompletion wc;
    try {
      wc = connection->awaitRecvCompletion();
    } catch (const std::runtime_error &e) {
      POCL_MSG_ERR("%s: ERROR: Receive failed: %s\n", id_str.c_str(), e.what());
      eh->requestExit("RDMA receive failed", -1);
      break;
    }

    if (netstat) {
      // The buffer contents are sent with IBV_WRITE and we have no way of
      // knowing the buffer size so just log the difference as requested here.
      // "Requested" statistics are going to be wacky but ah well...
      netstat->rxRequested(40); // GRH is always counted
      netstat->rxConfirmed(wc.byte_len);
      uint64_t data_size = transfer_size(requests_buf.at(wc.wr_id));
      netstat->rxRequested(data_size);
      netstat->rxConfirmed(data_size);
    }

    // The command metadata IBV_SEND is placed in the send list after the
    // RDMA_WRITE of the contents, so by this time we can be certain that the
    // transfer is complete and can safely proceed to...

    /************** Forward the Request to virtual context ******************/
    Request *request = new Request;
    request->Body = requests_buf.at(wc.wr_id);

    POCL_MSG_PRINT_GENERAL(
        "%s: received IBV_SEND for message ID: %" PRIu64 " type: %s\n",
        id_str.c_str(), uint64_t(request->Body.msg_id),
        request_to_str((RequestMessageType)request->Body.message_type));

    switch (request->Body.message_type) {
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
    case MessageType_CompileProgramFromSource:
    case MessageType_BuildProgramFromBinary:
    case MessageType_BuildProgramFromSPIRV:
    case MessageType_CompileProgramFromSPIRV:
    case MessageType_BuildProgramWithBuiltins:
    case MessageType_LinkProgram:
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
      virtualContext->notifyEvent(request->Body.event_id, CL_SUCCESS);
      delete request;
      break;
    }

    default: {
      virtualContext->unknownRequest(request);
      break;
    }
    }

    /******** Post new receive request, reusing this RequestMsg_t  ********/
    if (netstat)
      netstat->rxRequested(sizeof(RequestMsg_t));
    try {
      connection->post(ReceiveRequest{
          wc.wr_id,
          {{*requests_buf, (int)wc.wr_id * (ptrdiff_t)sizeof(RequestMsg_t),
            sizeof(RequestMsg_t)}}});
    } catch (const std::runtime_error &e) {
      POCL_MSG_ERR("%s: ERROR: Post recv request failed: %s\n", id_str.c_str(),
                   e.what());
      eh->requestExit("Posting RDMA recv request failed", -1);
      return;
    }
  }
}
