/* virtual_cl_context.cc - pocld class that holds all resources of a session

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
#include <memory>
#include <unordered_set>

#include "common.hh"

#ifdef ENABLE_RDMA
#include "rdma_reply_th.hh"
#include "rdma_request_th.hh"
#endif

#include "shared_cl_context.hh"
#include "virtual_cl_context.hh"

#include "daemon.hh"
#include "peer_handler.hh"
#include "reply_th.hh"
#include "tracing.h"
#include "traffic_monitor.hh"

#include "messages.h"

SharedContextBase *createSharedCLContext(cl::Platform *platform, size_t pid,
                                         VirtualContextBase *v,
                                         ReplyQueueThread *slow,
                                         ReplyQueueThread *fast);

/****************************************************************************************************************/
/****************************************************************************************************************/
/****************************************************************************************************************/

#define INIT_VARS                                                              \
  int err = CL_SUCCESS;                                                        \
  size_t i, j;                                                                 \
  uint64_t id = req->req.obj_id;

#define FOR_EACH_CONTEXT_DO(CODE)                                              \
  for (i = 0; i < SharedContextList.size(); ++i) {                             \
    err = SharedContextList[i]->CODE;                                          \
    if (err != CL_SUCCESS)                                                     \
      break;                                                                   \
  }

#define FOR_EACH_CONTEXT_UNDO(CODE)                                            \
  if (err != CL_SUCCESS) {                                                     \
    for (j = 0; j < i; ++j) {                                                  \
      SharedContextList[i]->CODE;                                              \
    }                                                                          \
  }

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

typedef std::vector<SharedContextBase *> ContextVector;

#ifdef ENABLE_RDMA
class SVMDeleter {
  cl_context _c;

public:
  SVMDeleter(cl_context c) { _c = c; }
  void operator()(char *p) { clSVMFree(_c, p); }
};
#endif

class VirtualCLContext : public VirtualContextBase {
  PoclDaemon *Daemon;
  ReplyQueueThreadUPtr write_slow;
  ReplyQueueThreadUPtr write_fast;
#ifdef ENABLE_RDMA
  RdmaReplyThreadUPtr write_rdma;
  RdmaRequestThreadUPtr read_rdma;
#endif
  PeerHandlerUPtr peers;
  uint32_t peer_id;
  std::atomic_int command_fd;
  std::atomic_int stream_fd;

  ExitHelper exit_helper;

  std::vector<cl::Platform> PlatformList;
  std::vector<SharedContextBase *> SharedContextList;
  size_t TotalDevices;

  size_t current_printf_position;
  std::mutex printf_lock;

  std::condition_variable main_cond;
  std::mutex main_mutex;
  std::deque<Request *> main_que;

#ifdef ENABLE_RDMA
  std::shared_ptr<RdmaConnection> client_rdma;
#ifdef RDMA_USE_SVM
  std::unordered_map<uint32_t, std::unique_ptr<char, SVMDeleter>>
      rdma_shadow_buffers;
#else
  std::unordered_map<uint32_t, std::unique_ptr<char[]>> rdma_shadow_buffers;
#endif
  std::unordered_map<uint32_t, RdmaBufferData> client_mem_regions;
  std::mutex client_regions_mutex;
  uint32_t client_uses_rdma;
#endif
  TrafficMonitor *netstat;

  std::unordered_set<uint32_t> BufferIDset;
  std::unordered_set<uint32_t> SamplerIDset;
  std::unordered_set<uint32_t> ImageIDset;
  std::unordered_set<uint32_t> QueueIDset;
  std::unordered_set<uint32_t> ProgramIDset;
  std::unordered_set<uint32_t> KernelIDset;

  std::unordered_map<uint32_t, ContextVector> ProgramPlatformBuildMap;

public:
  // create device / platform / context backends
  VirtualCLContext() = default;

  ~VirtualCLContext() {
    // stop threads
    assert(exit_helper.exit_requested());
    POCL_MSG_PRINT_GENERAL("VCTX: DEST\n");

    // make sure no shared context tries to broadcast stuff
    std::unique_lock<std::mutex> lock(main_mutex);
    // TOFIX: peers cannot be freed without crashing, let them leak for now.
    // peers.reset();
    for (auto i : SharedContextList) {
      delete i;
    }
    SharedContextList.clear();
    PlatformList.clear();
  }

  /****************************************************************************************************************/
  virtual size_t init(PoclDaemon *d, ClientConnections_t conns,
                      uint64_t session, CreateOrAttachSessionMsg_t &params);

  virtual void updateSockets(std::optional<int> command_fd,
                             std::optional<int> stream_fd) override;

  virtual void nonQueuedPush(Request *req) override;

  virtual void queuedPush(Request *req) override;

#ifdef ENABLE_RDMA
  virtual bool clientUsesRdma() override { return (client_uses_rdma != 0); };

  virtual char *getRdmaShadowPtr(uint32_t id) override {
    return rdma_shadow_buffers.at(id).get();
  };
#endif

  virtual void requestExit(int code, const char *reason) override;

  virtual void broadcastToPeers(const Request &req) override;

  virtual void notifyEvent(uint64_t event_id, cl_int status) override;

  virtual void unknownRequest(Request *req) override;

  virtual int run() override;

  virtual SharedContextBase *getDefaultContext() override {
    return SharedContextList.empty() ? nullptr : SharedContextList[0];
  };

private:
  int checkPlatformDeviceValidity(Request *req);

  size_t initPlatforms();

  void ServerInfo(Request *req, Reply *rep);

  void ConnectPeer(Request *req, Reply *rep);

  void CreateCmdQueue(Request *req, Reply *rep);

  void FreeCmdQueue(Request *req, Reply *rep);

  void CreateBuffer(Request *req, Reply *rep);

  void FreeBuffer(Request *req, Reply *rep);

  void BuildProgram(Request *req, Reply *rep, bool is_binary, bool is_builtin,
                    bool is_spirv);

  void FreeProgram(Request *req, Reply *rep);

  void CreateKernel(Request *req, Reply *rep);

  void FreeKernel(Request *req, Reply *rep);

  void CreateSampler(Request *req, Reply *rep);

  void FreeSampler(Request *req, Reply *rep);

  void CreateImage(Request *req, Reply *rep);

  void FreeImage(Request *req, Reply *rep);

  void DeviceInfo(Request *req, Reply *rep);

  void MigrateD2D(Request *req);
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

size_t VirtualCLContext::init(PoclDaemon *d, ClientConnections_t conns,
                              uint64_t session,
                              CreateOrAttachSessionMsg_t &params) {

  Daemon = d;
  current_printf_position = 0;
  TotalDevices = 0;
  command_fd = conns.fd_command;
  stream_fd = conns.fd_stream;
  peer_id = params.peer_id;
#ifdef ENABLE_RDMA
  client_uses_rdma = params.use_rdma;
  if (client_uses_rdma) {
    client_rdma = conns.rdma;
  }
#endif

  std::string id_string = std::to_string(session);
  netstat = new TrafficMonitor(&exit_helper, id_string);

#ifdef ENABLE_RDMA
  if (client_uses_rdma) {
    read_rdma = RdmaRequestThreadUPtr(
        new RdmaRequestThread(this, &exit_helper, netstat, "RDMA_R", conns.rdma,
                              &client_mem_regions, &client_regions_mutex));
    write_rdma = RdmaReplyThreadUPtr(
        new RdmaReplyThread(this, &exit_helper, netstat, "RDMA_W", conns.rdma,
                            &client_mem_regions, &client_regions_mutex));
  }
#endif
  write_slow = ReplyQueueThreadUPtr(
      new ReplyQueueThread(&stream_fd, this, &exit_helper, netstat, "WT_S"));
  write_fast = ReplyQueueThreadUPtr(
      new ReplyQueueThread(&command_fd, this, &exit_helper, netstat, "WT_F"));

  peers = PeerHandlerUPtr(new PeerHandler(peer_id, conns.incoming_peer_mutex,
                                          conns.incoming_peer_queue, this,
                                          &exit_helper, netstat));
  initPlatforms();

  POCL_MSG_PRINT_INFO("Created shared contexts for %" PRIuS
                      " platforms / %" PRIuS " devices\n",
                      PlatformList.size(), TotalDevices);

  return TotalDevices;
}

void VirtualCLContext::updateSockets(std::optional<int> fd_command,
                                     std::optional<int> fd_stream) {
  if (fd_command.has_value())
    command_fd = fd_command.value();
  if (fd_stream.has_value())
    stream_fd = fd_stream.value();
}

size_t VirtualCLContext::initPlatforms() {
  cl::Platform::get(&PlatformList);
  if (PlatformList.size() == 0) {
    POCL_MSG_ERR("No platforms found. \n");
    return 0;
  }

  SharedContextList.resize(PlatformList.size());

  for (size_t i = 0; i < PlatformList.size(); ++i) {
    SharedContextBase *p = createSharedCLContext(
        &(PlatformList[i]), i, this, write_slow.get(), write_fast.get());

    SharedContextList[i] = p;
    TotalDevices += p->numDevices();
  }

  POCL_MSG_PRINT_GENERAL("Initialized %" PRIuS
                         " platform%s with a total of %" PRIuS " device%s\n",
                         PlatformList.size(), PlatformList.empty() ? "" : "s",
                         TotalDevices, TotalDevices == 0 ? "" : "s");

  return PlatformList.size();
}

void VirtualCLContext::nonQueuedPush(Request *req) {

  if (req->req.message_type != MessageType_ServerInfo &&
      checkPlatformDeviceValidity(req))
    return;

  POCL_MSG_PRINT_GENERAL("VCTX NON-QUEUED PUSH (msg: %" PRIu64 ")\n",
                         uint64_t(req->req.msg_id));

  std::unique_lock<std::mutex> lock(main_mutex);
  main_que.push_back(req);
  main_cond.notify_one();
}

void VirtualCLContext::queuedPush(Request *req) {

  if (checkPlatformDeviceValidity(req))
    return;

  POCL_MSG_PRINT_GENERAL(
      "VCTX QUEUED PUSH (msg: %" PRIu64 ", event: %" PRIu64 ")\n",
      uint64_t(req->req.msg_id), uint64_t(req->req.event_id));
  SharedContextList[req->req.pid]->queuedPush(req);
}

void VirtualCLContext::notifyEvent(uint64_t event_id, cl_int status) {
  POCL_MSG_PRINT_EVENTS("Updating event %" PRIu64 " status to %d\n", event_id,
                        status);
  for (auto ctx : SharedContextList) {
    ctx->notifyEvent(event_id, status);
  }
}

void VirtualCLContext::requestExit(int code, const char *reason) {
  exit_helper.requestExit(reason, code);
}

void VirtualCLContext::broadcastToPeers(const Request &req) {
  std::unique_lock<std::mutex> lock(main_mutex);
  peers->broadcast(req);
}

void VirtualCLContext::unknownRequest(Request *req) {
  Reply *rep = new Reply(req);
  POCL_MSG_ERR("Unknown request type: %d\n", req->req.message_type);
  replyFail(&rep->rep, &req->req, CL_INVALID_OPERATION);
  write_fast->pushReply(rep);
}

int VirtualCLContext::checkPlatformDeviceValidity(Request *req) {
  if (req->req.message_type == MessageType_RdmaBufferRegistration)
    return 0;
  if (req->req.message_type == MessageType_MigrateD2D &&
      peer_id == req->req.m.migrate.source_peer_id) {
    uint32_t pid = req->req.m.migrate.source_pid;
    uint32_t did = req->req.m.migrate.source_pid;
    if ((pid < PlatformList.size()) &&
        (did < SharedContextList[pid]->numDevices()))
      return 0;
  }

  uint32_t pid = req->req.pid;
  uint32_t did = req->req.did;
  if ((pid < PlatformList.size()) &&
      (did < SharedContextList[pid]->numDevices()))
    return 0;

  Reply *reply = new Reply(req);

  int err =
      (pid < PlatformList.size() ? CL_INVALID_DEVICE : CL_INVALID_PLATFORM);
  replyFail(&reply->rep, &req->req, err);

  POCL_MSG_ERR("Message ID %" PRIu64 ": Unknown Platform ID %" PRIu32
               " or Device ID %" PRIu32 "\n",
               uint64_t(req->req.msg_id), uint32_t(req->req.pid),
               uint32_t(req->req.did));

  write_fast->pushReply(reply);
  return 1;
}

/****************************************************************************************************************/
/****************************************************************************************************************/

int VirtualCLContext::run() {
  Reply *reply;
  while (1) {

    if (exit_helper.exit_requested()) {
      auto e = exit_helper.status();
      POCL_MSG_PRINT_GENERAL("VCTX: exit req, status: %d\n", e);
      return e;
    }

    std::unique_lock<std::mutex> lock(main_mutex);
    if (main_que.size() > 0) {
      Request *request = main_que.front();
      main_que.pop_front();
      lock.unlock();

      reply = nullptr;
      if (request->req.message_type != MessageType_MigrateD2D &&
          request->req.message_type != MessageType_RdmaBufferRegistration) {
        reply = new Reply(request);
      }

      // PROCESSS REQUEST, then PUSH REPLY to WRITE Q

      switch (request->req.message_type) {
      case MessageType_ServerInfo:
        ServerInfo(request, reply);
        break;

      case MessageType_DeviceInfo:
        DeviceInfo(request, reply);
        break;

      case MessageType_ConnectPeer:
        ConnectPeer(request, reply);
        break;

      case MessageType_CreateBuffer:
        CreateBuffer(request, reply);
        break;

      case MessageType_FreeBuffer:
        FreeBuffer(request, reply);
        break;

      case MessageType_CreateCommandQueue:
        CreateCmdQueue(request, reply);
        break;

      case MessageType_FreeCommandQueue:
        FreeCmdQueue(request, reply);
        break;

      case MessageType_BuildProgramFromBinary:
        BuildProgram(request, reply, true, false, false);
        break;

      case MessageType_BuildProgramFromSource:
        BuildProgram(request, reply, false, false, false);
        break;

      case MessageType_BuildProgramFromSPIRV:
        BuildProgram(request, reply, false, false, true);
        break;

      case MessageType_BuildProgramWithBuiltins:
        BuildProgram(request, reply, false, true, false);
        break;

      case MessageType_FreeProgram:
        FreeProgram(request, reply);
        break;

      case MessageType_CreateKernel:
        CreateKernel(request, reply);
        break;

      case MessageType_FreeKernel:
        FreeKernel(request, reply);
        break;

      case MessageType_CreateSampler:
        CreateSampler(request, reply);
        break;

      case MessageType_FreeSampler:
        FreeSampler(request, reply);
        break;

      case MessageType_CreateImage:
        CreateImage(request, reply);
        break;

      case MessageType_FreeImage:
        FreeImage(request, reply);
        break;

      case MessageType_MigrateD2D:
        MigrateD2D(request);
        break;

      case MessageType_Shutdown:
        exit_helper.requestExit("Shutdown notification from client", 0);
        return 0;

      case MessageType_RdmaBufferRegistration:
#ifdef ENABLE_RDMA
        // unused existing fields are being repurposed for rdma info here:
        // uint32_t peer_id, uint32_t buf_id, uint32_t rkey, uint64_t vaddr
        peers->notifyRdmaBufferRegistration(
            request->req.cq_id, request->req.obj_id, request->req.did,
            request->req.msg_id);
#endif
        // Just ignore, this does not require a reply
        delete request;
        continue;

      default:
        reply->rep.data_size = 0;
        reply->rep.fail_details = CL_INVALID_OPERATION;
        reply->rep.failed = 1;
        reply->rep.message_type = MessageType_Failure;
        reply->extra_data.reset();
        reply->extra_size = 0;
        POCL_MSG_ERR("Unknown message type received: %" PRIu32 "\n",
                     uint32_t(request->req.message_type));
      }

      if (reply) {
        write_fast->pushReply(reply);
        // Reply frees the request when destroyed
      }

    } else {
      auto now = std::chrono::system_clock::now();
      std::chrono::duration<unsigned long> d(3);
      now += d;
      main_cond.wait_until(lock, now);
    }
  }
}

/****************************************************************************************************************/
/****************************************************************************************************************/

void VirtualCLContext::ConnectPeer(Request *req, Reply *rep) {
  INIT_VARS;
  std::array<uint8_t, AUTHKEY_LENGTH> authkey;
  std::memcpy(authkey.data(), req->req.m.connect_peer.authkey, AUTHKEY_LENGTH);
  err = peers->connectPeer(req->req.msg_id, req->req.m.connect_peer.address,
                           req->req.m.connect_peer.port,
                           req->req.m.connect_peer.session, authkey);
  RETURN_IF_ERR;
  replyOK(rep, MessageType_ConnectPeerReply);
}

/****************************************************************************************************************/

void VirtualCLContext::CreateCmdQueue(Request *req, Reply *rep) {
  INIT_VARS;
  CHECK_ID_NOT_EXISTS(QueueIDset, CL_INVALID_COMMAND_QUEUE);

  TP_CREATE_QUEUE(req->req.msg_id, req->req.client_did, id);
  err = SharedContextList[req->req.pid]->createQueue(
      id, req->req.did); // TODO queue flags
  TP_CREATE_QUEUE(req->req.msg_id, req->req.client_did, id);

  RETURN_IF_ERR;
  QueueIDset.insert(id);
  replyID(rep, MessageType_CreateCommandQueueReply, id);
}

void VirtualCLContext::FreeCmdQueue(Request *req, Reply *rep) {
  INIT_VARS;
  CHECK_ID_EXISTS(QueueIDset, CL_INVALID_COMMAND_QUEUE);

  TP_FREE_QUEUE(req->req.msg_id, req->req.client_did, id);
  err = SharedContextList[req->req.pid]->freeQueue(id);
  TP_FREE_QUEUE(req->req.msg_id, req->req.client_did, id);

  QueueIDset.erase(id);
  RETURN_IF_ERR;
  replyOK(rep, MessageType_FreeCommandQueueReply);
}

/****************************************************************************************************************/

void VirtualCLContext::CreateBuffer(Request *req, Reply *rep) {
  INIT_VARS;
  CHECK_ID_NOT_EXISTS(BufferIDset, CL_INVALID_MEM_OBJECT);

  CreateBufferMsg_t &m = req->req.m.create_buffer;

#ifdef ENABLE_RDMA
  uint32_t buf_size = m.size;
#ifdef RDMA_USE_SVM
  // SVM pointers only work within a single context
  assert(SharedContextList.size() == 1);
  char *b = (char *)clSVMAlloc(SharedContextList.front()->getHandle()(),
                               CL_MEM_READ_WRITE, buf_size, 0);
  if (!b) {
    POCL_MSG_ERR("SVM allocation of size %" PRIu32 " failed\n", buf_size);
    replyFail(&rep->rep, &req->req, CL_MEM_OBJECT_ALLOCATION_FAILURE);
    return;
  }
  std::unique_ptr<char, SVMDeleter> shadow_buf(
      b, SVMDeleter(SharedContextList.front()->getHandle()()));
#else
  std::unique_ptr<char[]> shadow_buf = std::make_unique<char[]>(buf_size);
#endif
  ibverbs::MemoryRegionPtr shadow_region = nullptr;
  if (client_uses_rdma) {
    try {
      POCL_MSG_PRINT_GENERAL("Registering memory region of size %" PRIu32 "\n",
                             buf_size);
      shadow_region = ibverbs::MemoryRegion::register_ptr(
          client_rdma->protectionDomain(), shadow_buf.get(), buf_size,
          ibverbs::MemoryRegion::Access::LocalWrite |
              ibverbs::MemoryRegion::Access::RemoteWrite |
              ibverbs::MemoryRegion::Access::RemoteRead);
      POCL_MSG_PRINT_GENERAL("Posting receive request\n");
      client_rdma->post(ReceiveRequest{0, {{shadow_region}}});
    } catch (const std::runtime_error &e) {
      POCL_MSG_ERR("%s", e.what());
      err = 1;
      replyFail(&rep->rep, &req->req, CL_MEM_OBJECT_ALLOCATION_FAILURE);
      FOR_EACH_CONTEXT_DO(freeBuffer(id, false));
      return;
    }

    RdmaBufferData meta = {shadow_buf.get(), shadow_region};
    client_mem_regions.insert({id, meta});
  }
  peers->rdmaRegisterBuffer(id, shadow_buf.get(), buf_size);
  rdma_shadow_buffers.insert({id, std::move(shadow_buf)});
#endif

  TP_CREATE_BUFFER(req->req.msg_id, req->req.client_did, id);

  uint64_t devaddr;
  FOR_EACH_CONTEXT_DO(
      createBuffer(id, m.size, m.flags, (void *)m.host_ptr, (void **)&devaddr));
  // Do not pass pointer to device_addr directly above since
  // it's a packed struct and the address might be unaligned.
  rep->rep.m.create_buffer.device_addr = devaddr;
  TP_CREATE_BUFFER(req->req.msg_id, req->req.client_did, id);
  FOR_EACH_CONTEXT_UNDO(freeBuffer(id, false));

  RETURN_IF_ERR;
  BufferIDset.insert(id);
  replyID(rep, MessageType_CreateBufferReply, id);
#ifdef ENABLE_RDMA
  if (client_uses_rdma) {
    CreateRdmaBufferReply_t *ext = new CreateRdmaBufferReply_t;
    ext->server_rkey = shadow_region->rkey();
    ext->server_vaddr = (uint64_t)shadow_buf.get();
    rep->rep.data_size = sizeof(CreateRdmaBufferReply_t);
    rep->extra_size = sizeof(CreateRdmaBufferReply_t);
    rep->extra_data.reset((uint8_t *)(ext));
  }
#endif
}

void VirtualCLContext::FreeBuffer(Request *req, Reply *rep) {
  INIT_VARS;
  if (!req->req.m.free_buffer.is_svm)
    CHECK_ID_EXISTS(BufferIDset, CL_INVALID_MEM_OBJECT);

  TP_FREE_BUFFER(req->req.msg_id, req->req.client_did, id);
  FOR_EACH_CONTEXT_DO(freeBuffer(id, req->req.m.free_buffer.is_svm));
  TP_FREE_BUFFER(req->req.msg_id, req->req.client_did, id);

  BufferIDset.erase(id);
#ifdef ENABLE_RDMA
  peers->rdmaUnregisterBuffer(id);
  if (client_uses_rdma) {
    client_mem_regions.erase(id);
  }
  rdma_shadow_buffers.erase(id);
#endif
  RETURN_IF_ERR;
  replyOK(rep, MessageType_FreeBufferReply);
}

#define WRITE_BYTES(var)                                                       \
  std::memcpy(buf, &var, sizeof(var));                                         \
  buf += sizeof(var);                                                          \
  assert((size_t)(buf - buffer) <= buffer_size);
#define WRITE_STRING(str, len)                                                 \
  std::memcpy(buf, str, len);                                                  \
  buf += len;                                                                  \
  assert((size_t)(buf - buffer) <= buffer_size);

void VirtualCLContext::BuildProgram(Request *req, Reply *rep, bool is_binary,
                                    bool is_builtin, bool is_spirv) {
  INIT_VARS;
  CHECK_ID_NOT_EXISTS(ProgramIDset, CL_INVALID_PROGRAM);

  BuildProgramMsg_t &m = req->req.m.build_program;

  POCL_MSG_PRINT_GENERAL(
      "VirtualCTX: build %s program %" PRIu64 " for %" PRIu32 " devices\n",
      (is_binary ? "binary"
                 : (is_builtin ? "builtin" : (is_spirv ? "SPIR-V" : "source"))),
      id, uint32_t(m.num_devices));
  // source / binary / builtin / SPIR-V must be provided as a payload
  assert(req->extra_size > 0);
  assert(m.num_devices);
  std::vector<uint32_t> DevList{};
  ContextVector ProgramContexts;

  std::unordered_map<uint64_t, std::vector<unsigned char>> output_binaries;
  std::unordered_map<uint64_t, std::vector<unsigned char>> input_binaries;
  std::unordered_map<uint64_t, std::string> build_logs;
  size_t num_kernels = 0;
  char *source = req->extra_data;
  size_t source_len = m.payload_size;
  char *options = req->extra_data2;

  if (is_binary || is_spirv) {
    source = nullptr;
    source_len = 0;
    unsigned char *buffer = (unsigned char *)req->extra_data;
    assert(req->extra_size == m.payload_size);
    unsigned char *buf = buffer;
    size_t buffer_size = m.payload_size;
    uint32_t n_binaries;
    n_binaries = *((uint32_t *)buf);
    buf += sizeof(uint32_t);
    assert(n_binaries == m.num_devices);

    for (i = 0; i < n_binaries; ++i) {
      uint32_t bin_size;
      bin_size = *((uint32_t *)buf);
      buf += sizeof(uint32_t);

      std::vector<unsigned char> binary(buf, buf + bin_size);
      buf += bin_size;
      assert((buf - buffer) <= (size_t)m.payload_size);
      uint64_t id = ((uint64_t)m.platforms[i] << 32) + m.devices[i];
      input_binaries[id] = std::move(binary);
    }
  }

  TP_BUILD_PROGRAM(req->req.msg_id, req->req.client_did, id);
  for (i = 0; i < SharedContextList.size(); ++i) {
    DevList.clear();
    for (j = 0; j < m.num_devices; ++j) {
      if (m.platforms[j] == i)
        DevList.push_back(m.devices[j]);
    }
    if (DevList.size() > 0) {
      err = SharedContextList[i]->buildProgram(
          id, DevList, source, source_len, is_binary, is_builtin, is_spirv,
          options, input_binaries, output_binaries, build_logs, num_kernels);
      if (err == CL_SUCCESS) {
        ProgramContexts.push_back(SharedContextList[i]);
      } else
        break;
    }
  }
  TP_BUILD_PROGRAM(req->req.msg_id, req->req.client_did, id);

  // output reply
  rep->extra_data.reset((uint8_t*)new char[MAX_REMOTE_BUILDPROGRAM_SIZE]);
  char *buffer = (char*)(rep->extra_data.get());
  size_t buffer_size = MAX_REMOTE_BUILDPROGRAM_SIZE;
  char *buf = buffer;

  // write build logs even on error
  WRITE_BYTES(m.num_devices);
  for (j = 0; j < m.num_devices; ++j) {
    uint64_t id = ((uint64_t)m.platforms[i] << 32) + m.devices[i];
    if (build_logs.find(id) == build_logs.end()) {
      uint32_t zero = 0;
      WRITE_BYTES(zero);
    } else {
      uint32_t build_log_size = build_logs[id].size();
      WRITE_BYTES(build_log_size);
      WRITE_STRING(build_logs[id].data(), build_log_size);
    }
  }

  if (err == CL_SUCCESS) {
    // write metadata
    assert(ProgramContexts.size() > 0);
    size_t kernel_meta_size = 0;
    ProgramContexts[0]->writeKernelMeta(id, buf, &kernel_meta_size);
    POCL_MSG_PRINT_GENERAL("Kernel meta size: %" PRIuS " \n", kernel_meta_size);
    assert(kernel_meta_size > 0);
    buf += kernel_meta_size;
  }

  if (err == CL_SUCCESS) {
    if (!is_binary && !is_builtin && !is_spirv) {
      // write binaries for source builds
      WRITE_BYTES(m.num_devices);
      for (j = 0; j < m.num_devices; ++j) {
        uint64_t id = (((uint64_t)m.platforms[j]) << 32) + m.devices[j];
        POCL_MSG_PRINT_GENERAL(
            "Looking for binary for Dev ID: %" PRIu32 " / %" PRIu32 " \n",
            uint32_t(m.platforms[j]), uint32_t(m.devices[j]));
        uint32_t binary_size = output_binaries[id].size();
        WRITE_BYTES(binary_size);
        assert(binary_size);
        WRITE_STRING(output_binaries[id].data(), binary_size);
      }
    }
  } else {
    for (auto sc : ProgramContexts) {
      sc->freeProgram(id);
    }
    ProgramContexts.clear();
  }

  rep->extra_size = (buf - buffer);

  RETURN_IF_ERR_DATA;
  ProgramIDset.insert(id);
  ProgramPlatformBuildMap[id] = std::move(ProgramContexts);
  replyData(rep, MessageType_BuildProgramReply, id, rep->extra_size);
}

void VirtualCLContext::FreeProgram(Request *req, Reply *rep) {
  INIT_VARS;
  CHECK_ID_EXISTS(ProgramIDset, CL_INVALID_PROGRAM);
  CHECK_ID_EXISTS2(ProgramPlatformBuildMap, CL_INVALID_PROGRAM, id);

  ContextVector &contexts = ProgramPlatformBuildMap[id];

  TP_FREE_PROGRAM(req->req.msg_id, req->req.client_did, id);
  for (i = 0; i < contexts.size(); ++i) {
    err = contexts[i]->freeProgram(id);
  }
  TP_FREE_PROGRAM(req->req.msg_id, req->req.client_did, id);

  ProgramIDset.erase(id);
  RETURN_IF_ERR;
  replyOK(rep, MessageType_FreeProgramReply);
}

void VirtualCLContext::CreateKernel(Request *req, Reply *rep) {
  INIT_VARS;
  CHECK_ID_NOT_EXISTS(KernelIDset, CL_INVALID_KERNEL);

  CreateKernelMsg_t &m = req->req.m.create_kernel;
  CHECK_ID_EXISTS2(ProgramPlatformBuildMap, CL_INVALID_PROGRAM,
                   uint32_t(m.prog_id));

  ContextVector &contexts = ProgramPlatformBuildMap[m.prog_id];

  TP_CREATE_KERNEL(req->req.msg_id, req->req.client_did, id);
  for (i = 0; i < contexts.size(); ++i) {
    err = contexts[i]->createKernel(id, m.prog_id, req->extra_data);
    if (err != CL_SUCCESS)
      break;
  }
  TP_CREATE_KERNEL(req->req.msg_id, req->req.client_did, id);
  if (err != CL_SUCCESS) {
    for (j = 0; j < i; ++j) {
      err = contexts[i]->freeKernel(id);
    }
  }

  RETURN_IF_ERR;
  KernelIDset.insert(id);
  replyID(rep, MessageType_CreateKernelReply, id);
}

void VirtualCLContext::FreeKernel(Request *req, Reply *rep) {
  INIT_VARS;
  CHECK_ID_EXISTS(KernelIDset, CL_INVALID_KERNEL);

  FreeKernelMsg_t &m = req->req.m.free_kernel;
  CHECK_ID_EXISTS2(ProgramPlatformBuildMap, CL_INVALID_PROGRAM,
                   uint32_t(m.prog_id));

  ContextVector &contexts = ProgramPlatformBuildMap[m.prog_id];

  TP_FREE_KERNEL(req->req.msg_id, req->req.client_did, req->req.obj_id);
  for (i = 0; i < contexts.size(); ++i) {
    err = contexts[i]->freeKernel(id);
  }
  TP_FREE_KERNEL(req->req.msg_id, req->req.client_did, req->req.obj_id);

  KernelIDset.erase(id);
  RETURN_IF_ERR;
  replyOK(rep, MessageType_FreeKernelReply);
}

/****************************************************************************************************************/

void VirtualCLContext::CreateImage(Request *req, Reply *rep) {
  INIT_VARS;
  CHECK_ID_NOT_EXISTS(ImageIDset, CL_INVALID_MEM_OBJECT);

  CreateImageMsg_t &m = req->req.m.create_image;

  TP_CREATE_IMAGE(req->req.msg_id, req->req.client_did, id);
  FOR_EACH_CONTEXT_DO(createImage(
      id, m.flags, m.channel_order, m.channel_data_type, m.type, m.width,
      m.height, m.depth, m.array_size, m.row_pitch, m.slice_pitch));
  TP_CREATE_IMAGE(req->req.msg_id, req->req.client_did, id);

  FOR_EACH_CONTEXT_UNDO(freeImage(id));

  RETURN_IF_ERR;
  ImageIDset.insert(id);
  replyID(rep, MessageType_CreateImageReply, id);
}

void VirtualCLContext::FreeImage(Request *req, Reply *rep) {
  INIT_VARS;
  CHECK_ID_EXISTS(ImageIDset, CL_INVALID_MEM_OBJECT);

  TP_FREE_IMAGE(req->req.msg_id, req->req.client_did, req->req.obj_id);
  FOR_EACH_CONTEXT_DO(freeImage(id));
  TP_FREE_IMAGE(req->req.msg_id, req->req.client_did, req->req.obj_id);

  ImageIDset.erase(id);
  RETURN_IF_ERR;
  replyOK(rep, MessageType_FreeImageReply);
}

/****************************************************************************************************************/

void VirtualCLContext::CreateSampler(Request *req, Reply *rep) {
  INIT_VARS;
  CHECK_ID_NOT_EXISTS(SamplerIDset, CL_INVALID_SAMPLER);

  CreateSamplerMsg_t &m = req->req.m.create_sampler;

  TP_CREATE_SAMPLER(req->req.msg_id, req->req.client_did, id);
  FOR_EACH_CONTEXT_DO(
      createSampler(id, m.normalized, m.address_mode, m.filter_mode));
  TP_CREATE_SAMPLER(req->req.msg_id, req->req.client_did, id);

  FOR_EACH_CONTEXT_UNDO(freeSampler(id));

  RETURN_IF_ERR;
  SamplerIDset.insert(id);
  replyID(rep, MessageType_CreateSamplerReply, id);
}

void VirtualCLContext::FreeSampler(Request *req, Reply *rep) {
  INIT_VARS;
  CHECK_ID_EXISTS(SamplerIDset, CL_INVALID_SAMPLER);

  TP_FREE_SAMPLER(req->req.msg_id, req->req.client_did, req->req.obj_id);
  FOR_EACH_CONTEXT_DO(freeSampler(id));
  TP_FREE_SAMPLER(req->req.msg_id, req->req.client_did, req->req.obj_id);

  SamplerIDset.erase(id);
  RETURN_IF_ERR;
  replyOK(rep, MessageType_FreeSamplerReply);
}

/****************************************************************************************************************/

void VirtualCLContext::MigrateD2D(Request *req) {
  MigrateD2DMsg_t &m = req->req.m.migrate;
  RequestMsg_t &r = req->req;
  EventTiming_t evt{};
  uint32_t mem_obj_id = r.obj_id;
  uint32_t size_buffer_id = m.size_id;
  uint32_t def_queue_id = DEFAULT_QUE_ID + m.source_did;
  int err;
  char *storage = nullptr;

  if (m.source_pid == r.pid && m.source_peer_id == peer_id &&
      m.dest_peer_id == peer_id) {
    POCL_MSG_PRINT_GENERAL("migration within 1 platform\n");
    SharedContextList[r.pid]->queuedPush(req);
  } else {
    r.m.migrate.is_external = 1;
    POCL_MSG_PRINT_GENERAL(
        "migration between 2 different platforms, SIZE: %" PRIu64
        "  QID: %" PRIu32 " DID: %" PRIu32 " CONTENT SIZE BUFFER: %" PRIu32
        "\n",
        uint64_t(m.size), uint32_t(r.cq_id), uint32_t(r.did),
        uint32_t(size_buffer_id));
    assert(m.size);

    // totally made up, but we immediately delete the event from EventMap
    uint64_t fake_ev_id = r.msg_id + (1UL << 50);

    if (m.source_peer_id == peer_id) {
      POCL_MSG_PRINT_GENERAL(
          "MigrateD2D: %s READ on PID: %" PRIu32 ", DID: %" PRIu32
          ", SRC EV ID: %" PRIu64 ", DST EV ID: %" PRIu64 "\n",
          (m.is_image == 0 ? "Buffer" : "Image"), uint32_t(m.source_pid),
          uint32_t(m.source_did), fake_ev_id, uint64_t(r.msg_id));
#ifdef ENABLE_RDMA
      req->extra_data = nullptr;
#ifndef RDMA_USE_SVM
      // No SVM, we have actual shadow buffers. Write data to shadow buffer but
      // do not pass it along as extra_data, rdma thread will fetch the
      // appropriate registration data
      storage = getRdmaShadowPtr(mem_obj_id);
#endif
#else
      // No RDMA, no persistent shadow buffers
      storage = new char[m.size];
      req->extra_data = storage;
#endif

#ifndef RDMA_USE_SVM
      SharedContextBase *src = SharedContextList[m.source_pid];
      if (m.is_image == 0) {
        uint64_t content_size;
        // TODO we should probably wait for events in DST SharedCLcontext
        // before launching readBuffer in SRC SharedCLcontext
        // enqueue a read buffer in the *source* context ...
        err = src->readBuffer(fake_ev_id, def_queue_id, mem_obj_id, 0,
                              size_buffer_id, m.size, 0, storage, &content_size,
                              evt, 0, nullptr);
        m.size = content_size;
      } else {
        sizet_vec3 origin = {0, 0, 0};
        sizet_vec3 region = {m.width, m.height, m.depth};

        // TODO we should probably wait for events in DST SharedCLcontext
        // before launching readBuffer in SRC SharedCLcontext
        // enqueue a read buffer in the *source* context ...
        err = src->readImageRect(fake_ev_id, def_queue_id, mem_obj_id, origin,
                                 region, storage, m.size, evt, 0, nullptr);
      }
      // TODO error handling
      assert(err == 0);
      // .... and wait for it here.
      src->waitAndDeleteEvent(fake_ev_id);
#endif
    }

    /* Write extra_size after possible content size has been read, just before
     * pushing the request on */
    req->extra_size = m.size;

    // .... and now we can push the writeBuffer to the queue
    if (m.dest_peer_id == peer_id) {
      SharedContextList[r.pid]->queuedPush(req);
    } else {
      peers->pushRequest(req, m.dest_peer_id);
    }
  }
}

/****************************************************************************************************************/
/****************************************************************************************************************/

void VirtualCLContext::ServerInfo(Request *req, Reply *rep) {
  rep->extra_size = PlatformList.size() * sizeof(uint32_t);
  rep->extra_data.reset(new uint8_t[rep->extra_size]);
  uint32_t *Counts = (uint32_t *)rep->extra_data.get();
  for (size_t i = 0; i < PlatformList.size(); ++i) {
    Counts[i] = SharedContextList.at(i)->numDevices();
  }
  replyData(rep, MessageType_ServerInfoReply, PlatformList.size(),
            rep->extra_size);
}

void VirtualCLContext::DeviceInfo(Request *req, Reply *rep) {
  DeviceInfo_t info{};

  // The device info contains various potentially long strings such as the
  // built-in kernels and the extensions lists. Handle them with a separate
  // dynamic string section at the end of the reply.
  std::vector<std::string> strings;

  // Store an empty string at offset 0.
  strings.push_back("");
  // The first string starts at offset 1.
  rep->rep.strings_size = 1;

  SharedContextList[req->req.pid]->getDeviceInfo(req->req.did, info, strings);

  for (const std::string &str : strings)
    rep->rep.strings_size += str.size() + 1;

  rep->extra_size = sizeof(info) + rep->rep.strings_size;
  rep->extra_data.reset(new uint8_t[rep->extra_size]);
  std::memcpy(rep->extra_data.get(), &info, sizeof(info));
  char *strings_pos = (char *)rep->extra_data.get() + sizeof(info);
  for (const std::string& str : strings) {
    // Append the strings to the string part of the reply, and
    // ensure that the strings are separated with \0.
    // str.c_str() is guaranteed to have a null byte after its last position
    std::memcpy(strings_pos, str.c_str(), str.size() + 1);
    strings_pos += str.size() + 1;
  }

  replyData(rep, MessageType_DeviceInfoReply, rep->extra_size);
}

/****************************************************************************************************************/
/****************************************************************************************************************/

VirtualContextBase *createVirtualContext(PoclDaemon *d,
                                         ClientConnections_t conns,
                                         uint64_t session,
                                         CreateOrAttachSessionMsg_t &params) {
  VirtualCLContext *vctx = new VirtualCLContext();
  vctx->init(d, conns, session, params);
  return vctx;
}

void startVirtualContextMainloop(VirtualContextBase *ctx) {
  static_cast<VirtualCLContext *>(ctx)->run();
}
