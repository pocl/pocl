/* rdma.cc - wrapper classes to make RDMA more ergonomic

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

#include <iostream>
#include <netdb.h>

#include "pocl_networking.h"
#include "rdma.hh"

namespace rdmacm {
EventChannel::EventChannel() : handle(nullptr) {
  stop_polling = 0;
  handle = rdma_create_event_channel();
  if (!handle) {
    const char *hint = strerror(errno);
    assert(0 && hint);
  } else {
    worker_th = std::thread(&EventChannel::poll_loop, this);
  }
}

EventChannel::~EventChannel() {
  if (handle) {
    stop_polling = 1;
    worker_th.join();
    for (rdma_cm_event *event : shared_queue) {
      rdma_ack_cm_event(event);
    }
    for (const auto &conn : new_connection_queue) {
      for (rdma_cm_event *event : conn.second) {
        rdma_ack_cm_event(event);
      }
    }
    rdma_destroy_event_channel(handle);
  }
}

void EventChannel::shutdown() {
  stop_polling = 1;
  std::scoped_lock l(new_connection_mutex, shared_queue_mutex);
  shared_queue_cond.notify_all();
  new_connection_cond.notify_all();
}

void EventChannel::poll_loop() {
  int err = 0;
  rdma_cm_event *event;
  while (!stop_polling) {
    err = rdma_get_cm_event(handle, &event);
    if (err) {
      shutdown();
      return;
    }
    // TODO: consider recovering from errors somehow?

    if (event->event == RDMA_CM_EVENT_DEVICE_REMOVAL)
      throw std::runtime_error("RDMAcm device has been removed");

    if (event->event == RDMA_CM_EVENT_CONNECT_REQUEST) {
      std::unique_lock<std::mutex> l(new_connection_mutex);
      new_connection_queue[event->id].push_back(event);
      new_connection_cond.notify_all();
      continue;
    }

    std::unique_lock<std::mutex> l(new_connection_mutex);
    auto it = new_connection_queue.find(event->id);
    if (it != new_connection_queue.end()) {
      new_connection_queue[event->id].push_back(event);
      new_connection_cond.notify_all();
    } else {
      l.unlock();
      std::unique_lock<std::mutex> l2(shared_queue_mutex);
      shared_queue.push_back(event);
      shared_queue_cond.notify_all();
    }
  }
}

rdma_cm_event *EventChannel::getNextForNewConnection(rdma_cm_id *cm_id) {
  rdma_cm_event *event = nullptr;
  while (!event && !stop_polling) {
    std::unique_lock<std::mutex> l(new_connection_mutex);
    if (cm_id) {
      auto it = new_connection_queue.find(cm_id);
      assert(it != new_connection_queue.end());
      if (!it->second.empty()) {
        event = it->second.front();
        it->second.pop_front();
      } // else wait until a new event is received
    } else {
      for (auto &it : new_connection_queue) {
        if (it.second.front()->event == RDMA_CM_EVENT_CONNECT_REQUEST) {
          event = it.second.front();
          it.second.pop_front();
        }
      }
    }

    if (!event && !stop_polling)
      new_connection_cond.wait(l);
  }

  return event;
}

void EventChannel::finalizeNewConnection(rdma_cm_id *cm_id) {
  std::scoped_lock l(new_connection_mutex, shared_queue_mutex);
  auto &conn = new_connection_queue.at(cm_id);
  shared_queue.insert(shared_queue.cend(), conn.begin(), conn.end());
  new_connection_queue.erase(cm_id);
  shared_queue_cond.notify_all();
}

rdma_cm_event *EventChannel::getNext() {
  rdma_cm_event *event = nullptr;
  while (!event && !stop_polling) {
    std::unique_lock<std::mutex> l(shared_queue_mutex);
    if (!stop_polling) {
      if (shared_queue.empty())
        shared_queue_cond.wait(l);
      else {
        event = shared_queue.front();
        shared_queue.pop_front();
      }
    }
  }

  return event;
}

Id::Id(EventChannelPtr ch, rdma_cm_id *id) : cm_channel(ch), handle(id) {}

Id::Id(EventChannelPtr ch) : cm_channel(ch), handle(nullptr) {
  int error = rdma_create_id(*cm_channel, &handle, NULL, RDMA_PS_TCP);
  if (error) {
    const char *hint = strerror(errno);
    assert(0 && hint);
  }
}

Id::~Id() {
  int error = rdma_destroy_id(handle);
  if (error) {
    const char *hint = strerror(errno);
    assert(0 && hint);
  }
}

} // namespace rdmacm

namespace ibverbs {
ProtectionDomain::ProtectionDomain(rdmacm::IdPtr id) : cm_id(id) {
  handle = ibv_alloc_pd(cm_id->ctx());
  assert(handle);
}

ProtectionDomain::~ProtectionDomain() {
  if (handle) {
    int res = ibv_dealloc_pd(handle);
    assert(res == 0);
  }
}

const MemoryRegion::Access MemoryRegion::Access::LocalRead = {0};
const MemoryRegion::Access MemoryRegion::Access::LocalWrite = {
    IBV_ACCESS_LOCAL_WRITE};
const MemoryRegion::Access MemoryRegion::Access::RemoteWrite = {
    IBV_ACCESS_REMOTE_WRITE};
const MemoryRegion::Access MemoryRegion::Access::RemoteRead = {
    IBV_ACCESS_REMOTE_READ};
const MemoryRegion::Access MemoryRegion::Access::RemoteAtomic = {
    IBV_ACCESS_REMOTE_ATOMIC};
const MemoryRegion::Access MemoryRegion::Access::WindowBinding = {
    IBV_ACCESS_MW_BIND};
const MemoryRegion::Access MemoryRegion::Access::ZeroBased = {
    IBV_ACCESS_ZERO_BASED};
const MemoryRegion::Access MemoryRegion::Access::OnDemand = {
    IBV_ACCESS_ON_DEMAND};
const MemoryRegion::Access MemoryRegion::Access::HugePages = {
    IBV_ACCESS_HUGETLB};
const MemoryRegion::Access MemoryRegion::Access::RelaxedOrdering = {
    IBV_ACCESS_RELAXED_ORDERING};

MemoryRegion::MemoryRegion(ProtectionDomainPtr pd, void *addr, size_t length,
                           MemoryRegion::Access access)
    : pd(pd), handle(nullptr), flags(access) {
  handle = ibv_reg_mr(*pd, addr, length, access.val);
  assert(handle && "Is your pinned memory limit too low?");
}

MemoryRegion::MemoryRegion(ProtectionDomainPtr pd, uint64_t offset,
                           size_t length, int fd, MemoryRegion::Access access)
    : pd(pd), handle(nullptr), flags(access) {
  handle = ibv_reg_dmabuf_mr(*pd, offset, length, /*iova*/ 0, fd, access.val);
  assert(handle);
}

CompletionChannel::CompletionChannel(rdmacm::IdPtr id) : cm_id(id) {
  handle = ibv_create_comp_channel(cm_id->ctx());
  assert(handle);
}

CompletionChannel::~CompletionChannel() {
  if (handle) {
    int res = ibv_destroy_comp_channel(handle);
    assert(res == 0);
  }
}

CompletionQueue::CompletionQueue(rdmacm::IdPtr cm_id, int capacity)
    : channel(cm_id) {
  handle = ibv_create_cq(cm_id->ctx(), capacity, nullptr, channel, 0);
  assert(handle);
  int err = ibv_req_notify_cq(handle, 0);
  const char *hint = strerror(err);
  assert(!err && hint);
}

CompletionQueue::~CompletionQueue() {
  if (handle) {
    int res = ibv_destroy_cq(handle);
    assert(res == 0);
  }
}

WorkCompletion CompletionQueue::awaitCompletion() {
  ibverbs::WorkCompletion wc;

  // ibv_req_notify_cq causes ONE (1) completion notification to be generated
  // for any work completion that arrives AFTER the call to ibv_req_notify_cq.
  // Existing work completions do not generate notifications after the fact.
  //
  // ibv_poll_cq gets a work completion from the queue if it contains any
  //
  // ibv_req_notify_cq and ibv_get_cq_event can be used to put the thread to
  // sleep until a new work completion and notification are generated
  //
  // Request a notification before polling to avoid a race between ibv_poll_cq
  // and ibv_req_notify_cq
  int error = ibv_req_notify_cq(handle, 0);
  if (error)
    throw std::runtime_error(strerror(error));

  while (ibv_poll_cq(handle, 1, &wc) != 1) {
    ibv_cq *comp_queue;
    void *context;
    if (ibv_get_cq_event(channel, &comp_queue, &context))
      throw std::runtime_error("Get RDMA channel event failed");
    assert(comp_queue == handle);
  }

  // Pop the work completion event from the queue
  ibv_ack_cq_events(handle, 1);

  return wc;
}

QueuePair::QueuePair(rdmacm::IdPtr cm_id, ProtectionDomainPtr pd)
    : cm_id(cm_id), send_cq(cm_id), recv_cq(cm_id) {
  ibv_qp_init_attr attributes;
  attributes.qp_context = nullptr;
  attributes.sq_sig_all = 0;
  attributes.srq = nullptr;
  attributes.cap.max_send_wr = WR_QUEUE_SIZE;
  attributes.cap.max_send_sge = MAX_SGE_COUNT;
  attributes.cap.max_recv_wr = WR_QUEUE_SIZE;
  attributes.cap.max_recv_sge = MAX_SGE_COUNT;
  attributes.cap.max_inline_data = 0;
  attributes.send_cq = send_cq;
  attributes.recv_cq = recv_cq;
  attributes.qp_type = IBV_QPT_RC;

  int err = rdma_create_qp(*cm_id, *pd, &attributes);
  const char *hint = strerror(err);
  assert(!err && hint);
}

QueuePair::~QueuePair() { rdma_destroy_qp(*cm_id); }
} // namespace ibverbs

ScatterGatherEntry::ScatterGatherEntry(ibverbs::MemoryRegionPtr mr)
    : sge{((mr->accessFlags() & ibverbs::MemoryRegion::Access::ZeroBased).val
               ? 0
               : (uint64_t)((ibv_mr *)**mr)->addr),
          (uint32_t)mr->handle->length, mr->handle->lkey},
      mr(mr) {}

ScatterGatherEntry::ScatterGatherEntry(ibverbs::MemoryRegionPtr mr,
                                       ptrdiff_t offset, uint32_t length)
    : sge{((mr->accessFlags() & ibverbs::MemoryRegion::Access::ZeroBased).val
               ? 0
               : (uint64_t)((ibv_mr *)**mr)->addr) +
              (int64_t)offset,
          length, ((ibv_mr *)**mr)->lkey},
      mr(mr) {}

const WorkRequest::Flags WorkRequest::Flags::None = {0};
const WorkRequest::Flags WorkRequest::Flags::Fence = {IBV_SEND_FENCE};
const WorkRequest::Flags WorkRequest::Flags::Signaled = {IBV_SEND_SIGNALED};
const WorkRequest::Flags WorkRequest::Flags::Solicited = {IBV_SEND_SOLICITED};
const WorkRequest::Flags WorkRequest::Flags::Inline = {IBV_SEND_INLINE};
const WorkRequest::Flags WorkRequest::Flags::OffloadIpChecksum = {
    IBV_SEND_IP_CSUM};

WorkRequest::WorkRequest() : wr{} {}

WorkRequest::WorkRequest(const WorkRequest &other)
    : wr{other.wr}, mr_list{other.mr_list},
      ibv_sg_list{other.ibv_sg_list}, next{} {
  wr.sg_list = ibv_sg_list.data();
  if (other.next) {
    next.reset(new WorkRequest(*other.next));
    wr.next = &next->wr;
  }
}

void WorkRequest::initWithSgList(uint64_t id, ibv_wr_opcode op,
                                 const std::vector<ScatterGatherEntry> &sg_list,
                                 Flags flags) {
  wr.wr_id = id;
  wr.opcode = (enum ibv_wr_opcode)op;
  wr.send_flags = flags;
  mr_list.reserve(sg_list.size());
  ibv_sg_list.reserve(sg_list.size());
  for (const ScatterGatherEntry &sge : sg_list) {
    mr_list.push_back(sge.mr);
    ibv_sg_list.push_back(sge.sge);
  }
  wr.num_sge = sg_list.size();
  wr.sg_list = ibv_sg_list.data();
}

void WorkRequest::chain(const WorkRequest &n) {
  next.reset(new WorkRequest(n));
  wr.next = &next->wr;
}

#define COMPUTE_TOTAL_SG_LENGTH                                                \
  size_t total_sg_length = 0;                                                  \
  do {                                                                         \
    for (const ScatterGatherEntry &sge : sg_list) {                            \
      total_sg_length += sge.sge.length ? sge.sge.length : INT32_MAX;          \
    }                                                                          \
  } while (0)

WorkRequest WorkRequest::Send(uint64_t wr_id,
                              const std::vector<ScatterGatherEntry> &sg_list,
                              WorkRequest::Flags flags) {
  /*
   * IBV_WR_SEND
   *
   * Write contents of the buffer regions specified in sg_list to the
   * sg_list specified in a ReceiveRequest on the remote end. The
   * ReceiveRequest will be consumed from the remote's queue. The sender
   * has no way of knowing where data was written.
   *
   * Maximum size is 2^31 bytes for ReliableConnecion and Unconnected type
   * queues and equal to the lowest MTU along the network path is for
   * UreliableDatagram type queues.
   */

  COMPUTE_TOTAL_SG_LENGTH;
  // only ReliableConnection is implemented for now
  assert(total_sg_length < INT32_MAX);

  WorkRequest w;
  w.initWithSgList(wr_id, IBV_WR_SEND, sg_list, flags);
  return w;
}

WorkRequest WorkRequest::Send(uint64_t wr_id,
                              const std::vector<ScatterGatherEntry> &sg_list,
                              uint32_t immediate, WorkRequest::Flags flags) {
  /*
   * IBV_WR_SEND_WITH_IMM
   *
   * Same as IBV_WR_SEND, but a 32 bit immediate value will be sent along with
   * the request. This value will be available in the WorkCompletion
   * associated with the ReceiveRequest that was consumed on the remote.
   */

  WorkRequest w = WorkRequest::Send(wr_id, sg_list, flags);
  w.wr.opcode = IBV_WR_SEND_WITH_IMM;
  w.wr.imm_data = immediate;
  return w;
}

WorkRequest WorkRequest::RdmaWrite(
    uint64_t wr_id, const std::vector<ScatterGatherEntry> &sg_list,
    uint64_t remote_addr, uint32_t rkey, WorkRequest::Flags flags) {
  /*
   * IBV_WR_RDMA_WRITE
   *
   * Write buffer contents from sg_list to the given virtual address /
   * rkey as one continuous block. No ReceiveRequest will be consumed on
   * the remote.
   *
   * Maximum size is 2^31 bytes.
   */

  COMPUTE_TOTAL_SG_LENGTH;
  assert(total_sg_length < INT32_MAX);

  WorkRequest w;
  w.initWithSgList(wr_id, IBV_WR_RDMA_WRITE, sg_list, flags);
  w.wr.wr.rdma.remote_addr = remote_addr;
  w.wr.wr.rdma.rkey = rkey;
  return w;
}

WorkRequest
WorkRequest::RdmaWrite(uint64_t wr_id,
                       const std::vector<ScatterGatherEntry> &sg_list,
                       uint64_t remote_addr, uint32_t rkey, uint32_t immediate,
                       WorkRequest::Flags flags) {
  /*
   * IBV_WR_RDMA_WRITE_WITH_IMM
   *
   * Same as IBV_WR_RDMA_WRITE, but a 32 bit immediate value will be sent
   * along with the request and a ReceiveRequest WILL be consumed on the
   * remote. The WorkCompletion associated with the consumed ReceiveRequest
   * will contain the given immediate value.
   */

  WorkRequest w =
      WorkRequest::RdmaWrite(wr_id, sg_list, remote_addr, rkey, flags);
  w.wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  w.wr.imm_data = immediate;
  return w;
}

WorkRequest
WorkRequest::RdmaRead(uint64_t wr_id,
                      const std::vector<ScatterGatherEntry> &sg_list,
                      uint64_t remote_addr, uint32_t rkey, Flags flags) {
  /*
   * IBV_WR_RDMA_READ
   *
   * Read a continuous block from the specified remote virtual address /
   * rkey and scatter the contents sequentially into the local buffers as
   * specified in sg_list. No ReceiveRequest will be consumed on the
   * remote. Maximum size is 2^31 bytes.
   */

  COMPUTE_TOTAL_SG_LENGTH;
  assert(total_sg_length < INT32_MAX);

  WorkRequest w;
  w.initWithSgList(wr_id, IBV_WR_RDMA_READ, sg_list, flags);
  w.wr.wr.rdma.remote_addr = remote_addr;
  w.wr.wr.rdma.rkey = rkey;
  return w;
}

WorkRequest WorkRequest::AtomicCompareAndSwap(
    uint64_t wr_id, const std::vector<ScatterGatherEntry> &sg_list,
    uint64_t remote_addr, uint32_t rkey, uint64_t compare, uint64_t swap,
    Flags flags) {
  /*
   * IBV_WR_ATOMIC_CMP_AND_SWP
   *
   * Read a 64 bit value from the remote virtual address / rkey, compare
   * it to the value from wr.atomic.compare_add and if they are equal,
   * write the value from wr.atomic.swap to that address atomically. The
   * original value will be written to the local buffers as specified in
   * sg_list. No ReceiverRequest will be consumed on the remote.
   */

  COMPUTE_TOTAL_SG_LENGTH;
  assert(total_sg_length >= sizeof(uint64_t));

  WorkRequest w;
  w.initWithSgList(wr_id, IBV_WR_ATOMIC_CMP_AND_SWP, sg_list, flags);
  w.wr.wr.atomic.remote_addr = remote_addr;
  w.wr.wr.atomic.rkey = rkey;
  w.wr.wr.atomic.compare_add = compare;
  w.wr.wr.atomic.swap = swap;
  return w;
}

WorkRequest WorkRequest::AtomicFetchAndAdd(
    uint64_t wr_id, const std::vector<ScatterGatherEntry> &sg_list,
    uint64_t remote_addr, uint32_t rkey, uint64_t add, Flags flags) {
  /*
   * IBV_WR_ATOMIC_FETCH_AND_ADD
   *
   * Read a 64 bit value from the remote virtual address / rkey, add the
   * value from wr.atomic.compare_add to it and write the result back to
   * the same address atomically. The original value will be written to
   * the local buffers as specified in sg_list. No ReceiverRequest will be
   * consumed on the remote.
   */

  COMPUTE_TOTAL_SG_LENGTH;
  assert(total_sg_length >= sizeof(uint64_t));

  WorkRequest w;
  w.initWithSgList(wr_id, IBV_WR_ATOMIC_FETCH_AND_ADD, sg_list, flags);
  w.wr.wr.atomic.remote_addr = remote_addr;
  w.wr.wr.atomic.rkey = rkey;
  w.wr.wr.atomic.compare_add = add;
  return w;
}

/* UNIMPLEMENTED VERBS: */
/*
 * IBV_WR_LOCAL_INV
 *
 * Invalidate the type 2 memory window associated with rkey
 */
/*
 * IBV_WR_BIND_MW
 *
 * Bind a type 2 memory window and set its rkey and properties as
 * specified by the rkey and bind_info parameters
 */
/*
 * IBV_WR_SEND_WITH_INV
 *
 * Same as Send, but additionally invalidates the remote type 2 memory
 * window specified by the rkey in invalidate_rkey.
 */
/*
 * IBV_WR_TSO
 *
 * Same as IBV_WR_SEND, but producing multiple messages using TCP Segmentation
 * Offload. The sg_list points to a TCP Stream buffer which will be
 * split into chunks of Maximum Segment Size (MTU). HDR will include a
 * TCP header for each segment. Only usable with UnreliableDatagram and
 * RawPacket queues.
 */
/*
 * IBV_WR_DRIVER1
 *
 * Do not use.
 */
/*
 * IBV_WR_ATOMIC_WRITE
 *
 * Atomically write 64 bits from given sg_list to the remote virtual
 * address / rkey. Must be supported by the HCA and enabled at QueuePair
 * creation time. New in IB SPEC 1.5, for more details see
 * https://lore.kernel.org/linux-rdma/20220418061244.89025-1-yangx.jy@fujitsu.com/
 */
#undef COMPUTE_TOTAL_SG_LENGTH

ReceiveRequest::ReceiveRequest(uint64_t wr_id,
                               const std::vector<ScatterGatherEntry> &sg_list)
    : wr{} {
  mr_list.reserve(sg_list.size());
  ibv_sg_list.reserve(sg_list.size());
  for (const ScatterGatherEntry &sge : sg_list) {
    mr_list.push_back(sge.mr);
    ibv_sg_list.push_back(sge.sge);
  }
  wr.num_sge = sg_list.size();
  wr.sg_list = ibv_sg_list.data();
  wr.wr_id = wr_id;
}

ReceiveRequest::ReceiveRequest(const ReceiveRequest &other)
    : wr{other.wr}, mr_list{other.mr_list}, ibv_sg_list{other.ibv_sg_list} {
  wr.sg_list = ibv_sg_list.data();
  if (other.next) {
    next.reset(new ReceiveRequest(*other.next));
    wr.next = &next->wr;
  }
}

void ReceiveRequest::chain(const ReceiveRequest &n) {
  next.reset(new ReceiveRequest(n));
  wr.next = &next->wr;
}

RdmaListener::RdmaListener()
    : listening_id(new rdmacm::Id(rdmacm::EventChannel::create())),
      listening(false) {}

void RdmaListener::listen(uint16_t port) {
  assert(!listening);
  int err;

  addrinfo *ai = pocl_resolve_address(nullptr, port, &err);
  if (err)
    throw std::runtime_error(gai_strerror(err));

  /* Try binding returned addresses until one works or we run out */
  for (addrinfo *a = ai; a != nullptr; a = a->ai_next) {
    err = rdma_bind_addr(*listening_id, a->ai_addr);
    if (!err)
      break;
  }

  freeaddrinfo(ai);
  if (err)
    throw std::runtime_error(strerror(errno));

  err = rdma_listen(*listening_id, 10);
  if (err)
    throw std::runtime_error(strerror(errno));

  listening = true;
}

RdmaConnection RdmaListener::accept() {
  assert(listening);

  int err;
  rdma_cm_event *event;
  rdmacm::IdPtr incoming_id;

  event = listening_id->cm_channel->getNextForNewConnection(nullptr);
  assert(event->event == RDMA_CM_EVENT_CONNECT_REQUEST);
  incoming_id.reset(new rdmacm::Id(listening_id->cm_channel, event->id));
  RdmaConnection connection(incoming_id);
  rdma_ack_cm_event(event);

  rdma_conn_param parameters = {};
  parameters.responder_resources = 1;
  err = rdma_accept(*incoming_id, &parameters);
  if (err)
    throw std::runtime_error(strerror(errno));

  event = listening_id->cm_channel->getNextForNewConnection(*incoming_id);
  if (event->event != RDMA_CM_EVENT_ESTABLISHED)
    throw std::runtime_error(
        std::string("Unexpected RDMAcm event while waiting for "
                    "RDMA_CM_EVENT_ESTABLISHED") +
        rdma_event_str(event->event));
  rdma_ack_cm_event(event);
  listening_id->cm_channel->finalizeNewConnection(*incoming_id);

  return connection;
}

RdmaConnection::RdmaConnection(rdmacm::IdPtr cm_id)
    : cm_id(cm_id), pd(ibverbs::ProtectionDomain::create(cm_id)),
      qp(ibverbs::QueuePair::create(cm_id, pd)) {}

RdmaConnection RdmaConnection::connect(const char *address, uint16_t port) {
  int err = 0;
  int timeout_ms = 5000;
  addrinfo *ai = pocl_resolve_address(address, port, &err);
  if (err)
    throw std::runtime_error(gai_strerror(err));

  rdmacm::IdPtr cm_id(new rdmacm::Id(rdmacm::EventChannel::create()));

  /* Try resolving returned addresses until one works or we run out. */
  for (addrinfo *a = ai; a != nullptr && !err; a = a->ai_next) {
    err = rdma_resolve_addr(*cm_id, NULL, ai->ai_addr, timeout_ms);
  }
  freeaddrinfo(ai);
  if (err)
    throw std::runtime_error(strerror(errno));

  // This is the active side of the connection and there won't be any other
  // IDs on this event channel so we'll just use the shared queue
  rdma_cm_event *event = cm_id->cm_channel->getNext();
  if (event->event != RDMA_CM_EVENT_ADDR_RESOLVED)
    throw std::runtime_error(
        std::string("Unexpected RDMAcm event while waiting for "
                    "RDMA_CM_EVENT_ADDR_RESOLVED: ") +
        rdma_event_str(event->event));

  rdma_ack_cm_event(event);

  // This fills out our ibv_context i.e. cm_id->verbs
  err = rdma_resolve_route(*cm_id, timeout_ms);
  if (err)
    throw std::runtime_error(strerror(errno));

  event = cm_id->cm_channel->getNext();
  if (event->event != RDMA_CM_EVENT_ROUTE_RESOLVED)
    throw std::runtime_error(
        std::string("Unexpected RDMAcm event while waiting for "
                    "RDMA_CM_EVENT_ROUTE_RESOLVED: ") +
        rdma_event_str(event->event));

  rdma_ack_cm_event(event);

  RdmaConnection conn(cm_id);

  rdma_conn_param conn_param = {};
  conn_param.initiator_depth = 1;
  conn_param.retry_count = 5;

  err = rdma_connect(*cm_id, &conn_param);
  if (err)
    throw std::runtime_error(strerror(errno));

  event = cm_id->cm_channel->getNext();

  // RDMA_CM_EVENT_CONNECT_RESPONSE won't be generated since RdmaConnection
  // sets up a QueuePair on the Id
  if (event->event != RDMA_CM_EVENT_ESTABLISHED)
    throw std::runtime_error(
        std::string("Unexpected RDMAcm event while waiting for "
                    "RDMA_CM_EVENT_ESTABLISHED") +
        rdma_event_str(event->event));

  rdma_ack_cm_event(event);

  return conn;
}

void RdmaConnection::post(const WorkRequest &req) {
  // why is the argument to ibv_post_send not const???
  ibv_send_wr wr = req.wr;
  ibv_send_wr *bad_wr;
  int error = ibv_post_send(*qp, &wr, &bad_wr);
  if (error)
    throw std::runtime_error(strerror(error));
}

void RdmaConnection::post(const ReceiveRequest &req) {
  // why is the argument to ibv_post_recv not const???
  ibv_recv_wr wr = req.wr;
  ibv_recv_wr *bad_wr;
  int error = ibv_post_recv(*qp, &wr, &bad_wr);
  if (error)
    throw std::runtime_error(strerror(error));
}

ibverbs::WorkCompletion RdmaConnection::awaitSendCompletion() {
  return qp->send_cq.awaitCompletion();
}

ibverbs::WorkCompletion RdmaConnection::awaitRecvCompletion() {
  return qp->recv_cq.awaitCompletion();
}
