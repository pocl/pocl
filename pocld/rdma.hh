/* rdma.hh - wrapper classes to make RDMA more ergonomic

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

#ifndef POCLD_RDMA_HH
#define POCLD_RDMA_HH

#include <array>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#define MAX_SGE_COUNT 10
#define WR_QUEUE_SIZE 10
#define DEFAULT_PORT 59274

// Forward declarations
namespace rdmacm {
class EventChannel;
typedef std::shared_ptr<EventChannel> EventChannelPtr;
class Id;
typedef std::shared_ptr<Id> IdPtr;
} // namespace rdmacm

namespace ibverbs {
class CompletionChannel;
class CompletionQueue;
class QueuePair;
typedef std::shared_ptr<QueuePair> QueuePairPtr;
class MemoryRegion;
typedef std::shared_ptr<MemoryRegion> MemoryRegionPtr;
class ProtectionDomain;
typedef std::shared_ptr<ProtectionDomain> ProtectionDomainPtr;
class WorkCompletion;
} // namespace ibverbs

class ScatterGatherEntry;
class RdmaListener;
class RdmaConnection;

// Actual interfaces
namespace rdmacm {

/** RAII wrapper for the RDMA Connection Manager Event Channel */
class EventChannel {
private:
  rdma_event_channel *handle;
  std::thread worker_th;

  std::atomic_int stop_polling;

  std::mutex new_connection_mutex;
  std::condition_variable new_connection_cond;
  std::unordered_map<rdma_cm_id *, std::deque<rdma_cm_event *>>
      new_connection_queue;

  std::mutex shared_queue_mutex;
  std::condition_variable shared_queue_cond;
  std::deque<rdma_cm_event *> shared_queue;

  friend class Id;
  friend class ::RdmaListener;
  friend class ::RdmaConnection;

  EventChannel();

  void poll_loop();

  /**
   * Get next event associated with given cm_id. All events MUST be
   * acknowledged with rdma_ack_cm_event once handled. Must not be called again
   * with the same cm_id after finalizeNewConnection.
   *
   * @param cm_id id that events should correspond to. If nullptr, returns any
   * RDMA_CM_EVENT_CONNECT_REQUEST in the new connection buffer or blocks
   * until one is received.
   */
  rdma_cm_event *getNextForNewConnection(rdma_cm_id *cm_id);

  void finalizeNewConnection(rdma_cm_id *cm_id);

  operator rdma_event_channel *() { return handle; }

public:
  static EventChannelPtr create() {
    return std::shared_ptr<EventChannel>(new EventChannel());
  }

  ~EventChannel();

  /**
   * All events MUST be acknowledged with rdma_ack_cm_event once handled.
   */
  rdma_cm_event *getNext();

  void shutdown();
};

/**
 * RAII wrapper for the RDMA Connection Manager ID. The ID is conceptually
 * equivalent to a socket.
 */
class Id {
private:
  EventChannelPtr cm_channel;
  rdma_cm_id *handle;

  Id() = delete;
  Id(EventChannelPtr);
  Id(EventChannelPtr, rdma_cm_id *);

  friend class ibverbs::ProtectionDomain;
  friend class ibverbs::CompletionChannel;
  friend class ibverbs::CompletionQueue;
  friend class ibverbs::QueuePair;
  friend class ::RdmaListener;
  friend class ::RdmaConnection;

  ibv_context *ctx() { return handle->verbs; }

public:
  ~Id();

  operator rdma_cm_id *() { return handle; }

  bool operator==(const Id &other) { return other.handle == handle; }

  bool operator==(rdma_cm_id *other) { return other == handle; }

  bool operator!=(const Id &other) { return other.handle != handle; }

  bool operator!=(rdma_cm_id *other) { return other != handle; }
};
} // namespace rdmacm

namespace ibverbs {
class CompletionChannel {
private:
  rdmacm::IdPtr cm_id;
  ibv_comp_channel *handle;

  friend class CompletionQueue;

  CompletionChannel(rdmacm::IdPtr);

  operator ibv_comp_channel *() { return handle; }

public:
  CompletionChannel() = delete;
  CompletionChannel(const CompletionChannel &) = delete;
  CompletionChannel(CompletionChannel &&) = delete;
  ~CompletionChannel();
  CompletionChannel &operator=(const CompletionChannel &) = delete;
  CompletionChannel &operator=(CompletionChannel &&) = delete;
};

class CompletionQueue {
private:
  CompletionChannel channel;
  ibv_cq *handle;

  friend class QueuePair;
  friend class ::RdmaConnection;

  CompletionQueue(rdmacm::IdPtr, int capacity = 10);

  operator ibv_cq *() { return handle; }

  WorkCompletion awaitCompletion();

public:
  CompletionQueue() = delete;
  CompletionQueue(const CompletionQueue &) = delete;
  CompletionQueue(CompletionQueue &&) = delete;
  ~CompletionQueue();
  CompletionQueue &operator=(const CompletionQueue &) = delete;
  CompletionQueue &operator=(CompletionQueue &&) = delete;
};

class QueuePair {
private:
  rdmacm::IdPtr cm_id;
  CompletionQueue send_cq;
  CompletionQueue recv_cq;
  QueuePair(rdmacm::IdPtr, ProtectionDomainPtr);

  friend class ::RdmaConnection;

  static QueuePairPtr create(rdmacm::IdPtr cm_id, ProtectionDomainPtr pd) {
    return std::shared_ptr<QueuePair>(new QueuePair(cm_id, pd));
  }

  operator ibv_qp *() { return cm_id->handle->qp; }

public:
  QueuePair() = delete;
  QueuePair(const QueuePair &) = delete;
  QueuePair(QueuePair &&) = delete;
  ~QueuePair();
  QueuePair &operator=(const QueuePair &) = delete;
  QueuePair &operator=(QueuePair &&) = delete;
};

/** RAII wrapper class for IBV protection domains */
class ProtectionDomain {
private:
  rdmacm::IdPtr cm_id;
  ibv_pd *handle;

  friend class MemoryRegion;
  friend class QueuePair;

  ProtectionDomain(rdmacm::IdPtr);

  operator ibv_pd *() { return handle; }

public:
  ProtectionDomain() = delete;
  ProtectionDomain(const ProtectionDomain &) = delete;
  ProtectionDomain(ProtectionDomain &&other) = delete;
  ~ProtectionDomain();
  ProtectionDomain &operator=(const ProtectionDomain &) = delete;
  ProtectionDomain &operator=(ProtectionDomain &&) = delete;

  static ProtectionDomainPtr create(rdmacm::IdPtr cm_id) {
    return std::shared_ptr<ProtectionDomain>(new ProtectionDomain(cm_id));
  };
};

/**
 * RAII wrapper class for memory region registrations. MR registration is slow
 * (has to call into kernel and pin the memory pages) so don't do it in a tight
 * loop.
 */
class MemoryRegion {
public:
  /** Access flags for MemoryRegion registrations */
  class Access {
  private:
    unsigned int val;

    Access(ibv_access_flags v) : val((unsigned int)v) {}

    Access(unsigned int v) : val(v) {}

    operator unsigned int() const { return val; }

    friend class MemoryRegion;
    friend class ::ScatterGatherEntry;

  public:
    Access() = delete;

    Access operator|(const Access &other) const { return val | other.val; }
    Access operator&(const Access &other) const { return val & other.val; }

    /** Default - local read access is always enabled */
    static const Access LocalRead;
    /**
     * Enable local write access. Required for use in Receive Requests and
     * for AtomicCompareAndSwap or AtomicFetchAndAdd to write to remote
     * content "locally".
     */
    static const Access LocalWrite;
    /**
     * Enable remote write access. Required for remote access with RdmaWrite
     * or RdmaWriteWithImmediate. LocalWrite MUST also be set.
     */
    static const Access RemoteWrite;
    /**
     * Enable remote read access. Required for remote access with RdmaRead.
     */
    static const Access RemoteRead;
    /**
     * Enable atomic operation access (if supported). Required for
     * AtomicCompareAndSwap  and AtomicFetchAndAdd.
     */
    static const Access RemoteAtomic;
    /** Enable memory window binding. */
    static const Access WindowBinding;
    /**
     * Use byte offset from beginning of MR to access this MR, instead of
     * the pointer address.
     */
    static const Access ZeroBased;
    /** Create and on-demand paging MR */
    static const Access OnDemand;
    /**
     * Huge pages are guaranteed to be used for this MR, applicable with
     * OnDemand in explicit mode only.
     */
    static const Access HugePages;
    /** Allow system to reorder accesses to the MR to improve performance. */
    static const Access RelaxedOrdering;
  };

private:
  ProtectionDomainPtr pd;
  ibv_mr *handle;
  Access flags;

  friend class ::ScatterGatherEntry;

  MemoryRegion(ProtectionDomainPtr pd, void *addr, size_t length,
               Access access = Access::LocalRead | Access::ZeroBased);

  MemoryRegion(ProtectionDomainPtr pd, uint64_t offset, size_t length, int fd,
               Access access = Access::LocalRead | Access::ZeroBased);

  ibv_mr *operator*() { return handle; }

public:
  MemoryRegion() = delete;
  MemoryRegion(const MemoryRegion &) = delete;
  MemoryRegion(MemoryRegion &&other) = delete;
  MemoryRegion &operator=(const MemoryRegion &) = delete;
  MemoryRegion &operator=(MemoryRegion &&) = delete;

  /**
   * Register a region of host memory for RDMA use
   * @param pd The protection domain that governs this memory region
   * @param addr Pointer to the start of the memory region to be registered
   * @param length Length of the memory region to be registered
   * @param access Bitfield of access qualifiers
   */
  static MemoryRegionPtr
  register_ptr(ProtectionDomainPtr pd, void *addr, size_t length,
               Access access = Access::LocalRead | Access::ZeroBased) {
    return std::shared_ptr<MemoryRegion>(
        new MemoryRegion(pd, addr, length, access));
  }

  /**
   * Register a portion of a dma-buf for RDMA use
   * @param pd The protection domain that governs this memory region
   * @param offset Offset from the start of the dma-buf
   * @param length Length of the memory region to be registered
   * @param fd The fd handle of the dma-buf
   * @param access Bitfield of access qualifiers
   */
  static MemoryRegionPtr register_dmabuf(ProtectionDomainPtr pd,
                                         uint64_t offset, size_t length, int fd,
                                         Access access = Access::LocalRead |
                                                         Access::ZeroBased) {
    return std::shared_ptr<MemoryRegion>(
        new MemoryRegion(pd, offset, length, fd, access));
  }

  uint32_t rkey() { return handle->rkey; }

  Access accessFlags() { return flags; }
};

class WorkCompletion : public ibv_wc {};
} // namespace ibverbs

/** Owned container with automatic RDMA memory region registration */
template <typename T> class RdmaBuffer {
private:
  static_assert(std::is_trivial_v<T>);
  size_t size;
  std::unique_ptr<T[]> ptr;
  ibverbs::MemoryRegionPtr mr;

public:
  RdmaBuffer(ibverbs::ProtectionDomainPtr pd, size_t elements,
             ibverbs::MemoryRegion::Access access =
                 ibverbs::MemoryRegion::Access::LocalRead |
                 ibverbs::MemoryRegion::Access::ZeroBased) {
    ptr.reset(new T[elements]);
    size = elements;
    mr = ibverbs::MemoryRegion::register_ptr(pd, (void *)ptr.get(),
                                             sizeof(T) * elements, access);
  }

  ibverbs::MemoryRegionPtr operator*() { return mr; }

  T &at(size_t i) {
    if (i > size)
      throw std::runtime_error("index out of bounds");
    return ptr[i];
  }
};

/** Defines (a subregion of) a memory region to be used for a work request */
class ScatterGatherEntry {
private:
  ibv_sge sge;
  ibverbs::MemoryRegionPtr mr;

  friend class WorkRequest;
  friend class ReceiveRequest;

public:
  ScatterGatherEntry() = delete;
  /**
   * Constructor for a ScatterGatherEntry that spans the entire given
   * MemoryRegion.
   */
  ScatterGatherEntry(ibverbs::MemoryRegionPtr mr);
  /**
   * Constructor for a ScatterGatherEntry that spans a subregion of the given
   * MemoryRegion
   */
  ScatterGatherEntry(ibverbs::MemoryRegionPtr mr, ptrdiff_t offset,
                     uint32_t length);
};

/**
 * Outgoing Work Request. These can be chained with the inherited 'next' field.
 */
class WorkRequest {
public:
  /** Bitmask representing WorkRequest properties. */
  class Flags {
  private:
    unsigned int val;

    operator unsigned int() const { return val; }
    friend class WorkRequest;

    Flags(ibv_send_flags v) : val((unsigned int)v) {}

    Flags(unsigned int v) : val(v) {}

  public:
    Flags() = delete;

    Flags operator|(const Flags &other) const { return val | other.val; }

    static const Flags None;

    /**
     * Wait for all RDMA Read and Atomic WRs submitted before this
     * WorkRequest to complete before starting to process it.
     */
    static const Flags Fence;

    /**
     * Generate a local WorkCompletion for this WR. Has no effect if the
     * QueuePair was created with sq_sig_all=1 in its attributes.
     */
    static const Flags Signaled;

    /**
     * Generate a 'solicited' event. If the remote is waiting for
     * 'solicited' events, it will be woken up. Only relevant for Send and
     * RdmaWrite type requests.
     */
    static const Flags Solicited;

    /**
     * Memory buffers will be inlined in the WorkRequest, meaning they will
     * be circulated through the CPU and lkey will be ignored. Data buffers
     * can be reused immediately after the send function returns. Without
     * this flag data buffers are in use until the WorkRequest is completed
     * and a WorkCompletion has been generated.
     *
     * Only valid for Send and RdmaWrite type
     * requests.
     *
     * Note that this is a nonstandard implementation extension. If
     * you aren't sure whether you need this, you probably don't.
     */
    static const Flags Inline;

    /**
     * Offload IPv4 and TCP/UDP checksum calculations. Only valid when
     * indicated as supported by the device capability flags in the device
     * attributes of the QueuePair.
     */
    static const Flags OffloadIpChecksum;
  };

private:
  ibv_send_wr wr;
  // Hold references to memory regions to ensure they won't be deleted
  // prematurely
  std::vector<ibverbs::MemoryRegionPtr> mr_list;
  // The actual scatter/gather entries referencing the memory regions in
  // mr_list
  std::vector<ibv_sge> ibv_sg_list;
  std::unique_ptr<WorkRequest> next;

  friend class RdmaConnection;

  WorkRequest();

  void initWithSgList(uint64_t id, ibv_wr_opcode op,
                      const std::vector<ScatterGatherEntry> &sg_list,
                      Flags flags);

public:
  WorkRequest(const WorkRequest &);
  /**
   * Constructs a WorkRequest with opcode IBV_WR_SEND: The content of the
   * local memory buffers will be sent to the remote. The sender does not
   * know where the data will be written on the remote end. A
   * ReceiveRequest will be consumed from the remote's receive queue. Make
   * sure the ReceiveRequest enqueued on the remote has enough space in its
   * sg_list to hold the entire contents of thus send.
   *
   * @param wr_id User-defined 64-bit value associated with this WR. Any
   * WorkCompletions (on both ends) generated for this WorkRequest will
   * contain this value.
   * @param sg_list Scatter/Gather array. Data will be read sequentially
   * from these as if they were a continuous block. If empty, a message
   * with a content size of 0 bytes will be sent. This list must not be
   * destroyed before the request is posted.
   */
  static WorkRequest Send(uint64_t wr_id,
                          const std::vector<ScatterGatherEntry> &sg_list,
                          Flags = Flags::None);

  /**
   * Constructs a WorkRequest with opcode IBV_WR_SEND_WITH_IMM: The content of
   * the local memory buffers will be sent to the remote. The sender does not
   * know where the data will be written on the remote end. A ReceiveRequest
   * will be consumed from the remote's receive queue.
   *
   * @param wr_id User-defined 64-bit value associated with this WR. Any
   * WorkCompletions (on both ends) generated for this WorkRequest will
   * contain this value.
   * @param sg_list Scatter/Gather array. Data will be read sequentially from
   * these as if they were a continuous block. If emtpy, a message with a
   * content size of 0 bytes will be sent. This list must not be destroyed
   * before the request is posted.
   * @param immediate 32 bits (network order) of auxiliary data that will be
   * placed in the WorkCompletion on the remote side.
   */
  static WorkRequest Send(uint64_t wr_id,
                          const std::vector<ScatterGatherEntry> &sg_list,
                          uint32_t immediate, Flags = Flags::None);

  /**
   * Constructs a WorkRequest with opcode IBV_WR_RDMA_WRITE: The content of
   * the local memory buffers will be written to a contiguous block of memory
   * in the remote's virtual address space (not necessarily physically
   * continuous). No Receive Reequests will be consumed on the remote.
   *
   * @param wr_id User-defined 64-bit value associated with this WR. Any
   * WorkCompletions (on both ends) generated for this WorkRequest will
   * contain this value.
   * @param sg_list Scatter/Gather array. Data will be read sequentially from
   * these as if they were a continuous block. This list must not be destroyed
   * before the request is posted.
   * @param remote_addr Start address for the write. The range from
   * remote_addr to remote_addr+sum(lengths of sg_list entries) must fit
   * within the remote MemoryRegion associated with rkey.
   * @param rkey Rkey of the remote MemoryRegion into which to write.
   */
  static WorkRequest RdmaWrite(uint64_t wr_id,
                               const std::vector<ScatterGatherEntry> &sg_list,
                               uint64_t remote_addr, uint32_t rkey,
                               Flags = Flags::None);

  /**
   * Constructs a WorkRequest with opcode IBV_WR_RDMA_WRITE_WITH_IMM: The
   * content of the local memory buffers will be written to a contiguous block
   * of memory in the remote's virtual address space (not necessarily
   * physically continuous). No Receive Reequests will be consumed on the
   * remote.
   *
   * @param wr_id User-defined 64-bit value associated with this WR. Any
   * WorkCompletions (on both ends) generated for this WorkRequest will
   * contain this value.
   * @param sg_list Scatter/Gather array. Data will be read sequentially from
   * these as if they were a continuous block. This list must not be destroyed
   * before the request is posted.
   * @param remote_addr Start address for the write. The range from
   * remote_addr to remote_addr+sum(lengths of sg_list entries) must fit
   * within the remote MemoryRegion associated with rkey.
   * @param rkey Rkey of the remote MemoryRegion into which to write.
   * @param immediate 32 bits (network order) of auxiliary data that will be
   * placed in the WorkCompletion on the remote side.
   */
  static WorkRequest RdmaWrite(uint64_t wr_id,
                               const std::vector<ScatterGatherEntry> &sg_list,
                               uint64_t remote_addr, uint32_t rkey,
                               uint32_t immediate, Flags = Flags::None);

  /**
   * Constructs a WorkRequest with opcode IBV_WR_RDMA_READ: Data is read from
   * a contiguous block of remote memory corresponding to the given rkey /
   * virtual address and written sequentially to the local buffer regions
   * specified in sg_list. No Receive Request will be consumed on the remote.
   *
   * @param wr_id User-defined 64-bit value associated with this WR. Any
   * WorkCompletions (on both ends) generated for this WorkRequest will
   * contain this value.
   * @param sg_list Scatter/Gather array. Data will be written sequentially to
   * these as if they were a continuous block.
   * @param remote_addr Remote start address for the read. The range from
   * remote_addr to remote_addr+sum(lengths of sg_list entries) must fit
   * within the remote MemoryRegion associated with rkey.
   * @param rkey Rkey of the remote MemoryRegion into which to write.
   */
  static WorkRequest RdmaRead(uint64_t wr_id,
                              const std::vector<ScatterGatherEntry> &sg_list,
                              uint64_t remote_addr, uint32_t rkey,
                              Flags = Flags::None);

  /**
   * Constructs a WorkRequest with opcode IBV_WR_ATOMIC_CMP_AND_SWP: A 64 bit
   * value is read from the given rkey / virtual address and compared to the
   * 'compare' value. If the values are the same, the remote location is
   * atomically overwritten with the 'swap' value. The original value is
   * written to the local regions designated by sg_list. No ReceiveRequest
   * will be consumed on the remote.
   *
   * @param wr_id User-defined 64-bit value associated with this WR. Any
   * WorkCompletions (on both ends) generated for this WorkRequest will
   * contain this value.
   * @param sg_list Scatter/Gather array. The original value of the remote
   * memory will be written sequentially to these as if they were a continuous
   * block. Total length of the entries must be at least 64 bits.
   * @param remote_addr Remote start address for the read. The associated
   * MemoryRegion must continue for at least 64 bits from this address.
   * @param rkey Rkey of the remote MemoryRegion from which to read.
   * @param compare The value to compare the remote value to.
   * @param swap The value to overwrite the remote location with.
   */
  static WorkRequest
  AtomicCompareAndSwap(uint64_t wr_id,
                       const std::vector<ScatterGatherEntry> &sg_list,
                       uint64_t remote_addr, uint32_t rkey, uint64_t compare,
                       uint64_t swap, Flags flags);

  /**
   * Constructs a WorkRequest with opcode IBV_WR_ATOMIC_FETCH_AND_ADD: A 64
   * bit value is read from the given rkey / virtual address, summed with
   * 'add' and the result is atomically written back to the remote location.
   * The original value is written to the local regions designated by sg_list.
   * No ReceiveRequest will be consumed on the remote.
   *
   * @param wr_id User-defined 64-bit value associated with this WR. Any
   * WorkCompletions (on both ends) generated for this WorkRequest will
   * contain this value.
   * @param sg_list Scatter/Gather array. The original value of the remote
   * memory will be written sequentially to these as if they were a continuous
   * block. Total length of the entries must be at least 64 bits.
   * @param remote_addr Remote start address for the read. The associated
   * MemoryRegion must continue for at least 64 bits from this address.
   * @param rkey Rkey of the remote MemoryRegion from which to read.
   * @param compare The value to be added to the remote value.
   */
  static WorkRequest AtomicFetchAndAdd(
      uint64_t wr_id, const std::vector<ScatterGatherEntry> &sg_list,
      uint64_t remote_addr, uint32_t rkey, uint64_t add, Flags flags);

  /**
   * @param next WorkRequest that should be performed after this. A copy of
   * next will be stored in this.
   */
  void chain(const WorkRequest &next);
};

/** Incoming Work Request. */
class ReceiveRequest {
private:
  ibv_recv_wr wr;
  // Hold references to memory regions to ensure they won't be deleted
  // prematurely
  std::vector<ibverbs::MemoryRegionPtr> mr_list;
  // The actual scatter/gather entries referencing the memory regions in
  // mr_list
  std::vector<ibv_sge> ibv_sg_list;
  std::unique_ptr<ReceiveRequest> next;

  friend class RdmaConnection;

public:
  ReceiveRequest() = delete;
  ReceiveRequest(const ReceiveRequest &other);

  /**
   * @param wr_id User-defined 64-bit value associated with this RR. The
   * WorkCompletion generated for this ReceiveRequest will contain this value.
   * @param sg_list received data will be sequentially written to the defined
   * buffer regions as if they were a contiguous block of memory.
   */
  ReceiveRequest(uint64_t wr_id,
                 const std::vector<ScatterGatherEntry> &sg_list);

  /**
   * @param next ReceiveRequest that should be performed after this. A copy of
   * next will be stored in this.
   */
  void chain(const ReceiveRequest &next);
};

class RdmaConnection;
typedef std::shared_ptr<RdmaConnection> RdmaConnectionPtr;

/** Helper class that takes care of setting up incoming RDMAcm connections */
class RdmaListener {
private:
  rdmacm::IdPtr listening_id;
  bool listening;

public:
  RdmaListener();

  /** Begin listening for RDMAcm connection requests. */
  void listen(uint16_t port = DEFAULT_PORT);

  /**
   * Block until a connection request is received, then return a
   * RdmaConnectionPtr after setting up the connection. Accepting a connection
   * will not stop the listening. NOT thread safe.
   */
  RdmaConnection accept();

  rdmacm::EventChannelPtr eventChannel() { return listening_id->cm_channel; }
};

class RdmaConnection {
private:
  rdmacm::IdPtr cm_id;
  ibverbs::ProtectionDomainPtr pd;
  ibverbs::QueuePairPtr qp;

  friend class RdmaListener;

  RdmaConnection(rdmacm::IdPtr);

public:
  RdmaConnection() = delete;
  /** Helper function for establishing an outbound connection */
  static RdmaConnection connect(const char *address,
                                uint16_t port = DEFAULT_PORT);
  /**
   * Push one or more work requests to the outbound work queue  See
   * WorkRequest::chain
   */
  void post(const WorkRequest &);
  /**
   * Push one or more work requests to the inbound work queue. See
   * ReceiveRequest::chain
   */
  void post(const ReceiveRequest &);
  /** Block until a Work Completion is generated on the outbound work queue */
  ibverbs::WorkCompletion awaitSendCompletion();
  /** Block until a Work Completion is generated on the inbound work queue */
  ibverbs::WorkCompletion awaitRecvCompletion();

  ibverbs::ProtectionDomainPtr protectionDomain() { return pd; };
  rdmacm::IdPtr id() { return cm_id; }
};

#endif
