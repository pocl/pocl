/* daemon.hh - interface & types for the PoclDaemon class

   Copyright (c) 2024 Jan Solanti / Tampere University

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

#include <atomic>
#include <cstdint>
#include <mutex>
#include <sys/socket.h>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common.hh"
#include "connection.hh"
#ifdef ENABLE_RDMA
#include "guarded_queue.hh"
#endif
#include "virtual_cl_context.hh"
#ifdef ENABLE_REMOTE_ADVERTISEMENT_AVAHI
#include "avahi_advertise.hh"
#endif

/** Helper struct to hold the port numbers that the server listens on */
struct ServerPorts {
  /** Port for "fast" incoming client connections (small commands, low latency
   * settings) */
  uint16_t command;
  /** Port for "slow" incoming client connections (bulk data transfer, large
   * internal buffers) */
  uint16_t stream;
  /** Port for incoming P2P server connections */
  uint16_t peer;
#ifdef ENABLE_RDMA
  /** Port for incoming P2P server RDMAcm connections */
  uint16_t peer_rdma;
  /** Port for incoming client RDMAcm connections */
  uint16_t rdma;
#endif
};

/**
 * A wrapper class to hold all state of a single server instance. This is mainly
 * for keeping shared variables in one place and out of global scope.
 */
class PoclDaemon {
public:
  ~PoclDaemon();

  /**
   * Sets up client listener sockets, binds them to the given address/ports and
   * begins listening for connection requests. Launches threads for listening
   * for P2P server connections and RDMAcm connections (both client and server)
   * and finally launches the main I/O thread running
   * `readAllClientSocketsThread()`
   */
  int launch(std::string ListenAddress, struct ServerPorts &ports,
             bool UseVsock = false);

  /**
   * Main function of the client I/O thread. Polls client sockets for new
   * connections and open connections for new requests. Pushes requests to their
   * respective context once they are fully read.
   *
   * The main loop within this function consists of 3 phases: the poll(), the
   * reads and the reaping of closed fds.
   *
   * First, a vector of pollfd descriptors is constructed for all open sockets.
   * This vector is cached across iterations and rebuilt from scratch whenever
   * the list of open fds changes. Once the list is constructed (if needed),
   * poll(2) is invoked, putting the thread to sleep until something happens.
   *
   * Once poll returns, two things happen: if there are new connection requests
   * on the client listener sockets, they are accepted.
   *
   * After the special case of the listener sockets, if there are further events
   * in the poll result, the respective fds are handed to Result::read for
   * reading a piece of the next command sent over that socket. Similar to the
   * pollfd list, the function keeps a list of in-flight Requests for this
   * purpose. Once a valid handshake or request for a new session has been
   * received, the fd is handed off to the respective VirtualCLContext.
   *
   * Finally, the function goes over the "dead" fds list, closes them and
   * removes the fd and its in-flight Request.
   */
  void readAllClientSocketsThread();

  /**
   * \brief Push Ctx to the deleter thread to be destroyed
   *
   * Properly deinitializing a VirtualCLContext can take a bit of time, so that
   * process is delegated to a separate thread in order to not block the socket
   * polling thread.
   */
  void releaseContextDeferred(VirtualContextBase *Ctx) {
    std::unique_lock<std::mutex> L(ContextDeleterMtx);
    ContextDeleteQueue.push_back(Ctx);
    ContextDeleterCond.notify_one();
  }

  /**
   * Main loop for the thread in charge of deleting VirtualCLContexts
   */
  void contextDeleterThread();

  /** Block until the main I/O and context deleter threads exit. */
  void waitForExit() {
    if (ClientPoller.joinable())
      ClientPoller.join();
    if (ContextDeleter.joinable()) {
      ContextDeleterCond.notify_one();
      ContextDeleter.join();
    }
  }

  /* returns nullptr on error */
  VirtualContextBase *performSessionSetup(std::shared_ptr<Connection> Conn,
                                          Request *R);

private:
  ExitHelper exit_helper;
  /** Port numbers that the server is listening on */
  struct ServerPorts ListenPorts;
  std::vector<std::shared_ptr<Connection>> OpenClientConnections;
  size_t NumListenFds;
  std::mutex SessionListMtx;
  std::unordered_map<uint64_t, VirtualContextBase *> ClientSessions;
  std::unordered_map<uint64_t, std::array<uint8_t, AUTHKEY_LENGTH>> SessionKeys;
  std::thread ContextDeleter;
  std::mutex ContextDeleterMtx;
  std::condition_variable ContextDeleterCond;
  std::vector<VirtualContextBase *> ContextDeleteQueue;
  std::atomic_uint64_t LastSessionId;
  std::thread ClientPoller;
  peer_listener_data_t PeerListenerData;
  std::thread PeerListenerThread;
#ifdef ENABLE_REMOTE_ADVERTISEMENT_AVAHI
  AvahiAdvertise *avahiAdvertiseP;
#endif
#ifdef ENABLE_REMOTE_ADVERTISEMENT_DHT
  std::thread DHTThread;
  friend void StartDHTAdvert(PoclDaemon *d, addrinfo *RA,
                             struct ServerPorts &Ports);
#endif
#ifdef ENABLE_RDMA
  RdmaListener rdma_listener;
  std::thread pl_rdma_event_th;
  std::thread client_rdma_event_th;
  GuardedQueue<rdma_cm_event *> cm_event_queue;
  std::unordered_map<rdma_cm_id *, VirtualContextBase *> cm_id_to_vctx;
  std::mutex cm_id_to_vctx_mutex;
#endif
};
