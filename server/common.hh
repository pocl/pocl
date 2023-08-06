/* common.hh - miscellaneous macros, types and classes used in several places
   around pocld

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

#ifndef POCL_REMOTE_COMMON_HH
#define POCL_REMOTE_COMMON_HH

#include "pocld_config.h"

#include <dlfcn.h>

#include <algorithm>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <atomic>
#include <condition_variable>
#include <mutex>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

using Time = std::chrono::steady_clock;
using ms = std::chrono::milliseconds;

using float_sec = std::chrono::duration<double>;
using float_time_point = std::chrono::time_point<Time, float_sec>;

#define MS_PER_S 1000

#include "common_cl.hh"
#include "pocl_debug.h"
#include "session.hh"

#ifdef ENABLE_RDMA
#include "guarded_queue.hh"
#include "rdma.hh"
#endif

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

#ifdef ENABLE_RDMA
ptrdiff_t transfer_src_offset(const RequestMsg_t &msg);
uint64_t transfer_size(const RequestMsg_t &msg);
#endif

#define RETURN_IF_ERR_CODE(code)                                               \
  do {                                                                         \
    int err = code;                                                            \
    if (err == 0)                                                              \
      rep->rep.failed = 0;                                                     \
    else {                                                                     \
      POCL_MSG_ERR("reply FAIL with: %d\n", err);                              \
      rep->rep.data_size = 0;                                                  \
      rep->rep.fail_details = err;                                             \
      rep->rep.failed = 1;                                                     \
      rep->rep.obj_id = 0;                                                     \
      rep->rep.message_type = MessageType_Failure;                             \
      rep->extra_data = nullptr;                                               \
      rep->extra_size = 0;                                                     \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define RETURN_IF_ERR                                                          \
  do {                                                                         \
    if (err == 0)                                                              \
      rep->rep.failed = 0;                                                     \
    else {                                                                     \
      POCL_MSG_ERR("reply FAIL with: %d\n", err);                              \
      rep->rep.data_size = 0;                                                  \
      rep->rep.fail_details = err;                                             \
      rep->rep.failed = 1;                                                     \
      rep->rep.obj_id = 0;                                                     \
      rep->rep.message_type = MessageType_Failure;                             \
      rep->extra_data = nullptr;                                               \
      rep->extra_size = 0;                                                     \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define RETURN_IF_ERR_DATA                                                     \
  do {                                                                         \
    if (err == 0)                                                              \
      rep->rep.failed = 0;                                                     \
    else {                                                                     \
      POCL_MSG_ERR("reply FAIL with DATA: %d\n", err);                         \
      rep->rep.data_size = rep->extra_size;                                    \
      rep->rep.fail_details = err;                                             \
      rep->rep.failed = 1;                                                     \
      rep->rep.obj_id = 0;                                                     \
      rep->rep.message_type = MessageType_Failure;                             \
      if (rep->extra_size == 0)                                                \
        rep->extra_data = nullptr;                                             \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define CHECK_ID_EXISTS2(set, err, ID)                                         \
  do {                                                                         \
    if (set.find(ID) == set.end()) {                                           \
      POCL_MSG_ERR("Can't find object with ID " #ID "; reply FAIL with: %d",   \
                   err);                                                       \
      rep->rep.data_size = 0;                                                  \
      rep->rep.fail_details = err;                                             \
      rep->rep.failed = 1;                                                     \
      rep->rep.obj_id = 0;                                                     \
      rep->rep.message_type = MessageType_Failure;                             \
      rep->extra_data = nullptr;                                               \
      rep->extra_size = 0;                                                     \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define CHECK_ID_EXISTS(set, err) CHECK_ID_EXISTS2(set, err, id)

#define CHECK_ID_NOT_EXISTS(set, err)                                          \
  do {                                                                         \
    if (set.find(id) != set.end()) {                                           \
      POCL_MSG_ERR("FOUND object with ID %" PRIu32 "; reply FAIL with: %d",    \
                   id, err);                                                   \
      rep->rep.data_size = 0;                                                  \
      rep->rep.fail_details = err;                                             \
      rep->rep.failed = 1;                                                     \
      rep->rep.obj_id = 0;                                                     \
      rep->rep.message_type = MessageType_Failure;                             \
      rep->extra_data = nullptr;                                               \
      rep->extra_size = 0;                                                     \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define CHECK_READ(readb, call, name)                                          \
  do {                                                                         \
    readb = call;                                                              \
    if (readb < 0) {                                                           \
      int e = errno;                                                           \
      POCL_MSG_ERR("%s: error %d on " #call "!\n", name, e);                   \
      delete request;                                                          \
      eh->requestExit(name, e);                                                \
      return;                                                                  \
    }                                                                          \
    if (readb == 0) {                                                          \
      POCL_MSG_ERR("%s: Filedescriptor closed (client disconnect) on %s.\n",   \
                   name, #call);                                               \
      delete request;                                                          \
      eh->requestExit(name, 0);                                                \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define CHECK_READ_RETRY(readb, call, name)                                    \
  do {                                                                         \
    readb = call;                                                              \
    if (readb < 0) {                                                           \
      int e = errno;                                                           \
      POCL_MSG_ERR("%s: error on " #call ": %s, fd=%d!\n", name, strerror(e),  \
                   fd);                                                        \
      delete request;                                                          \
      goto RETRY;                                                              \
    }                                                                          \
    if (readb == 0) {                                                          \
      delete request;                                                          \
      goto RETRY;                                                              \
    }                                                                          \
  } while (0)

#define CHECK_READ_RET(readb, call, name, ret)                                 \
  do {                                                                         \
    readb = call;                                                              \
    if (readb < 0) {                                                           \
      int e = errno;                                                           \
      POCL_MSG_ERR("%s: error on " #call ": %s\n", name, strerror(e));         \
      eh->requestExit(name, e);                                                \
      return ret;                                                              \
    }                                                                          \
    if (readb == 0) {                                                          \
      POCL_MSG_ERR("%s: Filedescriptor closed (client disconnect) on %s\n",    \
                   name, #call);                                               \
      eh->requestExit(name, 0);                                                \
      return ret;                                                              \
    }                                                                          \
  } while (0)

#define CHECK_WRITE(call, name)                                                \
  do {                                                                         \
    int res = call;                                                            \
    if (res < 0) {                                                             \
      int e = errno;                                                           \
      eh->requestExit(name, e);                                                \
      delete r;                                                                \
      POCL_MSG_ERR("%s: error on " #call ": %s!\n", name, strerror(e));        \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define CHECK_WRITE_RETRY(call, name)                                          \
  do {                                                                         \
    int res = call;                                                            \
    if (res < 0) {                                                             \
      int e = errno;                                                           \
      POCL_MSG_ERR("%s: error on " #call ": %s!\n", name, strerror(e));        \
      goto RETRY;                                                              \
    }                                                                          \
  } while (0)

#define CHECK_WRITE_RET(call, name, ret)                                       \
  do {                                                                         \
    int res = call;                                                            \
    if (res < 0) {                                                             \
      int e = errno;                                                           \
      eh->requestExit(name, e);                                                \
      POCL_MSG_ERR("%s: error on " #call ": %s!\n", name, strerror(e));        \
      return ret;                                                              \
    }                                                                          \
  } while (0)

typedef std::array<size_t, 3> sizet_vec3;

#define COPY_VEC3(new_vec, old_vec)                                            \
  sizet_vec3 new_vec;                                                          \
  new_vec[0] = old_vec.x;                                                      \
  new_vec[1] = old_vec.y;                                                      \
  new_vec[2] = old_vec.z;

class TrafficMonitor;

ssize_t read_full(int fd, void *p, size_t total, TrafficMonitor *);

int write_full(int fd, void *p, size_t total, TrafficMonitor *);

#ifdef ENABLE_RDMA
struct RdmaBufferData {
  char *shadow_buf;
  ibverbs::MemoryRegionPtr shadow_region;
};
struct RdmaRemoteBufferData {
  uint64_t address;
  uint32_t rkey;
};
#endif

struct peer_connection_t {
  int fd;
#ifdef ENABLE_RDMA
  std::shared_ptr<RdmaConnection> rdma;
#endif
};

class VirtualContextBase;

struct peer_listener_data_t {
  unsigned short port;
  unsigned short peer_rdma_port;
  std::mutex mutex;
  std::unordered_map<std::string, std::pair<std::condition_variable,
                                            std::vector<peer_connection_t>> *>
      incoming_peers;
#ifdef ENABLE_RDMA
  std::shared_ptr<RdmaListener> rdma_listener;
  std::mutex vctx_map_mutex;
  std::unordered_map<std::string, VirtualContextBase *> vctx_map;
  std::mutex peer_cm_id_to_vctx_mutex;
  std::unordered_map<rdma_cm_id *, VirtualContextBase *> peer_cm_id_to_vctx;
#endif
};

class Request {

public:
  RequestMsg_t req;

  uint64_t *waitlist;
  uint32_t waitlist_size;

  char *extra_data;
  size_t extra_size;

  char *extra_data2;
  size_t extra_size2;

  // server host timestamps for network comm
  uint64_t read_start_timestamp_ns;
  uint64_t read_end_timestamp_ns;

  Request()
      : req(), waitlist(nullptr), waitlist_size(0), extra_data(nullptr),
        extra_size(0), extra_data2(nullptr), extra_size2(0) {}

  // Deep copying constructor
  Request(const Request &r)
      : req(r.req), waitlist(nullptr), waitlist_size(r.waitlist_size),
        extra_data(nullptr), extra_size(r.extra_size), extra_data2(nullptr),
        extra_size2(r.extra_size2) {
    if (r.waitlist) {
      waitlist = new uint64_t[waitlist_size];
      std::memcpy(waitlist, r.waitlist, sizeof(uint64_t) * waitlist_size);
    }

    if (r.extra_data) {
      extra_data = new char[extra_size];
      std::memcpy(extra_data, r.extra_data, extra_size);
    }

    if (r.extra_data2) {
      extra_data = new char[extra_size2];
      std::memcpy(extra_data2, r.extra_data2, extra_size2);
    }
  }

  ~Request() {
    if (waitlist)
      delete[] waitlist;
    if (extra_data)
      delete[] extra_data;
    if (extra_data2)
      delete[] extra_data2;
  }
};

class Reply {

public:
  ReplyMsg_t rep;
  Request *req;
  char *extra_data;
  size_t extra_size;
  cl::Event event;
  // server host timestamps for network comm
  uint64_t write_start_timestamp_ns;

  Reply()
      : rep(), req(nullptr), extra_data(nullptr), extra_size(0),
        event(nullptr) {}

  ~Reply() {
    if (req)
      delete req;
    if (extra_data)
      delete[] extra_data;
  }
};

void replyID(Reply *rep, ReplyMessageType t, uint32_t id);

void replyFail(ReplyMsg_t *rep, RequestMsg_t *req, int err);

void replyOK(Reply *rep, ReplyMessageType t);

void replyData(Reply *rep, ReplyMessageType t, size_t data_size);

void replyData(Reply *rep, ReplyMessageType t, uint32_t id, size_t data_size);

/******************/

void replyOK(Reply *rep, EventTiming_t &evt, ReplyMessageType t);

void replyData(Reply *rep, EventTiming_t &evt, ReplyMessageType t, uint32_t id,
               size_t data_size);

void replyData(Reply *rep, EventTiming_t &evt, ReplyMessageType t,
               size_t data_size);

class ExitHelper {
  int exit_status;
  int requested_exit;
  mutable std::mutex exit_mutex;
  std::condition_variable exit_condvar;

public:
  ExitHelper() : exit_status(0), requested_exit(0){};

  int requestExit(const char *msg, int status) {
    std::unique_lock<std::mutex> lock(exit_mutex);

    if (requested_exit > 0) {
      POCL_MSG_PRINT_GENERAL("%s : EXIT ALREADY requested, returning\n", msg);
      return 1;
    }
    POCL_MSG_PRINT_GENERAL("%s : EXIT requested \n", msg);
    requested_exit = 13;
    exit_status = status;
    lock.unlock();
    exit_condvar.notify_one();
    return 0;
  }

  void waitUntilExit() {
    std::unique_lock<std::mutex> lock(exit_mutex);

    while (requested_exit != 13)
      exit_condvar.wait(lock);
  }

  int exit_requested() const {
    std::unique_lock<std::mutex> lock(exit_mutex);
    return requested_exit > 0;
  }

  int status() const {
    std::unique_lock<std::mutex> lock(exit_mutex);
    return exit_status;
  }
};

struct EventPair {
  cl::Event native;
  cl::UserEvent user;
};

std::string hexstr(const std::string &);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
