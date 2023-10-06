/* common.cc -- shared helper function impls used in pocld

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

#include "common.hh"
#include "traffic_monitor.hh"

#include <arpa/inet.h>
#include <cassert>
#include <iomanip>
#include <netdb.h>
#include <sstream>
#include <sys/uio.h>
#include <unistd.h>

ssize_t read_full(int fd, void *p, size_t total, TrafficMonitor *tm) {
  //      float_time_point start = Time::now();

  size_t readb = 0;
  ssize_t res;
  char *ptr = static_cast<char *>(p);
  if (tm)
    tm->rxRequested(total);
  while (readb < total) {
    size_t remain = total - readb;
    res = ::read(fd, ptr + readb, remain);
    if (res < 0) {
      int e = errno;
      if (e == EAGAIN || e == EWOULDBLOCK)
        continue;
      else
        return -1;
    }
    if (res == 0) { // EOF
      return 0;
    }
    readb += (size_t)res;
    if (tm)
      tm->rxConfirmed(res);
  }

  return static_cast<ssize_t>(total);
  //      float_time_point end = Time::now();
  //      float_sec elapsed_seconds = end-start;
  //      std::cerr << "read " << total << " bytes   @ " <<
  //      elapsed_seconds.count() << "\n";
}

int write_full(int fd, void *p, size_t total, TrafficMonitor *tm) {
  //      float_time_point start = Time::now();

  size_t written = 0;
  ssize_t res;
  char *ptr = static_cast<char *>(p);
  if (tm)
    tm->txSubmitted(total);
  while (written < total) {
    size_t remain = total - written;
    res = ::write(fd, ptr + written, remain);
    if (res < 0) {
      int e = errno;
      if (e == EAGAIN || e == EWOULDBLOCK)
        continue;
      else
        return -1;
    }
    written += (size_t)res;
    if (tm)
      tm->txConfirmed(res);
  }
  return 0;

  //      float_time_point end = Time::now();
  //      float_sec elapsed_seconds = end-start;
  //      std::cerr << "written " << total << " bytes   @ " <<
  //      elapsed_seconds.count() << "\n";
}

void replyID(Reply *rep, ReplyMessageType t, uint32_t id) {
  rep->rep.message_type = t;
  rep->rep.failed = 0;
  rep->rep.fail_details = 0;
  rep->rep.data_size = 0;
  rep->rep.obj_id = id;
}

void replyOK(Reply *rep, ReplyMessageType t) { replyID(rep, t, 0); }

void replyOK(Reply *rep, EventTiming_t &evt, ReplyMessageType t) {
  rep->rep.message_type = t;
  rep->rep.failed = 0;
  rep->rep.fail_details = 0;
  rep->rep.data_size = 0;
  rep->rep.obj_id = 0;
}

void replyFail(ReplyMsg_t *rep, RequestMsg_t *req, int err) {
  rep->message_type = MessageType_Failure;
  rep->failed = 1;
  rep->fail_details = err;
  rep->data_size = 0;
  rep->obj_id = 0;

  if (req) {
    rep->obj_id = req->obj_id;
    rep->client_did = req->client_did;
    rep->did = req->did;
    rep->pid = req->pid;
    rep->msg_id = req->msg_id;
  }
}

void replyData(Reply *rep, ReplyMessageType t, uint32_t id, size_t data_size) {
  rep->rep.message_type = t;
  rep->rep.failed = 0;
  rep->rep.fail_details = 0;
  rep->rep.data_size = data_size;
  rep->rep.obj_id = id;
}

void replyData(Reply *rep, ReplyMessageType t, size_t data_size) {
  replyData(rep, t, 0, data_size);
}

void replyData(Reply *rep, EventTiming_t &evt, ReplyMessageType t, uint32_t id,
               size_t data_size) {
  replyData(rep, t, data_size);
}

void replyData(Reply *rep, EventTiming_t &evt, ReplyMessageType t,
               size_t data_size) {
  replyData(rep, evt, t, 0, data_size);
}

std::string hexstr(const std::string &i) {
  std::stringstream ss;
  ss << std::hex << std::setfill('0');
  for (char b : i) {
    ss << std::setw(2) << (int)(uint8_t)b;
  }

  return std::string(ss.str());
}

std::string describe_sockaddr(struct sockaddr *addr, unsigned addr_size) {
  std::string ip_str(INET6_ADDRSTRLEN, '\0');
  if (addr->sa_family == AF_INET)
    inet_ntop(addr->sa_family, &((struct sockaddr_in *)addr)->sin_addr,
              ip_str.data(), ip_str.capacity());
  else if (addr->sa_family == AF_INET6)
    inet_ntop(addr->sa_family, &((struct sockaddr_in6 *)addr)->sin6_addr,
              ip_str.data(), ip_str.capacity());
  else
    ip_str = "[unknown address family]";
  const char *end =
      (const char *)memchr(ip_str.c_str(), '\0', ip_str.capacity());
  if (end)
    ip_str.resize(end - ip_str.c_str());

  std::string host_str(NI_MAXHOST, '\0');
  int res = getnameinfo(addr, addr_size, host_str.data(), host_str.capacity(),
                        NULL, 0, 0);
  end = (const char *)memchr(host_str.c_str(), '\0', host_str.capacity());
  if (end)
    host_str.resize(end - host_str.c_str());

  return host_str + " (" + ip_str + ")";
}

#ifdef ENABLE_RDMA
ptrdiff_t transfer_src_offset(const RequestMsg_t &msg) {
  switch (msg.message_type) {
  case MessageType_ReadBuffer:
    return msg.m.read.src_offset;
    break;
  case MessageType_MigrateD2D:
    return 0;
    break;
  default:
    assert(!"Unhandled RDMA message");
    return 0;
    break;
  }
}

uint64_t transfer_size(const RequestMsg_t &msg) {
  switch (msg.message_type) {
  case MessageType_ReadBuffer:
    return msg.m.read.content_size;
    break;
  case MessageType_MigrateD2D:
    return msg.m.migrate.size;
    break;
  default:
    assert(!"Unhandled RDMA message");
    return 0;
    break;
  }
}
#endif
