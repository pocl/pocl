/* connection.hh - Interface of a wrapper class for various connection types

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

#include <cassert>
#include <sys/socket.h>

#include "connection.hh"
#include "pocl_networking.h"
#include "traffic_monitor.hh"

#define COMMAND_SOCKET_BUFSIZE (4 * 1024)
#define STREAM_SOCKET_BUFSIZE (4 * 1024 * 1024)

Connection::Connection(transport_domain_t Domain, int Fd,
                       std::shared_ptr<TrafficMonitor> Meter)
    : Domain(Domain), Fd(Fd), Meter(Meter) {}

Connection::~Connection() {
  shutdown(Fd, SHUT_RDWR);
  close(Fd);
}

void Connection::configure(bool LowLatency) {
  size_t BufSize = LowLatency ? COMMAND_SOCKET_BUFSIZE : STREAM_SOCKET_BUFSIZE;
  switch (Domain) {
  case TransportDomain_Inet:
    // function only actually checks whether ai_family is AF_VSOCK
    pocl_remote_client_set_socket_options(Fd, BufSize, LowLatency, AF_UNSPEC);
    break;
  case TransportDomain_Vsock:
    pocl_remote_client_set_socket_options(Fd, BufSize, LowLatency, AF_VSOCK);
    break;
  default:
    break;
  }
}

ssize_t Connection::writeFull(const void *Source, size_t Bytes) {
  size_t Written = 0;
  ssize_t Res;
  const char *Ptr = static_cast<const char *>(Source);

  if (Meter.get())
    Meter->txSubmitted(Bytes);

  while (Written < Bytes) {
    size_t Remaining = Bytes - Written;
    Res = ::write(Fd, Ptr + Written, Remaining);
    if (Res < 0) {
      int e = errno;
      if (e == EAGAIN || e == EWOULDBLOCK)
        continue;
      else
        return -1;
    }
    Written += (size_t)Res;

    if (Meter.get())
      Meter->txConfirmed(Res);
  }

  return 0;
}

int Connection::readFull(void *Destination, size_t Bytes) {
  size_t readb = 0;
  ssize_t res;
  char *Ptr = static_cast<char *>(Destination);
  if (Meter.get())
    Meter->rxRequested(Bytes);
  while (readb < Bytes) {
    size_t Remain = Bytes - readb;
    res = ::read(Fd, Ptr + readb, Remain);
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
    if (Meter.get())
      Meter->rxConfirmed(res);
  }

  return static_cast<ssize_t>(Bytes);
}

int Connection::readReentrant(void *Destination, size_t Bytes,
                              size_t *Tracker) {
  if (*Tracker == Bytes)
    return 0;

  ssize_t readb;
  readb = ::read(Fd, (char *)Destination + *Tracker, Bytes - *Tracker);
  if (readb < 0)
    return errno;

  if (readb == 0)
    return EPIPE;

  *Tracker += readb;
  if (*Tracker != Bytes)
    return EAGAIN;
  return 0;
}

int Connection::pollableFd() { return Fd; }

std::string Connection::describe() {
  switch (Domain) {
  case TransportDomain_Unix:
    return "unix:fd=" + std::to_string(Fd);
  case TransportDomain_Inet:
    return "tcp:fd=" + std::to_string(Fd);
  case TransportDomain_Vsock:
    return "vsock:fd=" + std::to_string(Fd);
  default:
    assert(!"Unhandled transport domain");
    return "[unknown]";
  }
}