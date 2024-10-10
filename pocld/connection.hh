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

#ifndef POCL_CONNECTION_HH
#define POCL_CONNECTION_HH

#include <memory>
#include <unistd.h>

#include "pocl_networking.h"
#include "traffic_monitor.hh"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

class Connection {
public:
  struct Parameters {
    int BufSize;
    int IsFast;
  };

  Connection() = delete;
  Connection(transport_domain_t Domain, int Fd,
             std::shared_ptr<TrafficMonitor> Meter);
  ~Connection();
  void configure(bool LowLatency);
  ssize_t writeFull(const void *Source, size_t Bytes);
  int readFull(void *Destination, size_t Bytes);
  int readReentrant(void *Destination, size_t Bytes, size_t *Tracker);
  int pollableFd();
  std::string describe();
  transport_domain_t domain() { return Domain; }
  void setMeter(std::shared_ptr<TrafficMonitor> M) { Meter = M; }
  std::shared_ptr<TrafficMonitor> meter() { return Meter; }

private:
  std::shared_ptr<TrafficMonitor> Meter;
  transport_domain_t Domain;
  int Fd;
  // TODO: also wrap RdmaConnection?
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif