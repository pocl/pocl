/* traffic_monitor.hh - pocld helper for tracking network use per session

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

#ifndef POCLD_TRAFFIC_MONITOR_HH
#define POCLD_TRAFFIC_MONITOR_HH

#include <filesystem>
#include <string>

#include "common.hh"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

class TrafficMonitor {
  std::atomic_uint64_t tx_bytes_submitted;
  std::atomic_uint64_t tx_bytes_confirmed;
  std::atomic_uint64_t rx_bytes_requested;
  std::atomic_uint64_t rx_bytes_confirmed;
  ExitHelper *eh;
  std::thread file_thread;
  std::string client_id;
  std::filesystem::path base_path;

public:
  TrafficMonitor(ExitHelper *e, std::string &client_id);
  ~TrafficMonitor();

  void fileWriterThread();

  inline void txSubmitted(uint64_t bytes) { tx_bytes_submitted += bytes; }
  inline void txConfirmed(uint64_t bytes) { tx_bytes_confirmed += bytes; }
  inline void rxRequested(uint64_t bytes) { rx_bytes_requested += bytes; }
  inline void rxConfirmed(uint64_t bytes) { rx_bytes_confirmed += bytes; }
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
