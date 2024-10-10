/* traffic_monitor.cc - pocld helper for logging network traffic per session

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

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

#include "common.hh"
#include "pocl_debug.h"
#include "traffic_monitor.hh"

TrafficMonitor::TrafficMonitor(ExitHelper *e, std::string &client_id)
    : tx_bytes_submitted(0), tx_bytes_confirmed(0), rx_bytes_requested(0),
      rx_bytes_confirmed(0), eh(e), client_id(client_id) {
  const char *env_p = std::getenv("POCL_TRAFFIC_LOG_DIR");
  if (env_p == nullptr || env_p[0] == '\0') {
    POCL_MSG_PRINT_INFO(
        "POCL_TRAFFIC_LOG_DIR not set, stats will not be written to disk\n");
  } else {
    base_path = env_p;
    file_thread = std::thread{&TrafficMonitor::fileWriterThread, this};
  }
}

TrafficMonitor::~TrafficMonitor() {
  eh->requestExit("BandwidthMonitor exiting", 0);
  if (file_thread.joinable())
    file_thread.join();
}

void TrafficMonitor::fileWriterThread() {
  const auto starttime =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::stringstream filename;
  filename << "pocld-bandwidth-"
           << std::put_time(std::localtime(&starttime), "%F-%H.%M.%S") << "-"
           << client_id << ".csv";

  std::ofstream f(base_path / filename.str(), std::ios::out | std::ios::trunc);
  f << "timestamp,tx_bytes_submitted,tx_bytes_confirmed,rx_bytes_requested,rx_"
       "bytes_confirmed"
    << std::endl;
  std::string fieldsep = ",";
  std::string linesep = "\n";

  while (!eh->exit_requested()) {
    f << std::chrono::steady_clock::now().time_since_epoch().count() << fieldsep
      << tx_bytes_submitted << fieldsep << tx_bytes_confirmed << fieldsep
      << rx_bytes_requested << fieldsep << rx_bytes_confirmed << fieldsep
      << linesep;

    using std::chrono::operator""ms;
    std::this_thread::sleep_for(10ms);
  }
}
