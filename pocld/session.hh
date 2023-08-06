/* session.hh - pocld session store

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

#ifndef POCL_SESSION_HH
#define POCL_SESSION_HH

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

#ifdef ENABLE_RDMA
#include "rdma.hh"
#endif

#include "messages.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

struct peer_connection_t;

struct client_connections_t {
  std::atomic_int *fd_command;
  std::atomic_int *fd_stream;
  std::mutex *incoming_peer_mutex;
  std::pair<std::condition_variable, std::vector<peer_connection_t>>
      *incoming_peer_queue;
#ifdef ENABLE_RDMA
  // TODO this does not really work with reconnecting
  std::shared_ptr<RdmaConnection> rdma;
#endif
};

bool pass_new_client_fds(int fd_command, int fd_stream,
                         const std::string &session);
client_connections_t register_new_session(int fd_command, int fd_stream,
                                          std::string &session);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif // SESSION_HH
