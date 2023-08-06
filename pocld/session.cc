/* session.cc - pocld client session store

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

#include <cstdint>
#include <cstring>
#include <mutex>
#include <random>
#include <unordered_map>

#include <sys/socket.h>

#include "common.hh"
#include "pocl_debug.h"
#include "session.hh"

std::unordered_map<std::string, client_connections_t> session_store;

bool pass_new_client_fds(int fd_command, int fd_stream,
                         const std::string &session) {
  // look up session in store and pass fds via unix socket
  auto it = session_store.find(session);
  if (it == session_store.end()) {
    // TODO: handle this gracefully
    POCL_MSG_WARN("SERVER: Client tried to connect to invalid session\n");
    return false;
  }

  *session_store[session].fd_command = fd_command;
  *session_store[session].fd_stream = fd_stream;
  return true;
}

client_connections_t register_new_session(int fd_command, int fd_stream,
                                          std::string &session) {
  std::random_device rd;
  std::default_random_engine dice(rd());
  std::uniform_int_distribution<uint8_t> dist(0, UINT8_MAX);

  client_connections_t session_conns = {new std::atomic_int(fd_command),
                                        new std::atomic_int(fd_stream)};

  // TODO: give up at some point and fail?
  bool unique = false;
  while (!unique) {
    for (int i = 0; i < SESSION_ID_LENGTH; i++) {
      session.push_back(dist(dice));
    }
    unique =
        session_store.insert(std::make_pair(hexstr(session), session_conns))
            .second;
  }

  return session_conns;
}
