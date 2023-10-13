/* request.hh - class that represents a command received from the client

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

#ifndef POCL_REMOTE_REQUEST_HH
#define POCL_REMOTE_REQUEST_HH

#include "messages.h"
#include <cstring>

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

class Request {

public:
  uint32_t req_size;
  size_t req_size_read;
  RequestMsg_t req;
  size_t req_read;

  uint64_t *waitlist;
  uint64_t waitlist_size;
  size_t waitlist_read;

  char *extra_data;
  uint64_t extra_size;
  size_t extra_read;

  char *extra_data2;
  uint64_t extra_size2;
  size_t extra_read2;

  // server host timestamps for network comm
  uint64_t read_start_timestamp_ns;
  uint64_t read_end_timestamp_ns;

  bool fully_read;

  Request()
      : req_size(0), req_size_read(0), req(), req_read(0), waitlist(nullptr),
        waitlist_size(0), waitlist_read(0), extra_data(nullptr), extra_size(0),
        extra_read(0), extra_data2(nullptr), extra_size2(0), extra_read2(0),
        read_start_timestamp_ns(0), read_end_timestamp_ns(0),
        fully_read(false) {}

  // Deep copying constructor
  Request(const Request &r)
      : req_size(r.req_size), req(r.req), req_size_read(r.req_size_read),
        waitlist(nullptr), waitlist_size(r.waitlist_size),
        waitlist_read(r.waitlist_read), extra_data(nullptr),
        extra_size(r.extra_size), extra_read(r.extra_read),
        extra_data2(nullptr), extra_size2(r.extra_size2),
        extra_read2(r.extra_read2),
        read_start_timestamp_ns(r.read_start_timestamp_ns),
        read_end_timestamp_ns(r.read_end_timestamp_ns),
        fully_read(r.fully_read) {
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

  bool read(int fd);
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
