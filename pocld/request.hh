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

#include <cstring>
#include <vector>

#include "connection.hh"
#include "messages.h"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

/// Helper class for reading a Request from memory instead of a Connection
/// stream
class ByteReader {
public:
  ByteReader() = delete;
  ByteReader(uint8_t *Start, size_t Len)
      : StartPtr(Start), Offset(0), Length(Len) {}
  bool eof() { return Offset >= Length; }
  int readFull(void *Destination, size_t Bytes);
  int readReentrant(void *Destination, size_t Bytes, size_t *Tracker);
  std::string describe();

private:
  uint8_t *StartPtr;
  size_t Offset;
  size_t Length;
};

class Request {
public:
  /// Size, in bytes, of the main request body (up to sizeof RequestMsg_t)
  uint32_t BodySize = 0;
  /// Tracker for how many bytes of req_size have been read from the network
  /// socket
  size_t BodySizeBytesRead = 0;
  /// Main body of the reuqest
  RequestMsg_t Body{};
  /// Tracker for how many bytes of the main request body have been read from
  /// the network socket
  size_t BodyBytesRead = 0;

  /// List of event ids that must complete before this Request can be processed
  std::vector<uint64_t> Waitlist;
  /// Tracker for how many bytes of the waitlist have been read
  size_t WaitlistBytesRead = 0;

  /// Auxiliary data required for the Request (buffer contents, program binaries
  /// etc)
  std::vector<uint8_t> ExtraData;
  /// Size of the auxiliary data buffer
  uint64_t ExtraDataSize = 0;
  /// Tracker for how many bytes of the auxiliary data buffer have been read
  /// from the network socket
  size_t ExtraDataBytesRead = 0;

  /// Second auxiliary data required for the Request
  std::vector<uint8_t> ExtraData2;
  /// Size of the auxiliary data buffer
  uint64_t ExtraData2Size = 0;
  /// Tracker for how many bytes of the second auxiliary data buffer have been
  /// read from the network socket
  size_t ExtraData2BytesRead = 0;

  /// Server side timestamp for when reading the request from the network socket
  /// started
  uint64_t ReadStartTimestampNS = 0;
  /// Server side timestamp for when the last bit of data for the request has
  /// been successfully read from the network socket.
  uint64_t ReadEndTimestampNS = 0;

  /// Flag indicating that the request has been fully read from the network
  /// socket. Set at the very end of the read() function.
  bool IsFullyRead = false;

  /// Incrementally reads the request from given Connection. Returns true on
  /// success and false if an error occurs while reading. Call repeatedly until
  /// `fully_read` gets set to true.
  bool read(Connection *);

  /// Attempts to copy a whole Request from a ByteReader. Returns true if
  /// Request was fully read, false otherwise.
  bool readFull(ByteReader *);
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
