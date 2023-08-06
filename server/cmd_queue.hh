/* cmd_queue.hh -- a class that handles enqueuing commands and collecting
   results

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

#ifndef POCL_REMOTE_CMD_QUEUE_HH
#define POCL_REMOTE_CMD_QUEUE_HH

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

class ReplyQueueThread;
class SharedContextBase;

class CommandQueue {
  SharedContextBase *backend;
  size_t exit_requested;
  uint32_t queue_id;
  uint32_t dev_id;
  ReplyQueueThread *write_slow, *write_fast;
  std::vector<Request *> pending;

public:
  CommandQueue(SharedContextBase *b, uint32_t queue_id, uint32_t did,
               ReplyQueueThread *s, ReplyQueueThread *f);

  ~CommandQueue();

  void push(Request *request);

  void notify();

private:
  bool TryRun(Request *request);

  void RunCommand(Request *request);

  void MigrateMemObj(uint32_t queue_id, Request *req, Reply *rep);

  void ReadBuffer(uint32_t queue_id, Request *req, Reply *rep);

  void WriteBuffer(uint32_t queue_id, Request *req, Reply *rep);

  void CopyBuffer(uint32_t queue_id, Request *req, Reply *rep);

  void ReadBufferRect(uint32_t queue_id, Request *req, Reply *rep);

  void WriteBufferRect(uint32_t queue_id, Request *req, Reply *rep);

  void CopyBufferRect(uint32_t queue_id, Request *req, Reply *rep);

  void FillBuffer(uint32_t queue_id, Request *req, Reply *rep);

  void RunKernel(uint32_t queue_id, Request *req, Reply *rep);

  /******************/

  void FillImage(uint32_t queue_id, Request *req, Reply *rep);

  void ReadImageRect(uint32_t queue_id, Request *req, Reply *rep);

  void WriteImageRect(uint32_t queue_id, Request *req, Reply *rep);

  void CopyBuffer2Image(uint32_t queue_id, Request *req, Reply *rep);

  void CopyImage2Buffer(uint32_t queue_id, Request *req, Reply *rep);

  void CopyImage2Image(uint32_t queue_id, Request *req, Reply *rep);
};

typedef std::unique_ptr<CommandQueue> CommandQueueUPtr;

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
