/* virtual_cl_context.hh - pocld class that holds all resources of a session

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

#ifndef POCL_REMOTE_VIRTUAL_CL_HH
#define POCL_REMOTE_VIRTUAL_CL_HH

#include "common.hh"

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

class SharedContextBase;

class VirtualContextBase {

public:
  virtual ~VirtualContextBase() {}

  virtual void nonQueuedPush(Request *req) = 0;

  virtual void queuedPush(Request *req) = 0;

#ifdef ENABLE_RDMA
  virtual bool clientUsesRdma() = 0;

  virtual char *getRdmaShadowPtr(uint32_t id) = 0;
#endif

  virtual void requestExit(int code, const char *reason) = 0;

  virtual void broadcastToPeers(const Request &req) = 0;

  virtual void notifyEvent(uint64_t event_id, cl_int status) = 0;

  virtual void unknownRequest(Request *req) = 0;

  virtual int run() = 0;

  virtual SharedContextBase *getDefaultContext() = 0;
};

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#endif
