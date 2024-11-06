/* pocl_ipc_mutex.h: Named interprocess mutex that does not cause deadlock if a
   process dies or exits without unlocking.

   Copyright (c) 2024 Henry Linjamäki / Intel Finland Oy

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "pocl_debug.h"
#include "pocl_ipc_mutex.h"

#include <windows.h>

#include <cassert>
#include <string>

int pocl_ipc_mutex_create(const char *Name, pocl_ipc_mutex_t *IpcMtx) {
  assert(IpcMtx && "IpcMtx must not be nullptr!");
  if (std::string(Name).size() > MAX_PATH) {
    POCL_MSG_ERR("The mutex name exceeds an OS limit.");
    return -1;
  }

  HANDLE MtxHandle = CreateMutexA(NULL, FALSE, Name);
  if (MtxHandle == NULL) {
    POCL_MSG_ERR("CreateMutexA returned an error. GetLastError()=%zu\n",
                 static_cast<size_t>(GetLastError()));
    return -1;
  }
  IpcMtx->handle = static_cast<void *>(MtxHandle);
  return 0;
}

int pocl_ipc_mutex_lock(pocl_ipc_mutex_t IpcMtx) {
  assert(IpcMtx.handle && "pocl_ipc_mutex_create() must be called first!");
  auto MtxHandle = static_cast<HANDLE>(IpcMtx.handle);
  switch (WaitForSingleObject(MtxHandle, INFINITE)) {
  case WAIT_ABANDONED:
    POCL_MSG_WARN(
        "Another thread/process terminated before unlocking this mutex.");
    // FALL-THROUGH
  case WAIT_OBJECT_0:
    return 0;

  case WAIT_FAILED:
    POCL_MSG_WARN("Couldn't unlock an IPC mutex. GetLastError()=%zu\n",
                  static_cast<size_t>(GetLastError()));
    return -1;

  case WAIT_TIMEOUT:
  default:
    assert(!"Unexpected wait state!");
    return -1;
  }

  return 0;
}

int pocl_ipc_mutex_create_and_lock(const char *Name, pocl_ipc_mutex_t *IpcMtx) {
  assert(IpcMtx && "IpcMtx must not be nullptr!");
  if (int error = pocl_ipc_mutex_create(Name, IpcMtx))
    return error;
  if (int error = pocl_ipc_mutex_lock(*IpcMtx)) {
    pocl_ipc_mutex_release(IpcMtx);
    return error;
  }
  return 0;
}

void pocl_ipc_mutex_release(pocl_ipc_mutex_t *IpcMtx) {
  assert(IpcMtx && "Invalid pocl_ipc_mutex_t handle!");
  auto MtxHandle = static_cast<HANDLE>(IpcMtx->handle);
  if (!CloseHandle(MtxHandle)) {
    POCL_MSG_WARN("CloseHandle returned an error. GetLastError()=%zu\n",
                  static_cast<size_t>(GetLastError()));
    assert(!"Failed to release an IPC mutex!");
  }
  IpcMtx->handle = NULL;
}

void pocl_ipc_mutex_unlock_and_release(pocl_ipc_mutex_t *IpcMtx) {
  assert(IpcMtx && "Invalid pocl_ipc_mutex_t handle!");
  if (IpcMtx->handle == nullptr)
    return;

  auto MtxHandle = static_cast<HANDLE>(IpcMtx->handle);
  if (!ReleaseMutex(MtxHandle)) {
    POCL_MSG_WARN("ReleaseMutex returned an error. GetLastError()=%zu\n",
                  static_cast<size_t>(GetLastError()));
    assert(!"Failed to release an IPC mutex!");
  }
  pocl_ipc_mutex_release(IpcMtx);
}
