/* pocl_ipc_mutex.h: Named interprocess mutex

   Copyright (c) 2024 Henry Linjamäki / Intel Finland Oy

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
/* \file pocl_ip_mutex.h named interprocess mutex that does not
 * cause deadlock if a process dies or exits without unlocking
 * first. Beware that deadlocks are still possible within process if a
 * thread forgets to unlock the mutex.  */

#ifndef POCL_IPC_MUTEX_H
#define POCL_IPC_MUTEX_H

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct pocl_ipc_mutex_s
{
  void *handle;
} pocl_ipc_mutex_t;

/** Creates a mutex by 'name' or opens an existing one.
 *
 * On success, the return value is zero and a valid mutex object returned
 * via 'ipc_mtx'.  */
int pocl_ipc_mutex_create (const char *name, pocl_ipc_mutex_t *ipc_mtx);

/** Lock the mutex.
 *
 * The call is blocked if another process or thread by the same name
 * has locked the mutex. The mutex is unlocked in case the process
 * holding the lock terminates without unlocking first.
 *
 * The 'ípc_mtx' must be a valid object created by pocl_ipc_mutex_create().
 *
 * If mutex is locked successfully, zero is returned. Otherwise, on an error,
 * non-zero is returned.  */
int pocl_ipc_mutex_lock (pocl_ipc_mutex_t ipc_mtx);

/** Combined variant of pocl_ipc_mutex_create() and pocl_ipc_mutex_lock().  */
int pocl_ipc_mutex_create_and_lock (const char *name,
                                      pocl_ipc_mutex_t *ipc_mtx);

/** Releases the mutex.
 *
 * The behavior is undefined if the mutex is still locked.  */
void pocl_ipc_mutex_release (pocl_ipc_mutex_t *ipc_mtx);

/** Unlocks the mutex and then releases the mutex object
 *
 * The behavior is undefined if the mutex has not been locked.  */
void pocl_ipc_mutex_unlock_and_release (pocl_ipc_mutex_t *ipc_mtx);

#ifdef __cplusplus
}
#endif

#endif
