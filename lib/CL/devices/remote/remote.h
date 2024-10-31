/* remote.h - a pocl device driver which controls remote devices

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

#ifndef POCL_REMOTE_H
#define POCL_REMOTE_H

#include "bufalloc.h"
#include "pocl_cl.h"
#include "pocl_icd.h"

#include "prototypes.inc"

typedef struct remote_server_data_s remote_server_data_t;
typedef struct remote_device_data_s
{
  remote_server_data_t *server;
  cl_image_format *supported_image_formats;
  char *build_hash;
  unsigned remote_device_index;
  unsigned remote_platform_index;
  unsigned local_did;

  /* SVM support: The SVM memory pool region in the remote device's
     memory. */
  size_t device_svm_region_start_addr;
  size_t device_svm_region_size;

  /* The difference between host and device SVM region starting
     addresses (device minus host start address). That is, this offset
     must be added to host SVM addresses to end up with a device SVM
     address and vice versa. The addition can wraparound, which is defined
     behavior with unsigned values in C. Ideally, this offset would
     be always zero to avoid address translation overheads, but
     it's difficult to guarantee without HW support, so generally we must
     be ready for a non-zero offset and deal with it. */
  size_t svm_region_offset;

  /* migrated -> ready to launch queue */
  _cl_command_node *work_queue;
  /* finished queue */
  _cl_command_node *finished_list;

  /* driver wake + lock */
  ALIGN_CACHE (pocl_lock_t wq_lock);
  ALIGN_CACHE (pocl_cond_t wakeup_cond);
  ALIGN_CACHE (pocl_lock_t mem_lock);

  /* device pthread */
  pocl_thread_t driver_thread_id;
  size_t driver_thread_exit_requested;

} remote_device_data_t;

GEN_PROTOTYPES (remote)

typedef struct peer_list_s peer_list_t;

peer_list_t *pocl_remote_get_peer_list (int base_id, unsigned device_count);

cl_int pocl_remote_setup_peer_mesh (struct pocl_device_ops *ops);

#endif /* POCL_REMOTE_H */
