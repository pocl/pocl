/* pocl_rdma.h

   Copyright (c) 2021 Tampere University

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

#ifndef POCL_RDMA_H
#define POCL_RDMA_H

#include "pocl_cl.h"

#ifdef __cplusplus
#include <cstdint>
extern "C"
{
#else
#include <stdint.h>
#endif

#ifdef __GNUC__
#pragma GCC visibility push(hidden)
#endif

  // Avoid including rdma headers
  struct ibv_qp;
  struct ibv_mr;

  typedef struct rdma_data_s
  {
    struct rdma_event_channel *cm_channel;
    struct rdma_cm_id *cm_id;
    struct ibv_pd *p_domain;
    struct ibv_comp_channel *comp_channel_in;
    struct ibv_cq *cq_in;
    struct ibv_comp_channel *comp_channel_out;
    struct ibv_cq *cq_out;
  } rdma_data_t;

  int rdma_init_id (rdma_data_t *ctx);
  int rdma_init_cq (rdma_data_t *ctx);
  void rdma_uninitialize (rdma_data_t *ctx);

  void *rdma_get_cm_channel ();
  void *rdma_get_cm_ident ();

  void *rdma_register_mem_region (rdma_data_t *ctx, void *ptr,
                                  unsigned long size);
  int rdma_unregister_mem_region (void *mem_region);

  void rdma_setup_recv_request (rdma_data_t *ctx, void *ptr,
                                unsigned long size, uint64_t msg_id,
                                struct ibv_mr *mem_region);

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif

#endif
