/* pocl_rdma.c

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

#include "pocl_rdma.h"
#include "messages.h"

#include <stdio.h>

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

int
rdma_init_id (rdma_data_t *ctx)
{
  struct rdma_cm_id *cm_id = NULL;
  POCL_MSG_PRINT_INFO ("Creating CM event channel\n");
  struct rdma_event_channel *cm_channel = rdma_create_event_channel ();
  if (!cm_channel)
    {
      goto ERR;
    }

  POCL_MSG_PRINT_INFO ("Creating CM identifier\n");
  int error = rdma_create_id (cm_channel, &cm_id, NULL, RDMA_PS_TCP);
  if (error)
    {
      POCL_MSG_ERR ("Create CM identifier failed -> %s \n", strerror (errno));
      goto ERR;
    }

  ctx->cm_channel = cm_channel;
  ctx->cm_id = cm_id;

  return 0;

ERR:

  if (cm_id)
    rdma_destroy_id (cm_id);
  if (cm_channel)
    rdma_destroy_event_channel (cm_channel);

  return -1;
}

int
rdma_init_cq (rdma_data_t *ctx)
{

  rdma_data_t tmp;
  memset (&tmp, 0, sizeof (rdma_data_t));

  POCL_MSG_PRINT_INFO ("Allocating protection domain\n");
  tmp.p_domain = ibv_alloc_pd (ctx->cm_id->verbs);
  if (!tmp.p_domain)
    {
      POCL_MSG_ERR ("Allocate protection domain failed\n");
      goto ERR;
    }

  POCL_MSG_PRINT_INFO ("Creating in/out complete channels and queues\n");
  tmp.comp_channel_out = ibv_create_comp_channel (ctx->cm_id->verbs);
  if (!tmp.comp_channel_out)
    {
      POCL_MSG_ERR ("Create out complete channel failed\n");
      goto ERR;
    }
  tmp.comp_channel_in = ibv_create_comp_channel (ctx->cm_id->verbs);
  if (!tmp.comp_channel_in)
    {
      POCL_MSG_ERR ("Create in complete channel failed\n");
      goto ERR;
    }

  // TODO: This value needs to be tweaked
  int cq_capacity = 10;

  tmp.cq_out = ibv_create_cq (ctx->cm_id->verbs, cq_capacity, NULL,
                              tmp.comp_channel_out, 0);
  if (!tmp.cq_out)
    {
      POCL_MSG_ERR ("Create complete queue (out) failed\n");
      goto ERR;
    }
  if (ibv_req_notify_cq (tmp.cq_out, 0))
    {
      POCL_MSG_ERR ("Set complete queue (out) notify failed\n");
      goto ERR;
    }

  tmp.cq_in = ibv_create_cq (ctx->cm_id->verbs, cq_capacity, NULL,
                             tmp.comp_channel_in, 0);
  if (!tmp.cq_in)
    {
      POCL_MSG_ERR ("Create complete queue (in) failed\n");
      goto ERR;
    }
  if (ibv_req_notify_cq (tmp.cq_in, 0))
    {
      POCL_MSG_ERR ("Set complete queue (in) notify failed\n");
      goto ERR;
    }

  // POCL_MSG_PRINT_INFO ("Creating queue pair\n");
  //  TODO: These values need to be tweaked
  struct ibv_qp_init_attr qp_attr = {};
  qp_attr.cap.max_send_wr = 10;
  qp_attr.cap.max_send_sge = 10;
  qp_attr.cap.max_recv_wr = 10;
  qp_attr.cap.max_recv_sge = 10;

  qp_attr.send_cq = tmp.cq_out;
  qp_attr.recv_cq = tmp.cq_in;
  qp_attr.qp_type = IBV_QPT_RC;

  int error = rdma_create_qp (ctx->cm_id, tmp.p_domain, &qp_attr);
  if (error)
    {
      POCL_MSG_ERR ("Create queue pair failed: %s\n", strerror (errno));
      goto ERR;
    }

  ctx->p_domain = tmp.p_domain;
  ctx->comp_channel_out = tmp.comp_channel_out;
  ctx->cq_out = tmp.cq_out;
  ctx->comp_channel_in = tmp.comp_channel_in;
  ctx->cq_in = tmp.cq_in;

  return 0;

ERR:
  rdma_uninitialize (&tmp);

  return -1;
}

void
rdma_uninitialize (rdma_data_t *ctx)
{
  // There is no separate QP instance, instead it is bound to the identifier.
  // Regardless, it presumably should also be destroyed at this time if one
  // exists.
  // TODO: We assume that if the CQs exist, then a QP also exists, but is
  // this always the case?

  if (ctx->cq_in && ctx->cq_out && ctx->cm_id)
    rdma_destroy_qp (ctx->cm_id);
  if (ctx->cq_in)
    ibv_destroy_cq (ctx->cq_in);
  if (ctx->cq_out)
    ibv_destroy_cq (ctx->cq_out);
  if (ctx->comp_channel_in)
    ibv_destroy_comp_channel (ctx->comp_channel_in);
  if (ctx->comp_channel_out)
    ibv_destroy_comp_channel (ctx->comp_channel_out);
  if (ctx->p_domain)
    ibv_dealloc_pd (ctx->p_domain);
  if (ctx->cm_id)
    {
      rdma_disconnect (ctx->cm_id);
      rdma_destroy_id (ctx->cm_id);
    }
  if (ctx->cm_channel)
    rdma_destroy_event_channel (ctx->cm_channel);
  memset (ctx, 0, sizeof (rdma_data_t));
}

void *
rdma_register_mem_region (rdma_data_t *ctx, void *ptr, unsigned long size)
{
  // TODO The maximum size of the block that can be registered is limited to
  // device_attr.max_mr_size
  POCL_MSG_PRINT_REMOTE ("Registering memory region\n");
  struct ibv_mr *mem_region
      = ibv_reg_mr (ctx->p_domain, ptr, size,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ
                        | IBV_ACCESS_REMOTE_WRITE);
  if (!mem_region)
    {
      int e = errno;
      POCL_MSG_ERR ("Register memory region failed: %d = %s\n", e,
                    strerror (e));
      return NULL;
    }
  return mem_region;
}

int
rdma_unregister_mem_region (void *mem_region)
{
  POCL_MSG_PRINT_REMOTE ("UnRegistering memory region\n");
  int err = ibv_dereg_mr (mem_region);
  if (err)
    {
      POCL_MSG_ERR ("UnRegister memory region failed\n");
      return -1;
    }
  return 0;
}

void
rdma_setup_recv_request (rdma_data_t *ctx, void *ptr, unsigned long size,
                         uint64_t msg_id, struct ibv_mr *mem_region)
{
  if (!mem_region)
    {
      mem_region = rdma_register_mem_region (ctx, ptr, size);
    }

  struct ibv_sge sge = {};
  sge.addr = (uintptr_t)mem_region->addr;
  sge.length = size;
  sge.lkey = mem_region->lkey;

  struct ibv_recv_wr recv_wr = {};
  recv_wr.wr_id = msg_id;
  recv_wr.sg_list = &sge;
  recv_wr.num_sge = 1;

  struct ibv_recv_wr *bad_recv_wr;
  if (ibv_post_recv (ctx->cm_id->qp, &recv_wr, &bad_recv_wr))
    {
      POCL_MSG_ERR ("Receive request was invalid");
    }
}
