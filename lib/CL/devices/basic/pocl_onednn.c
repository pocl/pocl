
#include "pocl_util.h"

#include "dnnl.h"

#include "onednn_utils.h"

#include "pocl_onednn.h"

//#define REORDER_INPUT

typedef struct
{
  int nargs;
  dnnl_exec_arg_t *args;
} args_t;

static void
prepare_arg_node (args_t *node, int nargs)
{
  node->args = (dnnl_exec_arg_t *)malloc (sizeof (dnnl_exec_arg_t) * nargs);
  node->nargs = nargs;
}
static void
free_arg_node (args_t *node)
{
  free (node->args);
}

static void
set_arg (dnnl_exec_arg_t *arg, int arg_idx, dnnl_memory_t memory)
{
  arg->arg = arg_idx;
  arg->memory = memory;
}

static void
init_data_memory (uint32_t dim, const dnnl_dim_t *dims,
                  dnnl_format_tag_t user_tag, dnnl_engine_t engine,
                  const float *data, dnnl_memory_t *memory)
{
  dnnl_memory_desc_t user_md;
  CHECK (
      dnnl_memory_desc_init_by_tag (&user_md, dim, dims, dnnl_f32, user_tag));
  CHECK (dnnl_memory_create (memory, &user_md, engine, DNNL_MEMORY_ALLOCATE));
  write_to_dnnl_memory (data, *memory);
}

dnnl_status_t
prepare_reorder (
    dnnl_memory_t *user_memory,               // in
    const dnnl_memory_desc_t *prim_memory_md, // in
    dnnl_engine_t prim_engine,                // in: primitive's engine
    int dir_is_user_to_prim,    // in: user -> prim or prim -> user
    dnnl_memory_t *prim_memory, // out: primitive's memory created
    dnnl_primitive_t *reorder,  // out: reorder primitive created
    uint32_t *net_index, // primitive index in net (inc if reorder created)
    dnnl_primitive_t *net, args_t *net_args)
{ // net params
  const dnnl_memory_desc_t *user_memory_md;
  dnnl_memory_get_memory_desc (*user_memory, &user_memory_md);

  dnnl_engine_t user_mem_engine;
  dnnl_memory_get_engine (*user_memory, &user_mem_engine);

  if (!dnnl_memory_desc_equal (user_memory_md, prim_memory_md))
    {
      CHECK (dnnl_memory_create (prim_memory, prim_memory_md, prim_engine,
                                 DNNL_MEMORY_ALLOCATE));

      dnnl_primitive_desc_t reorder_pd;
      if (dir_is_user_to_prim)
        {
          CHECK (dnnl_reorder_primitive_desc_create (
              &reorder_pd, user_memory_md, user_mem_engine, prim_memory_md,
              prim_engine, NULL));
        }
      else
        {
          CHECK (dnnl_reorder_primitive_desc_create (
              &reorder_pd, prim_memory_md, prim_engine, user_memory_md,
              user_mem_engine, NULL));
        }
      CHECK (dnnl_primitive_create (reorder, reorder_pd));
      CHECK (dnnl_primitive_desc_destroy (reorder_pd));

      net[*net_index] = *reorder;
      prepare_arg_node (&net_args[*net_index], 2);
      set_arg (&net_args[*net_index].args[0], DNNL_ARG_FROM,
               dir_is_user_to_prim ? *user_memory : *prim_memory);
      set_arg (&net_args[*net_index].args[1], DNNL_ARG_TO,
               dir_is_user_to_prim ? *prim_memory : *user_memory);
      (*net_index)++;
    }
  else
    {
      *prim_memory = NULL;
      *reorder = NULL;
    }

  return dnnl_success;
}

void
conv_onecnn (const float *inputData, const float *filterData,
             float *outputData, float alpha, float beta, const int *inDims,
             const int *filDims, const int *outDims, const int *inStride,
             const int *outStride, const int *stride, const int *pad,
             const int *dilation)
{
  dnnl_dims_t src_dims = { inDims[0], inDims[1], inDims[2], inDims[3] };
  dnnl_dims_t weights_dims
      = { filDims[0], filDims[1], filDims[2], filDims[3] };
  dnnl_dims_t dst_dims = { outDims[0], outDims[1], outDims[2], outDims[3] };

  dnnl_dims_t conv_strides = { stride[0], stride[1] };
  dnnl_dims_t conv_padding = { pad[0], pad[1] };
  dnnl_dims_t conv_dilates = { dilation[0] - 1, dilation[1] - 1 };

  uint32_t n = 0;
  dnnl_primitive_t net[10];
  args_t net_args[10];

  dnnl_engine_t engine;
  dnnl_engine_kind_t kind;
  if (dnnl_engine_get_count (dnnl_gpu))
    {
      POCL_MSG_PRINT_INFO ("OneDNN running with GPU\n");
      kind = dnnl_gpu;
    }
  else
    {
      POCL_MSG_PRINT_INFO ("OneDNN running with CPU\n");
      kind = dnnl_cpu;
    }
  CHECK (dnnl_engine_create (&engine, kind, 0));

  dnnl_memory_t conv_user_src_memory, conv_user_weights_memory;
  init_data_memory (4, src_dims, dnnl_nchw, engine, inputData,
                    &conv_user_src_memory);
  init_data_memory (4, weights_dims, dnnl_oihw, engine, filterData,
                    &conv_user_weights_memory);

  dnnl_memory_desc_t conv_src_md, conv_weights_md, conv_dst_md;
  CHECK (dnnl_memory_desc_init_by_tag (&conv_src_md, 4, src_dims, dnnl_f32,
                                       dnnl_format_tag_any));
  CHECK (dnnl_memory_desc_init_by_tag (&conv_weights_md, 4, weights_dims,
                                       dnnl_f32, dnnl_format_tag_any));
  CHECK (dnnl_memory_desc_init_by_tag (&conv_dst_md, 4, dst_dims, dnnl_f32,
                                       dnnl_nchw));

  dnnl_convolution_desc_t conv_any_desc;
  CHECK (dnnl_dilated_convolution_forward_desc_init (
      &conv_any_desc, dnnl_forward, dnnl_convolution_direct, &conv_src_md,
      &conv_weights_md, NULL, &conv_dst_md, conv_strides, conv_dilates,
      conv_padding, conv_padding));

  dnnl_primitive_desc_t conv_pd;
  CHECK (dnnl_primitive_desc_create (&conv_pd, &conv_any_desc, NULL, engine,
                                     NULL));

  dnnl_memory_t conv_internal_src_memory, conv_internal_weights_memory,
      conv_internal_dst_memory;

  const dnnl_memory_desc_t *dst_md
      = dnnl_primitive_desc_query_md (conv_pd, dnnl_query_dst_md, 0);
  CHECK (dnnl_memory_create (&conv_internal_dst_memory, dst_md, engine,
                             DNNL_MEMORY_ALLOCATE));

#ifdef REORDER_INPUT
  // create reorder primitives between user data and convolution srcs
  // if required
  dnnl_primitive_t conv_reorder_src, conv_reorder_weights;

  const dnnl_memory_desc_t *src_md
      = dnnl_primitive_desc_query_md (conv_pd, dnnl_query_src_md, 0);
  CHECK (prepare_reorder (&conv_user_src_memory, src_md, engine, 1,
                          &conv_internal_src_memory, &conv_reorder_src, &n,
                          net, net_args));

  const dnnl_memory_desc_t *weights_md
      = dnnl_primitive_desc_query_md (conv_pd, dnnl_query_weights_md, 0);
  CHECK (prepare_reorder (&conv_user_weights_memory, weights_md, engine, 1,
                          &conv_internal_weights_memory, &conv_reorder_weights,
                          &n, net, net_args));

  dnnl_memory_t conv_src_memory = conv_internal_src_memory
                                      ? conv_internal_src_memory
                                      : conv_user_src_memory;
  dnnl_memory_t conv_weights_memory = conv_internal_weights_memory
                                          ? conv_internal_weights_memory
                                          : conv_user_weights_memory;
#else
  dnnl_memory_t conv_src_memory = conv_user_src_memory;
  dnnl_memory_t conv_weights_memory = conv_user_weights_memory;
#endif

  // finally create a convolution primitive
  dnnl_primitive_t conv;
  CHECK (dnnl_primitive_create (&conv, conv_pd));
  net[n] = conv;

  prepare_arg_node (&net_args[n], 3);
  set_arg (&net_args[n].args[0], DNNL_ARG_SRC, conv_src_memory);
  set_arg (&net_args[n].args[1], DNNL_ARG_WEIGHTS, conv_weights_memory);
  set_arg (&net_args[n].args[2], DNNL_ARG_DST, conv_internal_dst_memory);
  n++;

  dnnl_stream_t stream;
  CHECK (dnnl_stream_create (&stream, engine, dnnl_stream_default_flags));
  for (uint32_t i = 0; i < n; ++i)
    {
      CHECK (dnnl_primitive_execute (net[i], stream, net_args[i].nargs,
                                     net_args[i].args));
    }
  CHECK (dnnl_stream_wait (stream));

  read_from_dnnl_memory (outputData, conv_internal_dst_memory);

  for (uint32_t i = 0; i < n; ++i)
    {
      free_arg_node (&net_args[i]);
    }
  CHECK (dnnl_primitive_desc_destroy (conv_pd));
  dnnl_stream_destroy (stream);

  dnnl_memory_destroy (conv_user_src_memory);
  dnnl_memory_destroy (conv_user_weights_memory);
  dnnl_memory_destroy (conv_internal_src_memory);
  dnnl_memory_destroy (conv_internal_weights_memory);
  dnnl_memory_destroy (conv_internal_dst_memory);
#ifdef REORDER_INPUT
  dnnl_primitive_destroy (conv_reorder_src);
  dnnl_primitive_destroy (conv_reorder_weights);
#endif
  dnnl_primitive_destroy (conv);

  dnnl_engine_destroy (engine);
}