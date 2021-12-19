/* pocl_lttng.h: LTTNG tracepoints provider interface

   Copyright (c) 2015 Clément Léger / Kalray

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

#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER pocl_trace

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "./pocl_lttng.h"

#if !defined(POCL_LLTTNG_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define POCL_LLTTNG_H

#include <lttng/tracepoint.h>

/******************************************************************************/
/******************************************************************************/

/**
 *  NDRange kernel tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  ndrange_kernel,
  TP_ARGS(
    uint64_t, event_id,
    uint32_t, evt_status,
    uint32_t, dev_id,
    uint32_t, queue_id,
    uint32_t, kernel_id,
    const char*, kernel_name
  ),
  TP_FIELDS(
    ctf_integer_hex(uint64_t, event_id, event_id)
    ctf_integer_hex(uint32_t, evt_status, evt_status)
    ctf_integer_hex(uint32_t, dev_id, dev_id)
    ctf_integer_hex(uint32_t, queue_id, queue_id)
    ctf_integer_hex(uint32_t, kernel_id, kernel_id)
    ctf_string(kernel_name, kernel_name)
  )
)

/**
 *  R/W Buffer tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  read_buffer,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, obj_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, obj_id, obj_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  write_buffer,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, obj_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, obj_id, obj_id)
  )
)

/**
 *  Copy Buffer tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  copy_buffer,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, src_id,
      uint32_t, dst_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, src_id, src_id)
      ctf_integer_hex(uint32_t, dst_id, dst_id)
  )
)

/**
 *  Fill Buffer tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  fill_buffer,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, obj_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, obj_id, obj_id)
  )
)


/**
 *  R/W Buffer tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  read_buffer_rect,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, obj_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, obj_id, obj_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  write_buffer_rect,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, obj_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, obj_id, obj_id)
  )
)

/**
 *  Copy Buffer tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  copy_buffer_rect,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, src_id,
      uint32_t, dst_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, src_id, src_id)
      ctf_integer_hex(uint32_t, dst_id, dst_id)
  )
)




/**
 * IMAGEs
 */
TRACEPOINT_EVENT(
  pocl_trace,
  read_image_rect,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, obj_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, obj_id, obj_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  write_image_rect,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, obj_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, obj_id, obj_id)
  )
)


TRACEPOINT_EVENT(
  pocl_trace,
  copy_image_rect,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, src_id,
      uint32_t, dst_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, src_id, src_id)
      ctf_integer_hex(uint32_t, dst_id, dst_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  copy_image2buf,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, src_id,
      uint32_t, dst_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, src_id, src_id)
      ctf_integer_hex(uint32_t, dst_id, dst_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  copy_buf2image,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, src_id,
      uint32_t, dst_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, src_id, src_id)
      ctf_integer_hex(uint32_t, dst_id, dst_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  fill_image,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, obj_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, obj_id, obj_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  map_image,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, obj_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, obj_id, obj_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  map_buffer,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, obj_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, obj_id, obj_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  unmap_memobj,
  TP_ARGS(
      uint64_t, event_id,
      uint32_t, evt_status,
      uint32_t, dev_id,
      uint32_t, queue_id,
      uint32_t, obj_id
  ),
  TP_FIELDS(
      ctf_integer_hex(uint64_t, event_id, event_id)
      ctf_integer_hex(uint32_t, evt_status, evt_status)
      ctf_integer_hex(uint32_t, dev_id, dev_id)
      ctf_integer_hex(uint32_t, queue_id, queue_id)
      ctf_integer_hex(uint32_t, obj_id, obj_id)
  )
)

/**************************************************************************/
/**************************************************************************/

/**
 *  Create / Free Queue tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  create_queue,
  TP_ARGS(
    size_t, context_id,
    size_t, queue_id
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, queue_id, queue_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  free_queue,
  TP_ARGS(
    size_t, context_id,
    size_t, queue_id
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, queue_id, queue_id)
  )
)



/**
 *  Create / Free buffer tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  create_buffer,
  TP_ARGS(
    size_t, context_id,
    size_t, buffer_id
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, buffer_id, buffer_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  free_buffer,
  TP_ARGS(
    size_t, context_id,
    size_t, buffer_id
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, buffer_id, buffer_id)
  )
)




/**
 *  Create / Free program tracepoint
 */

TRACEPOINT_EVENT(
  pocl_trace,
  build_program,
  TP_ARGS(
    size_t, context_id,
    size_t, program_id
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, program_id, program_id)
  )
)


TRACEPOINT_EVENT(
  pocl_trace,
  create_program,
  TP_ARGS(
    size_t, context_id,
    size_t, program_id
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, program_id, program_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  free_program,
  TP_ARGS(
    size_t, context_id,
    size_t, program_id
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, program_id, program_id)
  )
)



/**
 *  Create / Free kernel tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  create_kernel,
  TP_ARGS(
    size_t, context_id,
    size_t, kernel_id,
    const char*, kernel_name
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, kernel_id, kernel_id)
      ctf_string(kernel_name, kernel_name)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  free_kernel,
  TP_ARGS(
    size_t, context_id,
    size_t, kernel_id,
    const char*, kernel_name
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, kernel_id, kernel_id)
      ctf_string(kernel_name, kernel_name)
  )
)


/**
 *  Create / Free image tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  create_image,
  TP_ARGS(
    size_t, context_id,
    size_t, image_id
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, image_id, image_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  free_image,
  TP_ARGS(
    size_t, context_id,
    size_t, image_id
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, image_id, image_id)
  )
)

/**
 *  Create / Free sampler tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  create_sampler,
  TP_ARGS(
    size_t, context_id,
    size_t, sampler_id
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, sampler_id, sampler_id)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  free_sampler,
  TP_ARGS(
    size_t, context_id,
    size_t, sampler_id
  ),
  TP_FIELDS(
      ctf_integer_hex(size_t, context_id, context_id)
      ctf_integer_hex(size_t, sampler_id, sampler_id)
  )
)

#endif /* POCL_LLTTNG_H */

#include <lttng/tracepoint-event.h>
