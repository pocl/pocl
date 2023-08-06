/* pocl_lttng.h: LTTNG tracepoints provider interface

   Copyright (c) 2015 Clément Léger / Kalray

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

#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER pocld_trace

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "pocld_lttng.h"

#if !defined(POCL_LLTTNG_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define POCL_LLTTNG_H

#include <lttng/tracepoint.h>

/******************************************************************************
*******************************************************************************
******************************************************************************/

/**
 *  MSG received event
 */
TRACEPOINT_EVENT (pocld_trace, msg_received,
                  TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t,
                           queue_id, int, type),
                  TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                                 ctf_integer_hex (uint32_t, dev_id, dev_id)
                                     ctf_integer_hex (uint32_t, queue_id,
                                                      queue_id)
                                         ctf_integer (int, type, type)))

/**
 *  MSG sent event
 */
TRACEPOINT_EVENT (pocld_trace, msg_sent,
                  TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t,
                           queue_id, int, type),
                  TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                                 ctf_integer_hex (uint32_t, dev_id, dev_id)
                                     ctf_integer_hex (uint32_t, queue_id,
                                                      queue_id)
                                         ctf_integer (int, type, type)))

/******************************************************************************
*******************************************************************************
******************************************************************************/

/**
 *  NDRange kernel tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, ndrange_kernel,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, uint32_t,
             kernel_id, int, event_status),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)
                           ctf_integer_hex (uint32_t, kernel_id, kernel_id)
                               ctf_integer (int, event_status, event_status)))

// kernel arg setup
TRACEPOINT_EVENT (
    pocld_trace, ndrange_setup,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, uint32_t,
             kernel_id, int, event_status),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)
                           ctf_integer_hex (uint32_t, kernel_id, kernel_id)
                               ctf_integer (int, event_status, event_status)))

/**
 *  Fill Buffer tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, fill_buffer,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, uint32_t,
             buffer_id, size_t, size, int, event_status),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)
                           ctf_integer_hex (uint32_t, buffer_id, buffer_id)
                               ctf_integer_hex (size_t, size, size)
                                   ctf_integer (int, event_status,
                                                event_status)))

/******************************************************************************
*******************************************************************************
******************************************************************************/

/**
 *  R/W Buffer tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, read_buffer,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, uint32_t,
             buffer_id, size_t, read_size, int, event_status),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)
                           ctf_integer_hex (uint32_t, buffer_id, buffer_id)
                               ctf_integer_hex (size_t, read_size, read_size)
                                   ctf_integer (int, event_status,
                                                event_status)))

TRACEPOINT_EVENT (
    pocld_trace, write_buffer,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, uint32_t,
             buffer_id, size_t, write_size, int, event_status),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)
                           ctf_integer_hex (uint32_t, buffer_id, buffer_id)
                               ctf_integer_hex (size_t, write_size, write_size)
                                   ctf_integer (int, event_status,
                                                event_status)))

/**
 *  Copy Buffer tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, copy_buffer,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, uint32_t,
             src_buffer_id, uint32_t, dst_buffer_id, size_t, copy_size, int,
             event_status),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id) ctf_integer_hex (
        uint32_t, dev_id, dev_id) ctf_integer_hex (uint32_t, queue_id,
                                                   queue_id)
                   ctf_integer_hex (uint32_t, src_buffer_id, src_buffer_id)
                       ctf_integer_hex (uint32_t, dst_buffer_id, dst_buffer_id)
                           ctf_integer_hex (size_t, copy_size, copy_size)
                               ctf_integer (int, event_status, event_status)))

/******************************************************************************
*******************************************************************************
******************************************************************************/

/**
 *  R/W Buffer tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, read_buffer_rect,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, uint32_t,
             buffer_id, size_t, x, size_t, y, size_t, z, int, event_status),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)
                           ctf_integer_hex (uint32_t, buffer_id, buffer_id)
                               ctf_integer_hex (size_t, x, x)
                                   ctf_integer_hex (size_t, y, y)
                                       ctf_integer_hex (size_t, z, z)
                                           ctf_integer (int, event_status,
                                                        event_status)))

TRACEPOINT_EVENT (
    pocld_trace, write_buffer_rect,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, uint32_t,
             buffer_id, size_t, x, size_t, y, size_t, z, int, event_status),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)
                           ctf_integer_hex (uint32_t, buffer_id, buffer_id)
                               ctf_integer_hex (size_t, x, x)
                                   ctf_integer_hex (size_t, y, y)
                                       ctf_integer_hex (size_t, z, z)
                                           ctf_integer (int, event_status,
                                                        event_status)))

/**
 *  Copy Buffer tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, copy_buffer_rect,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, uint32_t,
             src_buffer_id, uint32_t, dst_buffer_id, size_t, x, size_t, y,
             size_t, z, int, event_status),
    TP_FIELDS (
        ctf_integer_hex (uint64_t, msg_id,
                         msg_id) ctf_integer_hex (uint32_t, dev_id, dev_id)
            ctf_integer_hex (uint32_t, queue_id, queue_id)
                ctf_integer_hex (uint32_t, src_buffer_id, src_buffer_id)
                    ctf_integer_hex (uint32_t, dst_buffer_id, dst_buffer_id)
                        ctf_integer_hex (size_t, x, x) ctf_integer_hex (size_t,
                                                                        y, y)
                            ctf_integer_hex (size_t, z, z)
                                ctf_integer (int, event_status, event_status)))

/******************************************************************************
*******************************************************************************
******************************************************************************/

/**
 *  Map tracepoint
 */
/*
TRACEPOINT_EVENT(
  pocld_trace,
  map,
  TP_ARGS(
    int, event_status,
    uint32_t, buffer_id,
  ),
  TP_FIELDS(
    ctf_integer(int, event_status, event_status)
    ctf_integer_hex(uint32_t, buffer_id, buffer_id)
  )
)
*/

/**
 *  Map tracepoint
 */
/*
TRACEPOINT_EVENT(
  pocld_trace,
  unmap,
  TP_ARGS(
    int, event_status,
    uint32_t, buffer_id,
  ),
  TP_FIELDS(
    ctf_integer(int, event_status, event_status)
    ctf_integer_hex(uint32_t, buffer_id, buffer_id)
  )
)
*/

/******************************************************************************
*******************************************************************************
******************************************************************************/

TRACEPOINT_EVENT (
    pocld_trace, fill_image,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, uint32_t,
             image_id, int, event_status),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)
                           ctf_integer_hex (uint32_t, image_id, image_id)
                               ctf_integer (int, event_status, event_status)))

/**
 *  R/W image tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, read_image_rect,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, uint32_t,
             image_id, size_t, x, size_t, y, size_t, z, int, event_status),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)
                           ctf_integer_hex (uint32_t, image_id, image_id)
                               ctf_integer_hex (size_t, x, x)
                                   ctf_integer_hex (size_t, y, y)
                                       ctf_integer_hex (size_t, z, z)
                                           ctf_integer (int, event_status,
                                                        event_status)))

TRACEPOINT_EVENT (
    pocld_trace, write_image_rect,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, uint32_t,
             image_id, size_t, x, size_t, y, size_t, z, int, event_status),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)
                           ctf_integer_hex (uint32_t, image_id, image_id)
                               ctf_integer_hex (size_t, x, x)
                                   ctf_integer_hex (size_t, y, y)
                                       ctf_integer_hex (size_t, z, z)
                                           ctf_integer (int, event_status,
                                                        event_status)))

/**
 *  Copy image tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, copy_image_rect,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, uint32_t, queue_id, size_t,
             src_image_id, size_t, dst_image_id, size_t, x, size_t, y, size_t,
             z, int, event_status),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)
                           ctf_integer_hex (size_t, src_image_id, src_image_id)
                               ctf_integer_hex (size_t, dst_image_id,
                                                dst_image_id)
                                   ctf_integer_hex (size_t, x, x)
                                       ctf_integer_hex (size_t, y, y)
                                           ctf_integer_hex (size_t, z, z)
                                               ctf_integer (int, event_status,
                                                            event_status)))

/******************************************************************************
*******************************************************************************
******************************************************************************/

/**
 *  Create / Free Queue tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, create_queue,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, size_t, queue_id),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)))

TRACEPOINT_EVENT (
    pocld_trace, free_queue,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, size_t, queue_id),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, queue_id, queue_id)))

/**
 *  Create / Free buffer tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, create_buffer,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, size_t, buffer_id),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, buffer_id, buffer_id)))

TRACEPOINT_EVENT (
    pocld_trace, free_buffer,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, size_t, buffer_id),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, buffer_id, buffer_id)))

/**
 *  Create / Free program tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, build_program,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, size_t, program_id),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (size_t, program_id, program_id)))

TRACEPOINT_EVENT (
    pocld_trace, free_program,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, size_t, program_id),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (size_t, program_id, program_id)))

/**
 *  Create / Free kernel tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, create_kernel,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, size_t, kernel_id),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, kernel_id, kernel_id)))

TRACEPOINT_EVENT (
    pocld_trace, free_kernel,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, size_t, kernel_id),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, kernel_id, kernel_id)))

/**
 *  Create / Free image tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, create_image,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, size_t, image_id),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, image_id, image_id)))

TRACEPOINT_EVENT (
    pocld_trace, free_image,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, size_t, image_id),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, image_id, image_id)))

/**
 *  Create / Free sampler tracepoint
 */
TRACEPOINT_EVENT (
    pocld_trace, create_sampler,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, size_t, sampler_id),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, sampler_id, sampler_id)))

TRACEPOINT_EVENT (
    pocld_trace, free_sampler,
    TP_ARGS (uint64_t, msg_id, uint32_t, dev_id, size_t, sampler_id),
    TP_FIELDS (ctf_integer_hex (uint64_t, msg_id, msg_id)
                   ctf_integer_hex (uint32_t, dev_id, dev_id)
                       ctf_integer_hex (uint32_t, sampler_id, sampler_id)))

#endif /* POCL_LLTTNG_H */

#include <lttng/tracepoint-event.h>
