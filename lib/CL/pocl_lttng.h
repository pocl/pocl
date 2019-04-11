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

/**
 *  NDRange kernel tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  ndrange_kernel,
  TP_ARGS(
    unsigned long, event_id,
    int, event_status,
    const char*, kernel_name
  ),
  TP_FIELDS(
    ctf_integer(unsigned long, event_id, event_id)
    ctf_integer(int, event_status, event_status)
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
    unsigned long, event_id,
    int, event_status,
    const void *, host_ptr,
    size_t, buffer_size
  ),
  TP_FIELDS(
    ctf_integer(unsigned long, event_id, event_id)
    ctf_integer(int, event_status, event_status)
    ctf_integer_hex(const void *, host_ptr, host_ptr)
    ctf_integer_hex(size_t, buffer_size, buffer_size)
  )
)

TRACEPOINT_EVENT(
  pocl_trace,
  write_buffer,
  TP_ARGS(
    unsigned long, event_id,
    int, event_status,
    const void *, host_ptr,
    size_t, buffer_size
  ),
  TP_FIELDS(
    ctf_integer(unsigned long, event_id, event_id)
    ctf_integer(int, event_status, event_status)
    ctf_integer_hex(const void *, host_ptr, host_ptr)
    ctf_integer_hex(size_t, buffer_size, buffer_size)
  )
)

/**
 *  Copy Buffer tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  copy_buffer,
  TP_ARGS(
    unsigned long, event_id,
    int, event_status,
    size_t, buffer_size
  ),
  TP_FIELDS(
    ctf_integer(unsigned long, event_id, event_id)
    ctf_integer(int, event_status, event_status)
    ctf_integer_hex(size_t, buffer_size, buffer_size)
  )
)

/**
 *  Fill Buffer tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  fill_buffer,
  TP_ARGS(
    unsigned long, event_id,
    int, event_status,
    size_t, fill_size
  ),
  TP_FIELDS(
      ctf_integer(unsigned long, event_id, event_id)
    ctf_integer(int, event_status, event_status)
    ctf_integer_hex(size_t, fill_size, fill_size)
  )
)

/**
 *  Map tracepoint
 */
TRACEPOINT_EVENT(
  pocl_trace,
  map,
  TP_ARGS(
    unsigned long, event_id,
    int, event_status,
    size_t, fill_size
  ),
  TP_FIELDS(
    ctf_integer(unsigned long, event_id, event_id)
    ctf_integer(int, event_status, event_status)
    ctf_integer_hex(size_t, fill_size, fill_size)
  )
)

/**
 *  Generic tracepoint for other commands
 */
TRACEPOINT_EVENT(
  pocl_trace,
  command,
  TP_ARGS(
    unsigned long, event_id,
    int, event_status,
    const char *, cmd_type
  ),
  TP_FIELDS(
    ctf_integer(unsigned long, event_id, event_id)
    ctf_integer(int, event_status, event_status)
    ctf_string(cmd_type, cmd_type)
  )
)

/**
 *  Default tracepoint
 */

#endif /* POCL_LLTTNG_H */

#include <lttng/tracepoint-event.h>
