/* tracing.h - lttng tracepoint macros

   Copyright (c) 2018 Michal Babej / Tampere University of Technology

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

#include "pocld_config.h"

/* command execution status */
#ifndef CL_READ
#define CL_READ 0x1
#endif

#ifndef CL_RUNNING
#define CL_RUNNING 0x2
#endif

#ifndef CL_FINISHED
#define CL_FINISHED 0x3
#endif

#ifndef CL_WRITTEN
#define CL_WRITTEN 0x4
#endif

#ifdef HAVE_LTTNG_UST

#include "pocld_lttng.h"

#define TP_MSG_RECEIVED(msg_id, dev_id, queue_id, type)                       \
  tracepoint (pocld_trace, msg_received, msg_id, dev_id, queue_id, type);
#define TP_MSG_SENT(msg_id, dev_id, failed, type)                             \
  tracepoint (pocld_trace, msg_sent, msg_id, dev_id, failed, type);

#define TP_READ_BUFFER(msg_id, dev_id, queue_id, obj_id, size, stat)          \
  tracepoint (pocld_trace, read_buffer, msg_id, dev_id, queue_id, obj_id,     \
              size, stat);
#define TP_WRITE_BUFFER(msg_id, dev_id, queue_id, obj_id, size, stat)         \
  tracepoint (pocld_trace, write_buffer, msg_id, dev_id, queue_id, obj_id,    \
              size, stat);
#define TP_COPY_BUFFER(msg_id, dev_id, queue_id, src, dst, size, stat)        \
  tracepoint (pocld_trace, copy_buffer, msg_id, dev_id, queue_id, src, dst,   \
              size, stat);

#define TP_READ_BUFFER_RECT(msg_id, dev_id, queue_id, obj_id, x, y, z, stat)  \
  tracepoint (pocld_trace, read_buffer_rect, msg_id, dev_id, queue_id,        \
              obj_id, x, y, z, stat);
#define TP_WRITE_BUFFER_RECT(msg_id, dev_id, queue_id, obj_id, x, y, z, stat) \
  tracepoint (pocld_trace, write_buffer_rect, msg_id, dev_id, queue_id,       \
              obj_id, x, y, z, stat);
#define TP_COPY_BUFFER_RECT(msg_id, dev_id, queue_id, src, dst, x, y, z,      \
                            stat)                                             \
  tracepoint (pocld_trace, copy_buffer_rect, msg_id, dev_id, queue_id, src,   \
              dst, x, y, z, stat);

#define TP_FILL_BUFFER(msg_id, dev_id, queue_id, obj_id, size, stat)          \
  tracepoint (pocld_trace, fill_buffer, msg_id, dev_id, queue_id, obj_id,     \
              size, stat);
#define TP_NDRANGE_KERNEL(msg_id, dev_id, queue_id, kernel_id, stat)          \
  tracepoint (pocld_trace, ndrange_kernel, msg_id, dev_id, queue_id,          \
              kernel_id, stat);
#define TP_NDRANGE_SETUP(msg_id, dev_id, queue_id, kernel_id, stat)           \
  tracepoint (pocld_trace, ndrange_setup, msg_id, dev_id, queue_id,           \
              kernel_id, stat);

#define TP_READ_IMAGE_RECT(msg_id, dev_id, queue_id, obj_id, x, y, z, stat)   \
  tracepoint (pocld_trace, read_image_rect, msg_id, dev_id, queue_id, obj_id, \
              x, y, z, stat);
#define TP_WRITE_IMAGE_RECT(msg_id, dev_id, queue_id, obj_id, x, y, z, stat)  \
  tracepoint (pocld_trace, write_image_rect, msg_id, dev_id, queue_id,        \
              obj_id, x, y, z, stat);
#define TP_COPY_IMAGE_RECT(msg_id, dev_id, queue_id, src, dst, x, y, z, stat) \
  tracepoint (pocld_trace, copy_image_rect, msg_id, dev_id, queue_id, src,    \
              dst, x, y, z, stat);
#define TP_FILL_IMAGE(msg_id, dev_id, queue_id, obj_id, stat)                 \
  tracepoint (pocld_trace, fill_image, msg_id, dev_id, queue_id, obj_id, stat);

#define TP_CREATE_QUEUE(msg_id, dev_id, queue_id)                             \
  tracepoint (pocld_trace, create_queue, msg_id, dev_id, queue_id);
#define TP_FREE_QUEUE(msg_id, dev_id, queue_id)                               \
  tracepoint (pocld_trace, free_queue, msg_id, dev_id, queue_id);

#define TP_CREATE_BUFFER(msg_id, dev_id, buffer_id)                           \
  tracepoint (pocld_trace, create_buffer, msg_id, dev_id, buffer_id);
#define TP_FREE_BUFFER(msg_id, dev_id, buffer_id)                             \
  tracepoint (pocld_trace, free_buffer, msg_id, dev_id, buffer_id);

#define TP_BUILD_PROGRAM(msg_id, dev_id, program_id)                          \
  tracepoint (pocld_trace, build_program, msg_id, dev_id, program_id);
#define TP_FREE_PROGRAM(msg_id, dev_id, program_id)                           \
  tracepoint (pocld_trace, free_program, msg_id, dev_id, program_id);

#define TP_CREATE_KERNEL(msg_id, dev_id, kernel_id)                           \
  tracepoint (pocld_trace, create_kernel, msg_id, dev_id, kernel_id);
#define TP_FREE_KERNEL(msg_id, dev_id, kernel_id)                             \
  tracepoint (pocld_trace, free_kernel, msg_id, dev_id, kernel_id);

#define TP_CREATE_IMAGE(msg_id, dev_id, image_id)                             \
  tracepoint (pocld_trace, create_image, msg_id, dev_id, image_id);
#define TP_FREE_IMAGE(msg_id, dev_id, image_id)                               \
  tracepoint (pocld_trace, free_image, msg_id, dev_id, image_id);

#define TP_CREATE_SAMPLER(msg_id, dev_id, sampler_id)                         \
  tracepoint (pocld_trace, create_sampler, msg_id, dev_id, sampler_id);
#define TP_FREE_SAMPLER(msg_id, dev_id, sampler_id)                           \
  tracepoint (pocld_trace, free_sampler, msg_id, dev_id, sampler_id);

#else

#define TP_MSG_RECEIVED(msg_id, dev_id, queue_id, type)
#define TP_MSG_SENT(msg_id, dev_id, failed, type)

#define TP_READ_BUFFER(msg_id, dev_id, queue_id, obj_id, size, stat)
#define TP_WRITE_BUFFER(msg_id, dev_id, queue_id, obj_id, size, stat)
#define TP_COPY_BUFFER(msg_id, dev_id, queue_id, src, dst, size, stat)

#define TP_READ_BUFFER_RECT(msg_id, dev_id, queue_id, obj_id, x, y, z, stat)
#define TP_WRITE_BUFFER_RECT(msg_id, dev_id, queue_id, obj_id, x, y, z, stat)
#define TP_COPY_BUFFER_RECT(msg_id, dev_id, queue_id, src, dst, x, y, z, stat)

#define TP_FILL_BUFFER(msg_id, dev_id, queue_id, obj_id, size, stat)
#define TP_NDRANGE_KERNEL(msg_id, dev_id, queue_id, kernel_id, stat)
#define TP_NDRANGE_SETUP(msg_id, dev_id, queue_id, kernel_id, stat)

#define TP_READ_IMAGE_RECT(msg_id, dev_id, queue_id, obj_id, x, y, z, stat)
#define TP_WRITE_IMAGE_RECT(msg_id, dev_id, queue_id, obj_id, x, y, z, stat)
#define TP_COPY_IMAGE_RECT(msg_id, dev_id, queue_id, src, dst, x, y, z, stat)
#define TP_FILL_IMAGE(msg_id, dev_id, queue_id, obj_id, stat)

#define TP_CREATE_QUEUE(msg_id, dev_id, queue_id)
#define TP_FREE_QUEUE(msg_id, dev_id, queue_id)

#define TP_CREATE_BUFFER(msg_id, dev_id, buffer_id)
#define TP_FREE_BUFFER(msg_id, dev_id, buffer_id)

#define TP_BUILD_PROGRAM(msg_id, dev_id, program_id)
#define TP_FREE_PROGRAM(msg_id, dev_id, program_id)

#define TP_CREATE_KERNEL(msg_id, dev_id, kernel_id)
#define TP_FREE_KERNEL(msg_id, dev_id, kernel_id)

#define TP_CREATE_IMAGE(msg_id, dev_id, image_id)
#define TP_FREE_IMAGE(msg_id, dev_id, image_id)

#define TP_CREATE_SAMPLER(msg_id, dev_id, sampler_id)
#define TP_FREE_SAMPLER(msg_id, dev_id, sampler_id)

#endif
