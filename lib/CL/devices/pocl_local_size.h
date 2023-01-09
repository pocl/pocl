/* pocl_local_size.c - Different means for optimizing the local size.

   Copyright (c) 2011-2019 pocl developers
                 2020 Pekka Jääskeläinen

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
#ifndef POCL_LOCAL_SIZE_H
#define POCL_LOCAL_SIZE_H

#include "pocl_cl.h"
/* The generic local size optimizer used by default, in case there's no target
 * specific one defined in the device driver. */
POCL_EXPORT
void pocl_default_local_size_optimizer (cl_device_id dev, cl_kernel kernel,
                                        unsigned device_i,
                                        size_t global_x, size_t global_y,
                                        size_t global_z, size_t *local_x,
                                        size_t *local_y, size_t *local_z);

/* Can be used for devices which support only small work-groups and prefer
 * them to be maximally utilized to use as many of the SIMD lanes as possible.
 * High compute unit utilization is only a secondary goal which typicaly
 * results as a side effect from the small work-groups. Performs an exhaustive
 * search, thus should not be used with devices with a large work-group
 * support. */
POCL_EXPORT
void pocl_wg_utilization_maximizer (cl_device_id dev, cl_kernel kernel,
                                    unsigned device_i,
                                    size_t global_x, size_t global_y,
                                    size_t global_z, size_t *local_x,
                                    size_t *local_y, size_t *local_z);
#endif
