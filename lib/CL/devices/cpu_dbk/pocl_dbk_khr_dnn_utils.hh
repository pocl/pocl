/* pocl_dbk_khr_dnn_utils.h - cpu implementation of neural network related DBKs.

   Copyright (c) 2024 Robin Bijl / Tampere University

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

#ifndef POCL_DBK_KHR_DNN_UTILS_H
#define POCL_DBK_KHR_DNN_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "pocl_cl.h"

POCL_EXPORT int
pocl_cpu_execute_dbk_khr_nms_box(cl_program program, cl_kernel kernel,
                                 pocl_kernel_metadata_t *meta, cl_uint dev_i,
                                 struct pocl_argument *arguments);

#ifdef __cplusplus
}
#endif

#endif // POCL_DBK_KHR_DNN_UTILS_H
