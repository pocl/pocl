/* pocl_dkb_khr_dnn_utils_shared.h - collection of neural network util DBKs.

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

#ifndef POCL_DBK_KHR_DNN_UTILS_SHARED_H
#define POCL_DBK_KHR_DNN_UTILS_SHARED_H

#include "pocl_builtin_kernels.h"

int pocl_validate_dnn_utils_attrs (cl_dbk_id_exp kernel_id,
                                   const void *kernel_attributes);

int pocl_release_dnn_utils_attrs (cl_dbk_id_exp kernel_id,
                                  void *kernel_attributes);

void *pocl_copy_dnn_utils_attrs (cl_dbk_id_exp kernel_id,
                                 const void *kernel_attributes);

#endif // POCL_DBK_KHR_DNN_UTILS_SHARED_H
