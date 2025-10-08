/* pocl_dbk_util.h - utility functions for DBK descriptions.

   Copyright (c) 2025 Henry Linjamäki / Intel Finland Oy

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

#ifndef POCL_DBK_UTILS_H
#define POCL_DBK_UTILS_H

#include "pocl_cl.h"

#ifdef __cplusplus
extern "C"
{
#endif

POCL_EXPORT
cl_int pocl_dbk_unpack_bin_operands (cl_dbk_id_exp kernel_id,
                                     const void *kernel_attributes,
                                     const cl_tensor_desc_exp *ops_out[3]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* POCL_DBK_UTILS_H */
