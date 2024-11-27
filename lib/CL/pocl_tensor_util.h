/* pocl_tensor_util.h - Tensor related utilities

   Copyright (c) 2024 Michal Babej / Intel Finland Oy

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

#include "pocl_cl.h"

int pocl_check_tensor_layout (cl_uint rank,
                              const cl_tensor_shape_exp *shape,
                              cl_tensor_layout_type_exp layout_type,
                              const void *layout);

int pocl_check_tensor_desc (const cl_tensor_desc_exp *tdesc);

int pocl_copy_tensor_desc2mem (cl_mem mem, const cl_tensor_desc_exp *tdesc);

int pocl_copy_tensor_desc_layout (cl_tensor_desc_exp *dest,
                                  const cl_tensor_desc_exp *src);

POCL_EXPORT
int pocl_tensor_type_is_int (cl_tensor_datatype_exp dtype);

POCL_EXPORT
int pocl_tensor_type_size (cl_tensor_datatype_exp dtype);

size_t pocl_tensor_data_size (const cl_tensor_desc_exp *t);

cl_bool
pocl_tensor_dtype_value_equals (const cl_tensor_datatype_exp dtype,
                                const cl_tensor_datatype_value_exp *value,
                                cl_double double_const,
                                cl_long long_const,
                                cl_ulong ulong_const,
                                char fp8_const,
                                char int4_const);
