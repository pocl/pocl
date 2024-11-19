/* pocl_dbk_khr_onnxrt_cpu.c - ONNXRuntime Defined Built-in Kernels interfaces.

   Copyright (c) 2024 Jan Solanti <jan.solanti@tuni.fi>

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

#ifndef POCL_ONNXRT_H
#define POCL_ONNXRT_H

#include <stddef.h>

#include "pocl_cl.h"
#include "pocl_export.h"

typedef struct onnxrt_instance onnxrt_instance_t;

POCL_EXPORT
cl_int
pocl_create_ort_instance (const cl_dbk_attributes_onnx_inference_exp *attrs,
                          onnxrt_instance_t **onnxrt);

POCL_EXPORT
cl_int pocl_destroy_ort_instance (onnxrt_instance_t **onnxrt);

POCL_EXPORT
cl_int pocl_perform_ort_inference (onnxrt_instance_t *oi,
                                   const uint64_t *input_offsets,
                                   char *input_data,
                                   const uint64_t *output_offsets,
                                   char *output_storage);

#endif
