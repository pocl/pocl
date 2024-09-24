/* pocl_dbk_khr_onnxrt_shared.h - Defined Built-in Kernels interfaces.

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
#ifndef _POCL_DBK_KHR_ONNXRT_SHARED_H_
#define _POCL_DBK_KHR_ONNXRT_SHARED_H_

#include <CL/cl_exp_defined_builtin_kernels.h>
#include <CL/cl_platform.h>
#include <stddef.h>

#include "pocl_export.h"

POCL_EXPORT
    cl_dbk_attributes_khr_onnx_inference *pocl_copy_onnx_inference_dbk_attributes (
    const cl_dbk_attributes_khr_onnx_inference *src);

POCL_EXPORT
void pocl_release_onnx_inference_dbk_attributes (
    cl_dbk_attributes_khr_onnx_inference *attrs);

#endif //_POCL_DBK_KHR_ONNXRT_SHARED_H_
