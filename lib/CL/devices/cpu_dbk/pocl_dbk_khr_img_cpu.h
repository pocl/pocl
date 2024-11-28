/* pocl_dbk_khr_img_cpu.h - cpu implementation of image related dbks.

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
#ifndef _POCL_DBK_KHR_IMG_CPU_H_
#define _POCL_DBK_KHR_IMG_CPU_H_

#include "pocl_cl.h"
#include "pocl_export.h"

/**
 * Convert an yuv mem_obj into a rbg mem_obj.
 *
 * \param program [in] used to retrieve device data.
 * \param kernel [out] the jpeg state is retrieved from the kernel data.
 * \param meta [in] used for retrieving dbk attributes.
 * \param dev_i [in] used to retrieve device data.
 * \param arguments [in/out] pointers to data to process and store.
 */
POCL_EXPORT int
pocl_cpu_execute_dbk_exp_img_yuv2rgb (cl_program program,
                                      cl_kernel kernel,
                                      pocl_kernel_metadata_t *meta,
                                      cl_uint dev_i,
                                      struct pocl_argument *arguments);

#endif //_POCL_DBK_KHR_IMG_CPU_H_
