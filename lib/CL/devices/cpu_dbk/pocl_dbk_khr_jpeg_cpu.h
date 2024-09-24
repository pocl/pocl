/* pocl_dbk_khr_jpeg_cpu.h - JPEG Defined Built-in Kernel functions for CPU
   devices.

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

#ifndef _POCL_DBK_KHR_JPEG_CPU_H_
#define _POCL_DBK_KHR_JPEG_CPU_H_

#include "pocl_cl.h"

/**
 * JPEG encode an RGB image in the first argument and store
 * the result and size in the second two arguments.
 *
 * \param program [in] used to retrieve device data
 * \param kernel [out] the JPEG state is retrieved from the kernel data.
 * \param meta [in] used for retrieving DBK attributes.
 * \param dev_i [in] used to retrieve device data.
 * \param arguments [in/out] pointers to data to process and store.
 */
POCL_EXPORT int
pocl_cpu_execute_dbk_khr_jpeg_encode (cl_program program,
                                      cl_kernel kernel,
                                      pocl_kernel_metadata_t *meta,
                                      cl_uint dev_i,
                                      struct pocl_argument *arguments);

/**
 * Create and init the JPEG state, including a turbo JPEG handle.
 *
 * \param attributes [in] khr_jpeg_encode attributes used to configure turbo
 * JPEG.
 * \param status [out] returns state of operation.
 * \return pointer to created state needed to encode an image.
 */
POCL_EXPORT void *pocl_cpu_init_dbk_khr_jpeg_encode (void const *attributes,
                                                     int *status);

/**
 * Release and cleanup JPEG DBK state.
 *
 * \param kernel_data [in/out] state to be released nd set to NULL.
 * \return CL_SUCCESS and otherwise an OpenCL error.
 */
POCL_EXPORT int pocl_cpu_destroy_dbk_khr_jpeg_encode (void **kernel_data);

/**
 * Decompress the given JPEG image and image size in the first
 * and second arguments respectively and store the resulting RGB image in the
 * third argument.
 *
 * \param program [in] used to retrieve device data.
 * \param kernel [out] the JPEG state is retrieved from the kernel data.
 * \param meta [in] used for retrieving DBK attributes.
 * \param dev_i [in] used to retrieve device data.
 * \param arguments [in/out] pointers to data to process and store.
 */
POCL_EXPORT int
pocl_cpu_execute_dbk_khr_jpeg_decode (cl_program program,
                                      cl_kernel kernel,
                                      pocl_kernel_metadata_t *meta,
                                      cl_uint dev_i,
                                      struct pocl_argument *arguments);

/**
 * Create and init the JPEG state, including a turbo JPEG handle.
 *
 * \param attributes [in] currently not used as there are no required
 * attributes.
 * \param status [out] returns state of operation.
 * \return pointer to created state needed to encode an image.
 */
POCL_EXPORT void *pocl_cpu_init_dbk_khr_jpeg_decode (void const *attributes,
                                                     int *status);

/**
 * Release and cleanup JPEG DBK state.
 *
 * \param kernel_data [in/out] state to be released and set to NULL.
 * \return CL_SUCCESS and otherwise an OpenCL error.
 */
POCL_EXPORT int pocl_cpu_destroy_dbk_khr_jpeg_decode (void **kernel_data);

#endif //_POCL_DBK_KHR_JPEG_CPU_H_
