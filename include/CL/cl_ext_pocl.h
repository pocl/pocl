/*******************************************************************************
 * Copyright (c) 2021 Tampere University
 *               2023-2024 Pekka Jääskeläinen / Intel Finland Oy
 *
 * PoCL-specific proof-of-concept (draft) or finalized OpenCL extensions.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
 * KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
 * SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
 *    https://www.khronos.org/registry/
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 ******************************************************************************/

#include <CL/cl_ext.h>

#ifndef __CL_EXT_POCL_H
#define __CL_EXT_POCL_H

#ifdef __cplusplus
extern "C"
{
#endif

/* cl_pocl_content_size should be defined in CL/cl_ext.h; however,
 * if we PoCL is built against the system headers, it's possible
 * that they have an outdated version of CL/cl_ext.h.
 * In that case, add the extension here.
 */

#ifndef cl_pocl_content_size

#define cl_pocl_content_size 1

extern CL_API_ENTRY cl_int CL_API_CALL
clSetContentSizeBufferPoCL(
    cl_mem    buffer,
    cl_mem    content_size_buffer) CL_API_SUFFIX__VERSION_1_2;

typedef CL_API_ENTRY cl_int
(CL_API_CALL *clSetContentSizeBufferPoCL_fn)(
    cl_mem    buffer,
    cl_mem    content_size_buffer) CL_API_SUFFIX__VERSION_1_2;

#endif

/* cl_ext_buffer_device_address (experimental version)
 */

#ifndef cl_ext_buffer_device_address
#define cl_ext_buffer_device_address 1

/* clCreateBuffer() flag: A new cl_mem_flag CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT:
 * When set to CL_TRUE, specifies that the buffer must have a single fixed
 * device-side address for its lifetime, and the address can be queried via
 * clGetMemObjectInfo.
 *
 * Each device in the context can have their own (fixed) device-side address
 * and a copy of the created buffer which are synchronized implicitly by the
 * runtime.
 *
 * The flag might imply that the buffer will be "pinned" permanently to
 * a device's memory, but might not be necessarily so, as long as the address
 * range of the buffer remains constant.
 *
 * The device addresses of sub-buffers derived from
 * CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT allocated buffers can be computed by
 * adding the sub-buffer origin to the device-specific start address.
 */
#define CL_MEM_DEVICE_PRIVATE_ADDRESS_EXT 0x5000

/* clGetMemObjectInfo(): A new cl_mem_info type CL_MEM_DEVICE_PTR_EXT:
 * Returns the device address for a buffer allocated with
 * CL_MEM_DEVICE_ADDRESS_EXT. If the buffer was not created with the flag,
 * returns CL_INVALID_MEM_OBJECT.
 */
#define CL_MEM_DEVICE_ADDRESS_EXT 0x5001

typedef cl_ulong cl_mem_device_address_EXT;

/* clSetKernelExecInfo(): CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT:
 * Similar to CL_KERNEL_EXEC_INFO_SVM_PTRS except for CL_MEM_DEVICE_ADDRESS_EXT
 * device pointers: If a device pointer accessed by a kernel is not passed as
 * an argument, it must be set by this property.
 */
#define CL_KERNEL_EXEC_INFO_DEVICE_PTRS_EXT 0x5002

/* A new function clSetKernelArgDevicePointerEXT() for setting raw device
 * pointers as kernel arguments. */

typedef cl_int (CL_API_CALL *clSetKernelArgDevicePointerEXT_fn) (
    cl_kernel kernel, cl_uint arg_index, cl_mem_device_address_EXT dev_addr);

/* cl_ext_buffer_device_address (experimental stage) */
#endif

#define CL_DEVICE_REMOTE_TRAFFIC_STATS_POCL 0x4501

#define CL_DEVICE_REMOTE_SERVER_IP_POCL 0x4503

#define CL_DEVICE_REMOTE_SERVER_PORT_POCL 0x4504

  /***********************************
   * cl_pocl_svm_rect +
   * cl_pocl_command_buffer_svm +
   * cl_pocl_command_buffer_host_buffer
   * extensions
   ************************************/

#ifdef cl_khr_command_buffer

// SVM memory command-buffer functions (clCommandSVMMemcpyPOCL etc)
#define cl_pocl_command_buffer_svm 1

// cl_mem & host command-buffer functions (clCommandReadBuffer etc)
#define cl_pocl_command_buffer_host_buffer 1

// these are separate from command-buffers
// clEnqueueSVMMemFillRectPOCL, clEnqueueSVMMemcpyRectPOCL
#define cl_pocl_svm_rect 1

/****************************************************/

/* cl_device_command_buffer_capabilities_khr - bitfield */
#define CL_COMMAND_BUFFER_CAPABILITY_PROFILING_POCL  (1 << 8)

/* cl_command_buffer_flags_khr */
#define CL_COMMAND_BUFFER_PROFILING_POCL              (1 << 8)

/* cl_command_buffer_info_khr */
#define CL_COMMAND_BUFFER_INFO_PROFILING_POCL                     0x1299

/* cl_command_type */
/* To be used by clGetEventInfo: */
/* TODO use values from an assigned range */
#define CL_COMMAND_SVM_MEMCPY_RECT_POCL                       0x1210
#define CL_COMMAND_SVM_MEMFILL_RECT_POCL                      0x1211

typedef cl_int (CL_API_CALL *
clCommandSVMMemcpyPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *dst_ptr,
    const void *src_ptr,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandSVMMemcpyRectPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *dst_ptr,
    const void *src_ptr,
    const size_t *dst_origin,
    const size_t *src_origin,
    const size_t *region,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandSVMMemfillPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *svm_ptr,
    size_t size,
    const void *pattern,
    size_t pattern_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);


typedef cl_int (CL_API_CALL *
clCommandSVMMemfillRectPOCL_fn)(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *svm_ptr,
    const size_t *origin,
    const size_t *region,
    size_t row_pitch,
    size_t slice_pitch,
    const void *pattern,
    size_t pattern_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);







typedef cl_int (CL_API_CALL *
clCommandReadBufferPOCL_fn)(cl_command_buffer_khr command_buffer,
                        cl_command_queue command_queue,
                        cl_mem buffer,
                        size_t offset,
                        size_t size,
                        void *ptr,
                        cl_uint num_sync_points_in_wait_list,
                        const cl_sync_point_khr* sync_point_wait_list,
                        cl_sync_point_khr* sync_point,
                        cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandReadBufferRectPOCL_fn)(cl_command_buffer_khr command_buffer,
                            cl_command_queue command_queue,
                            cl_mem buffer,
                            const size_t *buffer_origin,
                            const size_t *host_origin,
                            const size_t *region,
                            size_t buffer_row_pitch,
                            size_t buffer_slice_pitch,
                            size_t host_row_pitch,
                            size_t host_slice_pitch,
                            void *ptr,
                            cl_uint num_sync_points_in_wait_list,
                            const cl_sync_point_khr* sync_point_wait_list,
                            cl_sync_point_khr* sync_point,
                            cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandReadImagePOCL_fn)(cl_command_buffer_khr command_buffer,
                       cl_command_queue command_queue,
                       cl_mem               image,
                       const size_t *       origin, /* [3] */
                       const size_t *       region, /* [3] */
                       size_t               row_pitch,
                       size_t               slice_pitch,
                       void *               ptr,
                       cl_uint num_sync_points_in_wait_list,
                       const cl_sync_point_khr* sync_point_wait_list,
                       cl_sync_point_khr* sync_point,
                       cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandWriteBufferPOCL_fn)(cl_command_buffer_khr command_buffer,
                         cl_command_queue command_queue,
                         cl_mem buffer,
                         size_t offset,
                         size_t size,
                         const void *ptr,
                         cl_uint num_sync_points_in_wait_list,
                         const cl_sync_point_khr* sync_point_wait_list,
                         cl_sync_point_khr* sync_point,
                         cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandWriteBufferRectPOCL_fn)(cl_command_buffer_khr command_buffer,
                             cl_command_queue command_queue,
                             cl_mem buffer,
                             const size_t *buffer_origin,
                             const size_t *host_origin,
                             const size_t *region,
                             size_t buffer_row_pitch,
                             size_t buffer_slice_pitch,
                             size_t host_row_pitch,
                             size_t host_slice_pitch,
                             const void *ptr,
                             cl_uint num_sync_points_in_wait_list,
                             const cl_sync_point_khr* sync_point_wait_list,
                             cl_sync_point_khr* sync_point,
                             cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clCommandWriteImagePOCL_fn)(cl_command_buffer_khr command_buffer,
                        cl_command_queue    command_queue,
                        cl_mem              image,
                        const size_t *      origin, /*[3]*/
                        const size_t *      region, /*[3]*/
                        size_t              row_pitch,
                        size_t              slice_pitch,
                        const void *        ptr,
                        cl_uint num_sync_points_in_wait_list,
                        const cl_sync_point_khr* sync_point_wait_list,
                        cl_sync_point_khr* sync_point,
                        cl_mutable_command_khr* mutable_handle);

typedef cl_int (CL_API_CALL *
clEnqueueSVMMemcpyRectPOCL_fn) (cl_command_queue command_queue,
                            cl_bool blocking,
                            void *dst_ptr,
                            const void *src_ptr,
                            const size_t *dst_origin,
                            const size_t *src_origin,
                            const size_t *region,
                            size_t dst_row_pitch,
                            size_t dst_slice_pitch,
                            size_t src_row_pitch,
                            size_t src_slice_pitch,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list,
                            cl_event *event);

typedef cl_int (CL_API_CALL *
clEnqueueSVMMemFillRectPOCL_fn) (cl_command_queue  command_queue,
                             void *            svm_ptr,
                             const size_t *    origin,
                             const size_t *    region,
                             size_t            row_pitch,
                             size_t            slice_pitch,
                             const void *      pattern,
                             size_t            pattern_size,
                             size_t            size,
                             cl_uint           num_events_in_wait_list,
                             const cl_event *  event_wait_list,
                             cl_event *        event);


#ifndef CL_NO_PROTOTYPES

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemcpyPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *dst_ptr,
    const void *src_ptr,
    size_t size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemcpyRectPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *dst_ptr,
    const void *src_ptr,
    const size_t *dst_origin,
    const size_t *src_origin,
    const size_t *region,
    size_t dst_row_pitch,
    size_t dst_slice_pitch,
    size_t src_row_pitch,
    size_t src_slice_pitch,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemfillPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *svm_ptr,
    size_t size,
    const void *pattern,
    size_t pattern_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandSVMMemfillRectPOCL(
    cl_command_buffer_khr command_buffer,
    cl_command_queue command_queue,
    void *svm_ptr,
    const size_t *origin,
    const size_t *region,
    size_t row_pitch,
    size_t slice_pitch,
    const void *pattern,
    size_t pattern_size,
    cl_uint num_sync_points_in_wait_list,
    const cl_sync_point_khr* sync_point_wait_list,
    cl_sync_point_khr* sync_point,
    cl_mutable_command_khr* mutable_handle);




extern CL_API_ENTRY cl_int CL_API_CALL
clCommandReadBufferPOCL(cl_command_buffer_khr command_buffer,
                        cl_command_queue command_queue,
                        cl_mem buffer,
                        size_t offset,
                        size_t size,
                        void *ptr,
                        cl_uint num_sync_points_in_wait_list,
                        const cl_sync_point_khr* sync_point_wait_list,
                        cl_sync_point_khr* sync_point,
                        cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandReadBufferRectPOCL(cl_command_buffer_khr command_buffer,
                            cl_command_queue command_queue,
                            cl_mem buffer,
                            const size_t *buffer_origin,
                            const size_t *host_origin,
                            const size_t *region,
                            size_t buffer_row_pitch,
                            size_t buffer_slice_pitch,
                            size_t host_row_pitch,
                            size_t host_slice_pitch,
                            void *ptr,
                            cl_uint num_sync_points_in_wait_list,
                            const cl_sync_point_khr* sync_point_wait_list,
                            cl_sync_point_khr* sync_point,
                            cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandReadImagePOCL(cl_command_buffer_khr command_buffer,
                       cl_command_queue command_queue,
                       cl_mem               image,
                       const size_t *       origin, /* [3] */
                       const size_t *       region, /* [3] */
                       size_t               row_pitch,
                       size_t               slice_pitch,
                       void *               ptr,
                       cl_uint num_sync_points_in_wait_list,
                       const cl_sync_point_khr* sync_point_wait_list,
                       cl_sync_point_khr* sync_point,
                       cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandWriteBufferPOCL(cl_command_buffer_khr command_buffer,
                         cl_command_queue command_queue,
                         cl_mem buffer,
                         size_t offset,
                         size_t size,
                         const void *ptr,
                         cl_uint num_sync_points_in_wait_list,
                         const cl_sync_point_khr* sync_point_wait_list,
                         cl_sync_point_khr* sync_point,
                         cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandWriteBufferRectPOCL(cl_command_buffer_khr command_buffer,
                             cl_command_queue command_queue,
                             cl_mem buffer,
                             const size_t *buffer_origin,
                             const size_t *host_origin,
                             const size_t *region,
                             size_t buffer_row_pitch,
                             size_t buffer_slice_pitch,
                             size_t host_row_pitch,
                             size_t host_slice_pitch,
                             const void *ptr,
                             cl_uint num_sync_points_in_wait_list,
                             const cl_sync_point_khr* sync_point_wait_list,
                             cl_sync_point_khr* sync_point,
                             cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clCommandWriteImagePOCL(cl_command_buffer_khr command_buffer,
                        cl_command_queue    command_queue,
                        cl_mem              image,
                        const size_t *      origin, /*[3]*/
                        const size_t *      region, /*[3]*/
                        size_t              row_pitch,
                        size_t              slice_pitch,
                        const void *        ptr,
                        cl_uint num_sync_points_in_wait_list,
                        const cl_sync_point_khr* sync_point_wait_list,
                        cl_sync_point_khr* sync_point,
                        cl_mutable_command_khr* mutable_handle);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemcpyRectPOCL (cl_command_queue command_queue,
                            cl_bool blocking,
                            void *dst_ptr,
                            const void *src_ptr,
                            const size_t *dst_origin,
                            const size_t *src_origin,
                            const size_t *region,
                            size_t dst_row_pitch,
                            size_t dst_slice_pitch,
                            size_t src_row_pitch,
                            size_t src_slice_pitch,
                            cl_uint num_events_in_wait_list,
                            const cl_event *event_wait_list,
                            cl_event *event);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueSVMMemFillRectPOCL (cl_command_queue  command_queue,
                             void *            svm_ptr,
                             const size_t *    origin,
                             const size_t *    region,
                             size_t            row_pitch,
                             size_t            slice_pitch,
                             const void *      pattern,
                             size_t            pattern_size,
                             size_t            size,
                             cl_uint           num_events_in_wait_list,
                             const cl_event *  event_wait_list,
                             cl_event *        event);

extern CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgDevicePointerEXT (
    cl_kernel kernel, cl_uint arg_index, cl_mem_device_address_EXT dev_addr);

#endif // CL_NO_PROTOTYPES

#endif // cl_khr_command_buffer

#ifdef __cplusplus
}
#endif

#endif /* __CL_EXT_POCL_H */
